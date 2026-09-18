"""Convert token-level BIO JSONL to GLiNER2 training JSONL.

Input rows look like {"tokens": [...], "ner_tags": ["O", "B-X", "I-X", ...]}.
Output rows are GLiNER2 records, one per line:

  {"id": "<family>/<split>/<n>", "input": "<text>",
   "output": {"entities": {"<label>": ["<verbatim mention>", ...], ...}}}

Guarantees (the format rules that matter for GLiNER2):
  * input  = tokens joined with single spaces (kvkk pre-tokenized style), or
    naturally-spaced Turkish legal text with --detokenize (see below);
  * every mention is sliced from the final input text via char-offset
    tracking, hence a verbatim substring of input BY CONSTRUCTION — no
    normalization ever;
  * every record declares ALL family labels, absent ones as EMPTY lists
    (never {"entities": {}} — sanitize drops those);
  * per-record per-label mention strings are deduped case-insensitively
    (GLiNER2 matching is case-insensitive all-occurrence; repeats only burn
    max_gold_per_query capacity);
  * splits are deterministic in --seed, assigned at the ORIGINAL-row level
    BEFORE any chunking, ids numbered sequentially per split.

Detokenization (--detokenize): rebuilds natural Turkish legal-text spacing
from the token stream via a single join-decision function with char-offset
tracking. Rules (every glue char is a single non-word char that GLiNER2's
WhitespaceTokenSplitter re-separates as its own \\S token, so spaced and glued
forms produce IDENTICAL token sequences — training whole-word matching is
provably unaffected, even when a join crosses a span boundary):
  * . , ; : ! ? ) ] } ” » ’ ' attach to the LEFT token;
  * ( [ { “ « ‘ attach to the RIGHT token;
  * straight " alternates open/close per record (parity state);
  * ’ or ' followed by a lowercase word glues BOTH sides (clitic: X ' in ->
    X'in) — safe because the splitter re-separates the apostrophe, so a
    detached mention "X" still whole-word matches inside "X'in";
  * . between digit tokens glues BOTH sides (14 . 09 . 2005 -> 14.09.2005);
  * : between 1-2 digit tokens glues BOTH sides (14 : 30 -> 14:30);
  * . between single UPPERCASE letters continuing with . glues BOTH sides
    (R . K . -> R.K.);
  * / between word tokens glues BOTH sides (2024 / 610 -> 2024/610);
  * % followed by a digit glues RIGHT (Turkish style: % 20 -> %20; the data
    shows % always precedes its number, so no attach-left for %);
  * runs of >= 3 '.' tokens with uniform span membership (fully inside one
    span or fully outside all spans) collapse to a standalone '...'
    (anonymization); runs straddling a span boundary are split at membership
    changes and only uniform sub-runs of >= 3 collapse — offsets stay exact;
  * deliberately NOT applied: gluing '-' or '_' (the splitter folds
    word-word into ONE token -> would break whole-word matching), gluing
    around '@' (email/@mention alternations fold into one token),
    reconstructing URLs (www.x folds into one token), and '…'/'...' stay
    standalone (dominant use here is anonymization placeholders).
Mentions are ALWAYS slices of the final text (never independently joined).

Row-level cleanup (flags):
  --dedup ci-keep-first    case-insensitive duplicate texts: keep first row
  --dedup conflict-drop    exact duplicate texts: keep first if the tag
                           sequences agree, drop ALL members if they conflict
                           (contradictory supervision)
  --min-tokens N           drop rows shorter than N tokens

Capacity strategy (kavram_terim): --chunk splits any row with more than
--chunk-b-threshold B-tags of one label OR more than --chunk-token-threshold
tokens at '.' sentence boundaries (never inside a span) into greedy-packed
chunks of at most --chunk-max-tokens tokens. Chunking conserves spans exactly;
the residual over-64 chunks are left for on_capacity_exceeded=skip_sample.
Mention lists are NEVER capped: under all-occurrence matching, dropping a
string would demote every occurrence to a negative.

Mention filters (olay_olgu): --filter-junk-mentions drops a span whose surface
is (a) pure punctuation, (b) a single token with <= 3 alphabetic characters
(',', 've', 'ile', bare numbers), or (c) > 30% punctuation characters
(dot-run fragments). The row itself stays, possibly as an all-negative record.

Eval hygiene (olay_olgu): --stratify samples validation/test proportionally by
per-label presence pattern (preserves zero-entity and OLAY/OLGU-present
rates); --exclude-unk-from-eval forces rows containing a literal '[UNK]'
token into train.

A conversion_stats.json with exact drop accounting is written to --outdir.

Usage:
  python scripts/convert_bio_to_gliner.py \
      --in datasets/kavram_terim/v1/raw_bio.jsonl \
      --outdir datasets/kavram_terim/v1/final --family kavram_terim \
      --label-map '{"TERIM": "hukuki terim", "KAVRAM": "hukuki kavram"}' \
      --seed 42 --splits 0.8,0.1,0.1 --val-size 5000 --test-size 5000 \
      --val-subset 2000 --dedup ci-keep-first --min-tokens 6 \
      --chunk --chunk-b-threshold 48 --chunk-token-threshold 200 \
      --chunk-max-tokens 150 --detokenize

  python scripts/convert_bio_to_gliner.py \
      --in datasets/olay_olgu/v1/raw_bio.jsonl \
      --outdir datasets/olay_olgu/v1/final --family olay_olgu \
      --label-map '{"OLAY": "olay anlatımı", "OLGU": "olgu tespiti"}' \
      --seed 42 --splits 0.8,0.1,0.1 --val-size 5000 --test-size 10000 \
      --val-subset 2000 --dedup conflict-drop --filter-junk-mentions \
      --stratify --exclude-unk-from-eval --detokenize
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

SPLIT_NAMES = ("train", "validation", "test")
UNK_TOKEN = "[UNK]"

# --- detokenization ---------------------------------------------------------
# Every char these rules glue is a single non-word char that GLiNER2's
# WhitespaceTokenSplitter re-separates as its own \S token (verified against
# GLiNER2/gliner2/processing/word_splitter.py), so spaced and glued forms
# tokenize identically and whole-word matching survives every join, including
# joins across span boundaries. '-' and '_' are excluded on purpose: the
# splitter's \w+(?:[-_]\w+)* folds word-word into ONE token.
PUNCT_ATTACH_LEFT = {".", ",", ";", ":", "!", "?", ")", "]", "}", "”", "»", "’", "'"}
PUNCT_ATTACH_RIGHT = {"(", "[", "{", "“", "«", "‘"}
WORDY_RE = re.compile(r"\w+(?:[-_]\w+)*")


def _wordy(u: str) -> bool:
    return bool(WORDY_RE.fullmatch(u))


def _single_upper(u: str) -> bool:
    return len(u) == 1 and u.isalpha() and u.isupper()


def _no_space_before(units: list, k: int, quote_open: bool) -> bool:
    """Decide the separator between output units k-1 and k ('' vs ' ')."""
    u0 = units[k][0]
    u1 = units[k - 1][0]
    u2 = units[k - 2][0] if k >= 2 else ""
    u3 = units[k + 1][0] if k + 1 < len(units) else ""
    if u0 in PUNCT_ATTACH_LEFT:
        return True
    if u1 in PUNCT_ATTACH_RIGHT:
        return True
    if (u0 == '"' or u1 == '"') and quote_open:
        return True  # closing straight quote / right after opening one
    if u1 in {"'", "’"} and u0 and u0[0].islower() and _wordy(u0):
        return True  # apostrophe clitic: X ' in -> X'in
    if u1 == "." and u2.isdigit() and u0.isdigit():
        return True  # dates / section numbers: 14 . 09 . 2005 -> 14.09.2005
    if u1 == ":" and u2.isdigit() and u0.isdigit() and len(u2) <= 2 and len(u0) <= 2:
        return True  # clock times: 14 : 30 -> 14:30
    if u1 == "." and _single_upper(u2) and _single_upper(u0) and u3 == ".":
        return True  # abbreviation runs: R . K . -> R.K.
    if u0 == "/" and _wordy(u1) and _wordy(u3):
        return True  # 2024 / 610 -> 2024/610 (left seam)
    if u1 == "/" and _wordy(u2) and _wordy(u0):
        return True  # 2024 / 610 -> 2024/610 (right seam)
    if u1 == "%" and u0[:1].isdigit():
        return True  # Turkish percent: % 20 -> %20
    return False


def space_join_offsets(tokens: list[str]) -> tuple[str, list[int], list[int]]:
    """' '.join with per-token char offsets (legacy path, byte-identical)."""
    parts, starts, ends, pos = [], [], [], 0
    for i, tok in enumerate(tokens):
        if i:
            parts.append(" ")
            pos += 1
        starts.append(pos)
        parts.append(tok)
        pos += len(tok)
        ends.append(pos)
    return "".join(parts), starts, ends


def detokenize_piece(
    tokens: list[str],
    spans: list[tuple[int, int, str]],
) -> tuple[str, list[int], list[int]]:
    """Tokens -> naturally spaced text + per-token char offsets.

    Mention strings must be sliced as text[starts[s]:ends[e-1]]. Dot runs are
    collapsed only within uniform span membership, so a slice can never begin
    or end inside a collapsed unit.
    """
    n = len(tokens)
    span_id = [-1] * n
    for sid, (s, e, _) in enumerate(spans):
        for i in range(s, e):
            span_id[i] = sid

    units: list[tuple[str, int, int]] = []  # (surface, first_tok, last_tok)
    i = 0
    while i < n:
        if tokens[i] == ".":
            j = i
            while j < n and tokens[j] == ".":
                j += 1
            # split the maximal run at span-membership changes; collapse each
            # uniform sub-run of >= 3 dots to a standalone '...'
            a = i
            while a < j:
                b = a
                while b < j and span_id[b] == span_id[a]:
                    b += 1
                if b - a >= 3:
                    units.append(("...", a, b - 1))
                else:
                    for t in range(a, b):
                        units.append((".", t, t))
                a = b
            i = j
            continue
        units.append((tokens[i], i, i))
        i += 1

    parts: list[str] = []
    starts = [0] * n
    ends = [0] * n
    pos = 0
    quote_open = False
    for k, (surface, first, last) in enumerate(units):
        if k and not _no_space_before(units, k, quote_open):
            parts.append(" ")
            pos += 1
        if surface == '"':
            quote_open = not quote_open
        parts.append(surface)
        for t in range(first, last + 1):
            starts[t] = pos
            ends[t] = pos + len(surface)
        pos += len(surface)
    return "".join(parts), starts, ends


def extract_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """BIO tags -> [(start, end_exclusive, raw_label)]. Lenient on stray I-."""
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    label = ""
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start, label = i, tag[2:]
        elif tag.startswith("I-"):
            if start is None or tag[2:] != label:  # stray I- (never in this data)
                if start is not None:
                    spans.append((start, i, label))
                start, label = i, tag[2:]
        else:  # O
            if start is not None:
                spans.append((start, i, label))
                start = None
    if start is not None:
        spans.append((start, len(tags), label))
    return spans


def junk_reason(mention: str) -> str | None:
    """Why a mention surface is junk, or None if it should be kept."""
    chars = mention.replace(" ", "")
    if not chars:
        return "pure_punctuation"
    n_punct = sum(1 for c in chars if unicodedata.category(c).startswith("P"))
    if n_punct == len(chars):
        return "pure_punctuation"
    if " " not in mention and sum(1 for c in mention if c.isalpha()) <= 3:
        return "short_single_token"
    if n_punct / len(chars) > 0.30:
        return "punct_ratio_gt_30"
    return None


def chunk_row(
    tokens: list[str], spans: list[tuple[int, int, str]], max_tokens: int
) -> list[tuple[list[str], list[tuple[int, int, str]]]]:
    """Split at '.' sentence boundaries (never inside a span), greedy-pack
    sentences into chunks of <= max_tokens. A single over-long sentence stays
    one oversized chunk. Conserves every span exactly (asserted)."""
    n = len(tokens)
    inside = bytearray(n + 1)
    for s, e, _ in spans:
        for b in range(s + 1, e):
            inside[b] = 1
    boundaries = [i + 1 for i, tok in enumerate(tokens) if tok == "." and 0 < i + 1 < n and not inside[i + 1]]
    segments, prev = [], 0
    for b in boundaries:
        segments.append((prev, b))
        prev = b
    if prev < n:
        segments.append((prev, n))

    chunks: list[tuple[int, int]] = []
    cur_s = cur_e = 0
    for s, e in segments:  # segments are contiguous: s == cur_e
        if cur_e > cur_s and e - cur_s > max_tokens:
            chunks.append((cur_s, cur_e))
            cur_s = s
        cur_e = e
    if cur_e > cur_s:
        chunks.append((cur_s, cur_e))

    out = []
    for cs, ce in chunks:
        sub = [(s - cs, e - cs, lab) for s, e, lab in spans if s >= cs and e <= ce]
        out.append((tokens[cs:ce], sub))
    assert sum(len(sub) for _, sub in out) == len(spans), "chunking lost a span"
    return out


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def largest_remainder(sizes: dict, total: int) -> dict:
    """Allocate `total` across strata proportionally to `sizes` (deterministic)."""
    pool = sum(sizes.values())
    quotas = {k: total * v / pool for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    leftovers = sorted(sizes, key=lambda k: (-(quotas[k] - alloc[k]), str(k)))
    for k in leftovers[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--label-map", required=True, help='JSON: BIO label -> GLiNER2 label name, e.g. \'{"TERIM": "hukuki terim"}\''
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        default="0.8,0.1,0.1",
        help="train,validation,test fractions (used only when --val-size/--test-size are not given)",
    )
    parser.add_argument(
        "--val-size", type=int, default=None, help="absolute validation rows (original rows, pre-chunking)"
    )
    parser.add_argument("--test-size", type=int, default=None, help="absolute test rows (original rows, pre-chunking)")
    parser.add_argument(
        "--val-subset",
        type=int,
        default=None,
        help="also write validation_<N>.jsonl, a fixed-seed subset of validation records (verbatim, same ids)",
    )
    parser.add_argument("--dedup", choices=["none", "ci-keep-first", "conflict-drop"], default="none")
    parser.add_argument("--min-tokens", type=int, default=1, help="drop rows with fewer than N tokens")
    parser.add_argument("--chunk", action="store_true")
    parser.add_argument("--chunk-b-threshold", type=int, default=48)
    parser.add_argument("--chunk-token-threshold", type=int, default=200)
    parser.add_argument("--chunk-max-tokens", type=int, default=150)
    parser.add_argument("--filter-junk-mentions", action="store_true")
    parser.add_argument(
        "--stratify", action="store_true", help="stratify validation/test by per-label presence pattern"
    )
    parser.add_argument(
        "--exclude-unk-from-eval", action="store_true", help="rows containing a literal [UNK] token go to train"
    )
    parser.add_argument(
        "--detokenize",
        action="store_true",
        help="rebuild natural Turkish legal-text spacing "
        "(see module docstring); mentions become slices "
        "of the final text via char-offset tracking",
    )
    args = parser.parse_args()

    label_map: dict[str, str] = json.loads(args.label_map)
    label_names = list(label_map.values())  # declaration order for every record
    raw_labels = list(label_map)
    fractions = [float(x) for x in args.splits.split(",")]
    assert len(fractions) == 3 and abs(sum(fractions) - 1.0) < 1e-6, args.splits

    # ---------------- pass 1: per-row metadata, drops, split map -------------
    n_rows = 0
    meta = []  # (n_tokens, span_counts_tuple, has_unk, presence_tuple)
    ci_first: dict[str, int] = {}
    exact_groups: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    drop_reason: dict[int, str] = {}

    for index, row in enumerate(iter_rows(args.src)):
        tokens, tags = row["tokens"], row["ner_tags"]
        assert len(tokens) == len(tags), f"row {index}: token/tag mismatch"
        unknown = {t[2:] for t in tags if t != "O"} - set(raw_labels)
        assert not unknown, f"row {index}: labels not in --label-map: {unknown}"
        spans = extract_spans(tags)
        counts = tuple(sum(1 for _, _, lab in spans if lab == rl) for rl in raw_labels)
        has_unk = UNK_TOKEN in tokens
        presence = None
        if args.stratify:
            if args.filter_junk_mentions and args.detokenize:
                # judge junk on the same surfaces pass 2 will emit
                dtext, dstarts, dends = detokenize_piece(tokens, spans)
                keep = [lab for s, e, lab in spans if not junk_reason(dtext[dstarts[s] : dends[e - 1]])]
            else:
                keep = [
                    lab for s, e, lab in spans if not (args.filter_junk_mentions and junk_reason(" ".join(tokens[s:e])))
                ]
            presence = tuple(rl in keep for rl in raw_labels)
        text = " ".join(tokens)
        if args.dedup == "ci-keep-first":
            key = md5(text.lower())
            if key in ci_first:
                drop_reason[index] = "duplicate_text_ci"
            else:
                ci_first[key] = index
        elif args.dedup == "conflict-drop":
            exact_groups[md5(text)].append((index, md5("|".join(tags))))
        meta.append((len(tokens), counts, has_unk, presence))
        n_rows += 1

    if args.dedup == "conflict-drop":
        for members in exact_groups.values():
            if len(members) < 2:
                continue
            if len({tags_h for _, tags_h in members}) == 1:
                for idx, _ in members[1:]:
                    drop_reason[idx] = "duplicate_text_same_tags"
            else:
                for idx, _ in members:
                    drop_reason[idx] = "duplicate_text_conflicting_tags"

    for index, (n_tok, _, _, _) in enumerate(meta):
        if index not in drop_reason and n_tok < args.min_tokens:
            drop_reason[index] = "min_tokens"

    kept = [i for i in range(n_rows) if i not in drop_reason]
    n_kept = len(kept)
    test_size = args.test_size if args.test_size is not None else round(n_kept * fractions[2])
    val_size = args.val_size if args.val_size is not None else round(n_kept * fractions[1])
    assert test_size + val_size < n_kept, "splits leave no train rows"

    split_of: dict[int, str] = {}
    if args.stratify:
        eligible, forced = [], []
        for i in kept:
            (forced if (args.exclude_unk_from_eval and meta[i][2]) else eligible).append(i)
        strata: dict[tuple, list[int]] = collections.defaultdict(list)
        for i in eligible:
            strata[meta[i][3]].append(i)
        sizes = {k: len(v) for k, v in strata.items()}
        test_alloc = largest_remainder(sizes, test_size)
        val_alloc = largest_remainder(sizes, val_size)
        rng = random.Random(args.seed)
        for key in sorted(strata):
            idxs = strata[key]
            rng.shuffle(idxs)
            t, v = test_alloc[key], val_alloc[key]
            assert t + v <= len(idxs), f"stratum {key} too small"
            for i in idxs[:t]:
                split_of[i] = "test"
            for i in idxs[t : t + v]:
                split_of[i] = "validation"
            for i in idxs[t + v :]:
                split_of[i] = "train"
        for i in forced:
            split_of[i] = "train"
    else:
        order = kept[:]
        random.Random(args.seed).shuffle(order)
        for i in order[:test_size]:
            split_of[i] = "test"
        for i in order[test_size : test_size + val_size]:
            split_of[i] = "validation"
        for i in order[test_size + val_size :]:
            split_of[i] = "train"

    # ---------------- pass 2: build + write records --------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    writers = {s: (args.outdir / f"{s}.jsonl").open("w", encoding="utf-8") for s in SPLIT_NAMES}
    rec_counter = {s: 0 for s in SPLIT_NAMES}
    rows_per_split = collections.Counter()
    final_mentions = {name: collections.Counter() for name in label_names}
    junk = {name: collections.Counter() for name in label_names}
    dedup_mention = collections.Counter()
    over_capacity = collections.Counter()
    dropped_spans = {rl: collections.Counter() for rl in raw_labels}
    rows_chunked = chunks_emitted = unk_rows_in_train = 0

    for index, row in enumerate(iter_rows(args.src)):
        reason = drop_reason.get(index)
        if reason is not None:
            for rl, c in zip(raw_labels, meta[index][1]):
                dropped_spans[rl][reason] += c
            continue
        tokens, tags = row["tokens"], row["ner_tags"]
        spans = extract_spans(tags)
        split = split_of[index]
        rows_per_split[split] += 1
        if args.exclude_unk_from_eval and meta[index][2]:
            unk_rows_in_train += 1

        counts = meta[index][1]
        if args.chunk and (len(tokens) > args.chunk_token_threshold or any(c > args.chunk_b_threshold for c in counts)):
            pieces = chunk_row(tokens, spans, args.chunk_max_tokens)
            rows_chunked += 1
            chunks_emitted += len(pieces)
        else:
            pieces = [(tokens, spans)]

        for piece_tokens, piece_spans in pieces:
            if args.detokenize:
                text, starts, ends = detokenize_piece(piece_tokens, piece_spans)
            else:
                text, starts, ends = space_join_offsets(piece_tokens)
            entities = {name: [] for name in label_names}
            seen = {name: set() for name in label_names}
            for s, e, raw in piece_spans:
                name = label_map[raw]
                mention = text[starts[s] : ends[e - 1]]
                if args.filter_junk_mentions:
                    why = junk_reason(mention)
                    if why:
                        junk[name][why] += 1
                        continue
                key = mention.lower()
                if key in seen[name]:
                    dedup_mention[name] += 1
                    continue
                seen[name].add(key)
                entities[name].append(mention)
            for name in label_names:
                final_mentions[name][split] += len(entities[name])
            if any(len(v) > 64 for v in entities.values()):
                over_capacity[split] += 1
            record = {
                "id": f"{args.family}/{split}/{rec_counter[split]}",
                "input": text,
                "output": {"entities": entities},
            }
            writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            rec_counter[split] += 1

    for handle in writers.values():
        handle.close()

    if args.val_subset:
        val_path = args.outdir / "validation.jsonl"
        lines = val_path.read_text(encoding="utf-8").splitlines()
        picks = sorted(random.Random(args.seed).sample(range(len(lines)), args.val_subset))
        subset_path = args.outdir / f"validation_{args.val_subset}.jsonl"
        with subset_path.open("w", encoding="utf-8") as fout:
            for i in picks:
                fout.write(lines[i] + "\n")

    stats = {
        "family": args.family,
        "input": str(args.src),
        "seed": args.seed,
        "label_map": label_map,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rows": {
            "total": n_rows,
            "kept": n_kept,
            "dropped": dict(collections.Counter(drop_reason.values())),
        },
        "rows_per_split": dict(rows_per_split),
        "records_per_split": dict(rec_counter),
        "chunking": {"rows_chunked": rows_chunked, "chunks_emitted": chunks_emitted},
        "unk_rows_forced_to_train": unk_rows_in_train,
        "spans_dropped_with_rows": {rl: dict(c) for rl, c in dropped_spans.items()},
        "mentions_junk_filtered": {n: dict(c) for n, c in junk.items()},
        "mentions_deduped_in_record": dict(dedup_mention),
        "final_mentions": {n: dict(c) for n, c in final_mentions.items()},
        "records_over_64_unique_mentions": dict(over_capacity),
    }
    stats_path = args.outdir / "conversion_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{args.family}: {n_rows} rows -> kept {n_kept} (dropped {dict(collections.Counter(drop_reason.values()))})")
    print(f"  rows/split      {dict(rows_per_split)}")
    print(f"  records/split   {dict(rec_counter)}")
    if args.chunk:
        print(f"  chunking        {rows_chunked} rows -> {chunks_emitted} chunks")
    print("  final mentions  " + ", ".join(f"{n}={sum(c.values())}" for n, c in final_mentions.items()))
    print(f"  over-64-unique  {dict(over_capacity)} (left for skip_sample)")
    print(f"  stats -> {stats_path}")


if __name__ == "__main__":
    main()
