"""Validate a convert_bio_to_gliner.py output directory against its raw BIO file.

Checks (hard-fails on any violation):
  1. every record has exactly the keys {id, input, output}, output has exactly
     {entities}, and entities declares EXACTLY the family's label set (empty
     lists as negatives) — no {"entities": {}} records can exist;
  2. ids are '<family>/<split>/<n>' with n sequential from 0, file order;
  3. every mention is a non-empty string, a verbatim substring of its input,
     AND whole-word matches its input under GLiNER2's REAL word splitter +
     lowercasing: the mention's WhitespaceTokenSplitter token sequence must
     occur as a contiguous subsequence of the input's token sequence —
     exactly processor._tokenize_text + _find_sublist semantics (the splitter
     is imported from the gliner2 package if available, else loaded from
     GLiNER2/gliner2/processing/word_splitter.py in --gliner2-repo); the
     failure count is reported and must be 0;
  4. per-record per-label mentions are unique case-insensitively;
  5. mention conservation vs raw:  raw B-spans  ==  spans dropped with rows
     + junk-filtered mentions + within-record dedup merges + final mentions
     (exact, per label);
  6. validation_<N>.jsonl is a verbatim line-subset of validation.jsonl;
  7. prints per-split record counts, per-label final mention counts, records
     over 64 unique mentions per label, and a seeded 5-record spot check
     printed verbatim.

Usage:
  python scripts/validate_gliner_conversion.py \
      --raw datasets/kavram_terim/v1/raw_bio.jsonl \
      --dir datasets/kavram_terim/v1/final --family kavram_terim --seed 42
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
from pathlib import Path

SPLIT_NAMES = ("train", "validation", "test")
SEP = "\x00"  # token-sequence delimiter for the subsequence check


def load_word_splitter(repo: Path):
    """Return (splitter instance, provenance) using GLiNER2's real code.

    Prefers the installed gliner2 package (conda env); falls back to loading
    word_splitter.py straight from the GLiNER2 repo checkout, which needs no
    torch. Both paths execute the identical real module.
    """
    try:
        from gliner2.processing.word_splitter import WhitespaceTokenSplitter

        return WhitespaceTokenSplitter(), "gliner2 package import"
    except Exception:
        path = repo / "gliner2" / "processing" / "word_splitter.py"
        spec = importlib.util.spec_from_file_location("g2_word_splitter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.WhitespaceTokenSplitter(), f"loaded from {path}"


def make_matcher(splitter):
    """Real GLiNER2 train-time matching: tokenize mention and input with the
    real splitter (lower=True, as processor._tokenize_text does) and require
    the mention token sequence as a contiguous subsequence of the input's
    (what _find_sublist checks). Implemented as a delimiter-joined substring
    search for speed — exactly equivalent for contiguous token runs."""

    def tokens_of(text: str) -> list[str]:
        return [tok for tok, _, _ in splitter(text, True)]

    def whole_word_ok(mention: str, input_key: str) -> bool:
        mention_tokens = tokens_of(mention)
        if not mention_tokens or all(t == "" for t in mention_tokens):
            return False  # _find_sublist returns no-match for empty queries
        return SEP + SEP.join(mention_tokens) + SEP in input_key

    def input_key_of(text: str) -> str:
        return SEP + SEP.join(tokens_of(text)) + SEP

    return whole_word_ok, input_key_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dir", dest="outdir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spot", type=int, default=5)
    parser.add_argument(
        "--gliner2-repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "GLiNER2",
        help="GLiNER2 checkout (fallback source of the real word splitter when the package isn't importable)",
    )
    args = parser.parse_args()

    splitter, provenance = load_word_splitter(args.gliner2_repo)
    whole_word_ok, input_key_of = make_matcher(splitter)
    print(f"word splitter: {provenance}")

    stats = json.loads((args.outdir / "conversion_stats.json").read_text(encoding="utf-8"))
    label_map: dict[str, str] = stats["label_map"]
    label_names = list(label_map.values())
    label_set = set(label_names)

    # ---- raw per-label B-span counts ---------------------------------------
    raw_spans = collections.Counter()
    raw_rows = 0
    with args.raw.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            raw_rows += 1
            for tag in json.loads(line)["ner_tags"]:
                if tag.startswith("B-"):
                    raw_spans[tag[2:]] += 1

    # ---- walk the converted splits -----------------------------------------
    counts = {}
    final_mentions = {name: collections.Counter() for name in label_names}
    over_capacity = collections.Counter()
    empty_records = collections.Counter()
    n_mentions_checked = 0
    whole_word_failures: list[str] = []
    all_lines: list[tuple[str, str]] = []  # (split, raw line) for spot check
    for split in SPLIT_NAMES:
        path = args.outdir / f"{split}.jsonl"
        n = 0
        with path.open(encoding="utf-8") as fin:
            for k, line in enumerate(fin):
                rec = json.loads(line)
                assert set(rec) == {"id", "input", "output"}, f"{path}:{k} keys {set(rec)}"
                assert set(rec["output"]) == {"entities"}, f"{path}:{k} output keys"
                assert rec["id"] == f"{args.family}/{split}/{k}", f"{path}:{k} id={rec['id']}"
                entities = rec["output"]["entities"]
                assert set(entities) == label_set, f"{path}:{k} labels {sorted(entities)} != {sorted(label_set)}"
                input_key = input_key_of(rec["input"])
                any_mention = over = False
                for name, mentions in entities.items():
                    assert isinstance(mentions, list), f"{path}:{k} {name} not a list"
                    lowered = set()
                    for m in mentions:
                        assert isinstance(m, str) and m, f"{path}:{k} bad mention {m!r}"
                        assert m in rec["input"], f"{path}:{k} not substring: {m!r}"
                        n_mentions_checked += 1
                        if not whole_word_ok(m, input_key):
                            whole_word_failures.append(f"{path}:{k} {m!r}")
                        key = m.lower()
                        assert key not in lowered, f"{path}:{k} ci-duplicate: {m!r}"
                        lowered.add(key)
                    final_mentions[name][split] += len(mentions)
                    over = over or len(mentions) > 64
                    any_mention = any_mention or bool(mentions)
                if over:
                    over_capacity[split] += 1
                if not any_mention:
                    empty_records[split] += 1
                all_lines.append((split, line.rstrip("\n")))
                n += 1
        counts[split] = n

    # ---- validation subset is a verbatim line-subset ------------------------
    subset_report = "none"
    subsets = sorted(args.outdir.glob("validation_*.jsonl"))
    if subsets:
        val_lines = set((args.outdir / "validation.jsonl").read_text(encoding="utf-8").splitlines())
        for sub in subsets:
            sub_lines = sub.read_text(encoding="utf-8").splitlines()
            assert len(set(sub_lines)) == len(sub_lines), f"{sub}: duplicate lines"
            missing = [l for l in sub_lines if l not in val_lines]
            assert not missing, f"{sub}: {len(missing)} lines not in validation.jsonl"
            subset_report = f"{sub.name}: {len(sub_lines)} records, verbatim subset OK"

    # ---- conservation per label --------------------------------------------
    print(f"== {args.family} ==")
    print(
        f"raw rows: {raw_rows}  (stats says total={stats['rows']['total']}, "
        f"kept={stats['rows']['kept']}, dropped={stats['rows']['dropped']})"
    )
    print(f"records per split: {counts}  (original rows per split: {stats['rows_per_split']})")
    print(f"all-negative records per split: {dict(empty_records)}")
    print(
        f"records with a >64-unique-mention label: {dict(over_capacity)} "
        f"(stats: {stats['records_over_64_unique_mentions']})"
    )
    print(
        f"real-splitter whole-word check: {len(whole_word_failures)} failures "
        f"/ {n_mentions_checked} mentions (GLiNER2 splitter+lowercasing)"
    )
    for line in whole_word_failures[:20]:
        print(f"    FAIL {line}")
    assert not whole_word_failures, f"{len(whole_word_failures)} mentions fail GLiNER2 whole-word matching"
    print(f"validation subset: {subset_report}")
    print("conservation (exact, per label):")
    ok = True
    for raw_label, name in label_map.items():
        raw_n = raw_spans[raw_label]
        with_rows = sum(stats["spans_dropped_with_rows"][raw_label].values())
        junked = sum(stats["mentions_junk_filtered"][name].values())
        merged = stats["mentions_deduped_in_record"].get(name, 0)
        final = sum(final_mentions[name].values())
        lhs, rhs = raw_n, with_rows + junked + merged + final
        status = "OK" if lhs == rhs else "MISMATCH"
        ok = ok and lhs == rhs
        print(
            f"  {raw_label:>7} -> {name!r}: raw {lhs} == row-drops {with_rows} "
            f"+ junk {junked} + in-record-dedup {merged} + final {final} "
            f"= {rhs}  [{status}]"
        )
        print(f"           final per split: {dict(final_mentions[name])}")
        if stats["mentions_junk_filtered"][name]:
            print(f"           junk reasons: {stats['mentions_junk_filtered'][name]}")
    assert ok, "mention conservation failed"

    # ---- seeded spot check --------------------------------------------------
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(all_lines)), min(args.spot, len(all_lines)))
    print(f"\nspot check ({len(picks)} seeded-random records, verbatim):")
    for i in sorted(picks):
        split, line = all_lines[i]
        rec = json.loads(line)
        print(f"--- [{split}] {rec['id']}")
        print(line)
        input_key = input_key_of(rec["input"])
        for name, mentions in rec["output"]["entities"].items():
            for m in mentions:
                assert m in rec["input"] and whole_word_ok(m, input_key)
            print(f"    {name}: {len(mentions)} mentions, all verbatim + real-splitter whole-word in input")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
