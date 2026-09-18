#!/usr/bin/env python3
"""Build `processed/nm-kvkk-pii-6K`: the e07 + e08 synthetic Turkish PII documents as a GLiNER2 train/test set,
in a Turkish and an English variant, split by hardness (a hard document names more than one person).

Record format — the one verified in the e07 training builds (`scripts/convert_relations_train.py::build_record`, imported
here unchanged so records stay byte-compatible with the builds that passed the overfit gate):

    {"input": "<document text>",
     "output": {"entities":  {<label name>: [verbatim surface, ...], ...},   # every label declared, [] = negative
                "relations": [{<relation name>: {"head": "<surface>", "tail": "<surface>"}}, ...]}}
                                                                            # + NEG_PER_RECORD absent types with ""

Language variants never mix: in `tr/` every entity label and every relation name is Turkish, in `en/` both are
English (`scripts/nm_kvkk_6k_labels.py`).

Split. Documents are grouped by brief (the document-type description they were written from) and whole groups go
to one side, so no brief is seen in both train and test. TEST takes --test-n documents, allocated over
(source × density profile × hardness) by largest remainder with a fixed hard quota, sampled per cell from sorted
bundle ids under --seed. TRAIN is everything else plus every zero-PII document as a pure-negative record.

Outputs per language: train.jsonl, test.jsonl, test_hard.jsonl, test_easy.jsonl, labels.json.
Shared: split_report.json, README.md.

Usage (kvkk-synth env; no sibling-repo imports, the record builder is vendored below):
    python kvkk_synth/dataset/build_6k.py [--runs kvkk_synth/runs/<a> kvkk_synth/runs/<b>] [--out datasets/synth/<name>]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # [pkg] kvkk_synth/dataset/build_6k.py -> repository root
# [pkg] vendored from scripts/convert_relations_train.py and gliner2/processing/word_splitter.py,
# so the builder has no sibling-repo dependency. Keep byte-compatible with the builds that passed the overfit gate:
# do not edit build_record. The tokenizer is GLiNER2's WhitespaceTokenSplitter regex, lowered, as the processor sees it.
import re as _re
from collections.abc import Mapping
from typing import Any

_TOKEN = _re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S""",
    _re.VERBOSE | _re.IGNORECASE,
)
NEG_PER_RECORD = 4  # absent relation types declared as pure negatives per record


def tokenize(text: str) -> list[str]:
    """Exactly the processor's whole-word view: whitespace splitter, lowered."""
    return [m.group(0).lower() for m in _TOKEN.finditer(text)]


def whole_word_matches(surface_tokens: list[str], text_tokens: list[str]) -> int:
    """Count of whole-word occurrences (processor._find_sublist semantics)."""
    if not surface_tokens:
        return 0
    n = len(surface_tokens)
    return sum(1 for i in range(len(text_tokens) - n + 1) if text_tokens[i : i + n] == surface_tokens)


def build_record(
    row_id: str,
    text: str,
    entities: list[Mapping[str, Any]],
    relations: list[Mapping[str, Any]],
    label_map: dict[str, str],
    all_entity_labels: list[str],
    all_relation_types: list[str],
    rng: random.Random,
    warnings: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    text_tokens = tokenize(text)

    def matches(surface: str) -> int:
        return whole_word_matches(tokenize(surface), text_tokens)

    # --- entities: all labels declared (entity-pilot convention), verbatim slices ---
    ent_out: dict[str, list[str]] = {name: [] for name in all_entity_labels}
    for ent in entities:
        name = label_map[ent["label"]]  # KeyError on unknown label, on purpose
        surface = text[ent["start"] : ent["end"]]
        if surface != ent["span"]:
            warnings.append({"record": row_id, "kind": "span_slice_mismatch", "span": ent["span"], "slice": surface})
        if not surface.strip() or surface in ent_out[name]:
            continue
        if matches(surface) == 0:
            # training RAISES on unfound entity surfaces -> must drop, loudly
            warnings.append({"record": row_id, "kind": "entity_surface_unmatchable", "label": name, "surface": surface})
            continue
        ent_out[name].append(surface)

    # --- coref clusters: id -> unique surfaces (offset slices, order kept) ---
    clusters: dict[str, list[str]] = collections.defaultdict(list)
    for ent in entities:
        surface = text[ent["start"] : ent["end"]]
        if surface.strip() and surface not in clusters[ent["id"]]:
            clusters[ent["id"]].append(surface)

    def matchable(eid: str) -> list[str]:
        return [s for s in clusters[eid] if matches(s) > 0]

    # --- relations: one instance per unique (type, head_surface, tail_surface) ---
    rel_out: list[dict[str, dict[str, str]]] = []
    seen: set = set()
    stats = {"triples": len(relations), "triples_covered": 0, "instances": 0, "negatives": 0}
    for rel in relations:
        heads, tails = matchable(rel["head"]), matchable(rel["tail"])
        if not heads or not tails:
            warnings.append(
                {
                    "record": row_id,
                    "kind": "triple_unmatchable",
                    "relation": rel["relation"],
                    "head_cluster": clusters[rel["head"]],
                    "tail_cluster": clusters[rel["tail"]],
                }
            )
            continue
        emitted = False
        for hs in heads:
            for ts in tails:
                key = (rel["relation"], hs, ts)
                if key in seen:
                    emitted = True  # covered by an identical earlier instance
                    continue
                seen.add(key)
                rel_out.append({rel["relation"]: {"head": hs, "tail": ts}})
                emitted = True
        if emitted:
            stats["triples_covered"] += 1
    stats["instances"] = len(rel_out)

    # --- negatives: sampled absent types, declared with empty head/tail ---
    present = {r["relation"] for r in relations}
    absent = [t for t in all_relation_types if t not in present]
    for neg in rng.sample(absent, min(NEG_PER_RECORD, len(absent))):
        rel_out.append({neg: {"head": "", "tail": ""}})
        stats["negatives"] += 1

    record = {"id": row_id, "input": text, "output": {"entities": ent_out, "relations": rel_out}}
    return record, stats


sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402  label vocabularies (tr / en)

RUNS = Path(__file__).resolve().parents[1] / "runs"  # [pkg] kvkk_synth/runs/<date>-<id>/
# The shipped nm-kvkk-pii-6K set was built from the e07 (1,000 briefs) and e08 (5,000 briefs) runs.
# Without --runs, every run folder under kvkk_synth/runs/ that has dataset/records_full.jsonl is used.
DEFAULT_SOURCES = [
    (p.name.split("-", 3)[-1], p) for p in sorted(RUNS.glob("*-*")) if (p / "dataset/records_full.jsonl").exists()
]
TAXONOMY = Path(__file__).resolve().parents[1] / "taxonomy"  # [pkg]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def brief_key(bundle):
    br = bundle.get("brief") or {}
    return hashlib.md5((br.get("description", "") or br.get("document_type", "")).strip().lower().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "datasets/synth/nm-kvkk-pii-6K"))
    ap.add_argument("--test-n", type=int, default=600)
    ap.add_argument("--test-hard", type=int, default=240, help="documents in TEST that name more than one person")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="[pkg] run folders to build from (default: the e07 + e08 runs of the shipped set); tag = folder name after the date",
    )
    a = ap.parse_args()
    sources = [(Path(r).name.split("-", 3)[-1], Path(r)) for r in a.runs] if a.runs else DEFAULT_SOURCES
    if not sources:
        sys.exit("no run folders with dataset/records_full.jsonl found; pass --runs kvkk_synth/runs/<date>-<id> ...")
    out_root = Path(a.out)
    rng = random.Random(a.seed)

    sys.path.insert(0, str(TAXONOMY))
    import kvkk_sampler as ks

    ents_meta, rels_meta = ks.load_taxonomy()
    maps = L.build(ents_meta, rels_meta)

    # ---- collect documents ---------------------------------------------------------------------------------------
    docs, negatives = [], []
    for tag, run in sources:
        bundles = {b["bundle_id"]: b for b in load_jsonl(run / "bundles.jsonl")}
        for r in load_jsonl(run / "dataset/records_full.jsonl"):
            b = bundles[r["bundle_id"]]
            docs.append(
                {
                    "id": f"{tag}-{r['bundle_id']}",
                    "source": tag,
                    "text": r["text"],
                    "entities": r["entities"],
                    "relations": r["relations"],
                    "profile": r.get("profile") or b.get("profile") or "normal",
                    "persons": len(r.get("persons_used", [])) or 1,
                    "brief": brief_key(b),
                    "document_type": (b.get("brief") or {}).get("document_type", ""),
                }
            )
        for r in load_jsonl(run / "dataset/negatives.jsonl"):
            negatives.append(
                {
                    "id": f"{tag}-neg-{r['bundle_id']}",
                    "source": tag,
                    "text": r["text"],
                    "entities": [],
                    "relations": [],
                    "profile": "zero",
                    "persons": 0,
                    "brief": brief_key(bundles[r["bundle_id"]]),
                    "document_type": "",
                }
            )
    for d in docs:
        d["hard"] = d["persons"] > 1
    print(f"{len(docs):,} documents ({sum(d['hard'] for d in docs):,} hard) + {len(negatives)} zero-PII negatives")

    # ---- brief-disjoint, stratified test split --------------------------------------------------------------------
    groups = collections.defaultdict(list)
    for d in docs:
        groups[d["brief"]].append(d)

    # a brief is "hard" if any of its documents is; whole briefs move together
    def cell(g):
        d = g[0]
        return (d["source"], d["profile"], any(x["hard"] for x in g))

    by_cell = collections.defaultdict(list)
    for k, g in groups.items():
        by_cell[cell(g)].append(k)

    hard_cells = {c: ks_ for c, ks_ in by_cell.items() if c[2]}
    easy_cells = {c: ks_ for c, ks_ in by_cell.items() if not c[2]}

    def take(cells, want, counted):
        """Largest-remainder allocation of `want` over cells, then seeded sampling of whole briefs.
        `counted(doc)` says which documents count toward the quota — for the hard half only the multi-person
        documents do, so a brief that also carries easy documents does not eat the hard budget."""
        sizes = {c: sum(sum(1 for d in groups[k] if counted(d)) for k in ks_) for c, ks_ in cells.items()}
        total = sum(sizes.values()) or 1
        exact = {c: want * s / total for c, s in sizes.items()}
        alloc = {c: int(v) for c, v in exact.items()}
        for c in sorted(exact, key=lambda c: -(exact[c] - alloc[c]))[: max(0, want - sum(alloc.values()))]:
            alloc[c] += 1
        picked = []
        for c, ks_ in cells.items():
            keys = sorted(ks_)
            rng.shuffle(keys)
            got = 0
            for k in keys:
                if got >= alloc[c]:
                    break
                picked.append(k)
                got += sum(1 for d in groups[k] if counted(d))
        return picked

    test_briefs = set(take(hard_cells, a.test_hard, lambda d: d["hard"]))
    test_briefs |= set(
        take(
            {c: [k for k in ks_ if k not in test_briefs] for c, ks_ in easy_cells.items()},
            a.test_n - sum(len(groups[k]) for k in test_briefs),
            lambda d: True,
        )
    )
    test = [d for k in test_briefs for d in groups[k]]
    # brief-disjointness covers the zero-PII negatives too: they carry the brief of the document they replaced
    train = [d for d in docs + negatives if d["brief"] not in test_briefs]
    shared = {d["brief"] for d in train} & test_briefs
    assert not shared, f"{len(shared)} briefs in both train and test"
    print(
        f"split: train {len(train):,} (incl. {sum(1 for d in train if d['persons'] == 0)} negatives) | test {len(test):,} "
        f"(hard {sum(d['hard'] for d in test)}) | briefs disjoint: yes"
    )

    # ---- write both language variants ------------------------------------------------------------------------------
    report = {
        "seed": a.seed,
        "sources": {t: str(p) for t, p in sources},
        "record_format": "gliner2 (build_record)",
        "negatives_per_record": NEG_PER_RECORD,
        "counts": {},
        "warnings": {},
    }
    for lang in ("tr", "en"):
        m = maps[lang]
        label_map = m["entity_labels"]
        all_labels = sorted(set(label_map.values()))
        rel_name = m["relation_names"]
        all_rel = sorted(set(rel_name.values()))
        d_out = out_root / lang
        d_out.mkdir(parents=True, exist_ok=True)
        warnings, counts = [], collections.Counter()

        def convert(items):
            rows = []
            for d in items:
                rels = [{**x, "relation": rel_name[x["relation"]]} for x in d["relations"]]
                rec, st = build_record(
                    d["id"],
                    d["text"],
                    d["entities"],
                    rels,
                    label_map,
                    all_labels,
                    all_rel,
                    random.Random(int(hashlib.sha256(f"{a.seed}:{d['id']}".encode()).hexdigest()[:16], 16)),
                    warnings,
                )
                counts["triples"] += st["triples"]
                counts["triples_covered"] += st["triples_covered"]
                counts["instances"] += st["instances"]
                counts["negatives"] += st["negatives"]
                rows.append((d, rec))
            return rows

        train_rows, test_rows = convert(train), convert(test)

        def dump(path, rows, with_id=False):
            with open(path, "w", encoding="utf-8") as f:
                for d, rec in rows:
                    f.write(json.dumps(({"id": d["id"]} if with_id else {}) | rec, ensure_ascii=False) + "\n")
            return len(rows)

        n_tr = dump(d_out / "train.jsonl", train_rows)
        n_te = dump(d_out / "test.jsonl", test_rows, with_id=True)
        n_h = dump(d_out / "test_hard.jsonl", [r for r in test_rows if r[0]["hard"]], with_id=True)
        n_e = dump(d_out / "test_easy.jsonl", [r for r in test_rows if not r[0]["hard"]], with_id=True)
        json.dump(
            {
                "language": lang,
                "entity_labels": m["entity_labels"],
                "relation_names": rel_name,
                "relation_descriptions": m["relation_descriptions"],
                "distinct_entity_labels": len(all_labels),
                "distinct_relation_names": len(all_rel),
            },
            open(d_out / "labels.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=1,
        )
        report["counts"][lang] = {
            "train": n_tr,
            "test": n_te,
            "test_hard": n_h,
            "test_easy": n_e,
            "entity_labels": len(all_labels),
            "relation_names": len(all_rel),
            "relation_triples": counts["triples"],
            "triples_covered": counts["triples_covered"],
            "relation_instances": counts["instances"],
            "declared_negatives": counts["negatives"],
        }
        report["warnings"][lang] = collections.Counter(w["kind"] for w in warnings)
        print(
            f"  [{lang}] train {n_tr:,} · test {n_te} (hard {n_h} / easy {n_e}) · "
            f"{len(all_labels)} entity labels · {len(all_rel)} relation names · "
            f"{counts['triples_covered']:,}/{counts['triples']:,} triples covered · warnings {dict(report['warnings'][lang])}"
        )

    report["strata"] = {
        "test": dict(
            collections.Counter(f"{d['source']}/{d['profile']}/{'hard' if d['hard'] else 'easy'}" for d in test)
        ),
        "train": dict(
            collections.Counter(f"{d['source']}/{d['profile']}/{'hard' if d.get('hard') else 'easy'}" for d in train)
        ),
    }
    report["hardness"] = {
        "train_hard": sum(1 for d in train if d.get("hard")),
        "test_hard": sum(1 for d in test if d["hard"]),
    }
    json.dump(report, open(out_root / "split_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {out_root}")


if __name__ == "__main__":
    main()
