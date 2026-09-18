"""Convert tr_pii_relations_v2.parquet into GLiNER2 RELATION-training JSONL.

Record format (verified by the relation-format recon + smoke run):
  {"input": "<doc text>",
   "output": {"entities":  {turkish_label: [verbatim surfaces]},   # all 56 labels declared, [] = negative
              "relations": [{"<type>": {"head": "<surface>", "tail": "<surface>"}}, ...]}}

- Entity surfaces are verbatim offset slices, deduped per (record, label), and
  MUST whole-word match (training raises on unfound entity surfaces:
  boundary_preprocessing.py:430-445), so unmatchable surfaces are dropped with
  a warning.
- Relation head/tail are verbatim SURFACE STRINGS matched lowercased
  whole-word, all occurrences; gold = cross-product of occurrences. Coref: each
  triple's head/tail id resolves to its cluster's unique surfaces; one instance
  is emitted per unique (type, head_surface, tail_surface). Surfaces that never
  whole-word match contribute zero gold pairs SILENTLY, so unmatchable surfaces
  are replaced by matchable cluster members (or the triple is flagged).
- Per record, NEG_PER_RECORD absent relation types are declared as
  {"type": {"head": "", "tail": ""}} = pure negative supervision (verified).
- Relation types keep their English gold names; no relation_descriptions
  (validate_data would strip them silently anyway).

Split (doc-level, from the recon's seeded search, seed 42, first satisfying
trial): TEST bundles {1, 5, 14, 15, 33, 37}; the other 23 docs train. The
script re-verifies the constraint that every multi-doc relation type keeps at
least one train doc, and documents train-only / test-only types.

Outputs:
  datasets/kvkk_relations/v2/train_probe.jsonl   (23 records)
  datasets/kvkk_relations/v2/test_probe.jsonl    (6 records)
  results/relations_probe/overfit4.jsonl         (4 richest train records, for the overfit gate)
  results/relations_probe/gold_train.json        (relations_gold.json filtered to the train split)
  results/relations_probe/gold_test.json         (... test split)
  results/relations_probe/gold_overfit.json      (... the 4 overfit records)
  results/relations_probe/split_report.json      (split, type strata, counts, warnings)

Existing eval assets (entities_eval.jsonl, relations_gold.json) are never touched.

Usage:
  python scripts/convert_relations_train.py
"""

from __future__ import annotations

import collections
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "GLiNER2"))  # read-only import of the word splitter

from gliner2.processing.word_splitter import WhitespaceTokenSplitter  # noqa: E402

# The 29-document v2 relations pilot (private). Only build_record / tokenize are imported from this module by
# the e07 converter and the kvkk_synth dataset builder; the pilot conversion itself needs this parquet.
SOURCE_PARQUET = PROJECT_ROOT / "datasets/kvkk_relations/v2/tr_pii_relations_v2.parquet"
GOLD_PATH = PROJECT_ROOT / "datasets" / "kvkk_relations" / "v2" / "relations_gold.json"
OUT_DIR = PROJECT_ROOT / "datasets" / "kvkk_relations" / "v2"
AUX_DIR = PROJECT_ROOT / "results" / "relations_probe"

SEED = 42
TEST_BUNDLES = {1, 5, 14, 15, 33, 37}  # recon's seeded doc-level split (seed 42)
NEG_PER_RECORD = 4  # absent relation types declared as pure negatives per record
OVERFIT_N = 4

_splitter = WhitespaceTokenSplitter()


def tokenize(text: str) -> list[str]:
    """Exactly the processor's whole-word view: whitespace splitter, lowered."""
    return [tok for tok, _, _ in _splitter(text, lower=True)]


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


def convert() -> None:
    import pandas as pd

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    label_map: dict[str, str] = gold["entity_label_mapping"]
    all_entity_labels: list[str] = gold["eval_labels"]
    all_relation_types: list[str] = gold["relation_types"]
    gold_by_id = {r["id"]: r for r in gold["records"]}

    df = pd.read_parquet(SOURCE_PARQUET).sort_values("bundle_id")
    warnings: list[dict[str, Any]] = []
    train_records, test_records = [], []
    per_record_stats: dict[str, dict[str, int]] = {}
    type_docs_train: dict[str, set] = collections.defaultdict(set)
    type_docs_test: dict[str, set] = collections.defaultdict(set)

    for _, row in df.iterrows():
        bundle = int(row["bundle_id"])
        row_id = f"kvkk_relations/v2/bundle_{bundle}"
        entities = [dict(e) for e in row["entities"]]
        relations = [dict(r) for r in row["relations"]]
        rng = random.Random(f"neg-{SEED}-{bundle}")  # deterministic per record
        record, stats = build_record(
            row_id, row["text"], entities, relations, label_map, all_entity_labels, all_relation_types, rng, warnings
        )
        per_record_stats[row_id] = stats
        target = type_docs_test if bundle in TEST_BUNDLES else type_docs_train
        for rel in relations:
            target[rel["relation"]].add(bundle)
        (test_records if bundle in TEST_BUNDLES else train_records).append(record)

    # --- split verification (recon constraint) ---
    multi_doc = {
        t for t in all_relation_types if len(type_docs_train.get(t, set()) | type_docs_test.get(t, set())) >= 2
    }
    violated = [t for t in multi_doc if not type_docs_train.get(t)]
    assert not violated, f"multi-doc types with no train doc: {violated}"
    train_only = sorted(t for t in all_relation_types if type_docs_train.get(t) and not type_docs_test.get(t))
    test_only = sorted(t for t in all_relation_types if type_docs_test.get(t) and not type_docs_train.get(t))
    both = sorted(t for t in all_relation_types if type_docs_train.get(t) and type_docs_test.get(t))

    # --- conservation checks ---
    n_source_triples = sum(s["triples"] for s in per_record_stats.values())
    n_covered = sum(s["triples_covered"] for s in per_record_stats.values())
    n_gold = sum(len(r["triples"]) for r in gold["records"])
    assert n_source_triples == n_gold == 102, (n_source_triples, n_gold)
    for rec in train_records + test_records:
        for item in rec["output"]["relations"]:
            ((rtype, args),) = item.items()
            assert rtype in all_relation_types, rtype
            if args["head"]:  # positive instance: surfaces verbatim in text
                assert args["head"] in rec["input"] and args["tail"] in rec["input"], rec["id"]
        for label, surfaces in rec["output"]["entities"].items():
            assert label in all_entity_labels
            for s in surfaces:
                assert s in rec["input"], (rec["id"], label, s)

    # --- write training files (strip the id key: trainer expects input/output only) ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUX_DIR.mkdir(parents=True, exist_ok=True)

    def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps({"input": rec["input"], "output": rec["output"]}, ensure_ascii=False) + "\n")

    dump_jsonl(OUT_DIR / "train_probe.jsonl", train_records)
    dump_jsonl(OUT_DIR / "test_probe.jsonl", test_records)

    # --- overfit set: the OVERFIT_N train records with most source triples ---
    ranked = sorted(
        train_records, key=lambda r: (-per_record_stats[r["id"]]["triples"], int(r["id"].rsplit("_", 1)[1]))
    )
    overfit = ranked[:OVERFIT_N]
    dump_jsonl(AUX_DIR / "overfit4.jsonl", overfit)

    # --- gold subsets for scripts/eval_relations.py ---
    def dump_gold(path: Path, ids: list[str]) -> None:
        subset = dict(gold)
        subset["records"] = [gold_by_id[i] for i in ids]
        path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dump_gold(AUX_DIR / "gold_train.json", [r["id"] for r in train_records])
    dump_gold(AUX_DIR / "gold_test.json", [r["id"] for r in test_records])
    dump_gold(AUX_DIR / "gold_overfit.json", [r["id"] for r in overfit])

    report = {
        "seed": SEED,
        "test_bundles": sorted(TEST_BUNDLES),
        "train_records": [r["id"] for r in train_records],
        "test_records": [r["id"] for r in test_records],
        "overfit_records": [r["id"] for r in overfit],
        "relation_types": {
            "n": len(all_relation_types),
            "train_and_test": both,
            "train_only": train_only,
            "test_only_zero_shot": test_only,
        },
        "triples": {
            "source": n_source_triples,
            "covered": n_covered,
            "train_instances": sum(per_record_stats[r["id"]]["instances"] for r in train_records),
            "test_instances": sum(per_record_stats[r["id"]]["instances"] for r in test_records),
            "negatives_per_record": NEG_PER_RECORD,
        },
        "per_record": per_record_stats,
        "warnings": warnings,
    }
    (AUX_DIR / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"train_probe.jsonl: {len(train_records)} records | test_probe.jsonl: {len(test_records)}")
    print(f"triples: {n_source_triples} source, {n_covered} covered by >=1 matchable instance")
    print(f"instances: train {report['triples']['train_instances']}, test {report['triples']['test_instances']}")
    print(f"types: {len(both)} in both, {len(train_only)} train-only, {len(test_only)} test-only(zero-shot)")
    print(f"overfit set: {[r['id'] for r in overfit]}")
    print(f"warnings: {len(warnings)}")
    for w in warnings:
        print("  WARN", json.dumps(w, ensure_ascii=False)[:200])


if __name__ == "__main__":
    convert()
