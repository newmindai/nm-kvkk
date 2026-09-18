"""Build the relation gold JSON that scripts/eval_relations.py consumes from an
nm-kvkk-pii-6K test split (which ships in GLiNER2 *training* record format).

Input  (test.jsonl):  {"id", "input", "output": {"entities": {name: [surfaces]},
                       "relations": [{name: {"head": s, "tail": s}}, ...]}}
Output (gold.json):   {"source", "eval_labels", "relation_types",
                       "relation_descriptions", "records": [{"id", "triples": [
                         {"relation", "head_mentions", "tail_mentions",
                          "head_label", "tail_label"}]}]}

Relation instances with an empty head or tail are the dataset's *declared
negatives* and are skipped — they are supervision, not gold triples.

The queried schema is the FULL vocabulary from labels.json (115 entity names /
113 relation names in tr), not just the types present in the split: that is the
production setting and the one the models were trained on.

Usage:
  python scripts/convert_nm6k_gold.py --data-root datasets/synth/nm-kvkk-pii-6K \\
      --variant tr --out-dir datasets/nm6k/tr
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPLITS = ("test", "test_hard", "test_easy")


def load_labels(path: Path) -> dict[str, Any]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    for key in ("entity_labels", "relation_names", "relation_descriptions"):
        if key not in labels:
            raise SystemExit(f"{path} has no {key!r}")
    return labels


def build_gold(
    records: list[dict[str, Any]], labels: dict[str, Any], source: str, relation_naming: str = "names"
) -> dict[str, Any]:
    """One gold document per record; skips declared negatives (empty head or tail).

    relation_naming="ids" queries the taxonomy ids ("national_id_of") instead of the
    variant's names ("national id of") — the vocabulary the pre-nm6k English models
    (rele07, rele07-orig, relprobe29) were trained on.
    """
    eval_labels = sorted(set(labels["entity_labels"].values()))
    name_to_id = {name: rel_id for rel_id, name in labels["relation_names"].items()}
    rename = {name: (name_to_id[name] if relation_naming == "ids" else name) for name in name_to_id}
    relation_types = sorted(set(rename.values()))
    descriptions = {rename[name]: desc for name, desc in labels["relation_descriptions"].items() if name in rename}
    missing = [r for r in relation_types if r not in descriptions]
    if missing:
        raise SystemExit(f"no description for relation names: {missing[:5]}")

    out_records, skipped, stats = [], 0, Counter()
    for record in records:
        entities = record["output"].get("entities", {})
        surface_label = {}
        for label, surfaces in entities.items():
            for surface in surfaces:
                surface_label.setdefault(surface.strip().casefold(), label)
        triples = []
        for instance in record["output"].get("relations", []):
            for name, pair in instance.items():
                head, tail = (pair.get("head") or "").strip(), (pair.get("tail") or "").strip()
                if not head or not tail:
                    skipped += 1
                    continue
                if name not in rename:
                    raise SystemExit(f"record {record.get('id')}: relation {name!r} is not in labels.json")
                triples.append(
                    {
                        "relation": rename[name],
                        "head_mentions": [head],
                        "tail_mentions": [tail],
                        "head_label": surface_label.get(head.casefold()),
                        "tail_label": surface_label.get(tail.casefold()),
                    }
                )
                stats[rename[name]] += 1
        out_records.append({"id": record["id"], "triples": triples})
    return {
        "source": source,
        "eval_labels": eval_labels,
        "relation_types": relation_types,
        "relation_descriptions": {name: descriptions[name] for name in relation_types},
        "records": out_records,
        "stats": {
            "n_records": len(out_records),
            "n_triples": sum(stats.values()),
            "n_declared_negatives_skipped": skipped,
            "n_types_with_support": len(stats),
            "support": dict(stats.most_common()),
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="nm-kvkk-pii-6K directory")
    parser.add_argument("--variant", choices=["tr", "en"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument(
        "--relation-naming",
        choices=["names", "ids"],
        default="names",
        help="'ids' queries national_id_of instead of the variant's 'national id of'",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="JSON array of record ids to drop (e.g. documents an older model trained on)",
    )
    parser.add_argument("--suffix", default="", help="appended to the output file names, e.g. _clean")
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="write empty relation descriptions: the evaluators then query BARE relation names (gold_*_nodesc.json)",
    )
    args = parser.parse_args(argv)

    root = args.data_root / args.variant
    labels = load_labels(root / "labels.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        path = root / f"{split}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any("id" not in r for r in records):
            raise SystemExit(f"{path} has records without an id — cannot build gold")
        if args.exclude:
            drop = set(json.loads(args.exclude.read_text(encoding="utf-8")))
            before = len(records)
            records = [r for r in records if r["id"] not in drop]
            print(f"  {split}: dropped {before - len(records)} of {before} excluded records")
            # the entities file handed to eval_relations must hold exactly these records
            ents = args.out_dir / f"{split}{args.suffix}.jsonl"
            ents.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        gold = build_gold(records, labels, str(path), args.relation_naming)
        if args.no_descriptions:
            gold["relation_descriptions"] = {name: "" for name in gold["relation_types"]}
        out = args.out_dir / f"gold_{split}{args.suffix}.json"
        out.write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
        s = gold["stats"]
        print(
            f"{out}: {s['n_records']} records · {s['n_triples']} triples · {s['n_types_with_support']}/"
            f"{len(gold['relation_types'])} types with support · {s['n_declared_negatives_skipped']} negatives skipped"
        )


if __name__ == "__main__":
    main()
