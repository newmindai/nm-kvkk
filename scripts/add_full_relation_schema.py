"""Make every nm6k training record carry the FULL relation schema the evaluators query with.

Why: the nm6k records declare all 115 entity labels (absent ones as empty negatives) but only the
relation types present in the document plus ~4 empty negatives — 6.6 relation names per record, no
descriptions. Evaluation (scripts/eval_relations.py, scripts/bench_gt.py) sends
all 113 types WITH descriptions. For the Mursit encoder that prompt is 5x longer than anything seen
in training and entity recall collapses with prompt length.

This writes a copy where each record's `relations` list has one `{name: {"head": "", "tail": ""}}`
negative for every absent type and `relation_descriptions` for all types, so the training prompt
matches the eval prompt. The trainer passes record["output"] to the processor as the schema, which
reads `relation_descriptions` (processor.py: _process_relations).

  python scripts/add_full_relation_schema.py --in datasets/nm6k/tr/train_clean.jsonl \
      --labels configs/labels/nm6k_vocab_tr.json --out datasets/nm6k/tr/train_clean_fullrel.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def expand(record: dict, relation_names: list, descriptions: dict, with_descriptions: bool = True) -> dict:
    out = dict(record["output"])
    items = list(out.get("relations", []))
    present = {next(iter(item)) for item in items}
    items += [{name: {"head": "", "tail": ""}} for name in relation_names if name not in present]
    out["relations"] = items
    if with_descriptions:
        out["relation_descriptions"] = {name: descriptions[name] for name in relation_names}
    return {**record, "output": out}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--no-descriptions", action="store_true", help="all 113 types as bare names (prompt ~1/3 the length)"
    )
    args = ap.parse_args(argv)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    names = sorted(set(labels["relation_names"].values()))
    descriptions = labels["relation_descriptions"]
    missing = [n for n in names if n not in descriptions]
    if missing:
        raise SystemExit(f"no description for: {missing}")
    n = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for line in args.inp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fh.write(
                json.dumps(expand(json.loads(line), names, descriptions, not args.no_descriptions), ensure_ascii=False)
                + "\n"
            )
            n += 1
    print(
        f"{args.out}: {n} records, {len(names)} relation types per record ({'names only' if args.no_descriptions else 'with descriptions'})"
    )


if __name__ == "__main__":
    main()
