"""Score every prediction file of a run with the same code, on a label subset.

  python scripts/score_run.py --run results/benchmark/vekaletname                # stack21 (default)
  python scripts/score_run.py --run results/benchmark/nm6k --subset kvkk19 --per-label
Writes <run>/results_<subset>.json and <run>/results_<subset>.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kvkkbench import labels as L  # noqa: E402
from kvkkbench import scoring as SC


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--subset", default="stack21", choices=list(L.SUBSETS))
    ap.add_argument("--per-label", action="store_true")
    ap.add_argument(
        "--exclude-docs",
        type=Path,
        default=None,
        help="JSON list of docnames to drop (e.g. configs/labels/nm6k_contaminated_for_rele07.json)",
    )
    ap.add_argument("--tag", default="", help="suffix for the results files (e.g. clean550)")
    args = ap.parse_args(argv)
    gold = [json.loads(l) for l in (args.run / "gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.exclude_docs:
        drop = set(json.loads(args.exclude_docs.read_text(encoding="utf-8")))
        gold = [d for d in gold if d["docname"] not in drop]
    run_info = json.loads((args.run / "manifest" / "_run.json").read_text(encoding="utf-8"))
    manifest = {"set": run_info["set"], "models": {}}
    for f in sorted((args.run / "manifest").glob("*.json")):
        if f.name.startswith("_"):
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        manifest["models"][m["name"]] = m
    ids = L.SUBSETS[args.subset]
    rows = []
    for name, m in manifest["models"].items():
        p = args.run / "predictions" / m["file"]
        if not p.exists():
            continue
        r = SC.score(gold, SC.load_predictions(p), ids)
        r.update(
            {
                "model": name,
                "unit": m.get("unit", "sentence"),
                "hardware": m.get("hardware", ""),
                "timing": SC.load_timing(p),
            }
        )
        rows.append(r)
    out = {
        "set": manifest["set"],
        "subset": args.subset,
        "n_docs": len(gold),
        "n_gold_subset": sum(1 for d in gold for s in d["spans"] if s["label"] in set(ids)),
        "rows": rows,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    (args.run / f"results_{args.subset}{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    md = (
        f"# {manifest['set']} — {len(gold)} documents, {out['n_gold_subset']} gold spans in the subset\n\n"
        + SC.markdown_table(rows, args.subset)
    )
    if args.per_label:
        labels = [i for i in ids if any(r["strict"]["per_label"].get(i, {}).get("gold", 0) for r in rows)]
        md += (
            "\n\nPer-label strict F1 (gold count in the header):\n\n| model | "
            + " | ".join(f"{l} ({rows[0]['strict']['per_label'].get(l, {}).get('gold', 0)})" for l in labels)
            + " |\n|---|"
            + "---:|" * len(labels)
            + "\n"
        )
        for r in sorted(rows, key=lambda r: -r["strict"]["micro"]["f1"]):
            md += (
                f"| {r['model']} | "
                + " | ".join(f"{r['strict']['per_label'].get(l, {}).get('f1', 0) * 100:.0f}" for l in labels)
                + " |\n"
            )
    (args.run / f"results_{args.subset}{suffix}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
