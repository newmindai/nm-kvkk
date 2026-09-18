"""Fill REPORT_template.md with the generated tables, examples, CMS sweep table and
summary bullets -> REPORT.md. Run after make_charts.py / make_report_tables.py / make_examples.py.

Usage:
  python scripts/make_report.py --template path/to/REPORT_template.md --tables results/bench_report/tables.md \\
      --examples results/bench_report/examples.md --summary results/bench_report/summary.json \\
      [--cms results/bench_report/cms_sweep_summary.json] \\
      --bundle kvkk-rele07tr-bench_2026-09-09 --out results/bench_report/REPORT.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


def split_tables(md: str) -> dict[str, str]:
    parts = re.split(r"^### (T\d) — ", md, flags=re.M)
    tables = {}
    for i in range(1, len(parts), 2):
        tables[parts[i]] = (
            "**" + parts[i + 1].split("\n", 1)[0].strip() + "**\n" + parts[i + 1].split("\n", 1)[1].strip()
        )
    return tables


def cms_table(summary_json: Path) -> str:
    """From results/bench_report/cms_sweep_summary.json (per-group counts of the flag sweep)."""
    r = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = [
        "| group | docs | personal data | with a person name | Madde 6 (special category) | company ids only* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    tot = {"docs": 0, "has_kvkk": 0, "has_name": 0, "has_madde6": 0, "company_ids_only": 0}
    for g, c in sorted(r["per_group"].items(), key=lambda kv: -kv[1]["docs"]):
        rows.append(
            f"| {g} | {c['docs']} | {c['has_kvkk']} ({100 * c['has_kvkk'] / c['docs']:.0f} %) | {c['has_name']} | {c['has_madde6']} | {c['company_ids_only']} |"
        )
        for k in tot:
            tot[k] += c[k]
    rows.append(
        f"| **all** | {tot['docs']} | **{tot['has_kvkk']} ({100 * tot['has_kvkk'] / tot['docs']:.1f} %)** | {tot['has_name']} | {tot['has_madde6']} | {tot['company_ids_only']} |"
    )
    return "\n".join(rows)


def summary_bullets(summary: dict) -> str:
    e07, gt = summary["e07"], summary["vekaletname"]

    def f(x):
        return f"{100 * x:.1f}"

    b = []
    if "rele07tr" in e07 and e07["rele07tr"].get("entities"):
        t = e07["rele07tr"]
        b.append(
            f"- **e07 test**: entities micro F1 **{f(t['entities']['f1'])}** (macro {f(t['entities']['macro_f1'])}); relations strict F1 **{f(t['relations']['f1'])}** "
            f"(P {f(t['relations']['precision'])} / R {f(t['relations']['recall'])})"
            + (
                f"; multi-person subset {f(summary['e07_multiperson']['rele07tr']['f1'])}"
                if "rele07tr" in summary["e07_multiperson"]
                else ""
            )
            + "."
        )
        if "rele07-kvkk" in e07 and e07["rele07-kvkk"].get("relations"):
            b.append(
                f"- Renaming the relation types to Turkish (`rele07` → `rele07tr`) is worth **+{100 * (t['relations']['f1'] - e07['rele07-kvkk']['relations']['f1']):.1f}** relation F1 points on e07 at no entity cost "
                f"({f(e07['rele07-kvkk']['entities']['f1'])} → {f(t['entities']['f1'])})."
            )
    if "rele07tr" in gt:
        v = gt["rele07tr"]
        b.append(
            f"- **vekaletname_v1 (real documents)**: entities strict F1 **{f(v['entities']['strict']['f1'])}** (boundary-lenient {f(v['entities']['lenient']['f1'])}); "
            f"relations strict F1 **{f(v['relations_filtered']['strict']['f1'])}** with the type filter (P {f(v['relations_filtered']['strict']['precision'])} / "
            f"R {f(v['relations_filtered']['strict']['recall'])}), {f(v['relations']['strict']['f1'])} without."
        )
        zs = gt.get("gliner25-orig")
        if zs:
            b.append(
                f"- The open zero-shot base scores {f(zs['entities']['strict']['f1'])} / {f(zs['relations']['strict']['f1'])} on the same real documents; "
                f"the KVKK entity-only champion {f(gt['kvkk-champion']['entities']['strict']['f1'])} (its 19 training labels contain no person or company name)."
                if "kvkk-champion" in gt
                else ""
            )
    b.append(
        "- Relations: recall is high, precision is the gap — attributes get attached to the wrong person in multi-person documents (examples in §6); the type filter and a higher threshold (0.6–0.7) recover part of it (§5.4)."
    )
    return "\n".join(x for x in b if x)


def thr_vekaletname(gt_dir: Path) -> str:
    """One line per rele07tr vekaletname run at a non-default threshold (rele07tr_thr<x>.json), if present."""
    rows = []
    for path in sorted(gt_dir.glob("rele07tr_thr*.json")):
        if path.name.endswith("_gtonly.json"):
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        o, of = rep["overall"], rep["overall_filtered"]
        rows.append(
            f"| {rep['threshold']:g} | {100 * o['entities']['strict']['f1']:.1f} | {100 * o['relations']['strict']['precision']:.1f} | "
            f"{100 * o['relations']['strict']['recall']:.1f} | {100 * o['relations']['strict']['f1']:.1f} | "
            f"{100 * of['relations']['strict']['precision']:.1f} | {100 * of['relations']['strict']['recall']:.1f} | {100 * of['relations']['strict']['f1']:.1f} |"
        )
    if len(rows) < 2:
        return ""
    return (
        "The same check on the real documents (`rele07tr`, vekaletname_v1):\n\n| threshold | entities F1 | relations P | R | F1 | type-filtered P | R | F1 |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cms", type=Path, default=None, help="results/bench_report/cms_sweep_summary.json")
    parser.add_argument("--bundle", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    text = args.template.read_text(encoding="utf-8")
    tables = split_tables(args.tables.read_text(encoding="utf-8"))
    for key, val in tables.items():
        text = text.replace("{{" + key + "}}", val)
    text = re.sub(r"\{\{T\d\}\}", "_(table not available — result files missing)_", text)
    text = text.replace(
        "{{EXAMPLES}}",
        args.examples.read_text(encoding="utf-8") if args.examples.exists() else "_(examples not generated)_",
    )
    text = text.replace("{{SUMMARY_BULLETS}}", summary_bullets(json.loads(args.summary.read_text(encoding="utf-8"))))
    text = text.replace(
        "{{CMS_TABLE}}", cms_table(args.cms) if args.cms and args.cms.exists() else "_(sweep results not included)_"
    )
    text = text.replace("{{THR_VEKALETNAME}}", thr_vekaletname(args.summary.parent.parent / "gt_vekaletname"))
    text = text.replace("{{DATE}}", dt.date.today().isoformat()).replace("{{BUNDLE}}", args.bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
