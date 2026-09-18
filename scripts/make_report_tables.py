"""Print the Markdown tables of REPORT.md from results/bench_report/summary.json
(written by scripts/make_charts.py) so every number in the report is reproducible.

Usage:  python scripts/make_report_tables.py [--summary results/bench_report/summary.json] [--out results/bench_report/tables.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def get(d: dict[str, Any], *keys) -> float | None:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/bench_report/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("results/bench_report/tables.md"))
    args = parser.parse_args(argv)
    s = json.loads(args.summary.read_text(encoding="utf-8"))
    models = s["models"]
    e07, mp, gt, thr = s["e07"], s["e07_multiperson"], s["vekaletname"], s["thresholds"]
    out = []

    out.append("### T1 — Headline: micro F1 at threshold 0.5, same data for every model\n")
    out.append(
        "| model | relations queried as | e07 test · entities | e07 test · relations (strict) | vekaletname · entities | vekaletname · relations (strict) | vekaletname · relations (type-filtered) |"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for m in models:
        t = m["tag"]
        rel = m["relations"] != "none"
        out.append(
            f"| {m['name']} | {m['relations']} | {pct(get(e07, t, 'entities', 'f1'))} | {pct(get(e07, t, 'relations', 'f1')) if rel else 'n/a'} | "
            f"{pct(get(gt, t, 'entities', 'strict', 'f1'))} | {pct(get(gt, t, 'relations', 'strict', 'f1')) if rel else 'n/a'} | "
            f"{pct(get(gt, t, 'relations_filtered', 'strict', 'f1')) if rel else 'n/a'} |"
        )

    out.append("\n### T2 — synthetic-v2 (e07) test set, 100 documents · threshold 0.5\n")
    out.append(
        "| model | entities P | R | F1 (micro) | F1 (macro) | relations P | R | F1 strict | F1 direction-lenient | multi-person F1 |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        t = m["tag"]
        e, r = e07.get(t, {}).get("entities"), e07.get(t, {}).get("relations")
        out.append(
            f"| {m['name']} | {pct(get(e or {}, 'precision'))} | {pct(get(e or {}, 'recall'))} | {pct(get(e or {}, 'f1'))} | {pct(get(e or {}, 'macro_f1'))} | "
            f"{pct(get(r or {}, 'precision'))} | {pct(get(r or {}, 'recall'))} | {pct(get(r or {}, 'f1'))} | {pct(get(r or {}, 'lenient_f1'))} | "
            f"{pct(get(mp, t, 'f1'))} |"
        )

    out.append("\n### T3 — vekaletname_v1, 20 real documents · threshold 0.5\n")
    out.append(
        "| model | entities P | R | F1 strict | F1 boundary-lenient | match / label≠ / boundary≠ / FP / FN | relations P | R | F1 strict | F1 type-filtered | P (filtered) | R (filtered) |"
    )
    out.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        t = m["tag"]
        v = gt.get(t)
        if not v:
            out.append(f"| {m['name']} | n/a | | | | | | | | | | |")
            continue
        c = v["entities"]["counts"]
        rel = m["relations"] != "none"
        rp = lambda *k: pct(get(v, *k)) if rel else "n/a"
        out.append(
            f"| {m['name']} | {pct(v['entities']['strict']['precision'])} | {pct(v['entities']['strict']['recall'])} | {pct(v['entities']['strict']['f1'])} | "
            f"{pct(v['entities']['lenient']['f1'])} | {c['match']} / {c['label_mismatch']} / {c['boundary_mismatch']} / {c['fp']} / {c['fn']} | "
            f"{rp('relations', 'strict', 'precision')} | {rp('relations', 'strict', 'recall')} | {rp('relations', 'strict', 'f1')} | "
            f"{rp('relations_filtered', 'strict', 'f1')} | {rp('relations_filtered', 'strict', 'precision')} | {rp('relations_filtered', 'strict', 'recall')} |"
        )

    if "rele07tr" in gt:
        out.append("\n### T4 — vekaletname_v1 per label, rele07tr vs rele07 (kvkk base) · exact span + label\n")
        out.append("| label (taxonomy id) | gold | rele07tr P | R | F1 | rele07 F1 |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for k, v in gt["rele07tr"]["per_label"].items():
            if v["gold"] == 0:
                continue
            o = gt.get("rele07-kvkk", {}).get("per_label", {}).get(k, {})
            out.append(
                f"| {k} | {v['gold']} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {pct(o.get('f1'))} |"
            )
        fp_only = [(k, v["pred"]) for k, v in gt["rele07tr"]["per_label"].items() if v["gold"] == 0]
        if fp_only:
            out.append(
                "\nPredicted labels with no gold support (rele07tr): "
                + ", ".join(f"{k} ({n})" for k, n in fp_only)
                + "."
            )

        out.append("\n### T5 — vekaletname_v1 per relation type, rele07tr · type-filtered\n")
        out.append("| relation type | gold | pred | TP | P | R | F1 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for k, v in gt["rele07tr"]["per_relation_filtered"].items():
            if v["gold"] == 0 and v["pred"] < 2:
                continue
            out.append(
                f"| {k} | {v['gold']} | {v['pred']} | {v['tp']} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} |"
            )

    if thr:
        out.append("\n### T6 — e07 test relations vs decision threshold (strict F1; P / R for rele07tr)\n")
        ts = sorted({float(t) for v in thr.values() for t in v})
        out.append("| model | " + " | ".join(f"{t:g}" for t in ts) + " |")
        out.append("|---|" + "---:|" * len(ts))
        for m in models:
            t = m["tag"]
            if t in thr:
                out.append(f"| {m['name']} | " + " | ".join(pct(get(thr[t], f"{x:g}", "f1")) for x in ts) + " |")
        if "rele07tr" in thr:
            out.append(
                "| rele07tr precision | "
                + " | ".join(pct(get(thr["rele07tr"], f"{x:g}", "precision")) for x in ts)
                + " |"
            )
            out.append(
                "| rele07tr recall | " + " | ".join(pct(get(thr["rele07tr"], f"{x:g}", "recall")) for x in ts) + " |"
            )

    if "rele07tr" in e07 and e07["rele07tr"].get("relations"):
        out.append("\n### T7 — e07 test per relation type (gold support ≥ 5), strict F1\n")
        pt = e07["rele07tr"]["relations"]["per_type"]
        rows = [k for k, v in sorted(pt.items(), key=lambda kv: -kv[1]["support"]) if v["support"] >= 5]
        cols = [m for m in models if e07.get(m["tag"], {}).get("relations")]
        out.append("| type | gold | " + " | ".join(m["tag"] for m in cols) + " |")
        out.append("|---|---:|" + "---:|" * len(cols))
        for k in rows:
            out.append(
                f"| {k} | {pt[k]['support']} | "
                + " | ".join(pct(get(e07[m["tag"]]["relations"]["per_type"], k, "f1")) for m in cols)
                + " |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
