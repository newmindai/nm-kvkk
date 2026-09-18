"""Collect every benchmark result into one summary.json and draw the report
charts (PNG, light mode, reference palette).

Inputs (all optional per model — missing cells are shown as "n/a"):
  <e07-dir>/<tag>/entities_test.json      scripts/eval_checkpoint.py on the e07 test set
  <e07-dir>/<tag>/rel_test.json           scripts/eval_relations.py, thresholds 0.5 (+0.3/0.4/0.6/0.7)
  <e07-dir>/<tag>/rel_multiperson.json    same on the 15-doc multi-person subset
  <gt-dir>/<tag>_thr0.5.json              scripts/bench_gt.py on vekaletname_v1

Usage:
  python scripts/make_charts.py --e07-dir results/bench_e07_models --gt-dir results/gt_vekaletname --out results/bench_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Model lineage, display order and short names (the report uses the same order everywhere).
MODELS: list[dict[str, str]] = [
    {"tag": "gliner25-orig", "name": "gliner2.5-multi-v1 (zero-shot)", "relations": "en"},
    {"tag": "piifilter", "name": "gliner2-privacy-filter-PII (zero-shot)", "relations": "none"},
    {"tag": "kvkk-champion", "name": "kvkk-champion (labeldiv 6ep)", "relations": "none"},
    {"tag": "relprobe29", "name": "relprobe29 (23 docs)", "relations": "en"},
    {"tag": "rele07-orig", "name": "rele07-orig (pristine base)", "relations": "en"},
    {"tag": "rele07-kvkk", "name": "rele07 (kvkk base)", "relations": "en"},
    {"tag": "rele07tr", "name": "rele07tr (Turkish relations)", "relations": "tr"},
]
# Reference palette (dataviz skill, light mode) — categorical slots in validated order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def relation_name_map() -> dict[str, str]:
    """Turkish relation name -> relation id (rele07tr reports key per-type results by the Turkish query name)."""
    here = Path(__file__).resolve().parent
    for cand in (
        here.parent / "configs/labels/e07_relation_mapping_tr.json",
        here.parent / "data/e07/relation_mapping_tr.json",
    ):
        if cand.exists():
            raw = json.loads(cand.read_text(encoding="utf-8"))
            return {v["tr"]: k for k, v in raw.items() if not k.startswith("_")}
    return {}


def collect(e07_dir: Path, gt_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": [], "e07": {}, "e07_multiperson": {}, "vekaletname": {}, "thresholds": {}}
    tr_to_rel = relation_name_map()
    summary["relation_tr"] = {v: k for k, v in tr_to_rel.items()}
    for m in MODELS:
        tag = m["tag"]
        summary["models"].append(m)
        ent = load(e07_dir / tag / "entities_test.json")
        rel = load(e07_dir / tag / "rel_test.json")
        multi = load(e07_dir / tag / "rel_multiperson.json")
        gt = load(gt_dir / f"{tag}_thr0.5.json")
        cell: dict[str, Any] = {}
        if ent:
            mt = ent["metrics"]
            cell["entities"] = {
                "precision": mt["exact_precision_micro"],
                "recall": mt["exact_recall_micro"],
                "f1": mt["exact_f1_micro"],
                "macro_f1": mt["exact_f1_macro"],
                "per_label": ent["report"]["per_label"],
            }
        if rel:
            bt = rel["by_threshold"]
            k = "0.5" if "0.5" in bt else str(rel["primary_threshold"])
            per_type = {tr_to_rel.get(name, name): v for name, v in bt[k]["strict"]["per_type"].items()}
            cell["relations"] = {
                **bt[k]["strict"]["micro"],
                "lenient_f1": bt[k]["lenient_direction"]["micro"]["f1"],
                "per_type": per_type,
                "n_gold": rel["n_gold_triples"],
            }
            summary["thresholds"][tag] = {
                t: {
                    "precision": v["strict"]["micro"]["precision"],
                    "recall": v["strict"]["micro"]["recall"],
                    "f1": v["strict"]["micro"]["f1"],
                }
                for t, v in bt.items()
            }
        summary["e07"][tag] = cell
        if multi:
            bt = multi["by_threshold"]
            k = "0.5" if "0.5" in bt else str(multi["primary_threshold"])
            summary["e07_multiperson"][tag] = {**bt[k]["strict"]["micro"], "n_gold": multi["n_gold_triples"]}
        if gt:
            o, of = gt["overall"], gt["overall_filtered"]
            summary["vekaletname"][tag] = {
                "entities": {
                    "strict": o["entities"]["strict"],
                    "lenient": o["entities"]["lenient"],
                    "counts": o["entities"]["counts"],
                    "n_gold": o["entities"]["n_gold"],
                    "n_pred": o["entities"]["n_pred"],
                },
                "relations": {
                    "strict": o["relations"]["strict"],
                    "lenient": o["relations"]["lenient"],
                    "counts": o["relations"]["counts"],
                },
                "relations_filtered": {
                    "strict": of["relations"]["strict"],
                    "lenient": of["relations"]["lenient"],
                    "counts": of["relations"]["counts"],
                },
                "per_label": o["per_label"],
                "per_relation": o["per_relation"],
                "per_relation_filtered": of["per_relation"],
                "elapsed_s": gt.get("elapsed_s"),
                "device": gt.get("device"),
            }
    return summary


# ------------------------------------------------------------------ charts


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbar(ax, names: list[str], values: list[float | None], title: str, color: str) -> None:
    ys = list(range(len(names)))[::-1]
    for y, v in zip(ys, values):
        if v is None:
            ax.text(0.01, y, "n/a", va="center", ha="left", fontsize=9, color=MUTED)
        else:
            ax.barh(y, v, height=0.55, color=color, edgecolor="white", linewidth=1)
            ax.text(v + 0.015, y, f"{100 * v:.1f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(ys, names)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    _style(ax, title)


def fig_headline(summary: dict[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    names = [m["name"] for m in summary["models"]]
    tags = [m["tag"] for m in summary["models"]]
    rels = [m["relations"] for m in summary["models"]]
    e07, gt = summary["e07"], summary["vekaletname"]
    panels = [
        ("synthetic-v2 (e07) test · entities F1", [e07.get(t, {}).get("entities", {}).get("f1") for t in tags]),
        ("vekaletname_v1 · entities F1", [gt.get(t, {}).get("entities", {}).get("strict", {}).get("f1") for t in tags]),
        (
            "synthetic-v2 (e07) test · relations F1 (strict)",
            [e07.get(t, {}).get("relations", {}).get("f1") if r != "none" else None for t, r in zip(tags, rels)],
        ),
        (
            "vekaletname_v1 · relations F1 (strict, type-filtered)",
            [
                gt.get(t, {}).get("relations_filtered", {}).get("strict", {}).get("f1") if r != "none" else None
                for t, r in zip(tags, rels)
            ],
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    for ax, (title, values) in zip(axes.flat, panels):
        hbar(ax, names, values, title, PALETTE[0])
    fig.suptitle("Same data, every model — micro F1 at threshold 0.5", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig1_headline.png", dpi=170)
    plt.close(fig)


def fig_dots(
    rows: list[str],
    series: dict[str, dict[str, float | None]],
    title: str,
    xlabel: str,
    out: Path,
    name: str,
    row_note: dict[str, str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 2.2))
    ys = list(range(len(rows)))[::-1]
    for i, (label, values) in enumerate(series.items()):
        xs = [values.get(r) for r in rows]
        ax.scatter(
            [x for x in xs if x is not None],
            [y for x, y in zip(xs, ys) if x is not None],
            s=46,
            color=PALETTE[i],
            edgecolor="white",
            linewidth=1,
            label=label,
            zorder=3,
        )
    ax.set_yticks(ys, [f"{r}  ({row_note[r]})" if row_note and r in row_note else r for r in rows])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    for y in ys:
        ax.axhline(y, color=GRID, linewidth=0.6, zorder=1)
    _style(ax, title)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5 / (0.42 * len(rows) + 2.2) - 0.06),
        ncol=min(len(series), 4),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_threshold(summary: dict[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    thr = summary["thresholds"]
    if not thr:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    i = 0
    for m in summary["models"]:
        t = m["tag"]
        if t not in thr:
            continue
        xs = sorted(float(k) for k in thr[t])
        ys = [thr[t][f"{x:g}"]["f1"] for x in xs]
        ax1.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=PALETTE[i], label=m["name"])
        i += 1
    ax1.set_xlim(0.28, 0.72)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax1.set_xlabel("decision threshold", fontsize=9, color=INK2)
    _style(ax1, "e07 test · relations strict F1 vs threshold")
    ax1.grid(axis="y", color=GRID, linewidth=0.6)
    ax1.grid(axis="x", visible=False)
    ax1.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    t = "rele07tr" if "rele07tr" in thr else next(iter(thr))
    xs = sorted(float(k) for k in thr[t])
    for j, (metric, style) in enumerate((("precision", "-"), ("recall", "--"), ("f1", ":"))):
        ax2.plot(
            xs,
            [thr[t][f"{x:g}"][metric] for x in xs],
            style,
            marker="o",
            markersize=5,
            linewidth=2,
            color=PALETTE[j],
            label=metric,
        )
    ax2.set_xlim(0.28, 0.72)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax2.set_xlabel("decision threshold", fontsize=9, color=INK2)
    _style(ax2, f"{t} · relations precision / recall / F1 vs threshold")
    ax2.grid(axis="y", color=GRID, linewidth=0.6)
    ax2.grid(axis="x", visible=False)
    ax2.legend(frameon=False, fontsize=9, loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig4_threshold.png", dpi=170)
    plt.close(fig)


def fig_multiperson(summary: dict[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [m for m in summary["models"] if m["tag"] in summary["e07_multiperson"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.4))
    ys = list(range(len(rows)))[::-1]
    for y, m in zip(ys, rows):
        full = summary["e07"][m["tag"]]["relations"]["f1"]
        multi = summary["e07_multiperson"][m["tag"]]["f1"]
        ax.barh(
            y + 0.17,
            full,
            height=0.3,
            color=PALETTE[0],
            edgecolor="white",
            label="full test (100 docs)" if y == ys[0] else None,
        )
        ax.barh(
            y - 0.17,
            multi,
            height=0.3,
            color=PALETTE[1],
            edgecolor="white",
            label="multi-person subset (15 docs)" if y == ys[0] else None,
        )
        ax.text(full + 0.01, y + 0.17, f"{100 * full:.1f}", va="center", fontsize=8.5, color=INK)
        ax.text(multi + 0.01, y - 0.17, f"{100 * multi:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [m["name"] for m in rows])
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    _style(ax, "e07 test · relations strict F1 — all documents vs multi-person documents")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig5_multiperson.png", dpi=170)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e07-dir", type=Path, default=Path("results/bench_e07_models"))
    parser.add_argument("--gt-dir", type=Path, default=Path("results/gt_vekaletname"))
    parser.add_argument("--out", type=Path, default=Path("results/bench_report"))
    parser.add_argument("--min-gold", type=int, default=3, help="per-label / per-type charts: minimum gold support")
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["IBM Plex Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    summary = collect(args.e07_dir, args.gt_dir)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    fig_headline(summary, args.out)

    # per-label on vekaletname (labels with >= min_gold gold spans), 4 models
    gt = summary["vekaletname"]
    show = [t for t in ("gliner25-orig", "kvkk-champion", "rele07-kvkk", "rele07tr") if t in gt]
    if show:
        base = gt[show[-1]]["per_label"]
        rows = [k for k, v in base.items() if v["gold"] >= args.min_gold]
        fig_dots(
            rows,
            {
                next(m["name"] for m in MODELS if m["tag"] == t): {
                    r: gt[t]["per_label"].get(r, {}).get("f1") for r in rows
                }
                for t in show
            },
            f"vekaletname_v1 · entity F1 per label (gold support ≥ {args.min_gold})",
            "F1 (exact span + label)",
            args.out,
            "fig2_per_label_vekaletname.png",
            {r: f"n={base[r]['gold']}" for r in rows},
        )

    # per-relation-type on e07 (types with >= 5 gold triples), relation models
    e07 = summary["e07"]
    show = [t for t in ("gliner25-orig", "relprobe29", "rele07-kvkk", "rele07tr") if e07.get(t, {}).get("relations")]
    if show:
        base = e07[show[-1]]["relations"]["per_type"]
        rows = [k for k, v in sorted(base.items(), key=lambda kv: -kv[1]["support"]) if v["support"] >= 5]
        fig_dots(
            rows,
            {
                next(m["name"] for m in MODELS if m["tag"] == t): {
                    r: e07[t]["relations"]["per_type"].get(r, {}).get("f1") for r in rows
                }
                for t in show
            },
            "e07 test · relation F1 per type (gold support ≥ 5)",
            "strict F1",
            args.out,
            "fig3_per_relation_e07.png",
            {r: f"{summary['relation_tr'].get(r, '')} · n={base[r]['support']}" for r in rows},
        )

    fig_threshold(summary, args.out)
    fig_multiperson(summary, args.out)
    print(f"wrote {args.out}/summary.json and {len(list(args.out.glob('fig*.png')))} figures")


if __name__ == "__main__":
    main()
