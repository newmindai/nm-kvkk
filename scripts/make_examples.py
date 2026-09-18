"""Render report examples (Markdown) from a vekaletname benchmark report
(scripts/bench_gt.py output) and, optionally, from an e07 relation report.

Inline markup (predictions vs gold, on the verbatim document text):
  [text]{label}            predicted span that matches gold (same span, same label)
  [text]{pred≠gold}        predicted span where gold has another label / other boundaries
  [text]{label}?           predicted span with no gold counterpart (FP — or a gold miss)
  ⟨text⟩{label}            gold span the model did not predict (FN)
Relations: one table per example — status, head → relation → tail, confidence.

Selection heuristics (one example each, all from real predictions):
  clean         highest relation F1 among docs with >= 3 gold relations
  wrong-person  multi-person doc with the most relation FPs whose head matched a gold head
  line-break    an entity FN whose gold text spans a line break
  gold-miss?    the most confident entity FP (candidate annotation miss)
  boundary      a boundary mismatch with the same label
  dense         (e07) the record with the most gold triples

Usage:
  python scripts/make_examples.py --report results/gt_vekaletname/rele07tr_thr0.5.json \\
      --gt datasets/gt/vekaletname_v1/gt.jsonl \\
      [--e07-report results/bench_e07_models/kvkk-tr-v1/rel_test.json --e07-entities datasets/kvkk_relations/e07tr/entities_test_e07tr.jsonl \\
       --e07-gold datasets/kvkk_relations/e07tr/gold_test_e07tr.json] --out results/bench_report/examples.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROJECT_ROOT = HERE.parent


def _first(*candidates: Path) -> Path:
    return next((c for c in candidates if c.exists()), candidates[0])


DEFAULT_LABELS = _first(
    PROJECT_ROOT / "configs/labels/taxonomy_labels_tr.json", PROJECT_ROOT / "data/taxonomy_labels_tr.json"
)
DEFAULT_REL_MAP = _first(
    PROJECT_ROOT / "configs/labels/e07_relation_mapping_tr.json", PROJECT_ROOT / "data/e07/relation_mapping_tr.json"
)


def tr_name(labels: dict[str, dict[str, Any]], node_id: str) -> str:
    return labels.get(node_id, {}).get("tr", node_id)


def markup(text: str, entity_items: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> str:
    """Inline-marked document text; items carry status/gold/pred with offsets."""
    marks = []
    for i in entity_items:
        g, p = i.get("gold"), i.get("pred")
        T = "\0"  # placeholder for the span text (labels may contain braces, so no str.format)
        if i["status"] == "match":
            marks.append((p["start"], p["end"], "[" + T + "]{" + tr_name(labels, p["label"]) + "}"))
        elif i["status"] in ("label_mismatch", "boundary_mismatch"):
            marks.append(
                (
                    p["start"],
                    p["end"],
                    "[" + T + "]{" + tr_name(labels, p["label"]) + "≠" + tr_name(labels, g["label"]) + "}",
                )
            )
            if i["status"] == "boundary_mismatch":
                marks.append((g["start"], g["end"], "⟨" + T + "⟩{gold: " + tr_name(labels, g["label"]) + "}"))
        elif i["status"] == "fp":
            marks.append((p["start"], p["end"], "[" + T + "]{" + tr_name(labels, p["label"]) + "}?"))
        elif i["status"] == "fn":
            marks.append((g["start"], g["end"], "⟨" + T + "⟩{" + tr_name(labels, g["label"]) + "}"))
    # non-overlapping render: later-start first, drop overlaps with already placed marks
    placed: list[tuple] = []
    for s, e, fmt in sorted(marks, key=lambda m: (m[0], -(m[1] - m[0]))):
        if all(e <= a or s >= b for a, b, _ in placed):
            placed.append((s, e, fmt))
    out, cursor = [], 0
    for s, e, fmt in sorted(placed):
        out.append(text[cursor:s])
        out.append(fmt.replace("\0", text[s:e]))
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


def relations_table(items: list[dict[str, Any]], gold_entities: list[dict[str, Any]], rel_tr: dict[str, str]) -> str:
    by_id = {e["id"]: e for e in gold_entities}
    icon = {"match": "✓", "reversed": "↔", "fp": "✗", "fn": "○", "dropped": "⌀"}
    order = {"match": 0, "reversed": 1, "fp": 2, "dropped": 3, "fn": 4}
    rows = ["| | head | relation | tail | conf. |", "|---|---|---|---|---|"]
    for i in sorted(items, key=lambda x: (order.get(x["status"], 9), x["relation"])):
        if i.get("pred"):
            h, t, c = i["pred"]["head"]["text"], i["pred"]["tail"]["text"], f"{i['pred']['confidence']:.2f}"
        else:
            h, t, c = (
                by_id.get(i["gold"]["head"], {}).get("span", i["gold"]["head"]),
                by_id.get(i["gold"]["tail"], {}).get("span", i["gold"]["tail"]),
                "",
            )
        note = (
            " (not in gold)"
            if i["status"] == "fp"
            else " (gold only)"
            if i["status"] == "fn"
            else " (type filter)"
            if i["status"] == "dropped"
            else ""
        )
        rows.append(
            f"| {icon.get(i['status'], i['status'])} | {_cell(h)} | `{rel_tr.get(i['relation'], i['relation'])}` {i['relation']}{note} | {_cell(t)} | {c} |"
        )
    return "\n".join(rows)


def _cell(s: Any) -> str:
    return str(s).replace("\n", " ").replace("|", "\\|")


def pick_examples(report: dict[str, Any], gt: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    docs = report["per_doc"]
    items = report["items"]
    picks: list[dict[str, Any]] = []

    def n_persons(d: str) -> int:
        return len({e["id"] for e in gt[d]["entities"] if e["label"] == "full_name" and not e.get("same_as")})

    cands = [d for d in docs if len(gt[d]["relations"]) >= 3]
    if cands:
        d = max(cands, key=lambda d: docs[d]["scores"]["relations_filtered"]["strict"]["f1"])
        picks.append({"kind": "clean", "caption": "A clean case — every gold relation recovered.", "doc": d})
    cands = [d for d in docs if n_persons(d) >= 2]
    if cands:
        d = max(cands, key=lambda d: docs[d]["scores"]["relations_filtered"]["counts"]["fp"])
        picks.append(
            {
                "kind": "wrong-person",
                "caption": "Multi-person document — attributes attached to the wrong person.",
                "doc": d,
            }
        )
    for d in docs:
        if any(
            i["status"] == "fn" and "\n" in gt[d]["text"][i["gold"]["start"] : i["gold"]["end"]]
            for i in items[d]["entities"]
        ):
            picks.append({"kind": "line-break", "caption": "Gold span broken across a line break.", "doc": d})
            break
    fps = [(i["pred"]["confidence"], d) for d in docs for i in items[d]["entities"] if i["status"] == "fp"]
    if fps:
        picks.append(
            {
                "kind": "gold-miss?",
                "caption": "Most confident prediction absent from gold — model error or annotation miss.",
                "doc": max(fps)[1],
            }
        )
    for d in docs:
        if any(i["status"] == "boundary_mismatch" and i["same_label"] for i in items[d]["entities"]):
            picks.append({"kind": "boundary", "caption": "Same label, different boundaries.", "doc": d})
            break
    seen, uniq = set(), []
    for p in picks:
        if p["doc"] not in seen:
            uniq.append(p)
            seen.add(p["doc"])
    return uniq


def render_vekaletname(
    report: dict[str, Any],
    gt: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    rel_tr: dict[str, str],
    use_filtered: bool = True,
) -> str:
    out = []
    for n, ex in enumerate(pick_examples(report, gt), start=1):
        d = ex["doc"]
        rec = gt[d]
        ents = report["items"][d]["entities"]
        rel_items = report["items"][d]["relations"]
        if use_filtered:
            dropped_keys = {
                (r["relation"], r["head"]["start"], r["tail"]["start"])
                for r in report["predictions"][d]["dropped_relations"]
            }
            rel_items = [
                i
                for i in rel_items
                if not (
                    i.get("pred")
                    and (i["relation"], i["pred"]["head"]["start"], i["pred"]["tail"]["start"]) in dropped_keys
                )
            ]
            rel_items += [
                {"status": "dropped", "relation": r["relation"], "pred": r, "gold": None}
                for r in report["predictions"][d]["dropped_relations"]
            ]
        s = report["per_doc"][d]["scores"]
        out.append(f"### Example {n} — {ex['caption']}\n")
        out.append(
            f"*{rec['file_type']}* · `{d}` · entities F1 {100 * s['entities']['strict']['f1']:.0f} · "
            f"relations F1 {100 * s['relations_filtered' if use_filtered else 'relations']['strict']['f1']:.0f}\n"
        )
        out.append("```text\n" + markup(rec["text"], ents, labels).strip() + "\n```\n")
        out.append(relations_table(rel_items, rec["entities"], rel_tr) + "\n")
    return "\n".join(out)


def render_e07(
    rel_report: dict[str, Any], entities_file: Path, gold_file: Path, rel_tr: dict[str, str], max_words: int = 320
) -> str:
    """One dense e07 record (most gold triples among records <= max_words): relation verdicts from the
    eval_relations report (details = one row per gold/predicted triple with verdict TP/FP/FN)."""
    texts = {}
    for line in Path(entities_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            texts[r["id"]] = r["input"]
    gold = json.loads(Path(gold_file).read_text(encoding="utf-8"))
    short = [r for r in gold["records"] if len(texts.get(r["id"], "").split()) <= max_words] or gold["records"]
    rec = max(short, key=lambda r: len(r["triples"]))
    thr = "0.5" if "0.5" in rel_report["by_threshold"] else str(rel_report["primary_threshold"])
    rows_for = [d for d in rel_report["by_threshold"][thr]["details"] if d.get("record") == rec["id"]]
    tr_to_id = {v: k for k, v in rel_tr.items()}
    text = texts[rec["id"]]
    out = [
        f"### Example — dense synthetic record (e07 test) · `{rec['id']}` · {len(rec['triples'])} gold triples\n",
        "```text\n" + text.strip() + "\n```\n",
    ]
    if rows_for:
        icon = {"TP": "✓", "FP": "✗", "FN": "○"}
        order = {"TP": 0, "FP": 1, "FN": 2}
        rows = ["| | head | relation | tail | conf. |", "|---|---|---|---|---|"]
        for d in sorted(rows_for, key=lambda d: (order.get(d["verdict"], 9), d["relation"])):
            rel_id = tr_to_id.get(d["relation"], d["relation"])
            note = " (not in gold)" if d["verdict"] == "FP" else " (gold only)" if d["verdict"] == "FN" else ""
            conf = f"{d['confidence']:.2f}" if d.get("confidence") is not None else ""
            rows.append(
                f"| {icon.get(d['verdict'], d['verdict'])} | {_cell(d['head'])} | `{rel_tr.get(rel_id, d['relation'])}` {rel_id}{note} | {_cell(d['tail'])} | {conf} |"
            )
        out.append("\n".join(rows) + "\n")
    else:
        out.append("_(no per-record details for this record in the relation report)_\n")
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--relation-map-tr", type=Path, default=DEFAULT_REL_MAP)
    parser.add_argument("--e07-report", type=Path, default=None)
    parser.add_argument("--e07-entities", type=Path, default=None)
    parser.add_argument("--e07-gold", type=Path, default=None)
    parser.add_argument("--unfiltered", action="store_true", help="show relations before the type filter")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    gt = {
        r["docname"]: r for r in (json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip())
    }
    labels = {e["id"]: e for e in json.loads(args.labels.read_text(encoding="utf-8"))}
    rel_tr = {
        k: v["tr"]
        for k, v in json.loads(args.relation_map_tr.read_text(encoding="utf-8")).items()
        if not k.startswith("_")
    }
    parts = [
        f"<!-- generated by scripts/make_examples.py from {args.report.name} -->\n",
        "Markup: `[text]{label}` predicted and in gold · `[text]{pred≠gold}` label or boundary disagreement · "
        "`[text]{label}?` predicted, not in gold · `⟨text⟩{label}` gold only. "
        "Relations: ✓ in gold · ✗ not in gold · ↔ reversed direction · ⌀ removed by the type filter · ○ gold only.\n",
        render_vekaletname(report, gt, labels, rel_tr, use_filtered=not args.unfiltered),
    ]
    if args.e07_report and args.e07_entities and args.e07_gold:
        parts.append(
            render_e07(
                json.loads(args.e07_report.read_text(encoding="utf-8")), args.e07_entities, args.e07_gold, rel_tr
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
