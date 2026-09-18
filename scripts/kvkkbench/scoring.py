"""Score prediction files against gold on a label subset, with the same comparison code as the
ground-truth benchmark (scripts/gt_compare.py): strict + lenient offsets, per label, micro, macro."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from gt_compare import compare_entities, prf  # noqa: E402


def _restrict(spans: Iterable[dict], ids: set) -> list[dict]:
    return [s for s in spans if s["label"] in ids]


def score(gold_docs: list[dict], preds: dict[str, list[dict]], subset_ids: list[str]) -> dict:
    ids = set(subset_ids)
    strict, lenient = defaultdict(Counter), defaultdict(Counter)
    tot = Counter()
    n_gold = n_pred = 0
    for doc in gold_docs:
        g = _restrict(doc["spans"], ids)
        p = _restrict(preds.get(doc["docname"], []), ids)
        n_gold += len(g)
        n_pred += len(p)
        for it in compare_entities(g, p):
            gl = it["gold"]["label"] if it["gold"] else None
            pl = it["pred"]["label"] if it["pred"] else None
            tot[it["status"]] += 1
            # strict: exact offsets + label
            if it["status"] == "match":
                strict[gl]["tp"] += 1
            else:
                if pl:
                    strict[pl]["fp"] += 1
                if gl:
                    strict[gl]["fn"] += 1
            # lenient: overlapping offsets with the same label count
            if it["status"] == "match" or (it["status"] == "boundary_mismatch" and it["same_label"]):
                lenient[gl]["tp"] += 1
            else:
                if pl:
                    lenient[pl]["fp"] += 1
                if gl:
                    lenient[gl]["fn"] += 1

    def table(c):
        return {
            k: {
                **prf(v["tp"], v["fp"], v["fn"]),
                "tp": v["tp"],
                "fp": v["fp"],
                "fn": v["fn"],
                "gold": v["tp"] + v["fn"],
            }
            for k, v in sorted(c.items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"]))
        }

    st, le = table(strict), table(lenient)

    def micro(c):
        return prf(sum(v["tp"] for v in c.values()), sum(v["fp"] for v in c.values()), sum(v["fn"] for v in c.values()))

    def macro(t):  # over labels with gold support in this set
        f = [v["f1"] for v in t.values() if v["gold"] > 0]
        return {"f1": statistics.mean(f) if f else 0.0, "n_labels": len(f)}

    return {
        "n_gold": n_gold,
        "n_pred": n_pred,
        "counts": dict(tot),
        "strict": {"micro": micro(strict), "macro": macro(st), "per_label": st},
        "lenient": {"micro": micro(lenient), "macro": macro(le), "per_label": le},
    }


def load_predictions(path: Path) -> dict[str, list[dict]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["docname"]] = r["spans"]
    return out


def load_timing(path: Path) -> dict:
    tot_s = 0.0
    n_sent = 0
    chars = 0
    n_docs = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("timing", {})
        tot_s += t.get("model_s", 0.0)
        n_sent += t.get("n_units", 0)
        chars += t.get("chars", 0)
        n_docs += 1
    return {
        "model_s": tot_s,
        "n_docs": n_docs,
        "n_units": n_sent,
        "chars": chars,
        "ms_per_unit": 1000 * tot_s / n_sent if n_sent else None,
        "chars_per_s": chars / tot_s if tot_s else None,
        "s_per_doc": tot_s / n_docs if n_docs else None,
    }


def markdown_table(rows: list[dict], subset: str) -> str:
    hdr = (
        "| model | unit | hardware | strict P | R | **F1** | macro F1 | lenient F1 | lenient macro | n pred | ms/unit | s/doc |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = ""
    for r in sorted(rows, key=lambda r: -r["strict"]["micro"]["f1"]):
        s, l, t = r["strict"], r["lenient"], r["timing"]
        body += (
            f"| {r['model']} | {r.get('unit', 'sentence')} | {r.get('hardware', '')} | {s['micro']['precision'] * 100:.1f} | {s['micro']['recall'] * 100:.1f} | "
            f"**{s['micro']['f1'] * 100:.1f}** | {s['macro']['f1'] * 100:.1f} | {l['micro']['f1'] * 100:.1f} | {l['macro']['f1'] * 100:.1f} | {r['n_pred']} | "
            f"{t['ms_per_unit']:.0f} | {t['s_per_doc']:.2f} |\n"
            if t.get("ms_per_unit") is not None
            else f"| {r['model']} | {r.get('unit', 'sentence')} | {r.get('hardware', '')} | {s['micro']['precision'] * 100:.1f} | {s['micro']['recall'] * 100:.1f} | "
            f"**{s['micro']['f1'] * 100:.1f}** | {s['macro']['f1'] * 100:.1f} | {l['micro']['f1'] * 100:.1f} | {l['macro']['f1'] * 100:.1f} | {r['n_pred']} | — | — |\n"
        )
    return f"Subset **{subset}** ({len(rows)} models), gold restricted to the subset labels.\n\n" + hdr + body
