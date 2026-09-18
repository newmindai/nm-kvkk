"""Compare model predictions with the vekaletname ground truth (entities + relations).

Gold record (datasets/gt/<set>/gt.jsonl): `spans` = non-overlapping NER view
[{start, end, label, text, entity_id}], `entities` = [{id, label, span, role, same_as,
mentions:[{start,end}]}], `relations` = [{head, relation, tail}] over entity ids.
Predictions: flat entity spans [{label, start, end, text, confidence}] (taxonomy node ids)
and relations [{relation, head:{text,start,end,confidence}, tail:{...}}] (taxonomy ids).

Entity statuses: match (same offsets + label) · label_mismatch (same offsets, other label) ·
boundary_mismatch (overlapping offsets; `same_label` says whether the label agrees) · fp · fn.
Relation statuses: match · reversed (head/tail swapped) · fp · fn. Relation arguments match
a gold entity when the predicted span overlaps ANY mention of that entity or of an entity
linked to it by `same_as` (coreference-aware, like scripts/eval_relations.py). Matching is
greedy one-to-one; exact wins before overlap.

Scores: strict = match only; lenient = + boundary_mismatch with the same label (entities)
or + reversed (relations).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


# ------------------------------------------------------------------ entities


def compare_entities(gold: Sequence[dict[str, Any]], pred: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One item per matched pair / unmatched span, sorted by position."""
    gold_left = list(range(len(gold)))
    pred_left = list(range(len(pred)))
    items: list[dict[str, Any]] = []

    def take(gi: int, pi: int, status: str) -> None:
        gold_left.remove(gi)
        pred_left.remove(pi)
        items.append(
            {
                "status": status,
                "start": min(gold[gi]["start"], pred[pi]["start"]),
                "end": max(gold[gi]["end"], pred[pi]["end"]),
                "gold": gold[gi],
                "pred": pred[pi],
                "same_label": gold[gi]["label"] == pred[pi]["label"],
            }
        )

    # 1. exact offsets + label
    for gi in list(gold_left):
        for pi in list(pred_left):
            if (gold[gi]["start"], gold[gi]["end"], gold[gi]["label"]) == (
                pred[pi]["start"],
                pred[pi]["end"],
                pred[pi]["label"],
            ):
                take(gi, pi, "match")
                break
    # 2. exact offsets, other label
    for gi in list(gold_left):
        for pi in list(pred_left):
            if (gold[gi]["start"], gold[gi]["end"]) == (pred[pi]["start"], pred[pi]["end"]):
                take(gi, pi, "label_mismatch")
                break
    # 3. overlapping offsets, largest overlap first
    pairs = sorted(((_overlap(gold[gi], pred[pi]), gi, pi) for gi in gold_left for pi in pred_left), reverse=True)
    for ov, gi, pi in pairs:
        if ov > 0 and gi in gold_left and pi in pred_left:
            take(gi, pi, "boundary_mismatch")
    for pi in pred_left:
        items.append(
            {
                "status": "fp",
                "start": pred[pi]["start"],
                "end": pred[pi]["end"],
                "gold": None,
                "pred": pred[pi],
                "same_label": False,
            }
        )
    for gi in gold_left:
        items.append(
            {
                "status": "fn",
                "start": gold[gi]["start"],
                "end": gold[gi]["end"],
                "gold": gold[gi],
                "pred": None,
                "same_label": False,
            }
        )
    return sorted(items, key=lambda i: (i["start"], i["end"]))


def score_entities(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(i["status"] for i in items)
    n_pred = counts["match"] + counts["label_mismatch"] + counts["boundary_mismatch"] + counts["fp"]
    n_gold = counts["match"] + counts["label_mismatch"] + counts["boundary_mismatch"] + counts["fn"]
    strict_tp = counts["match"]
    lenient_tp = strict_tp + sum(1 for i in items if i["status"] == "boundary_mismatch" and i["same_label"])
    return {
        "strict": prf(strict_tp, n_pred - strict_tp, n_gold - strict_tp),
        "lenient": prf(lenient_tp, n_pred - lenient_tp, n_gold - lenient_tp),
        "counts": {k: counts.get(k, 0) for k in ("match", "label_mismatch", "boundary_mismatch", "fp", "fn")},
        "n_gold": n_gold,
        "n_pred": n_pred,
    }


# ------------------------------------------------------------------ relations


def entity_mentions(rec: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    """entity id -> all (start, end) mentions of it and of every entity coreferent via same_as."""
    parent = {e["id"]: e["id"] for e in rec["entities"]}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    for e in rec["entities"]:
        if e.get("same_as") and e["same_as"] in parent:
            parent[find(e["id"])] = find(e["same_as"])
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for e in rec["entities"]:
        groups[find(e["id"])].extend((m["start"], m["end"]) for m in e["mentions"])
    return {e["id"]: sorted(set(groups[find(e["id"])])) for e in rec["entities"]}


def _hits(span: dict[str, Any], mentions: Sequence[tuple[int, int]]) -> bool:
    return any(min(span["end"], b) - max(span["start"], a) > 0 for a, b in mentions)


def compare_relations(rec: dict[str, Any], pred: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    mentions = entity_mentions(rec)
    gold = rec["relations"]
    gold_left = list(range(len(gold)))
    pred_left = list(range(len(pred)))
    items: list[dict[str, Any]] = []

    def take(gi: int, pi: int, status: str) -> None:
        gold_left.remove(gi)
        pred_left.remove(pi)
        items.append({"status": status, "relation": gold[gi]["relation"], "gold": gold[gi], "pred": pred[pi]})

    for strict in (True, False):
        for pi in list(pred_left):
            for gi in list(gold_left):
                if gold[gi]["relation"] != pred[pi]["relation"]:
                    continue
                h, t = mentions.get(gold[gi]["head"], []), mentions.get(gold[gi]["tail"], [])
                if strict and _hits(pred[pi]["head"], h) and _hits(pred[pi]["tail"], t):
                    take(gi, pi, "match")
                    break
                if not strict and _hits(pred[pi]["head"], t) and _hits(pred[pi]["tail"], h):
                    take(gi, pi, "reversed")
                    break
    for pi in pred_left:
        items.append({"status": "fp", "relation": pred[pi]["relation"], "gold": None, "pred": pred[pi]})
    for gi in gold_left:
        items.append({"status": "fn", "relation": gold[gi]["relation"], "gold": gold[gi], "pred": None})
    return items


def score_relations(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(i["status"] for i in items)
    n_pred = counts["match"] + counts["reversed"] + counts["fp"]
    n_gold = counts["match"] + counts["reversed"] + counts["fn"]
    strict_tp, lenient_tp = counts["match"], counts["match"] + counts["reversed"]
    return {
        "strict": prf(strict_tp, n_pred - strict_tp, n_gold - strict_tp),
        "lenient": prf(lenient_tp, n_pred - lenient_tp, n_gold - lenient_tp),
        "counts": {k: counts.get(k, 0) for k in ("match", "reversed", "fp", "fn")},
        "n_gold": n_gold,
        "n_pred": n_pred,
    }


# ------------------------------------------------------------------ type constraints


def load_constraints(path, tr_to_id: dict[str, str], tr_to_rel: dict[str, str]) -> dict[str, set]:
    """configs/relation_constraints.json ({turkish relation: [[head tr label, tail tr label], ...]},
    the (head label, tail label) pairs observed in training) -> {relation id: {(head id, tail id)}}.
    Entries whose relation or labels are not in the bridges are skipped."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed: dict[str, set] = {}
    for rel_tr, pairs in raw.items():
        rel = tr_to_rel.get(rel_tr)
        if rel is None:
            continue
        for head_tr, tail_tr in pairs:
            if head_tr in tr_to_id and tail_tr in tr_to_id:
                allowed.setdefault(rel, set()).add((tr_to_id[head_tr], tr_to_id[tail_tr]))
    return allowed


def apply_type_constraints(
    pred_relations: Sequence[dict[str, Any]], pred_entities: Sequence[dict[str, Any]], allowed: dict[str, set]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split predicted relations into (kept, dropped) by the training-derived type constraints.

    A side's label is the label of the predicted entity span overlapping it. With every label
    queried, a side that overlaps no predicted entity is invalid (mirrors scripts/demo.py's
    flag_type_violations under a full query). Types without a constraint entry pass through.
    Dropped items get `violation` = reason string."""

    def label_at(span: dict[str, Any]):
        best, best_ov = None, 0
        for e in pred_entities:
            ov = _overlap(span, e)
            if ov > best_ov:
                best, best_ov = e["label"], ov
        return best

    kept, dropped = [], []
    for r in pred_relations:
        pairs = allowed.get(r["relation"])
        if not pairs:
            kept.append(r)
            continue
        h, t = label_at(r["head"]), label_at(r["tail"])
        if h is None or t is None:
            dropped.append(r | {"violation": f"unlabelled side (head={h}, tail={t})"})
        elif (h, t) not in pairs:
            dropped.append(r | {"violation": f"({h}, {t}) never seen for {r['relation']} in training"})
        else:
            kept.append(r)
    return kept, dropped


# ------------------------------------------------------------------ aggregate


def aggregate(docs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Micro scores over documents ({entity_items, relation_items}) plus per-label / per-relation tables."""
    ent_items: list[dict[str, Any]] = []
    rel_items: list[dict[str, Any]] = []
    for d in docs:
        ent_items.extend(d["entity_items"])
        rel_items.extend(d["relation_items"])

    per_label: dict[str, Counter] = defaultdict(Counter)
    for i in ent_items:
        if i["gold"]:
            per_label[i["gold"]["label"]]["gold"] += 1
        if i["pred"]:
            per_label[i["pred"]["label"]]["pred"] += 1
        if i["status"] == "match":
            per_label[i["gold"]["label"]]["tp"] += 1
        else:
            if i["pred"]:
                per_label[i["pred"]["label"]]["fp"] += 1
            if i["gold"]:
                per_label[i["gold"]["label"]]["fn"] += 1
    per_relation: dict[str, Counter] = defaultdict(Counter)
    for i in rel_items:
        c = per_relation[i["relation"]]
        if i["gold"]:
            c["gold"] += 1
        if i["pred"]:
            c["pred"] += 1
        if i["status"] == "match":
            c["tp"] += 1
        else:
            if i["pred"]:
                c["fp"] += 1
            if i["gold"]:
                c["fn"] += 1

    def table(counters: dict[str, Counter]) -> dict[str, dict[str, Any]]:
        return {
            k: {"gold": c["gold"], "pred": c["pred"], **prf(c["tp"], c["fp"], c["fn"])}
            for k, c in sorted(counters.items(), key=lambda kv: (-kv[1]["gold"], kv[0]))
        }

    return {
        "entities": score_entities(ent_items),
        "relations": score_relations(rel_items),
        "per_label": table(per_label),
        "per_relation": table(per_relation),
    }
