"""String-level span scoring for GLiNER2 entity extraction.

GLiNER2 training data carries entities as surface strings (no offsets), and the
model itself treats every occurrence of a string as one mention. This module
scores at the same granularity: per label, the *set* of normalized surface
strings predicted for a sentence is compared with the gold set.

``score_examples`` is pure. ``make_compute_metrics`` builds the hook that
``ExtractorTrainer(compute_metrics=...)`` calls with ``(model, eval_dataset)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

Gold = Mapping[str, Iterable[Any]]
Example = tuple[Gold, Gold]


def _normalize(value: Any) -> str:
    if isinstance(value, dict):  # tolerate include_spans / include_confidence output
        value = value.get("text", "")
    return " ".join(str(value).split())


def _mentions(entities: Gold | None, label: str) -> set:
    if not entities:
        return set()
    return {m for m in (_normalize(v) for v in (entities.get(label) or [])) if m}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision/recall/F1 with the sklearn zero-division convention (0.0)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_examples(examples: Sequence[Example], labels: Sequence[str]) -> dict[str, Any]:
    """Exact string-set matching per label.

    Returns ``{"per_label": {label: {tp, fp, fn, support, precision, recall, f1}},
    "micro": {...}, "macro": {precision, recall, f1, n_labels}}``. Macro averages
    only labels with gold support; micro pools the counts of all labels.
    """
    counts = {label: [0, 0, 0, 0] for label in labels}  # tp, fp, fn, support
    for gold, pred in examples:
        for label in labels:
            g, p = _mentions(gold, label), _mentions(pred, label)
            c = counts[label]
            c[0] += len(g & p)
            c[1] += len(p - g)
            c[2] += len(g - p)
            c[3] += len(g)

    per_label = {}
    for label in labels:
        tp, fp, fn, support = counts[label]
        precision, recall, f1 = _prf(tp, fp, fn)
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    precision, recall, f1 = _prf(tp, fp, fn)
    micro = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    supported = [per_label[label] for label in labels if per_label[label]["support"] > 0]
    n = len(supported)
    macro = {
        "precision": sum(s["precision"] for s in supported) / n if n else 0.0,
        "recall": sum(s["recall"] for s in supported) / n if n else 0.0,
        "f1": sum(s["f1"] for s in supported) / n if n else 0.0,
        "n_labels": n,
    }
    return {"per_label": per_label, "micro": micro, "macro": macro}


def _record_spec(record: Mapping[str, Any]):
    """The (spec, labels) one record declares; validates descriptions cover the labels."""
    output = record["output"]
    names = list((output.get("entities") or {}).keys())
    descriptions = output.get("entity_descriptions") or None
    if descriptions:
        missing = [n for n in names if n not in descriptions]
        if missing:
            raise ValueError(f"record {record.get('id')}: entity_descriptions missing for {missing}")
        return {name: descriptions[name] for name in names}, names
    return names, names


def _schema_groups(records: Sequence[Mapping[str, Any]]):
    """Group record indices by declared schema (label names + descriptions).

    A mixed eval set (rows from several datasets, e.g. KVKK + NER) yields one
    group per distinct schema, in first-seen order; extraction then runs once
    per group so every record is queried with exactly the labels it declares.
    Returns ``[(spec, labels, indices)]``.
    """
    indices: dict[tuple, list[int]] = {}
    specs: dict[tuple, tuple[Any, list[str]]] = {}
    for i, record in enumerate(records):
        spec, labels = _record_spec(record)
        key = (tuple(labels), tuple(sorted(spec.items())) if isinstance(spec, dict) else None)
        indices.setdefault(key, []).append(i)
        specs[key] = (spec, labels)
    return [(specs[key][0], specs[key][1], rows) for key, rows in indices.items()]


def flatten_report(report: dict[str, Any], prefix: str = "exact") -> dict[str, float]:
    flat = {}
    for avg in ("micro", "macro"):
        for key in ("precision", "recall", "f1"):
            flat[f"{prefix}_{key}_{avg}"] = report[avg][key]
    for label, stats in report["per_label"].items():
        for key in ("precision", "recall", "f1", "support"):
            flat[f"{key}/{label}"] = stats[key]
    return flat


def make_compute_metrics(
    batch_size: int = 16,
    threshold: float = 0.5,
    errors_path: Path | None = None,
    max_errors: int = 200,
) -> Callable[[Any, Any], dict[str, float]]:
    """Build the ``compute_metrics(model, eval_dataset)`` hook for ExtractorTrainer.

    The hook runs ``model.batch_extract_entities`` over ``eval_dataset.data`` using
    the label spec declared in the records (names, or name -> description when the
    records carry ``entity_descriptions``). Records with different schemas (a mixed
    eval set) are extracted in one batch per schema and scored into one merged
    report. It scores exact string matches, optionally
    writes mismatching rows to ``errors_path`` (JSONL, keyed by ``id``), and returns
    flat scalar metrics. The full report of the last call is kept on
    ``hook.last_report`` for printing.
    """

    def compute_metrics(model, eval_dataset) -> dict[str, float]:
        # Under DDP the trainer passes the DistributedDataParallel wrapper;
        # batch_extract_entities lives on the wrapped model.
        model = getattr(model, "module", model)
        records = list(eval_dataset.data)
        if not records:
            return {}
        # The trainer calls this hook on every rank with the full eval set.
        # Shard extraction (the expensive part) across ranks, all-gather the
        # predictions, then score the merged set identically on every rank so
        # save_best sees the same metric everywhere.
        import torch.distributed as dist

        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if world > 1 else 0
        predictions: list[Any] = [None] * len(records)
        group_of: list[int] = [0] * len(records)
        labels: list[str] = []
        for group_index, (spec, group_labels, rows) in enumerate(_schema_groups(records)):
            labels.extend(name for name in group_labels if name not in labels)
            for i in rows:
                group_of[i] = group_index
            mine = [i for i in rows if i % world == rank]
            if not mine:
                continue
            group_predictions = model.batch_extract_entities(
                [records[i]["input"] for i in mine], spec, batch_size=batch_size, threshold=threshold
            )
            if len(group_predictions) != len(mine):
                raise ValueError(
                    f"model returned {len(group_predictions)} predictions "
                    f"for {len(mine)} texts (schema group {group_index})"
                )
            for i, prediction in zip(mine, group_predictions):
                predictions[i] = prediction
        if world > 1:
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, predictions)
            for other in gathered:
                for i, prediction in enumerate(other):
                    if prediction is not None:
                        predictions[i] = prediction
        examples: list[Example] = []
        mismatches: list[tuple[int, dict]] = []  # (record index, row); group_of gives the schema
        for index, (record, prediction) in enumerate(zip(records, predictions)):
            gold = record["output"].get("entities") or {}
            pred = (prediction or {}).get("entities") or {}
            examples.append((gold, pred))
            if any(_mentions(gold, label) != _mentions(pred, label) for label in labels):
                mismatches.append(
                    (index, {"id": record.get("id"), "text": record["input"], "gold": gold, "predicted": pred})
                )
        report = score_examples(examples, labels)
        compute_metrics.last_report = report
        if errors_path is not None:
            # Cap round-robin across schema groups (a mixed eval file is usually
            # block-ordered, and a flat cap would starve the later datasets),
            # then restore record order for readability.
            per_group: dict[int, list[tuple[int, dict]]] = {}
            for index, row in mismatches:
                per_group.setdefault(group_of[index], []).append((index, row))
            interleaved = (
                [item for batch in zip_longest(*per_group.values()) for item in batch if item is not None]
                if per_group
                else []
            )
            chosen = sorted(interleaved[:max_errors], key=lambda item: item[0])
            target = Path(errors_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                for _, row in chosen:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        flat = flatten_report(report)
        flat["n_examples"] = len(records)
        flat["n_mismatched_examples"] = len(mismatches)
        return flat

    compute_metrics.last_report = None
    return compute_metrics


__all__ = ["score_examples", "make_compute_metrics", "flatten_report"]
