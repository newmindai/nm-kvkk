"""Turn model entity spans into the BIO JSONL rows the project's converters
consume ({"tokens": [...], "ner_tags": ["O", "B-X", "I-X", ...]}, see
scripts/convert_bio_to_gliner.py).

Tokens come from GLiNER2's own whitespace word splitter (apostrophes and
punctuation are separate tokens, so "Ayşe'nin" -> Ayşe ' nin and a suffix can
stay outside a span). Tags use taxonomy node ids (ASCII), never the Turkish
query names. BIO is flat, so overlapping predictions are resolved first:
highest confidence, then longest, then earliest. One row per PAGE; an entity
that crosses a page boundary restarts with B- on the next page.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "GLiNER2"))  # local clone wins over any installed gliner2

from gliner2.processing.word_splitter import resolve_word_splitter  # noqa: E402

_SPLITTER = resolve_word_splitter(None)

Token = tuple[str, int, int]


def tokenize(text: str) -> list[Token]:
    """(token, char_start, char_end) using the model's word splitter, original case."""
    return list(_SPLITTER(text, lower=False))


def flatten_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedy non-overlapping selection: confidence desc, length desc, start asc; result sorted by start."""
    ranked = sorted(entities, key=lambda e: (-float(e.get("confidence", 0.0)), -(e["end"] - e["start"]), e["start"]))
    kept: list[dict[str, Any]] = []
    for candidate in ranked:
        if all(candidate["end"] <= k["start"] or candidate["start"] >= k["end"] for k in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda e: e["start"])


def model_entities_to_flat(entities: dict[str, list[dict[str, Any]]], tr_to_id: dict[str, str]) -> list[dict[str, Any]]:
    """{turkish label: [{text,start,end,confidence}]} -> [{label: node id, text, start, end, confidence}].

    Raises KeyError for a label outside the bridge so a schema drift is loud, not silent.
    """
    flat: list[dict[str, Any]] = []
    for name, mentions in entities.items():
        node_id = tr_to_id[name]
        for m in mentions:
            flat.append(
                {
                    "label": node_id,
                    "text": m["text"],
                    "start": int(m["start"]),
                    "end": int(m["end"]),
                    "confidence": float(m.get("confidence", 0.0)),
                }
            )
    return sorted(flat, key=lambda e: (e["start"], e["end"]))


def _pages(text: str, page_offsets: Sequence[int]) -> list[tuple[int, int, int]]:
    """[(page_no, start, end)] from page start offsets."""
    starts = sorted(int(o) for o in page_offsets) or [0]
    if starts[0] != 0:
        starts.insert(0, 0)
    bounds = starts + [len(text)]
    return [(i + 1, bounds[i], bounds[i + 1]) for i in range(len(starts))]


def bio_rows(
    docname: str, meta: dict[str, Any], text: str, page_offsets: Sequence[int], entities: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One BIO row per non-empty page. `entities` = [{label, start, end, confidence}] in document offsets."""
    flat = flatten_entities(entities)
    tokens = tokenize(text)
    rows: list[dict[str, Any]] = []
    for page_no, p_start, p_end in _pages(text, page_offsets):
        page_tokens = [t for t in tokens if p_start <= t[1] < p_end]
        if not page_tokens:
            continue
        tags, confs = [], []
        open_entity = None  # the entity the previous token belonged to (within this page)
        for _, t_start, t_end in page_tokens:
            hit = next((e for e in flat if t_start < e["end"] and t_end > e["start"]), None)
            if hit is None:
                tags.append("O")
                confs.append(0.0)
                open_entity = None
            else:
                tags.append(("I-" if hit is open_entity else "B-") + hit["label"])
                confs.append(round(hit["confidence"], 4))
                open_entity = hit
        rows.append(
            {
                "id": f"{docname}#p{page_no}",
                "docname": docname,
                "page": page_no,
                "group": meta.get("group"),
                "file_type": meta.get("file_type"),
                "tokens": [t[0] for t in page_tokens],
                "ner_tags": tags,
                "confidences": confs,
            }
        )
    return rows
