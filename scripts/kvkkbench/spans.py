"""Span utilities shared by every adapter: token offsets, BIO decoding, normalisation, stacking."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

TRAILING_PUNCT = ",.;:!?)]}\"'"
LEADING_PUNCT = "([{\"'"
WORD_RE = re.compile(r"\S+")


def token_offsets(tokens: Sequence[str], text: str) -> list[tuple[int, int]]:
    """Offsets for tokens whose concatenation is the text (the :5550/:5558 servers' tokens carry
    their trailing spaces). Falls back to sequential search when the concatenation differs."""
    offs, cur = [], 0
    if "".join(tokens) == text:
        for t in tokens:
            offs.append((cur, cur + len(t)))
            cur += len(t)
        return offs
    for t in tokens:
        core = t.strip()
        i = text.find(core, cur) if core else cur
        if i < 0:
            i = cur
        offs.append((i, i + len(core)))
        cur = i + len(core)
    return offs


def words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def decode_bio(tags: Sequence[str], offsets: Sequence[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """BIO/IOB2 tags over tokens -> (start, end, type). An I- without a matching open span starts one."""
    spans, cur = [], None
    for tag, (s, e) in zip(tags, offsets):
        if not tag or tag == "O":
            if cur:
                spans.append(tuple(cur))
                cur = None
            continue
        prefix, _, typ = tag.partition("-")
        if prefix == "B" or cur is None or cur[2] != typ:
            if cur:
                spans.append(tuple(cur))
            cur = [s, e, typ]
        else:
            cur[1] = e
    if cur:
        spans.append(tuple(cur))
    return spans


def normalize(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace, then unbalanced leading/trailing punctuation — applied to every model alike."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while end > start and text[end - 1] in TRAILING_PUNCT:
        end -= 1
    while start < end and text[start] in LEADING_PUNCT:
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def make_span(
    text: str, start: int, end: int, label: str, native: str | None = None, confidence: float | None = None
) -> dict | None:
    start, end = (
        max(0, min(int(start), len(text))),
        max(0, min(int(end), len(text))),
    )  # GLiNER can report end = len+1 (virtual terminal period)
    s, e = normalize(text, start, end)
    if e <= s:
        return None
    d = {"start": s, "end": e, "label": label, "text": text[s:e]}
    if native is not None:
        d["native_label"] = native
    if confidence is not None:
        d["confidence"] = round(float(confidence), 4)
    return d


def overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def stack(primary: Iterable[dict], secondary: Iterable[dict]) -> list[dict]:
    """Production priority merge: keep every primary span, add secondary spans that overlap none of them."""
    out = list(primary)
    for s in secondary:
        if not any(overlaps(s, p) for p in out):
            out.append(s)
    return sorted(out, key=lambda x: (x["start"], x["end"]))
