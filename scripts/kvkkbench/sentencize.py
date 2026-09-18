"""Sentence splitting with document offsets: a regex splitter, applied identically to every model.
`align` maps a list of sentence strings back onto the text (used by the tests and by any external splitter you
plug in)."""

from __future__ import annotations

import re

_LOCAL = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9\"(])")


def local_split(text: str) -> list[tuple[int, int]]:
    spans, cur = [], 0
    for m in _LOCAL.finditer(text):
        if m.start() > cur:
            spans.append((cur, m.start()))
        cur = m.end()
    if cur < len(text):
        spans.append((cur, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def align(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    """Locate each returned sentence in the text (verbatim, then whitespace-insensitively)."""
    spans, cur = [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        i = text.find(s, cur)
        if i >= 0:
            spans.append((i, i + len(s)))
            cur = i + len(s)
            continue
        pat = r"\s*".join(re.escape(w) for w in s.split())
        m = re.compile(pat).search(text, cur)
        if not m:
            raise ValueError(f"sentence not found in text: {s[:60]!r}")
        spans.append((m.start(), m.end()))
        cur = m.end()
    return spans


def split_with_offsets(text: str) -> tuple[list[tuple[int, int]], str]:
    """-> ([(start, end)], source); source is always 'local' in this release."""
    return local_split(text), "local"
