"""Benchmark sets in one shape: [{docname, text, spans: [{start, end, label}]}] with taxonomy ids."""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VEKALETNAME = PROJECT_ROOT / "datasets/gt/vekaletname_v1/gt.jsonl"
NM6K_TEST = PROJECT_ROOT / "datasets/nm6k/tr/test.jsonl"
NM6K_LABELS = PROJECT_ROOT / "configs/labels/nm6k_labels_tr.json"


def load_vekaletname(path: Path = VEKALETNAME) -> list[dict]:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [
        {
            "docname": r["docname"],
            "text": r["text"],
            "spans": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in r["spans"]],
        }
        for r in recs
    ]


def _find_all(text: str, surface: str) -> list[tuple]:
    pat = re.compile(r"(?<!\w)" + re.escape(surface) + r"(?!\w)")
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def load_nm6k(path: Path = NM6K_TEST, labels: Path = NM6K_LABELS, limit: int = 0) -> list[dict]:
    """nm6k gold is (label name -> surfaces); every whole-word occurrence of a surface becomes a span,
    overlaps resolved longest-first. Label names map to taxonomy ids (first entry wins)."""
    name_to_id: dict[str, str] = {}
    for e in json.loads(labels.read_text(encoding="utf-8")):
        name_to_id.setdefault(e["tr"], e["id"])
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r["input"]
        cands = []
        for name, surfaces in r["output"]["entities"].items():
            node = name_to_id[name]
            for surf in surfaces:
                for s, e in _find_all(text, surf):
                    cands.append((e - s, s, e, node))
        cands.sort(key=lambda c: (-c[0], c[1]))
        taken: list[dict] = []
        for _, s, e, node in cands:
            if not any(s < t["end"] and t["start"] < e for t in taken):
                taken.append({"start": s, "end": e, "label": node})
        out.append({"docname": r["id"], "text": text, "spans": sorted(taken, key=lambda t: (t["start"], t["end"]))})
        if limit and len(out) >= limit:
            break
    return out


MIXED_V2 = PROJECT_ROOT / "datasets/gt/mixed_v2/gt.jsonl"


def load_mixed_v2(path: Path = MIXED_V2, limit: int = 0) -> list[dict]:
    """Real-world dataset ground truth v2, accepted documents (scripts/convert_mixed_v2.py). Same schema as vekaletname_v1."""
    return load_vekaletname(path)


SETS = {"vekaletname": load_vekaletname, "nm6k": load_nm6k, "mixed_v2": load_mixed_v2}
