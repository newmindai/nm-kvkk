"""Adapter for our GLiNER2 extractors (and the open GLiNER2 baselines)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .spans import make_span

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GlinerModel:
    def __init__(self, path: str, names: dict[str, str], device: str = "auto", threshold: float = 0.5):
        for c in (PROJECT_ROOT / "GLiNER2",):
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
        import torch

        from gliner2 import AutoExtractor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model = AutoExtractor.from_pretrained(
            str(PROJECT_ROOT / path) if not Path(path).is_absolute() else path, map_location=device
        )
        self.model.eval()
        self.names = names  # {query name: taxonomy id}
        self.threshold = threshold
        self.schema = self.model.create_schema().entities(list(names))

    def predict(self, text: str) -> tuple[list[dict], float]:
        t = time.perf_counter()
        out = self.model.extract(
            text, self.schema, threshold=self.threshold, include_confidence=True, include_spans=True
        )
        dt = time.perf_counter() - t
        spans = []
        for name, mentions in out.get("entities", {}).items():
            node = self.names[name]
            for m in mentions:
                sp = make_span(text, int(m["start"]), int(m["end"]), node, native=name, confidence=m.get("confidence"))
                if sp:
                    spans.append(sp)
        return sorted(spans, key=lambda s: (s["start"], s["end"])), dt
