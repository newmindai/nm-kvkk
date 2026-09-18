"""Adapter for Hugging Face BIO token classifiers (ytu-ce-cosmos/modernbert-tr-pii-ner), used the way
its model card prescribes: whitespace words in with `is_split_into_words=True`, the first sub-word's
label per word, BIO decoded over words. Sub-word aggregation on raw text splits "ahmet@ornek.com" and
"A.Ş." into several entities; word-level labelling does not."""

from __future__ import annotations

import time

from .spans import decode_bio, make_span, words_with_offsets


class HFTokenClassifier:
    def __init__(self, model_id: str, mapping: dict[str, str], device: str = "auto", max_length: int = 8192):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id).to(device).eval()
        self.id2label = self.model.config.id2label
        self.mapping = mapping
        self.model_id = model_id
        self.max_length = max_length

    def predict(self, text: str) -> tuple[list[dict], float]:
        import torch

        words = words_with_offsets(text)
        if not words:
            return [], 0.0
        t = time.perf_counter()
        enc = self.tok(
            [w for w, _, _ in words],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = enc.word_ids(0)
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        pred = logits.argmax(-1).tolist()
        tags = ["O"] * len(words)
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            tags[wid] = self.id2label[pred[pos]]
        dt = time.perf_counter() - t
        out = []
        for s, e, typ in decode_bio(tags, [(a, b) for _, a, b in words]):
            node = self.mapping.get(typ)
            if node:
                sp = make_span(text, s, e, node, native=typ)
                if sp:
                    out.append(sp)
        return sorted(out, key=lambda s: (s["start"], s["end"])), dt
