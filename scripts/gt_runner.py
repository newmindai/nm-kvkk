"""Shared runner for the ground-truth benchmark (vekaletname_v1 and any set in the
same gt.jsonl schema): loads a checkpoint, queries it with the taxonomy entity
labels (Turkish) and the relation types in the naming the model was trained on,
and compares against gold with scripts/gt_compare.py.

Used by scripts/bench_gt.py; scripts/eval_all-style recipes call that.

Relation naming modes:
  tr    Turkish names + Turkish descriptions (configs/labels/e07_relation_mapping_tr.json) — rele07tr
  en    English type ids + English descriptions (configs/labels/e07_mapping.json) — rele07, relprobe29, pristine
  none  entities only (kvkk-champion: relation head frozen dead; piifilter: no relation head)
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for candidate in (HERE, PROJECT_ROOT / "GLiNER2"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cms_bio import flatten_entities, model_entities_to_flat  # noqa: E402
from gt_compare import (
    aggregate,
    apply_type_constraints,
    compare_entities,
    compare_relations,  # noqa: E402
    load_constraints,
    score_entities,
    score_relations,
)


def find_file(*candidates: Path) -> Path | None:
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def load_gt(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_relation_predictions(raw: dict, name_to_rel: dict) -> list:
    """{query name: [{head:{text,start,end,confidence}, tail:{...}} | (head, tail)]} -> flat list keyed by relation id."""
    preds = []
    for name, instances in raw.items():
        rel_id = name_to_rel.get(name, name)
        for inst in instances:
            if isinstance(inst, dict) and isinstance(inst.get("head"), dict):
                head, tail = inst["head"], inst["tail"]
            else:  # surfaces only (no include_spans) — offsets unknown
                h, t = inst if isinstance(inst, (list, tuple)) else (inst.get("head"), inst.get("tail"))
                head, tail = {"text": h, "start": -1, "end": -1}, {"text": t, "start": -1, "end": -1}
            preds.append(
                {
                    "relation": rel_id,
                    "head": dict(head),
                    "tail": dict(tail),
                    "confidence": min(float(head.get("confidence", 0) or 0), float(tail.get("confidence", 0) or 0)),
                }
            )
    return preds


class GTRunner:
    def __init__(
        self,
        model: Path,
        device: str = "auto",
        relation_mode: str = "tr",
        labels: Path | None = None,
        relation_map_tr: Path | None = None,
        mapping_en: Path | None = None,
        constraints: Path | None = None,
    ):
        import torch

        from gliner2 import AutoExtractor

        labels = find_file(labels, PROJECT_ROOT / "configs/labels/taxonomy_labels_tr.json")
        relation_map_tr = find_file(relation_map_tr, PROJECT_ROOT / "configs/labels/e07_relation_mapping_tr.json")
        mapping_en = find_file(mapping_en, PROJECT_ROOT / "configs/labels/e07_mapping.json")
        constraints = find_file(constraints, PROJECT_ROOT / "configs/relation_constraints.json")
        if labels is None or relation_map_tr is None or mapping_en is None:
            raise FileNotFoundError("taxonomy_labels_tr.json / relation_mapping_tr.json / mapping.json not found")

        self.labels = json.loads(labels.read_text(encoding="utf-8"))
        # FIRST wins: several taxonomy ids can share one query name (nm6k merges
        # email/KEP, TCKN/YKN, phone/fax). The labels file is in taxonomy order, so
        # the canonical id is the one a prediction maps back to.
        self.tr_to_id = {}
        for e in self.labels:
            self.tr_to_id.setdefault(e["tr"], e["id"])
        self.id_to_entry = {e["id"]: e for e in self.labels}
        tr_map = {
            k: v for k, v in json.loads(relation_map_tr.read_text(encoding="utf-8")).items() if not k.startswith("_")
        }
        en_map = json.loads(mapping_en.read_text(encoding="utf-8"))
        en_desc = en_map.get("relation_descriptions", {})
        self.relation_mode = relation_mode
        if relation_mode == "tr":
            self.relations = {k: {"name": v["tr"], "desc": v["desc_tr"]} for k, v in tr_map.items()}
        elif relation_mode == "en":
            self.relations = {
                k: {"name": k, "desc": en_desc[k]} for k in en_map.get("relation_types", []) if k in en_desc
            }
        else:
            self.relations = {}
        self.rel_tr = {k: v["tr"] for k, v in tr_map.items()}
        self.name_to_rel = {v["name"]: k for k, v in self.relations.items()}
        self.allowed = (
            load_constraints(constraints, self.tr_to_id, {v["tr"]: k for k, v in tr_map.items()}) if constraints else {}
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_path = Path(model)
        self.name = self.model_path.name
        started = time.time()
        self.model = AutoExtractor.from_pretrained(str(model), map_location=device)
        self.model.eval()
        self.load_s = round(time.time() - started, 1)
        self.lock = threading.Lock()

    # ------------------------------------------------------------ one document
    def _extract(
        self,
        text: str,
        entity_names: list[str],
        rels: dict[str, str],
        threshold: float,
        prompt_mode: str,
        chunk_size: int,
    ):
        """Run the model in one of three prompt shapes and return (entities, relation_extraction) outputs.

        joint    one schema carrying every entity label and relation type — the default and what
                 the demo shows
        split    entities alone, then every relation type alone (two passes)
        chunked  entities alone, then relation types in groups of `chunk_size`
        Shorter prompts matter for encoders whose recall falls with prompt length
        for mDeBERTa the modes should agree."""
        kw = dict(threshold=threshold, include_confidence=True, include_spans=True)
        if prompt_mode == "joint" or not rels:
            schema = self.model.create_schema().entities(entity_names)
            if rels:
                schema = schema.relations(rels)
            out = self.model.extract(text, schema, **kw)
            return out.get("entities", {}), out.get("relation_extraction", {})
        if prompt_mode not in ("split", "chunked"):
            raise ValueError(f"unknown prompt_mode {prompt_mode!r}")
        ent_out = self.model.extract(text, self.model.create_schema().entities(entity_names), **kw).get("entities", {})
        items = list(rels.items())
        size = len(items) if prompt_mode == "split" else max(1, chunk_size)
        rel_out: dict[str, Any] = {}
        for i in range(0, len(items), size):
            out = self.model.extract(text, self.model.create_schema().relations(dict(items[i : i + size])), **kw)
            rel_out.update(out.get("relation_extraction", {}))
        return ent_out, rel_out

    def run_doc(
        self,
        rec: dict[str, Any],
        threshold: float = 0.5,
        gt_only: bool = False,
        gt_labels: set | None = None,
        gt_relations: set | None = None,
        prompt_mode: str = "joint",
        chunk_size: int = 20,
    ) -> dict[str, Any]:
        entries = [e for e in self.labels if not gt_only or (gt_labels and e["id"] in gt_labels)]
        rels = {
            v["name"]: v["desc"]
            for k, v in self.relations.items()
            if not gt_only or (gt_relations and k in gt_relations)
        }
        with self.lock:
            started = time.time()
            ent_out, rel_out = self._extract(
                rec["text"], [e["tr"] for e in entries], rels, threshold, prompt_mode, chunk_size
            )
            elapsed = round(time.time() - started, 2)

        pred_entities = flatten_entities(model_entities_to_flat(ent_out, self.tr_to_id))
        pred_relations = parse_relation_predictions(rel_out, self.name_to_rel)
        kept, dropped = apply_type_constraints(pred_relations, pred_entities, self.allowed)

        entity_items = compare_entities(rec["spans"], pred_entities)
        relation_items = compare_relations(rec, pred_relations)
        relation_items_filtered = compare_relations(rec, kept)
        return {
            "docname": rec["docname"],
            "threshold": threshold,
            "gt_only": gt_only,
            "relation_mode": self.relation_mode,
            "prompt_mode": prompt_mode,
            "chunk_size": chunk_size if prompt_mode == "chunked" else None,
            "n_labels_queried": len(entries),
            "n_relations_queried": len(rels),
            "elapsed_s": elapsed,
            "pred_entities": [e | {"tr": self.id_to_entry[e["label"]]["tr"]} for e in pred_entities],
            "pred_relations": [r | {"tr": self.rel_tr.get(r["relation"], r["relation"])} for r in pred_relations],
            "dropped_relations": [r | {"tr": self.rel_tr.get(r["relation"], r["relation"])} for r in dropped],
            "entity_items": entity_items,
            "relation_items": relation_items,
            "relation_items_filtered": relation_items_filtered,
            "scores": {
                "entities": score_entities(entity_items),
                "relations": score_relations(relation_items),
                "relations_filtered": score_relations(relation_items_filtered),
            },
        }

    # ------------------------------------------------------------ report
    def build_report(
        self,
        results: dict[str, dict[str, Any]],
        gt_name: str,
        threshold: float,
        gt_only: bool,
        n_docs_total: int,
        prompt_mode: str = "joint",
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        agg = aggregate(results.values())
        agg_f = aggregate(
            [
                {"entity_items": r["entity_items"], "relation_items": r["relation_items_filtered"]}
                for r in results.values()
            ]
        )
        return {
            "model": self.name,
            "model_path": str(self.model_path),
            "relation_mode": self.relation_mode,
            "gt": gt_name,
            "threshold": threshold,
            "gt_only": gt_only,
            "device": self.device,
            "prompt_mode": prompt_mode,
            "chunk_size": chunk_size,
            "n_docs": len(results),
            "n_docs_total": n_docs_total,
            "overall": agg,
            "overall_filtered": {"relations": agg_f["relations"], "per_relation": agg_f["per_relation"]},
            "per_doc": {d: {"scores": r["scores"], "elapsed_s": r["elapsed_s"]} for d, r in sorted(results.items())},
            "predictions": {
                d: {
                    "entities": r["pred_entities"],
                    "relations": r["pred_relations"],
                    "dropped_relations": r["dropped_relations"],
                }
                for d, r in sorted(results.items())
            },
            "items": {
                d: {"entities": r["entity_items"], "relations": r["relation_items"]} for d, r in sorted(results.items())
            },
        }
