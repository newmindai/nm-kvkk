"""Stratified pilot subset: every label gets a floor, negatives get a floor, long rows are excluded,
selection is deterministic and the output is the converter's JSONL format."""

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from make_pilot import build_pilot, select_pilot_rows


def _rows(n=400, seed=0):
    rng = random.Random(seed)
    labels = ["A", "B", "C"]
    rows = []
    for i in range(n):
        n_words = rng.choice([5, 10, 20, 60])
        text = " ".join(f"w{j}" for j in range(n_words))
        ents = []
        if rng.random() > 0.1:  # 10 % negatives
            for lab in labels:
                if rng.random() < {"A": 0.6, "B": 0.3, "C": 0.05}[lab]:
                    ents.append({"start": 0, "end": 2, "label": lab, "text": "w0"})
        rows.append({"id": f"r{i}", "text": text, "entities": ents})
    return rows


def _meta(rows):
    return [{"n_words": len(r["text"].split()), "labels": {e["label"] for e in r["entities"]}} for r in rows]


def test_selection_respects_floors_length_and_target():
    rows = _rows()
    picked = select_pilot_rows(_meta(rows), target=120, min_per_label=15, min_negatives=8, max_words=30, seed=1)
    assert len(picked) == 120 and len(set(picked)) == 120
    assert all(len(rows[i]["text"].split()) <= 30 for i in picked)
    for lab in ["A", "B", "C"]:
        assert sum(lab in {e["label"] for e in rows[i]["entities"]} for i in picked) >= 15
    assert sum(not rows[i]["entities"] for i in picked) >= 8


def test_selection_is_deterministic_and_sorted():
    meta = _meta(_rows())
    a = select_pilot_rows(meta, target=50, min_per_label=5, min_negatives=3, max_words=30, seed=7)
    b = select_pilot_rows(meta, target=50, min_per_label=5, min_negatives=3, max_words=30, seed=7)
    assert a == b == sorted(a)


def test_build_pilot_writes_converter_records(tmp_path):
    rows = _rows(200)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")
    label_map = {
        "A": {"name": "a-name", "description": "d"},
        "B": {"name": "b-name", "description": "d"},
        "C": {"name": "c-name", "description": "d"},
    }
    (tmp_path / "labels.json").write_text(json.dumps(label_map), encoding="utf-8")
    out = build_pilot(
        tmp_path / "train.parquet",
        tmp_path / "labels.json",
        tmp_path / "pilot.jsonl",
        target=40,
        min_per_label=5,
        min_negatives=2,
        max_words=30,
        seed=3,
    )
    lines = [json.loads(l) for l in Path(out).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 40
    assert all(set(l) == {"id", "input", "output"} for l in lines)
    assert all(list(l["output"]["entities"]) == ["a-name", "b-name", "c-name"] for l in lines)
    assert "entity_descriptions" not in lines[0]["output"]
    stats = json.loads((tmp_path / "pilot.stats.json").read_text(encoding="utf-8"))
    assert stats["rows"] == 40 and stats["negatives"] >= 2
