"""Parquet row -> GLiNER2 JSONL record conversion."""

import pytest
from convert_to_gliner_jsonl import row_to_record, select_indices

LABELS = {
    "PER": {"name": "kişi", "description": "kişi adı"},
    "LOC": {"name": "yer", "description": "şehir veya ülke"},
}
TEXT = "Ahmet Yılmaz İstanbul ' da Ahmet Yılmaz ile görüştü ."
ROW = {
    "id": "ner/validation/7",
    "text": TEXT,
    "entities": [
        {"start": 0, "end": 12, "label": "PER", "text": "Ahmet Yılmaz"},
        {"start": 13, "end": 21, "label": "LOC", "text": "İstanbul"},
        {"start": 27, "end": 39, "label": "PER", "text": "Ahmet Yılmaz"},
    ],
}


def test_entities_become_label_name_to_unique_surface_strings():
    rec = row_to_record(ROW, LABELS)
    assert rec["output"]["entities"] == {"kişi": ["Ahmet Yılmaz"], "yer": ["İstanbul"]}


def test_every_label_is_declared_even_when_absent():
    row = dict(ROW, entities=[ROW["entities"][0]])
    rec = row_to_record(row, LABELS)
    assert rec["output"]["entities"] == {"kişi": ["Ahmet Yılmaz"], "yer": []}


def test_row_without_entities_declares_all_labels_empty():
    rec = row_to_record(dict(ROW, entities=[]), LABELS)
    assert rec["output"]["entities"] == {"kişi": [], "yer": []}


def test_input_text_and_id_are_preserved_verbatim():
    rec = row_to_record(ROW, LABELS)
    assert rec["input"] == TEXT
    assert rec["id"] == "ner/validation/7"


def test_surface_string_is_sliced_from_offsets_not_stored_text():
    row = dict(ROW, entities=[dict(ROW["entities"][1], text="WRONG")])
    rec = row_to_record(row, LABELS)
    assert rec["output"]["entities"]["yer"] == ["İstanbul"]


def test_descriptions_included_only_when_requested():
    with_desc = row_to_record(ROW, LABELS, with_descriptions=True)
    assert with_desc["output"]["entity_descriptions"] == {"kişi": "kişi adı", "yer": "şehir veya ülke"}
    assert "entity_descriptions" not in row_to_record(ROW, LABELS)["output"]


def test_unknown_label_code_raises():
    row = dict(ROW, entities=[dict(ROW["entities"][0], label="XXX")])
    with pytest.raises(KeyError):
        row_to_record(row, LABELS)


def test_label_order_follows_the_label_map():
    rec = row_to_record(ROW, {"LOC": LABELS["LOC"], "PER": LABELS["PER"]})
    assert list(rec["output"]["entities"]) == ["yer", "kişi"]


def test_select_indices_is_deterministic_sorted_and_bounded():
    a = select_indices(total=1000, n=10, seed=42)
    b = select_indices(total=1000, n=10, seed=42)
    assert a == b
    assert a == sorted(a) and len(set(a)) == 10 and all(0 <= i < 1000 for i in a)
    assert select_indices(total=1000, n=10, seed=1) != a


def test_select_indices_returns_everything_when_n_is_not_positive_or_too_large():
    assert select_indices(total=5, n=0, seed=42) == [0, 1, 2, 3, 4]
    assert select_indices(total=5, n=99, seed=42) == [0, 1, 2, 3, 4]
