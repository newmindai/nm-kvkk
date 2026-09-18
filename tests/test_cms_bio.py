"""BIO export for span predictions (scripts/cms_bio.py)."""

from cms_bio import bio_rows, flatten_entities, model_entities_to_flat, tokenize


def ent(label, start, end, confidence=0.9):
    return {"label": label, "start": start, "end": end, "confidence": confidence}


# ---------------------------------------------------------------- tokenize


def test_tokenize_uses_the_model_word_splitter_and_separates_clitics():
    tokens = tokenize("Ayşe'nin TCKN: 123")
    assert [t[0] for t in tokens] == ["Ayşe", "'", "nin", "TCKN", ":", "123"]
    assert tokens[0] == ("Ayşe", 0, 4) and tokens[1] == ("'", 4, 5)


# ---------------------------------------------------------------- bio_rows


def test_single_token_entity_gets_b_tag_and_confidence():
    text = "TCKN: 12345678901"
    rows = bio_rows("d1", {}, text, [0], [ent("national_id_number", 6, 17, 0.87)])
    assert len(rows) == 1
    assert rows[0]["tokens"] == ["TCKN", ":", "12345678901"]
    assert rows[0]["ner_tags"] == ["O", "O", "B-national_id_number"]
    assert rows[0]["confidences"] == [0.0, 0.0, 0.87]


def test_multi_token_entity_is_b_then_i():
    text = "Adı: Ayşe Nur Çelik ."
    rows = bio_rows("d1", {}, text, [0], [ent("full_name", 5, 19)])
    assert rows[0]["ner_tags"] == ["O", "O", "B-full_name", "I-full_name", "I-full_name", "O"]


def test_suffix_token_stays_outside_the_span():
    text = "Ayşe'nin adresi"
    rows = bio_rows("d1", {}, text, [0], [ent("first_name", 0, 4)])
    assert rows[0]["tokens"] == ["Ayşe", "'", "nin", "adresi"]
    assert rows[0]["ner_tags"] == ["B-first_name", "O", "O", "O"]


def test_row_carries_doc_metadata_and_page_number():
    rows = bio_rows("doc_x", {"group": "cv", "file_type": "Özgeçmiş"}, "a b", [0], [])
    assert rows[0]["id"] == "doc_x#p1"
    assert rows[0]["docname"] == "doc_x" and rows[0]["page"] == 1
    assert rows[0]["group"] == "cv" and rows[0]["file_type"] == "Özgeçmiş"


def test_pages_become_separate_rows():
    text = "sayfa bir\nsayfa iki"
    rows = bio_rows("d1", {}, text, [0, 10], [ent("city", 16, 19)])
    assert [r["page"] for r in rows] == [1, 2]
    assert rows[0]["tokens"] == ["sayfa", "bir"] and rows[0]["ner_tags"] == ["O", "O"]
    assert rows[1]["tokens"] == ["sayfa", "iki"] and rows[1]["ner_tags"] == ["O", "B-city"]


def test_entity_crossing_a_page_boundary_restarts_with_b_on_the_next_page():
    text = "Ahmet\nYılmaz"
    rows = bio_rows("d1", {}, text, [0, 6], [ent("full_name", 0, 12)])
    assert rows[0]["ner_tags"] == ["B-full_name"]
    assert rows[1]["ner_tags"] == ["B-full_name"]


def test_empty_page_is_skipped():
    text = "a\n\n\nb"
    rows = bio_rows("d1", {}, text, [0, 2, 3], [])
    assert [r["page"] for r in rows] == [1, 3]


def test_bio_rows_flattens_overlaps_itself():
    text = "Ayşe Nur Çelik"
    rows = bio_rows("d1", {}, text, [0], [ent("full_name", 0, 14, 0.9), ent("first_name", 0, 4, 0.95)])
    # BIO is flat: the more confident span wins, the overlapped one is dropped.
    assert rows[0]["ner_tags"] == ["B-first_name", "O", "O"]


# ---------------------------------------------------------- flatten_entities


def test_flatten_keeps_the_more_confident_of_two_overlapping_spans():
    kept = flatten_entities([ent("a", 0, 5, 0.6), ent("b", 3, 8, 0.8)])
    assert [e["label"] for e in kept] == ["b"]


def test_flatten_breaks_confidence_ties_by_length_then_position():
    kept = flatten_entities([ent("short", 0, 3, 0.7), ent("long", 0, 8, 0.7), ent("later", 9, 12, 0.7)])
    assert [e["label"] for e in kept] == ["long", "later"]


def test_flatten_output_is_sorted_by_start():
    kept = flatten_entities([ent("b", 10, 12, 0.9), ent("a", 0, 2, 0.5)])
    assert [e["start"] for e in kept] == [0, 10]


def test_flatten_keeps_adjacent_non_overlapping_spans():
    kept = flatten_entities([ent("a", 0, 4, 0.9), ent("b", 4, 8, 0.9)])
    assert len(kept) == 2


# ----------------------------------------------------- model_entities_to_flat


def test_model_output_is_mapped_from_turkish_names_to_node_ids():
    out = {"kimlik numarası": [{"text": "123", "start": 6, "end": 9, "confidence": 0.9}], "ad soyad": []}
    flat = model_entities_to_flat(out, {"kimlik numarası": "national_id_number", "ad soyad": "full_name"})
    assert flat == [{"label": "national_id_number", "text": "123", "start": 6, "end": 9, "confidence": 0.9}]
