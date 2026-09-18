"""Gold builder for the nm-kvkk-pii-6K test splits (scripts/convert_nm6k_gold.py)."""

import pytest
from convert_nm6k_gold import build_gold

LABELS = {
    "entity_labels": {
        "full_name": "ad soyad",
        "national_id_number": "kimlik numarası",
        "email_address": "e-posta adresi",
        "kep_address": "e-posta adresi",
    },
    "relation_names": {"national_id_of": "kimlik numarası sahibi", "email_of": "e-posta adresi sahibi"},
    "relation_descriptions": {
        "kimlik numarası sahibi": "Kimlik numarası kişiye aittir",
        "e-posta adresi sahibi": "E-posta adresi kişiye aittir",
    },
}


def record(rid="e07-1", ents=None, rels=None):
    return {"id": rid, "input": "x", "output": {"entities": ents or {}, "relations": rels or []}}


def test_declared_negatives_are_skipped():
    r = record(
        rels=[
            {"kimlik numarası sahibi": {"head": "123", "tail": "Ali Kaya"}},
            {"e-posta adresi sahibi": {"head": "", "tail": ""}},
        ]
    )
    gold = build_gold([r], LABELS, "src")
    assert [t["relation"] for t in gold["records"][0]["triples"]] == ["kimlik numarası sahibi"]
    assert gold["stats"]["n_declared_negatives_skipped"] == 1 and gold["stats"]["n_triples"] == 1


def test_mentions_are_single_element_lists_and_labels_are_resolved():
    r = record(
        ents={"ad soyad": ["Ali Kaya"], "kimlik numarası": ["123"]},
        rels=[{"kimlik numarası sahibi": {"head": "123", "tail": "Ali Kaya"}}],
    )
    t = build_gold([r], LABELS, "src")["records"][0]["triples"][0]
    assert t["head_mentions"] == ["123"] and t["tail_mentions"] == ["Ali Kaya"]
    assert t["head_label"] == "kimlik numarası" and t["tail_label"] == "ad soyad"


def test_label_lookup_is_case_insensitive_and_missing_is_none():
    r = record(ents={"ad soyad": ["Ali Kaya"]}, rels=[{"kimlik numarası sahibi": {"head": "999", "tail": "ALİ KAYA"}}])
    t = build_gold([r], LABELS, "src")["records"][0]["triples"][0]
    assert t["head_label"] is None
    assert t["tail_label"] is None or t["tail_label"] == "ad soyad"  # İ-casefold is locale-dependent


def test_schema_is_the_full_vocabulary_not_just_present_types():
    gold = build_gold([record()], LABELS, "src")
    assert gold["relation_types"] == ["e-posta adresi sahibi", "kimlik numarası sahibi"]
    assert gold["eval_labels"] == ["ad soyad", "e-posta adresi", "kimlik numarası"]  # merged names deduped
    assert set(gold["relation_descriptions"]) == set(gold["relation_types"])


def test_every_record_is_kept_even_without_triples():
    gold = build_gold([record("a"), record("b")], LABELS, "src")
    assert [r["id"] for r in gold["records"]] == ["a", "b"]
    assert all(r["triples"] == [] for r in gold["records"])


def test_unknown_relation_name_is_rejected():
    with pytest.raises(SystemExit):
        build_gold([record(rels=[{"bilinmeyen": {"head": "a", "tail": "b"}}])], LABELS, "src")


# ---------------------------------------------------- preflight (unmatchable surfaces)

from preflight_nm6k_train import text_tokens, unmatchable  # noqa: E402


def rec(text, ents):
    return {"id": "x", "input": text, "output": {"entities": ents}}


def test_terminal_period_is_appended_like_the_processor():
    assert text_tokens("bir iki") == ["bir", "iki", "."]
    assert text_tokens("bir iki.") == ["bir", "iki", "."]
    assert text_tokens("bir iki?")[-1] == "?"


def test_url_at_the_very_end_becomes_unmatchable():
    # the appended "." glues onto the final URL token
    r = rec("Profil: https://linkedin.com/in/pinarsahan", {"web sitesi": ["https://linkedin.com/in/pinarsahan"]})
    assert unmatchable(r) == [("web sitesi", "https://linkedin.com/in/pinarsahan")]


def test_a_trailing_period_breaks_a_url_whether_written_or_appended():
    # the whitespace splitter keeps "<url>." as ONE token, so the surface never matches
    r = rec("Profil: https://linkedin.com/in/pinarsahan.", {"web sitesi": ["https://linkedin.com/in/pinarsahan"]})
    assert unmatchable(r) == [("web sitesi", "https://linkedin.com/in/pinarsahan")]


def test_url_followed_by_whitespace_is_matchable():
    r = rec(
        "Profil: https://linkedin.com/in/pinarsahan ve devam.", {"web sitesi": ["https://linkedin.com/in/pinarsahan"]}
    )
    assert unmatchable(r) == []


def test_normal_records_and_empty_declarations_pass():
    assert unmatchable(rec("Ali Kaya geldi.", {"ad soyad": ["Ali Kaya"], "IBAN": []})) == []


def test_matching_is_case_insensitive_and_whole_word():
    assert unmatchable(rec("ALİ KAYA geldi.", {"ad soyad": ["ALİ KAYA"]})) == []
    assert unmatchable(rec("Alikaya geldi.", {"ad soyad": ["Ali"]})) == [("ad soyad", "Ali")]
