"""Unit tests for the cross-model benchmark library (scripts/kvkkbench)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kvkkbench import labels as L  # noqa: E402
from kvkkbench import scoring as SC  # noqa: E402
from kvkkbench import sentencize as Z  # noqa: E402
from kvkkbench import spans as S  # noqa: E402


def test_token_offsets_concatenation_and_fallback():
    text = "Ahmet Yılmaz'ın no 123."
    toks = ["Ahmet ", "Yılmaz", "'", "ın ", "no ", "123", "."]
    assert "".join(toks) == text
    offs = S.token_offsets(toks, text)
    assert [text[s:e] for s, e in offs] == toks
    # fallback: tokens without the trailing spaces
    offs2 = S.token_offsets(["Ahmet", "Yılmaz", "'", "ın", "no", "123", "."], text)
    assert text[offs2[1][0] : offs2[1][1]] == "Yılmaz" and text[offs2[5][0] : offs2[5][1]] == "123"


def test_decode_bio_handles_B_I_and_orphan_I():
    tags = ["O", "B-PNO", "I-PNO", "I-PNO", "O", "I-IDN", "B-EMA", "B-EMA"]
    offs = [(0, 1), (2, 6), (7, 10), (11, 13), (14, 15), (16, 27), (28, 40), (41, 50)]
    assert S.decode_bio(tags, offs) == [(2, 13, "PNO"), (16, 27, "IDN"), (28, 40, "EMA"), (41, 50, "EMA")]


def test_normalize_strips_space_and_trailing_punctuation():
    text = "tel 0532 123 45 67, e-posta (ahmet@x.com)."
    assert text[slice(*S.normalize(text, 4, 19))] == "0532 123 45 67"
    assert text[slice(*S.normalize(text, 28, 42))] == "ahmet@x.com"
    assert S.make_span(text, 4, 4, "x") is None


def test_stack_keeps_primary_and_non_overlapping_secondary():
    ner = [{"start": 0, "end": 12, "label": "full_name"}]
    kvkk = [
        {"start": 5, "end": 9, "label": "national_id_number"},
        {"start": 20, "end": 31, "label": "national_id_number"},
    ]
    out = S.stack(ner, kvkk)
    assert [(s["start"], s["end"]) for s in out] == [(0, 12), (20, 31)]


def test_align_sentences_verbatim_and_whitespace_insensitive():
    text = "Birinci cümle.  İkinci\ncümle burada. Üçüncü."
    spans = Z.align(text, ["Birinci cümle.", "İkinci cümle burada.", "Üçüncü."])
    assert [text[s:e] for s, e in spans] == ["Birinci cümle.", "İkinci\ncümle burada.", "Üçüncü."]
    with pytest.raises(ValueError):
        Z.align(text, ["yok böyle bir cümle"])


def test_local_split_offsets_cover_sentences():
    text = "Ahmet geldi. Sonra gitti! Peki?"
    assert [text[s:e] for s, e in Z.local_split(text)] == ["Ahmet geldi.", "Sonra gitti!", "Peki?"]


def test_scoring_strict_lenient_macro():
    gold = [
        {
            "docname": "d",
            "text": "x",
            "spans": [
                {"start": 0, "end": 5, "label": "full_name"},
                {"start": 10, "end": 21, "label": "national_id_number"},
                {"start": 30, "end": 34, "label": "iban"},
            ],
        }
    ]
    preds = {
        "d": [
            {"start": 0, "end": 5, "label": "full_name"},  # strict tp
            {"start": 10, "end": 22, "label": "national_id_number"},  # boundary -> lenient tp
            {"start": 50, "end": 55, "label": "phone_number"},  # fp
            {"start": 60, "end": 65, "label": "job_title"},
        ]
    }  # outside subset: ignored
    r = SC.score(gold, preds, L.STACK21)
    assert r["n_gold"] == 3 and r["n_pred"] == 3
    assert r["strict"]["micro"]["f1"] == pytest.approx(2 * (1 / 3) * (1 / 3) / (2 / 3))
    assert r["lenient"]["micro"]["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    assert r["strict"]["macro"]["n_labels"] == 3
    assert r["strict"]["per_label"]["iban"]["fn"] == 1 and r["lenient"]["per_label"]["national_id_number"]["tp"] == 1


def test_subset_names_cover_all_ids_in_every_vocab():
    for vocab in L.VOCABS:
        names = L.subset_names(vocab, "stack21")
        assert set(names.values()) == set(L.STACK21)
        assert len(names) == 21
    assert set(L.MAPPING_KVKK.values()) == set(L.KVKK19)


def test_nm6k_loader_makes_offset_spans(tmp_path):
    import json

    from kvkkbench import data as D

    rec = {
        "id": "t1",
        "input": "Ahmet Yılmaz 12345678901 numaralı. Ahmet Yılmaz tekrar.",
        "output": {
            "entities": {"ad soyad": ["Ahmet Yılmaz"], "kimlik numarası": ["12345678901"], "IBAN": []},
            "relations": [],
        },
    }
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    docs = D.load_nm6k(p)
    labels = sorted((s["label"], docs[0]["text"][s["start"] : s["end"]]) for s in docs[0]["spans"])
    assert labels == [
        ("full_name", "Ahmet Yılmaz"),
        ("full_name", "Ahmet Yılmaz"),
        ("national_id_number", "12345678901"),
    ]


def test_make_span_clamps_offsets_past_the_text():
    text = "Ahmet Yılmaz"
    sp = S.make_span(text, 6, 13, "full_name")
    assert sp is not None and sp["text"] == "Yılmaz" and sp["end"] == 12
