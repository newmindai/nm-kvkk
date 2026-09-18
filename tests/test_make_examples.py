"""Inline markup for report examples (scripts/make_examples.py)."""

from make_examples import markup

LABELS = {
    "full_name": {"tr": "ad soyad"},
    "company_tax_number": {"tr": "şirket vergi numarası"},
    "city": {"tr": "şehir"},
}


def item(status, gold=None, pred=None, same_label=False):
    return {"status": status, "gold": gold, "pred": pred, "same_label": same_label}


def test_markup_all_statuses_and_labels_with_special_characters():
    text = "Ali Kaya 123 İzmir Ankara"
    items = [
        item(
            "match",
            gold={"start": 0, "end": 8, "label": "full_name"},
            pred={"start": 0, "end": 8, "label": "full_name"},
        ),
        item("fp", pred={"start": 9, "end": 12, "label": "company_tax_number"}),
        item("fn", gold={"start": 13, "end": 18, "label": "city"}),
        item(
            "label_mismatch",
            gold={"start": 19, "end": 25, "label": "city"},
            pred={"start": 19, "end": 25, "label": "full_name"},
        ),
    ]
    assert (
        markup(text, items, LABELS)
        == "[Ali Kaya]{ad soyad} [123]{şirket vergi numarası}? ⟨İzmir⟩{şehir} [Ankara]{ad soyad≠şehir}"
    )


def test_markup_drops_overlapping_marks_and_keeps_text_intact():
    text = "abcdef"
    items = [
        item("fp", pred={"start": 0, "end": 4, "label": "city"}),
        item("fn", gold={"start": 2, "end": 6, "label": "city"}),
    ]
    out = markup(text, items, LABELS)
    assert out == "[abcd]{şehir}?ef"
