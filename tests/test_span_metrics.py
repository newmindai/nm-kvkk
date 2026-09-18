"""String-level span scoring used for the zero-shot baseline and for fine-tuning-time eval."""

import json

import pytest
from span_metrics import make_compute_metrics, score_examples


def label(report, name):
    return report["per_label"][name]


def test_exact_match_is_a_true_positive():
    r = score_examples([({"kişi": ["Ahmet Yılmaz"]}, {"kişi": ["Ahmet Yılmaz"]})], labels=["kişi"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fp"], label(r, "kişi")["fn"]) == (1, 0, 0)
    assert label(r, "kişi")["f1"] == 1.0


def test_boundary_miss_counts_as_fp_and_fn():
    r = score_examples([({"kişi": ["Ahmet Yılmaz"]}, {"kişi": ["Ahmet"]})], labels=["kişi"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fp"], label(r, "kişi")["fn"]) == (0, 1, 1)


def test_wrong_label_counts_against_both_labels():
    r = score_examples([({"kişi": ["Ahmet"], "yer": []}, {"kişi": [], "yer": ["Ahmet"]})], labels=["kişi", "yer"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fn"]) == (0, 1)
    assert (label(r, "yer")["tp"], label(r, "yer")["fp"]) == (0, 1)


def test_duplicate_strings_collapse_to_one_mention():
    r = score_examples([({"kişi": ["Ahmet", "Ahmet"]}, {"kişi": ["Ahmet", "Ahmet", "Ahmet"]})], labels=["kişi"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fp"], label(r, "kişi")["fn"]) == (1, 0, 0)


def test_predictions_with_no_gold_are_false_positives_with_zero_scores():
    r = score_examples([({"kişi": []}, {"kişi": ["Ahmet"]})], labels=["kişi"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fp"], label(r, "kişi")["fn"]) == (0, 1, 0)
    assert (label(r, "kişi")["precision"], label(r, "kişi")["recall"], label(r, "kişi")["f1"]) == (0.0, 0.0, 0.0)


def test_missing_predictions_are_false_negatives():
    r = score_examples([({"kişi": ["Ahmet"]}, {"kişi": []})], labels=["kişi"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fp"], label(r, "kişi")["fn"]) == (0, 0, 1)
    assert label(r, "kişi")["recall"] == 0.0


def test_label_missing_from_prediction_dict_is_treated_as_no_predictions():
    r = score_examples([({"kişi": ["Ahmet"]}, {})], labels=["kişi"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fn"]) == (0, 1)


def test_whitespace_differences_are_normalized_before_matching():
    r = score_examples([({"kişi": ["Ahmet  Yılmaz "]}, {"kişi": ["Ahmet Yılmaz"]})], labels=["kişi"])
    assert label(r, "kişi")["tp"] == 1


def test_matching_is_case_sensitive():
    r = score_examples([({"kişi": ["Ahmet"]}, {"kişi": ["ahmet"]})], labels=["kişi"])
    assert (label(r, "kişi")["tp"], label(r, "kişi")["fp"], label(r, "kişi")["fn"]) == (0, 1, 1)


def test_micro_pools_counts_and_macro_averages_labels_with_support():
    examples = [
        ({"a": ["x"], "b": ["y"], "c": []}, {"a": ["x"], "b": [], "c": []}),
    ]
    r = score_examples(examples, labels=["a", "b", "c"])
    assert (r["micro"]["tp"], r["micro"]["fp"], r["micro"]["fn"]) == (1, 0, 1)
    assert r["micro"]["precision"] == 1.0
    assert r["micro"]["recall"] == 0.5
    assert r["micro"]["f1"] == pytest.approx(2 / 3)
    # macro over labels with gold support only: a (1.0) and b (0.0); c has no support
    assert r["macro"]["f1"] == pytest.approx(0.5)
    assert r["macro"]["n_labels"] == 2
    assert label(r, "c")["support"] == 0


def test_support_counts_unique_gold_mentions_per_example():
    examples = [({"a": ["x", "x"]}, {"a": []}), ({"a": ["x"]}, {"a": []})]
    r = score_examples(examples, labels=["a"])
    assert label(r, "a")["support"] == 2


class FakeExtractor:
    """Stands in for AutoExtractor: records the label spec it was asked for."""

    def __init__(self, answer):
        self.answer = answer
        self.specs = []

    def batch_extract_entities(self, texts, entity_types, batch_size=8, threshold=0.5, **_):
        self.specs.append(entity_types)
        return [{"entities": dict(self.answer)} for _ in texts]


class FakeDataset:
    def __init__(self, records):
        self.data = records


def test_hook_passes_plain_label_list_when_records_have_no_descriptions():
    model = FakeExtractor({"kişi": ["Ahmet"], "yer": []})
    ds = FakeDataset([{"id": "r0", "input": "Ahmet geldi .", "output": {"entities": {"kişi": ["Ahmet"], "yer": []}}}])
    metrics = make_compute_metrics(batch_size=4, threshold=0.5)(model, ds)
    assert model.specs == [["kişi", "yer"]]
    assert metrics["exact_f1_micro"] == 1.0
    assert metrics["f1/kişi"] == 1.0
    assert metrics["support/kişi"] == 1


def test_hook_passes_name_to_description_dict_when_records_have_descriptions():
    model = FakeExtractor({"kişi": ["Ahmet"]})
    ds = FakeDataset(
        [
            {
                "id": "r0",
                "input": "Ahmet geldi .",
                "output": {"entities": {"kişi": ["Ahmet"]}, "entity_descriptions": {"kişi": "kişi adı"}},
            }
        ]
    )
    make_compute_metrics(batch_size=4, threshold=0.5)(model, ds)
    assert model.specs == [{"kişi": "kişi adı"}]


def test_hook_writes_error_rows_with_ids(tmp_path):
    model = FakeExtractor({"kişi": ["Mehmet"]})
    ds = FakeDataset(
        [
            {"id": "r0", "input": "Ahmet geldi .", "output": {"entities": {"kişi": ["Ahmet"]}}},
            {"id": "r1", "input": "Mehmet geldi .", "output": {"entities": {"kişi": ["Mehmet"]}}},
        ]
    )
    errors = tmp_path / "errors.jsonl"
    make_compute_metrics(batch_size=4, threshold=0.5, errors_path=errors)(model, ds)
    rows = [json.loads(line) for line in errors.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["r0"]
    assert rows[0]["gold"] == {"kişi": ["Ahmet"]}
    assert rows[0]["predicted"] == {"kişi": ["Mehmet"]}


class FakeExtractorByText:
    """Per-text canned answers; records each (texts, spec) call."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def batch_extract_entities(self, texts, entity_types, batch_size=8, threshold=0.5, **_):
        self.calls.append((list(texts), entity_types))
        return [{"entities": dict(self.answers[t])} for t in texts]


def test_hook_scores_mixed_schema_records_per_group(tmp_path):
    """A mixed eval set (e.g. KVKK + NER rows) is extracted per schema group and scored jointly."""
    ds = FakeDataset(
        [
            {"id": "k0", "input": "iban satiri", "output": {"entities": {"IBAN": ["TR12"], "adres": []}}},
            {"id": "n0", "input": "kisi satiri", "output": {"entities": {"kişi": ["Ahmet Yılmaz"]}}},
            {"id": "k1", "input": "adres satiri", "output": {"entities": {"IBAN": [], "adres": ["Çankaya"]}}},
        ]
    )
    model = FakeExtractorByText(
        {
            "iban satiri": {"IBAN": ["TR12"], "adres": []},
            "kisi satiri": {"kişi": []},  # miss -> fn
            "adres satiri": {"IBAN": ["TR99"], "adres": ["Çankaya"]},  # extra IBAN -> fp
        }
    )
    errors = tmp_path / "errors.jsonl"
    flat = make_compute_metrics(batch_size=4, threshold=0.5, errors_path=errors)(model, ds)

    # one batch_extract call per schema, each with only its own rows, order preserved
    assert model.calls == [
        (["iban satiri", "adres satiri"], ["IBAN", "adres"]),
        (["kisi satiri"], ["kişi"]),
    ]
    # merged counts: IBAN tp1 fp1, adres tp1, kişi fn1 -> micro tp2 fp1 fn1
    assert flat["exact_f1_micro"] == pytest.approx(2 / 3)
    assert flat["precision/IBAN"] == 0.5
    assert flat["f1/adres"] == 1.0
    assert (flat["recall/kişi"], flat["support/kişi"]) == (0.0, 1)
    assert flat["n_examples"] == 3
    assert flat["n_mismatched_examples"] == 2
    assert [json.loads(l)["id"] for l in errors.read_text(encoding="utf-8").splitlines()] == ["n0", "k1"]


def test_hook_groups_by_descriptions_as_well():
    """Same label names with different descriptions are separate schemas."""
    ds = FakeDataset(
        [
            {
                "id": "a",
                "input": "x",
                "output": {"entities": {"kişi": []}, "entity_descriptions": {"kişi": "ad soyad"}},
            },
            {"id": "b", "input": "y", "output": {"entities": {"kişi": []}}},
        ]
    )
    model = FakeExtractorByText({"x": {"kişi": []}, "y": {"kişi": []}})
    make_compute_metrics(batch_size=4, threshold=0.5)(model, ds)
    assert model.calls == [(["x"], {"kişi": "ad soyad"}), (["y"], ["kişi"])]


def test_hook_still_rejects_missing_descriptions_for_declared_labels():
    ds = FakeDataset(
        [
            {
                "id": "a",
                "input": "x",
                "output": {"entities": {"kişi": [], "yer": []}, "entity_descriptions": {"kişi": "ad soyad"}},
            },
        ]
    )
    with pytest.raises(ValueError):
        make_compute_metrics(batch_size=4, threshold=0.5)(FakeExtractorByText({"x": {}}), ds)


def test_overlapping_label_names_pool_counts_across_schemas():
    """The same label name in two schemas is one label: counts pool into one row."""
    ds = FakeDataset(
        [
            {"id": "a", "input": "x", "output": {"entities": {"tarih": ["1 Ocak"], "IBAN": []}}},
            {"id": "b", "input": "y", "output": {"entities": {"tarih": ["2 Şubat"], "kişi": []}}},
        ]
    )
    model = FakeExtractorByText({"x": {"tarih": ["1 Ocak"], "IBAN": []}, "y": {"tarih": [], "kişi": []}})
    flat = make_compute_metrics(batch_size=4, threshold=0.5)(model, ds)
    assert len(model.calls) == 2
    assert (flat["support/tarih"], flat["recall/tarih"], flat["precision/tarih"]) == (2, 0.5, 1.0)


def test_prediction_count_mismatch_raises_instead_of_silent_fn():
    class ShortExtractor:
        def batch_extract_entities(self, texts, entity_types, batch_size=8, threshold=0.5, **_):
            return [{"entities": {}}]  # one short

    ds = FakeDataset(
        [
            {"id": "a", "input": "x", "output": {"entities": {"kişi": []}}},
            {"id": "b", "input": "y", "output": {"entities": {"kişi": []}}},
        ]
    )
    with pytest.raises(ValueError):
        make_compute_metrics(batch_size=4, threshold=0.5)(ShortExtractor(), ds)


def test_error_rows_split_across_schema_groups_when_capped(tmp_path):
    """max_errors must not be eaten entirely by the first schema block of a mixed file."""
    recs, answers = [], {}
    for i in range(3):  # 3 kvkk mismatches first ...
        recs.append({"id": f"k{i}", "input": f"kt{i}", "output": {"entities": {"IBAN": ["TR1"]}}})
        answers[f"kt{i}"] = {"IBAN": []}
    for i in range(2):  # ... then 2 ner mismatches
        recs.append({"id": f"n{i}", "input": f"nt{i}", "output": {"entities": {"kişi": ["Ali Veli"]}}})
        answers[f"nt{i}"] = {"kişi": []}
    errors = tmp_path / "errors.jsonl"
    make_compute_metrics(batch_size=4, threshold=0.5, errors_path=errors, max_errors=4)(
        FakeExtractorByText(answers), FakeDataset(recs)
    )
    ids = [json.loads(l)["id"] for l in errors.read_text(encoding="utf-8").splitlines()]
    assert ids == ["k0", "k1", "n0", "n1"]  # round-robin across groups, file kept in record order
