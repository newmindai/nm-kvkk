"""Gold-vs-prediction comparison for the vekaletname GT demo (scripts/gt_compare.py)."""

from gt_compare import (
    aggregate,
    compare_entities,
    compare_relations,
    entity_mentions,
    prf,
    score_entities,
    score_relations,
)


def g(start, end, label, eid="e1"):
    return {"start": start, "end": end, "label": label, "text": "t", "entity_id": eid}


def p(start, end, label, conf=0.9):
    return {"start": start, "end": end, "label": label, "text": "t", "confidence": conf}


# ------------------------------------------------------------------ entities


def test_exact_match_is_match():
    items = compare_entities([g(0, 5, "full_name")], [p(0, 5, "full_name")])
    assert [i["status"] for i in items] == ["match"]
    assert items[0]["gold"]["entity_id"] == "e1" and items[0]["pred"]["confidence"] == 0.9


def test_same_span_other_label_is_label_mismatch():
    items = compare_entities([g(0, 5, "full_name")], [p(0, 5, "first_name")])
    assert [i["status"] for i in items] == ["label_mismatch"]


def test_overlapping_span_is_boundary_mismatch_and_records_whether_label_agrees():
    items = compare_entities(
        [g(0, 10, "full_address")],
        [
            p(0, 6, "full_address"),
        ],
    )
    assert items[0]["status"] == "boundary_mismatch" and items[0]["same_label"] is True
    items = compare_entities([g(0, 10, "full_address")], [p(3, 12, "city")])
    assert items[0]["status"] == "boundary_mismatch" and items[0]["same_label"] is False


def test_unmatched_sides_become_fp_and_fn_sorted_by_position():
    items = compare_entities([g(20, 25, "iban")], [p(0, 3, "city")])
    assert [(i["status"], i["start"]) for i in items] == [("fp", 0), ("fn", 20)]


def test_exact_match_wins_over_boundary_when_both_are_possible():
    # pred A exactly matches gold 1; pred B overlaps gold 1 too and must become fp, not steal it
    items = compare_entities([g(0, 5, "x")], [p(0, 5, "x"), p(3, 8, "x")])
    assert sorted(i["status"] for i in items) == ["fp", "match"]


def test_scores_strict_and_lenient():
    items = compare_entities(
        [g(0, 5, "a"), g(10, 20, "b"), g(30, 35, "c")], [p(0, 5, "a"), p(10, 16, "b"), p(30, 35, "z"), p(50, 55, "a")]
    )
    s = score_entities(items)
    # strict: 1 tp (a); pred count 4; gold count 3
    assert (s["strict"]["tp"], s["strict"]["fp"], s["strict"]["fn"]) == (1, 3, 2)
    # lenient: a + boundary-with-same-label b = 2 tp
    assert (s["lenient"]["tp"], s["lenient"]["fp"], s["lenient"]["fn"]) == (2, 2, 1)
    assert s["counts"] == {"match": 1, "label_mismatch": 1, "boundary_mismatch": 1, "fp": 1, "fn": 0}


def test_prf_handles_zero():
    assert prf(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert prf(2, 2, 0)["precision"] == 0.5 and prf(2, 0, 2)["recall"] == 0.5


# ------------------------------------------------------------------ relations

REC = {
    "text": "x",
    "entities": [
        {
            "id": "e1",
            "label": "full_name",
            "span": "Ali Kaya",
            "role": "vekil",
            "same_as": None,
            "mentions": [{"start": 0, "end": 8}, {"start": 50, "end": 58}],
        },
        {
            "id": "e2",
            "label": "full_name",
            "span": "A. Kaya",
            "role": None,
            "same_as": "e1",
            "mentions": [{"start": 100, "end": 107}],
        },
        {
            "id": "e3",
            "label": "national_id_number",
            "span": "123",
            "role": None,
            "same_as": None,
            "mentions": [{"start": 20, "end": 23}],
        },
        {
            "id": "e4",
            "label": "full_address",
            "span": "Adres",
            "role": None,
            "same_as": None,
            "mentions": [{"start": 30, "end": 35}],
        },
    ],
    "relations": [
        {"head": "e3", "relation": "national_id_of", "tail": "e1"},
        {"head": "e4", "relation": "residence_of", "tail": "e1"},
    ],
}


def pr(relation, hs, he, ts, te, conf=0.8):
    return {
        "relation": relation,
        "head": {"text": "h", "start": hs, "end": he, "confidence": conf},
        "tail": {"text": "t", "start": ts, "end": te, "confidence": conf},
    }


def test_entity_mentions_merges_same_as_groups():
    m = entity_mentions(REC)
    assert {(a, b) for a, b in m["e1"]} == {(0, 8), (50, 58), (100, 107)}
    assert m["e2"] == m["e1"] and m["e3"] == [(20, 23)]


def test_relation_match_via_any_coreferent_mention():
    items = compare_relations(REC, [pr("national_id_of", 20, 23, 100, 107)])  # tail = the alias mention
    assert [i["status"] for i in items] == ["match", "fn"]
    assert items[0]["gold"]["head"] == "e3" and items[1]["relation"] == "residence_of"


def test_reversed_direction_is_lenient_not_strict():
    items = compare_relations(REC, [pr("national_id_of", 0, 8, 20, 23)])
    assert [i["status"] for i in items] == ["reversed", "fn"]
    s = score_relations(items)
    assert s["strict"]["tp"] == 0 and s["lenient"]["tp"] == 1


def test_wrong_type_or_unknown_argument_is_fp():
    items = compare_relations(REC, [pr("phone_of", 20, 23, 0, 8), pr("national_id_of", 20, 23, 200, 205)])
    assert sorted(i["status"] for i in items) == ["fn", "fn", "fp", "fp"]


def test_each_gold_relation_is_credited_once():
    items = compare_relations(REC, [pr("national_id_of", 20, 23, 0, 8), pr("national_id_of", 20, 23, 50, 58)])
    assert sorted(i["status"] for i in items) == ["fn", "fp", "match"]


def test_argument_matching_tolerates_partial_overlap():
    items = compare_relations(REC, [pr("residence_of", 30, 33, 0, 8)])  # head covers part of the address
    assert items[0]["status"] == "match"


# ------------------------------------------------------------------ aggregate


def test_aggregate_micro_and_per_label():
    d1 = {
        "entity_items": compare_entities([g(0, 5, "a"), g(10, 15, "b")], [p(0, 5, "a"), p(10, 15, "c")]),
        "relation_items": compare_relations(REC, [pr("national_id_of", 20, 23, 0, 8)]),
    }
    d2 = {
        "entity_items": compare_entities([g(0, 5, "a")], [p(0, 5, "a"), p(7, 9, "a")]),
        "relation_items": compare_relations(REC, []),
    }
    agg = aggregate([d1, d2])
    assert (
        agg["entities"]["strict"]["tp"] == 2
        and agg["entities"]["strict"]["fp"] == 2
        and agg["entities"]["strict"]["fn"] == 1
    )
    assert agg["per_label"]["a"]["tp"] == 2 and agg["per_label"]["a"]["fp"] == 1 and agg["per_label"]["a"]["gold"] == 2
    assert agg["per_label"]["b"]["fn"] == 1 and agg["per_label"]["c"]["fp"] == 1 and agg["per_label"]["c"]["gold"] == 0
    assert agg["relations"]["strict"]["tp"] == 1 and agg["relations"]["strict"]["fn"] == 3
    assert agg["per_relation"]["national_id_of"]["gold"] == 2 and agg["per_relation"]["residence_of"]["fn"] == 2


# ------------------------------------------------------------------ constraints

from gt_compare import apply_type_constraints, load_constraints  # noqa: E402


def test_load_constraints_maps_turkish_names_to_ids(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(
        '{"telefon numarası sahibi": [["telefon numarası", "ad soyad"]], "bilinmeyen": [["x", "y"]]}', encoding="utf-8"
    )
    allowed = load_constraints(
        path,
        tr_to_id={"telefon numarası": "phone_number", "ad soyad": "full_name"},
        tr_to_rel={"telefon numarası sahibi": "phone_of"},
    )
    assert allowed == {"phone_of": {("phone_number", "full_name")}}


def test_apply_type_constraints_keeps_allowed_drops_wrong_types_and_unlabelled_sides():
    ents = [p(0, 5, "phone_number"), p(10, 18, "full_name"), p(30, 40, "full_address")]
    rels = [
        pr("phone_of", 0, 5, 10, 18),  # allowed
        pr("phone_of", 30, 40, 10, 18),  # head is an address -> wrong type
        pr("phone_of", 50, 55, 10, 18),  # head overlaps no predicted entity -> invalid side
        pr("email_of", 0, 5, 10, 18),
    ]  # no constraint for this type -> kept
    allowed = {"phone_of": {("phone_number", "full_name")}}
    kept, dropped = apply_type_constraints(rels, ents, allowed)
    assert [r["head"]["start"] for r in kept] == [0, 0] and [r["relation"] for r in kept] == ["phone_of", "email_of"]
    assert len(dropped) == 2 and all(d["violation"] for d in dropped)
