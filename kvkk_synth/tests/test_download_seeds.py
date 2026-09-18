import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import download_seeds as ds

RAW = os.path.join(ROOT, "briefs", "raw", "data")


def test_catalogue_rule_on_a_toy_frame():
    df = pd.DataFrame(
        [
            {
                "uid": "u1",
                "domain": "A",
                "document_type": "Invoice",
                "document_description": "d1",
                "document_format": "structured",
            },
            {
                "uid": "u2",
                "domain": "A",
                "document_type": "invoice ",
                "document_description": "d2",
                "document_format": "unstructured",
            },
            {
                "uid": "u3",
                "domain": "A",
                "document_type": "Invoice",
                "document_description": "d2",
                "document_format": "unstructured",
            },
            {
                "uid": "u4",
                "domain": "B",
                "document_type": "Memo",
                "document_description": "m",
                "document_format": "structured",
            },
        ]
    )
    c = ds.build_catalogue(df)
    assert list(c.columns) == [
        "domain",
        "type_norm",
        "uid",
        "document_type",
        "document_description",
        "document_format",
        "n",
    ]
    inv = c[c.type_norm == "invoice"].iloc[0]
    assert (
        inv.uid == "u1"
        and inv.document_type == "Invoice"
        and inv.document_description == "d2"
        and inv.document_format == "unstructured"
        and inv.n == 3
    )
    assert len(c) == 2


def test_fresh_rule_on_a_toy_frame():
    tr = pd.DataFrame(
        [
            {
                "domain": "A",
                "type_norm": "x",
                "uid": "1",
                "document_type": "X",
                "document_description": "",
                "document_format": "s",
                "n": 1,
            },
            {
                "domain": "A",
                "type_norm": "y",
                "uid": "2",
                "document_type": "Y",
                "document_description": "",
                "document_format": "s",
                "n": 1,
            },
        ]
    )
    te = pd.DataFrame(
        [
            {
                "domain": "B",
                "type_norm": "x",
                "uid": "3",
                "document_type": "X",
                "document_description": "",
                "document_format": "s",
                "n": 1,
            },
            {
                "domain": "B",
                "type_norm": "z",
                "uid": "4",
                "document_type": "Z",
                "document_description": "",
                "document_format": "s",
                "n": 1,
            },
            {
                "domain": "B",
                "type_norm": "w",
                "uid": "5",
                "document_type": "W",
                "document_description": "",
                "document_format": "s",
                "n": 1,
            },
        ]
    )
    f = ds.build_fresh(tr, te, {"y", "w"})
    assert sorted(f.uid) == ["1", "4"]  # y used, x present in train, w used


@pytest.mark.skipif(not os.path.exists(RAW), reason="raw Nemotron files not downloaded")
def test_rebuilt_catalogues_equal_shipped():
    ds.check_catalogues(ROOT)  # raises AssertionError on any difference


def test_geo_counts_expected():
    assert ds.GEO_EXPECTED == {"cities": 81, "towns": 975, "neighbourhoods": 74659, "streets": 1097149}
