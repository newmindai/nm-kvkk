import importlib.util
import json
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
G = os.path.abspath(os.path.join(ROOT, ".."))  # the repository root


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vendored_tokenizer_and_record_match_the_original():
    b = load("build_6k", os.path.join(ROOT, "dataset", "build_6k.py"))
    assert b.tokenize("Ali ali@x.com Ankara'da 05xx-123") == ["ali", "ali@x.com", "ankara", "'", "da", "05xx-123"]
    if not os.path.exists(G):
        pytest.skip("sibling repo not present")
    sys.path.insert(0, os.path.join(G, "GLiNER2"))
    sys.path.insert(0, os.path.join(G, "scripts"))
    import convert_relations_train as orig

    recs = [
        json.loads(l) for l in open(os.path.join(ROOT, "examples", "e08", "records-sample.jsonl"), encoding="utf-8")
    ]
    labels = sorted({e["label"] for r in recs for e in r["entities"]})
    rels = sorted({x["relation"] for r in recs for x in r["relations"]}) + ["zzz_absent_a", "zzz_absent_b"]
    lm = {l: l.replace("_", " ") for l in labels}
    for i, r in enumerate(recs):
        for x in r["entities"]:
            x.setdefault("span", r["text"][x["start"] : x["end"]])
        a, sa = b.build_record(
            f"r{i}", r["text"], r["entities"], r["relations"], lm, [lm[l] for l in labels], rels, random.Random(1), []
        )
        o, so = orig.build_record(
            f"r{i}", r["text"], r["entities"], r["relations"], lm, [lm[l] for l in labels], rels, random.Random(1), []
        )
        assert a == o and sa == so
        assert b.tokenize(r["text"]) == orig.tokenize(r["text"])
