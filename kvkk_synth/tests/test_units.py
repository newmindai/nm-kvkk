import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPE = os.path.join(ROOT, "pipeline")
EX = os.path.join(ROOT, "examples", "e08-dry")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_harmony_fixes_suffix_without_moving_offsets():
    h = load("harmony", os.path.join(PIPE, "v2", "harmony.py"))
    text = "Yılmaz'in dosyası ve Zeynep'de kaldı."
    spans = [{"start": 0, "end": 6, "span": "Yılmaz"}, {"start": 21, "end": 27, "span": "Zeynep"}]
    assert text[21:27] == "Zeynep"
    fixed, n = h.fix_text(text, spans)
    assert len(fixed) == len(text) and fixed.startswith("Yılmaz'ın") and "Zeynep'te" in fixed and n == 2


def test_kinship_claims_are_sexed_and_neutral():
    k = load("kinship_claims", os.path.join(PIPE, "kinship_claims.py"))
    assert (
        k.sentence("child_of", "A", "B", "M") == "A, B'in oğludur."
        and k.sentence("sibling_of", "A", "B", "F") == "A, B'in kız kardeşidir."
    )
    assert "okul" not in k.sentence("child_of", "A", "B", "F")
    assert k.sentence("phone_of", "A", "B", "F") is None


def test_tiers_validator_rejects_placeholders_and_digits():
    t = load("tiers", os.path.join(PIPE, "tiers.py"))
    name_re = re.compile(r"Ahmet Yılmaz")
    assert t.validate("job_title", "Kıdemli Yazılım Mühendisi", set(), name_re)[0]
    assert not t.validate("job_title", "XYZ", set(), name_re)[0]
    assert not t.validate("job_title", "12345", set(), name_re)[0]
    assert not t.validate("job_title", "Ahmet Yılmaz", set(), name_re)[0]
    assert not t.validate("iban", "TR12", set(), name_re)[0]  # not a tier-B label


def test_parser_reproduces_the_e08_dry_run(tmp_path):
    for f in ("bundles.jsonl", "raw-dry.parquet", "dry2-records.jsonl", "dry2-rejects.jsonl"):
        shutil.copy(os.path.join(EX, f), tmp_path / f)
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(PIPE, "v2", "parse_facts.py"),
            "--raw",
            "raw-dry.parquet",
            "--bundles",
            "bundles.jsonl",
            "--facts-optional",
            "--prefix",
            "check-",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    got = {json.loads(l)["text"]: json.loads(l) for l in open(tmp_path / "check-records.jsonl", encoding="utf-8")}
    exp = {json.loads(l)["text"]: json.loads(l) for l in open(tmp_path / "dry2-records.jsonl", encoding="utf-8")}
    assert set(got) == set(exp), (len(got), len(exp))
    for text, e in exp.items():
        assert [(x["start"], x["end"], x["label"]) for x in got[text]["entities"]] == [
            (x["start"], x["end"], x["label"]) for x in e["entities"]
        ]
    rej = [json.loads(l) for l in open(tmp_path / "check-rejects.jsonl", encoding="utf-8")]
    assert len(rej) == sum(1 for _ in open(tmp_path / "dry2-rejects.jsonl", encoding="utf-8"))


def test_every_example_span_is_offset_exact():
    for l in open(os.path.join(ROOT, "examples", "e08", "records-sample.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        for e in r["entities"]:
            assert r["text"][e["start"] : e["end"]] == e.get("span", e.get("text"))
