import collections
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_sample_is_balanced_capped_and_shaped(tmp_path):
    out = tmp_path / "briefs.jsonl"
    cat = tmp_path / "briefs_catalogue.jsonl"
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "pipeline", "sample_briefs.py"),
            "--catalogue",
            os.path.join(ROOT, "briefs", "nemotron_catalogue.parquet"),
            "--n",
            "300",
            "--seed",
            "1",
            "--max-uses",
            "3",
            "--out",
            str(out),
            "--catalogue-out",
            str(cat),
        ],
        check=True,
    )
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 300 and [r["bundle_id"] for r in rows] == list(range(1, 301))
    assert set(rows[0]) == {
        "bundle_id",
        "uid",
        "domain",
        "document_type",
        "document_format",
        "description",
        "source",
        "text",
    }
    uses = collections.Counter(r["uid"] for r in rows)
    assert max(uses.values()) <= 3
    per_domain = collections.Counter(r["domain"] for r in rows)
    assert max(per_domain.values()) - min(per_domain.values()) <= 2
    assert rows[0]["text"].startswith(rows[0]["document_type"] + " (" + rows[0]["domain"])
    catrows = [json.loads(l) for l in open(cat, encoding="utf-8")]
    assert {r["uid"] for r in catrows} == set(uses) and "bundle_id" not in catrows[0]


def test_reuse_kicks_in_when_n_exceeds_distinct_briefs(tmp_path):
    out = tmp_path / "b.jsonl"
    cat = tmp_path / "c.jsonl"
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "pipeline", "sample_briefs.py"),
            "--catalogue",
            os.path.join(ROOT, "briefs", "nemotron_catalogue_fresh_for_e08.parquet"),
            "--n",
            "5000",
            "--seed",
            "8",
            "--out",
            str(out),
            "--catalogue-out",
            str(cat),
        ],
        check=True,
    )
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    uses = collections.Counter(r["uid"] for r in rows)
    assert len(rows) == 5000 and len(uses) == 1789 and max(uses.values()) == 3
