import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def source(script):
    return open(os.path.join(ROOT, "pipeline", script), encoding="utf-8").read()


def test_sampler_defaults_are_run_folder_relative():
    for f in ("sampler8.py", "sampler7.py"):
        s = source(f)
        assert (
            'default="briefs.jsonl"' in s
            and 'default="scores-e5.json"' in s
            and 'default="scores-roles.json"' in s
            and 'default="bundles.jsonl"' in s
        ), f
        assert "os.path.join(RUN" not in s and "RUN = " not in s, f


def test_sampler8_help_runs_with_the_package_geo():
    if not os.path.exists(os.path.join(ROOT, "geo", "cities.parquet")):
        pytest.skip("geo tables not built: python download_seeds.py --geo")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "pipeline", "sampler8.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0 and "--briefs" in r.stdout, r.stderr[-1500:]


def test_build_dataset_parquet_name_from_run_folder():
    s = source("build_dataset.py")
    assert "tr_pii_relations_e08.parquet" not in s and "RUN_ID" in s


def test_embed_score_nodes_default_is_the_package_taxonomy():
    assert '"taxonomy", "nodes.jsonl"' in source("embed_score.py")
