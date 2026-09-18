import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAVE_GEO = all(
    os.path.exists(os.path.join(ROOT, "geo", f"{t}.parquet")) for t in ("cities", "towns", "neighbourhoods", "streets")
)


@pytest.mark.skipif(not HAVE_GEO, reason="geo tables not built (download_seeds.py --geo)")
def test_dry_run_renders_prompts_without_api(tmp_path):
    env = {**os.environ, "OPENROUTER_API_KEY": "", "KVKK_PY": sys.executable, "KVKK_RUNS": str(tmp_path)}
    r = subprocess.run(
        ["zsh", os.path.join(ROOT, "new_run.sh"), "dry", "--n", "5"], env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr[-2000:]
    run = next(p for p in tmp_path.iterdir() if p.name.endswith("-dry"))
    assert (run / "briefs.jsonl").exists() and (run / "config.yaml").exists()
    r = subprocess.run(
        ["zsh", os.path.join(ROOT, "run.sh"), "--dry", "--limit", "5"], cwd=run, env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert (
        (run / "scores-e5.json").exists()
        and (run / "bundles.jsonl").exists()
        and (run / "soundness_report.json").exists()
    )
    rendered = [json.loads(l) for l in open(run / "prompts-rendered.jsonl", encoding="utf-8")]
    assert len(rendered) == 5 and all("===SON===" in x["user"] for x in rendered)
    assert not (run / "raw.parquet").exists()
    assert "DRY RUN DONE" in open(run / "run.log", encoding="utf-8").read()
