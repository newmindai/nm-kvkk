#!/usr/bin/env python3
"""G5 oracle: nothing that must stay local is addable. Prints GIT HYGIENE OK when clean."""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
out = subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, capture_output=True, text=True
).stdout
paths = [l[3:] for l in out.splitlines()]
bad = [
    p
    for p in paths
    if p.startswith("experiments/")
    or (p.startswith("kvkk_synth/runs/") and p != "kvkk_synth/runs/README.md")
    or p.startswith("kvkk_synth/geo/raw/")
    or p.startswith("kvkk_synth/briefs/raw/")
    or (p.startswith("kvkk_synth/geo/") and p.endswith(".parquet"))
    or "__pycache__" in p
    or p.endswith(".env")
]
print(f"{len(paths)} addable paths, {len(bad)} that must stay local")
for p in bad[:20]:
    print("  BAD", p)
print("GIT HYGIENE OK" if not bad else "GIT HYGIENE FAILED")
sys.exit(0 if not bad else 1)
