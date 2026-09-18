#!/usr/bin/env python3
"""G3 oracle: the newest runs/*-smoke folder holds a dataset with >= 3 accepted documents, every span offset-exact,
and a viewer. Prints SMOKE OK only when all of that holds."""

import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
runs = sorted(glob.glob(os.path.join(ROOT, "runs", "*-smoke")))
if not runs:
    print("SMOKE FAILED: no runs/*-smoke")
    sys.exit(1)
run = runs[-1]
st_path = os.path.join(run, "dataset", "stats.json")
if not os.path.exists(st_path):
    print(f"SMOKE FAILED: {st_path} missing")
    sys.exit(1)
st = json.load(open(st_path, encoding="utf-8"))
recs = [json.loads(l) for l in open(os.path.join(run, "dataset", "records.jsonl"), encoding="utf-8")]
bad = [
    (r.get("bundle_id"), e)
    for r in recs
    for e in r["entities"]
    if r["text"][e["start"] : e["end"]] != e.get("span", e.get("text"))
]
print(
    f"{run}: documents {st['documents']}, buckets {st['buckets']}, spans {sum(len(r['entities']) for r in recs)}, offset mismatches {len(bad)}"
)
ok = (
    st["documents"] >= 3
    and not bad
    and os.path.exists(os.path.join(run, "viewer.html"))
    and "CHAIN DONE" in open(os.path.join(run, "run.log"), encoding="utf-8").read()
)
print("SMOKE OK" if ok else "SMOKE FAILED")
sys.exit(0 if ok else 1)
