#!/usr/bin/env python3
"""G4 oracle: re-judge 10 e08 documents with the ported coherence judge and compare with the Data Designer verdicts
of the run (judge-e08.json). Spends about $0.01; needs OPENROUTER_API_KEY and the e08 run folder."""

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E08 = os.path.abspath(
    os.path.join(ROOT, "..", "experiments", "kvkk_relations_pilot", "runs", "2026-09-09-e08-luna-5000")
)
OUT = os.path.join(ROOT, "examples", "judge-port-check")
if "OPENROUTER_API_KEY" not in os.environ:  # the oracle runs outside run.sh: read the repo-root .env quietly
    envf = os.path.join(ROOT, "..", ".env")
    if os.path.exists(envf):
        for line in open(envf, encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
if not os.path.exists(E08):
    print("JUDGE PORT FAILED: e08 run folder not present")
    sys.exit(1)
os.makedirs(OUT, exist_ok=True)
truth = {int(k): v for k, v in json.load(open(os.path.join(E08, "judge-e08.json"), encoding="utf-8")).items()}
recs = [json.loads(l) for l in open(os.path.join(E08, "dataset", "records_full.jsonl"), encoding="utf-8")]
sample = [r for r in recs if r["bundle_id"] in truth][:10]
with open(os.path.join(OUT, "records.jsonl"), "w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
subprocess.run(
    [
        sys.executable,
        os.path.join(ROOT, "pipeline", "judge_coherence.py"),
        "--records",
        "records.jsonl",
        "--tag",
        "port",
        "--workers",
        "4",
    ],
    cwd=OUT,
    check=True,
)
got = {int(k): v for k, v in json.load(open(os.path.join(OUT, "judge-port.json"), encoding="utf-8")).items()}
within = sum(1 for b in got if abs(got[b]["coherence"] - truth[b]["coherence"]) <= 1)
json.dump(
    {b: {"port": got[b]["coherence"], "e08": truth[b]["coherence"]} for b in got},
    open(os.path.join(OUT, "comparison.json"), "w"),
    indent=1,
)
print(f"judge port: {len(got)} scored, within 1 point of e08 on {within}")
ok = len(got) == 10 and within >= 8
print("JUDGE PORT OK" if ok else "JUDGE PORT FAILED")
sys.exit(0 if ok else 1)
