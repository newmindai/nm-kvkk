#!/usr/bin/env python3
"""Repair pass (design §6, one round): every flagged document and every fixable reject goes back to the writer
(GPT-5.6 Luna on flex) with its own tagged draft, the parser's problem list and the allowed values/kinds; it
rewrites minimally — tag the untagged, remove or replace what is not allowed, fix broken tags — and the result is
re-parsed with the same guards. Fixable = anything except truncation / reasoning leak / zero-PII violations.
Run from the run folder with OPENROUTER_API_KEY set: python code/repair.py [--workers 12]
Writes raw-repair.parquet (same columns as raw.parquet) and repair-todo.jsonl (the inputs), then parse with --prefix repair-.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor, as_completed

import config  # [pkg]
import prompts  # [pkg] prompt files

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=config.get("writer.workers"))
ap.add_argument("--model", default=config.get("writer.model"))
a = ap.parse_args()  # [pkg] config.yaml
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import tiers  # noqa: E402

bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open("bundles.jsonl", encoding="utf-8"))}
UNFIXABLE = ("truncated", "reasoning", "zero-PII")
todo = []
for f, kind in (("flagged.jsonl", "flagged"), ("rejects.jsonl", "rejected")):
    if not os.path.exists(f):
        continue
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        probs = r.get("flags") or r.get("problems") or []
        if any(any(u in p for u in UNFIXABLE) for p in probs) or r.get("profile") == "zero":
            continue
        todo.append(
            {
                "bundle_id": r["bundle_id"],
                "kind": kind,
                "problems": probs,
                "text_tagged": r["text_tagged"],
                "genre": r["genre"],
                "document_format": r["document_format"],
                "profile": r.get("profile"),
            }
        )
with open("repair-todo.jsonl", "w", encoding="utf-8") as fh:
    for t in todo:
        fh.write(json.dumps(t, ensure_ascii=False) + "\n")
print(
    f"repair candidates: {len(todo)} ({sum(1 for t in todo if t['kind'] == 'flagged')} flagged, {sum(1 for t in todo if t['kind'] == 'rejected')} rejected)",
    flush=True,
)

PROMPT = prompts.load("repair.txt")  # [pkg] see kvkk_synth/prompts/


def fix(t):
    b = bundles[t["bundle_id"]]
    ents = "\n".join(f"  - [{e['value']}]{e['label']}" for e in b["entities"])
    persons = "; ".join(f"{p['name']} ({p['role_en']})" for p in b.get("persons", [])) or "the subject only"
    body = {
        "model": a.model,
        "temperature": 0.3,
        "max_tokens": 12288,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(
                    problems="\n".join(f"- {p.replace('FLAG ', '')}" for p in t["problems"]),
                    entities=ents,
                    persons=persons,
                    kinds=", ".join(sorted(tiers.TIER_B)),
                    text=t["text_tagged"],
                ),
            }
        ],
        "provider": {"only": config.get("writer.provider"), "allow_fallbacks": False},
        "reasoning": {"effort": config.get("writer.repair_reasoning_effort")},
        "usage": {"include": True},
    }
    for attempt in range(3):
        try:
            j = requests.post(
                URL,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json=body,
                timeout=600,
            ).json()
            txt = j["choices"][0]["message"]["content"]
            u = j.get("usage", {})
            return {
                "bundle_id": t["bundle_id"],
                "scenario": "pool",
                "genre": t["genre"],
                "document_format": t["document_format"],
                "profile": t["profile"],
                "text_tagged": txt,
                "cost": u.get("cost", 0.0),
                "repair_of": t["kind"],
            }
        except Exception as e:
            print("retry", t["bundle_id"], attempt, str(e)[:80], file=sys.stderr)
            time.sleep(3)
    return None


t0 = time.time()
out = []
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for res in (f.result() for f in as_completed([ex.submit(fix, t) for t in todo])):
        if res:
            out.append(res)
df = pd.DataFrame(sorted(out, key=lambda r: r["bundle_id"]))
df.to_parquet("raw-repair.parquet", index=False)
print(
    f"repair: {len(out)} rewritten, ${df.cost.sum() if len(df) else 0:.3f}, {round(time.time() - t0)}s -> raw-repair.parquet",
    flush=True,
)
