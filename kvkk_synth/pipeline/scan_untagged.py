#!/usr/bin/env python3
"""Untagged-PII scan (design §6, flagging only): DeepSeek reads each accepted document WITH its tags and lists
any personal data that is not inside a tag. Documents with findings move from records_full.jsonl to
flagged.jsonl (appended, with `flags`), so nothing untagged reaches the training records silently.
Run from the run folder with OPENROUTER_API_KEY set: python code/scan_untagged.py [--workers 8]
"""

import argparse
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor, as_completed

import config  # [pkg]
import prompts  # [pkg] prompt files

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=config.get("judge.workers"))
ap.add_argument("--prefix", default="")
a = ap.parse_args()
MODEL = config.get("judge.model")
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]  # [pkg]
PROMPT = prompts.load("scan_untagged.txt")  # [pkg] see kvkk_synth/prompts/

recs = [json.loads(l) for l in open(f"{a.prefix}records_full.jsonl", encoding="utf-8")]


def scan(r):
    body = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 800,
        "messages": [{"role": "user", "content": PROMPT.format(text=r["text_tagged"])}],
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
        "usage": {"include": True},
    }
    for attempt in range(3):
        try:
            j = requests.post(
                URL,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json=body,
                timeout=180,
            ).json()
            found = json.loads(re.search(r"\{.*\}", j["choices"][0]["message"]["content"], re.S).group(0)).get(
                "untagged", []
            )
            clean = r["text"]
            found = [
                f
                for f in found
                if isinstance(f, dict)
                and f.get("quote")
                and f["quote"] in clean
                and not any(e["start"] <= clean.find(f["quote"]) < e["end"] for e in r["entities"])
            ]
            return r["bundle_id"], found, j.get("usage", {}).get("cost", 0.0)
        except Exception as e:
            print("retry", r["bundle_id"], attempt, str(e)[:80], file=sys.stderr)
    return r["bundle_id"], None, 0.0  # [e08] None = scan failed


t0 = time.time()
results = {}
cost = 0.0
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for bid, found, c in (f.result() for f in as_completed([ex.submit(scan, r) for r in recs])):
        results[bid] = found
        cost += c
keep, moved = [], []
for r in recs:
    f = results.get(r["bundle_id"], [])
    if f is None:
        r["flags"] = ["FLAG untagged scan failed"]
        r["untagged_scan"] = None
        moved.append(r)
        continue  # [e08]
    if f:
        r["flags"] = [f"FLAG untagged personal data: {x['quote'][:40]} ({x.get('kind', '?')})" for x in f]
        r["untagged_scan"] = f
        moved.append(r)
    else:
        r["untagged_scan"] = []
        keep.append(r)
with open(f"{a.prefix}records_full.jsonl", "w", encoding="utf-8") as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{a.prefix}records.jsonl", "w", encoding="utf-8") as fh:
    for r in keep:
        fh.write(
            json.dumps(
                {k: r[k] for k in ("text", "entities", "relations") if k in r}
                | ({"relations_negative": r["relations_negative"]} if r.get("relations_negative") else {}),
                ensure_ascii=False,
            )
            + "\n"
        )
with open(f"{a.prefix}flagged.jsonl", "a", encoding="utf-8") as fh:
    for r in moved:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(
    f"untagged scan: {len(recs)} docs, {len(moved)} flagged ({sum(len(r['untagged_scan']) for r in moved)} findings), {len(keep)} clean | ${cost:.3f}, {round(time.time() - t0)}s"
)
