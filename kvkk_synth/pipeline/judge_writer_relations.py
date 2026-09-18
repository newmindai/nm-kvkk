#!/usr/bin/env python3
"""Confirm the pending relations of writer-added facts (design §4.6): for each accepted document, DeepSeek
(reasoning off) says whether the document attributes each writer-added value to the subject. Confirmed
relations are appended to the record's `relations`; every verdict is kept in `relations_writer`.
Run from the run folder with OPENROUTER_API_KEY set: python code/judge_writer_relations.py
"""

import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # [pkg]
import prompts  # [pkg] prompt files

MODEL = config.get("judge.model")  # [pkg]
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taxonomy"))  # [pkg]
import kvkk_sampler as ks  # noqa: E402

_, rels_meta = ks.load_taxonomy()
rel_tr = {r["id"]: r["labels_tr"][0].replace("_", " ") for r in rels_meta}

PROMPT = prompts.load("judge_writer_relations.txt")  # [pkg] see kvkk_synth/prompts/

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--prefix", default="")
PFX = _ap.parse_args().prefix
recs = [json.loads(l) for l in open(f"{PFX}records_full.jsonl", encoding="utf-8")]
n_conf = n_pend = 0
cost = 0.0
t0 = time.time()
for r in recs:
    pend = r.get("relations_writer", [])
    if not pend:
        continue
    byid = {e["id"]: e for e in r["entities"]}
    subj = next(
        e["span"]
        for e in r["entities"]
        if e["id"] == [x for x in (pend[0]["head"], pend[0]["tail"]) if not x.startswith("w")][0]
    )
    claims = []
    for i, x in enumerate(pend):
        wid = x["head"] if x["head"].startswith("w") else x["tail"]
        e = byid[wid]
        claims.append(
            f"{i}: «{e['span']}» ({e['label'].replace('_', ' ')}) — relation «{rel_tr.get(x['relation'], x['relation'])}» to {subj}"
        )
    body = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 1500,
        "messages": [
            {"role": "user", "content": PROMPT.format(text=r["text"], subject=subj, claims="\n".join(claims))}
        ],
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
        "usage": {"include": True},
    }
    verdict = {}
    for attempt in range(3):
        try:
            j = requests.post(
                URL,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json=body,
                timeout=180,
            ).json()
            txt = j["choices"][0]["message"]["content"]
            verdict = {c["index"]: c for c in json.loads(re.search(r"\{.*\}", txt, re.S).group(0))["checks"]}
            cost += j.get("usage", {}).get("cost", 0.0)
            break
        except Exception as e:
            print("retry", r["bundle_id"], attempt, str(e)[:80], file=sys.stderr)
    for i, x in enumerate(pend):
        c = verdict.get(i, {})
        x["status"] = "confirmed" if c.get("attributed") else "unconfirmed"
        x["evidence"] = c.get("evidence", "")
        n_pend += 1
        if x["status"] == "confirmed":
            n_conf += 1
            r["relations"].append({"head": x["head"], "relation": x["relation"], "tail": x["tail"]})
with open(f"{PFX}records_full.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{PFX}records.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(
            json.dumps({"text": r["text"], "entities": r["entities"], "relations": r["relations"]}, ensure_ascii=False)
            + "\n"
        )
print(
    f"writer relations: {n_conf} confirmed / {n_pend} pending, ${cost:.4f}, {round(time.time() - t0)}s -> records_full.jsonl, records.jsonl"
)
