#!/usr/bin/env python3
"""Relation judge for e07: for every accepted document, DeepSeek decides per kept relation whether the document
states or unambiguously implies it (evidence quoted). Unexpressed relations are removed from `relations` and
kept in `relations_unexpressed`. Bundle relations use the sampler's fact sentence; writer-added and secondary-
person relations get a constructed one. Run from the run folder: python code/verify_relations7.py [--workers 8]
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
ap.add_argument("--records", default="records_full.jsonl")
a = ap.parse_args()
MODEL = config.get("judge.model")
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]  # [pkg]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taxonomy"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # [pkg]
import kinship_claims  # noqa: E402
import kvkk_sampler as ks

_, rels_meta = ks.load_taxonomy()
rel_tr = {r["id"]: r["labels_tr"][0].replace("_", " ") for r in rels_meta}
bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open("bundles.jsonl", encoding="utf-8"))}
PROMPT = prompts.load("judge_relations.txt")  # [pkg] see kvkk_synth/prompts/


def facts_for(r):
    b = bundles[r["bundle_id"]]
    byid = {e["id"]: e for e in r["entities"]}
    known = {(x["head"], x["relation"], x["tail"]): x.get("fact_tr") for x in b.get("relations", [])}
    out = []
    persons = {p["id"]: p for p in b.get("persons", [])}
    for x in r["relations"]:
        k = (x["head"], x["relation"], x["tail"])
        f = known.get(k)
        if x["relation"] in kinship_claims.SENT:  # [e08] neutral sexed claim, not the scenario template
            p = (
                persons.get(x["head"])
                if persons.get(x["head"], {}).get("kind") not in (None, "subject")
                else persons.get(x["tail"])
            )
            h, t = byid.get(x["head"]), byid.get(x["tail"])
            if p and h and t:
                f = kinship_claims.sentence(x["relation"], h["span"], t["span"], p.get("sex"))
        if not f:
            h, t = byid.get(x["head"]), byid.get(x["tail"])
            f = f"{h['span'] if h else x['head']} — {rel_tr.get(x['relation'], x['relation'])} — {t['span'] if t else x['tail']}"
        out.append(f)
    return out


def judge(r):
    facts = facts_for(r)
    if not facts:
        return r["bundle_id"], {}, 0.0
    body = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(text=r["text"], facts="\n".join(f"{i}: {f}" for i, f in enumerate(facts))),
            }
        ],
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
            v = {
                c["index"]: c
                for c in json.loads(re.search(r"\{.*\}", j["choices"][0]["message"]["content"], re.S).group(0))[
                    "checks"
                ]
            }
            return r["bundle_id"], v, j.get("usage", {}).get("cost", 0.0)
        except Exception as e:
            print("retry", r["bundle_id"], attempt, str(e)[:80], file=sys.stderr)
    return r["bundle_id"], None, 0.0  # [e08] None = judge failed


recs = [json.loads(l) for l in open(a.records, encoding="utf-8")]
t0 = time.time()
verdicts = {}
cost = 0.0
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for bid, v, c in (f.result() for f in as_completed([ex.submit(judge, r) for r in recs])):
        verdicts[bid] = v
        cost += c
kept = dropped = 0
for r in recs:
    v = verdicts.get(r["bundle_id"])
    keep, drop = [], []
    if v is None and r["relations"]:
        r["judge_failed"] = "relations"
        r["flags"] = list(r.get("flags") or []) + ["FLAG relation judge failed"]  # [e08] never silently keep unverified
        v = {i: {"expressed": True, "evidence": "JUDGE_FAILED"} for i in range(len(r["relations"]))}
    for i, x in enumerate(r["relations"]):
        c = (v or {}).get(i)
        (keep if (c is not None and c.get("expressed")) else drop).append(
            {**x, "evidence": (c or {}).get("evidence", "")}
        )
    r["relations"] = [{k: x[k] for k in ("head", "relation", "tail")} for x in keep]
    r["relations_evidence"] = keep
    r["relations_unexpressed"] = drop
    kept += len(keep)
    dropped += len(drop)
with open(a.records, "w", encoding="utf-8") as fh:
    for r in recs:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("records.jsonl", "w", encoding="utf-8") as fh:
    for r in recs:
        fh.write(
            json.dumps(
                {
                    "text": r["text"],
                    "entities": r["entities"],
                    "relations": r["relations"],
                    **({"relations_negative": r["relations_negative"]} if r.get("relations_negative") else {}),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
print(
    f"relation judge: {kept} kept, {dropped} dropped as unexpressed ({dropped / max(kept + dropped, 1):.1%}) | ${cost:.3f}, {round(time.time() - t0)}s"
)
