#!/usr/bin/env python3
"""Attribution check for the persons axis: for every used NEGATIVE pair (a secondary person's value that is
NOT the subject's), DeepSeek (reasoning off) says whether the document nevertheless attributes the value to
the subject. A violated negative is an attribution leak — the writer blurred whose data it is.
Also re-checks every used positive relation that involves a secondary person (the value IS that person's).
Run from the run folder with OPENROUTER_API_KEY set: python code/judge_negatives.py
Writes negatives verdicts into records_full.jsonl (`negatives_judge`) and prints the counts.
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

PROMPT = prompts.load("judge_negatives.txt")  # [pkg] see kvkk_synth/prompts/

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--prefix", default="")
PFX = _ap.parse_args().prefix
recs = [json.loads(l) for l in open(f"{PFX}records_full.jsonl", encoding="utf-8")]
n_neg = n_viol = n_pos = n_pos_ok = 0
cost = 0.0
t0 = time.time()
for r in recs:
    persons = {p["id"]: p for p in r.get("persons", [])}
    if len(persons) < 2:
        continue
    byid = {e["id"]: e for e in r["entities"]}
    subj_id = next(p["id"] for p in persons.values() if p["kind"] == "subject")
    claims, kinds = [], []
    ROLE_OBJ = {
        "authorised_signatory_of",
        "owner_of_company",
        "witness_of",
    }  # [e08] person -> company / record: "X is the signatory of «Y»"
    P2P_ROLE = {
        "represented_by",
        "guarantor_of",
        "heir_of",
        "tenant_of",
    }  # [e08] person <-> person: judged by the relation judge, not here

    def claim(name, vid, rel):
        if rel in ROLE_OBJ:
            return f"{name} is the {rel_tr.get(rel, rel)} of «{byid[vid]['span']}» ({byid[vid]['label'].replace('_', ' ')})"
        return f"«{byid[vid]['span']}» ({byid[vid]['label'].replace('_', ' ')}) is {name}'s ({rel_tr.get(rel, rel)})"

    for n in r.get("relations_negative", []):  # expected: NOT attributed to the subject
        vid = n["head"] if n["head"] != subj_id else n["tail"]
        if vid not in byid:
            continue
        claims.append(f"{len(claims)}: " + claim(persons[subj_id]["name"], vid, n["relation"]))
        kinds.append(("neg", n))
    sec_ids = {pid for pid in persons if pid != subj_id}
    for x in r["relations"]:  # positives of secondary persons: expected attributed
        pid = x["head"] if x["head"] in sec_ids else (x["tail"] if x["tail"] in sec_ids else None)
        if (
            pid is None
            or x["relation"]
            in ("spouse_of", "mother_of", "father_of", "child_of", "sibling_of", "relative_of", "emergency_contact_of")
            or x["relation"] in P2P_ROLE
        ):
            continue
        vid = x["tail"] if x["head"] == pid else x["head"]
        if vid not in byid:
            continue
        claims.append(f"{len(claims)}: " + claim(persons[pid]["name"], vid, x["relation"]))
        kinds.append(("pos", x))
    if not claims:
        continue
    body = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 1500,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(
                    text=r["text"],
                    persons="; ".join(f"{p['name']} ({p['role_en']})" for p in persons.values()),
                    claims="\n".join(claims),
                ),
            }
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
            verdict = {
                c["index"]: c
                for c in json.loads(re.search(r"\{.*\}", j["choices"][0]["message"]["content"], re.S).group(0))[
                    "checks"
                ]
            }
            cost += j.get("usage", {}).get("cost", 0.0)
            break
        except Exception as e:
            print("retry", r["bundle_id"], attempt, str(e)[:80], file=sys.stderr)
    out = []
    for i, (kind, x) in enumerate(kinds):
        c = verdict.get(i, {})
        att = bool(c.get("attributed"))
        if kind == "neg":
            n_neg += 1
            n_viol += att
            out.append(
                {
                    **x,
                    "kind": "negative",
                    "attributed_to_subject": att,
                    "violated": att,
                    "evidence": c.get("evidence", ""),
                }
            )
        else:
            n_pos += 1
            n_pos_ok += att
            out.append({**x, "kind": "secondary_positive", "attributed": att, "evidence": c.get("evidence", "")})
    r["negatives_judge"] = out
with open(f"{PFX}records_full.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(
    f"negatives: {n_neg} checked, {n_viol} violated (value wrongly attributed to the subject) | secondary positives: {n_pos_ok}/{n_pos} attributed | ${cost:.4f}, {round(time.time() - t0)}s"
)
