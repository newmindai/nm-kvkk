#!/usr/bin/env python3
"""Targeted repair of relatives whose sex, name and kinship words disagree (e07 dataset fix).

For every accepted document with a related person whose first name is not clearly of the role's sex, or whose
kinship word in the text contradicts the name's sex:
  1. decide the person's sex: the role's required sex (mother F, father M), else the sex the text already uses
     (kızı/oğlu …), else the name's pool sex, else random;
  2. if the current first name is not clearly of that sex, draw a new full name from the strict pool and
     rename the entity in the bundle;
  3. ask the writer (GPT-5.6 Luna, flex) to rewrite minimally: rename every occurrence, use the correct Turkish
     kinship words and gendered references, change nothing else;
  4. write raw-sexfix.parquet for parse_facts.py --prefix sexfix-, and the updated bundles to bundles.jsonl.
Run from the run folder with OPENROUTER_API_KEY set: python code/repair_sex.py
"""

import csv
import json
import os
import random
import re
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor, as_completed

import config  # [pkg]
import prompts  # [pkg] prompt files

random.seed(5)
NAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2", "names")  # [pkg]
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = config.get("writer.model")  # [pkg]
sexof = {r["name"]: r["sex"] for r in csv.DictReader(open(os.path.join(NAMES, "first_names.csv"), encoding="utf-8"))}
strict = {
    s: [
        r["name"]
        for r in csv.DictReader(open(os.path.join(NAMES, "first_names.csv"), encoding="utf-8"))
        if r["sex"] == s
    ]
    for s in "FM"
}
surnames = [r["name"] for r in csv.DictReader(open(os.path.join(NAMES, "surnames.csv"), encoding="utf-8"))]
ROLE_SEX = {"mother_of": "F", "father_of": "M"}
WORDS = {
    "F": {
        "child_of": "kızı",
        "sibling_of": "kız kardeşi / ablası",
        "spouse_of": "eşi (kadın)",
        "relative_of": "akrabası (kadın)",
        "emergency_contact_of": "acil durum kişisi (kadın)",
        "mother_of": "annesi",
        "father_of": "babası",
    },
    "M": {
        "child_of": "oğlu",
        "sibling_of": "erkek kardeşi / abisi",
        "spouse_of": "eşi (erkek)",
        "relative_of": "akrabası (erkek)",
        "emergency_contact_of": "acil durum kişisi (erkek)",
        "mother_of": "annesi",
        "father_of": "babası",
    },
}
FEM = re.compile(r"\b(kızı|kız kardeşi|ablası|hanım|bayan)\b", re.I)
MAL = re.compile(r"\b(oğlu|erkek kardeşi|abisi|ağabeyi|bey|bay)\b", re.I)

bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open("bundles.jsonl", encoding="utf-8"))}
recs = {r["bundle_id"]: r for r in (json.loads(l) for l in open("dataset/records_full.jsonl", encoding="utf-8"))}
todo = []
for bid, r in recs.items():
    b = bundles[bid]
    fixes = []
    for p in b.get("persons", [])[1:]:
        if p["kind"] != "related" or p["id"] not in r.get("persons_used", []):
            continue
        first = p["name"].split()[0]
        s = sexof.get(first, "?")
        near = " ".join(
            r["text"][max(0, m.start() - 60) : m.end() + 60] for m in re.finditer(re.escape(p["name"]), r["text"])
        )
        said = (
            "F"
            if FEM.search(near) and not MAL.search(near)
            else ("M" if MAL.search(near) and not FEM.search(near) else None)
        )
        need = ROLE_SEX.get(p["role"]) or (
            {"F": "M", "M": "F"}[b["subject_sex"]] if p["role"] == "spouse_of" and b.get("subject_sex") else None
        )  # spouse = opposite sex of the subject
        if need and s == need and (said in (None, need)):
            continue
        if not need and s in "FM" and said in (None, s):
            continue
        sex = need or said or (s if s in "FM" else random.choice("FM"))
        if s != sex:  # rename to a clearly sexed name
            new = f"{random.choice(strict[sex])} {random.choice(surnames)}"
            while new in r["text"]:
                new = f"{random.choice(strict[sex])} {random.choice(surnames)}"
        else:
            new = p["name"]
        fixes.append(
            {"pid": p["id"], "old": p["name"], "new": new, "sex": sex, "role": p["role"], "word": WORDS[sex][p["role"]]}
        )
    if fixes:
        todo.append((bid, fixes))
print(f"documents to fix: {len(todo)} | persons: {sum(len(f) for _, f in todo)}", flush=True)

PROMPT = prompts.load("repair_sex.txt")  # [pkg] see kvkk_synth/prompts/


def fix(bid, fixes):
    r = recs[bid]
    lines = "\n".join(
        f"- {f['old']} → {f['new']}: this person is {'female' if f['sex'] == 'F' else 'male'}, the subject's {f['role'].replace('_of', '').replace('_', ' ')} — Turkish role word: {f['word']}"
        for f in fixes
    )
    body = {
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": 12288,
        "messages": [{"role": "user", "content": PROMPT.format(fixes=lines, text=r["text_tagged"])}],
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
            return {
                "bundle_id": bid,
                "scenario": "pool",
                "genre": r["genre"],
                "document_format": r["document_format"],
                "profile": r.get("profile"),
                "text_tagged": j["choices"][0]["message"]["content"],
                "cost": j.get("usage", {}).get("cost", 0.0),
            }
        except Exception as e:
            print("retry", bid, attempt, str(e)[:80], file=sys.stderr)
            time.sleep(3)
    return None


t0 = time.time()
out = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for res in (f.result() for f in as_completed([ex.submit(fix, bid, fx) for bid, fx in todo])):
        if res:
            out.append(res)
# update the bundles: renamed entities, sex and role words on the persons
for bid, fixes in todo:
    b = bundles[bid]
    for f in fixes:
        for e in b["entities"]:
            if e["id"] == f["pid"]:
                e["value"] = f["new"]
        for p in b["persons"]:
            if p["id"] == f["pid"]:
                p["name"] = f["new"]
                p["sex"] = f["sex"]
                p["role_tr"] = f["word"]
        for x in b["relations"]:
            x["fact_tr"] = x["fact_tr"].replace(f["old"], f["new"])
with open("bundles.jsonl", "w", encoding="utf-8") as fh:
    for bid in sorted(bundles):
        fh.write(json.dumps(bundles[bid], ensure_ascii=False) + "\n")
df = pd.DataFrame(sorted(out, key=lambda r: r["bundle_id"]))
df.to_parquet("raw-sexfix.parquet", index=False)
json.dump(todo, open("sexfix-todo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(
    f"sex fix: {len(out)} documents rewritten, ${df.cost.sum() if len(df) else 0:.4f}, {round(time.time() - t0)}s -> raw-sexfix.parquet, bundles.jsonl updated"
)
