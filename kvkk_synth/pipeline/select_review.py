#!/usr/bin/env python3
"""Pick a stratified review sample from dataset/records_full.jsonl (default 48 documents: by profile, with every
multi-person and writer-added document over-represented, plus a few repaired ones) and write eval/chunk-NN.jsonl
in the reviewer format of build_review.py. Run from the run folder: python code/select_review.py --n 48
"""

import argparse
import json
import os
import random

random.seed(11)
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--size", type=int, default=6)
a = ap.parse_args()
recs = [json.loads(l) for l in open("dataset/records_full.jsonl", encoding="utf-8")]
bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open("bundles.jsonl", encoding="utf-8"))}
_jf = (
    "judge.json" if os.path.exists("judge.json") else next(iter(sorted(__import__("glob").glob("judge-*.json"))), None)
)  # [pkg] the coherence judge writes judge-<tag>.json
judge = {int(k): v for k, v in json.load(open(_jf, encoding="utf-8")).items()} if _jf else {}
groups = {
    "multi": [r for r in recs if len(r.get("persons_used", [])) > 1],
    "writer": [r for r in recs if r.get("writer_added") and len(r.get("persons_used", [])) <= 1],
    "repair": [r for r in recs if r.get("stage") == "repair"],
    "zero": [r for r in recs if r.get("negative")],
    "dense": [
        r
        for r in recs
        if r.get("profile") == "dense" and not r.get("writer_added") and len(r.get("persons_used", [])) <= 1
    ],
    "sparse": [
        r
        for r in recs
        if r.get("profile") == "sparse" and not r.get("writer_added") and len(r.get("persons_used", [])) <= 1
    ],
    "normal": [
        r
        for r in recs
        if r.get("profile") == "normal" and not r.get("writer_added") and len(r.get("persons_used", [])) <= 1
    ],
}
quota = {"multi": 12, "writer": 8, "repair": 6, "zero": 4, "dense": 8, "sparse": 5, "normal": 5}
chosen, seen = [], set()
for g, q in quota.items():
    pool = [r for r in groups[g] if r["bundle_id"] not in seen]
    random.shuffle(pool)
    for r in pool[:q]:
        chosen.append(r)
        seen.add(r["bundle_id"])
rest = [r for r in recs if r["bundle_id"] not in seen]
random.shuffle(rest)
chosen += rest[: max(0, a.n - len(chosen))]
chosen = chosen[: a.n]
rows = []
for r in sorted(chosen, key=lambda r: r["bundle_id"]):
    b = bundles[r["bundle_id"]]
    brief = b["brief"]
    j = judge.get(r["bundle_id"], {})
    used = {e["id"] for e in r["entities"]}
    rows.append(
        {
            "bundle_id": r["bundle_id"],
            "domain": brief["domain"],
            "document_type": brief["document_type"],
            "document_description": brief["description"],
            "document_format": brief["document_format"],
            "profile": r.get("profile"),
            "fact_target": r.get("fact_target"),
            "zero_pii_document": bool(r.get("negative")),
            "stage": r.get("stage"),
            "subject": next((p["name"] for p in b.get("persons", []) if p["kind"] == "subject"), None),
            "named_persons": [
                {"name": p["name"], "role": p["role_en"], "kind": p["kind"]} for p in b.get("persons", [])
            ],
            "facts_offered": [
                {"fact": x["fact_tr"], "relation": x["relation"], "used": x["head"] in used and x["tail"] in used}
                for x in b.get("relations", [])
            ],
            "entities_offered": [
                {"id": e["id"], "label": e["label"], "value": e["value"], "tier": "B" if False else None}
                for e in b.get("entities", [])
            ],
            "writer_allowed_kinds": b.get("writer_allowed", []),
            "writer_added": r.get("writer_added", []),
            "relations_writer": r.get("relations_writer", []),
            "relations_negative_used": r.get("relations_negative", []),
            "relations_kept": r["relations"],
            "relations_unexpressed": r.get("relations_unexpressed", []),
            "entities_bound": [
                {
                    "id": e["id"],
                    "label": e["label"],
                    "span": e["span"],
                    **({"source": e["source"]} if e.get("source") else {}),
                }
                for e in r["entities"]
            ],
            "parser_decision": "accepted",
            "parser_problems": [],
            "cheap_judge": {"coherence": j.get("coherence"), "reason": j.get("reason")} if j else None,
            "locale": b.get("locale"),
            "text_tagged": r["text_tagged"],
        }
    )
os.makedirs("eval", exist_ok=True)
n = 0
for i in range(0, len(rows), a.size):
    n += 1
    with open(f"eval/chunk-{n:02d}.jsonl", "w", encoding="utf-8") as f:
        for x in rows[i : i + a.size]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
from collections import Counter

print(
    f"review sample: {len(rows)} docs in {n} chunks | groups: {dict(Counter(next(g for g, rr in groups.items() if any(x['bundle_id'] == r['bundle_id'] for x in rr)) for r in chosen if any(any(x['bundle_id'] == r['bundle_id'] for x in rr) for rr in groups.values())))}"
)
