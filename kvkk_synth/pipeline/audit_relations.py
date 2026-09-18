#!/usr/bin/env python3
"""[e08] Mid-run audit gate (design §4.8, pipeline review item 8). After the first block of documents has been generated,
parsed and relation-judged, decide whether the relation mechanism is broken — not whether a type is merely rare or
rarely chosen. Two kinds of relation are judged by different standards:

  * pool relations: the writer picks ~fact_target facts out of 6–15 offered, so an optional relation is expected to be
    expressed in only a fraction of the bundles that offered it (block 1 of e08: 3–40 %). Broken = offered in
    ≥ --min-offered bundles and expressed in < --min-rate of them (default 2 %: zero or near-zero means the parser or
    the judge is dropping it systematically).
  * person relations (kinship + role relations of the persons axis): the secondary person MUST be named, so among the
    accepted documents that name them the relation should be expressed. Broken = ≥ --min-persons cases and expressed
    in < --min-person-rate of them (default 50 %). This is the check that caught the 'oğlu' template claim.
  * first-pass acceptance of the block's non-zero documents below --min-accept.
Exit 1 (stopping the chain) on any of the three. Writes audit-<n>.json. Run from the run folder:
  python code/audit_relations.py --n 2000
"""

import argparse
import json
import sys
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bundles", default="bundles.jsonl")
ap.add_argument("--records", default="records_full.jsonl")
ap.add_argument("--min-offered", type=int, default=50)
ap.add_argument("--min-rate", type=float, default=0.02)
ap.add_argument("--min-persons", type=int, default=10)
ap.add_argument("--min-person-rate", type=float, default=0.5)
ap.add_argument("--min-accept", type=float, default=0.80)
a = ap.parse_args()

bundles = [json.loads(l) for l in open(a.bundles, encoding="utf-8")][: a.n]
ids = {b["bundle_id"] for b in bundles}
recs = {r["bundle_id"]: r for r in (json.loads(l) for l in open(a.records, encoding="utf-8")) if r["bundle_id"] in ids}
nonzero = [b for b in bundles if b["profile"] != "zero"]
rel_ids = {x["relation"] for b in bundles for x in b["relations"]} | {
    "child_of",
    "spouse_of",
    "sibling_of",
    "mother_of",
    "father_of",
    "relative_of",
    "emergency_contact_of",
}
person_rels = {
    p["role"] for b in nonzero for p in b.get("persons", [])[1:] if p["role"] in rel_ids
}  # counterparts / bystanders have no relation by design
offered, expressed = Counter(), Counter()
for b in nonzero:
    for rid in {x["relation"] for x in b["relations"]}:
        offered[rid] += 1
for r in recs.values():
    if r.get("negative"):
        continue
    for rid in {x["relation"] for x in r["relations"]}:
        expressed[rid] += 1
accept = len([r for r in recs.values() if not r.get("negative")]) / max(len(nonzero), 1)
pool = {
    rid: {"offered": offered[rid], "expressed": expressed[rid], "rate": round(expressed[rid] / offered[rid], 3)}
    for rid in offered
    if rid not in person_rels
}
broken_pool = {rid: v for rid, v in pool.items() if v["offered"] >= a.min_offered and v["rate"] < a.min_rate}

# person relations: named in an accepted document -> expressed / judge dropped / relation absent
pc = defaultdict(Counter)
for b in nonzero:
    r = recs.get(b["bundle_id"])
    for p in b.get("persons", [])[1:]:
        rel = p["role"]
        if rel not in person_rels:
            continue
        c = pc[rel]
        c["offered"] += 1
        if r is None:
            c["doc_not_accepted"] += 1
            continue
        if p["id"] not in r.get("persons_used", []):
            c["person_not_named"] += 1
            continue
        c["named"] += 1
        if any(x["relation"] == rel and p["id"] in (x["head"], x["tail"]) for x in r["relations"]):
            c["expressed"] += 1
        elif any(
            x["relation"] == rel and p["id"] in (x["head"], x["tail"]) for x in r.get("relations_unexpressed", [])
        ):
            c["judge_dropped"] += 1
        else:
            c["relation_absent"] += 1
persons = {
    rel: {**c, "rate_when_named": round(c["expressed"] / c["named"], 3) if c["named"] else None}
    for rel, c in pc.items()
}
broken_persons = {
    rel: v for rel, v in persons.items() if v["named"] >= a.min_persons and v["rate_when_named"] < a.min_person_rate
}

report = {
    "block": a.n,
    "records": len(recs),
    "nonzero_bundles": len(nonzero),
    "acceptance_first_pass": round(accept, 3),
    "relation_types_offered": len(offered),
    "relation_types_expressed": len(expressed),
    "broken_pool": broken_pool,
    "broken_persons": broken_persons,
    "pool": pool,
    "persons": persons,
}
json.dump(report, open(f"audit-{a.n}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in report.items() if k not in ("pool", "persons")}, ensure_ascii=False))
print(
    "person relations (expressed / named):",
    {rel: f"{v.get('expressed', 0)}/{v.get('named', 0)}" for rel, v in sorted(persons.items())},
)
if broken_pool or broken_persons or accept < a.min_accept:
    print(
        f"AUDIT STOP: pool {len(broken_pool)} broken, persons {len(broken_persons)} broken, acceptance {accept:.1%} (floor {a.min_accept:.0%})",
        file=sys.stderr,
    )
    sys.exit(1)
print("audit passed")
