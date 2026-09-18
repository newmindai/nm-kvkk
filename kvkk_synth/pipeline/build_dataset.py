#!/usr/bin/env python3
"""Assemble the e07 dataset: accepted documents (first pass) + accepted repairs, minus anything flagged or
discarded, into dataset/ with the taxonomy schema, the GLiNER conversion, a typed parquet, stats and an audit
of the distributions the design cares about. Also writes manifest.jsonl (one row per bundle: what happened).
Run from the run folder: python code/build_dataset.py
"""

import json
import os
import statistics as st
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taxonomy"))
import kvkk_sampler as ks  # noqa: E402  # [pkg]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


bundles = {b["bundle_id"]: b for b in load("bundles.jsonl")}
first = {r["bundle_id"]: r for r in load("records_full.jsonl")}
repaired = {r["bundle_id"]: r for r in load("repair-records_full.jsonl")}
flagged = {r["bundle_id"]: r for r in load("flagged.jsonl")}
rejects = {r["bundle_id"]: r for r in load("rejects.jsonl")}
rep_rej = {r["bundle_id"]: r for r in load("repair-rejects.jsonl")}
rep_flag = {r["bundle_id"]: r for r in load("repair-flagged.jsonl")}
sexfix = {
    r["bundle_id"]: r for r in load("sexfix-records_full.jsonl")
}  # relatives with corrected sex/name/kinship words: overrides both passes
# [e08] sex-inconsistent identity fields (a male subject with a "kızlık soyadı") — found by a reviewer on chunk 02;
# the sampler must not offer maiden_name to a male subject (fixed in sampler8 for later runs), and any document that
# already carries one is flagged rather than shipped, because the text itself says the wrong thing.
SEX_ONLY = {"maiden_name": "F"}
# [e08] "Kadın" is a real but rare given name; in a PII document it reads as the common noun "woman"
# (a reviewer read "Kadın Düzgün" as a leaked role word). Excluded from the name pools for later runs.
AMBIGUOUS_FIRST_NAMES = {"Kadın"}
REF_YEAR = 2026


def _self_contradictory(r):
    """[e08] the document states an age and a birth date that cannot both be true (the sampler drew the age
    independently when date_of_birth had not been generated yet; fixed in sampler8)."""
    ages = [e["span"] for e in r.get("entities", []) if e["label"] == "age"]
    dobs = [e["span"] for e in r.get("entities", []) if e["label"] == "date_of_birth"]
    if not ages or not dobs:
        return False
    try:
        return abs((REF_YEAR - int(dobs[0][-4:])) - int(ages[0])) > 2
    except ValueError:
        return False


for src, dst in ((first, flagged), (repaired, rep_flag), (sexfix, flagged)):
    for bid in [
        bid
        for bid, r in src.items()
        if (
            bundles.get(bid, {}).get("subject_sex")
            and any(SEX_ONLY.get(e["label"]) not in (None, bundles[bid]["subject_sex"]) for e in r.get("entities", []))
        )
        or any(
            e["label"] in ("full_name", "first_name")
            and e["span"].split()[:1]
            and e["span"].split()[0] in AMBIGUOUS_FIRST_NAMES
            for e in r.get("entities", [])
        )
        or _self_contradictory(r)
    ]:
        r = src.pop(bid)
        r["flags"] = list(r.get("flags") or []) + [
            "FLAG sex-inconsistent field, ambiguous given name, or age contradicting the birth date"
        ]
        dst.setdefault(bid, r)

# [e08] a record that a judge could not verify (judge_failed / FLAG added after parsing) is flagged, never accepted
for src, dst in ((first, flagged), (repaired, rep_flag), (sexfix, flagged)):
    for bid in [
        bid
        for bid, r in src.items()
        if r.get("judge_failed") or any(str(f).startswith("FLAG") for f in (r.get("flags") or []))
    ]:
        dst.setdefault(bid, src.pop(bid))

final, manifest = {}, []
for bid in bundles:
    if bid in sexfix:
        final[bid] = {**sexfix[bid], "bucket": "accepted", "stage": "sexfix"}
        manifest.append(
            {"bundle_id": bid, "bucket": "accepted", "stage": "sexfix", "was": "gen" if bid in first else "repair"}
        )
    elif bid in first:
        final[bid] = {**first[bid], "bucket": "accepted", "stage": "gen"}
        manifest.append({"bundle_id": bid, "bucket": "accepted", "stage": "gen"})
    elif bid in repaired:
        final[bid] = {**repaired[bid], "bucket": "accepted", "stage": "repair"}
        manifest.append(
            {
                "bundle_id": bid,
                "bucket": "accepted",
                "stage": "repair",
                "was": "flagged" if bid in flagged else "rejected",
            }
        )
    elif bid in rep_flag:
        manifest.append({"bundle_id": bid, "bucket": "flagged", "stage": "repair", "flags": rep_flag[bid].get("flags")})
    elif bid in rep_rej:
        manifest.append(
            {"bundle_id": bid, "bucket": "discarded", "stage": "repair", "problems": rep_rej[bid].get("problems")}
        )
    elif bid in flagged:
        manifest.append({"bundle_id": bid, "bucket": "flagged", "stage": "gen", "flags": flagged[bid].get("flags")})
    elif bid in rejects:
        manifest.append(
            {"bundle_id": bid, "bucket": "discarded", "stage": "gen", "problems": rejects[bid].get("problems")}
        )
    else:
        manifest.append({"bundle_id": bid, "bucket": "missing", "stage": "gen"})
os.makedirs("dataset", exist_ok=True)
recs_all = [final[b] for b in sorted(final)]
negs = [r for r in recs_all if r.get("negative")]
recs = [
    r for r in recs_all if not r.get("negative")
]  # negatives kept apart (dataset/negatives.jsonl), not in the training records
with open("manifest.jsonl", "w", encoding="utf-8") as f:
    for m in manifest:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
with open("dataset/records.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(
            json.dumps(
                {
                    "text": r["text"],
                    "entities": [{k: e[k] for k in ("id", "label", "span", "start", "end")} for e in r["entities"]],
                    "relations": r["relations"],
                    **({"relations_negative": r["relations_negative"]} if r.get("relations_negative") else {}),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
with open("dataset/records_full.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("dataset/gliner.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(
            json.dumps(
                ks.to_gliner({"text": r["text"], "entities": r["entities"], "relations": r["relations"]}),
                ensure_ascii=False,
            )
            + "\n"
        )
with open("dataset/negatives.jsonl", "w", encoding="utf-8") as f:
    for r in negs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("dataset/discarded.jsonl", "w", encoding="utf-8") as f:
    for m in manifest:
        if m["bucket"] in ("discarded", "flagged"):
            src = (
                rep_rej.get(m["bundle_id"])
                or rep_flag.get(m["bundle_id"])
                or flagged.get(m["bundle_id"])
                or rejects.get(m["bundle_id"])
            )
            if src:
                f.write(json.dumps({**src, "bucket": m["bucket"], "stage": m["stage"]}, ensure_ascii=False) + "\n")
rows = []
for r in recs:
    b = bundles[r["bundle_id"]]
    rows.append(
        {
            "bundle_id": r["bundle_id"],
            "bucket": r["bucket"],
            "stage": r["stage"],
            "profile": r.get("profile"),
            "negative": bool(r.get("negative")),
            "persons": len(b.get("persons", [])) or (1 if not r.get("negative") else 0),
            "persons_used": len(r.get("persons_used", [])),
            "domain": b["brief"]["domain"],
            "document_type": b["brief"]["document_type"],
            "document_format": b["brief"]["document_format"],
            "chars": len(r["text"]),
            "mentions": len(r["entities"]),
            "non_name_mentions": len(
                [e for e in r["entities"] if e["label"] not in ("full_name", "first_name", "last_name")]
            ),
            "writer_added": len(r.get("writer_added", [])),
            "relations": len(r["relations"]),
            "negatives": len(r.get("relations_negative", [])),
            "text": r["text"],
            "entities": json.dumps(r["entities"], ensure_ascii=False),
            "relations_json": json.dumps(r["relations"], ensure_ascii=False),
        }
    )
RUN_ID = os.path.basename(os.getcwd())  # [pkg] runs/<date>-<id>
pd.DataFrame(rows).to_parquet(f"dataset/tr_pii_relations_{RUN_ID}.parquet", index=False)

labels = Counter(e["label"] for r in recs for e in r["entities"])
rels = Counter(x["relation"] for r in recs for x in r["relations"])
nonneg = [r for r in recs if not r.get("negative")]
by_prof = {}
for p in ("zero", "sparse", "normal", "dense"):
    rr = [r for r in recs if r.get("profile") == p]
    if rr:
        by_prof[p] = {
            "docs": len(rr),
            "mentions_per_doc": round(st.mean(len(r["entities"]) for r in rr), 1),
            "chars_median": int(st.median(len(r["text"]) for r in rr)),
        }
stats = {
    "documents": len(recs),
    "negatives_kept_apart": len(negs),
    "buckets": dict(Counter(m["bucket"] for m in manifest)),
    "stages": dict(Counter(m["stage"] for m in manifest if m["bucket"] == "accepted")),
    "entity_mentions": sum(labels.values()),
    "distinct_labels": f"{len(labels)}/118",
    "relations": sum(rels.values()),
    "distinct_relation_types": f"{len(rels)}/{len(ks.load_taxonomy()[1])}",
    "negatives_pairs": sum(len(r.get("relations_negative", [])) for r in recs),
    "zero_pii_documents": sum(1 for r in recs if r.get("negative")),
    "multi_person_documents": sum(1 for r in recs if len(r.get("persons_used", [])) > 1),
    "writer_added_facts": sum(len(r.get("writer_added", [])) for r in recs),
    "mentions_per_doc": round(st.mean(len(r["entities"]) for r in nonneg), 1),
    "non_name_mentions_per_doc": round(
        st.mean(
            len([e for e in r["entities"] if e["label"] not in ("full_name", "first_name", "last_name")])
            for r in nonneg
        ),
        1,
    ),
    "subject_name_share_of_mentions": round(labels["full_name"] / max(sum(labels.values()), 1), 2),
    "chars_median": int(st.median(len(r["text"]) for r in recs)),
    "by_profile": by_prof,
    "label_counts": dict(labels.most_common()),
    "relation_counts": dict(rels.most_common()),
    "labels_below_floor_of_10": sorted(n for n in ks.load_taxonomy()[0] if labels[n] < 10),
}
json.dump(stats, open("dataset/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(
    json.dumps(
        {k: v for k, v in stats.items() if k not in ("label_counts", "relation_counts", "labels_below_floor_of_10")},
        ensure_ascii=False,
    )
)
print("labels below 10 docs:", len(stats["labels_below_floor_of_10"]))
