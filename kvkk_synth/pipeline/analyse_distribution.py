#!/usr/bin/env python3
"""[e08] Entity- and relation-type distribution of the finished training set, in the shape e07's analysis had
(analysis/entity_distribution.{md,json}) plus a relations section, since relations are the point of this build.
Run from the run folder: python code/analyse_distribution.py
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taxonomy"))  # [pkg]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvkk_sampler as ks
import tiers  # noqa: E402

ents_meta, rels_meta = ks.load_taxonomy()
rmap = {r["id"]: r for r in rels_meta}
recs = [json.loads(l) for l in open("dataset/records_full.jsonl", encoding="utf-8")]
bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open("bundles.jsonl", encoding="utf-8"))}

mentions, docs_with, writer_added = Counter(), defaultdict(set), Counter()
offered = defaultdict(set)
for r in recs:
    for e in r["entities"]:
        mentions[e["label"]] += 1
        docs_with[e["label"]].add(r["bundle_id"])
        if e.get("source") == "writer":
            writer_added[e["label"]] += 1
    for p in bundles[r["bundle_id"]].get("pool", []):
        offered[p["node"]].add(r["bundle_id"])
total = sum(mentions.values())
import re as _re

group_of = {
    n: _re.sub(r"^\d+_|\.json$", "", m.get("_file", "?")) for n, m in ents_meta.items()
}  # taxonomy file = group
madde6_labels = {n for n, m in ents_meta.items() if "Madde 6" in (m.get("_sensitivity") or "")}
tier_of = {n: ("B" if n in tiers.TIER_B else "C" if n in tiers.TIER_C else "A") for n in ents_meta}

rel_count, rel_docs = Counter(), defaultdict(set)
for r in recs:
    for x in r["relations"]:
        rel_count[x["relation"]] += 1
        rel_docs[x["relation"]].add(r["bundle_id"])
rel_total = sum(rel_count.values())
neg_count = Counter(x["relation"] for r in recs for x in r.get("relations_negative", []))
person_rels = {
    "spouse_of",
    "mother_of",
    "father_of",
    "child_of",
    "sibling_of",
    "relative_of",
    "emergency_contact_of",
    "represented_by",
    "guarantor_of",
    "heir_of",
    "tenant_of",
    "authorised_signatory_of",
    "owner_of_company",
    "witness_of",
}
struct_rels = {r["id"] for r in rels_meta if "full_name" not in r["head_types"] + r["tail_types"]}

ent = sorted(mentions.items(), key=lambda kv: -kv[1])
entropy = -sum((c / total) * math.log2(c / total) for c in mentions.values())
madde6 = sum(1 for r in recs if any(e["label"] in madde6_labels for e in r["entities"]))
by_group, by_tier = defaultdict(Counter), defaultdict(Counter)
gdocs, tdocs = defaultdict(set), defaultdict(set)
for lab, c in mentions.items():
    by_group[group_of.get(lab, "?")]["mentions"] += c
    by_group[group_of.get(lab, "?")]["labels"] += 1
    by_tier[tier_of[lab]]["mentions"] += c
    by_tier[tier_of[lab]]["labels"] += 1
    gdocs[group_of.get(lab, "?")] |= docs_with[lab]
    tdocs[tier_of[lab]] |= docs_with[lab]
group_total = Counter(group_of.get(n, "?") for n in ents_meta)
tier_total = Counter(tier_of[n] for n in ents_meta)

out = {
    "documents": len(recs),
    "mentions": total,
    "labels_seen": len(mentions),
    "labels_total": len(ents_meta),
    "labels_in_10plus_docs": sum(1 for l, d in docs_with.items() if len(d) >= 10),
    "entropy_bits": round(entropy, 2),
    "full_name_share": round(mentions["full_name"] / total, 3),
    "top5_share": round(sum(c for _, c in ent[:5]) / total, 3),
    "top10_share": round(sum(c for _, c in ent[:10]) / total, 3),
    "madde6_documents": madde6,
    "per_label": {
        lab: {
            "mentions": c,
            "documents": len(docs_with[lab]),
            "offered_in": len(offered[lab]),
            "use_rate": round(len(docs_with[lab]) / len(offered[lab]), 3) if offered[lab] else None,
            "writer_added": writer_added[lab],
            "tier": tier_of[lab],
            "group": group_of.get(lab, "?"),
        }
        for lab, c in ent
    },
    "never_seen": sorted(n for n in ents_meta if n not in mentions),
    "relations": {
        "total": rel_total,
        "types_seen": len(rel_count),
        "types_total": len(rels_meta),
        "person_relations": sum(rel_count[r] for r in person_rels),
        "structural_relations": sum(rel_count[r] for r in struct_rels),
        "negatives": sum(neg_count.values()),
        "per_type": {r: {"count": c, "documents": len(rel_docs[r])} for r, c in rel_count.most_common()},
        "never_seen": sorted(r for r in rmap if r not in rel_count),
    },
}
os.makedirs("analysis", exist_ok=True)
json.dump(out, open("analysis/entity_distribution.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

L = []
L.append(
    f"# e08 — entity- and relation-type distribution of the training set ({len(recs):,} documents, {total:,} tagged mentions)\n"
)
L.append(
    "Source: `dataset/records_full.jsonl`; numbers in `entity_distribution.json` (per label: mentions, documents, offered, use rate,\nwriter-added; per relation: count, documents). Produced by `code/analyse_distribution.py`.\n"
)
L.append("## 1. Concentration\n")
L.append("| measure | value |\n|---|---|")
L.append(f"| labels seen | {len(mentions)} of {len(ents_meta)} ({len(out['never_seen'])} never appear) |")
L.append(f"| labels in ≥ 10 documents | {out['labels_in_10plus_docs']} |")
L.append(f"| share of all mentions taken by `full_name` | {out['full_name_share']:.1%} |")
L.append(f"| top 5 labels | {out['top5_share']:.1%} of mentions |")
L.append(f"| top 10 labels | {out['top10_share']:.1%} |")
L.append(
    f"| entropy of the label distribution | {out['entropy_bits']} bits (uniform over {len(ents_meta)} would be {math.log2(len(ents_meta)):.2f}) |"
)
L.append(f"| documents with a special-category (Madde 6) label | {madde6:,} ({madde6 / len(recs):.1%}) |\n")
L.append("## 2. By taxonomy group\n")
L.append("| group | mentions | share | labels seen | documents |\n|---|---|---|---|---|")
for g, c in sorted(by_group.items(), key=lambda kv: -kv[1]["mentions"]):
    L.append(
        f"| {g} | {c['mentions']:,} | {c['mentions'] / total:.1%} | {c['labels']} / {group_total[g]} | {len(gdocs[g]):,} |"
    )
L.append("\n## 3. By tier (who generated the value)\n")
L.append("| tier | mentions | share | labels seen | documents |\n|---|---|---|---|---|")
names = {
    "A": "A · sampler, checksum recipes + e08 generators",
    "B": "B · writer-owned free text",
    "C": "C · real geography",
}
for t in ("A", "B", "C"):
    c = by_tier[t]
    L.append(
        f"| {names[t]} | {c['mentions']:,} | {c['mentions'] / total:.1%} | {c['labels']} / {tier_total[t]} | {len(tdocs[t]):,} |"
    )
L.append("\n## 4. Relations\n")
L.append("| measure | value |\n|---|---|")
L.append(f"| relation instances | {rel_total:,} |")
L.append(f"| relation types seen | {len(rel_count)} of {len(rels_meta)} |")
L.append(f"| person relations (kinship + roles) | {out['relations']['person_relations']:,} |")
L.append(f"| structural relations (value ↔ value) | {out['relations']['structural_relations']:,} |")
L.append(f"| explicit negatives | {out['relations']['negatives']:,} |")
L.append(f"| relations per document | {rel_total / len(recs):.1f} |\n")
L.append("### Role relations (the vekaletname axis)\n")
L.append("| relation | instances | documents |\n|---|---|---|")
for r in (
    "represented_by",
    "guarantor_of",
    "authorised_signatory_of",
    "owner_of_company",
    "tenant_of",
    "heir_of",
    "witness_of",
    "property_at",
):
    L.append(f"| {r} | {rel_count[r]} | {len(rel_docs[r])} |")
L.append("\n### The 15 most frequent relation types\n")
L.append("| relation | instances | documents |\n|---|---|---|")
for r, c in rel_count.most_common(15):
    L.append(f"| {r} | {c:,} | {len(rel_docs[r]):,} |")
L.append(
    f"\nRelation types never instantiated ({len(out['relations']['never_seen'])}): {', '.join(out['relations']['never_seen']) or '—'}\n"
)
L.append("## 5. The 25 most frequent labels\n")
L.append(
    "| label | tier | mentions | documents | offered in | use rate | writer-added |\n|---|---|---|---|---|---|---|"
)
for lab, c in ent[:25]:
    o = len(offered[lab])
    L.append(
        f"| {lab} | {tier_of[lab]} | {c:,} | {len(docs_with[lab]):,} | {o:,} | {(len(docs_with[lab]) / o):.0%} | {writer_added[lab]} |"
        if o
        else f"| {lab} | {tier_of[lab]} | {c:,} | {len(docs_with[lab]):,} | – | – | {writer_added[lab]} |"
    )
L.append(
    f"\n## 6. Thin labels\n\nLabels in fewer than 10 documents ({sum(1 for l in ents_meta if len(docs_with[l]) < 10)}): "
    + ", ".join(
        f"{l} ({len(docs_with[l])})"
        for l in sorted(ents_meta, key=lambda l: len(docs_with[l]))
        if len(docs_with[l]) < 10
    )
    + "\n"
)
open("analysis/entity_distribution.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(
    f"analysis/entity_distribution.md + .json | {len(mentions)} labels, {len(rel_count)} relation types, {rel_total:,} relations"
)
