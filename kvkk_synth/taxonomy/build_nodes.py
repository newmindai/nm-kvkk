#!/usr/bin/env python3
"""e03 inputs: the writer decisions of e01 and e02 as labels, the 24 briefs as queries, the 118
taxonomy nodes as documents. Run from the run folder.
  labels.jsonl  {bundle_id, node, used_e01, used_e02, used_any, used_both, omitted_both}
  briefs.jsonl  {bundle_id, document_type, domain, document_format, description, text}
  nodes.jsonl   {id, group, text}  — label + aliases + description + example sentences
"""

import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "kvkk-taxonomy-v2-2"))
import kvkk_sampler as ks  # noqa: E402

TAG = re.compile(r"\[([^\[\]]+)\]([a-z][a-z0-9_]*)")
RUNS = {
    "e01": os.path.join(ROOT, "runs", "2026-09-08-e01-muse13-v22-random"),
    "e02": glob.glob(os.path.join(ROOT, "runs", "2026-09-08-e02-*"))[0],
}

used = {}  # (bid, node) -> {tag: bool}
briefs = {}
for tag, d in RUNS.items():
    bundles = {
        b["bundle_id"]: b for b in (json.loads(l) for l in open(os.path.join(d, "bundles.jsonl"), encoding="utf-8"))
    }
    docs = {}
    for f in ("records_full.jsonl", "rejects.jsonl"):
        for l in open(os.path.join(d, f), encoding="utf-8"):
            r = json.loads(l)
            docs[r["bundle_id"]] = r
    for bid, b in bundles.items():
        br = b["brief"]
        briefs[bid] = {
            "bundle_id": bid,
            "document_type": br["document_type"],
            "domain": br["domain"],
            "document_format": br["document_format"],
            "description": br["description"],
            "text": f"{br['document_type']} ({br['domain']}; {br['document_format']} document). {br['description']}",
        }
        r = docs.get(bid)
        if not r:
            continue
        tagged = {(m.group(1).lower(), m.group(2)) for m in TAG.finditer(r["text_tagged"])}
        for e in b["entities"]:
            if e["id"] == b["subject_id"] or e.get("optional"):
                continue
            used.setdefault((bid, e["label"]), {})[tag] = (e["value"].lower(), e["label"]) in tagged

with open("labels.jsonl", "w", encoding="utf-8") as f:
    for (bid, node), v in sorted(used.items()):
        a, b2 = v.get("e01"), v.get("e02")
        f.write(
            json.dumps(
                {
                    "bundle_id": bid,
                    "node": node,
                    "used_e01": a,
                    "used_e02": b2,
                    "used_any": bool(a or b2),
                    "used_both": bool(a and b2),
                    "omitted_both": (a is False and b2 is False),
                }
            )
            + "\n"
        )
with open("briefs.jsonl", "w", encoding="utf-8") as f:
    for bid in sorted(briefs):
        f.write(json.dumps(briefs[bid], ensure_ascii=False) + "\n")

ents, _ = ks.load_taxonomy()
GROUP = {
    "01": "person",
    "02": "identity",
    "03": "contact",
    "04": "civil",
    "05": "financial",
    "06": "digital",
    "07": "location",
    "08": "vehicle",
    "09": "health",
    "10": "special",
    "11": "employment",
    "12": "organization",
    "13": "legal",
    "15": "registry",
}
with open("nodes.jsonl", "w", encoding="utf-8") as f:
    for nid, e in ents.items():
        aliases = [a for a in e["labels_en"][1:] + e["labels_tr"] if a != nid][:6]
        examples = "; ".join(x.get("text", "") for x in e.get("examples", []) if x.get("text"))[:300]
        text = f"{nid.replace('_', ' ')} / {e['labels_tr'][0].replace('_', ' ')}"
        if aliases:
            text += f" ({', '.join(a.replace('_', ' ') for a in aliases)})"
        text += f". {e['description']}"
        if e.get("validation"):
            text += f" Format: {e['validation']}."
        if examples:
            text += f" Examples: {examples}"
        f.write(json.dumps({"id": nid, "group": GROUP[e["_file"][:2]], "text": text}, ensure_ascii=False) + "\n")
print(f"labels {len(used)} | briefs {len(briefs)} | nodes {len(ents)}")
