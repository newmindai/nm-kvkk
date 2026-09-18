#!/usr/bin/env python3
"""[pkg] Draw the briefs of a run from a Nemotron catalogue: round-robin over domains (balanced), within a domain
by descending n (the common document types first) with a per-brief cap, then further passes reusing briefs up to
--max-uses when N exceeds the distinct briefs (e08: 5,000 documents over 1,787 briefs, at most 3 uses each).
Writes briefs.jsonl (one row per document, bundle_id 1..N) and briefs_catalogue.jsonl (one row per distinct brief,
the embedding input). Run: python sample_briefs.py --catalogue ../briefs/nemotron_catalogue.parquet --n 1000"""

import argparse
import json
import random

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--catalogue", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=8)
ap.add_argument("--max-uses", type=int, default=3)
ap.add_argument("--out", default="briefs.jsonl")
ap.add_argument("--catalogue-out", default="briefs_catalogue.jsonl")
a = ap.parse_args()
rng = random.Random(a.seed)
cat = pd.read_parquet(a.catalogue)
by_dom = {
    d: g.sample(frac=1, random_state=a.seed).sort_values("n", ascending=False, kind="stable").to_dict("records")
    for d, g in cat.groupby("domain")
}
domains = sorted(by_dom)
rng.shuffle(domains)
uses = {}
picked = []
for use in range(1, a.max_uses + 1):
    idx = {d: 0 for d in domains}
    while len(picked) < a.n and any(idx[d] < len(by_dom[d]) for d in domains):
        for d in domains:
            if len(picked) >= a.n:
                break
            if idx[d] >= len(by_dom[d]):
                continue
            row = by_dom[d][idx[d]]
            idx[d] += 1
            if uses.get(row["uid"], 0) >= use:
                continue
            uses[row["uid"]] = uses.get(row["uid"], 0) + 1
            picked.append(row)
    if len(picked) >= a.n:
        break
if len(picked) < a.n:
    raise SystemExit(f"only {len(picked)} briefs available at max {a.max_uses} uses; lower --n or raise --max-uses")
rng.shuffle(picked)


def brief(row):
    return {
        "uid": row["uid"],
        "domain": row["domain"],
        "document_type": row["document_type"],
        "document_format": row["document_format"],
        "description": row["document_description"],
        "source": "nemotron",
        "text": f"{row['document_type']} ({row['domain']}; {row['document_format']} document). {row['document_description']}",
    }


with open(a.out, "w", encoding="utf-8") as f:
    for i, row in enumerate(picked, 1):
        f.write(json.dumps({"bundle_id": i, **brief(row)}, ensure_ascii=False) + "\n")
seen = set()
with open(a.catalogue_out, "w", encoding="utf-8") as f:
    for row in picked:
        if row["uid"] in seen:
            continue
        seen.add(row["uid"])
        f.write(json.dumps(brief(row), ensure_ascii=False) + "\n")
print(
    f"briefs: {len(picked)} documents over {len(seen)} briefs, {len({r['domain'] for r in picked})} domains, max uses {max(uses.values())} -> {a.out}, {a.catalogue_out}"
)
