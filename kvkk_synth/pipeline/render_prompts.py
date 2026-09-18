#!/usr/bin/env python3
"""[pkg] Render the writer prompts for a bundles file without calling any model (run.sh --dry). Uses the same
build_seed / Template path as gen_direct.py, so what is written here is what the writer would receive."""

import argparse
import json
import os
import sys

from jinja2 import Template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_facts as g

ap = argparse.ArgumentParser()
ap.add_argument("--bundles", default="bundles.jsonl")
ap.add_argument("--out", default="prompts-rendered.jsonl")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
ents_meta, _ = g.ks.load_taxonomy()
bundles = [json.loads(l) for l in open(a.bundles, encoding="utf-8")]
if a.limit:
    bundles = bundles[: a.limit]
nonzero = [b for b in bundles if b["profile"] != "zero"]
seed = g.build_seed(nonzero, ents_meta, optional_facts=True).set_index("bundle_id") if nonzero else None
T_OPT, T_ZERO = Template(g.PROMPT_OPTIONAL), Template(g.PROMPT_ZERO)
with open(a.out, "w", encoding="utf-8") as f:
    for b in bundles:
        br = b["brief"]
        if b["profile"] == "zero":
            user = T_ZERO.render(
                genre=br["document_type"],
                domain=br["domain"],
                description=br["description"],
                document_format=br["document_format"],
            )
        else:
            user = T_OPT.render(**seed.loc[b["bundle_id"]].to_dict())
        f.write(
            json.dumps(
                {"bundle_id": b["bundle_id"], "profile": b["profile"], "system": g.SYSTEM, "user": user},
                ensure_ascii=False,
            )
            + "\n"
        )
print(f"rendered {len(bundles)} prompts -> {a.out}")
