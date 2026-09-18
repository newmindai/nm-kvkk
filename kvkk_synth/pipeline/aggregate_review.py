#!/usr/bin/env python3
"""Aggregate the Sonnet reviewer outputs (eval/out-NN.json) of one experiment.

Checks every quoted issue against the document (exact substring), then writes eval/summary.json and
eval/issues.jsonl and prints the table used in the run README. Run from the run folder.
"""

import glob
import json
import os
from collections import Counter

DIMS = ["fluency", "coherence", "tag_boundaries", "relation_expression", "completeness_and_leaks"]
text = {}
for f in glob.glob("eval/chunk-*.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        text[r["bundle_id"]] = r
_jf = (
    "judge.json" if os.path.exists("judge.json") else next(iter(sorted(glob.glob("judge-*.json"))), None)
)  # [e08] judge-<tag>.json
judge = (
    {int(k): v for k, v in json.load(open(_jf, encoding="utf-8")).items() if v.get("coherence") is not None}
    if _jf
    else {}
)

docs, obs, recs = [], [], []
for f in sorted(glob.glob("eval/out-*.json")):
    d = json.load(open(f, encoding="utf-8"))
    for x in d["docs"]:
        x["_file"] = os.path.basename(f)
        docs.append(x)
    obs += d.get("global_observations", [])
    recs += d.get("recommendations", [])
seen = set()
docs = [x for x in docs if not (x["bundle_id"] in seen or seen.add(x["bundle_id"]))]

bad_quotes = 0
for x in docs:
    t = text[x["bundle_id"]]["text_tagged"]
    for i in x.get("issues", []):
        i["quote_ok"] = i.get("quote", "") in t
        bad_quotes += not i["quote_ok"]


def num(x, k):
    """[e08] a count field; some reviewers write a list of items instead of a number."""
    v = x.get(k, 0)
    return len(v) if isinstance(v, (list, tuple)) else (v if isinstance(v, (int, float)) else 0)


n = len(docs)
means = {k: round(sum(x["scores"][k] for x in docs) / n, 2) for k in DIMS}
verdicts = Counter(x["verdict"] for x in docs)
issue_types = Counter(i["type"] for x in docs for i in x.get("issues", []) if i["quote_ok"])
parser_dis = [
    (x["bundle_id"], x["parser_agreement"]["note"])
    for x in docs
    if not x.get("parser_agreement", {}).get("agree", True)
]
coh_pairs = [(judge[x["bundle_id"]]["coherence"], x["scores"]["coherence"]) for x in docs if x["bundle_id"] in judge]
agree1 = sum(1 for a, b in coh_pairs if abs(a - b) <= 1)
summary = {
    "documents_evaluated": n,
    "mean_scores": means,
    "overall_mean": round(sum(sum(x["scores"][k] for k in DIMS) / len(DIMS) for x in docs) / n, 2),
    "verdicts": dict(verdicts),
    "issue_types": dict(issue_types.most_common()),
    "docs_with_no_issues": sum(1 for x in docs if not [i for i in x.get("issues", []) if i["quote_ok"]]),
    "invalid_quotes_dropped": bad_quotes,
    "parser_disagreements": parser_dis,
    "facts_used_naturally_mean": round(sum(num(x, "facts_used_naturally") for x in docs) / n, 2),
    "counts": {
        k: sum(num(x, k) for x in docs)
        for k in (
            "facts_used_naturally",
            "facts_forced",
            "persons_natural",
            "persons_forced",
            "attribution_errors",
            "sex_errors",
            "writer_added_ok",
            "writer_added_bad",
            "tierb_misses",
            "structural_ok",
            "structural_bad",
        )
    },  # [e08]
    "value_realism_mean": round(
        sum(num(x, "value_realism") for x in docs if num(x, "value_realism"))
        / max(sum(1 for x in docs if num(x, "value_realism")), 1),
        2,
    ),
    "zero_pii_leaks": sum(num(x, "leaked_pii") for x in docs),
    "cheap_judge_vs_sonnet_coherence": {
        "n": len(coh_pairs),
        "within_1_point": agree1,
        "cheap_mean": round(sum(a for a, _ in coh_pairs) / max(len(coh_pairs), 1), 2),
        "sonnet_mean": round(sum(b for _, b in coh_pairs) / max(len(coh_pairs), 1), 2),
    },
    "per_doc": [
        {
            "bundle_id": x["bundle_id"],
            "document_type": text[x["bundle_id"]]["document_type"],
            "parser": text[x["bundle_id"]]["parser_decision"],
            "verdict": x["verdict"],
            "scores": x["scores"],
            "cheap_judge": judge.get(x["bundle_id"], {}).get("coherence"),
            "issues": len([i for i in x.get("issues", []) if i["quote_ok"]]),
        }
        for x in sorted(docs, key=lambda x: x["bundle_id"])
    ],
    "global_observations": obs,
    "recommendations": recs,
}
json.dump(summary, open("eval/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open("eval/issues.jsonl", "w", encoding="utf-8") as f:
    for x in docs:
        for i in x.get("issues", []):
            f.write(json.dumps({"bundle_id": x["bundle_id"], "verdict": x["verdict"], **i}, ensure_ascii=False) + "\n")

print(
    json.dumps(
        {k: v for k, v in summary.items() if k not in ("per_doc", "global_observations", "recommendations")},
        ensure_ascii=False,
        indent=1,
    )
)
print("\n| bundle | document type | parser | Sonnet verdict | flu | coh | tag | rel | leak | cheap judge | issues |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p in summary["per_doc"]:
    s = p["scores"]
    print(
        f"| {p['bundle_id']} | {p['document_type']} | {p['parser']} | {p['verdict']} | {s['fluency']} | {s['coherence']} | {s['tag_boundaries']} | {s['relation_expression']} | {s['completeness_and_leaks']} | {p['cheap_judge']} | {p['issues']} |"
    )
print("\nOBSERVATIONS:")
[print(" -", o) for o in obs]
print("\nRECOMMENDATIONS:")
[print(" -", r) for r in recs]
