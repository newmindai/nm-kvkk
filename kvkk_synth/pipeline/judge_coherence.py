#!/usr/bin/env python3
"""Cheap LLM coherence judge (DeepSeek v4 Flash, reasoning off), calibrated in the pilots against the Sonnet
reviewers (r = 0.66). Scores each accepted document's plausibility as a host for its facts (1-5), flags whether
the document type is a plausible host at all, and names the forced facts.

[pkg] Ported off NeMo Data Designer to the same direct threaded OpenRouter call the other judges use. The prompt
(prompts/judge_coherence.txt) and the verdict schema are unchanged; output judge-<tag>.json has the same shape.
Run from the run folder: python ../../pipeline/judge_coherence.py --records dataset/records_full.jsonl --tag <id>
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from jinja2 import Template
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # [pkg]
import prompts

URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = prompts.load("judge_coherence.txt")  # [pkg] see kvkk_synth/prompts/
# [pkg] Data Designer appended the structured-output schema to the prompt for its LLMStructuredColumnConfig; the direct
# call has to say it explicitly. The prompt file stays byte-equal to the e08 constant; this line is the schema Verdict below.
SCHEMA = (
    '\n\nAnswer with one JSON object and nothing else: {"coherence": <integer 1-5>, "plausible_host": <true|false>, '
    '"forced_facts": [<short Turkish phrase>, ...], "reason": "<one sentence>"}'
)
T = Template(PROMPT + SCHEMA)


class Verdict(BaseModel):
    coherence: int = Field(ge=1, le=5)
    plausible_host: bool
    forced_facts: list[str]
    reason: str


def parse_verdict(text):
    m = re.search(r"\{.*\}", text, re.S)
    return Verdict.model_validate_json(m.group(0)).model_dump()


def body_for(rec, model):
    extra = {"reasoning": {"enabled": False}} if model.startswith("deepseek/") else {}
    return {
        "model": model,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": T.render(
                    document_type=rec.get("document_type", rec.get("genre")),
                    domain=rec.get("domain", rec.get("scenario")),
                    text=rec["text"],
                ),
            }
        ],
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
        **extra,
    }


def judge_one(rec, post, model=None):
    """post(body) -> parsed JSON response. Three attempts; None verdict when all fail (the caller counts it)."""
    body = body_for(rec, model or config.get("judge.model"))
    for attempt in range(3):
        try:
            j = post(body)
            return (
                rec["bundle_id"],
                parse_verdict(j["choices"][0]["message"]["content"]),
                (j.get("usage") or {}).get("cost", 0.0),
            )
        except Exception as e:
            print("retry", rec["bundle_id"], attempt, str(e)[:80], file=sys.stderr)
    return rec["bundle_id"], None, 0.0


def _http_post(body):
    import requests

    key = os.environ["OPENROUTER_API_KEY"]
    return requests.post(
        URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=180
    ).json()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="dataset/records_full.jsonl")
    ap.add_argument("--model", default=config.get("judge.model"))
    ap.add_argument("--tag", default="ds")
    ap.add_argument("--workers", type=int, default=config.get("judge.workers"))
    ap.add_argument(
        "--truth", default="none", help="a {bundle_id: {coherence, verdict}} JSON to calibrate against, or none"
    )
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.records, encoding="utf-8")]
    t0 = time.time()
    out = {}
    cost = 0.0
    failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for bid, v, c in (
            f.result() for f in as_completed([ex.submit(judge_one, r, _http_post, a.model) for r in recs])
        ):
            cost += c
            if v is None:
                failed += 1
            else:
                out[int(bid)] = v
    json.dump(out, open(f"judge-{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter

    print(
        f"coherence judge: {len(out)} scored, {failed} failed | mean {sum(v['coherence'] for v in out.values()) / max(len(out), 1):.2f} | "
        f"distribution {dict(sorted(Counter(v['coherence'] for v in out.values()).items()))} | ${cost:.3f}, {round(time.time() - t0)}s"
    )
    if a.truth != "none" and os.path.exists(a.truth):
        import statistics as st

        truth = {int(k): v for k, v in json.load(open(a.truth)).items()}
        pairs = [(out[b]["coherence"], truth[b]["coherence"]) for b in out if b in truth]
        if pairs:
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
            mx, my = st.mean(xs), st.mean(ys)
            r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (
                sum((x - mx) ** 2 for x in xs) ** 0.5 * sum((y - my) ** 2 for y in ys) ** 0.5
            )
            print(
                f"  vs truth: n={len(pairs)} judge {mx:.2f} vs truth {my:.2f} r={r:.2f}, within 1 point {sum(abs(x - y) <= 1 for x, y in zip(xs, ys)) / len(pairs):.0%}"
            )
