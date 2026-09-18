#!/usr/bin/env python3
"""Threaded generation for e07 (no Data Designer): renders the same prompts as gen_facts.py and calls OpenRouter
directly with N workers. Writes raw.parquet (bundle_id, scenario, genre, document_format, profile, text_tagged,
tokens) and appends token totals to gen.log. Resumable: rows already in raw.parquet are skipped.
Usage (from code/v2): python ../gen_direct.py --bundles ../../bundles.jsonl --out ../../raw.parquet --workers 16
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from jinja2 import Template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # [pkg]
import gen_facts as g

ap = argparse.ArgumentParser()
ap.add_argument("--bundles", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=config.get("writer.model"))
ap.add_argument("--workers", type=int, default=config.get("writer.workers"))  # [pkg] config.yaml
ap.add_argument("--max-tokens", type=int, default=config.get("writer.max_tokens"))
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
ents_meta, _ = g.ks.load_taxonomy()
bundles = [json.loads(l) for l in open(a.bundles, encoding="utf-8")]
if a.limit:
    bundles = bundles[: a.limit]
seed = g.build_seed([b for b in bundles if b["profile"] != "zero"], ents_meta, optional_facts=True).set_index(
    "bundle_id"
)
T_OPT, T_ZERO = Template(g.PROMPT_OPTIONAL), Template(g.PROMPT_ZERO)
done = {}
if os.path.exists(a.out):
    for r in pd.read_parquet(a.out).to_dict("records"):
        if str(r.get("text_tagged") or "").strip():
            done[int(r["bundle_id"])] = r  # [e08] empty outputs are retried on resume
todo = [b for b in bundles if b["bundle_id"] not in done]
print(f"{len(bundles)} bundles, {len(done)} already done, {len(todo)} to generate with {a.workers} workers", flush=True)


def render(b):
    br = b["brief"]
    if b["profile"] == "zero":
        return T_ZERO.render(
            genre=br["document_type"],
            domain=br["domain"],
            description=br["description"],
            document_format=br["document_format"],
        )
    row = seed.loc[b["bundle_id"]].to_dict()
    return T_OPT.render(**row)


def call(b):
    body = {
        "model": a.model,
        "temperature": config.get("writer.temperature"),
        "top_p": config.get("writer.top_p"),
        "max_tokens": a.max_tokens,  # [pkg]
        "messages": [{"role": "system", "content": g.SYSTEM}, {"role": "user", "content": render(b)}],
        "provider": {"only": config.get("writer.provider"), "allow_fallbacks": False},
        "reasoning": {"effort": config.get("writer.reasoning_effort")},
        "usage": {"include": True},
    }
    r = None
    for attempt in range(4):
        try:
            r = requests.post(
                URL,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json=body,
                timeout=600,
            )
            j = r.json()
            txt = j["choices"][0]["message"]["content"]
            u = j.get("usage", {})
            return {
                "bundle_id": b["bundle_id"],
                "scenario": b.get("scenario", "pool"),
                "genre": b["brief"]["document_type"],
                "document_format": b["brief"]["document_format"],
                "profile": b["profile"],
                "text_tagged": txt,
                "tokens_in": u.get("prompt_tokens", 0),
                "tokens_out": u.get("completion_tokens", 0),
                "tokens_reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                "cost": u.get("cost", 0.0),
            }
        except Exception as e:
            print(
                f"retry {b['bundle_id']} {attempt}: {str(e)[:100]} {getattr(r, 'text', '')[:120]}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(3 * (attempt + 1))
    return {
        "bundle_id": b["bundle_id"],
        "scenario": b.get("scenario", "pool"),
        "genre": b["brief"]["document_type"],
        "document_format": b["brief"]["document_format"],
        "profile": b["profile"],
        "text_tagged": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_reasoning": 0,
        "cost": 0.0,
    }


t0 = time.time()
lock = threading.Lock()
n = 0
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for res in (f.result() for f in as_completed([ex.submit(call, b) for b in todo])):
        with lock:
            done[res["bundle_id"]] = res
            n += 1
            if n % 25 == 0 or n == len(todo):
                pd.DataFrame(sorted(done.values(), key=lambda r: r["bundle_id"])).to_parquet(a.out, index=False)
                print(
                    f"  {n}/{len(todo)} done, {round(time.time() - t0)}s, cost so far ${sum(r['cost'] for r in done.values()):.3f}",
                    flush=True,
                )
df = pd.DataFrame(sorted(done.values(), key=lambda r: r["bundle_id"]))
df.to_parquet(a.out, index=False)
tot = {k: int(df[k].sum()) for k in ("tokens_in", "tokens_out", "tokens_reasoning")}
print(
    f"WROTE {a.out} {df.shape} | tokens in={tot['tokens_in']:,} out={tot['tokens_out']:,} (reasoning {tot['tokens_reasoning']:,}) | cost ${df.cost.sum():.3f} | empty outputs {(df.text_tagged == '').sum()} | {round(time.time() - t0)}s",
    flush=True,
)
