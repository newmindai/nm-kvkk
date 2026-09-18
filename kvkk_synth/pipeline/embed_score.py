#!/usr/bin/env python3
"""Cosine scores brief × node with a local sentence-transformers model. Run from the run folder.
Usage: python code/embed_score.py --model google/embeddinggemma-300m --tag gemma
       python code/embed_score.py --model intfloat/multilingual-e5-large-instruct --tag e5
Writes scores-<tag>.json: {"model", "seconds", "scores": {bundle_id: {node: cosine}}}
"""

import argparse
import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

TASK = "identify which categories of personal data would appear inside a document of this type"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--briefs", default="briefs.jsonl")
ap.add_argument(
    "--nodes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taxonomy", "nodes.jsonl")
)  # [pkg]
a = ap.parse_args()

briefs = [json.loads(l) for l in open(a.briefs, encoding="utf-8")]
key = lambda b: str(b.get("uid") or b["bundle_id"])
nodes = [json.loads(l) for l in open(a.nodes, encoding="utf-8")]
t0 = time.time()
model = SentenceTransformer(a.model, device=a.device, token=os.environ.get("HF_TOKEN"))
if "embeddinggemma" in a.model:
    q = model.encode([b["text"] for b in briefs], prompt=f"task: {TASK} | query: ", normalize_embeddings=True)
    d = model.encode([n["text"] for n in nodes], prompt="title: none | text: ", normalize_embeddings=True)
elif "e5" in a.model and "instruct" in a.model:
    q = model.encode([f"Instruct: {TASK.capitalize()}\nQuery: {b['text']}" for b in briefs], normalize_embeddings=True)
    d = model.encode([n["text"] for n in nodes], normalize_embeddings=True)
else:
    q = model.encode([b["text"] for b in briefs], normalize_embeddings=True)
    d = model.encode([n["text"] for n in nodes], normalize_embeddings=True)
sims = np.asarray(q) @ np.asarray(d).T
out = {
    "model": a.model,
    "seconds": round(time.time() - t0, 1),
    "dims": int(np.asarray(q).shape[1]),
    "scores": {key(b): {n["id"]: float(sims[i, j]) for j, n in enumerate(nodes)} for i, b in enumerate(briefs)},
}
json.dump(out, open(f"scores-{a.tag}.json", "w", encoding="utf-8"), indent=0)
print(f"{a.model}: {len(briefs)}×{len(nodes)} cosines in {out['seconds']}s (dims {out['dims']}) -> scores-{a.tag}.json")
