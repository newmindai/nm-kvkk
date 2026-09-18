# 2026-09-14-smoke — the package's first end-to-end run (5 documents)

Purpose: prove that a fresh run folder created by `new_run.sh` goes through all 16 steps of `run.sh` with the
real writer and judges, under the single `kvkk-synth` environment. All values in the documents are synthetic.

## Inputs
| input | file | what |
|---|---|---|
| briefs | briefs.jsonl | 5 documents from the base catalogue, seed 8 |
| config | config.yaml | package defaults (GPT-5.6 Luna on flex, DeepSeek v4 Flash judges, block 2000) |

## What changed since the previous run
This is the packaged e08 code (`pipeline/`, changes marked `[pkg]`), not the pilot tree.

## What happened
`run.sh --limit 5`: embeddings 5 × 118 nodes + 23 roles (32 s), sampler 5 bundles (1 sparse / 2 normal / 2 dense,
2 multi-person, 4 negatives), writer 5/5 in 40 s ($0.006), parse 5 accepted / 0 rejected (41 mentions, 23 relations,
1 harmony fix), relation judge 23/23 expressed, audit gate passed (acceptance threshold relaxed below 50 documents),
writer-relation judge 0 pending, negatives 2/2 kept apart, untagged scan 0 findings, repair 0 candidates
(step 9 crashed once on a `[pkg]` one-liner bug, fixed, resumed with `--from 9`), sex check 0 fixes, coherence
judge mean 4.00 (3, 3, 4, 5, 5), review sample 1 chunk of 5, viewer written. Total model cost about $0.01.

## Results
`dataset/stats.json`: 5 documents, 41 mentions on 16 labels, 23 relations on 18 types, 2 negative pairs,
median 2,758 characters; every span offset-exact (`tests/check_smoke.py`).

## Reading
Five documents say nothing about distribution; they say the chain runs. `examples/e08/` has the numbers of a real build.
