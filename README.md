# nm-kvkk — Turkish KVKK personal-data extraction with GLiNER2.5

Training recipes, data pipeline and benchmarks for GLiNER2.5 models that extract **KVKK personal-data entities**
(a 118-node taxonomy, Turkish labels) and the **relations that attach them to persons** (92–113 Turkish-named
relation types) from Turkish legal and corporate text in one pass. Everything that produced the shipped models is
here: the patched library, the exact configurations, the synthetic-data generator, and the scoring scripts behind
every number in our reports.

```
GLiNER2/        the GLiNER2 library (fastino-ai/GLiNER2 2.0.0) with our patches: native tokenization for byte-level
                BPE encoders, a bf16 loss fix, a non-finite-gradient guard — see GLiNER2/CHANGES.md
scripts/        every tool: train.py, data converters, evaluators, the cross-model benchmark (kvkkbench/), demo.py
configs/        labels/ (vocabularies) · recipes/*.json (the training configurations that worked) · benchmark_models.json
recipes/        one shell script per workflow; runs unchanged on a laptop, one GPU, or inside a SLURM job
launchers/      SLURM wrappers + the H100 container definition (site settings live in launchers/env.sh, never in git)
kvkk_synth/     the synthetic Turkish PII + relations generator (correct by construction: sampler → writer → parser → judges)
datasets/       local data (ignored); the label vocabularies the scripts need are in configs/labels/
models/         empty; python scripts/download_models.py (models/README.md)
results/        runtime output (ignored)
tests/          pytest for the scripts and the benchmark library
```

## Quickstart — local (Apple Silicon, Linux, CPU)

```bash
conda env create -f environment.yml && conda activate gliner2-kvkk     # or: pip install -r requirements.txt (Python ≥ 3.10)
python scripts/download_models.py                                       # gliner2.5-multi-v1 + Mursit-Base (+ Mursit-Base-4k) into models/
python scripts/download_datasets.py                                     # nm-kvkk-pii-6K into datasets/nm6k/ (recipe P below)
recipes/smoke.sh                                                        # 20 training steps + both evaluators on a toy set → "SMOKE OK"
python -m pytest tests -q                                               # unit tests of the scripts (no model needed)
```

The library is used from `GLiNER2/` through `PYTHONPATH` (the recipes set it). Do not `pip install gliner2` next to
it: the shipped models rely on the patches.

## Quickstart — H100 cluster (SLURM + Singularity/Apptainer)

```bash
# once, on a machine with internet + Docker:  build launchers/gliner2_h100.sif , download models/, rsync both
cp launchers/env.sh.example launchers/env.sh && $EDITOR launchers/env.sh   # PROJECT, SIF, account, queues
QOS=debug launchers/submit.sh launchers/test_gpu_debug.sh                 # container sees the GPUs, NCCL works
QOS=debug launchers/submit.sh launchers/smoke_debug.sh                    # trainer + evaluators inside the container
launchers/submit.sh launchers/train_relations_e07tr_1gpu.sh               # a real recipe
```

Every launcher just sets the SLURM resources and calls the same `recipes/*.sh` you run locally; `PY`, `DEVICE`,
`NPROC` and `OFFLINE` come from `launchers/env.sh`.

## The recipes (what worked)

The hyper-parameters are in the configs under `configs/recipes/`; the traps are listed under "Things you must know before training" below.

| # | recipe | config | base → data | H100 time | what it gives you |
|---|---|---|---|---|---|
| 1 | `recipes/train_entities_kvkk.sh` | `entities_kvkk_labeldiv_6ep.json` | gliner2.5-multi-v1 → 50k pseudonymised real KVKK rows with label-name aliases, 6 epochs | 13 min on 4 GPUs | the **KVKK entity model**: wording-robust (0.932 canonical / 0.930 unseen aliases); base of everything below |
| 2 | `recipes/train_relations_e07tr.sh` | `relations_e07tr.json` | recipe 1 → 850 synthetic-v2 (e07) docs, Turkish relation names, 2,000 steps, 1 GPU | 12 min | **`gliner2.5-kvkk-tr-v1`** (`rele07tr`, the reference model): entities + relations, e07 test 96.2 / 69.3, real documents 75.8 / 38.5 |
| 3 | `recipes/train_curriculum.sh` | `curriculum_stage1_mixed.json` → `curriculum_stage2_nm6k_short.json` | gliner2.5 → 100k pseudonymised real KVKK+NER rows → nm-kvkk-pii-6K synthetic, **stopped early** with a checkpoint sweep | ~40 min + sweep | the best real-document model so far (82.2 / 37.9, 44.8 with type-filtered relations, at 500 stage-2 steps, one seed) |
| 4 | `recipes/train_mursit.sh` | `mursit_stage1_mixed.json` → `mursit_stage2_nm6k_fullrel.json` | Mursit-Base-4k (Turkish ModernBERT) + gliner2.5 head → same curriculum with the full relation schema in every record | ~40 min | a 3–5× faster encoder that works (nm6k 97.5 / 92.8 with bare relation names; real docs 68.9 / 36.4) |
| P | `recipes/train_public_nm6k.sh` | `public_nm6k.json` | gliner2.5-multi-v1 → nm-kvkk-pii-6K (the published synthetic set), 4,000 steps with a checkpoint sweep | ~25 min + sweep | **the recipe anyone can run**: the v2 stage-2 configuration on open inputs only, without the private real-text stage |
| — | `recipes/smoke.sh` | `local_smoke.json` | toy data | 1 min | proves the installation |
| — | `configs/recipes/local_entities_kvkk.json` | — | recipe 1 adapted to a 24 GB laptop (fp32, batch 4×8, 256 words) | hours on MPS | validates the pipeline end to end, not the numbers |

Scoring a checkpoint on every benchmark whose data you have: `MODEL=… TAG=… RELATIONS=tr recipes/eval_all.sh`.
Trying a released checkpoint by hand: `python scripts/download_models.py --only gliner2.5-kvkk-tr-v1 && python scripts/demo.py --model models/gliner2.5-kvkk-tr-v1` → http://localhost:8765.

## Data

Three data recipes feed the training recipes:

- **A** `recipes/data_kvkk_entities.sh` — pseudonymised real KVKK / NER parquet (real text, personal values masked and
  re-populated with synthetic entities) → stratified pilot subsets, label-name aliases,
  wording-benchmark variants, the 100k mixed set.
- **B** `recipes/data_synth_to_gliner.sh` — `kvkk_synth` run folders → `datasets/nm6k/{tr,en}` — the nm-kvkk-pii-6K set (brief-disjoint
  split, relation gold, preflight, full-schema file).
- **C** `recipes/data_e07_relations.sh <run>` — one `kvkk_synth` run → the synthetic-v2 (e07 / e07tr) relation sets of recipe 2.

`kvkk_synth/` generates the synthetic documents: a sampler owns every fact (checksum-valid identifiers, real
addresses, relatives, relations allowed by the taxonomy), a reasoning writer writes the document with inline tags,
a parser binds every tag back to an offered value, judges verify each relation. About 1 h and $1.50 per 1,000
documents (`kvkk_synth/README.md`). The real-text sets and the real-document gold sets are private. nm-kvkk-pii-6K is published:
`python scripts/download_datasets.py` places it under `datasets/nm6k/` (https://huggingface.co/datasets/newmindai/nm-kvkk-pii-6K),
and `recipes/train_public_nm6k.sh` trains from the open base on it. The e07 set is not distributed; the generator above rebuilds it.
Released checkpoints on Hugging Face:
- `gliner2.5-kvkk-tr-v2` — https://huggingface.co/newmindai/gliner2.5-kvkk-tr-v2 — the model to start with: GLiNER2.5-multi → 100k pseudonymised real Turkish legal/KVKK sentences → nm-kvkk-pii-6K (Turkish label names); best on real documents and on relations
- `gliner2.5-kvkk-tr-v1` — https://huggingface.co/newmindai/gliner2.5-kvkk-tr-v1 — GLiNER2.5-multi → KVKK50k → e07 (Turkish label names); best on the 19 KVKK fields
- `gliner2.5-mursit-kvkk-tr-v1` — https://huggingface.co/newmindai/gliner2.5-mursit-kvkk-tr-v1 — the same recipe as v2 on the Mursit-Base Turkish ModernBERT encoder, 3–5× faster; needs the GLiNER2 library in this repository

Each repository also ships `kvkk_schema.json` (every label and relation type with the query names that model was
trained on) and `extract_kvkk.py` (text in, JSON keyed by taxonomy ids out — see `scripts/extract_kvkk.py`).

The Mursit-based model needs the GLiNER2 library in this repository (`PYTHONPATH=GLiNER2`); with the PyPI package it
loads without error but produces wrong output, because its native-tokenization and pooling settings are ignored. The two
mDeBERTa-based models also work with `gliner2` 2.0.0 from PyPI.

## Benchmarks

Four benchmarks: the synthetic-v2 e07 test, the nm-kvkk-pii-6K test, real documents, and label-wording robustness.
The metrics are exact-match span F1 and coreference-aware strict / direction-lenient relation F1. Not every benchmark
is valid for every model generation (test-set contamination), and every relation number must state its prompt shape
(`models/README.md`). `scripts/run_benchmark.py` + `scripts/score_run.py` form the cross-model harness that scored our
checkpoints against open baselines on the same gold; the reference numbers are in the recipe table above.

## Things you must know before training

The short list — each cost real time:

1. **bf16 on CUDA is the default and safe only on this patched library** (`marginal_pair_consistency_loss`
   produced NaN gradients in bf16 upstream: finite forward, frozen run at loss 0.0). Off CUDA the trainer uses fp32.
2. **Never fine-tune with `enable_relations: false`** — it freezes the relation head while the encoder moves, and the
   checkpoint can never do relations again. Leave relations enabled even when the data has none.
3. **Relations training runs on one GPU.** The OOM-skip path is not collective; 4-GPU DDP looks successful and then
   dies in NCCL. Entity-only training with DDP is fine.
4. **Synthetic data overfits the generator.** In-distribution scores rise while real-document scores fall over the
   same checkpoints. Stop early and pick checkpoints on real documents.
5. **Prompt shape is a per-model hyper-parameter** (with / without relation descriptions, joint vs split) worth 5–10
   points. Mursit models need the full relation schema in training and bare names at inference.
6. Gold surfaces must be **whole-word matchable** after the processor appends a period to texts without terminal
   punctuation — `scripts/preflight_nm6k_train.py` catches the records that would kill a run.
7. Threshold 0.5 is the reporting baseline; **0.6–0.7 is the better relation operating point**.

## Credits and license

Built on [GLiNER2](https://github.com/fastino-ai/GLiNER2) by Fastino AI (Apache-2.0) and the
[`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1) checkpoint; Turkish encoders
[`newmindai/Mursit-Base`](https://huggingface.co/newmindai/Mursit-Base) / `Mursit-Large`. Synthetic data seeds:
NVIDIA Nemotron-PII catalogue (CC-BY-4.0, document briefs only), Turkish address hierarchy (MIT / CC0) — see
`kvkk_synth/README.md`. This repository is released under the Apache License 2.0 (`LICENSE`).
