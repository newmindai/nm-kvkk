# kvkk_synth — synthetic Turkish PII documents with relations, correct by construction

A pipeline that builds a KVKK-taxonomy training set (118 entity types, 113 relation types) without ever asking a
model to annotate text. A **sampler** decides the facts a document may carry (checksum-valid identifiers, real
addresses, sexed relatives, the relations between them); a reasoning **writer** writes the document with each value
tagged inline as `[value]label`; a **parser** binds every tag back to an offered value and rejects anything
invented; **judges** confirm each relation is stated and no value of another person is attributed to the subject.
This is the code that produced the e07 (1,000 documents) and e08 (5,000 documents) runs behind
`processed/nm-kvkk-pii-6K`.

## Quickstart (five documents, about one cent)

```bash
cd kvkk_synth
conda env create -f environment.yml && conda activate kvkk-synth
echo 'OPENROUTER_API_KEY=sk-or-...' >> ../.env          # repository-root .env (see .env.example), never committed
python download_seeds.py                                # Nemotron-PII catalogue, geo tables, e5 model (local)
./new_run.sh smoke --n 5                                # runs/<date>-smoke/ with briefs + config snapshot
cd runs/$(date +%Y-%m-%d)-smoke
../../run.sh --dry --limit 5                            # embeddings + sampler + rendered prompts, no API call
../../run.sh --limit 5                                  # the whole chain on 5 documents (about $0.01)
open viewer.html; cat dataset/stats.json
```

Offline checks that need no key: `python -m pytest tests -q` and `python download_seeds.py --check`.

## A full build

```bash
./new_run.sh mybuild --n 5000 --catalogue fresh         # fresh = the e08 catalogue (no type shared with e07)
cd runs/$(date +%Y-%m-%d)-mybuild && ../../run.sh
```

| run | documents | model cost | wall clock |
|---|---|---|---|
| e07 | 1,000 | $1.49 | about 1 h |
| e08 | 5,000 | $9.03 | 7 h (generation 2.5 h, judges and repair the rest) |

The chain generates a first block (`run.block` in `config.yaml`, 2,000), parses and judges it, then runs the
**audit gate**: it stops the chain if a relation type offered often enough is never expressed, if a named
relative's relation is expressed in under half of the documents that name them, or if first-pass acceptance
falls under 80 %. Read `audit-<n>.json`, fix, and resume with `../../run.sh --from 5`. Every step is resumable:
the writer skips bundles already in `raw.parquet`.

## What each stage guarantees

| step | module | guarantee | writes |
|---|---|---|---|
| 2 | `embed_score.py`, `role_scores.py` | every brief scored against every entity node and every person role, locally | `scores-e5.json`, `scores-roles.json` |
| 3 | `sampler8.py` | density profile from the brief's PII affinity; a pool of candidate facts; values that pass real validators; sexed relatives; relations allowed by the taxonomy; explicit negatives; a soundness report before any spend | `bundles.jsonl`, `soundness_report.json` |
| 4–6 | `gen_direct.py`, `v2/parse_facts.py` | tagged text; accepted only if every tag is an offered value (or a valid writer-owned kind), nothing offered is untagged, no other personal data appears, the marker is present | `raw.parquet`, `records_full.jsonl`, `rejects.jsonl`, `flagged.jsonl` |
| 4c, 11 | `verify_relations7.py` | every kept relation confirmed by the judge with quoted evidence; unverifiable records flagged, never accepted | `records_full.jsonl` (in place) |
| 4d | `audit_relations.py` | the chain stops when the relation mechanism is broken | `audit-<n>.json` |
| 7 | `judge_writer_relations.py`, `judge_negatives.py` | writer-added facts attributed to the subject; no secondary person's value attributed to the subject | `records_full.jsonl` (in place) |
| 8 | `scan_untagged.py` | untagged personal data flags the document | `flagged.jsonl` |
| 9–10 | `repair.py` | one rewrite round by the same writer; repaired documents pass every check again | `raw-repair.parquet`, `repair-*.jsonl` |
| 12 | `build_dataset.py` | one bucket per bundle (accepted, flagged, discarded); zero-PII documents apart; statistics | `dataset/`, `manifest.jsonl` |
| 13 | `repair_sex.py` | relatives' name, role word and kinship word agree | `raw-sexfix.parquet`, `sexfix-*` |
| 14 | `judge_coherence.py` | a 1–5 coherence score per document (a stratifier, never a filter) | `judge-<id>.json` |
| 15 | `select_review.py`, `aggregate_review.py` | a stratified review sample; every reviewer quote verified before it counts | `eval/` |
| 16 | `analyse_distribution.py`, `build_viewer7.py` | label and relation distributions; a browsable viewer | `analysis/`, `viewer.html` |

## The review step is manual

Step 15 writes `eval/chunk-NN.jsonl` and stops. The reviewers are Claude Sonnet subagents run from Claude Code with
`prompts/review/REVIEWER_PROMPT.md` (rubric: `prompts/review/RUBRIC.md`), one per chunk, each writing
`eval/out-NN.json`; then `python ../../pipeline/aggregate_review.py`. Claude never runs through the OpenRouter
key: the reviewers are the measurement, not part of the generation loop.

## Knobs

`config.yaml` (snapshotted into each run folder): writer and judge model ids, the provider pin, reasoning
effort, temperature, workers, the audit block size, the review sample size, the sampler seed. Sampler constants
stay in `pipeline/sampler8.py` with their e08 values: `PROFILE_SHARES` (zero / sparse / normal / dense =
8 / 22 / 45 / 25 %), `PROFILE` (pool size and fact targets per profile), `LABEL_FLOOR` (150 bundles), the relation
floor (40 bundles per sampler-owned type), the role threshold (0.02) and the multi-person share (25 %).

## Building a GLiNER2 training set from runs

From the repository root, in the training environment (not this one):

```bash
recipes/data_synth_to_gliner.sh kvkk_synth/runs/<a> kvkk_synth/runs/<b>     # -> datasets/nm6k/{tr,en}/ (recipes 3, 4, benchmarks)
recipes/data_e07_relations.sh kvkk_synth/runs/<a>                            # -> datasets/kvkk_relations/e07{,tr}/ (recipe 2)
```

The first runs `dataset/build_6k.py` (two label-vocabulary variants `tr/`, `en/`, brief-disjoint train/test,
`test_hard` = documents naming more than one person), then the relation-gold conversion, the trainer preflight and the
full-schema file. The record builder is vendored from `scripts/convert_relations_train.py`, byte-compatible with the
builds that passed the overfit gate (`tests/test_build_6k.py` proves it).

## Rules of the pipeline

- Paid model calls never start without an explicit go; `--dry` runs are free.
- Embeddings run locally.
- Long documents are wanted. Density is the quality axis, length is a diversity axis.
- Zero-PII documents are kept apart from training records.
- Run folders, raw seeds, geo tables and `.env` stay local and are never committed.
- Every value in a generated document is synthetic: names, identifiers, addresses, e-mail addresses and phone
  numbers are produced by the sampler; any resemblance to a real person is coincidental.

## Provenance and licences

Nemotron-PII catalogue (NVIDIA, CC-BY-4.0; only domain, type, description, format, never text or values) ·
address hierarchy: `berkanumutlu/php-turkiye-il-ilce-adres` (MIT, source adres.nvi.gov.tr) cross-checked with
`ceyyyh/turkiye_mahalleleri` (CC0) · taxonomy v2.2 and the
8 role relations: in-repo (`taxonomy/`).

## Layout

```
README.md, environment.yml, config.yaml
download_seeds.py      seeds: catalogue (+ --check), geo tables, e5 model
new_run.sh, run.sh     run-folder creation; the 16-step resumable chain (--dry, --limit, --from)
taxonomy/              v2.2 + roles extension, kvkk_sampler.py, tr_heuristics.py, nodes.jsonl
prompts/               writer, repair and judge prompts as files; review/ rubric and reviewer prompt
pipeline/              the e08 code with [pkg]-marked mechanical changes; prompts.py, config.py, sample_briefs.py, render_prompts.py
briefs/                the two shipped catalogues + provenance; raw/ (download)
geo/                   build_geo.py + provenance; *.parquet rebuilt by download_seeds.py
examples/              e08 summary artefacts, the 50-document dry run, the smoke run
dataset/               build_6k.py (GLiNER2 records), labels.py
tests/                 pytest suite (offline) and the gate oracles
runs/                  one folder per build (gitignored)
```
