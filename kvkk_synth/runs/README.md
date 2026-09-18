# runs/ — one folder per build

`new_run.sh <id> --n <documents> [--catalogue base|fresh] [--seed 8]` creates `runs/<YYYY-MM-DD>-<id>/` with
`config.yaml` (snapshot), `briefs.jsonl`, `briefs_catalogue.jsonl` and a `README.md` stub. `run.sh` is then
executed from inside that folder and writes everything else next to the inputs:

`scores-e5.json`, `scores-roles.json` (embeddings) · `bundles.jsonl`, `soundness_report.json` (sampler) ·
`raw.parquet` (writer) · `records_full.jsonl`, `records.jsonl`, `rejects.jsonl`, `flagged.jsonl` (parser) ·
`audit-<n>.json` (gate) · `raw-repair.parquet`, `repair-*.jsonl` (repair) · `raw-sexfix.parquet`, `sexfix-*`
(sex check) · `judge-<id>.json` (coherence) · `dataset/` (records, gliner, parquet, negatives, discarded,
stats) · `manifest.jsonl` · `analysis/` · `eval/chunk-*.jsonl` (review input) · `viewer.html` · `run.log`.

Every run folder is gitignored except this file. Keep the folder: it is the audit trail of the dataset.
Write the run's README the way `examples/e08/README.md` is written: inputs, what changed, what happened,
results, reading.
