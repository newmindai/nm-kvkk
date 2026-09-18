#!/bin/zsh
# [pkg] The e08 chain, run from inside runs/<date>-<id>/. Every step is resumable: the writer skips bundles already in
# raw.parquet, and --from N restarts at step N. --dry runs steps 2-3 and renders the writer prompts instead of calling
# the API (no spend). Secrets come from the repo-root .env (OPENROUTER_API_KEY), never printed.
#   run.sh [--dry] [--limit N] [--from STEP]
set -euo pipefail
PKG=${0:a:h}; PIPE=$PKG/pipeline; PY=${KVKK_PY:-python}
DRY=0; LIMIT=0; FROM=2
while [[ $# -gt 0 ]]; do case $1 in --dry) DRY=1; shift;; --limit) LIMIT=$2; shift 2;; --from) FROM=$2; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
[[ -f $PKG/../.env ]] && { set -a; source $PKG/../.env; set +a; }
export TOKENIZERS_PARALLELISM=false
[[ -f briefs.jsonl && -f briefs_catalogue.jsonl ]] || { echo "run.sh must be executed inside a run folder created by new_run.sh"; exit 1; }
cfg() { $PY -c "import sys; sys.path.insert(0,'$PIPE'); import config; print(config.get('$1'))"; }
BLOCK=$(cfg run.block); EMB=$(cfg embeddings.model); REVIEW_N=$(cfg run.review_n); REVIEW_CHUNK=$(cfg run.review_chunk)
WJ=$(cfg writer.workers); JW=$(cfg judge.workers); SEED=$(cfg sampler.seed)
RUN_ID=$(basename $PWD)
LIM=(); [[ $LIMIT -gt 0 ]] && LIM=(--limit $LIMIT)
BLK=$BLOCK; [[ $LIMIT -gt 0 && $LIMIT -lt $BLOCK ]] && BLK=$LIMIT
AUDIT=(); [[ $LIMIT -gt 0 && $LIMIT -lt 50 ]] && AUDIT=(--min-accept 0)     # the acceptance threshold is meaningless on a tiny smoke run
step() { echo; echo "=== $1 ($(date +%H:%M:%S)) ==="; }
at() { [[ $1 -ge $FROM ]]; }
exec > >(tee -a run.log) 2>&1
echo "run.sh in $PWD: dry=$DRY limit=$LIMIT from=$FROM block=$BLK writer_workers=$WJ judge_workers=$JW"

at 2 && { step "2 embeddings"; $PY $PIPE/embed_score.py --model $EMB --tag e5 --briefs briefs_catalogue.jsonl; $PY $PIPE/role_scores.py; }
at 3 && { step "3 sample"; $PY $PIPE/sampler8.py --seed $SEED $LIM --out bundles.jsonl; }
if [[ $DRY -eq 1 ]]; then step "dry: render prompts"; $PY $PIPE/render_prompts.py --bundles bundles.jsonl $LIM --out prompts-rendered.jsonl; echo; echo "=== DRY RUN DONE ($(date +%H:%M:%S)) ==="; exit 0; fi
[[ -n ${OPENROUTER_API_KEY:-} ]] || { echo "OPENROUTER_API_KEY is not set (put it in the repo-root .env)"; exit 1; }
at 4 && { step "4a generate block 1"; $PY $PIPE/gen_direct.py --bundles bundles.jsonl --out raw.parquet --workers $WJ --limit $BLK
          step "4b parse block 1";    $PY $PIPE/v2/parse_facts.py --raw raw.parquet --bundles bundles.jsonl --facts-optional
          step "4c relation judge";   $PY $PIPE/verify_relations7.py --workers $JW
          step "4d audit gate";       $PY $PIPE/audit_relations.py --n $BLK $AUDIT; }
at 5 && { step "5 generate rest";     $PY $PIPE/gen_direct.py --bundles bundles.jsonl --out raw.parquet --workers $WJ $LIM; }
at 6 && { step "6 parse";             $PY $PIPE/v2/parse_facts.py --raw raw.parquet --bundles bundles.jsonl --facts-optional; }
at 7 && { step "7 writer judges";     $PY $PIPE/judge_writer_relations.py; $PY $PIPE/judge_negatives.py; }
at 8 && { step "8 untagged scan";     $PY $PIPE/scan_untagged.py --workers $JW; }
rows() { [[ -f $1 ]] && $PY -c "import pandas as pd, sys; sys.exit(0 if len(pd.read_parquet('$1')) else 1)"; }   # true when the parquet has rows
at 9 && { step "9 repair";            $PY $PIPE/repair.py --workers $WJ
          if rows raw-repair.parquet; then $PY $PIPE/v2/parse_facts.py --raw raw-repair.parquet --bundles bundles.jsonl --facts-optional --prefix repair-; else echo "nothing to repair"; fi; }
at 10 && { step "10 judges on repaired"; if [[ -f repair-records_full.jsonl ]]; then $PY $PIPE/judge_writer_relations.py --prefix repair-; $PY $PIPE/judge_negatives.py --prefix repair-; $PY $PIPE/scan_untagged.py --workers $JW --prefix repair-; else echo "no repaired documents"; fi; }
at 11 && { step "11 relation judge";  $PY $PIPE/verify_relations7.py --workers $JW; if [[ -f repair-records_full.jsonl ]]; then $PY $PIPE/verify_relations7.py --workers $JW --records repair-records_full.jsonl; fi; }
at 12 && { step "12 dataset";         $PY $PIPE/build_dataset.py; }
at 13 && { step "13 sex check";       $PY $PIPE/repair_sex.py
           if rows raw-sexfix.parquet; then
             $PY $PIPE/v2/parse_facts.py --raw raw-sexfix.parquet --bundles bundles.jsonl --facts-optional --prefix sexfix-
             $PY $PIPE/verify_relations7.py --workers $JW --records sexfix-records_full.jsonl; $PY $PIPE/build_dataset.py; fi; }
at 14 && { step "14 coherence";       $PY $PIPE/judge_coherence.py --records dataset/records_full.jsonl --tag $RUN_ID --workers $JW --truth none; }
at 15 && { step "15 review sample";   $PY $PIPE/select_review.py --n $REVIEW_N --size $REVIEW_CHUNK
           echo "REVIEW: eval/chunk-*.jsonl are written. In Claude Code, dispatch one Sonnet subagent per chunk with"
           echo "        $PKG/prompts/review/REVIEWER_PROMPT.md, save eval/out-NN.json, then: $PY $PIPE/aggregate_review.py"; }
at 16 && { step "16 analysis + viewer"; $PY $PIPE/analyse_distribution.py; $PY $PIPE/build_viewer7.py; }
echo; echo "=== CHAIN DONE ($(date +%H:%M:%S)) ==="
