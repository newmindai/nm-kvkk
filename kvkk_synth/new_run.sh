#!/bin/zsh
# [pkg] new_run.sh <id> --n <documents> [--catalogue base|fresh] [--seed 8]
# Creates kvkk_synth/runs/<YYYY-MM-DD>-<id>/ (or $KVKK_RUNS/<...>) with config.yaml (snapshot), briefs.jsonl,
# briefs_catalogue.jsonl and a README.md stub. Then: cd into it and run ../../run.sh
set -euo pipefail
HERE=${0:a:h}; PY=${KVKK_PY:-python}
ID=${1:?usage: new_run.sh <id> --n N [--catalogue base|fresh] [--seed 8]}; shift
N=100; CAT=base; SEED=8
while [[ $# -gt 0 ]]; do case $1 in --n) N=$2; shift 2;; --catalogue) CAT=$2; shift 2;; --seed) SEED=$2; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
if [[ $CAT == fresh ]]; then CATF=$HERE/briefs/nemotron_catalogue_fresh_for_e08.parquet; else CATF=$HERE/briefs/nemotron_catalogue.parquet; fi
RUN=${KVKK_RUNS:-$HERE/runs}/$(date +%Y-%m-%d)-$ID
[[ -e $RUN ]] && { echo "exists: $RUN"; exit 1; }
mkdir -p $RUN/dataset $RUN/analysis $RUN/eval
cp $HERE/config.yaml $RUN/config.yaml
$PY $HERE/pipeline/sample_briefs.py --catalogue $CATF --n $N --seed $SEED --out $RUN/briefs.jsonl --catalogue-out $RUN/briefs_catalogue.jsonl
cat > $RUN/README.md <<EOF
# $(basename $RUN) — <one line: what this run is for>

## Inputs
| input | file | what |
|---|---|---|
| briefs | briefs.jsonl | $N documents from the $CAT catalogue, seed $SEED, at most 3 uses per brief |
| config | config.yaml | snapshot at creation |

## What changed since the previous run

## What happened

## Results

## Reading
EOF
echo "created $RUN"
echo "next: cd $RUN && $HERE/run.sh --dry --limit 5     (no API calls)"
echo "      cd $RUN && $HERE/run.sh                     (the full chain)"
