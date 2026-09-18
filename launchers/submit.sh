#!/bin/bash
# Submit a launcher with the account / queue / log path from launchers/env.sh, so the launchers themselves
# contain nothing site-specific.
#   launchers/submit.sh launchers/train_relations_e07tr_1gpu.sh              # production queue
#   QOS=debug launchers/submit.sh launchers/smoke_debug.sh                     # debug queue (SLURM_QOS_DEBUG)
#   launchers/submit.sh launchers/train_curriculum_1gpu.sh --time=06:00:00     # extra sbatch options pass through
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] || { echo "launchers/env.sh not found — copy launchers/env.sh.example and edit it"; exit 1; }
set -a; source "$HERE/env.sh"; set +a
LAUNCHER=${1:?usage: launchers/submit.sh launchers/<launcher>.sh [sbatch options]}; shift
QOS_SELECTED=${SLURM_QOS}
[ "${QOS:-}" = "debug" ] && QOS_SELECTED=${SLURM_QOS_DEBUG:-$SLURM_QOS}
mkdir -p "$LOGDIR"
sbatch --account="$SLURM_ACCOUNT" --qos="$QOS_SELECTED" --output="$LOGDIR/%x-%j.out" --chdir="$PROJECT" --export=ALL "$@" "$LAUNCHER"
