#!/bin/bash
# Smoke test on the debug queue: container GPU + trainer + evaluators (QOS=debug launchers/submit.sh launchers/smoke_debug.sh).
# Submit: launchers/submit.sh launchers/smoke_debug.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=smoke_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:20:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=1
bash recipes/smoke.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
