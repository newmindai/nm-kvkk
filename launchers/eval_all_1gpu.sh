#!/bin/bash
# Score one checkpoint on every available benchmark: MODEL=... TAG=... RELATIONS=tr|en|none launchers/submit.sh launchers/eval_all_1gpu.sh
# Submit: launchers/submit.sh launchers/eval_all_1gpu.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=eval_all_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=1
bash recipes/eval_all.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
