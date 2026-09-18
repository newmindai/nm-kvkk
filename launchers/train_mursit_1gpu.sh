#!/bin/bash
# RECIPE 4: Mursit encoder, stage 1 + full-schema stage 2 (~40 min of training on an H100) + bare-name evaluation.
# Submit: launchers/submit.sh launchers/train_mursit_1gpu.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=train_mursit_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=1
bash recipes/train_mursit.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
