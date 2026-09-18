#!/bin/bash
# RECIPE 3: two-stage curriculum with early stopping (stage 1 ~13 min, stage 2 ~25 min, checkpoint sweep ~1 h on an H100).
# Submit: launchers/submit.sh launchers/train_curriculum_1gpu.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=train_curriculum_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=1
bash recipes/train_curriculum.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
