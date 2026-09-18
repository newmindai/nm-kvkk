#!/bin/bash
# PUBLIC RECIPE: gliner2.5-multi-v1 -> nm-kvkk-pii-6K from open inputs (training ~25 min, checkpoint sweep ~1 h on an H100).
# Submit: launchers/submit.sh launchers/train_public_nm6k_1gpu.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=train_public_nm6k_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=03:00:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=1
bash recipes/train_public_nm6k.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
