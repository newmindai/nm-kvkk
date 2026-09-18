#!/bin/bash
# RECIPE 1: KVKK entity champion on 4 GPUs (DDP via torchrun), ~13 min of training on 4 x H100.
# Submit: launchers/submit.sh launchers/train_entities_kvkk_ddp4.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=train_entities_kvkk_ddp4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=01:30:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=4
bash recipes/train_entities_kvkk.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
