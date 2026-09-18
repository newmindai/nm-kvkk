#!/bin/bash
# RECIPE 2: entities + relations (rele07tr) on ONE GPU, ~12 min of training + evaluations. BASE=<checkpoint> to change the base.
# Submit: launchers/submit.sh launchers/train_relations_e07tr_1gpu.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=train_relations_e07tr_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=1
bash recipes/train_relations_e07tr.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
