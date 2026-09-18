#!/bin/bash
# Container + 4-GPU NCCL sanity test on the debug queue (launchers/test_ddp.py).
# Submit: launchers/submit.sh launchers/test_gpu_debug.sh   (account, queue and log path come from launchers/env.sh)
#SBATCH --job-name=test_gpu_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=00:15:00
set -euo pipefail
: "${PROJECT:?submit through launchers/submit.sh so launchers/env.sh is exported}"
set -a; source "$PROJECT/launchers/env.sh"; set +a
cd "$PROJECT"
[ -n "${CONTAINER_MODULE:-}" ] && module load "$CONTAINER_MODULE"
export NPROC=4
bash launchers/run_test_ddp.sh "$@"
echo "=== JOB $SLURM_JOB_NAME DONE"
