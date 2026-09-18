#!/bin/bash
# Container + multi-GPU sanity test, called by launchers/test_gpu_debug.sh (PY, NPROC and the module come from env.sh).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-python}"; NPROC="${NPROC:-1}"
export PYTHONPATH="$REPO/GLiNER2${PYTHONPATH:+:$PYTHONPATH}"
echo "=== 1) GPU visibility"
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count(), '|', torch.cuda.get_device_name(0))"
echo "=== 2) gliner2 + dependencies import"
$PY -c "import gliner2, transformers, peft, safetensors, yaml, sentencepiece; print('imports OK | transformers', transformers.__version__, '| gliner2 from', gliner2.__file__)"
echo "=== 3) NCCL all_reduce over $NPROC GPUs"
$PY -m torch.distributed.run --standalone --nproc_per_node="$NPROC" launchers/test_ddp.py
echo "=== ALL TESTS PASSED"
