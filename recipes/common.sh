# Sourced by every recipe in recipes/. One script per recipe runs unchanged on a laptop, a single GPU box and
# inside a SLURM job — only these environment variables differ:
#
#   PY        how to run python                default: python
#             cluster: PY="singularity exec --nv $SIF python"  (launchers/env.sh)
#   DEVICE    cuda | mps | cpu | auto           default: auto (cuda > mps > cpu)
#   NPROC     GPUs for torchrun                 default: 1 (plain python); >1 = torchrun --standalone --nproc_per_node
#   RESULTS   results root                      default: results
#   OFFLINE   1 = HF_HUB_OFFLINE (no hub lookups; models are folders under models/)   default: 0
#
# Helpers: train <args>   scripts/train.py under python or torchrun
#          need <paths>   abort with a pointer to the README when an input is missing
#          step <text>    timestamped section header
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-python3}"
DEVICE="${DEVICE:-auto}"
NPROC="${NPROC:-1}"
RESULTS="${RESULTS:-results}"
THRESHOLDS="${THRESHOLDS:-0.5,0.3,0.4,0.6,0.7}"
export PYTHONPATH="$REPO/GLiNER2${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
[ "${OFFLINE:-0}" = "1" ] && export HF_HUB_OFFLINE=1
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

# The evaluation scripts take an explicit device; resolve "auto" once.
if [ "$DEVICE" = "auto" ]; then
  DEV=$($PY -c "import torch;print('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')")
else
  DEV="$DEVICE"
fi

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

need() {
  local missing=0
  for f in "$@"; do
    [ -e "$f" ] || { echo "missing input: $f"; missing=1; }
  done
  [ "$missing" = 0 ] || { echo "see README.md (data) and models/README.md (checkpoints) for how to obtain these"; exit 1; }
}

train() {
  if [ "$NPROC" -gt 1 ]; then
    $PY -m torch.distributed.run --standalone --nproc_per_node="$NPROC" scripts/train.py "$@"
  else
    $PY scripts/train.py "$@"
  fi
}

# gold_<split>_nodesc.json from gold_<split>.json: the same triples queried with bare relation names
make_nodesc_gold() {
  $PY - "$1" "$2" <<'EOF'
import json, sys
g = json.load(open(sys.argv[1], encoding="utf-8"))
g["relation_descriptions"] = {k: "" for k in g["relation_types"]}
json.dump(g, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
EOF
}

echo "recipe $(basename "$0") · repo $REPO · python: $PY · device: $DEV · nproc: $NPROC · results: $RESULTS"
