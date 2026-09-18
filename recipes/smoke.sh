#!/bin/bash
# Installation smoke test: 20 training steps (entities + relations) from gliner2.5-multi-v1 on a generated toy set,
# then both evaluators on the resulting checkpoint. ~2 min on a laptop, ~1 min on a GPU. Needs models/gliner2.5-multi-v1.
#   recipes/smoke.sh                       # local
#   launchers/submit.sh launchers/smoke_debug.sh   # cluster debug queue
source "$(dirname "$0")/common.sh"
need models/gliner2.5-multi-v1
OUT="$RESULTS/smoke"; NAME=local_smoke
step "0) toy dataset"
$PY scripts/make_toy_dataset.py --out datasets/toy
step "1) train $NAME ($DEV, ${STEPS:-20} steps)"
train --config configs/recipes/local_smoke.json --train datasets/toy/train.jsonl \
      --device "$DEVICE" --out "$OUT" --run-name "$NAME" --max-steps "${STEPS:-20}" "$@"
CKPT="$OUT/$NAME/checkpoints/final"
[ -d "$CKPT" ] || { echo "no checkpoint written at $CKPT"; exit 1; }
step "2) entities on the toy test set"
$PY scripts/eval_checkpoint.py --model "$CKPT" --data datasets/toy/test.jsonl --device "$DEV" --batch-size 4 --threshold 0.5 --out "$OUT/entities_test.json"
step "3) relations on the toy test set"
$PY scripts/eval_relations.py --model "$CKPT" --gold datasets/toy/gold_test.json --entities datasets/toy/test.jsonl \
    --thresholds 0.5 --device "$DEV" --out "$OUT/rel_test.json"
echo; echo "SMOKE OK — trainer, checkpointing and both evaluators work on this machine (results under $OUT; the scores are meaningless)."
