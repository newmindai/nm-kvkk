#!/bin/bash
# RECIPE 2 — entities + relations in one model, "rele07tr", the production recipe.
#   base = the KVKK champion (recipe 1 or models/kvkk-champion), data = e07tr (850 records, Turkish relation names),
#   2,000 steps, batch 4, ONE GPU (DDP is unsafe for relations training), bf16. 12 min on an H100.
#   Result: e07 test entities 96.2 / relations 69.3 (strict F1, thr 0.5); real documents 75.8 / 38.5.
#
#   cluster:  launchers/submit.sh launchers/train_relations_e07tr_1gpu.sh
#   local:    BASE=models/kvkk-champion recipes/train_relations_e07tr.sh        (hours on MPS; STEPS=200 for a check)
# Inputs: datasets/kvkk_relations/e07tr/{train.jsonl,entities_test_e07tr.jsonl,gold_test_e07tr.json,gold_multiperson_e07tr.json}
#         (recipes/data_e07_relations.sh), a base checkpoint with a relation head (BASE).
source "$(dirname "$0")/common.sh"
BASE=${BASE:-models/kvkk-champion}
NAME=${NAME:-relations_e07tr}
D=${D:-datasets/kvkk_relations/e07tr}
TAG=${TAG:-e07tr}
OUT="$RESULTS/relations"
need "$BASE" "$D/train.jsonl" "$D/entities_test_$TAG.jsonl" "$D/gold_test_$TAG.json"
[ "$NPROC" -eq 1 ] || { echo "relations training runs on ONE GPU: the OOM-skip path is not collective"; exit 1; }

step "1) train $NAME from $BASE"
train --config configs/recipes/relations_e07tr.json --from-pretrained "$BASE" --train "$D/train.jsonl" \
      --device "$DEVICE" --max-steps "${STEPS:-2000}" --run-name "$NAME" --out "$OUT" "$@"
CKPT="$OUT/$NAME/checkpoints/final"
[ -d "$CKPT" ] || { echo "no final checkpoint at $CKPT"; exit 1; }

step "2) relations: e07 test (thresholds $THRESHOLDS; 0.5 is the reporting baseline, 0.6-0.7 the better operating point)"
$PY scripts/eval_relations.py --model "$CKPT" --gold "$D/gold_test_$TAG.json" --entities "$D/entities_test_$TAG.jsonl" \
    --thresholds "$THRESHOLDS" --device "$DEV" --out "$OUT/$NAME/rel_test.json"
if [ -f "$D/gold_multiperson_$TAG.json" ]; then
  step "3) relations: multi-person subset"
  $PY scripts/eval_relations.py --model "$CKPT" --gold "$D/gold_multiperson_$TAG.json" --entities "$D/entities_test_$TAG.jsonl" \
      --thresholds "$THRESHOLDS" --device "$DEV" --out "$OUT/$NAME/rel_multiperson.json"
fi
step "4) entities: e07 test (batch 4 — larger batches OOM mDeBERTa attention on long documents)"
$PY scripts/eval_checkpoint.py --model "$CKPT" --data "$D/entities_test_$TAG.jsonl" --device "$DEV" --batch-size 4 --threshold 0.5 \
    --out "$OUT/$NAME/entities_test.json"
if [ -f datasets/kvkk_relations/v2/relations_gold.json ]; then
  step "5) relations: 29-document pilot (29 docs, English-typed gold)"
  $PY scripts/eval_relations.py --model "$CKPT" --gold datasets/kvkk_relations/v2/relations_gold.json \
      --entities datasets/kvkk_relations/v2/entities_eval.jsonl --thresholds 0.5 --device "$DEV" --out "$OUT/$NAME/rel_v2external.json"
fi
for gt in datasets/gt/vekaletname_v1/gt.jsonl datasets/gt/mixed_v2/gt.jsonl; do
  [ -f "$gt" ] || continue
  step "6) real documents: $gt (joint prompt + descriptions, the shape this recipe scores best in)"
  $PY scripts/bench_gt.py --model "$CKPT" --relations tr --tag "$NAME" --gt "$gt" --device "$DEV" --out-dir "$OUT/$NAME/gt_$(basename "$(dirname "$gt")")"
done
echo; echo "DONE $NAME — checkpoint: $CKPT"
