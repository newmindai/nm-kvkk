#!/bin/bash
# RECIPE 3 — two-stage curriculum with EARLY STOPPING: the best real-document model.
#   stage 1: gliner2.5-multi-v1 -> 100k REAL KVKK+NER rows (entities only, relations left enabled), 13,000 steps (~13 min H100)
#   stage 2: -> nm6k synthetic entities + relations, SHORT: 4,000 steps with a checkpoint every 500 (~25 min)
#   then every checkpoint is scored; keep the one that is best on REAL documents (500 steps gave 82.2 / 44.8 on vekaletname_v1
#   vs 75.8 / 43.2 for rele07tr; by 4,000 steps it is back to ~72). In-distribution scores keep rising while real-document
#   scores fall — do not pick a checkpoint on the nm6k test alone.
#
#   cluster: launchers/submit.sh launchers/train_curriculum_1gpu.sh
#   env: S2_CFG (short|full config), S2_STEPS (4000), SKIP_STAGE1=1 to reuse an existing stage-1 checkpoint, STAGE1=<ckpt>
# Inputs: datasets/mixed/train_100000.jsonl (recipes/data_kvkk_entities.sh), datasets/nm6k/tr/{train_clean.jsonl,test.jsonl,gold_test.json}
#         (recipes/data_synth_to_gliner.sh), models/gliner2.5-multi-v1.
source "$(dirname "$0")/common.sh"
[ "$NPROC" -eq 1 ] || { echo "stage 2 trains relations: ONE GPU only (the OOM-skip path is not collective)"; exit 1; }
D=${D:-datasets/nm6k/tr}
OUT="$RESULTS/curriculum"
S1_NAME=${S1_NAME:-curriculum_stage1_mixed}
S1_CKPT=${STAGE1:-$OUT/$S1_NAME/checkpoints/final}
S2_CFG=${S2_CFG:-configs/recipes/curriculum_stage2_nm6k_short.json}
S2_STEPS=${S2_STEPS:-4000}
NAME=${NAME:-$(basename "$S2_CFG" .json)}
need datasets/mixed/train_100000.jsonl "$D/train_clean.jsonl" "$D/test.jsonl" "$D/gold_test.json" models/gliner2.5-multi-v1

if [ -d "$S1_CKPT" ] && [ -n "${SKIP_STAGE1:-}" ]; then
  step "1) stage 1: reusing $S1_CKPT"
else
  step "1) stage 1: real KVKK+NER text, 13,000 steps"
  train --config configs/recipes/curriculum_stage1_mixed.json --train datasets/mixed/train_100000.jsonl \
        --device "$DEVICE" --max-steps "${S1_STEPS:-13000}" --run-name "$S1_NAME" --out "$OUT"
fi
[ -d "$S1_CKPT" ] || { echo "stage 1 produced no checkpoint at $S1_CKPT"; exit 1; }

step "2) stage 2: nm6k entities + relations, $S2_STEPS steps ($S2_CFG)"
train --config "$S2_CFG" --from-pretrained "$S1_CKPT" --train "$D/train_clean.jsonl" \
      --device "$DEVICE" --max-steps "$S2_STEPS" --run-name "$NAME" --out "$OUT" "$@"

[ -f "$D/gold_test_nodesc.json" ] || make_nodesc_gold "$D/gold_test.json" "$D/gold_test_nodesc.json"
step "3) score every stage-2 checkpoint (in-distribution nm6k + real documents if present)"
SWEEP="$OUT/$NAME/sweep"; mkdir -p "$SWEEP"
for ck in "$OUT/$NAME"/checkpoints/checkpoint-* "$OUT/$NAME/checkpoints/final"; do
  [ -d "$ck" ] || continue
  s=$(basename "$ck" | sed 's/checkpoint-//')
  echo "--- checkpoint $s"
  $PY scripts/eval_checkpoint.py --model "$ck" --data "$D/test.jsonl" --device "$DEV" --batch-size 2 --threshold 0.5 --out "$SWEEP/entities_$s.json"
  $PY scripts/eval_relations.py --model "$ck" --gold "$D/gold_test.json" --entities "$D/test.jsonl" --thresholds 0.5 --device "$DEV" --out "$SWEEP/rel_$s.json"
  $PY scripts/eval_relations.py --model "$ck" --gold "$D/gold_test_nodesc.json" --entities "$D/test.jsonl" --thresholds 0.5 --device "$DEV" --out "$SWEEP/rel_${s}_nodesc.json"
  for gt in datasets/gt/vekaletname_v1/gt.jsonl datasets/gt/mixed_v2/gt.jsonl; do
    [ -f "$gt" ] || continue
    $PY scripts/bench_gt.py --model "$ck" --relations tr --labels configs/labels/nm6k_labels_tr.json --relation-map-tr configs/labels/nm6k_relmap_tr_nodesc.json \
        --tag "${NAME}_$s" --gt "$gt" --device "$DEV" --out-dir "$SWEEP/gt_$(basename "$(dirname "$gt")")"
  done
done
echo; echo "DONE $NAME — sweep results under $SWEEP (pick the checkpoint by the real-document scores)"
