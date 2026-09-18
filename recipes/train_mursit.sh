#!/bin/bash
# RECIPE 4 — the Mursit (Turkish ModernBERT) encoder that works.
#   stage 1: Mursit-Base-4k + gliner2.5 head (warm-started) -> 100k real KVKK+NER rows, 13,000 steps
#   stage 2: -> nm6k with the FULL relation schema in every record (all 113 types, descriptions), 13,000 steps
#   evaluate with BARE relation names. 4x faster than mDeBERTa, zero OOM skips; nm6k 97.5 entity-only / 92.8 REL,
#   real documents 68.9 / 36.4 (below rele07tr — a cheap encoder that works, not the best model).
#   Training Mursit without the full schema, or querying it with descriptions, collapses recall.
#
#   cluster: launchers/submit.sh launchers/train_mursit_1gpu.sh
# Inputs: models/Mursit-Base-4k (scripts/download_models.py), models/gliner2.5-multi-v1, datasets/mixed/train_100000.jsonl,
#         datasets/nm6k/tr/{train_clean.jsonl,labels.json,test.jsonl,gold_test.json}.
source "$(dirname "$0")/common.sh"
[ "$NPROC" -eq 1 ] || { echo "stage 2 trains relations: ONE GPU only"; exit 1; }
D=${D:-datasets/nm6k/tr}
OUT="$RESULTS/mursit"
S1_NAME=${S1_NAME:-mursit_stage1_mixed}
S1_CKPT=${STAGE1:-$OUT/$S1_NAME/checkpoints/final}
NAME=${NAME:-mursit_stage2_nm6k_fullrel}
need models/Mursit-Base-4k models/gliner2.5-multi-v1 datasets/mixed/train_100000.jsonl "$D/train_clean.jsonl" "$D/labels.json" "$D/test.jsonl" "$D/gold_test.json"

FULLREL="$D/train_clean_fullrel.jsonl"
if [ ! -f "$FULLREL" ]; then
  step "0) full-relation-schema training file"
  $PY scripts/add_full_relation_schema.py --in "$D/train_clean.jsonl" --labels "$D/labels.json" --out "$FULLREL"
fi
[ -f "$D/gold_test_nodesc.json" ] || make_nodesc_gold "$D/gold_test.json" "$D/gold_test_nodesc.json"

if [ -d "$S1_CKPT" ] && [ -n "${SKIP_STAGE1:-}" ]; then
  step "1) stage 1: reusing $S1_CKPT"
else
  step "1) stage 1: Mursit-Base-4k + warm-started head on real KVKK+NER text, 13,000 steps"
  train --config configs/recipes/mursit_stage1_mixed.json --train datasets/mixed/train_100000.jsonl \
        --device "$DEVICE" --max-steps "${S1_STEPS:-13000}" --run-name "$S1_NAME" --out "$OUT"
fi
[ -d "$S1_CKPT" ] || { echo "stage 1 produced no checkpoint at $S1_CKPT"; exit 1; }

step "2) stage 2: nm6k with the full relation schema, ${S2_STEPS:-13000} steps"
train --config configs/recipes/mursit_stage2_nm6k_fullrel.json --from-pretrained "$S1_CKPT" --train "$FULLREL" \
      --device "$DEVICE" --max-steps "${S2_STEPS:-13000}" --run-name "$NAME" --out "$OUT" "$@"
CKPT="$OUT/$NAME/checkpoints/final"
[ -d "$CKPT" ] || { echo "no final checkpoint at $CKPT"; exit 1; }

step "3) schema-token check (run.json['schema_tokens']) and entity-only evaluation"
$PY scripts/inspect_schema_tokens.py "$CKPT" || true
$PY scripts/eval_checkpoint.py --model "$CKPT" --data "$D/test.jsonl" --device "$DEV" --batch-size 2 --threshold 0.5 --out "$OUT/$NAME/entities_test.json"
step "4) joint prompt, BARE relation names (the shape this model was trained on)"
$PY scripts/eval_relations.py --model "$CKPT" --gold "$D/gold_test_nodesc.json" --entities "$D/test.jsonl" --thresholds "$THRESHOLDS" --device "$DEV" --out "$OUT/$NAME/rel_test_nodesc.json"
step "5) joint prompt WITH descriptions (for comparison — expect lower recall)"
$PY scripts/eval_relations.py --model "$CKPT" --gold "$D/gold_test.json" --entities "$D/test.jsonl" --thresholds 0.5 --device "$DEV" --out "$OUT/$NAME/rel_test.json"
for gt in datasets/gt/vekaletname_v1/gt.jsonl datasets/gt/mixed_v2/gt.jsonl; do
  [ -f "$gt" ] || continue
  step "6) real documents: $gt (bare names)"
  $PY scripts/bench_gt.py --model "$CKPT" --relations tr --labels configs/labels/nm6k_labels_tr.json --relation-map-tr configs/labels/nm6k_relmap_tr_nodesc.json \
      --tag "$NAME" --gt "$gt" --device "$DEV" --out-dir "$OUT/$NAME/gt_$(basename "$(dirname "$gt")")"
done
echo; echo "DONE $NAME — checkpoint: $CKPT"
