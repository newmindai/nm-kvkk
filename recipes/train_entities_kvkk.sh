#!/bin/bash
# RECIPE 1 — KVKK entity model, "the KVKK champion".
#   gliner2.5-multi-v1 -> 50k label-diverse KVKK rows, 6 epochs = 9,378 steps, effective batch 32, bf16 (CUDA).
#   Result on 4 x H100: 13 min; wording benchmark 0.932 canonical / 0.930 unseen aliases.
#
#   cluster (4 GPUs):  launchers/submit.sh launchers/train_entities_kvkk_ddp4.sh
#   single GPU:        NPROC=1 recipes/train_entities_kvkk.sh            (accumulation 4 keeps the effective batch at 32)
#   laptop (MPS):      CFG=configs/recipes/local_entities_kvkk.json NAME=local_entities_kvkk recipes/train_entities_kvkk.sh
# Inputs: datasets/kvkk/pilot/{train_50000_labeldiv,validation_2000}.jsonl (recipes/data_kvkk_entities.sh), models/gliner2.5-multi-v1.
source "$(dirname "$0")/common.sh"
CFG=${CFG:-configs/recipes/entities_kvkk_labeldiv_6ep.json}
NAME=${NAME:-entities_kvkk_labeldiv_6ep}
TRAIN=${TRAIN:-datasets/kvkk/pilot/train_50000_labeldiv.jsonl}
EVAL=${EVAL:-datasets/kvkk/pilot/validation_2000.jsonl}
OUT="$RESULTS/entities"
need "$TRAIN" "$EVAL" models/gliner2.5-multi-v1
EXTRA=()
if [ "$NPROC" -eq 1 ] && [ "$CFG" = "configs/recipes/entities_kvkk_labeldiv_6ep.json" ]; then
  EXTRA+=(--set gradient_accumulation_steps=4)   # the config is batch 8 per rank x 4 ranks
fi

step "1) train $NAME ($CFG)"
train --config "$CFG" --train "$TRAIN" --eval "$EVAL" --device "$DEVICE" --out "$OUT" --run-name "$NAME" --tensorboard ${EXTRA[@]+"${EXTRA[@]}"} "$@"
CKPT="$OUT/$NAME/checkpoints/best"; [ -d "$CKPT" ] || CKPT="$OUT/$NAME/checkpoints/final"
[ -d "$CKPT" ] || { echo "no checkpoint under $OUT/$NAME/checkpoints"; exit 1; }

step "2) label-wording benchmark: canonical + the alias variants that exist (scripts/make_eval_variants.py)"
for v in validation_2000 validation_2000_seenalias validation_2000_unseen1 validation_2000_unseen2; do
  f="datasets/kvkk/pilot/$v.jsonl"; [ -f "$f" ] || continue
  $PY scripts/eval_checkpoint.py --model "$CKPT" --data "$f" --device "$DEV" --batch-size "${EVAL_BS:-8}" --threshold 0.5 \
      --out "$OUT/$NAME/eval_$v.json"
done
echo; echo "DONE $NAME — checkpoint: $CKPT. To use it as the base of the relations recipe: BASE=$CKPT recipes/train_relations_e07tr.sh"
