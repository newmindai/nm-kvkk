#!/bin/bash
# PUBLIC RECIPE — gliner2.5-multi-v1 -> nm-kvkk-pii-6K (entities + relations) from open inputs only.
#   The released models add a stage on pseudonymised real text before this step (recipes 3 and 4); that text is not
#   distributed. This recipe trains the same stage-2 configuration directly on the published synthetic set, so anyone
#   can train a KVKK extractor from the open base and the open dataset: 4,000 steps with a checkpoint every 500, then
#   every checkpoint is scored on the nm6k test split (and on real documents if you have any under datasets/gt/).
#   Pick the checkpoint by real-document scores when you have them: in-distribution scores keep rising past the point
#   where real-document scores fall.
#
#   python scripts/download_models.py --only gliner2.5-multi-v1
#   python scripts/download_datasets.py                       # nm-kvkk-pii-6K -> datasets/nm6k/{tr,en}
#   recipes/train_public_nm6k.sh                              # laptop: STEPS=200 for a check (hours on MPS)
#   cluster: launchers/submit.sh launchers/train_public_nm6k_1gpu.sh
#   env: STEPS (4000), CFG (configs/recipes/public_nm6k.json), D (datasets/nm6k/tr), NAME
# Inputs: models/gliner2.5-multi-v1, datasets/nm6k/tr/{train.jsonl,test.jsonl,labels.json} — the published layout.
source "$(dirname "$0")/common.sh"
[ "$NPROC" -eq 1 ] || { echo "relations training runs on ONE GPU (the OOM-skip path is not collective)"; exit 1; }
D=${D:-datasets/nm6k/tr}
CFG=${CFG:-configs/recipes/public_nm6k.json}
STEPS=${STEPS:-4000}
NAME=${NAME:-$(basename "$CFG" .json)}
OUT="$RESULTS/public"
need models/gliner2.5-multi-v1 "$D/train.jsonl" "$D/test.jsonl" "$D/labels.json"

step "1) derive the training file and the relation gold from the published split (re-done when the split is newer)"
[ "$D/train_clean.jsonl" -nt "$D/train.jsonl" ] || $PY scripts/preflight_nm6k_train.py --data "$D/train.jsonl" --out "$D/train_clean.jsonl" --report "$D/preflight.json"
[ "$D/gold_test.json" -nt "$D/test.jsonl" ] || $PY scripts/convert_nm6k_gold.py --data-root "$(dirname "$D")" --variant "$(basename "$D")" --out-dir "$D" --splits test
[ "$D/gold_test_nodesc.json" -nt "$D/gold_test.json" ] || make_nodesc_gold "$D/gold_test.json" "$D/gold_test_nodesc.json"

step "2) train: gliner2.5-multi-v1 -> nm6k, $STEPS steps, checkpoint every 500 ($CFG)"
train --config "$CFG" --from-pretrained models/gliner2.5-multi-v1 --train "$D/train_clean.jsonl" \
      --device "$DEVICE" --max-steps "$STEPS" --run-name "$NAME" --out "$OUT" "$@"

step "3) score every checkpoint (nm6k test; real documents if present under datasets/gt/)"
SWEEP="$OUT/$NAME/sweep"; mkdir -p "$SWEEP"
for ck in "$OUT/$NAME"/checkpoints/checkpoint-* "$OUT/$NAME/checkpoints/final"; do
  [ -d "$ck" ] || continue
  s=$(basename "$ck" | sed 's/checkpoint-//')
  echo "--- checkpoint $s"
  $PY scripts/eval_checkpoint.py --model "$ck" --data "$D/test.jsonl" --device "$DEV" --batch-size 2 --threshold 0.5 --out "$SWEEP/entities_$s.json"
  $PY scripts/eval_relations.py --model "$ck" --gold "$D/gold_test.json" --entities "$D/test.jsonl" --thresholds 0.5 --device "$DEV" --out "$SWEEP/rel_$s.json"
  $PY scripts/eval_relations.py --model "$ck" --gold "$D/gold_test_nodesc.json" --entities "$D/test.jsonl" --thresholds 0.5 --device "$DEV" --out "$SWEEP/rel_${s}_nodesc.json"
  for gt in datasets/gt/*/gt.jsonl; do
    [ -f "$gt" ] || continue
    $PY scripts/bench_gt.py --model "$ck" --relations tr --labels configs/labels/nm6k_labels_tr.json --relation-map-tr configs/labels/nm6k_relmap_tr_nodesc.json \
        --tag "${NAME}_$s" --gt "$gt" --device "$DEV" --out-dir "$SWEEP/gt_$(basename "$(dirname "$gt")")"
  done
done
echo; echo "DONE $NAME — checkpoints under $OUT/$NAME/checkpoints, scores under $SWEEP"
