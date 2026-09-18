#!/bin/bash
# Score ONE checkpoint on every benchmark whose data is present. Same scripts, same thresholds
# as every number in README.md and models/README.md, so results are directly comparable.
#   MODEL=models/gliner2.5-kvkk-tr-v1 TAG=kvkk-tr-v1 RELATIONS=tr recipes/eval_all.sh
#   MODEL=results/curriculum/curriculum_stage2_nm6k_short/checkpoints/checkpoint-500 TAG=s2short-500 RELATIONS=tr VOCAB=nm6k recipes/eval_all.sh
#   MODEL=models/gliner2.5-multi-v1 TAG=gliner25-zero-shot RELATIONS=en recipes/eval_all.sh
# RELATIONS: tr = Turkish relation names (rele07tr family, nm6k-tr models) · en = English type ids · none = entity-only models
# VOCAB:     e07 (default; taxonomy names + e07 relation map) · nm6k (nm6k label/relation maps, bare names — the nm6k/Mursit models)
# PRE_NM6K=1 drops the 52 nm6k test documents that rele07 trained on (use for every model NOT trained on nm6k).
# Real-document sets (datasets/gt/*) are private; the recipe skips what is missing.
source "$(dirname "$0")/common.sh"
MODEL=${MODEL:?set MODEL=<checkpoint dir>}
TAG=${TAG:-$(basename "$MODEL")}
RELATIONS=${RELATIONS:-tr}
VOCAB=${VOCAB:-e07}
OUT="$RESULTS/eval/$TAG"; mkdir -p "$OUT"
need "$MODEL"
GT_ARGS=()
if [ "$VOCAB" = "nm6k" ]; then GT_ARGS=(--labels configs/labels/nm6k_labels_tr.json --relation-map-tr configs/labels/nm6k_relmap_tr_nodesc.json); fi

# 1) e07 test (100 synthetic docs; dead for anything trained on nm6k — 87 of its docs are in nm6k train)
if [ "$RELATIONS" = "tr" ]; then E=datasets/kvkk_relations/e07tr; T=e07tr; else E=datasets/kvkk_relations/e07; T=e07; fi
if [ -f "$E/entities_test_$T.jsonl" ] && [ -z "${SKIP_E07:-}" ]; then
  step "e07 test · entities"
  $PY scripts/eval_checkpoint.py --model "$MODEL" --data "$E/entities_test_$T.jsonl" --device "$DEV" --batch-size 4 --threshold 0.5 --out "$OUT/e07_entities.json"
  if [ "$RELATIONS" != "none" ]; then
    step "e07 test · relations ($THRESHOLDS) + multi-person subset"
    $PY scripts/eval_relations.py --model "$MODEL" --gold "$E/gold_test_$T.json" --entities "$E/entities_test_$T.jsonl" --thresholds "$THRESHOLDS" --device "$DEV" --out "$OUT/e07_rel_test.json"
    [ -f "$E/gold_multiperson_$T.json" ] && $PY scripts/eval_relations.py --model "$MODEL" --gold "$E/gold_multiperson_$T.json" --entities "$E/entities_test_$T.jsonl" --thresholds "$THRESHOLDS" --device "$DEV" --out "$OUT/e07_rel_multiperson.json"
  fi
fi

# 2) nm6k test (602 synthetic docs, 241 hard / 361 easy)
D=datasets/nm6k/tr
if [ -f "$D/test.jsonl" ]; then
  ENT="$D/test.jsonl"; GOLD="$D/gold_test.json"; SUF=""
  if [ -n "${PRE_NM6K:-}" ] && [ -f "$D/test_clean.jsonl" ]; then ENT="$D/test_clean.jsonl"; GOLD="$D/gold_test_clean.json"; SUF="_clean550"; fi
  step "nm6k test$SUF · entities (batch 2)"
  $PY scripts/eval_checkpoint.py --model "$MODEL" --data "$ENT" --device "$DEV" --batch-size 2 --threshold 0.5 --out "$OUT/nm6k_entities$SUF.json"
  if [ "$RELATIONS" != "none" ]; then
    [ -f "$D/gold_test_nodesc.json" ] || make_nodesc_gold "$D/gold_test.json" "$D/gold_test_nodesc.json"
    step "nm6k test$SUF · relations with descriptions and with bare names (prompt shape matters: report both)"
    $PY scripts/eval_relations.py --model "$MODEL" --gold "$GOLD" --entities "$ENT" --thresholds "$THRESHOLDS" --device "$DEV" --out "$OUT/nm6k_rel$SUF.json"
    [ -z "$SUF" ] && $PY scripts/eval_relations.py --model "$MODEL" --gold "$D/gold_test_nodesc.json" --entities "$ENT" --thresholds 0.5 --device "$DEV" --out "$OUT/nm6k_rel_nodesc.json"
    for split in test_hard test_easy; do
      [ -f "$D/gold_$split.json" ] && $PY scripts/eval_relations.py --model "$MODEL" --gold "$D/gold_$split.json" --entities "$D/$split.jsonl" --thresholds 0.5 --device "$DEV" --out "$OUT/nm6k_rel_$split.json"
    done
  fi
fi

# 3) real documents (private gold sets, character offsets, coreference-aware relations)
for gt in datasets/gt/vekaletname_v1/gt.jsonl datasets/gt/mixed_v2/gt.jsonl; do
  [ -f "$gt" ] || continue
  set_name=$(basename "$(dirname "$gt")")
  step "real documents · $set_name (joint prompt)"
  $PY scripts/bench_gt.py --model "$MODEL" --relations "$RELATIONS" "${GT_ARGS[@]}" --tag "$TAG" --gt "$gt" --device "$DEV" --out-dir "$OUT/gt_$set_name"
  [ "$RELATIONS" != "none" ] && $PY scripts/bench_gt.py --model "$MODEL" --relations "$RELATIONS" "${GT_ARGS[@]}" --tag "$TAG" --gt "$gt" --device "$DEV" --threshold 0.7 --out-dir "$OUT/gt_$set_name"
done
echo; echo "DONE — results under $OUT"
