#!/bin/bash
# DATA RECIPE B — kvkk_synth run folders -> datasets/nm6k/{tr,en}/ ready for recipes 3 and 4.
#   1. kvkk_synth/dataset/build_6k.py   brief-disjoint train/test split, Turkish and English label vocabularies
#   2. scripts/convert_nm6k_gold.py     relation gold for scripts/eval_relations.py (with descriptions and bare names)
#   3. scripts/preflight_nm6k_train.py  drop records whose surfaces the trainer's whole-word matcher cannot find
#   4. scripts/add_full_relation_schema.py  the full-schema training file the Mursit recipe needs
#
#   recipes/data_synth_to_gliner.sh                                    # every run under kvkk_synth/runs with a dataset/
#   recipes/data_synth_to_gliner.sh kvkk_synth/runs/2026-09-08-e07 kvkk_synth/runs/2026-09-09-e08
#   NAME=nm-kvkk-pii-6K TEST_N=600 TEST_HARD=240 SEED=42             # the shipped set's parameters (defaults)
# The reference nm-kvkk-pii-6K set itself is described in README.md (Data); this recipe rebuilds it from runs.
source "$(dirname "$0")/common.sh"
NAME=${NAME:-nm-kvkk-pii-6K}
SYNTH="datasets/synth/$NAME"
RUNS=("$@")

step "1) build $SYNTH from ${RUNS[*]:-every run folder with dataset/records_full.jsonl}"
$PY kvkk_synth/dataset/build_6k.py --out "$SYNTH" --test-n "${TEST_N:-600}" --test-hard "${TEST_HARD:-240}" --seed "${SEED:-42}" \
    ${RUNS:+--runs "${RUNS[@]}"}
cp "$SYNTH/split_report.json" datasets/nm6k/split_report.json

for v in tr en; do
  D="datasets/nm6k/$v"; mkdir -p "$D"
  step "2) $v: copy splits + labels, build relation gold"
  for f in train test test_hard test_easy; do cp "$SYNTH/$v/$f.jsonl" "$D/$f.jsonl"; done
  cp "$SYNTH/$v/labels.json" "$D/labels.json"
  $PY scripts/convert_nm6k_gold.py --data-root "$SYNTH" --variant "$v" --out-dir "$D"
  $PY scripts/convert_nm6k_gold.py --data-root "$SYNTH" --variant "$v" --out-dir "$D" --splits test --suffix _nodesc --no-descriptions
  if [ "$v" = tr ] && [ -f configs/labels/nm6k_contaminated_for_rele07.json ]; then
    # the 52 test documents that the e07-trained models saw in training: needed to score pre-nm6k models fairly
    $PY scripts/convert_nm6k_gold.py --data-root "$SYNTH" --variant tr --out-dir "$D" --splits test --suffix _clean \
        --exclude configs/labels/nm6k_contaminated_for_rele07.json || echo "(clean-550 subset skipped: ids differ from the shipped set)"
  fi
  step "3) $v: preflight against the trainer's whole-word matcher"
  $PY scripts/preflight_nm6k_train.py --data "$D/train.jsonl" --out "$D/train_clean.jsonl" --report "$D/preflight.json"
done

step "4) tr: full relation schema in every record (Mursit recipe)"
$PY scripts/add_full_relation_schema.py --in datasets/nm6k/tr/train_clean.jsonl --labels configs/labels/nm6k_vocab_tr.json --out datasets/nm6k/tr/train_clean_fullrel.jsonl
echo; echo "DONE — datasets/nm6k/{tr,en}: train_clean.jsonl (training), test*.jsonl + gold_*.json (evaluation), labels.json"
