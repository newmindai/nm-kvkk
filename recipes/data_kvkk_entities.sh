#!/bin/bash
# DATA RECIPE A — the pseudonymised real-text entity sets (real text, personal values masked and re-populated with synthetic entities): KVKK pilot subsets with label-name diversity, the
# label-wording eval variants, the NER pilot subsets and the 100k KVKK+NER mix used by the curriculum recipes.
#
# Inputs (parquet with character-offset entities, one row per text: {id, text, entities:[{start,end,label,text}]}):
#   datasets/kvkk/v2/final/{train,validation}.parquet   19 KVKK PII labels  (newmindai/turkish-kvkk, cleaned v2 splits)
#   datasets/ner/v2/final/{train,validation}.parquet    11 legal NER labels (newmindai/turkish-ner,  cleaned v2 splits)
# Label code -> Turkish name + description: configs/labels/labels_kvkk.json, configs/labels/labels_ner.json.
# Every subset is deterministic in --seed 42; the parameters below are the ones the shipped models were trained with
# (datasets/*/pilot/*.stats.json record them per file).
source "$(dirname "$0")/common.sh"
K=datasets/kvkk; N=datasets/ner
need "$K/v2/final/train.parquet" "$K/v2/final/validation.parquet"
mkdir -p "$K/pilot" "$N/pilot" datasets/mixed

step "1) KVKK pilot: 50k stratified train rows (<=256 words), 2k + 1k validation"
$PY scripts/make_pilot.py --parquet "$K/v2/final/train.parquet" --labels configs/labels/labels_kvkk.json --out "$K/pilot/train_50000.jsonl" \
    --target 50000 --min-per-label 4000 --min-negatives 1500 --max-words 256 --seed 42
$PY scripts/make_pilot.py --parquet "$K/v2/final/validation.parquet" --labels configs/labels/labels_kvkk.json --out "$K/pilot/validation_2000.jsonl" \
    --target 2000 --min-per-label 150 --min-negatives 60 --max-words 256 --seed 42
$PY scripts/make_pilot.py --parquet "$K/v2/final/validation.parquet" --labels configs/labels/labels_kvkk.json --out "$K/pilot/validation_1000.jsonl" \
    --target 1000 --min-per-label 75 --min-negatives 30 --max-words 256 --seed 42

step "2) label-name diversity for training (86 aliases over 19 labels, canonical kept at p=0.5)"
$PY scripts/augment_label_names.py --in "$K/pilot/train_50000.jsonl" --out "$K/pilot/train_50000_labeldiv.jsonl" --seed 42

step "3) label-wording benchmark variants of the 2k validation set (eval stays canonical for training)"
$PY scripts/make_eval_variants.py --check
$PY scripts/make_eval_variants.py --in "$K/pilot/validation_2000.jsonl" --out "$K/pilot/validation_2000_seenalias.jsonl" --variant seenalias --p-canonical 0.0 --seed 7
$PY scripts/make_eval_variants.py --in "$K/pilot/validation_2000.jsonl" --out "$K/pilot/validation_2000_unseen1.jsonl" --variant unseen1
$PY scripts/make_eval_variants.py --in "$K/pilot/validation_2000.jsonl" --out "$K/pilot/validation_2000_unseen2.jsonl" --variant unseen2

if [ -f "$N/v2/final/train.parquet" ]; then
  step "4) NER pilot (50k train, 1k validation) and the 100k KVKK+NER mix for the curriculum stage 1"
  $PY scripts/make_pilot.py --parquet "$N/v2/final/train.parquet" --labels configs/labels/labels_ner.json --out "$N/pilot/train_50000.jsonl" \
      --target 50000 --min-per-label 4000 --min-negatives 1500 --max-words 256 --seed 42
  $PY scripts/make_pilot.py --parquet "$N/v2/final/validation.parquet" --labels configs/labels/labels_ner.json --out "$N/pilot/validation_1000.jsonl" \
      --target 1000 --min-per-label 75 --min-negatives 30 --max-words 256 --seed 42
  cat "$K/pilot/train_50000.jsonl" "$N/pilot/train_50000.jsonl" > datasets/mixed/train_100000.jsonl      # the trainer shuffles
  cat "$K/pilot/validation_1000.jsonl" "$N/pilot/validation_1000.jsonl" > datasets/mixed/validation_2000.jsonl
else
  echo "(NER parquet not present: skipping the NER pilot and datasets/mixed — the curriculum recipes need them)"
fi
echo; echo "DONE — $K/pilot/{train_50000_labeldiv,validation_2000*}.jsonl for recipe 1; datasets/mixed/train_100000.jsonl for recipes 3 and 4"
