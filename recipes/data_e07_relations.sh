#!/bin/bash
# DATA RECIPE C — one kvkk_synth run -> the e07 / e07tr relation-training sets of recipe 2.
#   e07:   850 train records (785 accepted + 65 zero-PII negatives), 100 test docs, English relation type ids
#   e07tr: the SAME split with Turkish relation names (+ Turkish descriptions) in the gold and test files, and the
#          English names kept as training aliases at p=0.4 — the naming that gained +10 relation points.
#
#   recipes/data_e07_relations.sh kvkk_synth/runs/<date>-<id>
# The run's dataset/ folder must contain tr_pii_relations_<run>.parquet and negatives.jsonl (kvkk_synth/run.sh step 12).
# The Turkish relation map (configs/labels/e07_relation_mapping_tr.json) ships with the repository.
source "$(dirname "$0")/common.sh"
RUN=${1:?usage: recipes/data_e07_relations.sh kvkk_synth/runs/<date>-<id>}
PARQUET=$(ls "$RUN"/dataset/tr_pii_relations_*.parquet 2>/dev/null | head -1)
[ -n "$PARQUET" ] || { echo "no dataset/tr_pii_relations_*.parquet under $RUN (run kvkk_synth/run.sh through step 12 first)"; exit 1; }
need "$RUN/dataset/negatives.jsonl" configs/labels/e07_relation_mapping_tr.json
E07=datasets/kvkk_relations/e07; E07TR=datasets/kvkk_relations/e07tr
MAP="$E07/relation_mapping_tr.json"

step "1) e07 (English relation ids) from $PARQUET"
$PY scripts/convert_relations_e07.py --src-dir "$RUN/dataset" --src-parquet-name "$(basename "$PARQUET")" --out-dir "$E07" --tag e07
cp configs/labels/e07_relation_mapping_tr.json "$MAP"   # the e07 build overwrites its own folder; restore the shipped map

step "2) e07tr (Turkish relation names, English aliases at p=${ALIAS_P:-0.4}) derived from the frozen e07 build"
$PY scripts/convert_relations_e07.py --frozen-src "$E07" --relation-map "$MAP" --relation-alias-p "${ALIAS_P:-0.4}" --out-dir "$E07TR" --tag e07tr
echo; echo "DONE — $E07TR/{train.jsonl, entities_test_e07tr.jsonl, gold_test_e07tr.json, gold_multiperson_e07tr.json} for recipes/train_relations_e07tr.sh"
