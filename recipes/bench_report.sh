#!/bin/bash
# Regenerate the benchmark report (-> results/bench_report/) for one model:
# real-document set + e07 test, summary JSON, five charts, inline examples and the filled report.
#   MODEL=models/gliner2.5-kvkk-tr-v1 TAG=kvkk-tr-v1 RELATIONS=tr recipes/bench_report.sh
#   MODEL=... TAG=... recipes/bench_report.sh --skip-e07          # real documents only
# Other models' result files under results/bench_e07_models/<tag>/ and results/gt_vekaletname/<tag>_thr*.json are picked
# up by the chart step, so re-running this for several tags builds the comparison.
source "$(dirname "$0")/common.sh"
MODEL=${MODEL:-models/gliner2.5-kvkk-tr-v1}
TAG=${TAG:-kvkk-tr-v1}
RELATIONS=${RELATIONS:-tr}          # tr = Turkish relation names · en = English type ids · none = entities only
GT=${GT:-datasets/gt/vekaletname_v1/gt.jsonl}
need "$MODEL" "$GT" configs/labels/taxonomy_labels_tr.json configs/labels/e07_relation_mapping_tr.json

step "1) real documents: $GT (entities + relations, character offsets)"
$PY scripts/bench_gt.py --model "$MODEL" --relations "$RELATIONS" --tag "$TAG" --device "$DEV" --gt "$GT" --out-dir results/gt_vekaletname

if [ "${1:-}" != "--skip-e07" ]; then
  if [ "$RELATIONS" = "tr" ]; then D=datasets/kvkk_relations/e07tr; T=e07tr; else D=datasets/kvkk_relations/e07; T=e07; fi
  need "$D/entities_test_$T.jsonl" "$D/gold_test_$T.json"
  mkdir -p "results/bench_e07_models/$TAG"
  step "2) e07 test: entities"
  $PY scripts/eval_checkpoint.py --model "$MODEL" --data "$D/entities_test_$T.jsonl" --device "$DEV" --batch-size 4 --threshold 0.5 \
      --out "results/bench_e07_models/$TAG/entities_test.json"
  if [ "$RELATIONS" != "none" ]; then
    step "3) e07 test: relations ($THRESHOLDS) + multi-person subset"
    $PY scripts/eval_relations.py --model "$MODEL" --gold "$D/gold_test_$T.json" --entities "$D/entities_test_$T.jsonl" --thresholds "$THRESHOLDS" --device "$DEV" \
        --out "results/bench_e07_models/$TAG/rel_test.json"
    $PY scripts/eval_relations.py --model "$MODEL" --gold "$D/gold_multiperson_$T.json" --entities "$D/entities_test_$T.jsonl" --thresholds "$THRESHOLDS" --device "$DEV" \
        --out "results/bench_e07_models/$TAG/rel_multiperson.json"
  fi
fi

step "4) summary + charts + tables"
$PY scripts/make_charts.py --e07-dir results/bench_e07_models --gt-dir results/gt_vekaletname --out results/bench_report
$PY scripts/make_report_tables.py --summary results/bench_report/summary.json --out results/bench_report/tables.md
step "5) examples"
$PY scripts/make_examples.py --report "results/gt_vekaletname/${TAG}_thr0.5.json" --gt "$GT" \
    --labels configs/labels/taxonomy_labels_tr.json --relation-map-tr configs/labels/e07_relation_mapping_tr.json \
    --e07-report "results/bench_e07_models/$TAG/rel_test.json" --e07-entities datasets/kvkk_relations/e07tr/entities_test_e07tr.jsonl \
    --e07-gold datasets/kvkk_relations/e07tr/gold_test_e07tr.json --out results/bench_report/examples.md
step "6) report"
TEMPLATE=${TEMPLATE:-docs/reports/REPORT_template.md}     # the filled report needs a template (not shipped); steps 1-5 do not
if [ -f "$TEMPLATE" ]; then
  $PY scripts/make_report.py --template "$TEMPLATE" --tables results/bench_report/tables.md \
      --examples results/bench_report/examples.md --summary results/bench_report/summary.json --bundle "$(basename "$REPO")" \
      --out results/bench_report/REPORT.md
  echo; echo "DONE — results/bench_report/REPORT.md and figures regenerated"
else
  echo; echo "DONE — results/bench_report/ tables, figures and examples regenerated (no report template at $TEMPLATE: REPORT.md skipped)"
fi
