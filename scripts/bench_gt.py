"""Headless ground-truth benchmark: score one checkpoint on a gt.jsonl set
(default: vekaletname_v1) and write the
report JSON (overall + per label / per relation / per doc + every prediction
and comparison item).

Usage:
  python scripts/bench_gt.py --model models/gliner2.5-kvkk-tr-v1 --relations tr --tag kvkk-tr-v1
  python scripts/bench_gt.py --model models/gliner2.5-multi-v1 --relations en --tag gliner25-orig
  python scripts/bench_gt.py --model models/kvkk-champion --relations none --tag kvkk-champion
Options: --threshold 0.5 · --gt-only (query only the types present in the gold) · --out-dir results/gt_vekaletname
         --prompt-mode joint|split|chunked (--chunk-size 20): shorter prompts for encoders whose recall falls with prompt length
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gt_runner import GTRunner, find_file, load_gt  # noqa: E402

PROJECT_ROOT = HERE.parent
DEFAULT_GT = find_file(
    PROJECT_ROOT / "datasets/gt/vekaletname_v1/gt.jsonl", PROJECT_ROOT / "datasets/gt/mixed_v2/gt.jsonl"
) or Path("gt.jsonl")
logger = logging.getLogger("bench_gt")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tag", default=None, help="name used in the output file (default: checkpoint dir name)")
    parser.add_argument("--relations", choices=["tr", "en", "none"], default="tr")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gt-only", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--relation-map-tr", type=Path, default=None)
    parser.add_argument("--mapping-en", type=Path, default=None)
    parser.add_argument("--constraints", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results/gt_vekaletname")
    parser.add_argument(
        "--prompt-mode",
        choices=["joint", "split", "chunked"],
        default="joint",
        help="joint: one schema (default) · split: entities alone + relations alone · chunked: relations in groups",
    )
    parser.add_argument("--chunk-size", type=int, default=20, help="relation types per pass in chunked mode")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    records = load_gt(args.gt)
    gt_labels = {s["label"] for r in records for s in r["spans"]}
    gt_relations = {x["relation"] for r in records for x in r["relations"]}
    runner = GTRunner(
        args.model, args.device, args.relations, args.labels, args.relation_map_tr, args.mapping_en, args.constraints
    )
    tag = args.tag or runner.name
    logger.info(
        "%s (%s relations) on %s: %d docs, %d gold spans, %d gold relations · model loaded in %.1f s on %s",
        tag,
        args.relations,
        args.gt.parent.name,
        len(records),
        sum(len(r["spans"]) for r in records),
        sum(len(r["relations"]) for r in records),
        runner.load_s,
        runner.device,
    )

    results = {}
    started = time.time()
    for i, rec in enumerate(records, start=1):
        results[rec["docname"]] = runner.run_doc(
            rec, args.threshold, args.gt_only, gt_labels, gt_relations, args.prompt_mode, args.chunk_size
        )
        s = results[rec["docname"]]["scores"]
        logger.info(
            "%2d/%d %s · ent F1 %.3f · rel F1 %.3f (filtered %.3f) · %.1fs",
            i,
            len(records),
            rec["docname"][-12:],
            s["entities"]["strict"]["f1"],
            s["relations"]["strict"]["f1"],
            s["relations_filtered"]["strict"]["f1"],
            results[rec["docname"]]["elapsed_s"],
        )

    report = runner.build_report(
        results,
        args.gt.parent.name,
        args.threshold,
        args.gt_only,
        len(records),
        args.prompt_mode,
        args.chunk_size if args.prompt_mode == "chunked" else None,
    )
    report["tag"] = tag
    report["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    report["elapsed_s"] = round(time.time() - started, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pm = (
        ""
        if args.prompt_mode == "joint"
        else f"_pm-{args.prompt_mode}{args.chunk_size if args.prompt_mode == 'chunked' else ''}"
    )
    path = args.out_dir / f"{tag}_thr{args.threshold:g}{'_gtonly' if args.gt_only else ''}{pm}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    o = report["overall"]
    logger.info(
        "OVERALL %s: entities strict F1 %.3f (P %.3f R %.3f) lenient %.3f · relations strict F1 %.3f (P %.3f R %.3f) "
        "lenient %.3f · type-filtered strict F1 %.3f (P %.3f R %.3f) · %s",
        tag,
        o["entities"]["strict"]["f1"],
        o["entities"]["strict"]["precision"],
        o["entities"]["strict"]["recall"],
        o["entities"]["lenient"]["f1"],
        o["relations"]["strict"]["f1"],
        o["relations"]["strict"]["precision"],
        o["relations"]["strict"]["recall"],
        o["relations"]["lenient"]["f1"],
        report["overall_filtered"]["relations"]["strict"]["f1"],
        report["overall_filtered"]["relations"]["strict"]["precision"],
        report["overall_filtered"]["relations"]["strict"]["recall"],
        path,
    )


if __name__ == "__main__":
    main()
