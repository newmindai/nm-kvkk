"""Benchmark a saved GLiNER2 extractor checkpoint on a GLiNER2 JSONL eval set.

Loads a checkpoint directory (config.json + model.safetensors, as written by
ExtractorTrainer._save_checkpoint) via AutoExtractor.from_pretrained — which
dispatches on the saved architecture (boundary/span) — and scores it with the
exact string-set span metrics from scripts/span_metrics.py, i.e. the same hook
training uses. Prints a per-label table plus micro/macro P/R/F1 and optionally
writes the full report + flat metrics as JSON.

Usage:
  python scripts/eval_checkpoint.py \\
      --model results/pilot/A_kvkk_mursit_base_native_mean/checkpoints/best \\
      --data datasets/kvkk/pilot/validation_2000.jsonl --device cpu \\
      [--batch-size 16] [--threshold 0.5] [--limit N] [--out report.json]
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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "GLiNER2"))  # local clone wins over any installed gliner2

logger = logging.getLogger("eval_checkpoint")


class EvalDataset:
    """Minimal stand-in for the trainer's eval dataset: the metrics hook only reads .data."""

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def load_model(checkpoint: Path, device: str):
    import torch

    from gliner2 import AutoExtractor

    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is not available")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    model = AutoExtractor.from_pretrained(str(checkpoint), map_location=device)
    model.eval()
    return model


def print_report(report: dict[str, Any]) -> None:
    width = max(len("label"), *(len(label) for label in report["per_label"]))
    print(f"{'label':<{width}}  {'P':>6}  {'R':>6}  {'F1':>6}  {'support':>7}")
    for label, s in report["per_label"].items():
        print(f"{label:<{width}}  {s['precision']:>6.3f}  {s['recall']:>6.3f}  {s['f1']:>6.3f}  {s['support']:>7}")
    micro, macro = report["micro"], report["macro"]
    print("-" * (width + 33))
    print(f"{'micro':<{width}}  {micro['precision']:>6.3f}  {micro['recall']:>6.3f}  {micro['f1']:>6.3f}")
    print(
        f"{'macro':<{width}}  {macro['precision']:>6.3f}  {macro['recall']:>6.3f}  {macro['f1']:>6.3f}  "
        f"({macro['n_labels']} labels with support)"
    )


def run(args) -> dict[str, Any]:
    from span_metrics import make_compute_metrics

    records = load_records(args.data, args.limit)
    if not records:
        raise SystemExit(f"no records read from {args.data}")
    logger.info("loaded %d records from %s", len(records), args.data)

    model = load_model(args.model, args.device)
    logger.info("loaded %s from %s on %s", type(model).__name__, args.model, args.device)

    hook = make_compute_metrics(batch_size=args.batch_size, threshold=args.threshold)
    started = time.time()
    flat = hook(model, EvalDataset(records))
    elapsed = time.time() - started
    report = hook.last_report

    print_report(report)
    print(
        f"{len(records)} records | {int(flat['n_mismatched_examples'])} with mismatches | "
        f"{elapsed:.1f} s | threshold {args.threshold} | batch size {args.batch_size} | {args.device}"
    )

    result = {
        "model": str(args.model),
        "data": str(args.data),
        "device": args.device,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "limit": args.limit,
        "n_examples": len(records),
        "elapsed_s": round(elapsed, 1),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": flat,
        "report": report,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="checkpoint dir (config.json + model.safetensors)")
    parser.add_argument("--data", type=Path, required=True, help="GLiNER2 JSONL eval file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    parser.add_argument("--out", type=Path, default=None, help="write full report + flat metrics as JSON")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
