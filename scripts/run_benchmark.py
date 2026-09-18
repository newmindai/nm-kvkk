"""Cross-model benchmark: run every model through the same pipeline and write one prediction file per model.

  python scripts/run_benchmark.py --set nm6k --models all
  python scripts/run_benchmark.py --set nm6k --models rele07tr,cosmos-pii --limit 50
  python scripts/run_benchmark.py --set vekaletname --models rele07tr --unit doc      # document-level variant

Pipeline: documents -> sentencizer (cached in <run>/sentences.json) -> per unit: model prediction -> spans shifted
to document offsets -> taxonomy ids via the model's mapping / label file -> <run>/predictions/<model>.jsonl (one
line per document, with timing) -> <run>/manifest/. Score with scripts/score_run.py. Models are registered in
configs/benchmark_models.json (kinds: gliner = our checkpoints, hf_token = HF BIO token classifiers).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "GLiNER2"))
from kvkkbench import data as D  # noqa: E402
from kvkkbench import labels as L
from kvkkbench import sentencize as Z

REGISTRY = PROJECT_ROOT / "configs/benchmark_models.json"
RUNS = PROJECT_ROOT / "results/benchmark"

log = logging.getLogger("bench")


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def load_registry() -> dict:
    return {m["name"]: m for m in json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]}


def get_sentences(docs, run: Path):
    cache = run / "sentences.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out = {}
    for d in docs:
        spans, source = Z.split_with_offsets(d["text"])
        out[d["docname"]] = {"spans": spans, "source": source}
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log.info("sentencized %d docs", len(out))
    return out


def make_predictor(spec: dict, subset: str, threshold: float, device: str):
    kind = spec["kind"]
    if kind == "gliner":
        from kvkkbench.gliner import GlinerModel

        names = L.subset_names(spec["vocab"], subset)
        return GlinerModel(spec["path"], names, device=device, threshold=threshold).predict
    if kind == "hf_token":
        from kvkkbench.hf_token import HFTokenClassifier

        return HFTokenClassifier(spec["model"], L.MAPPINGS[spec["mapping"]], device=device).predict
    raise ValueError(f"unknown kind {kind}")


def run_model(spec, docs, sentences, run: Path, subset: str, unit: str, threshold: float, device: str) -> Path:
    name = spec["name"] + ("@doc" if unit == "doc" else "")
    out_path = run / "predictions" / f"{safe(name)}.jsonl"
    predict = make_predictor(spec, subset, threshold, device)
    ids = set(L.SUBSETS[subset])
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for i, d in enumerate(docs):
            text = d["text"]
            units = [(0, len(text))] if unit == "doc" else [tuple(x) for x in sentences[d["docname"]]["spans"]]
            spans, model_s = [], 0.0
            for s, e in units:
                seg = text[s:e]
                if not seg.strip():
                    continue
                try:
                    got, dtm = predict(seg)
                except Exception as exc:  # keep the run alive; the document counts as no predictions for this unit
                    log.warning("%s: %s unit %d-%d failed: %s", name, d["docname"], s, e, exc)
                    got, dtm = [], 0.0
                model_s += dtm
                for sp in got:
                    if sp["label"] in ids:
                        spans.append({**sp, "start": sp["start"] + s, "end": sp["end"] + s})
            fh.write(
                json.dumps(
                    {
                        "docname": d["docname"],
                        "model": name,
                        "spans": sorted(spans, key=lambda x: (x["start"], x["end"])),
                        "timing": {"model_s": round(model_s, 4), "n_units": len(units), "chars": len(text)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 25 == 0:
                log.info("%s: %d/%d docs", name, i + 1, len(docs))
    log.info("%s: done in %.0f s -> %s", name, time.time() - started, out_path.name)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(D.SETS), required=True)
    ap.add_argument("--models", default="all", help="'all' or comma-separated names from configs/benchmark_models.json")
    ap.add_argument("--subset", default="stack21", choices=list(L.SUBSETS))
    ap.add_argument("--unit", default="sentence", choices=["sentence", "doc"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None, help="run directory (default results/benchmark/<set>)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run = args.out or (RUNS / args.set)
    (run / "predictions").mkdir(parents=True, exist_ok=True)
    docs = D.SETS[args.set](limit=args.limit) if args.set in ("nm6k", "mixed_v2") else D.SETS[args.set]()
    if args.limit:
        docs = docs[: args.limit]
    log.info(
        "set %s: %d docs, %d gold spans (%d in %s)",
        args.set,
        len(docs),
        sum(len(d["spans"]) for d in docs),
        sum(1 for d in docs for s in d["spans"] if s["label"] in set(L.SUBSETS[args.subset])),
        args.subset,
    )
    sentences = get_sentences(docs, run)
    (run / "gold.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")

    reg = load_registry()
    names = list(reg) if args.models == "all" else [n.strip() for n in args.models.split(",")]
    specs = [reg[n] for n in names]
    (run / "manifest").mkdir(exist_ok=True)
    (run / "manifest" / "_run.json").write_text(
        json.dumps({"set": args.set, "subset": args.subset, "n_docs": len(docs)}, ensure_ascii=False), encoding="utf-8"
    )
    suffix = "@doc" if args.unit == "doc" else ""
    for spec in specs:
        name = spec["name"] + suffix
        target = run / "predictions" / f"{safe(name)}.jsonl"
        if args.skip_existing and target.exists():
            log.info("skip %s (exists)", name)
            continue
        try:
            path = run_model(spec, docs, sentences, run, args.subset, args.unit, args.threshold, args.device)
        except Exception as exc:
            log.error("%s FAILED: %s", name, exc)
            continue
        entry = {
            "name": name,
            **{k: v for k, v in spec.items() if k != "name"},
            "unit": args.unit,
            "threshold": args.threshold if spec["kind"] == "gliner" else None,
            "file": path.name,
            "finished": dt.datetime.now().isoformat(timespec="seconds"),
            "client": platform.node(),
        }
        (run / "manifest" / f"{safe(name)}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    log.info("manifest entries: %s", run / "manifest")


if __name__ == "__main__":
    main()
