"""Build a label-stratified pilot subset (GLiNER2 JSONL) from a parquet split.

Selection (deterministic, seeded): keep rows with <= max_words words; rarest label first,
top up rows containing each label to min_per_label; add all-negative rows to min_negatives;
fill uniformly to the target size. Rows are converted with the same rules as
scripts/convert_to_gliner_jsonl.py (label names from the labels JSON, every label declared,
no descriptions) and a <out>.stats.json is written next to the JSONL.

Usage:
  python scripts/make_pilot.py --parquet datasets/ner/v2/final/train.parquet --labels configs/labels/labels_ner.json \\
      --out datasets/ner/pilot/train_50000.jsonl --target 50000 --min-per-label 4000 --min-negatives 1500
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_gliner_jsonl import load_label_map, row_to_record  # noqa: E402


def select_pilot_rows(
    meta: Sequence[Mapping], target: int, min_per_label: int, min_negatives: int, max_words: int, seed: int
) -> list[int]:
    """Return sorted row indices. ``meta[i]`` needs ``n_words`` and ``labels`` (a set of codes)."""
    rng = random.Random(seed)
    eligible = [i for i, m in enumerate(meta) if m["n_words"] <= max_words]
    picked: set[int] = set()
    freq = collections.Counter(lab for i in eligible for lab in meta[i]["labels"])
    for lab in sorted(freq, key=lambda l: (freq[l], l)):  # rarest first
        have = sum(lab in meta[i]["labels"] for i in picked)
        need = max(min_per_label - have, 0)
        pool = [i for i in eligible if i not in picked and lab in meta[i]["labels"]]
        rng.shuffle(pool)
        picked.update(pool[:need])
    have_neg = sum(not meta[i]["labels"] for i in picked)
    pool = [i for i in eligible if i not in picked and not meta[i]["labels"]]
    rng.shuffle(pool)
    picked.update(pool[: max(min_negatives - have_neg, 0)])
    pool = [i for i in eligible if i not in picked]
    rng.shuffle(pool)
    picked.update(pool[: max(target - len(picked), 0)])
    return sorted(picked)[:target] if len(picked) > target else sorted(picked)


def build_pilot(
    parquet: Path,
    labels_json: Path,
    out: Path,
    target: int,
    min_per_label: int,
    min_negatives: int,
    max_words: int,
    seed: int,
) -> Path:
    import pyarrow.parquet as pq

    label_map = load_label_map(Path(labels_json))
    table = pq.read_table(parquet, columns=["id", "text", "entities"])
    texts = table.column("text").to_pylist()
    entities = table.column("entities").to_pylist()
    meta = [
        {"n_words": len(t.split()), "labels": {e["label"] for e in (ents or [])}} for t, ents in zip(texts, entities)
    ]
    chosen = select_pilot_rows(meta, target, min_per_label, min_negatives, max_words, seed)
    ids = table.column("id").to_pylist()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_label = collections.Counter()
    negatives = 0
    words = []
    with out.open("w", encoding="utf-8") as handle:
        for i in chosen:
            record = row_to_record(
                {"id": ids[i], "text": texts[i], "entities": entities[i]}, label_map, with_descriptions=False
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            n = sum(len(v) for v in record["output"]["entities"].values())
            negatives += n == 0
            for name, mentions in record["output"]["entities"].items():
                per_label[name] += len(mentions)
            words.append(meta[i]["n_words"])
    stats = {
        "source": str(parquet),
        "rows": len(chosen),
        "negatives": negatives,
        "max_words": max(words) if words else 0,
        "mean_words": round(sum(words) / len(words), 1) if words else 0,
        "unique_mentions_per_label": dict(per_label.most_common()),
        "params": {
            "target": target,
            "min_per_label": min_per_label,
            "min_negatives": min_negatives,
            "max_words": max_words,
            "seed": seed,
        },
    }
    out.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out}: {stats['rows']} rows, {negatives} negatives, max {stats['max_words']} words")
    print("  unique mentions per label:", stats["unique_mentions_per_label"])
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--min-per-label", type=int, default=0)
    parser.add_argument("--min-negatives", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    build_pilot(
        args.parquet,
        args.labels,
        args.out,
        args.target,
        args.min_per_label,
        args.min_negatives,
        args.max_words,
        args.seed,
    )


if __name__ == "__main__":
    main()
