"""Convert the processed parquet datasets (character-offset entities) to GLiNER2 JSONL.

Rules (see gliner2/training/data.py and processor.py for why):
- entities are emitted as label name -> list of unique surface strings sliced
  from the offsets (GLiNER2 has no offset input path);
- every label of the dataset is declared on every row, empty lists for absent
  labels, so negatives (and the abstention head) are supervised;
- rows with no entities are kept;
- text and ``id`` are preserved verbatim.

Usage:
  python scripts/convert_to_gliner_jsonl.py --dataset skb --split validation --sample 2000
  python scripts/convert_to_gliner_jsonl.py --dataset ner --split train
Add ``--descriptions`` to also emit ``entity_descriptions`` (suffix ``_desc``).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LabelMap = Mapping[str, Mapping[str, str]]  # code -> {"name": ..., "description": ...}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "datasets" / "processed"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "configs" / "labels"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets"


def row_to_record(row: Mapping[str, Any], label_map: LabelMap, with_descriptions: bool = False) -> dict[str, Any]:
    text = row["text"]
    names = [label_map[code]["name"] for code in label_map]
    entities: dict[str, list[str]] = {name: [] for name in names}
    seen = {name: set() for name in names}
    for entity in row.get("entities") or []:
        name = label_map[entity["label"]]["name"]  # KeyError on unknown code, on purpose
        surface = text[entity["start"] : entity["end"]]
        if surface.strip() and surface not in seen[name]:
            seen[name].add(surface)
            entities[name].append(surface)
    output: dict[str, Any] = {"entities": entities}
    if with_descriptions:
        output["entity_descriptions"] = {label_map[code]["name"]: label_map[code]["description"] for code in label_map}
    return {"id": row.get("id"), "input": text, "output": output}


def select_indices(total: int, n: int, seed: int) -> list[int]:
    """Deterministic sorted sample of row indices; everything when n <= 0 or n >= total."""
    if n <= 0 or n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


def load_label_map(path: Path) -> dict[str, dict[str, str]]:
    label_map = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [entry["name"] for entry in label_map.values()]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: label names must be unique, got {names}")
    for code, entry in label_map.items():
        if not entry.get("name") or not entry.get("description"):
            raise ValueError(f"{path}: label {code} needs both 'name' and 'description'")
    return label_map


def convert(
    dataset: str,
    split: str,
    sample: int,
    seed: int,
    with_descriptions: bool,
    data_root: Path,
    labels_dir: Path,
    out_dir: Path,
) -> Path:
    import pyarrow.parquet as pq

    label_map = load_label_map(labels_dir / f"labels_{dataset}.json")
    parquet_path = data_root / dataset / f"{split}.parquet"
    parquet = pq.ParquetFile(parquet_path)
    total = parquet.metadata.num_rows
    wanted = select_indices(total, sample, seed)
    wanted_set = set(wanted)

    suffix = (f"_s{sample}" if 0 < sample < total else "") + ("_desc" if with_descriptions else "")
    out_path = out_dir / dataset / f"{split}{suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_label = collections.Counter()
    written = empty = 0
    offset = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for batch in parquet.iter_batches(batch_size=20_000, columns=["id", "text", "entities"]):
            rows = batch.to_pylist()
            for local_index, row in enumerate(rows):
                if offset + local_index in wanted_set:
                    record = row_to_record(row, label_map, with_descriptions)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    n_entities = sum(len(v) for v in record["output"]["entities"].values())
                    empty += n_entities == 0
                    for name, mentions in record["output"]["entities"].items():
                        per_label[name] += len(mentions)
            offset += len(rows)

    print(f"{dataset}/{split}: wrote {written} of {total} rows -> {out_path}")
    print(f"  rows without entities: {empty}")
    print("  unique mentions per label:", dict(per_label.most_common()))
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["ner", "kvkk", "skb"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--sample", type=int, default=0, help="rows to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--descriptions", action="store_true", help="emit entity_descriptions")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    convert(
        args.dataset,
        args.split,
        args.sample,
        args.seed,
        args.descriptions,
        args.data_root,
        args.labels_dir,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
