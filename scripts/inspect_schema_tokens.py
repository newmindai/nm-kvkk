"""How distinct are the schema-token embeddings ([SEP_STRUCT], [SEP_TEXT], [P], [C], [E], [R], [L],
[EXAMPLE], [OUTPUT], [DESCRIPTION]) in a saved extractor? Reads model.safetensors + tokenizer.json
directly (no model construction).

Untrained rows (what resize_token_embeddings leaves) have identical norms, cosine ≈ 1 to the mean
embedding and to each other; gliner2.5-multi-v1's trained rows are distinct (pairwise ≈ 0.8, cosine to
the mean ≈ 0.1).

  python scripts/inspect_schema_tokens.py models/gliner2.5-multi-v1 models/gliner2.5-mursit-kvkk-tr-v1
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

SKIP = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}


def schema_ids(model_dir: Path) -> dict:
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    return {
        t["content"]: t["id"]
        for t in tj.get("added_tokens", [])
        if t.get("special") and t["content"].startswith("[") and t["content"] not in SKIP
    }


def inspect(model_dir: Path) -> dict:
    import torch
    from safetensors import safe_open

    files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors under {model_dir}")
    with safe_open(files[0], "pt") as f:
        key = next(
            k for k in f.keys() if k.endswith(("embeddings.tok_embeddings.weight", "embeddings.word_embeddings.weight"))
        )
        W = f.get_tensor(key).float()
    ids = schema_ids(model_dir)
    base = min(ids.values())
    reg, rows = W[:base], W[list(ids.values())]
    mean = reg.mean(0)
    cos_mean = torch.nn.functional.cosine_similarity(rows, mean.expand_as(rows), dim=1)
    pair = torch.nn.functional.cosine_similarity(rows[0].expand_as(rows[1:]), rows[1:], dim=1)
    return {
        "model": str(model_dir),
        "embedding": list(W.shape),
        "tokens": list(ids),
        "regular_row_norm": round(reg.norm(dim=1).mean().item(), 3),
        "row_norms": [round(x, 3) for x in rows.norm(dim=1).tolist()],
        "cos_to_mean": [round(x, 3) for x in cos_mean.tolist()],
        "cos_first_to_others": [round(x, 3) for x in pair.tolist()],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    for m in args.models:
        r = inspect(m)
        if args.json:
            print(json.dumps(r, ensure_ascii=False))
            continue
        print(f"{r['model']}: embedding {r['embedding']}, {len(r['tokens'])} schema tokens")
        print(f"   norm regular {r['regular_row_norm']} | schema rows {r['row_norms']}")
        print(f"   cos to mean {r['cos_to_mean']}")
        print(f"   cos first→others {r['cos_first_to_others']}")


if __name__ == "__main__":
    main()
