"""Materialise the base encoders/extractors as plain folders under models/ (no Hugging Face cache indirection).

Convention of this repository: every model is a self-contained directory under models/<name>/ and configs refer to
it by that relative path. Training clusters usually have no internet, so download here and rsync models/ over.

  python scripts/download_models.py                    # gliner2.5-multi-v1 + Mursit-Base (+ Mursit-Base-4k)
  python scripts/download_models.py --all              # also the open baselines (PII filter) and Mursit-Large
  python scripts/download_models.py --only gliner2.5-multi-v1
  python scripts/download_models.py --mursit-4k        # (re)build the 4k-context Mursit variant only

Mursit-Base-4k is our context extension of newmindai/Mursit-Base: byte-identical weights, max_position_embeddings
1024 -> 4096 and global_rope_theta 10000 -> 41829.4 (NTK-aware, scale 4, head_dim 64); local RoPE/window untouched.
It is needed because the nm6k schema prompt alone is 1,362 sub-words (2,189 with descriptions).
Set HF_TOKEN in the environment (or .env) only for gated repos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"

REPOS = {
    # name on disk           hub id                                        default?
    "gliner2.5-multi-v1": ("fastino/gliner2.5-multi-v1", True),
    "Mursit-Base": ("newmindai/Mursit-Base", True),
    "Mursit-Large": ("newmindai/Mursit-Large", False),
    "Mursit-Base-TR-Retrieval": ("newmindai/Mursit-Base-TR-Retrieval", False),
    "gliner2-privacy-filter-PII-multi": ("fastino/gliner2-privacy-filter-PII-multi", False),
    # released fine-tuned checkpoints (demo, evaluation recipes): --only gliner2.5-kvkk-tr-v1 ...
    "gliner2.5-kvkk-tr-v1": ("newmindai/gliner2.5-kvkk-tr-v1", False),
    "gliner2.5-kvkk-tr-v2": ("newmindai/gliner2.5-kvkk-tr-v2", False),
    "gliner2.5-mursit-kvkk-tr-v1": ("newmindai/gliner2.5-mursit-kvkk-tr-v1", False),
}
NTK_THETA = {"Mursit-Base": 41829.4, "Mursit-Large": 41829.4}  # 10000 * 4 ** (64 / (64 - 2))


def load_env_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
    return None


def download(name: str, repo: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    target = MODELS / name
    if (target / "config.json").exists() and any(target.glob("*.safetensors")):
        print(f"have   {target}")
        return target
    print(f"fetch  {repo} -> {target}")
    snapshot_download(
        repo,
        local_dir=str(target),
        token=token,
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.jpg", "*.png", "onnx/*"],
    )
    return target


def make_4k(base_name: str, force: bool = False) -> Path:
    """Copy tokenizer + config, patch the context length, symlink the weights (identical bytes)."""
    src = MODELS / base_name
    dst = MODELS / f"{base_name}-4k"
    if not (src / "config.json").exists():
        raise SystemExit(f"{src} is missing — download it first")
    if dst.exists() and not force:
        print(f"have   {dst}")
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["max_position_embeddings"] = 4096
    cfg["global_rope_theta"] = NTK_THETA.get(base_name, 41829.4)
    cfg["_nm_note"] = (
        f"Context extension of newmindai/{base_name}: max_position_embeddings 1024 -> 4096, global_rope_theta "
        f"10000.0 -> {cfg['global_rope_theta']} (NTK-aware, s=4, head_dim=64). local_rope_theta and local_attention=128 "
        "unchanged: local layers never see a position beyond their window. Weights are byte-identical to the base; "
        "the fine-tune adapts to the new theta."
    )
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    weights = dst / "model.safetensors"
    if weights.exists() or weights.is_symlink():
        weights.unlink()
    try:
        weights.symlink_to((src / "model.safetensors").resolve())
    except OSError:
        shutil.copy2(src / "model.safetensors", weights)
    print(f"built  {dst} (weights -> {src.name}/model.safetensors)")
    return dst


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every repo in the table, not only the defaults")
    ap.add_argument("--only", nargs="*", default=None, help="model names to fetch (keys of the table)")
    ap.add_argument("--mursit-4k", action="store_true", help="only (re)build the Mursit-*-4k context-extension folders")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    MODELS.mkdir(exist_ok=True)
    token = load_env_token()
    if args.mursit_4k:
        for base in ("Mursit-Base", "Mursit-Large"):
            if (MODELS / base / "config.json").exists():
                make_4k(base, args.force)
        return
    names = args.only or [n for n, (_, default) in REPOS.items() if default or args.all]
    for name in names:
        if name not in REPOS:
            raise SystemExit(f"unknown model {name!r}; known: {', '.join(REPOS)}")
        download(name, REPOS[name][0], token)
        if name in NTK_THETA:
            make_4k(name, args.force)
    print("\nmodels/ now holds:", ", ".join(sorted(p.name for p in MODELS.iterdir() if p.is_dir())))
    print("Fine-tuned release checkpoints (kvkk-champion, rele07tr, ...) are listed in models/README.md.")


if __name__ == "__main__":
    main()
