"""Materialise the published datasets as plain folders under datasets/ (no Hugging Face cache indirection).

  python scripts/download_datasets.py                    # nm-kvkk-pii-6K -> datasets/nm6k/{tr,en}
  python scripts/download_datasets.py --only nm-kvkk-pii-6K --force

The folder layout is the one the recipes expect (datasets/README.md). Training clusters usually have no internet,
so download here and rsync datasets/ over. Set HF_TOKEN in the environment (or .env) only for gated repos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from download_models import load_env_token

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    # name                 hub id                        folder under the repository      a file that proves it is there
    "nm-kvkk-pii-6K": ("newmindai/nm-kvkk-pii-6K", "datasets/nm6k", "tr/train.jsonl"),
}


def download(name: str, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    repo, folder, marker = DATASETS[name]
    target = PROJECT_ROOT / folder
    if (target / marker).exists() and not force:
        print(f"have   {target}")
        return target
    print(f"fetch  {repo} -> {target}")
    snapshot_download(repo, repo_type="dataset", local_dir=str(target), token=load_env_token())
    return target


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="dataset names to fetch (keys of the table)")
    ap.add_argument("--force", action="store_true", help="re-download even if the folder is populated")
    args = ap.parse_args(argv)
    for name in args.only or list(DATASETS):
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
        download(name, args.force)
    print("\nnext: recipes/train_public_nm6k.sh (gliner2.5-multi-v1 -> nm-kvkk-pii-6K)")


if __name__ == "__main__":
    main()
