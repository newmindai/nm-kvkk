#!/usr/bin/env python3
"""[pkg] Fetch and build the seed data the pipeline needs. Idempotent: existing files are kept.

  briefs  nvidia/Nemotron-PII (CC-BY-4.0) train + test parquet -> briefs/raw/data/, then the two catalogues
          (briefs/README.md has the rule); --check asserts the rebuild equals the shipped parquets column by column.
  geo     berkanumutlu/php-turkiye-il-ilce-adres (MIT) *.sql dumps + ceyyyh/turkiye_mahalleleri (CC0) csv -> geo/raw/,
          then geo/build_geo.py -> geo/*.parquet; --check compares row counts with GEO_EXPECTED.
  model   intfloat/multilingual-e5-large-instruct into the local Hugging Face cache (never an API embedder).

Usage: python download_seeds.py [--briefs] [--geo] [--model] [--check]      (no flag = all three, then check)
Set HF_TOKEN in the environment only if a download is gated.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
BRIEFS_RAW = os.path.join(ROOT, "briefs", "raw", "data")
GEO_RAW = os.path.join(ROOT, "geo", "raw")
COLS = ["uid", "domain", "document_type", "document_description", "document_format"]
GEO_FILES = ["cities.sql", "districts.sql", "neighbourhoods.sql", "streets.sql", "towns.sql"]
GEO_META = {"LICENSE": "LICENSE.berkanumutlu", "README.md": "README.berkanumutlu.md"}
GEO_REPO = "https://raw.githubusercontent.com/berkanumutlu/php-turkiye-il-ilce-adres/main/"
GEO_EXPECTED = {"cities": 81, "towns": 975, "neighbourhoods": 74659, "streets": 1097149}
CATALOGUE_COLUMNS = ["domain", "type_norm", "uid", "document_type", "document_description", "document_format", "n"]


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def build_catalogue(df):
    """One row per (domain, type_norm): uid/document_type from the first row in file order, the most frequent
    description and format, n = group size. Verified equal to the catalogue e07 and e08 were built from."""
    df = df.assign(type_norm=df.document_type.map(norm))
    rows = []
    for (dom, tn), g in df.groupby(["domain", "type_norm"], sort=True):
        first = g.iloc[0]
        rows.append(
            {
                "domain": dom,
                "type_norm": tn,
                "uid": first.uid,
                "document_type": first.document_type,
                "document_description": g.document_description.value_counts().index[0],
                "document_format": g.document_format.value_counts().index[0],
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)


def build_fresh(train_cat, test_cat, used_types):
    """The e08 catalogue: unused train types + test types that are new and unused."""
    a = train_cat[~train_cat.type_norm.isin(used_types)]
    b = test_cat[~test_cat.type_norm.isin(set(train_cat.type_norm)) & ~test_cat.type_norm.isin(used_types)]
    return pd.concat([a, b]).sort_values(["domain", "type_norm"]).reset_index(drop=True)


def fetch_briefs():
    from huggingface_hub import hf_hub_download

    os.makedirs(BRIEFS_RAW, exist_ok=True)
    for split in ("train", "test"):
        target = os.path.join(BRIEFS_RAW, f"{split}-00000-of-00001.parquet")
        if os.path.exists(target):
            print("have", target)
            continue
        p = hf_hub_download(
            "nvidia/Nemotron-PII",
            f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=os.path.join(ROOT, "briefs", "raw"),
            token=os.environ.get("HF_TOKEN"),
        )
        print("downloaded", p)
    rebuild_catalogues(write=not os.path.exists(os.path.join(ROOT, "briefs", "nemotron_catalogue.parquet")))


def rebuild_catalogues(write=False):
    tr = pd.read_parquet(os.path.join(BRIEFS_RAW, "train-00000-of-00001.parquet"), columns=COLS)
    te = pd.read_parquet(os.path.join(BRIEFS_RAW, "test-00000-of-00001.parquet"), columns=COLS)
    used = set(json.load(open(os.path.join(ROOT, "briefs", "e07_types.json"), encoding="utf-8")))
    cat, cat_te = build_catalogue(tr), build_catalogue(te)
    fresh = build_fresh(cat, cat_te, used)
    if write:
        cat.to_parquet(os.path.join(ROOT, "briefs", "nemotron_catalogue.parquet"), index=False)
        fresh.to_parquet(os.path.join(ROOT, "briefs", "nemotron_catalogue_fresh_for_e08.parquet"), index=False)
        print("wrote briefs/nemotron_catalogue.parquet and nemotron_catalogue_fresh_for_e08.parquet")
    return cat, fresh


def check_catalogues(root=ROOT):
    cat, fresh = rebuild_catalogues(write=False)
    for name, built in (("nemotron_catalogue.parquet", cat), ("nemotron_catalogue_fresh_for_e08.parquet", fresh)):
        shipped = pd.read_parquet(os.path.join(root, "briefs", name))
        m = shipped.merge(built, on=["domain", "type_norm"], how="outer", suffixes=("", "_r"), indicator=True)
        assert (m._merge == "both").all(), f"{name}: row sets differ ({(m._merge != 'both').sum()} rows)"
        for c in ("uid", "document_type", "document_description", "document_format", "n"):
            assert (m[c] == m[c + "_r"]).all(), f"{name}: column {c} differs in {(m[c] != m[c + '_r']).sum()} rows"
        print(f"catalogue check: {name} equal on {len(m)} rows")


def fetch_geo():
    os.makedirs(GEO_RAW, exist_ok=True)
    for f in GEO_FILES:
        target = os.path.join(GEO_RAW, f)
        if os.path.exists(target):
            print("have", target)
            continue
        urllib.request.urlretrieve(GEO_REPO + f, target)
        print("downloaded", target)
    for src, dst in GEO_META.items():
        target = os.path.join(GEO_RAW, dst)
        if not os.path.exists(target):
            urllib.request.urlretrieve(GEO_REPO + src, target)
            print("downloaded", target)
    csv = os.path.join(GEO_RAW, "turkiye_mahalleleri.cc0.csv")
    if not os.path.exists(csv):
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [f for f in list_repo_files("ceyyyh/turkiye_mahalleleri", repo_type="dataset") if f.endswith(".csv")]
        p = hf_hub_download(
            "ceyyyh/turkiye_mahalleleri", files[0], repo_type="dataset", token=os.environ.get("HF_TOKEN")
        )
        shutil.copyfile(p, csv)
        print("downloaded", csv, "from", files[0])
    if not all(os.path.exists(os.path.join(ROOT, "geo", f"{t}.parquet")) for t in GEO_EXPECTED):
        subprocess.run([sys.executable, os.path.join(ROOT, "geo", "build_geo.py")], check=True)
    else:
        print("have geo/*.parquet")


def check_geo():
    for t, n in GEO_EXPECTED.items():
        got = len(pd.read_parquet(os.path.join(ROOT, "geo", f"{t}.parquet"), columns=["id"]))
        assert got == n, f"geo {t}: {got} rows, expected {n}"
    print("geo check: row counts as documented")


def fetch_model():
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, os.path.join(ROOT, "pipeline"))
    import config

    SentenceTransformer(config.get("embeddings.model"), token=os.environ.get("HF_TOKEN"))
    print("model cached:", config.get("embeddings.model"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefs", action="store_true")
    ap.add_argument("--geo", action="store_true")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    allf = not (a.briefs or a.geo or a.model or a.check)
    if a.briefs or allf:
        fetch_briefs()
    if a.geo or allf:
        fetch_geo()
    if a.model or allf:
        fetch_model()
    if a.check or allf:
        check_catalogues()
        check_geo()
