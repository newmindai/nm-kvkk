"""Create eval-set variants for the label-name generalization benchmark.

Renames the entity dict KEYS of a GLiNER2 JSONL eval set; text and mentions
are untouched (mentions are verbatim substrings, so renaming keys never
affects span matching). Three variants:

  seenalias  per-record sampling from the TRAINING alias pools in
             augment_label_names.ALIASES (same rng scheme as training
             augmentation; --p-canonical 0 means never keep the canonical name)
  unseen1    fixed map UNSEEN_V1 (held-out synonyms, disjoint from training)
  unseen2    fixed map UNSEEN_V2 (a second held-out set, disjoint from both)

The held-out maps preserve meaning exactly (no narrowing, no cross-label
collisions) and share no string — case-insensitive — with the canonical names,
any training alias, or each other; check_maps() enforces this on every run.

Usage:
  python scripts/make_eval_variants.py --check
  python scripts/make_eval_variants.py \
      --in datasets/kvkk/pilot/validation_2000.jsonl \
      --out datasets/kvkk/pilot/validation_2000_seenalias.jsonl \
      --variant seenalias --p-canonical 0.0 --seed 7
  python scripts/make_eval_variants.py \
      --in datasets/kvkk/pilot/validation_2000.jsonl \
      --out datasets/kvkk/pilot/validation_2000_unseen1.jsonl --variant unseen1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment_label_names import ALIASES  # noqa: E402

# Held-out synonym maps: wording a Turkish legal/compliance user would type,
# never seen in training (neither canonical nor any ALIASES entry).
UNSEEN_V1 = {
    "adres": "tam adres",
    "doğum tarihi": "doğduğu tarih",
    "kredi kartı numarası": "kart no",
    "coğrafi koordinat": "koordinat bilgisi",
    "araç plakası": "plaka numarası",
    "müşteri numarası": "müşteri ID",
    "sürücü belgesi numarası": "şoför belgesi no",
    "e-posta adresi": "mail adresi",
    "referans kodu": "ref no",
    "IBAN": "uluslararası banka hesap numarası",
    "kimlik numarası": "TCKN",
    "IP adresi": "IP numarası",
    "MAC adresi": "cihaz MAC adresi",
    "pasaport numarası": "pasaport belge numarası",
    "doğum yeri": "doğduğu yer",
    "fotoğraf dosyası": "görsel dosyası",
    "telefon numarası": "tel no",
    "seri numarası": "serial no",
    "web sitesi": "URL",
}

UNSEEN_V2 = {
    "adres": "adres verisi",
    "doğum tarihi": "doğum tarihi verisi",
    "kredi kartı numarası": "ödeme kartı numarası",
    "coğrafi koordinat": "enlem ve boylam",
    "araç plakası": "tescil plakası",
    "müşteri numarası": "müşteri kodu",
    "sürücü belgesi numarası": "sürücü ehliyeti numarası",
    "e-posta adresi": "elektronik posta",
    "referans kodu": "reference number",
    "IBAN": "IBAN bilgisi",
    "kimlik numarası": "vatandaşlık numarası",
    "IP adresi": "internet protokol numarası",
    "MAC adresi": "MAC numarası",
    "pasaport numarası": "pasaport belge no",
    "doğum yeri": "birthplace",
    "fotoğraf dosyası": "image file",
    "telefon numarası": "iletişim numarası",
    "seri numarası": "seri kodu",
    "web sitesi": "web sayfası adresi",
}


def check_maps(verbose: bool = False) -> None:
    """Assert the held-out maps are complete, unique, and disjoint from training."""
    canonical = set(ALIASES)
    seen = {name.casefold() for name in ALIASES}
    seen |= {alias.casefold() for pool in ALIASES.values() for alias in pool}

    for tag, mapping in (("UNSEEN_V1", UNSEEN_V1), ("UNSEEN_V2", UNSEEN_V2)):
        assert set(mapping) == canonical, f"{tag} keys != 19 canonical labels"
        values = [name.casefold() for name in mapping.values()]
        assert len(set(values)) == len(values), f"{tag} has duplicate synonyms"
        clash = set(values) & seen
        assert not clash, f"{tag} overlaps training names: {sorted(clash)}"
    v1 = {name.casefold() for name in UNSEEN_V1.values()}
    v2 = {name.casefold() for name in UNSEEN_V2.values()}
    assert not v1 & v2, f"UNSEEN_V1/V2 overlap: {sorted(v1 & v2)}"

    if verbose:
        pool_size = len(seen)
        print(
            f"no-overlap check OK: 19+19 held-out names vs {pool_size} "
            "training names (canonical + aliases, case-insensitive) — "
            "0 collisions; V1 ∩ V2 = 0"
        )
        for tag, mapping in (("UNSEEN_V1", UNSEEN_V1), ("UNSEEN_V2", UNSEEN_V2)):
            print(f"\n{tag}:")
            for label in ALIASES:
                print(f"  {label:26s} -> {mapping[label]}")


def rename(entities: dict, variant: str, p_canonical: float, seed: int, index: int) -> dict:
    if variant == "unseen1":
        return {UNSEEN_V1[label]: mentions for label, mentions in entities.items()}
    if variant == "unseen2":
        return {UNSEEN_V2[label]: mentions for label, mentions in entities.items()}
    # seenalias: per-record rng, deterministic in (seed, record index)
    rng = random.Random(f"{seed}:{index}")
    renamed = {}
    for label, mentions in entities.items():
        pool = ALIASES[label]
        name = label if rng.random() < p_canonical else rng.choice(pool)
        renamed[name] = mentions
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path)
    parser.add_argument("--out", dest="dst", type=Path)
    parser.add_argument("--variant", choices=["seenalias", "unseen1", "unseen2"])
    parser.add_argument("--p-canonical", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check", action="store_true", help="print the no-overlap check and both maps, then exit")
    args = parser.parse_args()

    check_maps(verbose=args.check)
    if args.check:
        return
    if not (args.src and args.dst and args.variant):
        parser.error("--in, --out and --variant are required (or use --check)")

    n = 0
    with args.src.open(encoding="utf-8") as fin, args.dst.open("w", encoding="utf-8") as fout:
        for index, line in enumerate(fin):
            record = json.loads(line)
            entities = record["output"]["entities"]
            unknown = set(entities) - set(ALIASES)
            assert not unknown, f"record {index}: non-canonical labels {sorted(unknown)}"
            renamed = rename(entities, args.variant, args.p_canonical, args.seed, index)
            assert len(renamed) == len(entities), f"record {index}: name collision"
            record["output"]["entities"] = renamed
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"{n} records -> {args.dst} (variant={args.variant})")


if __name__ == "__main__":
    main()
