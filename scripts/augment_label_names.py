"""Diversify entity label NAMES in GLiNER2 JSONL (text and mentions untouched).

For each record, each declared label is independently either kept canonical
(P_CANONICAL) or replaced by a random alias — consistently across that record's
positives and empty-list negatives. Mentions are verbatim text strings, so
renaming keys never affects span matching. Eval sets stay canonical.

Usage:
  python scripts/augment_label_names.py \
      --in datasets/kvkk/pilot/train_50000.jsonl \
      --out datasets/kvkk/pilot/train_50000_labeldiv.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

P_CANONICAL = 0.5

# Meaning-preserving aliases only: no narrowing (e.g. NOT "cep telefonu" for
# telefon numarası — data contains landlines), no cross-label collisions
# (e.g. NOT "müşteri kimlik numarası" — collides with kimlik numarası).
ALIASES = {
    "adres": ["açık adres", "adres bilgisi", "posta adresi", "address"],
    "doğum tarihi": ["doğum tarihi bilgisi", "dogum tarihi", "date of birth"],
    "kredi kartı numarası": ["kart numarası", "kredi kartı no", "credit card number"],
    "coğrafi koordinat": ["koordinat", "GPS koordinatı", "coğrafi konum bilgisi", "geographic coordinates"],
    "araç plakası": ["plaka", "araç plaka numarası", "taşıt plakası", "license plate"],
    "müşteri numarası": ["müşteri no", "müşteri numarası bilgisi", "customer number"],
    "sürücü belgesi numarası": ["ehliyet numarası", "sürücü belgesi no", "ehliyet no", "driver's license number"],
    "e-posta adresi": ["e-posta", "email adresi", "e-mail", "elektronik posta adresi"],
    "referans kodu": ["referans numarası", "ref kodu", "işlem referans kodu", "reference code"],
    "IBAN": ["IBAN numarası", "iban no", "banka IBAN numarası"],
    "kimlik numarası": ["TC kimlik numarası", "kimlik no", "TC kimlik no", "national ID number"],
    "IP adresi": ["IP", "IP adres bilgisi", "internet protokol adresi", "IP address"],
    "MAC adresi": ["MAC", "MAC adres bilgisi", "MAC address"],
    "pasaport numarası": ["pasaport no", "pasaport numarası bilgisi", "passport number"],
    "doğum yeri": ["doğum yeri bilgisi", "dogum yeri", "place of birth"],
    "fotoğraf dosyası": ["fotoğraf", "foto dosyası", "resim dosyası", "photo file"],
    "telefon numarası": ["telefon no", "telefon numarası bilgisi", "phone number"],
    "seri numarası": ["seri no", "seri numarası bilgisi", "serial number"],
    "web sitesi": ["internet sitesi", "web adresi", "web sitesi adresi", "website"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", dest="dst", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    usage = collections.Counter()
    unknown = set()
    n = 0
    with args.src.open(encoding="utf-8") as fin, args.dst.open("w", encoding="utf-8") as fout:
        for index, line in enumerate(fin):
            record = json.loads(line)
            rng = random.Random((args.seed, index))
            entities = record["output"]["entities"]
            renamed = {}
            for label, mentions in entities.items():
                pool = ALIASES.get(label)
                if pool is None:
                    unknown.add(label)
                    name = label
                elif rng.random() < P_CANONICAL:
                    name = label
                else:
                    name = rng.choice(pool)
                renamed[name] = mentions
                usage[name] += 1
            record["output"]["entities"] = renamed
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"{n} records -> {args.dst}")
    if unknown:
        print("labels with no alias pool (kept canonical):", sorted(unknown))
    print("distinct label names in output:", len(usage))
    for name, count in usage.most_common(12):
        print(f"  {count:6d}  {name}")


if __name__ == "__main__":
    main()
