"""Write a tiny synthetic Turkish entities + relations set in GLiNER2 record format, for smoke tests only.

  python scripts/make_toy_dataset.py --out datasets/toy            # train.jsonl (32) + test.jsonl (8) + gold_test.json

The records follow every format rule the trainer enforces: every label declared in every record
(empty list = negative), surfaces copied verbatim from the text and whole-word matchable, relations as
{"<type>": {"head": surface, "tail": surface}} plus declared negative types with empty head/tail.
Nothing here is real personal data: names and numbers are generated.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIRST = ["Ayşe", "Mehmet", "Elif", "Can", "Zeynep", "Burak", "Deniz", "Selin", "Emre", "Fatma"]
LAST = ["Yılmaz", "Kaya", "Demir", "Çelik", "Şahin", "Arslan", "Doğan", "Koç", "Aydın", "Öztürk"]
CITIES = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]
LABELS = ["ad soyad", "telefon numarası", "e-posta adresi", "kimlik numarası", "şehir"]
RELATIONS = {
    "telefon numarası sahibi": "Telefon numarası kişiye aittir",
    "e-posta adresi sahibi": "E-posta adresi kişiye aittir",
    "kimlik numarası sahibi": "Kimlik numarası kişiye aittir",
    "ikamet ettiği şehir": "Kişinin yaşadığı şehir",
}


def make_record(rng: random.Random, idx: int):
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    phone = f"05{rng.randint(30, 59)} {rng.randint(100, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}"
    mail = (
        f"{name.split()[0].lower().replace('ş', 's').replace('ı', 'i').replace('ç', 'c')}{rng.randint(1, 99)}@ornek.com"
    )
    tckn = "".join(str(rng.randint(0, 9)) for _ in range(11))
    city = rng.choice(CITIES)
    template = rng.choice(
        [
            "{name} adlı müşterimizin telefon numarası {phone}, e-posta adresi {mail} olarak kayıtlıdır. Kimlik numarası {tckn} olan müşteri {city} ilinde ikamet etmektedir.",
            "Başvuru sahibi {name} ({tckn}) {city} adresinde oturmakta olup kendisine {phone} numarasından ve {mail} adresinden ulaşılabilir.",
            "Sayın {name}, {city} şubemize {phone} telefonu ve {mail} e-postası ile kayıt yaptırmıştır. T.C. kimlik no: {tckn}.",
        ]
    )
    text = template.format(name=name, phone=phone, mail=mail, tckn=tckn, city=city)
    entities = {
        "ad soyad": [name],
        "telefon numarası": [phone],
        "e-posta adresi": [mail],
        "kimlik numarası": [tckn],
        "şehir": [city],
    }
    relations = [
        {"telefon numarası sahibi": {"head": phone, "tail": name}},
        {"e-posta adresi sahibi": {"head": mail, "tail": name}},
        {"kimlik numarası sahibi": {"head": tckn, "tail": name}},
        {"ikamet ettiği şehir": {"head": name, "tail": city}},
    ]
    if idx % 4 == 3:  # every fourth record drops a fact and declares its types as negatives
        text = text.replace(f" {mail}", "").replace(mail, "")
        entities["e-posta adresi"] = []
        relations = [r for r in relations if "e-posta adresi sahibi" not in r] + [
            {"e-posta adresi sahibi": {"head": "", "tail": ""}}
        ]
    return {
        "id": f"toy-{idx}",
        "input": text,
        "output": {"entities": entities, "relations": relations, "relation_descriptions": dict(RELATIONS)},
    }


def gold_from(records):
    out = []
    for r in records:
        triples = []
        for inst in r["output"]["relations"]:
            for rel, pair in inst.items():
                if pair["head"] and pair["tail"]:
                    triples.append(
                        {
                            "relation": rel,
                            "head_mentions": [pair["head"]],
                            "tail_mentions": [pair["tail"]],
                            "head_label": None,
                            "tail_label": None,
                        }
                    )
        out.append({"id": r["id"], "triples": triples})
    return {
        "source": "scripts/make_toy_dataset.py",
        "eval_labels": LABELS,
        "relation_types": sorted(RELATIONS),
        "relation_descriptions": dict(RELATIONS),
        "records": out,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("datasets/toy"))
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train = [make_record(rng, i) for i in range(args.n_train)]
    test = [make_record(rng, args.n_train + i) for i in range(args.n_test)]
    for name, recs in (("train.jsonl", train), ("test.jsonl", test)):
        (args.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    (args.out / "gold_test.json").write_text(
        json.dumps(gold_from(test), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"{args.out}: {len(train)} train / {len(test)} test records, {len(LABELS)} labels, {len(RELATIONS)} relation types"
    )


if __name__ == "__main__":
    main()
