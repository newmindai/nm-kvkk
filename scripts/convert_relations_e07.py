"""Convert the e07 run (the synthetic-v2 dataset) into GLiNER2 RELATION-training train/test sets.

This is the e07 input-normalization path over the VERIFIED v2 converter: the
record format is produced by importing ``build_record`` from
``scripts/convert_relations_train.py`` UNCHANGED (the format that passed the
overfit gate at F1 0.824), so records stay byte-compatible with the v2 build:

  {"input": "<doc text>",
   "output": {"entities":  {turkish_label: [verbatim surfaces]},  # ALL labels declared, [] = negative
              "relations": [{"<type>": {"head": "<surface>", "tail": "<surface>"}}, ...]}}

e07 differences handled HERE (never inside the verified builder):
  - parquet ``entities`` / ``relations_json`` columns are JSON STRINGS -> json.loads;
  - 65 zero-PII negative docs (negatives.jsonl) are folded into TRAIN ONLY as
    pure-negative records: every entity label declared [], NEG_PER_RECORD
    sampled relation types declared {"head": "", "tail": ""} (the verified
    negative rule — build_record with empty entities/relations does exactly this);
  - the label mapping is the 59 v2 entries VERBATIM plus 54 new e07 labels
    (MAPPING_ADDITIONS below, same conventions: canonical KVKK Turkish where it
    exists, else descriptive lowercase Turkish; proper nouns keep their case);
  - relation descriptions: the 57 v2 descriptions verbatim
    (scripts/eval_relations.py RELATION_DESCRIPTIONS) plus 45 new ones phrased
    in the gold head->tail direction observed in the data.

Split (seed 42, deterministic): TEST = 100 accepted docs, stratified over
profile x persons(1 vs 2+) with a fixed multi-person quota of 15 (all 73
multi-person docs are precious); largest-remainder allocation per cell, then
per-cell seeded sampling over sorted bundle ids. TRAIN = remaining 785
accepted + all 65 negatives.

Outputs (datasets/kvkk_relations/e07/):
  train.jsonl                850 records (785 accepted + 65 negatives, no id key)
  test.jsonl                 100 records
  entities_test_e07.jsonl    test docs as {"id","input","output":{"entities":...}}
                             (the --entities file for eval_relations.py / eval_checkpoint.py)
  gold_test_e07.json         eval_relations.py gold schema, 100 test docs
  gold_multiperson_e07.json  ... the multi-person (persons>=2) test subset only
  mapping.json               full label mapping + relation descriptions + notes
  split_report.json          split, strata, conservation accounting, warnings

Validations run on every build: verbatim surfaces, all-labels declared per
record, triple conservation accounting, gold-vs-gold span_metrics == 1.0,
zero exact-text overlap with the frozen v2 pilot (29 docs), unique ids.

Turkish-first relation names (e07tr): --relation-map applies a
{english_type: {"tr": name, "desc_tr": description}} file (keys starting with
"_" are metadata and skipped). Gold files and test.jsonl then carry the
Turkish canonical types ONLY (plus Turkish descriptions); train.jsonl gets the
proven label-diversity recipe mirrored onto relation names: per record, each
distinct declared relation type independently KEEPS THE ENGLISH ORIGINAL with
probability --relation-alias-p (default 0.4) and is renamed to the Turkish
canonical otherwise — seeded per record id, consistent across that record's
positive and negative declarations. Everything upstream (split, negative
sampling, entity side, surfaces) is byte-identical to the plain e07 build.

Usage:
  python scripts/convert_relations_e07.py --src-dir kvkk_synth/runs/<date>-<id>/dataset
  python scripts/convert_relations_e07.py \\
      --relation-map configs/labels/e07_relation_mapping_tr.json \\
      --relation-alias-p 0.4 --out-dir datasets/kvkk_relations/e07tr --tag e07tr
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from convert_relations_train import NEG_PER_RECORD, build_record  # noqa: E402  (verified format)
from eval_relations import RELATION_DESCRIPTIONS as V2_RELATION_DESCRIPTIONS  # noqa: E402
from span_metrics import score_examples  # noqa: E402

# Source = the `dataset/` folder of a kvkk_synth run (tr_pii_relations_e07.parquet + negatives.jsonl).
# Set by --src-dir; the module-level names stay so the conversion functions read them unchanged.
SOURCE_DIR = Path("kvkk_synth/runs/<date>-<id>/dataset")
SOURCE_PARQUET = SOURCE_DIR / "tr_pii_relations_e07.parquet"
SOURCE_NEGATIVES = SOURCE_DIR / "negatives.jsonl"


def set_source(src_dir: Path, parquet_name: str = "tr_pii_relations_e07.parquet") -> None:
    global SOURCE_DIR, SOURCE_PARQUET, SOURCE_NEGATIVES
    SOURCE_DIR = Path(src_dir)
    SOURCE_PARQUET = SOURCE_DIR / parquet_name
    SOURCE_NEGATIVES = SOURCE_DIR / "negatives.jsonl"


V2_GOLD = PROJECT_ROOT / "datasets" / "kvkk_relations" / "v2" / "relations_gold.json"
V2_ENTITIES = PROJECT_ROOT / "datasets" / "kvkk_relations" / "v2" / "entities_eval.jsonl"
SHIPPED_E07_MAPPING = PROJECT_ROOT / "datasets" / "kvkk_relations" / "e07" / "mapping.json"  # tracked in git
OUT_DIR = PROJECT_ROOT / "datasets" / "kvkk_relations" / "e07"

SEED = 42
N_TEST = 100
N_TEST_MULTI = 15  # fixed quota out of the 73 multi-person docs

# --- 54 new e07 entity labels -> Turkish (v2 conventions; the 59 v2 entries are
# reused verbatim from relations_gold.json["entity_label_mapping"]). ---
MAPPING_ADDITIONS: dict[str, str] = {
    "advertising_id": "reklam kimliği",
    "association_foundation_membership": "dernek veya vakıf üyeliği",
    "bar_registry_number": "baro sicil numarası",
    "biometric_data_reference": "biyometrik veri",
    "cargo_tracking_number": "kargo takip numarası",
    "civil_registry_volume_number": "nüfus cilt numarası",
    "company_tax_number": "şirket vergi numarası",  # kept separate from vergi kimlik numarası (person vs company subject)
    "cookie_id": "çerez kimliği",
    "credit_score": "kredi notu",
    "crypto_wallet_address": "kripto cüzdan adresi",
    "device_id": "cihaz kimliği",
    "district": "ilçe",
    "doctor_license_number": "doktor diploma numarası",
    "drivers_license_number": "ehliyet numarası",
    "enforcement_file_number": "icra dosya numarası",
    "eprescription_number": "e-reçete numarası",
    "expert_registry_number": "bilirkişi sicil numarası",
    "family_order_number": "aile sıra numarası",
    "gender": "cinsiyet",
    "id_document_serial": "kimlik seri numarası",
    "independent_section_number": "bağımsız bölüm numarası",
    "individual_order_number": "birey sıra numarası",
    "initials": "isim baş harfleri",
    "insurance_policy_number": "sigorta poliçe numarası",
    "investigation_number": "soruşturma numarası",
    "judge_prosecutor_registry_number": "hakim savcı sicil numarası",
    "land_parcel_number": "ada parsel numarası",
    "loyalty_card_number": "sadakat kartı numarası",
    "maiden_name": "kızlık soyadı",
    "medula_tracking_number": "Medula takip numarası",
    "mersis_number": "MERSİS numarası",
    "middle_name": "ikinci ad",
    "mothers_maiden_name": "anne kızlık soyadı",
    "notary_record_number": "noter yevmiye numarası",
    "notification_barcode": "tebligat barkodu",
    "passport_mrz": "pasaport MRZ kodu",
    "performance_disciplinary": "performans ve disiplin bilgisi",
    "place_of_birth": "doğum yeri",
    "postal_code": "posta kodu",
    "provision_number": "provizyon numarası",
    "registered_place": "nüfusa kayıtlı yer",
    "risk_report": "risk raporu bilgisi",
    "signature_handwriting": "imza veya el yazısı",
    "sim_identifier": "SIM kart kimliği",
    "social_media_handle": "sosyal medya hesabı",
    "social_security_number": "sosyal güvenlik numarası",
    "student_number": "öğrenci numarası",
    "tax_office_name": "vergi dairesi adı",
    "title_honorific": "hitap unvanı",  # kept distinct from iş unvanı (job_title): "Av.", "Op. Dr." style honorifics
    "tracked_location": "takip edilen konum",  # kept distinct from coğrafi koordinat / işyeri konumu
    "trade_registry_number": "ticaret sicil numarası",
    "uets_address": "UETS adresi",
    "voice_record": "ses kaydı",
    "yupass_number": "Yupass numarası",
}

MAPPING_NOTES: dict[str, str] = {
    "company_tax_number": "NOT merged into 'vergi kimlik numarası' (tax_id_number): heads of company_tax_of "
    "are company subjects; merging would erase the person/company distinction.",
    "title_honorific": "NOT 'unvan' to avoid a stretch-collision with 'iş unvanı' (job_title); these are "
    "honorific prefixes such as 'Av.' and 'Op. Dr.'.",
    "tracked_location": "district/city-level tracked positions ('Fatih, İstanbul'); kept apart from "
    "'coğrafi koordinat' (gps_coordinates) and 'işyeri konumu' (workplace_location).",
    "social_security_number": "SGK-style numbers; kept apart from 'kimlik numarası' (national_id_number).",
    "risk_report": "spans are descriptive credit-risk statements, hence 'risk raporu bilgisi' not a bare id name.",
    "medula_tracking_number|mersis_number|uets_address|yupass_number|passport_mrz|sim_identifier": "proper "
    "system names keep their canonical casing (Medula, MERSİS, UETS, Yupass, MRZ, SIM).",
}

# --- 45 new e07 relation types: gold head->tail direction descriptions
# (directions verified against the head/tail label pairs in the data). ---
NEW_RELATION_DESCRIPTIONS: dict[str, str] = {
    "advertising_id_of": "Advertising identifier belongs to the person",
    "bar_registry_of": "Bar association registry number belongs to the person",
    "biometric_of": "Biometric data reference belongs to the person",
    "civil_registry_of": "Civil registry record detail belongs to the person",
    "company_tax_of": "Tax number belongs to the company",
    "cookie_of": "Cookie identifier tracks the person",
    "credit_score_of": "Credit score belongs to the person",
    "crypto_wallet_of": "Crypto wallet address belongs to the person",
    "device_used_by": "Device identifier is used by the person",
    "disciplinary_of": "Performance or disciplinary record belongs to the person",
    "drivers_license_of": "Driver's license number belongs to the person",
    "emergency_contact_of": "First person is the emergency contact of the second person",
    "eprescription_of": "E-prescription number belongs to the person",
    "expert_registry_of": "Court expert registry number belongs to the person",
    "former_surname_of": "Maiden or former surname belongs to the person",
    "gender_of": "Gender of the person",
    "id_serial_of": "Identity document serial number belongs to the person",
    "initials_of": "Name initials refer to the person",
    "insurance_policy_of": "Insurance policy number belongs to the person",
    "judiciary_registry_of": "Judge or prosecutor registry number belongs to the person",
    "located_at": "Tracked location is where the person was located",
    "loyalty_card_of": "Loyalty card number belongs to the person",
    "medula_of": "Medula tracking number belongs to the person",
    "member_of": "Person is a member of the organization",
    "mersis_of": "MERSIS number belongs to the company",
    "mothers_maiden_name_of": "Mother's maiden name belongs to the person",
    "mrz_of": "Passport MRZ code belongs to the person",
    "notary_act_of": "Notary record number belongs to the person",
    "notified_via": "Notification barcode is used to notify the person",
    "part_of_name": "Name component or honorific is part of the person's full name",
    "place_of_birth_of": "Place of birth of the person",
    "prescribed_by": "Doctor license number belongs to the prescribing doctor",
    "property_of": "Real estate parcel or unit number is owned by the person",
    "provision_of": "Provision number belongs to the person",
    "risk_report_of": "Credit risk report information belongs to the person",
    "shipment_of": "Cargo tracking number belongs to the person's shipment",
    "signature_of": "Signature or handwriting belongs to the person",
    "sim_of": "SIM card identifier belongs to the person",
    "social_security_of": "Social security number belongs to the person",
    "student_number_of": "Student number belongs to the person",
    "tax_office_of": "Tax office where the person is registered",
    "trade_registry_of": "Trade registry number belongs to the company",
    "uets_of": "UETS address belongs to the person",
    "voice_of": "Voice recording belongs to the person",
    "yupass_of": "Yupass number belongs to the person",
}


def largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Deterministic largest-remainder apportionment of `total` over cells."""
    pool = sum(sizes.values())
    quotas = {c: sizes[c] * total / pool for c in sizes}
    alloc = {c: int(quotas[c]) for c in sizes}
    for c in sorted(sizes, key=lambda c: (-(quotas[c] - alloc[c]), c)):
        if sum(alloc.values()) == total:
            break
        alloc[c] += 1
    assert sum(alloc.values()) == total
    assert all(alloc[c] <= sizes[c] for c in sizes)
    return alloc


def load_relation_map(path: Path) -> dict[str, dict[str, str]]:
    """{english_type: {"tr": ..., "desc_tr": ...}}; "_"-prefixed keys skipped."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rel_map = {k: v for k, v in raw.items() if not k.startswith("_")}
    for eng, entry in rel_map.items():
        assert entry.get("tr") and entry.get("desc_tr"), f"incomplete entry for {eng}"
    return rel_map


def convert_frozen(
    frozen_dir: Path, relation_map_path: Path, relation_alias_p: float, out_dir: Path, tag: str, frozen_tag: str = "e07"
) -> None:
    """Derive a Turkish-first build by RENAMING the frozen e07 outputs in place.

    Use this (not the parquet path) when the upstream parquet has drifted since
    the frozen build: texts, entity declarations, surfaces, split, and negative
    sampling stay byte-identical to <frozen_dir>; ONLY relation-type names (and
    gold relation types/descriptions) change. Train gets the labeldiv recipe —
    per record, each distinct declared relation type independently keeps the
    ENGLISH original with P=relation_alias_p, else the Turkish canonical; the
    rng is seeded per (SEED, record index), exactly the augment_label_names.py
    scheme, because frozen train.jsonl carries no ids. Test/gold are Turkish
    canonical only.
    """
    relation_map = load_relation_map(relation_map_path)
    frozen_mapping = json.loads((frozen_dir / "mapping.json").read_text(encoding="utf-8"))
    all_relation_types: list[str] = frozen_mapping["relation_types"]
    all_entity_labels: list[str] = frozen_mapping["eval_labels"]
    missing_map = sorted(t for t in all_relation_types if t not in relation_map)
    assert not missing_map, f"relation types without Turkish mapping: {missing_map}"
    tr_of = {t: relation_map[t]["tr"] for t in all_relation_types}
    tr_names = [tr_of[t] for t in all_relation_types]
    assert len(set(tr_names)) == len(tr_names), "duplicate Turkish relation names"
    assert not set(tr_names) & set(all_entity_labels), "relation name collides with entity label"
    assert not set(tr_names) & set(all_relation_types), "relation name collides with English type"
    desc_tr = {tr_of[t]: relation_map[t]["desc_tr"] for t in all_relation_types}
    allowed_names = set(all_relation_types) | set(tr_names)

    def rename_record(record: dict[str, Any], rng: random.Random | None) -> None:
        """rng None -> force Turkish; else labeldiv flip per distinct type."""
        rels = record["output"]["relations"]
        present = sorted({next(iter(item)) for item in rels})
        choice = {}
        for t in present:  # sorted -> rng draws independent of list order
            assert t in tr_of, f"unknown relation type in frozen file: {t}"
            keep_english = rng is not None and rng.random() < relation_alias_p
            choice[t] = t if keep_english else tr_of[t]
        record["output"]["relations"] = [{choice[next(iter(item))]: next(iter(item.values()))} for item in rels]
        for item in record["output"]["relations"]:  # well-formedness after rename
            ((rtype, args),) = item.items()
            assert rtype in allowed_names and set(args) == {"head", "tail"}
            if args["head"]:
                assert args["head"] in record["input"] and args["tail"] in record["input"]
            else:
                assert args["head"] == "" and args["tail"] == ""

    out_dir.mkdir(parents=True, exist_ok=True)
    rel_name_stats = {
        "train_pos_turkish": 0,
        "train_pos_english": 0,
        "train_neg_turkish": 0,
        "train_neg_english": 0,
        "train_records_all_turkish": 0,
        "train_records_mixed": 0,
        "train_records_all_english": 0,
    }
    with (
        (frozen_dir / "train.jsonl").open(encoding="utf-8") as fin,
        (out_dir / "train.jsonl").open("w", encoding="utf-8") as fout,
    ):
        n_train = 0
        for index, line in enumerate(fin):
            record = json.loads(line)
            rename_record(record, random.Random(f"reldiv-{SEED}-idx-{index}"))
            langs = set()
            for item in record["output"]["relations"]:
                ((name, args),) = item.items()
                lang = "english" if name in set(all_relation_types) else "turkish"
                langs.add(lang)
                kind = "pos" if args["head"] else "neg"
                rel_name_stats[f"train_{kind}_{lang}"] += 1
            key = (
                "train_records_all_turkish"
                if langs == {"turkish"}
                else "train_records_all_english"
                if langs == {"english"}
                else "train_records_mixed"
            )
            rel_name_stats[key] += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_train += 1
    with (
        (frozen_dir / "test.jsonl").open(encoding="utf-8") as fin,
        (out_dir / "test.jsonl").open("w", encoding="utf-8") as fout,
    ):
        n_test = 0
        for line in fin:
            record = json.loads(line)
            rename_record(record, rng=None)  # Turkish canonical only
            for item in record["output"]["relations"]:
                assert next(iter(item)) in desc_tr
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_test += 1

    # entity side: byte-for-byte the frozen entities file
    entities_src = frozen_dir / f"entities_test_{frozen_tag}.jsonl"
    (out_dir / f"entities_test_{tag}.jsonl").write_bytes(entities_src.read_bytes())

    n_gold = {}
    for kind in ("gold_test", "gold_multiperson"):
        gold = json.loads((frozen_dir / f"{kind}_{frozen_tag}.json").read_text(encoding="utf-8"))
        assert gold["relation_types"] == all_relation_types
        gold["relation_types"] = sorted(tr_names)
        gold["relation_descriptions"] = {n: desc_tr[n] for n in sorted(tr_names)}
        gold["relation_types_english"] = list(all_relation_types)
        gold["relation_name_mapping"] = dict(tr_of)
        for r in gold["records"]:
            for triple in r["triples"]:
                triple["relation_english"] = triple["relation"]
                triple["relation"] = tr_of[triple["relation"]]
        n_gold[kind] = sum(len(r["triples"]) for r in gold["records"])
        (out_dir / f"{kind}_{tag}.json").write_text(
            json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    mapping = dict(frozen_mapping)
    mapping["frozen_source"] = str(frozen_dir)
    mapping["relation_mapping_tr"] = {t: relation_map[t] for t in all_relation_types}
    mapping["relation_types_turkish"] = sorted(tr_names)
    mapping["relation_alias_p"] = relation_alias_p
    mapping["relation_naming_convention"] = (
        "her ilişki adı gold yönünün TAIL argümanını adlandıran isim tamlamasıdır; "
        "varsayılan şablon '<head özelliği> sahibi', sahiplik dışı bağlarda "
        "'<rolü> olduğu <tail türü>' / '<head> kullanıcısı' / '<head> muhatabı' / "
        "sıfat-fiil + kişi. Train: tip başına kayıt içinde tutarlı, seed'li yazı tura "
        f"(P(İngilizce alias)={relation_alias_p}); test/gold: yalnız Türkçe kanonik."
    )
    (out_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = json.loads((frozen_dir / "split_report.json").read_text(encoding="utf-8"))
    report["frozen_source"] = str(frozen_dir)
    report["relation_name_distribution"] = dict(
        rel_name_stats,
        alias_p=relation_alias_p,
        map=str(relation_map_path),
        rng_scheme=f"reldiv-{SEED}-idx-<record index>",
    )
    (out_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    s = rel_name_stats
    n_tr = s["train_pos_turkish"] + s["train_neg_turkish"]
    n_en = s["train_pos_english"] + s["train_neg_english"]
    print(f"frozen rename: {frozen_dir} -> {out_dir}")
    print(f"train.jsonl: {n_train} records | test.jsonl: {n_test} records (texts/entities frozen)")
    print(
        f"relation names (train): {n_tr} Turkish / {n_en} English declarations "
        f"(pos {s['train_pos_turkish']}/{s['train_pos_english']}, "
        f"neg {s['train_neg_turkish']}/{s['train_neg_english']}; alias_p={relation_alias_p}) | "
        f"records: {s['train_records_all_turkish']} all-TR, {s['train_records_mixed']} mixed, "
        f"{s['train_records_all_english']} all-EN | test+gold: Turkish canonical only"
    )
    print(
        f"gold: test {n_gold['gold_test']} triples, multi-person {n_gold['gold_multiperson']} triples "
        f"(Turkish canonical types + Turkish descriptions)"
    )


def convert(
    relation_map_path: Path | None = None, relation_alias_p: float = 0.4, out_dir: Path = OUT_DIR, tag: str = "e07"
) -> None:
    import pandas as pd

    # Base entity-label mapping: the 59 verified v2-pilot entries. Read from the (private) v2 gold when present,
    # otherwise from the shipped e07 mapping (identical entries; the e07 additions are removed and re-applied below).
    if V2_GOLD.exists():
        base_mapping: dict[str, str] = dict(json.loads(V2_GOLD.read_text(encoding="utf-8"))["entity_label_mapping"])
    elif SHIPPED_E07_MAPPING.exists():
        shipped = json.loads(SHIPPED_E07_MAPPING.read_text(encoding="utf-8"))["entity_label_mapping"]
        base_mapping = {k: v for k, v in shipped.items() if k not in MAPPING_ADDITIONS}
        print(f"v2 pilot gold not present; base mapping taken from {SHIPPED_E07_MAPPING} ({len(base_mapping)} entries)")
    else:
        raise SystemExit(f"need {V2_GOLD} or {SHIPPED_E07_MAPPING} for the base entity-label mapping")
    v2_gold = {"entity_label_mapping": base_mapping}
    label_map: dict[str, str] = dict(base_mapping)
    overlap_keys = set(label_map) & set(MAPPING_ADDITIONS)
    assert not overlap_keys, f"additions collide with v2 keys: {overlap_keys}"
    label_map.update(MAPPING_ADDITIONS)
    turkish_collisions = collections.Counter(label_map.values())
    v2_turkish = set(base_mapping.values())
    for eng, tr in MAPPING_ADDITIONS.items():
        assert tr not in v2_turkish, f"new label {eng} collides with existing Turkish name {tr}"
    relation_descriptions = dict(V2_RELATION_DESCRIPTIONS)
    assert not set(relation_descriptions) & set(NEW_RELATION_DESCRIPTIONS)
    relation_descriptions.update(NEW_RELATION_DESCRIPTIONS)

    df = pd.read_parquet(SOURCE_PARQUET).sort_values("bundle_id").reset_index(drop=True)
    assert bool((~df["negative"]).all()) and df["bundle_id"].is_unique

    # --- inventory: every label/type present must be covered ---
    docs: list[dict[str, Any]] = []
    label_counts: collections.Counter = collections.Counter()
    type_counts: collections.Counter = collections.Counter()
    for _, row in df.iterrows():
        entities = json.loads(row["entities"])  # e07: JSON-string column
        relations = json.loads(row["relations_json"])  # e07: JSON-string column
        for e in entities:
            label_counts[e["label"]] += 1
        for r in relations:
            type_counts[r["relation"]] += 1
        docs.append(
            {
                "bundle": int(row["bundle_id"]),
                "text": row["text"],
                "profile": row["profile"],
                "persons": int(row["persons"]),
                "entities": entities,
                "relations": relations,
            }
        )
    unmapped = sorted(l for l in label_counts if l not in label_map)
    undescribed = sorted(t for t in type_counts if t not in relation_descriptions)
    assert not unmapped, f"labels without mapping: {unmapped}"
    assert not undescribed, f"relation types without description: {undescribed}"

    all_entity_labels = sorted({label_map[l] for l in label_counts})
    all_relation_types = sorted(type_counts)

    negatives: list[dict[str, Any]] = []
    with SOURCE_NEGATIVES.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            assert rec["negative"] is True and not rec["entities"] and not rec["relations"]
            negatives.append({"bundle": int(rec["bundle_id"]), "text": rec["text"]})

    # --- benchmark isolation: zero exact-text overlap with the frozen v2 pilot ---
    e07_texts = {d["text"] for d in docs} | {n["text"] for n in negatives}
    if V2_ENTITIES.exists():
        v2_texts = {json.loads(l)["input"] for l in V2_ENTITIES.open(encoding="utf-8") if l.strip()}
        leak = v2_texts & e07_texts
        assert not leak, f"e07 shares {len(leak)} exact texts with the v2 benchmark"
    else:
        print("v2 pilot entities not present; skipping the exact-text overlap check against it")

    # --- stratified split: profile x persons(1 vs 2+), fixed multi-person quota ---
    cells: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for d in docs:
        cells[(d["profile"], "2+" if d["persons"] >= 2 else "1")].append(d["bundle"])
    multi_sizes = {p: len(cells[(p, "2+")]) for p in sorted({k[0] for k in cells}) if cells.get((p, "2+"))}
    single_sizes = {p: len(cells[(p, "1")]) for p in sorted({k[0] for k in cells}) if cells.get((p, "1"))}
    multi_alloc = largest_remainder(multi_sizes, N_TEST_MULTI)
    single_alloc = largest_remainder(single_sizes, N_TEST - N_TEST_MULTI)
    test_bundles: set = set()
    cell_report = {}
    for bucket, alloc in (("2+", multi_alloc), ("1", single_alloc)):
        for profile, k in alloc.items():
            pool = sorted(cells[(profile, bucket)])
            rng = random.Random(f"split-{SEED}-e07-{profile}-{bucket}")
            chosen = sorted(rng.sample(pool, k))
            test_bundles.update(chosen)
            cell_report[f"{profile}|persons={bucket}"] = {"pool": len(pool), "test": k, "train": len(pool) - k}
    assert len(test_bundles) == N_TEST

    # --- convert every doc through the VERIFIED builder ---
    warnings: list[dict[str, Any]] = []
    per_record_stats: dict[str, dict[str, int]] = {}
    train_records, test_records = [], []
    doc_meta: dict[str, dict[str, Any]] = {}
    for d in docs:
        row_id = f"kvkk_relations/e07/bundle_{d['bundle']}"
        rng = random.Random(f"neg-{SEED}-e07-{d['bundle']}")  # deterministic per record
        record, stats = build_record(
            row_id,
            d["text"],
            d["entities"],
            d["relations"],
            label_map,
            all_entity_labels,
            all_relation_types,
            rng,
            warnings,
        )
        per_record_stats[row_id] = stats
        doc_meta[row_id] = d
        (test_records if d["bundle"] in test_bundles else train_records).append(record)
    n_neg_records = 0
    for n in negatives:  # TRAIN ONLY: pure-negative records via the same builder
        row_id = f"kvkk_relations/e07/neg_{n['bundle']}"
        rng = random.Random(f"neg-{SEED}-e07-neg-{n['bundle']}")
        record, stats = build_record(
            row_id, n["text"], [], [], label_map, all_entity_labels, all_relation_types, rng, warnings
        )
        assert stats == {"triples": 0, "triples_covered": 0, "instances": 0, "negatives": NEG_PER_RECORD}, (
            row_id,
            stats,
        )
        per_record_stats[row_id] = stats
        train_records.append(record)
        n_neg_records += 1
    all_ids = [r["id"] for r in train_records + test_records]
    assert len(all_ids) == len(set(all_ids)), "duplicate record ids"

    # --- validation: verified-format invariants on every record ---
    for rec in train_records + test_records:
        assert set(rec["output"]["entities"]) == set(all_entity_labels), rec["id"]
        for label, surfaces in rec["output"]["entities"].items():
            for s in surfaces:
                assert s in rec["input"], (rec["id"], label, s)
        for item in rec["output"]["relations"]:
            ((rtype, args),) = item.items()
            assert rtype in all_relation_types, rtype
            if args["head"]:
                assert args["head"] in rec["input"] and args["tail"] in rec["input"], rec["id"]

    # --- conservation accounting ---
    n_source_triples = sum(len(d["relations"]) for d in docs)
    n_covered = sum(s["triples_covered"] for s in per_record_stats.values())
    n_mentions = sum(len(d["entities"]) for d in docs)
    assert n_source_triples == sum(type_counts.values())
    dropped_triples = [w for w in warnings if w["kind"] == "triple_unmatchable"]
    dropped_surfaces = [w for w in warnings if w["kind"] == "entity_surface_unmatchable"]
    assert n_covered == n_source_triples - len(dropped_triples)

    # --- gold files (eval_relations.py schema; triples keep ALL raw triples) ---
    def gold_record(row_id: str) -> dict[str, Any]:
        d = doc_meta[row_id]
        id2label: dict[str, str] = {}
        clusters: dict[str, list[str]] = collections.defaultdict(list)
        for e in d["entities"]:
            id2label.setdefault(e["id"], e["label"])
            surface = d["text"][e["start"] : e["end"]]
            if surface.strip() and surface not in clusters[e["id"]]:
                clusters[e["id"]].append(surface)
        triples = []
        for r in d["relations"]:
            triples.append(
                {
                    "relation": r["relation"],
                    "head_label": label_map[id2label[r["head"]]],
                    "head_label_raw": id2label[r["head"]],
                    "head_mentions": clusters[r["head"]],
                    "tail_label": label_map[id2label[r["tail"]]],
                    "tail_label_raw": id2label[r["tail"]],
                    "tail_mentions": clusters[r["tail"]],
                }
            )
        return {"id": row_id, "triples": triples}

    test_ids = [r["id"] for r in test_records]
    multi_test_ids = [i for i in test_ids if doc_meta[i]["persons"] >= 2]
    gold_common = {
        "source": str(SOURCE_PARQUET),
        "entity_label_mapping": label_map,
        "eval_labels": all_entity_labels,
        "relation_types": all_relation_types,
        "relation_descriptions": {t: relation_descriptions[t] for t in all_relation_types},
    }
    gold_test = dict(gold_common, records=[gold_record(i) for i in test_ids])
    gold_multi = dict(gold_common, records=[gold_record(i) for i in multi_test_ids])
    n_gold_triples = sum(len(r["triples"]) for r in gold_test["records"])
    n_test_source = sum(len(doc_meta[i]["relations"]) for i in test_ids)
    assert n_gold_triples == n_test_source  # gold conserves every raw test triple

    # --- gold-vs-gold entity sanity: span_metrics must be exactly 1.0 ---
    gold_entities = {r["id"]: r["output"]["entities"] for r in test_records}
    examples = [(gold_entities[i], gold_entities[i]) for i in test_ids]
    sanity = score_examples(examples, all_entity_labels)
    assert sanity["micro"]["f1"] == 1.0 and sanity["macro"]["f1"] == 1.0, sanity["micro"]

    # --- Turkish-first relation names (--relation-map) ---
    relation_map: dict[str, dict[str, str]] | None = None
    rel_name_stats: dict[str, int] | None = None
    if relation_map_path is not None:
        relation_map = load_relation_map(relation_map_path)
        missing_map = sorted(t for t in all_relation_types if t not in relation_map)
        assert not missing_map, f"relation types without Turkish mapping: {missing_map}"
        tr_of = {t: relation_map[t]["tr"] for t in all_relation_types}
        tr_names = [tr_of[t] for t in all_relation_types]
        assert len(set(tr_names)) == len(tr_names), "duplicate Turkish relation names"
        assert not set(tr_names) & set(all_entity_labels), "relation name collides with entity label"
        assert not set(tr_names) & set(all_relation_types), "relation name collides with English type"
        desc_tr = {tr_of[t]: relation_map[t]["desc_tr"] for t in all_relation_types}

        def rename_relations(record: dict[str, Any], force_turkish: bool) -> dict[str, str]:
            """Per-record type->name choice (proven labeldiv recipe, seeded)."""
            rels = record["output"]["relations"]
            present = sorted({next(iter(item)) for item in rels})
            rng = random.Random(f"reldiv-{SEED}-{record['id']}")
            choice = {}
            for t in present:  # sorted -> rng draws independent of list order
                keep_english = (not force_turkish) and rng.random() < relation_alias_p
                choice[t] = t if keep_english else tr_of[t]
            record["output"]["relations"] = [{choice[next(iter(item))]: next(iter(item.values()))} for item in rels]
            return choice

        rel_name_stats = {
            "train_pos_turkish": 0,
            "train_pos_english": 0,
            "train_neg_turkish": 0,
            "train_neg_english": 0,
            "train_records_all_turkish": 0,
            "train_records_mixed": 0,
            "train_records_all_english": 0,
        }
        for rec in train_records:  # TRAIN: diversity flip per record x type
            rename_relations(rec, force_turkish=False)
            langs = set()
            for item in rec["output"]["relations"]:
                ((name, args),) = item.items()
                lang = "english" if name in set(all_relation_types) else "turkish"
                langs.add(lang)
                kind = "pos" if args["head"] else "neg"
                rel_name_stats[f"train_{kind}_{lang}"] += 1
            key = (
                "train_records_all_turkish"
                if langs == {"turkish"}
                else "train_records_all_english"
                if langs == {"english"}
                else "train_records_mixed"
            )
            rel_name_stats[key] += 1
        for rec in test_records:  # TEST: Turkish canonical only (mirrors gold)
            rename_relations(rec, force_turkish=True)
            for item in rec["output"]["relations"]:
                ((name, _),) = item.items()
                assert name in desc_tr, f"non-Turkish type in test record: {name}"
        allowed_names = set(all_relation_types) | set(tr_names)
        for rec in train_records + test_records:  # well-formedness after rename
            for item in rec["output"]["relations"]:
                ((rtype, args),) = item.items()
                assert rtype in allowed_names, rtype
                if args["head"]:
                    assert args["head"] in rec["input"] and args["tail"] in rec["input"], rec["id"]
                else:
                    assert args["head"] == "" and args["tail"] == "", rec["id"]

        for g in (gold_test, gold_multi):  # gold: Turkish canonical ONLY + Turkish descs
            g["relation_types"] = sorted(tr_names)
            g["relation_descriptions"] = {n: desc_tr[n] for n in sorted(tr_names)}
            g["relation_types_english"] = list(all_relation_types)
            g["relation_name_mapping"] = dict(tr_of)
            for r in g["records"]:
                for triple in r["triples"]:
                    triple["relation_english"] = triple["relation"]
                    triple["relation"] = tr_of[triple["relation"]]

    # --- write outputs ---
    OUT_DIR = out_dir  # noqa: N806 (shadow module default with the CLI choice)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def dump_jsonl(path: Path, records: list[dict[str, Any]], keep_id: bool = False) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                if keep_id:
                    payload = {"id": rec["id"], "input": rec["input"], "output": rec["output"]}
                else:
                    payload = {"input": rec["input"], "output": rec["output"]}
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    dump_jsonl(OUT_DIR / "train.jsonl", train_records)
    dump_jsonl(OUT_DIR / "test.jsonl", test_records)
    entities_test = [
        {"id": r["id"], "input": r["input"], "output": {"entities": r["output"]["entities"]}} for r in test_records
    ]
    with (OUT_DIR / f"entities_test_{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for rec in entities_test:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for name, payload in ((f"gold_test_{tag}.json", gold_test), (f"gold_multiperson_{tag}.json", gold_multi)):
        (OUT_DIR / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mapping = {
        "source": str(SOURCE_PARQUET),
        "negatives_source": str(SOURCE_NEGATIVES),
        "reused_v2_mappings": len(v2_gold["entity_label_mapping"]),
        "new_mappings": len(MAPPING_ADDITIONS),
        "entity_label_mapping": label_map,
        "entity_label_mapping_new_keys": sorted(MAPPING_ADDITIONS),
        "mapping_notes": MAPPING_NOTES,
        "eval_labels": all_entity_labels,
        "relation_types": all_relation_types,
        "relation_descriptions": {t: relation_descriptions[t] for t in all_relation_types},
        "relation_descriptions_new_keys": sorted(set(NEW_RELATION_DESCRIPTIONS) & set(all_relation_types)),
        "label_counts": dict(sorted(label_counts.items())),
        "relation_type_counts": dict(sorted(type_counts.items())),
        "turkish_name_merges": {
            tr: [e for e, t in label_map.items() if t == tr] for tr, c in sorted(turkish_collisions.items()) if c > 1
        },
    }
    if relation_map is not None:
        mapping["relation_mapping_tr"] = {t: relation_map[t] for t in all_relation_types}
        mapping["relation_types_turkish"] = sorted(tr_names)
        mapping["relation_alias_p"] = relation_alias_p
        mapping["relation_naming_convention"] = (
            "her ilişki adı gold yönünün TAIL argümanını adlandıran isim tamlamasıdır; "
            "varsayılan şablon '<head özelliği> sahibi', sahiplik dışı bağlarda "
            "'<rolü> olduğu <tail türü>' / '<head> kullanıcısı' / '<head> muhatabı' / "
            "sıfat-fiil + kişi. Train: tip başına kayıt içinde tutarlı, seed'li yazı tura "
            f"(P(İngilizce alias)={relation_alias_p}); test/gold: yalnız Türkçe kanonik."
        )
    (OUT_DIR / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_ids = [r["id"] for r in train_records]
    multi_train = [i for i in train_ids if i in doc_meta and doc_meta[i]["persons"] >= 2]
    report = {
        "seed": SEED,
        "counts": {
            "accepted_docs": len(docs),
            "negative_docs": n_neg_records,
            "train_records": len(train_records),
            "test_records": len(test_records),
            "train_accepted": len(train_records) - n_neg_records,
            "multi_person_total": len(multi_train) + len(multi_test_ids),
            "multi_person_test": len(multi_test_ids),
            "multi_person_train": len(multi_train),
            "entity_labels": len(all_entity_labels),
            "relation_types": len(all_relation_types),
            "source_mentions": n_mentions,
            "source_triples": n_source_triples,
        },
        "strata": cell_report,
        "test_ids": test_ids,
        "multi_person_test_ids": multi_test_ids,
        "triples": {
            "source": n_source_triples,
            "covered": n_covered,
            "dropped_unmatchable": len(dropped_triples),
            "train_instances": sum(per_record_stats[i]["instances"] for i in train_ids),
            "test_instances": sum(per_record_stats[i]["instances"] for i in test_ids),
            "negative_instances_train": sum(per_record_stats[i]["negatives"] for i in train_ids),
            "negative_instances_test": sum(per_record_stats[i]["negatives"] for i in test_ids),
            "gold_test_triples": n_gold_triples,
            "gold_multiperson_triples": sum(len(r["triples"]) for r in gold_multi["records"]),
        },
        "entity_surfaces_dropped_unmatchable": len(dropped_surfaces),
        "v2_overlap_texts": 0,
        "gold_vs_gold_span_metrics": {"micro_f1": sanity["micro"]["f1"], "macro_f1": sanity["macro"]["f1"]},
        "warnings": warnings,
    }
    if rel_name_stats is not None:
        report["relation_name_distribution"] = dict(
            rel_name_stats, alias_p=relation_alias_p, map=str(relation_map_path)
        )
    (OUT_DIR / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    c, t = report["counts"], report["triples"]
    print(
        f"train.jsonl: {c['train_records']} records ({c['train_accepted']} accepted + {c['negative_docs']} negatives)"
    )
    print(
        f"test.jsonl:  {c['test_records']} records | multi-person: {c['multi_person_test']} test / {c['multi_person_train']} train (of {c['multi_person_total']})"
    )
    print(
        f"labels: {c['entity_labels']} Turkish (from {len(label_counts)} English) | relation types: {c['relation_types']}"
    )
    print(f"triples: {t['source']} source, {t['covered']} covered, {t['dropped_unmatchable']} dropped (unmatchable)")
    print(
        f"instances: train {t['train_instances']} (+{t['negative_instances_train']} negatives), "
        f"test {t['test_instances']} (+{t['negative_instances_test']} negatives)"
    )
    print(f"gold: test {t['gold_test_triples']} triples, multi-person {t['gold_multiperson_triples']} triples")
    print(
        f"entity surfaces dropped: {report['entity_surfaces_dropped_unmatchable']} | "
        f"gold-vs-gold span F1 micro={sanity['micro']['f1']:.3f} macro={sanity['macro']['f1']:.3f}"
    )
    print(f"strata: {json.dumps(cell_report)}")
    if rel_name_stats is not None:
        s = rel_name_stats
        n_tr = s["train_pos_turkish"] + s["train_neg_turkish"]
        n_en = s["train_pos_english"] + s["train_neg_english"]
        print(
            f"relation names (train): {n_tr} Turkish / {n_en} English declarations "
            f"(pos {s['train_pos_turkish']}/{s['train_pos_english']}, "
            f"neg {s['train_neg_turkish']}/{s['train_neg_english']}; alias_p={relation_alias_p}) | "
            f"records: {s['train_records_all_turkish']} all-TR, {s['train_records_mixed']} mixed, "
            f"{s['train_records_all_english']} all-EN | test+gold: Turkish canonical only"
        )
    print(f"warnings: {len(warnings)}")
    for w in warnings[:40]:
        print("  WARN", json.dumps(w, ensure_ascii=False)[:200])
    if len(warnings) > 40:
        print(f"  ... {len(warnings) - 40} more in split_report.json")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="dataset/ folder of a kvkk_synth run: <parquet> + negatives.jsonl (required unless --frozen-src)",
    )
    parser.add_argument(
        "--src-parquet-name", default="tr_pii_relations_e07.parquet", help="parquet file name inside --src-dir"
    )
    parser.add_argument(
        "--relation-map",
        type=Path,
        default=None,
        help="JSON {english_type: {'tr','desc_tr'}}; applies Turkish canonicals",
    )
    parser.add_argument(
        "--relation-alias-p",
        type=float,
        default=0.4,
        help="P(keep ENGLISH original) per record x type in TRAIN (needs --relation-map)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tag", default="e07", help="suffix for entities/gold file names")
    parser.add_argument(
        "--frozen-src",
        type=Path,
        default=None,
        help="derive outputs by renaming this frozen e07 build instead of "
        "rebuilding from the parquet (use when the parquet has drifted "
        "since the freeze; requires --relation-map)",
    )
    args = parser.parse_args(argv)
    assert 0.0 <= args.relation_alias_p <= 1.0
    if args.src_dir is not None:
        set_source(args.src_dir, args.src_parquet_name)
    elif args.frozen_src is None:
        parser.error("--src-dir (a kvkk_synth run's dataset/ folder) or --frozen-src is required")
    if args.frozen_src is not None:
        assert args.relation_map is not None, "--frozen-src requires --relation-map"
        convert_frozen(
            frozen_dir=args.frozen_src,
            relation_map_path=args.relation_map,
            relation_alias_p=args.relation_alias_p,
            out_dir=args.out_dir,
            tag=args.tag,
        )
    else:
        convert(
            relation_map_path=args.relation_map,
            relation_alias_p=args.relation_alias_p,
            out_dir=args.out_dir,
            tag=args.tag,
        )


if __name__ == "__main__":
    main()
