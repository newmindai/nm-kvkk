"""Label subsets and native-label -> taxonomy-id mappings."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT_ROOT / "configs" / "labels"
LABELS_DIR = CONFIGS / "benchmark"

# KVKK-19: the 19 PII labels of newmind's served KVKK models, in taxonomy ids (3 approximate matches)
KVKK19 = [
    "national_id_number",
    "phone_number",
    "email_address",
    "iban",
    "full_address",
    "date_of_birth",
    "place_of_birth",
    "card_number",
    "ip_address",
    "passport_number",
    "drivers_license_number",
    "license_plate",
    "id_document_serial",
    "gps_coordinates",
    "customer_number",
    "url_with_pii",
    "device_id",
    "social_media_handle",
    "audiovisual_record_reference",
]
NER2 = ["full_name", "company_name"]
STACK21 = KVKK19 + NER2
SUBSETS = {"kvkk19": KVKK19, "stack21": STACK21}

# served model codes -> taxonomy ids
MAPPING_KVKK = {
    "IDN": "national_id_number",
    "PNO": "phone_number",
    "EMA": "email_address",
    "IBN": "iban",
    "ADD": "full_address",
    "BRT": "date_of_birth",
    "PBT": "place_of_birth",
    "CCN": "card_number",
    "IPA": "ip_address",
    "PAS": "passport_number",
    "DLN": "drivers_license_number",
    "CRN": "license_plate",
    "SER": "id_document_serial",
    "COO": "gps_coordinates",
    "CUS": "customer_number",
    "WES": "url_with_pii",
    "MAC": "device_id",
    "HAS": "social_media_handle",
    "PHO": "audiovisual_record_reference",
}
MAPPING_NER = {"PER": "full_name", "COR": "company_name"}  # GOV CRT OOR PRO PUB LAW MNY DAT LOC: no counterpart
# ytu-ce-cosmos/modernbert-tr-pii-ner (25 types). Not mapped: DIN_ETNIK_SIYASI_TERIM and SAGLIK_BILGISI
# (each covers a whole taxonomy group, no single id). VKN -> tax_id_number by name ("vergi kimlik numarası").
MAPPING_COSMOS = {
    "KISI_AD_SOYAD": "full_name",
    "TCKN": "national_id_number",
    "TELEFON": "phone_number",
    "EMAIL": "email_address",
    "ADRES": "full_address",
    "POSTA_KODU": "postal_code",
    "DOGUM_TARIHI": "date_of_birth",
    "IBAN_TR": "iban",
    "HESAP_NO": "bank_account_number",
    "KART_NO": "card_number",
    "KART_CVV": "card_cvv",
    "KART_SON_KULLANMA": "card_expiry",
    "SWIFT_BIC": "swift_bic",
    "IP_ADRES": "ip_address",
    "PASAPORT_NO": "passport_number",
    "SURUCU_BELGESI_NO": "drivers_license_number",
    "KIMLIK_BELGE_NO": "id_document_serial",
    "PLAKA": "license_plate",
    "TUZEL_KISI": "company_name",
    "VKN": "tax_id_number",
    "MERSIS_NO": "mersis_number",
    "MESLEK_UNVAN": "job_title",
    "ETTN_EFATURA_ID": "invoice_document_id",
}
MAPPINGS = {"kvkk": MAPPING_KVKK, "ner": MAPPING_NER, "cosmos": MAPPING_COSMOS}
COSMOS23 = list(MAPPING_COSMOS.values())
COSMOS14 = [i for i in STACK21 if i in set(COSMOS23)]  # what the served models, cosmos and ours all have
SUBSETS.update({"cosmos14": COSMOS14, "cosmos23": COSMOS23})

VOCABS = {
    "taxonomy_tr": CONFIGS / "taxonomy_labels_tr.json",
    "nm6k_tr": CONFIGS / "nm6k_labels_tr.json",
    "nm6k_en": CONFIGS / "nm6k_labels_en.json",
}


def vocab_entries(vocab: str) -> list[dict]:
    return json.loads(VOCABS[vocab].read_text(encoding="utf-8"))


def subset_names(vocab: str, subset: str) -> dict[str, str]:
    """{query name: taxonomy id} for the subset in the given vocabulary (first entry wins on shared names)."""
    ids = set(SUBSETS[subset])
    out: dict[str, str] = {}
    for e in vocab_entries(vocab):
        if e["id"] in ids and e["tr"] not in out:
            out[e["tr"]] = e["id"]
    missing = ids - set(out.values())
    if missing:
        raise ValueError(f"{vocab} lacks names for {sorted(missing)}")
    return out


def write_subset_files() -> list[Path]:
    """Materialise configs/labels/benchmark/<subset>_<vocab>.json in the taxonomy_labels format (for bench_gt.py --labels)."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for subset in SUBSETS:
        for vocab in VOCABS:
            names = subset_names(vocab, subset)
            entries = [
                e for e in vocab_entries(vocab) if e["id"] in set(SUBSETS[subset]) and names.get(e["tr"]) == e["id"]
            ]
            p = LABELS_DIR / f"{subset}_{vocab}.json"
            p.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            written.append(p)
    for name, m in MAPPINGS.items():
        p = LABELS_DIR / f"mapping_{name}.json"
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written.append(p)
    return written
