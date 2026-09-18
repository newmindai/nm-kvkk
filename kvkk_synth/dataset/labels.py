"""Label vocabularies for the nm-kvkk-pii-6K dataset: one monolingual name per entity node and per relation type,
in Turkish and in English. No variant ever mixes the two languages.

Turkish entity names: the 113 verified names of the e07 build
(`configs/labels/e07_mapping.json::entity_label_mapping`) plus the 5 nodes e07 never produced.
Turkish relation names: the 92 verified names of `relation_mapping_tr.json` plus 21 written here in the same
convention — the name is a short noun phrase naming the TAIL argument of the gold head→tail direction:
  "<head özelliği> sahibi"          attribute → person  (adli sicil kaydı sahibi)
  "<rol> olduğu <tail türü>"        person → person/org (kefil olduğu kişi, imza yetkilisi olduğu şirket)
  participle + kişi                 recorded/observed   (görüntüsü kaydedilen kişi)
English names are derived systematically from the taxonomy ids (underscores → spaces), which keeps them unique
and, because every relation id ends in a preposition-like token (of / by / in / at), free of collisions with the
entity names. `check()` asserts both properties.
"""

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # kvkk_synth/dataset/labels.py -> repository root
E07_MAPPING = _REPO / "configs/labels/e07_mapping.json"
E07_REL_TR = _REPO / "configs/labels/e07_relation_mapping_tr.json"

# --- Turkish entity names for the 5 nodes the e07 build never saw -------------------------------------------------
ENTITY_TR_NEW = {
    "military_service_status": "askerlik durumu",
    "nickname": "lakap",
    "philosophical_belief": "felsefi inanç",
    "religious_belief": "dini inanç",
    "union_membership": "sendika üyeliği",
}

# --- Turkish relation names for the 21 types the e07 map lacks ----------------------------------------------------
RELATION_TR_NEW = {
    # attribute (head) → person (tail): "<head özelliği> sahibi"
    "attire_of": ("kılık kıyafet bilgisi sahibi", "Kılık kıyafet bilgisi kişiye aittir"),
    "belief_of": ("inanç bilgisi sahibi", "Dini veya felsefi inanç bilgisi kişiye aittir"),
    "conviction_of": ("adli sicil kaydı sahibi", "Ceza mahkumiyeti veya güvenlik tedbiri kaydı kişiye aittir"),
    "disability_of": ("engellilik durumu sahibi", "Engellilik durumu kişiye aittir"),
    "ethnicity_of": ("etnik köken bilgisi sahibi", "Etnik köken bilgisi kişiye aittir"),
    "foreigner_id_of": ("yabancı kimlik numarası sahibi", "Yabancı kimlik numarası kişiye aittir"),
    "military_status_of": ("askerlik durumu sahibi", "Askerlik durumu kişiye aittir"),
    "nickname_of": ("lakap sahibi", "Lakap kişiye aittir"),
    "sexual_life_of": ("cinsel hayat bilgisi sahibi", "Cinsel hayata ilişkin bilgi kişiye aittir"),
    "depicted_in": ("görüntüsü kaydedilen kişi", "Görsel veya işitsel kayıtta kişinin görüntüsü veya sesi yer alır"),
    # person (head) → person / organisation (tail): "<rol> olduğu <tail türü>"
    "authorised_signatory_of": (
        "imza yetkilisi olduğu şirket",
        "Kişi, şirketin imza yetkilisi veya yetkili temsilcisidir",
    ),
    "owner_of_company": ("sahibi olduğu şirket", "Kişi, şirketin sahibi veya ortağıdır"),
    "represented_by": ("vekili", "Kişi (vekalet veren), adına işlem yapmak üzere bir vekil tayin etmiştir"),
    "guarantor_of": ("kefil olduğu kişi", "Kişi (kefil), borçlunun borcuna veya sözleşmesine kefil olmuştur"),
    "heir_of": ("mirasçısı olduğu kişi", "Kişi, murisin mirasçısıdır"),
    "tenant_of": ("kiracısı olduğu kişi", "Kişi (kiracı), kiraya verenin taşınmazında kiracıdır"),
    "sibling_of": ("kardeşi olduğu kişi", "Kişi, diğer kişinin kardeşidir"),
    "witness_of": ("tanık olduğu işlem", "Kişi, noter işlemine veya dava dosyasına tanık olmuştur"),
    # value ↔ value
    "property_at": ("taşınmazın adresi", "Ada/parsel veya bağımsız bölüm künyesiyle belirtilen taşınmaz bu adrestedir"),
    "same_vehicle": ("aynı aracın kaydı", "Plaka ile şasi veya ruhsat numarası aynı araca aittir"),
    "verdict_in_case": ("hükmün verildiği dava", "Ceza mahkumiyeti bu dava dosyasında verilmiştir"),
}


def english(node_or_relation_id: str) -> str:
    return node_or_relation_id.replace("_", " ")


def build(ents_meta, rels_meta):
    """-> {"tr": {...}, "en": {...}} with entity_labels, relation_names and relation_descriptions per language."""
    m = json.loads(E07_MAPPING.read_text(encoding="utf-8"))
    ent_tr = dict(m["entity_label_mapping"])
    ent_tr.update(ENTITY_TR_NEW)
    desc_en = dict(m["relation_descriptions"])
    rel_tr_src = {k: v for k, v in json.loads(E07_REL_TR.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    rel_tr = {k: v["tr"] for k, v in rel_tr_src.items()}
    desc_tr = {k: v["desc_tr"] for k, v in rel_tr_src.items()}
    for rid, (name, desc) in RELATION_TR_NEW.items():
        rel_tr[rid] = name
        desc_tr[rid] = desc
    for rid, r in ((r["id"], r) for r in rels_meta):  # English descriptions for the types e07 never had
        desc_en.setdefault(rid, r["description"].split(";")[0].strip().rstrip("."))

    missing_e = sorted(n for n in ents_meta if n not in ent_tr)
    missing_r = sorted(r["id"] for r in rels_meta if r["id"] not in rel_tr)
    if missing_e or missing_r:
        raise SystemExit(f"no Turkish name for entities {missing_e} / relations {missing_r}")

    out = {
        "tr": {
            "entity_labels": {n: ent_tr[n] for n in ents_meta},
            "relation_names": {r["id"]: rel_tr[r["id"]] for r in rels_meta},
            "relation_descriptions": {rel_tr[r["id"]]: desc_tr[r["id"]] for r in rels_meta},
        },
        "en": {
            "entity_labels": {n: english(n) for n in ents_meta},
            "relation_names": {r["id"]: english(r["id"]) for r in rels_meta},
            "relation_descriptions": {english(r["id"]): desc_en[r["id"]] for r in rels_meta},
        },
    }
    check(out)
    return out


def check(maps):
    for lang, m in maps.items():
        ents = set(m["entity_labels"].values())
        rels = set(m["relation_names"].values())
        clash = ents & rels
        if clash:
            raise SystemExit(f"[{lang}] entity and relation names collide: {sorted(clash)}")
        dup = [v for v in rels if list(m["relation_names"].values()).count(v) > 1]
        if dup:
            raise SystemExit(f"[{lang}] duplicate relation names: {sorted(set(dup))}")
    return True
