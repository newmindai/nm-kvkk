#!/usr/bin/env python3
"""
kvkk_sampler.py — KVKK PII taksonomisi icin dogrulama + sentetik ornekleme araci.

Kullanim:
  python3 kvkk_sampler.py validate
  python3 kvkk_sampler.py sample  -n 200 --seed 42 --out data/records.jsonl
  python3 kvkk_sampler.py sample  -n 200 --hub 3 --negatives 0.2 --out data/hub.jsonl
  python3 kvkk_sampler.py convert --inp data/records.jsonl --format gliner --out data/gliner.jsonl

Cikti semasi (records):
  {"text": str,
   "entities":  [{"id","label","span","start","end"}],   # label = entity node id
   "relations": [{"head","relation","tail"}]}            # id referanslari; DOC_SUBJECT olabilir

Notlar:
- Sablonlar tohumdur; morfoloji (ek uyumu) LLM parafraz asamasinda duzeltilmelidir.
- On-tokenizasyon POLICY K1(a) kararini uygular: kesme isaretinde zorunlu bolme.
"""

import argparse
import glob
import json
import os
import random
import re
import sys

TAXO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomies")


# ---------------------------------------------------------------- yukleme
def load_taxonomy():
    entities, relations = {}, []
    for f in sorted(glob.glob(os.path.join(TAXO_DIR, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("layer") == "relation":
            for g in d["relation_groups"]:
                for r in g["relations"]:
                    r["_group"] = g["group"]
                    relations.append(r)
        else:
            for n in d["nodes"]:
                n["_file"] = os.path.basename(f)
                n["_sensitivity"] = d["kvkk_sensitivity"]
                n["_out_of_scope"] = d.get("out_of_kvkk_scope", False)
                entities[n["id"]] = n
    return entities, relations


# ---------------------------------------------------------------- dogrulama
LABEL_RE = re.compile(r"^[a-z0-9_]+$")


def validate(entities, relations):
    errs, warns = [], []
    seen = {}
    for nid, n in entities.items():
        for lab in n["labels_en"] + n["labels_tr"]:
            if not LABEL_RE.match(lab):
                errs.append(f"etiket kurali ihlali: {nid} -> '{lab}'")
            if lab in seen and seen[lab] != nid:
                warns.append(f"alias cakismasi: '{lab}' hem {seen[lab]} hem {nid}")
            seen[lab] = nid
    referenced = set()
    for r in relations:
        for t in r["head_types"] + r["tail_types"]:
            referenced.add(t)
            if t not in entities:
                errs.append(f"kirik referans: {r['id']} -> '{t}'")
        if not r.get("templates_tr"):
            warns.append(f"sablonsuz iliski: {r['id']}")
    uncovered = set(entities) - referenced
    for u in sorted(uncovered):
        errs.append(f"kapsanmayan entity (hicbir iliskide yok): {u}")
    return errs, warns


# ---------------------------------------------------------------- deger uretecleri
FIRST = [
    "Ayşe",
    "Mehmet",
    "Fatma",
    "Ali",
    "Zeynep",
    "Mustafa",
    "Elif",
    "Ahmet",
    "Emine",
    "Hasan",
    "Deniz",
    "Umut",
    "Cem",
    "Selin",
    "Burak",
]
LAST = [
    "Yılmaz",
    "Kaya",
    "Demir",
    "Şahin",
    "Çelik",
    "Yıldız",
    "Aydın",
    "Öztürk",
    "Arslan",
    "Doğan",
    "Koç",
    "Kurt",
    "Polat",
]
TITLES = ["Dr.", "Av.", "Prof. Dr.", "Uzm.", "Op. Dr.", "Müh."]
CITIES = ["İstanbul", "Ankara", "İzmir", "Bursa", "Sivas", "Trabzon", "Antalya", "Konya", "Eskişehir"]
DISTRICTS = ["Kadıköy", "Çankaya", "Bornova", "Nilüfer", "Osmangazi", "Muratpaşa", "Selçuklu"]
STREETS = ["Çamlık Mah. Gül Sok.", "Atatürk Cad.", "Cumhuriyet Mah. Lale Sok.", "Bahar Mah. Menekşe Cad."]
COMPANIES = ["X A.Ş.", "Yılmaz İnşaat Ltd. Şti.", "Delta Lojistik A.Ş.", "Nova Tekstil San. Tic. Ltd. Şti."]
CONDS = ["tip 2 diyabet", "hipertansiyon", "astım", "kalp yetmezliği", "migren"]
MENTAL = ["depresyon tedavisi", "anksiyete bozukluğu tanısı"]
MEDS = ["Metformin", "Parol", "Coraspin", "Ventolin"]
PROCS = ["bypass ameliyatı", "apandisit ameliyatı", "fizik tedavi", "kemoterapi"]
RELIG = ["Alevi inancına mensup", "Sünni", "ateist", "deist"]
ETHN = ["Kürt kökenli", "Çerkes kökenli", "Roman kökenli"]
POLIT = ["X partisi üyesi", "Y partisi sempatizanı"]
UNIONS = ["Türk-İş üyesi", "DİSK üyesi", "Eğitim-Sen üyesi"]
ASSOC = ["Kızılay Derneği üyesi", "TEMA Vakfı gönüllüsü"]
ATTIRE = ["Başörtülü", "Sarıklı ve cübbeli"]
SEXL = ["cinsel yönelimi", "eşcinsel olduğu"]
CRIM = ["2 yıl hapis cezası", "dolandırıcılık hükmü", "denetimli serbestlik kararı"]
JOBS = ["Makine Mühendisi", "satış müdürü", "hemşire", "öğretmen", "avukat"]
EDUS = ["2015 yılında İTÜ Bilgisayar Mühendisliği", "2010 yılında AÜ Hukuk Fakültesi"]
WORKH = ["2018-2021 arasında X A.Ş.'de satış müdürü", "2012-2016 arasında Delta Lojistik'te operasyon uzmanı"]
DISCP = ["kınama cezası", "uyarı cezası"]
TRACK = ["Ankara Çankaya", "İstanbul Maslak", "İzmir Alsancak"]


def _d(n):
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def gen_tckn():
    while True:
        d = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
        d10 = ((sum(d[0::2]) * 7) - sum(d[1::2])) % 10
        d11 = (sum(d) + d10) % 10
        s = "".join(map(str, d)) + str(d10) + str(d11)
        return s


def gen_iban():
    bban = _d(5) + "0" + _d(16)
    num = "".join(bban) + "292700"  # TR -> 29,27 ; 00
    check = 98 - (int(num) % 97)
    body = f"TR{check:02d}{bban}"
    return " ".join(body[i : i + 4] for i in range(0, len(body), 4))


def gen_luhn(n=16, prefix="4"):
    d = [int(prefix)] + [random.randint(0, 9) for _ in range(n - 2)]

    def luhn_check(ds):
        s = 0
        for i, x in enumerate(reversed(ds)):
            x = x * 2 if i % 2 == 0 else x
            s += x - 9 if x > 9 else x
        return (10 - s % 10) % 10

    d.append(luhn_check(d))
    s = "".join(map(str, d))
    return " ".join(s[i : i + 4] for i in range(0, len(s), 4))


def gen_imei():
    d = [random.randint(0, 9) for _ in range(14)]
    s = 0
    for i, x in enumerate(reversed(d)):
        x = x * 2 if i % 2 == 0 else x
        s += x - 9 if x > 9 else x
    return "".join(map(str, d)) + str((10 - s % 10) % 10)


def full_name():
    n = f"{random.choice(FIRST)} {random.choice(LAST)}"
    if random.random() < 0.2:
        n = f"{random.choice(TITLES)} {n}"
    if random.random() < 0.1:
        n = n.lower()
    return n


GEN = {
    "full_name": full_name,
    "first_name": lambda: random.choice(FIRST),
    "middle_name": lambda: random.choice(FIRST),
    "last_name": lambda: random.choice(LAST),
    "maiden_name": lambda: random.choice(LAST),
    "mothers_maiden_name": lambda: random.choice(LAST),
    "initials": lambda: ".".join(random.choice("ABCDEFGHKLMNSTUYZ") for _ in range(random.randint(2, 3))) + ".",
    "nickname": lambda: random.choice(["Topal Hasan", "Deli Cemal", "Küçük Ayşe"]),
    "title_honorific": lambda: random.choice(TITLES),
    "national_id_number": gen_tckn,
    "foreigner_id_number": lambda: "99" + _d(9),
    "id_document_serial": lambda: random.choice("ABCUVYZ") + _d(2) + random.choice("ABCUVYZ") + _d(5),
    "passport_number": lambda: random.choice("US") + _d(8),
    "drivers_license_number": lambda: _d(6),
    "tax_id_number": lambda: _d(10),
    "social_security_number": lambda: _d(13),
    "phone_number": lambda: (
        random.choice(["+90 5", "05"]) + f"{random.randint(30, 59)}" + " " + _d(3) + " " + _d(2) + " " + _d(2)
    ),
    "fax_number": lambda: "0" + str(random.choice([212, 216, 312])) + " " + _d(3) + " " + _d(2) + " " + _d(2),
    "email_address": lambda: (
        f"{random.choice(FIRST).lower()}.{random.choice(LAST).lower()}@example.com".replace("ı", "i")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ğ", "g")
    ),
    "kep_address": lambda: (
        f"{random.choice(FIRST).lower()}.{random.choice(LAST).lower()}@hs01.kep.tr".replace("ı", "i")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ğ", "g")
    ),
    "full_address": lambda: (
        f"{random.choice(STREETS)} No:{random.randint(1, 60)} D:{random.randint(1, 20)} {random.choice(DISTRICTS)}/{random.choice(CITIES)}"
    ),
    "street_line": lambda: f"{random.choice(STREETS)} No:{random.randint(1, 60)} D:{random.randint(1, 20)}",
    "district": lambda: random.choice(DISTRICTS),
    "city": lambda: random.choice(CITIES),
    "postal_code": lambda: _d(5),
    "date_of_birth": lambda: f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(1950, 2005)}",
    "place_of_birth": lambda: random.choice(CITIES),
    "age": lambda: str(random.randint(18, 85)),
    "gender": lambda: random.choice(["Kadın", "Erkek"]),
    "marital_status": lambda: random.choice(["Evli", "Bekar", "Boşanmış"]),
    "nationality": lambda: random.choice(["T.C.", "Alman", "Suriye"]),
    "blood_type": lambda: random.choice(["A Rh+", "0 Rh-", "B Rh+", "AB Rh+"]),
    "iban": gen_iban,
    "card_number": gen_luhn,
    "card_cvv": lambda: _d(3),
    "card_expiry": lambda: f"{random.randint(1, 12):02d}/{random.randint(26, 31)}",
    "bank_account_number": lambda: _d(8),
    "swift_bic": lambda: random.choice(["TGBATRIS", "ISBKTRIS", "AKBKTRIS"]),
    "salary_info": lambda: f"{random.randint(20, 150)}.000 TL",
    "ip_address": lambda: ".".join(str(random.randint(1, 254)) for _ in range(4)),
    "username": lambda: (
        f"{random.choice(FIRST).lower()[0]}{random.choice(LAST).lower()}{random.randint(10, 99)}".replace("ı", "i")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ğ", "g")
    ),
    "password": lambda: "Xk" + _d(1) + "!" + random.choice("mnpq") + _d(1) + "Pq",
    "device_id": gen_imei,
    "social_media_handle": lambda: (
        "@"
        + f"{random.choice(FIRST).lower()}_{random.choice(LAST).lower()}".replace("ı", "i")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ğ", "g")
    ),
    "url_with_pii": lambda: (
        "https://linkedin.com/in/"
        + f"{random.choice(FIRST).lower()}{random.choice(LAST).lower()}".replace("ı", "i")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ğ", "g")
    ),
    "session_or_token": lambda: "eyJhbGciOiJIUzI1NiJ9." + _d(6),
    "gps_coordinates": lambda: f"{random.uniform(36, 42):.4f}, {random.uniform(26, 45):.4f}",
    "tracked_location": lambda: random.choice(TRACK),
    "workplace_location": lambda: random.choice(["Maslak", "Kızılay", "Alsancak"]),
    "license_plate": lambda: (
        f"{random.randint(1, 81):02d} {''.join(random.choice('ABCDEFGHJKLMNPRSTUVYZ') for _ in range(random.randint(1, 3)))} {_d(random.randint(2, 4))}"
    ),
    "vin_chassis_number": lambda: "".join(random.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17)),
    "vehicle_registration_number": lambda: random.choice("ABC") + random.choice("ABC") + " " + _d(6),
    "medical_condition": lambda: random.choice(CONDS),
    "mental_health_info": lambda: random.choice(MENTAL),
    "medication": lambda: random.choice(MEDS),
    "medical_procedure": lambda: random.choice(PROCS),
    "disability_status": lambda: f"%{random.choice([40, 60, 80])} engelli",
    "health_report_id": lambda: f"{random.randint(2020, 2026)}/{_d(5)}",
    "ethnicity": lambda: random.choice(ETHN),
    "political_opinion": lambda: random.choice(POLIT),
    "religious_belief": lambda: random.choice(RELIG[:2]),
    "philosophical_belief": lambda: random.choice(RELIG[2:]),
    "attire_info": lambda: random.choice(ATTIRE),
    "union_membership": lambda: random.choice(UNIONS),
    "association_foundation_membership": lambda: random.choice(ASSOC),
    "sexual_life_info": lambda: random.choice(SEXL),
    "criminal_record": lambda: random.choice(CRIM),
    "biometric_data_reference": lambda: "parmak izi şablonu FP-" + _d(5),
    "genetic_data_reference": lambda: random.choice(["BRCA1 mutasyonu", "MTHFR varyantı"]),
    "job_title": lambda: random.choice(JOBS),
    "employee_id": lambda: _d(4),
    "work_history": lambda: random.choice(WORKH),
    "performance_disciplinary": lambda: random.choice(DISCP),
    "education_info": lambda: random.choice(EDUS),
    "company_name": lambda: random.choice(COMPANIES),
    "mersis_number": lambda: _d(16),
    "trade_registry_number": lambda: _d(6) + "-" + _d(1),
    "invoice_document_id": lambda: "550e8400-e29b-41d4-a716-" + _d(12),
    "legal_case_number": lambda: (
        f"{random.randint(2018, 2026)}/{random.randint(1, 999)} " + random.choice(["E.", "K."])
    ),
    "customer_number": lambda: _d(9),
    "subscription_number": lambda: _d(7),
    "order_reference": lambda: f"SP-{random.randint(2023, 2026)}-{_d(5)}",
    "audiovisual_record_reference": lambda: random.choice(
        [
            f"KAM-{random.randint(1, 9):02d} kamerasının {random.randint(0, 23):02d}:{random.randint(0, 59):02d} kaydı",
            f"IMG_{_d(4)}.jpg",
        ]
    ),
}

try:  # arastirma tabanli heuristik ureticiler varsa temel ureticileri gecersiz kil
    from tr_heuristics import HGEN

    GEN.update(HGEN)
except ImportError:
    pass


def surface(entities, node_id):
    if node_id in GEN:
        return GEN[node_id]()
    ex = [e.get("span") for e in entities[node_id]["examples"] if e.get("span")]
    if ex:
        return random.choice(ex)
    raise ValueError(f"deger uretilemedi: {node_id}")


# ---------------------------------------------------------------- ornekleme
def fill(template, head_val, tail_val):
    """Sablonu doldurur; yerlestirilen span'lerin karakter konumlarini dondurur."""
    text, spans = "", {}
    i = 0
    for m in re.finditer(r"\{(head|tail)\}", template):
        text += template[i : m.start()]
        val = head_val if m.group(1) == "head" else tail_val
        spans[m.group(1)] = (len(text), len(text) + len(val))
        text += val
        i = m.end()
    text += template[i:]
    return text, spans


def sample_record(entities, relations, hub=1):
    """hub=1: tek iliskili cumle. hub>1: ayni kisiye bagli coklu iliski, cumleler birlesir."""
    person = surface(entities, "full_name")
    rels = random.sample(relations, k=min(hub, len(relations)))
    text_parts, ents, rel_out, eid = [], [], [], 0
    offset = 0
    pid = None
    for r in rels:
        head_t = random.choice(r["head_types"])
        tail_t = random.choice(r["tail_types"])
        head_v = (
            person
            if head_t == "full_name"
            and "full_name" in r["head_types"]
            and hub > 1
            and r["_group"] in ("special_attribution", "legal_customer")
            else surface(entities, head_t)
        )
        tail_v = person if tail_t == "full_name" and hub > 1 else surface(entities, tail_t)
        if head_t == "full_name" and tail_t == "full_name" and head_v == tail_v:
            head_v = surface(entities, head_t)
        tmpl = random.choice(r["templates_tr"])
        sent, spans = fill(tmpl, head_v, tail_v)
        ids = {}
        for role, (s, e) in spans.items():
            val = head_v if role == "head" else tail_v
            nid = head_t if role == "head" else tail_t
            if hub > 1 and val == person and pid is not None:
                ids[role] = pid
            else:
                eid += 1
                ids[role] = f"e{eid}"
                if val == person:
                    pid = ids[role]
            ents.append({"id": ids[role], "label": nid, "span": val, "start": offset + s, "end": offset + e})
        rel_out.append(
            {"head": ids.get("head", "DOC_SUBJECT"), "relation": r["id"], "tail": ids.get("tail", "DOC_SUBJECT")}
        )
        text_parts.append(sent)
        offset += len(sent) + 1
    return {"text": " ".join(text_parts), "entities": ents, "relations": rel_out}


def sample_negative(entities):
    """Iki entity ayni cumlede ama iliskisiz: negatif es-gecis ornegi."""
    a, b = random.sample(["full_name", "phone_number", "company_name", "city", "license_plate"], 2)
    va, vb = surface(entities, a), surface(entities, b)
    tmpl = random.choice(
        [
            "Toplantıda {A} ve {B} ayrı ayrı gündeme geldi.",
            "Kayıtlarda {A} ile {B} arasında bağlantı kurulamadı.",
        ]
    )
    text = tmpl.replace("{A}", va).replace("{B}", vb)
    ents = []
    for i, (nid, v) in enumerate([(a, va), (b, vb)], 1):
        s = text.find(v)
        ents.append({"id": f"e{i}", "label": nid, "span": v, "start": s, "end": s + len(v)})
    return {"text": text, "entities": ents, "relations": []}


# ---------------------------------------------------------------- donusum
def pretokenize(text):
    """POLICY K1(a): whitespace bolme + kesme isaretinde zorunlu bolme."""
    toks = []
    for w in text.split():
        parts = re.split(r"(?<=\S)(['’])", w, maxsplit=1)
        if len(parts) == 3:
            toks.append(parts[0])
            toks.append(parts[1] + parts[2])
        else:
            toks.append(w)
    return toks


def char_to_token_spans(text, tokens, start, end):
    pos, bounds = 0, []
    for t in tokens:
        pos = text.find(t, pos)
        bounds.append((pos, pos + len(t)))
        pos += len(t)
    ts = te = None
    for i, (s, e) in enumerate(bounds):
        if ts is None and e > start:
            ts = i
        if s < end:
            te = i
    return ts, te


def to_gliner(rec):
    toks = pretokenize(rec["text"])
    ner = []
    idmap = {}
    for ent in rec["entities"]:
        ts, te = char_to_token_spans(rec["text"], toks, ent["start"], ent["end"])
        if ts is None:
            continue
        ner.append([ts, te, ent["label"]])
        idmap[ent["id"]] = [ts, te]
    rels = [
        {"head": idmap[r["head"]], "tail": idmap[r["tail"]], "relation": r["relation"]}
        for r in rec["relations"]
        if r["head"] in idmap and r["tail"] in idmap
    ]
    return {"tokenized_text": toks, "ner": ner, "relations": rels}


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sp = sub.add_parser("sample")
    sp.add_argument("-n", type=int, default=100)
    sp.add_argument("--hub", type=int, default=1)
    sp.add_argument("--negatives", type=float, default=0.15)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--out", default="records.jsonl")
    cp = sub.add_parser("convert")
    cp.add_argument("--inp", required=True)
    cp.add_argument("--format", choices=["gliner"], default="gliner")
    cp.add_argument("--out", required=True)
    a = ap.parse_args()

    entities, relations = load_taxonomy()
    if a.cmd == "validate":
        errs, warns = validate(entities, relations)
        for w in warns:
            print("UYARI:", w)
        for e in errs:
            print("HATA:", e)
        print(f"{len(entities)} entity, {len(relations)} iliski | {len(errs)} hata, {len(warns)} uyari")
        sys.exit(1 if errs else 0)
    if a.cmd == "sample":
        if a.seed is not None:
            random.seed(a.seed)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for _ in range(a.n):
                rec = (
                    sample_negative(entities)
                    if random.random() < a.negatives
                    else sample_record(entities, relations, hub=a.hub)
                )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{a.n} kayit -> {a.out}")
    if a.cmd == "convert":
        with open(a.inp, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8") as fo:
            for line in fi:
                fo.write(json.dumps(to_gliner(json.loads(line)), ensure_ascii=False) + "\n")
        print(f"donusum tamam -> {a.out}")


if __name__ == "__main__":
    main()
