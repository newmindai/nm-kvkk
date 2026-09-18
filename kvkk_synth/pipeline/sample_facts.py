#!/usr/bin/env python3
"""Fact-bundle sampler on top of kvkk-taxonomy v1.2.

Sampling decides everything a document must contain — the subject person, every entity
value, every relation — so the LLM only has to realize the facts in natural Turkish.
Soundness guarantees per bundle:
  * every relation instance satisfies the taxonomy's head_types / tail_types
  * the relation graph is connected and anchored on one subject person (closure adds the
    anchoring relation for any dangling entity, e.g. card_cvv -> card_number -> card_of -> person)
  * values pass real validators (TCKN checksum, IBAN mod-97, Luhn, initials/components
    consistent with the person's name and address, gender consistent with the name pool)
  * coverage: all 75 relations and all 83 nodes appear across the run (cycled sampling)

Usage: python sample_facts.py -n 100 --seed 42 --out bundles.jsonl
"""

import argparse
import json
import os
import random
import re
import string
import sys
from collections import Counter, defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "taxonomy"))  # [pkg]
import kvkk_sampler as ks  # noqa: E402

REF_YEAR = 2026

# ----------------------------------------------------------------------------- value pools
# Their pools are tiny (15 first names); widen them in place so their generators pick them up.
FIRST_F = [
    "Ayşe",
    "Fatma",
    "Emine",
    "Hatice",
    "Zeynep",
    "Elif",
    "Meryem",
    "Merve",
    "Büşra",
    "Esra",
    "Zehra",
    "Hülya",
    "Sevgi",
    "Nur",
    "Gül",
    "Derya",
    "Selin",
    "Gamze",
    "Özlem",
    "Yasemin",
    "Sibel",
    "Aslı",
    "Ceren",
    "Pınar",
    "Burcu",
    "Dilek",
    "Tuğba",
    "Nazlı",
    "Sevim",
    "Hanife",
]
FIRST_M = [
    "Mehmet",
    "Mustafa",
    "Ahmet",
    "Ali",
    "Hüseyin",
    "Hasan",
    "İbrahim",
    "Yusuf",
    "Murat",
    "Ömer",
    "Emre",
    "Burak",
    "Cem",
    "Kemal",
    "Serkan",
    "Volkan",
    "Kerem",
    "Onur",
    "Halil",
    "Osman",
    "Selim",
    "Tolga",
    "Umut",
    "Barış",
    "Erdem",
    "Fatih",
    "Gökhan",
    "Koray",
    "Levent",
    "Orhan",
]
LAST = [
    "Yılmaz",
    "Kaya",
    "Demir",
    "Şahin",
    "Çelik",
    "Yıldız",
    "Yıldırım",
    "Öztürk",
    "Aydın",
    "Özdemir",
    "Arslan",
    "Doğan",
    "Kılıç",
    "Aslan",
    "Çetin",
    "Kara",
    "Koç",
    "Kurt",
    "Özkan",
    "Şimşek",
    "Polat",
    "Korkmaz",
    "Erdoğan",
    "Güneş",
    "Akın",
    "Bulut",
    "Tekin",
    "Aksoy",
    "Turan",
    "Ateş",
]
COMPANIES = [
    "Kuzey Lojistik A.Ş.",
    "Yılmaz İnşaat Ltd. Şti.",
    "Delta Tekstil San. Tic. A.Ş.",
    "Anadolu Sigorta Aracılık Ltd. Şti.",
    "Marmara Bilişim Hizmetleri A.Ş.",
    "Ege Gıda Pazarlama A.Ş.",
    "Karadeniz Enerji Üretim A.Ş.",
    "Boğaziçi Sağlık Hizmetleri Ltd. Şti.",
    "Nova Yazılım A.Ş.",
    "Meriç Otomotiv Ticaret Ltd. Şti.",
    "Akdeniz Turizm İşletmeleri A.Ş.",
    "Toros Madencilik A.Ş.",
]
CITIES = [
    "İstanbul",
    "Ankara",
    "İzmir",
    "Bursa",
    "Antalya",
    "Adana",
    "Konya",
    "Gaziantep",
    "Kayseri",
    "Mersin",
    "Eskişehir",
    "Samsun",
    "Denizli",
    "Trabzon",
    "Erzurum",
    "Sivas",
    "Sakarya",
    "Muğla",
]
DISTRICTS = [
    "Kadıköy",
    "Çankaya",
    "Bornova",
    "Nilüfer",
    "Osmangazi",
    "Muratpaşa",
    "Selçuklu",
    "Seyhan",
    "Şahinbey",
    "Melikgazi",
    "Odunpazarı",
    "Atakum",
    "Ortahisar",
    "Yenimahalle",
    "Karşıyaka",
]
MAHALLE = [
    "Cumhuriyet",
    "Atatürk",
    "Yıldız",
    "Bahçelievler",
    "Çamlık",
    "Yeşiltepe",
    "Fevzi Çakmak",
    "Mimar Sinan",
    "Barbaros",
    "Esentepe",
    "Güzelyalı",
    "Zafer",
    "Hürriyet",
    "İnönü",
]
STREET = [
    "İstiklal Caddesi",
    "Gazi Bulvarı",
    "Menekşe Sokak",
    "Zafer Caddesi",
    "Papatya Sokak",
    "Lale Sokak",
    "Turgut Özal Bulvarı",
    "Şehitler Caddesi",
    "Gül Sokak",
    "Bağdat Caddesi",
]
BANK_CODES = [
    "00010",
    "00012",
    "00015",
    "00032",
    "00046",
    "00062",
    "00064",
    "00067",
    "00111",
    "00134",
    "00205",
    "00206",
]
BINS = ["4", "51", "52", "53", "54", "55", "9792"]

ks.FIRST[:] = FIRST_F + FIRST_M
ks.LAST[:] = LAST
ks.COMPANIES[:] = COMPANIES
ks.CITIES[:] = CITIES
ks.DISTRICTS[:] = DISTRICTS


def _d(n):
    return "".join(random.choice(string.digits) for _ in range(n))


def gen_iban():
    bban = random.choice(BANK_CODES) + "0" + _d(16)
    num = "".join(str(int(c, 36)) for c in bban + "TR00")
    body = f"TR{98 - int(num) % 97:02d}{bban}"
    return " ".join(body[i : i + 4] for i in range(0, len(body), 4))


def gen_card():
    p = random.choice(BINS)
    d = [int(c) for c in p] + [random.randint(0, 9) for _ in range(15 - len(p))]
    s = 0
    for i, x in enumerate(reversed(d)):
        x = x * 2 if i % 2 == 0 else x
        s += x - 9 if x > 9 else x
    d.append((10 - s % 10) % 10)
    s = "".join(map(str, d))
    return " ".join(s[i : i + 4] for i in range(0, len(s), 4))


# [e01] override removed: tr_heuristics.gen_iban (weighted TCMB bank codes) applies
# [e01] override removed: tr_heuristics.gen_card (TROY/Visa/MC BINs) applies
# [e01] override removed: tr_heuristics.gen_street applies
# [e01] override removed: tr_heuristics.gen_postal applies

ASCII = str.maketrans("ıİşŞçÇöÖüÜğĞ", "iIsScCoOuUgG")


def fold(s):
    return s.translate(ASCII).lower()


# ----------------------------------------------------------------------------- validators
def valid_tckn(v):
    if not re.fullmatch(r"[1-9]\d{10}", v):
        return False
    d = [int(c) for c in v]
    return d[9] == ((sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10) and d[10] == sum(d[:10]) % 10


def valid_iban(v):
    b = v.replace(" ", "")
    if not re.fullmatch(r"TR\d{24}", b):
        return False
    return int("".join(str(int(c, 36)) for c in b[4:] + b[:4])) % 97 == 1


def valid_luhn(v):
    ds = [int(c) for c in v.replace(" ", "")]
    if not ds:
        return False
    s = 0
    for i, x in enumerate(reversed(ds)):
        x = x * 2 if i % 2 == 1 else x
        s += x - 9 if x > 9 else x
    return s % 10 == 0


VALIDATORS = {
    "national_id_number": valid_tckn,
    "iban": valid_iban,
    "card_number": valid_luhn,
    "device_id": valid_luhn,
    "email_address": lambda v: re.fullmatch(r"[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}", v) is not None,
    "kep_address": lambda v: v.endswith(".kep.tr"),
    "date_of_birth": lambda v: re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", v) is not None,
    "phone_number": lambda v: re.fullmatch(r"(\+90 5|05)\d{2} \d{3} \d{2} \d{2}", v) is not None,
    "tax_id_number": lambda v: re.fullmatch(r"\d{10}", v) is not None,
    "card_cvv": lambda v: re.fullmatch(r"\d{3}", v) is not None,
    "card_expiry": lambda v: re.fullmatch(r"(0[1-9]|1[0-2])/\d{2}", v) is not None,
    "ip_address": lambda v: all(0 <= int(x) <= 255 for x in v.split(".")) and v.count(".") == 3,
    "license_plate": lambda v: re.fullmatch(r"\d{2} [A-Z]{1,3} \d{2,4}", v) is not None,
    "vin_chassis_number": lambda v: re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", v) is not None,
    "postal_code": lambda v: re.fullmatch(r"\d{5}", v) is not None,
}


# ----------------------------------------------------------------------------- person / context
class Person:
    def __init__(self, sex=None, full_parts=False):
        self.sex = sex or random.choice("FM")
        pool = FIRST_F if self.sex == "F" else FIRST_M
        self.first = random.choice(pool)
        self.middle = (
            random.choice([n for n in pool if n != self.first]) if full_parts or random.random() < 0.2 else None
        )
        self.last = random.choice(LAST)
        self.title = random.choice(ks.TITLES) if full_parts or random.random() < 0.2 else None
        self.full = " ".join(x for x in (self.title, self.first, self.middle, self.last) if x)

    def initials(self):
        return ".".join(n[0] for n in (self.first, self.middle, self.last) if n).translate(ASCII).upper() + "."


# Relation -> which side the subject person takes when the relation is person-anchored.
# (derived from head_types/tail_types: full_name only on one side)
def person_side(rel):
    h, t = "full_name" in rel["head_types"], "full_name" in rel["tail_types"]
    if h and t:
        return "kinship"
    if t:
        return "tail"
    if h:
        return "head"
    return None


# node -> (relation, side of the subject person) that connects a dangling entity to the subject
ANCHOR = {
    "card_number": ("card_of", "tail"),
    "iban": ("iban_of", "tail"),
    "bank_account_number": ("iban_of", "tail"),
    "username": ("account_of", "tail"),
    "social_media_handle": ("account_of", "tail"),
    "tracked_location": ("located_at", "tail"),
    "full_address": ("residence_of", "tail"),
    "license_plate": ("vehicle_plate_of", "tail"),
    "vin_chassis_number": ("vehicle_vin_of", "tail"),
    "vehicle_registration_number": ("vehicle_registration_of", "tail"),
    "legal_case_number": ("party_in_case", "head"),
    "company_name": ("employer_of", "tail"),
    "criminal_record": ("conviction_of", "tail"),
    "political_opinion": ("political_view_of", "tail"),
    "union_membership": ("member_of", "head"),
    "association_foundation_membership": ("member_of", "head"),
}
KIN_SEX = {"mother_of": "F", "father_of": "M"}
# descriptions that do not follow the "HEAD ... TAIL ..." reading pattern
FACT_OVERRIDES = {"relative_of": "HEAD, TAIL'in akrabasıdır (akrabalık türü belirtilmemiş)."}
MADDE6_GROUPS = {"health", "special_attribution"}


class Bundle:
    def __init__(self, ents_meta, rels_meta, bid, full_parts=False):
        self.meta, self.rmeta, self.bid = ents_meta, rels_meta, bid
        self.p = Person(full_parts=full_parts)  # part_of_name needs middle name + title to exist
        self.ctx = {}  # node -> value shared within the bundle
        self.entities = {}  # (node, value) -> id
        self.order = []  # ids in creation order
        self.relations = []  # (head_id, rel_id, tail_id)
        self.pid = self.ent("full_name", self.p.full)

    # -- entity bookkeeping
    def ent(self, node, value):
        key = (node, value)
        if key not in self.entities:
            self.entities[key] = f"e{len(self.entities) + 1}"
            self.order.append((self.entities[key], node, value))
        return self.entities[key]

    def _address(self):
        if "full_address" not in self.ctx:
            sl, di, ci, pc = (
                ks.GEN["street_line"](),
                random.choice(DISTRICTS),
                random.choice(CITIES),
                ks.GEN["postal_code"](),
            )
            self.ctx.update(street_line=sl, district=di, city=ci, postal_code=pc, full_address=f"{sl} {pc} {di}/{ci}")
        return self.ctx["full_address"]

    def value(self, node):
        p = self.p
        if node == "full_name":
            return p.full
        if node == "first_name":
            return p.first
        if node == "last_name":
            return p.last
        if node == "middle_name":
            return p.middle
        if node == "title_honorific":
            return p.title
        if node == "initials":
            return p.initials()
        if node == "gender":
            return "Kadın" if p.sex == "F" else "Erkek"
        if node == "maiden_name":
            return self.ctx.setdefault(node, random.choice([l for l in LAST if l != p.last]))
        if node == "date_of_birth":
            return self.ctx.setdefault(
                node, f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(1950, 2004)}"
            )
        if node == "age":
            if "date_of_birth" in self.ctx or random.random() < 0.5:
                dob = self.value("date_of_birth")
                return str(REF_YEAR - int(dob[-4:]) - random.choice([0, 1]))
            return self.ctx.setdefault(node, str(random.randint(18, 85)))
        if node == "email_address":
            return self.ctx.setdefault(
                node,
                f"{fold(p.first)}.{fold(p.last)}@{random.choice(['gmail.com', 'outlook.com', 'yandex.com', 'hotmail.com'])}",
            )
        if node == "kep_address":
            return self.ctx.setdefault(node, f"{fold(p.first)}.{fold(p.last)}@hs01.kep.tr")
        if node == "username":
            return self.ctx.setdefault(node, f"{fold(p.first)[0]}{fold(p.last)}{random.randint(10, 99)}")
        if node == "social_media_handle":
            return self.ctx.setdefault(node, f"@{fold(p.first)}_{fold(p.last)}")
        if node == "url_with_pii":
            return self.ctx.setdefault(node, f"https://linkedin.com/in/{fold(p.first)}{fold(p.last)}")
        if node in ("full_address", "street_line", "district", "city", "postal_code"):
            self._address()
            return self.ctx[node]
        return self.ctx.setdefault(node, ks.surface(self.meta, node))

    # -- relation instantiation
    def add_rel(self, head_id, rel_id, tail_id):
        tup = (head_id, rel_id, tail_id)
        if tup not in self.relations:
            self.relations.append(tup)

    def pick_type(self, options, usage):
        opts = list(options)
        if "middle_name" in opts and not self.p.middle:
            opts.remove("middle_name")
        if "title_honorific" in opts and not self.p.title:
            opts.remove("title_honorific")
        m = min(usage[o] for o in opts)
        return random.choice([o for o in opts if usage[o] == m])

    def instantiate(self, rel, usage):
        side = person_side(rel)
        if side == "kinship":
            other = Person(
                sex=KIN_SEX.get(rel["id"]) or ({"F": "M", "M": "F"}[self.p.sex] if rel["id"] == "spouse_of" else None)
            )
            if rel["id"] in KIN_SEX:  # subject is the child; the other is the parent
                p_side, o_side = "tail", "head"
            else:
                p_side, o_side = random.choice([("head", "tail"), ("tail", "head")])
            o_type = self.pick_type(rel[f"{o_side}_types"], usage)
            o_val = other.full if o_type == "full_name" else other.first
            oid = self.ent(o_type, o_val)
            usage[o_type] += 1
            self.add_rel(*(self.pid, rel["id"], oid) if p_side == "head" else (oid, rel["id"], self.pid))
            return
        if side == "tail":
            ht = self.pick_type(rel["head_types"], usage)
            usage[ht] += 1
            self.add_rel(self.ent(ht, self.value(ht)), rel["id"], self.pid)
            return
        if side == "head":
            tt = self.pick_type(rel["tail_types"], usage)
            usage[tt] += 1
            self.add_rel(self.pid, rel["id"], self.ent(tt, self.value(tt)))
            return
        ht = self.pick_type(rel["head_types"], usage)
        usage[ht] += 1
        tt = self.pick_type(rel["tail_types"], usage)
        usage[tt] += 1
        self.add_rel(self.ent(ht, self.value(ht)), rel["id"], self.ent(tt, self.value(tt)))

    # -- connectivity closure: anchor every dangling entity to the subject
    def close(self, usage):
        for _ in range(6):
            adj = defaultdict(set)
            for h, _, t in self.relations:
                adj[h].add(t)
                adj[t].add(h)
            seen, q = {self.pid}, deque([self.pid])
            while q:
                for n in adj[q.popleft()]:
                    if n not in seen:
                        seen.add(n)
                        q.append(n)
            dangling = [(i, n) for i, n, _ in self.order if i not in seen]
            if not dangling:
                return True
            for eid, node in dangling:
                if node in ANCHOR:
                    rel_id, pside = ANCHOR[node]
                    self.add_rel(*(self.pid, rel_id, eid) if pside == "head" else (eid, rel_id, self.pid))
        return False

    # -- export
    def to_json(self, ents_meta, rels_meta):
        rmap = {r["id"]: r for r in rels_meta}
        idv = {i: (n, v) for i, n, v in self.order}
        rels = []
        for h, r, t in self.relations:
            # the clause after ';' in some descriptions is annotation-guideline commentary
            # ("Madde 6 atif cozumu", "coreference kurar") — never show it to the generator
            desc = FACT_OVERRIDES.get(r) or rmap[r]["description"].split(";")[0].strip().rstrip(".") + "."
            assert "HEAD" in desc and "TAIL" in desc, f"fact template for {r} names no participants: {desc}"
            fact = desc.replace("HEAD", idv[h][1]).replace("TAIL", idv[t][1])
            rels.append({"head": h, "relation": r, "tail": t, "group": rmap[r]["_group"], "fact_tr": fact})
        groups = sorted({x["group"] for x in rels})
        return {
            "bundle_id": self.bid,
            "subject_id": self.pid,
            "subject_sex": self.p.sex,
            "entities": [
                {"id": i, "label": n, "value": v, "sensitivity": ents_meta[n]["_sensitivity"]} for i, n, v in self.order
            ],
            "relations": rels,
            "groups": groups,
            "madde6": bool(MADDE6_GROUPS & set(groups)),
        }


# ----------------------------------------------------------------------------- soundness
def check_bundle(b, ents_meta, rels_meta):
    problems = []
    rmap = {r["id"]: r for r in rels_meta}
    byid = {e["id"]: e for e in b["entities"]}
    subj = byid[b["subject_id"]]["value"]
    for r in b["relations"]:
        h, t = byid.get(r["head"]), byid.get(r["tail"])
        if not h or not t:
            problems.append(f"missing entity in {r}")
            continue
        if h["label"] not in rmap[r["relation"]]["head_types"]:
            problems.append(f"{r['relation']}: head type {h['label']} not allowed")
        if t["label"] not in rmap[r["relation"]]["tail_types"]:
            problems.append(f"{r['relation']}: tail type {t['label']} not allowed")
        if r["head"] == r["tail"]:
            problems.append(f"{r['relation']}: self-relation")
    for e in b["entities"]:
        v, n = e["value"], e["label"]
        if n in VALIDATORS and not VALIDATORS[n](v):
            problems.append(f"invalid {n}: {v}")
        if (
            n in ("first_name", "last_name", "middle_name", "title_honorific")
            and any(x["relation"] == "part_of_name" and x["head"] == e["id"] for x in b["relations"])
            and v not in subj
        ):
            problems.append(f"part_of_name component {v} not in {subj}")
        if n == "initials":
            expected = ".".join(w[0] for w in subj.split() if "." not in w).translate(ASCII).upper() + "."
            if v != expected:
                problems.append(f"initials {v} != {expected} for {subj}")
    fa = next((e["value"] for e in b["entities"] if e["label"] == "full_address"), None)
    for e in b["entities"]:
        if e["label"] in ("street_line", "district", "city", "postal_code") and fa and e["value"] not in fa:
            problems.append(f"address component {e['value']} not in {fa}")
    if b["subject_sex"] == "F" and any(e["label"] == "gender" and e["value"] != "Kadın" for e in b["entities"]):
        problems.append("gender inconsistent with subject")
    if b["subject_sex"] == "M" and any(e["label"] == "gender" and e["value"] != "Erkek" for e in b["entities"]):
        problems.append("gender inconsistent with subject")
    # connectivity
    adj = defaultdict(set)
    for r in b["relations"]:
        adj[r["head"]].add(r["tail"])
        adj[r["tail"]].add(r["head"])
    seen, q = {b["subject_id"]}, deque([b["subject_id"]])
    while q:
        for n in adj[q.popleft()]:
            if n not in seen:
                seen.add(n)
                q.append(n)
    if seen != {e["id"] for e in b["entities"]}:
        problems.append("graph not connected to subject")
    if len({e["id"] for e in b["entities"]}) != len(b["entities"]):
        problems.append("duplicate entity ids")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--madde6-boost", type=float, default=0.35)
    ap.add_argument("--out", default="bundles.jsonl")
    a = ap.parse_args()
    random.seed(a.seed)

    ents_meta, rels_meta = ks.load_taxonomy()
    errs, _ = ks.validate(ents_meta, rels_meta)
    assert not errs, errs
    m6 = [r for r in rels_meta if r["_group"] in MADDE6_GROUPS]
    usage = Counter()
    cycle = []
    bundles, report_problems = [], []
    for bid in range(1, a.n + 1):
        k = random.choice([2, 3, 3, 4])
        chosen = []
        while len(chosen) < k:
            if not cycle:
                cycle = random.sample(rels_meta, len(rels_meta))  # coverage: cycle through all 75
            r = cycle.pop()
            if r["id"] not in {c["id"] for c in chosen}:
                chosen.append(r)
        if random.random() < a.madde6_boost:
            chosen.append(random.choice(m6))
        b = Bundle(ents_meta, rels_meta, bid, full_parts=any(r["id"] == "part_of_name" for r in chosen))
        for r in chosen:
            b.instantiate(r, usage)
        assert b.close(usage), f"bundle {bid}: could not connect graph"
        js = b.to_json(ents_meta, rels_meta)
        probs = check_bundle(js, ents_meta, rels_meta)
        if probs:
            report_problems.append({"bundle_id": bid, "problems": probs})
        bundles.append(js)

    with open(a.out, "w", encoding="utf-8") as f:
        for b in bundles:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    rel_cov = Counter(r["relation"] for b in bundles for r in b["relations"])
    node_cov = Counter(e["label"] for b in bundles for e in b["entities"])
    report = {
        "bundles": len(bundles),
        "problems": report_problems,
        "relations_covered": f"{len(rel_cov)}/{len(rels_meta)}",
        "nodes_covered": f"{len(node_cov)}/{len(ents_meta)}",
        "missing_relations": sorted(set(r["id"] for r in rels_meta) - set(rel_cov)),
        "missing_nodes": sorted(set(ents_meta) - set(node_cov)),
        "relations_per_bundle": dict(Counter(len(b["relations"]) for b in bundles)),
        "entities_per_bundle": dict(Counter(len(b["entities"]) for b in bundles)),
        "madde6_share": sum(b["madde6"] for b in bundles) / len(bundles),
        "relation_counts": dict(rel_cov.most_common()),
        "group_counts": dict(Counter(r["group"] for b in bundles for r in b["relations"]).most_common()),
        "distinct_subject_names": len({b["entities"][0]["value"] for b in bundles}),
    }
    json.dump(report, open("soundness_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(
        f"{len(bundles)} bundles -> {a.out} | problems: {len(report_problems)} | "
        f"relations {report['relations_covered']} | nodes {report['nodes_covered']} | "
        f"madde6 share {report['madde6_share']:.2f} | distinct subjects {report['distinct_subject_names']}"
    )
    if report_problems:
        print(json.dumps(report_problems[:5], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
