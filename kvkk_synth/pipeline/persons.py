"""Persons axis (design §4.4): a document may name more than one person.

A secondary person is one of three kinds:
  related       — via a kinship / emergency-contact relation to the subject (the taxonomy already models these)
  counterpart   — a professional the subject deals with (physician, lawyer, expert, judge/prosecutor, HR contact),
                  with their own role-bound identifier; NOT related to the subject by any taxonomy relation
  bystander     — an unrelated person in the same document (witness, another customer, a colleague); no relation
Each secondary person carries 0–3 entities of their own, anchored to that person. Every entity of a counterpart or
bystander yields an explicit NEGATIVE pair against the subject (relations_negative): the value is in the text and it
is not the subject's — the signal a relation model needs.
"""

import random

KIN_RELS = ["spouse_of", "mother_of", "father_of", "child_of", "sibling_of", "relative_of", "emergency_contact_of"]
COUNTERPARTS = [  # role -> (title, role_en, role_tr, identifier nodes)
    ("physician", "Dr.", "the physician who examined / treated the subject", "hekim", ["doctor_license_number"]),
    ("lawyer", "Av.", "the subject's lawyer (a different person)", "avukat", ["bar_registry_number"]),
    ("expert", None, "the court-appointed expert on the subject's case", "bilirkişi", ["expert_registry_number"]),
    (
        "judge",
        None,
        "the judge or prosecutor handling the subject's file",
        "hakim/savcı",
        ["judge_prosecutor_registry_number"],
    ),
    (
        "hr_contact",
        None,
        "the HR / customer-relations officer handling the subject's request",
        "yetkili",
        ["employee_id", "phone_number", "email_address"],
    ),
]
BYSTANDERS = [
    ("witness", "a witness named in the document, unrelated to the subject", "tanık"),
    ("other_customer", "another customer / applicant listed in the same document", "diğer müşteri"),
    ("colleague", "a colleague of the subject mentioned in passing", "iş arkadaşı"),
]
OWN_ENTITY_NODES = [
    "phone_number",
    "email_address",
    "employee_id",
    "customer_number",
    "city",
]  # what a secondary person may carry
SHARES_PERSONS = [(1, 60), (2, 28), (3, 12)]  # design defaults
ROLE_WORDS = {  # [sexfix] (role_en, role_tr) per sex — the writer never has to guess kızı/oğlu
    "spouse_of": {"F": ("the subject's wife", "eşi (kadın)"), "M": ("the subject's husband", "eşi (erkek)")},
    "mother_of": {"F": ("the subject's mother", "annesi"), "M": ("the subject's mother", "annesi")},
    "father_of": {"F": ("the subject's father", "babası"), "M": ("the subject's father", "babası")},
    "child_of": {"F": ("the subject's daughter", "kızı"), "M": ("the subject's son", "oğlu")},
    "sibling_of": {
        "F": ("the subject's sister", "kız kardeşi / ablası"),
        "M": ("the subject's brother", "erkek kardeşi / abisi"),
    },
    "relative_of": {
        "F": ("a female relative of the subject", "akrabası (kadın)"),
        "M": ("a male relative of the subject", "akrabası (erkek)"),
    },
    "emergency_contact_of": {
        "F": ("the subject's emergency contact (a woman)", "acil durumda ulaşılacak kişi (kadın)"),
        "M": ("the subject's emergency contact (a man)", "acil durumda ulaşılacak kişi (erkek)"),
    },
}
SHARES_KIND = [("related", 50), ("counterpart", 30), ("bystander", 20)]


def _wchoice(pairs):
    vals, w = zip(*pairs)
    return random.choices(vals, w)[0]


def _person_value(b, p, node):
    """A value of `node` that belongs to secondary person `p` (not the subject)."""
    ks = b.ks
    if node == "email_address":
        return f"{b.fold(p.first)}.{b.fold(p.last)}@{random.choice(['gmail.com', 'hotmail.com', 'outlook.com', 'yandex.com'])}"
    if node == "city":
        return b.loc.rng.choice([b.loc.city, "Ankara", "İzmir", "İstanbul"])
    if node == "date_of_birth":
        return f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(1940, 2015)}"
    return ks.surface(b.meta, node)  # taxonomy generator, or an example span for the example-only nodes


def add_secondary(b, kind, rels_meta, usage, role=None):
    """Add one secondary person of `kind` to bundle `b`. Returns a persons-list entry or None."""
    if kind == "role":  # [e08]
        return add_role_person(b, role, rels_meta, usage)
    if kind == "related":
        rel_id = role or random.choice(KIN_RELS)
        # [sexfix] the relative's sex is decided here and passed everywhere: name pool (strict F/M), prompt, role word
        sex = {"mother_of": "F", "father_of": "M"}.get(rel_id) or (
            {"F": "M", "M": "F"}[b.p.sex] if rel_id == "spouse_of" else random.choice("FM")
        )
        saved = b.ks.TITLES[:]
        if rel_id != "spouse_of":
            b.ks.TITLES[:] = [""]  # no "Prof. Dr." on a child or a parent
        try:
            p = b.Person(sex=sex, strict=True)
        finally:
            b.ks.TITLES[:] = saved
        pid = b.ent("full_name", p.full)
        usage["full_name"] += 1
        # direction: the relative is the head, the subject the tail (annesi / babası / çocuğu / kardeşi / akrabası / eşi / acil kişisi of the subject)
        b.add_rel(pid, rel_id, b.pid)
        role_en, role_tr = ROLE_WORDS[rel_id][sex]
        ents, negatives = [pid], []
        if (
            random.random() < 0.85
        ):  # [e08] a relative usually carries one own datum (a parent's phone, a spouse's TCKN) -> a negative
            node = random.choice(["phone_number", "national_id_number", "email_address", "date_of_birth"])
            cands = [
                r
                for r in rels_meta
                if (node in r["head_types"] and "full_name" in r["tail_types"])
                or (node in r["tail_types"] and "full_name" in r["head_types"])
            ]
            subject_vals = {v for (n_, v), i_ in b.entities.items()}
            val = _person_value(b, p, node)
            for _ in range(20):
                if val not in subject_vals:
                    break
                val = _person_value(b, p, node)
            if cands and val not in subject_vals:
                rel = random.choice(cands)
                eid = b.ent(node, val)
                ents.append(eid)
                usage[node] += 1
                if node in rel["head_types"]:
                    b.add_rel(eid, rel["id"], pid)
                    negatives.append({"head": eid, "relation": rel["id"], "tail": b.pid})
                else:
                    b.add_rel(pid, rel["id"], eid)
                    negatives.append({"head": b.pid, "relation": rel["id"], "tail": eid})
        return {
            "id": pid,
            "name": p.full,
            "sex": sex,
            "kind": "related",
            "role": rel_id,
            "role_en": role_en,
            "role_tr": role_tr,
            "entities": ents,
            "negatives": negatives,
        }
    # counterpart or bystander: a fresh person from the pools
    p = b.Person(sex=random.choice("FM"), strict=True)  # [sexfix]
    if kind == "counterpart":
        role, title, role_en, role_tr, own_nodes = next(
            (c for c in COUNTERPARTS if c[0] == role), None
        ) or random.choice(COUNTERPARTS)
        name = f"{title} {p.first} {p.last}" if title else f"{p.first} {p.last}"
    else:
        role, role_en, role_tr = next((c for c in BYSTANDERS if c[0] == role), None) or random.choice(BYSTANDERS)
        own_nodes = random.sample(OWN_ENTITY_NODES, k=random.choice([1, 1, 2]))  # [e08] ≥ 1 own entity → ≥ 1 negative
        name = f"{p.first} {p.last}"
    pid = b.ent("full_name", name)
    ents, negatives, rels = [pid], [], []
    for node in own_nodes:
        # the relation that anchors this node to a person (same lookup as the pool sampler)
        cands = [
            r
            for r in rels_meta
            if (node in r["head_types"] and "full_name" in r["tail_types"])
            or (node in r["tail_types"] and "full_name" in r["head_types"])
        ]
        if not cands:
            continue
        rel = random.choice(cands)
        val = _person_value(b, p, node)
        subject_vals = {v for (n_, v), i_ in b.entities.items()}
        for _ in range(20):  # [e08] never reuse a value already in the bundle (a negative must not equal a positive)
            if val not in subject_vals:
                break
            val = _person_value(b, p, node)
        if val in subject_vals:
            continue
        eid = b.ent(node, val)
        ents.append(eid)
        usage[node] += 1
        if node in rel["head_types"]:
            b.add_rel(eid, rel["id"], pid)
            rels.append((eid, rel["id"], pid))
            negatives.append({"head": eid, "relation": rel["id"], "tail": b.pid})
        else:
            b.add_rel(pid, rel["id"], eid)
            rels.append((pid, rel["id"], eid))
            negatives.append({"head": b.pid, "relation": rel["id"], "tail": eid})
    return {
        "id": pid,
        "name": name,
        "sex": p.sex,
        "kind": kind,
        "role": role,
        "role_en": role_en,
        "role_tr": role_tr,
        "entities": ents,
        "negatives": negatives,
    }


# [e08] role relations from taxonomy-ext/16_relations_roles.json (design §4.8): which side the subject takes, what the
# secondary person carries (every own entity -> a negative against the subject), and the words the writer sees.
# 'variants' draw the direction: the subject can be the heir or the deceased, the tenant or the landlord.
ROLE_SPECS = {
    "represented_by": {
        "variants": [
            (
                "head",
                "the attorney-in-fact / representative (vekil) to whom the subject grants power of attorney",
                "vekili (vekalet alan)",
            )
        ],
        "title": ("Av.", 0.6),
        "own": ["bar_registry_number", "national_id_number", "phone_number"],
    },
    "guarantor_of": {
        "variants": [("tail", "the guarantor (kefil) who vouches for the subject's debt or contract", "kefili")],
        "title": None,
        "own": ["national_id_number", "phone_number", "email_address"],
    },
    "heir_of": {
        "variants": [
            ("tail", "the subject's heir (mirasçı)", "mirasçısı"),
            ("head", "the deceased person (muris) whose heir the subject is", "muris (miras bırakan)"),
        ],
        "title": None,
        "own": ["national_id_number", "phone_number"],
    },
    "tenant_of": {
        "variants": [
            ("tail", "the tenant who rents from the subject (the subject is the landlord)", "kiracısı"),
            ("head", "the landlord from whom the subject rents", "kiraya veren (ev sahibi)"),
        ],
        "title": None,
        "own": ["national_id_number", "phone_number", "email_address"],
    },
    "authorised_signatory_of": {
        "variants": [
            (
                None,
                "the authorised signatory who signs on behalf of the company named in the document",
                "şirket imza yetkilisi",
            )
        ],
        "title": None,
        "own": ["phone_number", "email_address"],
        "object": "company_name",
    },
    "owner_of_company": {
        "variants": [(None, "the owner / partner of the company named in the document", "şirket sahibi / ortağı")],
        "title": None,
        "own": ["phone_number", "email_address"],
        "object": "company_name",
    },
    "witness_of": {
        "variants": [(None, "a witness to the notarial act / court proceeding, unrelated to the subject", "tanık")],
        "title": None,
        "own": ["national_id_number", "phone_number"],
        "object": "record",
    },
}
RECORD_ANCHOR = {
    "legal_case_number": ("party_in_case", "head"),
    "notary_record_number": ("notary_act_of", "tail"),
}  # record <-> subject


def add_role_person(b, role, rels_meta, usage):
    """[e08] One secondary person in a role relation with the subject (or with the subject's company / record).
    Returns a persons-list entry (kind 'role') or None."""
    spec = ROLE_SPECS[role]
    subject_side, role_en, role_tr = random.choice(spec["variants"])
    p = b.Person(sex=random.choice("FM"), strict=True)
    title = spec["title"][0] if spec["title"] and random.random() < spec["title"][1] else None
    name = f"{title} {p.first} {p.last}" if title else f"{p.first} {p.last}"
    pid = b.ent("full_name", name)
    usage["full_name"] += 1
    ents, negatives = [pid], []
    subject_vals = {v for (n_, v), i_ in b.entities.items()}
    if subject_side == "head":
        b.add_rel(b.pid, role, pid)
    elif subject_side == "tail":
        b.add_rel(pid, role, b.pid)
    elif spec["object"] == "company_name":
        cid = next((i for (n_, v), i in b.entities.items() if n_ == "company_name"), None)
        if cid is None:
            cid = b.ent("company_name", b.ks.surface(b.meta, "company_name"))
            usage["company_name"] += 1
        b.add_rel(pid, role, cid)
        negatives.append(
            {"head": b.pid, "relation": role, "tail": cid}
        )  # the subject is NOT the company's signatory/owner
    else:  # record: the subject's case / notarial act, the secondary a witness to it
        node = next((n_ for (n_, v), i in b.entities.items() if n_ in RECORD_ANCHOR), None) or random.choice(
            list(RECORD_ANCHOR)
        )
        rid = next((i for (n_, v), i in b.entities.items() if n_ == node), None)
        if rid is None:
            rid = b.ent(node, b.value(node))
            usage[node] += 1
            arel, aside = RECORD_ANCHOR[node]
            b.add_rel(*((b.pid, arel, rid) if aside == "head" else (rid, arel, b.pid)))
        b.add_rel(pid, role, rid)
        negatives.append({"head": b.pid, "relation": role, "tail": rid})  # the subject is a party, not a witness
    own = [n for n in spec["own"] if n != "bar_registry_number" or title == "Av."]
    for node in random.sample(own, k=min(len(own), random.choice([1, 1, 2]))):
        cands = [
            r
            for r in rels_meta
            if (node in r["head_types"] and "full_name" in r["tail_types"])
            or (node in r["tail_types"] and "full_name" in r["head_types"])
        ]
        if not cands:
            continue
        rel = random.choice(cands)
        val = _person_value(b, p, node)
        for _ in range(20):
            if val not in subject_vals:
                break
            val = _person_value(b, p, node)
        if val in subject_vals:
            continue
        eid = b.ent(node, val)
        ents.append(eid)
        usage[node] += 1
        if node in rel["head_types"]:
            b.add_rel(eid, rel["id"], pid)
            negatives.append({"head": eid, "relation": rel["id"], "tail": b.pid})
        else:
            b.add_rel(pid, rel["id"], eid)
            negatives.append({"head": b.pid, "relation": rel["id"], "tail": eid})
    return {
        "id": pid,
        "name": name,
        "sex": p.sex,
        "kind": "role",
        "role": role,
        "role_en": role_en,
        "role_tr": role_tr,
        "entities": ents,
        "negatives": negatives,
    }


def draw_persons(b, rels_meta, usage, shares_persons=SHARES_PERSONS, shares_kind=SHARES_KIND):
    n = _wchoice(shares_persons)
    if n >= 3:
        n = random.choice([3, 3, 4])
    persons = [
        {
            "id": b.pid,
            "name": b.p.full,
            "kind": "subject",
            "role": "subject",
            "role_en": "the subject of the document",
            "entities": [],
            "negatives": [],
        }
    ]
    for _ in range(n - 1):
        entry = add_secondary(b, _wchoice(shares_kind), rels_meta, usage)
        if entry:
            persons.append(entry)
    return persons


def prompt_block(persons):
    lines = [
        f"- {p['name']}: {p['role_en']}"
        + (
            f" — {'female' if p['sex'] == 'F' else 'male'}; Turkish role word: {p['role_tr']}"
            if p.get("sex") and p.get("role_tr")
            else ""
        )
        for p in persons
    ]
    return (
        "NAMED PERSONS. Exactly these people are named in the document, each in the stated role, and EACH OF THEM MUST APPEAR at least once "
        "(tagged as [full name]full_name); anyone else is referred to by role only:\n"
        + "\n".join(lines)
        + "\nTheir listed facts remain candidates like the others, but every fact stays with the person it is listed for: a value listed for a "
        "secondary person must never be presented as the subject's, and vice versa."
    )
