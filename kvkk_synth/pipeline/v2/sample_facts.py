#!/usr/bin/env python3
"""v2 fact sampler — scenario-coherent bundles on top of the v1 machinery.

Changes vs v1 (FIX A):
  * relations are drawn from a primary scenario group plus compatible neighbours (scenarios.py),
    never from all 75 at random; the document genre comes from the scenario's pool
  * attire values are sexed; subject name parts are emitted as optional bindable entities
  * fact templates lose their parenthetical commentary
Usage: python sample_facts.py -n 45 --seed 7 --out bundles.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(PARENT, "..", "taxonomy"))  # [pkg]
import kvkk_sampler as ks  # noqa: E402
import sample_facts as v1  # noqa: E402  (pools, Person, Bundle, validators, check_bundle)
from scenarios import COMPAT, GENRES, NEIGHBOUR_ALLOW, WEIGHTS  # noqa: E402

ATTIRE_F = ["Başörtülü", "Çarşaflı", "Peçeli"]
ATTIRE_M = ["Sarıklı ve cübbeli", "Takkeli", "Sakallı ve cübbeli"]

# ---------------------------------------------------------------- name pools (names/*.csv)
# 2,224 first names + 1,222 surnames, each with a sampling probability `weight` (frequency-tempered
# and normalised per file, so common names dominate less).
# A full name is never reused within a run.
AMBIGUOUS_FIRST_NAMES = {"Kadın", "Erkek", "Bayan"}  # [e08] "Kadın Düzgün" reads as "woman Düzgün" in a PII document
NAMES_DIR = os.path.join(HERE, "names")
POOL = {"F": [], "M": [], "L": []}
WEIGHT = {"F": [], "M": [], "L": []}
POOL_STRICT = {"F": [], "M": []}
WEIGHT_STRICT = {"F": [], "M": []}  # [sexfix] names labelled F or M only, never U
if os.path.exists(os.path.join(NAMES_DIR, "first_names.csv")):
    import csv

    with open(os.path.join(NAMES_DIR, "first_names.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for sex in ("F", "M"):
                if row["sex"] in (sex, "U"):
                    POOL[sex].append(row["name"])
                    WEIGHT[sex].append(float(row["weight"]))
                if row["sex"] == sex:
                    POOL_STRICT[sex].append(row["name"])
                    WEIGHT_STRICT[sex].append(float(row["weight"]))
    with open(os.path.join(NAMES_DIR, "surnames.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            POOL["L"].append(row["name"])
            WEIGHT["L"].append(float(row["weight"]))
    for _s in ("F", "M"):  # [e08] common nouns that also exist as given names
        keep = [i for i, n in enumerate(POOL[_s]) if n not in AMBIGUOUS_FIRST_NAMES]
        POOL[_s] = [POOL[_s][i] for i in keep]
        WEIGHT[_s] = [WEIGHT[_s][i] for i in keep]
        keep = [i for i, n in enumerate(POOL_STRICT[_s]) if n not in AMBIGUOUS_FIRST_NAMES]
        POOL_STRICT[_s] = [POOL_STRICT[_s][i] for i in keep]
        WEIGHT_STRICT[_s] = [WEIGHT_STRICT[_s][i] for i in keep]
    v1.FIRST_F[:], v1.FIRST_M[:], v1.LAST[:] = POOL["F"], POOL["M"], POOL["L"]  # maiden names, guards, faker pools
    ks.FIRST[:] = POOL["F"] + POOL["M"]
    ks.LAST[:] = POOL["L"]

v1.ASCII.update(
    str.maketrans("âÂîÎûÛ", "aAiIuU")
)  # pool names with circumflex (Alâaddin) must fold for e-mails/usernames

USED_FULL_NAMES = set()


class PersonV2(v1.Person):
    """Frequency-tempered sampling from the big pools; no full name is reused within a run."""

    def __init__(self, sex=None, full_parts=False, strict=False):
        if not POOL["L"]:
            super().__init__(sex=sex, full_parts=full_parts)
            return
        self.sex = sex or random.choice("FM")
        pool, weight = (POOL_STRICT, WEIGHT_STRICT) if strict else (POOL, WEIGHT)  # [sexfix]
        for _ in range(50):
            self.first = random.choices(pool[self.sex], weight[self.sex])[0]
            self.last = random.choices(POOL["L"], WEIGHT["L"])[0]
            if (self.first, self.last) not in USED_FULL_NAMES:
                break
        USED_FULL_NAMES.add((self.first, self.last))
        self.middle = None
        if full_parts or random.random() < 0.2:
            self.middle = random.choices(pool[self.sex], weight[self.sex])[0]
            if self.middle == self.first:
                self.middle = None if not full_parts else random.choice(POOL[self.sex])
        self.title = random.choice(ks.TITLES) if full_parts or random.random() < 0.2 else None
        self.full = " ".join(x for x in (self.title, self.first, self.middle, self.last) if x)


v1.Person = PersonV2  # Bundle.__init__ and kinship instantiate() look Person up in v1 at call time

# taxonomy templates that encode a story (a rejection, a discrimination claim) are replaced by
# neutral record-style hints so the generator does not build the document around that story
HINT_OVERRIDES = {
    "attire_of": ["{tail} — kılık kıyafet bilgisi: {head}", "Kayıtlarda {tail} için {head} notu bulunmaktadır."],
    "sexual_life_of": [
        "{tail} ile ilgili kayıtlı beyan: {head}",
        "Dosyada {tail} hakkında {head} bilgisi yer almaktadır.",
    ],
}


def clean_fact(fact):
    fact = re.sub(r"\s*\([^)]*\)", "", fact)
    return re.sub(r"\s{2,}", " ", fact).strip()


class BundleV2(v1.Bundle):
    def __init__(self, ents_meta, rels_meta, bid, scenario, full_parts=False):
        super().__init__(ents_meta, rels_meta, bid, full_parts=full_parts)
        self.scenario = scenario
        self.optional = set()

    def value(self, node):
        if node == "attire_info":
            return self.ctx.setdefault(node, random.choice(ATTIRE_F if self.p.sex == "F" else ATTIRE_M))
        return super().value(node)

    def add_name_parts(self):
        """Subject's first/last name as optional entities: surname-only mentions bind, but the
        model is not required to produce them."""
        for node, val in (("first_name", self.p.first), ("last_name", self.p.last)):
            if (node, val) not in self.entities:
                self.optional.add(self.ent(node, val))

    def to_json(self, ents_meta, rels_meta, genre, fmt, neighbours):
        js = super().to_json(ents_meta, rels_meta)
        for e in js["entities"]:
            e["optional"] = e["id"] in self.optional
        rmap = {r["id"]: r for r in rels_meta}
        val = {e["id"]: e["value"] for e in js["entities"]}
        for r in js["relations"]:
            # hint the generator with the taxonomy's natural-Turkish templates (proper diacritics),
            # not the ASCII-folded descriptions it was copying verbatim
            tpl = (
                HINT_OVERRIDES.get(r["relation"])
                or [t for t in rmap[r["relation"]].get("templates_tr", []) if "{head}" in t and "{tail}" in t]
                or rmap[r["relation"]].get("templates_tr", [])
            )
            if tpl:
                r["fact_tr"] = random.choice(tpl).replace("{head}", val[r["head"]]).replace("{tail}", val[r["tail"]])
            else:
                r["fact_tr"] = clean_fact(r["fact_tr"])
        js.update(scenario=self.scenario, neighbours=neighbours, genre=genre, document_format=fmt)
        return js


def least_used(pool, usage, k, exclude=()):
    pool = [r for r in pool if r["id"] not in exclude]
    random.shuffle(pool)
    pool.sort(key=lambda r: usage[r["id"]])
    return pool[:k]


def pick_relations(primary, by_group, usage, vary=False):
    k_primary = min(len(by_group[primary]), random.choice([1, 1, 2, 3, 4, 5]) if vary else random.choice([2, 2, 3]))
    chosen = least_used(by_group[primary], usage, k_primary)
    neigh_pool, neighbours = [], []
    for g in COMPAT[primary]:
        allowed = NEIGHBOUR_ALLOW.get((primary, g))
        for r in by_group[g]:
            if allowed is None or r["id"] in allowed:
                neigh_pool.append(r)
    k_neigh = random.choice([0, 1, 1, 2, 3]) if vary else random.choice([1, 1, 2])
    for r in least_used(neigh_pool, usage, k_neigh, exclude={c["id"] for c in chosen}):
        chosen.append(r)
        neighbours.append(r["_group"])
    for r in chosen:
        usage[r["id"]] += 1
    return chosen, sorted(set(neighbours))


def check_v2(js, ents_meta, rels_meta):
    core = dict(js, entities=[e for e in js["entities"] if not e.get("optional")])
    problems = v1.check_bundle(core, ents_meta, rels_meta)
    subj = next(e["value"] for e in js["entities"] if e["id"] == js["subject_id"])
    for e in js["entities"]:
        if e.get("optional") and e["value"] not in subj:
            problems.append(f"optional name part {e['value']} not in subject {subj}")
    if js["subject_sex"] == "F" and any(e["value"] in ATTIRE_M for e in js["entities"]):
        problems.append("male attire on female subject")
    if js["subject_sex"] == "M" and any(e["value"] in ATTIRE_F for e in js["entities"]):
        problems.append("female attire on male subject")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="bundles.jsonl")
    ap.add_argument(
        "--vary", action="store_true", help="wider spread of facts per bundle (1–8) for optional-facts generation"
    )
    a = ap.parse_args()
    random.seed(a.seed)

    ents_meta, rels_meta = ks.load_taxonomy()
    by_group = defaultdict(list)
    for r in rels_meta:
        by_group[r["_group"]].append(r)
    usage, node_usage = Counter(), Counter()
    groups, weights = list(WEIGHTS), [WEIGHTS[g] for g in WEIGHTS]
    bundles, problems_all = [], []
    for bid in range(1, a.n + 1):
        primary = random.choices(groups, weights)[0]
        chosen, neighbours = pick_relations(primary, by_group, usage, vary=a.vary)
        b = BundleV2(ents_meta, rels_meta, bid, primary, full_parts=any(r["id"] == "part_of_name" for r in chosen))
        for r in chosen:
            b.instantiate(r, node_usage)
        assert b.close(node_usage), f"bundle {bid}: could not connect graph"
        b.add_name_parts()
        genre, fmt = random.choice(GENRES[primary])
        js = b.to_json(ents_meta, rels_meta, genre, fmt, neighbours)
        probs = check_v2(js, ents_meta, rels_meta)
        if probs:
            problems_all.append({"bundle_id": bid, "problems": probs})
        bundles.append(js)

    with open(a.out, "w", encoding="utf-8") as f:
        for b in bundles:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    rel_cov = Counter(r["relation"] for b in bundles for r in b["relations"])
    report = {
        "bundles": len(bundles),
        "problems": problems_all,
        "scenarios": dict(Counter(b["scenario"] for b in bundles)),
        "relations_covered": f"{len(rel_cov)}/{len(rels_meta)}",
        "nodes_covered": f"{len({e['label'] for b in bundles for e in b['entities']})}/{len(ents_meta)}",
        "relations_per_bundle": dict(Counter(len(b["relations"]) for b in bundles)),
        "groups_per_bundle": dict(Counter(len(b["groups"]) for b in bundles)),
        "madde6_share": round(sum(b["madde6"] for b in bundles) / len(bundles), 2),
        "distinct_subjects": len({b["entities"][0]["value"] for b in bundles}),
    }
    json.dump(report, open("soundness_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(
        f"{len(bundles)} bundles -> {a.out} | problems {len(problems_all)} | relations {report['relations_covered']} | "
        f"nodes {report['nodes_covered']} | groups/bundle {report['groups_per_bundle']} | madde6 share {report['madde6_share']}"
    )
    print("scenarios:", report["scenarios"])


if __name__ == "__main__":
    main()
