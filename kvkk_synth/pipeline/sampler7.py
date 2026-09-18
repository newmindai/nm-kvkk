#!/usr/bin/env python3
"""e07 sampler: 1,000 bundles from the brief catalogue with density profiles, centred embedding pools,
three value tiers, real geography and a brief-scored persons axis.

Per brief (briefs1000.jsonl, with a pre-drawn density profile):
  zero    -> no bundle (a zero-PII document); the writer gets the no-personal-data prompt
  sparse  -> fact target 1–2,  pool = 6 centred types + name (+ TCKN with p=.5)
  normal  -> fact target 2–4,  pool = 10 + name + TCKN
  dense   -> fact target 5–10, pool = 15 + name + TCKN, form/record layout requested
Centring: each node's cosine minus its mean over the 1,000 briefs (hub types stop scoring everywhere).
Tier B types in the pool are offered as writer-owned kinds, not instantiated; tier C from the Locale.
Persons: 1 / 2 / 3+ with shares 60/28/12, but the role of a secondary person is drawn only among roles
whose centred role-cosine to the brief is above a threshold (scores-roles.json); if none qualifies the
document stays single-person.
Run from code/:  python sampler7.py --out ../bundles.jsonl
"""

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "v2"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "taxonomy"))  # [pkg]
import importlib.util

import kvkk_sampler as ks  # noqa: E402
import sample_facts as v1  # noqa: E402

spec = importlib.util.spec_from_file_location("sf_v2", os.path.join(HERE, "v2", "sample_facts.py"))
sf2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf2)
import geoloc
import persons as pers  # noqa: E402
import tiers

PROFILE = {
    "sparse": {"k": 6, "target": (1, 2), "tckn_p": 0.5},
    "normal": {"k": 10, "target": (2, 4), "tckn_p": 1.0},
    "dense": {"k": 15, "target": (5, 10), "tckn_p": 1.0},
}
SHARES_PERSONS = [(1, 60), (2, 28), (3, 12)]
ROLE_KIND = {
    **{r: "related" for r in pers.KIN_RELS},
    **{r[0]: "counterpart" for r in pers.COUNTERPARTS},
    **{r[0]: "bystander" for r in pers.BYSTANDERS},
}
ROLE_THRESHOLD = 0.03  # centred cosine; roles below never appear
NAME_PARTS = {"middle_name", "title_honorific"}


class BundleV3(sf2.BundleV2):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.loc = geoloc.Locale(random)
        self.ks, self.fold, self.Person = ks, v1.fold, v1.Person

    def _address(self):
        if "full_address" not in self.ctx:
            L = self.loc
            self.ctx.update(
                street_line=L.street_line(),
                district=L.town,
                city=L.city,
                postal_code=L.zip,
                full_address=L.full_address(with_zip=random.random() < 0.6),
            )
        return self.ctx["full_address"]

    def value(self, node):
        if node == "place_of_birth":
            return self.ctx.setdefault(node, self.loc.place_of_birth())
        if node == "registered_place":
            return self.ctx.setdefault(node, self.loc.registered_place())
        if node == "tracked_location":
            return self.ctx.setdefault(node, self.loc.tracked_location())
        if node == "license_plate":
            import re as _re

            raw = super().value(node)
            v = _re.sub(r"^\d{2}", self.loc.plate, raw) if raw[:2].isdigit() else raw
            self.ctx[node] = v
            return v
        return super().value(node)


def relations_for(node, rels_meta):
    direct, other = [], []
    for r in rels_meta:
        side = v1.person_side(r)
        if node in r["head_types"] and side == "tail" or node in r["tail_types"] and side == "head":
            direct.append(r)
        elif node in r["head_types"] + r["tail_types"] and side is None:
            other.append(r)
    return direct or other


def add_node(b, node, rels_meta, usage):
    rels = relations_for(node, rels_meta)
    if not rels or node == "full_name":
        return False
    rel = random.choice(rels)
    snap = (copy.deepcopy(b.entities), list(b.order), list(b.relations), dict(b.ctx))
    try:
        val = b.value(node)
        if val is None:
            raise ValueError("no value")
        side = v1.person_side(rel)
        if side == "tail":
            b.add_rel(b.ent(node, val), rel["id"], b.pid)
        elif side == "head":
            b.add_rel(b.pid, rel["id"], b.ent(node, val))
        elif node in rel["head_types"]:
            tt = b.pick_type(rel["tail_types"], usage)
            b.add_rel(b.ent(node, val), rel["id"], b.ent(tt, b.value(tt)))
        else:
            ht = b.pick_type(rel["head_types"], usage)
            b.add_rel(b.ent(ht, b.value(ht)), rel["id"], b.ent(node, val))
        if not b.close(usage):
            raise ValueError("cannot anchor")
        usage[node] += 1
        return True
    except Exception:
        b.entities, b.order, b.relations, b.ctx = snap
        return False


def centred(scores):
    """node -> mean cosine over all briefs; returns a function brief_id -> {node: centred score}"""
    nodes = list(next(iter(scores.values())).keys())
    mean = {n: sum(s[n] for s in scores.values()) / len(scores) for n in nodes}
    return lambda bid: {n: scores[bid][n] - mean[n] for n in nodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefs", default="briefs.jsonl")  # [pkg] run-folder relative
    ap.add_argument("--scores", default="scores-e5.json")  # [pkg]
    ap.add_argument("--roles", default="scores-roles.json")  # [pkg]
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--out", default="bundles.jsonl")  # [pkg]
    a = ap.parse_args()
    random.seed(a.seed)

    ents_meta, rels_meta = ks.load_taxonomy()
    briefs = [json.loads(l) for l in open(a.briefs, encoding="utf-8")]
    node_scores = json.load(open(a.scores, encoding="utf-8"))["scores"]
    uid2bid = {str(b_["uid"]): str(b_["bundle_id"]) for b_ in briefs}  # embed_score keys by uid
    node_scores = {uid2bid.get(k, k): v for k, v in node_scores.items()}
    role_scores = json.load(open(a.roles, encoding="utf-8"))["scores"]
    cscore = centred(node_scores)
    crole = centred(role_scores)
    usage, node_usage = Counter(), Counter()
    out, stats = [], Counter()
    for br in briefs:
        bid = br["bundle_id"]
        prof = br["profile"]
        if prof == "zero":
            out.append(
                {
                    "bundle_id": bid,
                    "subject_id": None,
                    "subject_sex": None,
                    "entities": [],
                    "relations": [],
                    "groups": [],
                    "madde6": False,
                    "scenario": "zero",
                    "genre": br["document_type"],
                    "document_format": br["document_format"],
                    "profile": "zero",
                    "fact_target": 0,
                    "brief": {
                        "domain": br["domain"],
                        "document_type": br["document_type"],
                        "description": br["description"],
                        "document_format": br["document_format"],
                    },
                    "pool": [],
                    "persons": [],
                    "relations_negative": [],
                    "writer_allowed": [],
                }
            )
            stats["zero"] += 1
            continue
        P = PROFILE[prof]
        cs = cscore(str(bid))
        # least-used tie-break: among the top 2k, prefer types with low usage so coverage spreads (§4.4)
        ranked = sorted(cs, key=lambda n: -cs[n])
        cand = [n for n in ranked if n not in ("full_name", "national_id_number")][: 2 * P["k"]]
        cand.sort(key=lambda n: (node_usage[n] > 40, -cs[n]))  # types above the 40-doc floor go last
        pool = ["full_name"] + (["national_id_number"] if random.random() < P["tckn_p"] else []) + cand[: P["k"]]
        b = BundleV3(ents_meta, rels_meta, bid, scenario="pool", full_parts=bool(NAME_PARTS & set(pool)))
        added = ["full_name"]
        for node in pool[1:]:
            if node in tiers.TIER_B:
                continue
            if add_node(b, node, rels_meta, usage):
                added.append(node)
                node_usage[node] += 1
        # persons: shares as prior, role only if the brief supports it
        n = pers._wchoice(SHARES_PERSONS)
        if n >= 3:
            n = random.choice([3, 3, 4])
        rs = crole(str(bid))
        eligible = [r for r in rs if rs[r] >= ROLE_THRESHOLD]
        persons_list = [
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
            if not eligible:
                break
            weights = [max(rs[r], 1e-6) for r in eligible]
            role = random.choices(eligible, weights)[0]
            entry = pers.add_secondary(b, ROLE_KIND[role], rels_meta, usage, role=role)
            if entry:
                persons_list.append(entry)
                eligible = [r for r in eligible if r != role]
        b.add_name_parts()
        js = b.to_json(ents_meta, rels_meta, br["document_type"], br["document_format"], [])
        js.update(
            brief={
                "domain": br["domain"],
                "document_type": br["document_type"],
                "description": br["description"],
                "document_format": br["document_format"],
            },
            profile=prof,
            fact_target=random.randint(*P["target"]),
            pool=[
                {
                    "node": n_,
                    "cosine": round(node_scores[str(bid)].get(n_, 0.0), 4),
                    "centred": round(cs.get(n_, 0.0), 4),
                    "instantiated": n_ in added,
                    "tier": "B" if n_ in tiers.TIER_B else ("C" if n_ in tiers.TIER_C else "A"),
                }
                for n_ in pool
            ],
            writer_allowed=sorted(tiers.TIER_B),
            persons=persons_list,
            relations_negative=[x for p in persons_list for x in p["negatives"]],
            locale={"city": b.loc.city, "town": b.loc.town, "neighbourhood": b.loc.nb, "zip": b.loc.zip},
            role_scores={r: round(rs[r], 3) for r in eligible + [p["role"] for p in persons_list[1:]]},
        )
        js["fact_target"] = min(js["fact_target"], len(js["relations"])) or 1
        out.append(js)
        stats[prof] += 1
        stats[f"persons={len(persons_list)}"] += 1
        for p in persons_list[1:]:
            stats[f"kind={p['kind']}"] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for js in out:
            f.write(json.dumps(js, ensure_ascii=False) + "\n")
    nodes_cov = Counter(e["label"] for js in out for e in js["entities"])
    report = {
        "bundles": len(out),
        "profiles": {k: v for k, v in stats.items() if "=" not in k},
        "persons": {k: v for k, v in stats.items() if k.startswith("persons")},
        "secondary_kinds": {k: v for k, v in stats.items() if k.startswith("kind")},
        "negatives": sum(len(js["relations_negative"]) for js in out),
        "distinct_nodes_instantiated": len(nodes_cov),
        "node_counts": dict(nodes_cov.most_common()),
        "facts_per_bundle": dict(sorted(Counter(len(js["relations"]) for js in out).items())),
        "tier_b_kinds_offered": sum(1 for js in out for p in js["pool"] if p["tier"] == "B"),
        "tier_c_offered": sum(1 for js in out for p in js["pool"] if p["tier"] == "C" and p["instantiated"]),
    }
    json.dump(report, open("soundness_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "node_counts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
