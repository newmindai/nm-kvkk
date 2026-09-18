#!/usr/bin/env python3
"""e04 sampler: the candidate pool of a bundle is chosen by the brief, not by a random scenario.

For each of the 24 e01 briefs: take the 15 entity types whose e5 cosine to the brief is highest
(scores computed in e03, local multilingual-e5-large-instruct, node text = labels + aliases +
description + examples), always add full_name and national_id_number, and instantiate every type
with the relation that anchors it to the subject (or, for types with no person relation, the
relation that ties it to its carrier entity, which the connectivity closure then anchors).
Values come from the taxonomy's checksum recipes as in e01. The brief is attached to the bundle
so gen_facts.py uses it unchanged.
Run from code/v2:  python ../pool_sampler.py --scores ../../../2026-09-08-e03-brief-entity-relevance/scores-e5.json \
                    --briefs ../../../2026-09-08-e01-muse13-v22-random/bundles.jsonl --k 15 --out ../../bundles.jsonl
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
import sample_facts as v1  # noqa: E402  (v1 machinery: person_side, ANCHOR, check_bundle)

spec = importlib.util.spec_from_file_location("sf_v2", os.path.join(HERE, "v2", "sample_facts.py"))
sf2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf2)  # PersonV2, BundleV2, check_v2, name pools
import geoloc  # [e05] tier split + real geography
import persons as pers  # [e06] persons axis
import tiers

ALWAYS = ["full_name", "national_id_number"]


class BundleV3(sf2.BundleV2):
    """[e05] tier-C values from one real Locale per bundle."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.loc = geoloc.Locale(random)
        self.ks, self.fold, self.Person = ks, v1.fold, v1.Person  # [e06] for secondary persons

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
        if node == "license_plate":  # [e05] plate province = locale province
            import re as _re

            raw = super().value(node)
            v = _re.sub(r"^\d{2}", self.loc.plate, raw) if raw[:2].isdigit() else raw
            self.ctx[node] = v
            return v
        return super().value(node)


NAME_PARTS = {"middle_name", "title_honorific"}

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--briefs", required=True, help="e01 bundles.jsonl (brief per bundle_id)")
ap.add_argument("--k", type=int, default=15)
ap.add_argument("--seed", type=int, default=8)
ap.add_argument("--out", default="../../bundles.jsonl")
ap.add_argument("--persons", default="60,28,12", help="[e06] shares of 1 / 2 / 3+ named persons")
a = ap.parse_args()
random.seed(a.seed)

ents_meta, rels_meta = ks.load_taxonomy()
scores = json.load(open(a.scores, encoding="utf-8"))["scores"]
briefs = {b["bundle_id"]: b["brief"] for b in (json.loads(l) for l in open(a.briefs, encoding="utf-8"))}
rmap = {r["id"]: r for r in rels_meta}


def relations_for(node):
    """Relations that put `node` next to the subject directly, else any relation carrying it."""
    direct, other = [], []
    for r in rels_meta:
        side = v1.person_side(r)
        if node in r["head_types"] and side == "tail" or node in r["tail_types"] and side == "head":
            direct.append(r)
        elif node in r["head_types"] + r["tail_types"] and side is None:
            other.append(r)
    return direct or other


def add_node(b, node, usage):
    rels = relations_for(node)
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
    except Exception as e:
        b.entities, b.order, b.relations, b.ctx = snap
        print(f"  bundle {b.bid}: skip {node} ({e})", file=sys.stderr)
        return False


usage, rel_usage = Counter(), Counter()
bundles, problems_all, pool_log = [], [], {}
for bid in sorted(briefs):
    s = scores[str(bid)]
    ranked = [n for n, _ in sorted(s.items(), key=lambda x: -x[1])]
    pool = ALWAYS + [n for n in ranked if n not in ALWAYS][: a.k]
    b = BundleV3(ents_meta, rels_meta, bid, scenario="pool", full_parts=bool(NAME_PARTS & set(pool)))
    added = []
    for node in pool:
        if node == "full_name":
            added.append(node)
            continue
        if node in tiers.TIER_B:  # [e05] writer-owned: offered as a kind, not as a value
            continue
        if add_node(b, node, usage):
            added.append(node)
    shares = [(1, int(a.persons.split(",")[0])), (2, int(a.persons.split(",")[1])), (3, int(a.persons.split(",")[2]))]
    persons_list = pers.draw_persons(b, rels_meta, usage, shares_persons=shares)  # [e06]
    b.add_name_parts()
    brief = briefs[bid]
    js = b.to_json(ents_meta, rels_meta, brief["document_type"], brief["document_format"], [])
    js["brief"] = brief
    js["pool"] = [
        {
            "node": n,
            "cosine": round(s.get(n, 0.0), 4),
            "instantiated": n in added,
            "tier": "B" if n in tiers.TIER_B else ("C" if n in tiers.TIER_C else "A"),
        }
        for n in pool
    ]  # [e05]
    js["writer_allowed"] = sorted(tiers.TIER_B)
    js["persons"] = persons_list  # [e06]
    js["relations_negative"] = [n for p in persons_list for n in p["negatives"]]
    js["locale"] = {"city": b.loc.city, "town": b.loc.town, "neighbourhood": b.loc.nb, "zip": b.loc.zip}
    probs = sf2.check_v2(js, ents_meta, rels_meta)
    if len(persons_list) > 1:  # [e06] connectivity is per person, not per subject
        probs = [p for p in probs if p != "graph not connected to subject"]
        from collections import defaultdict, deque

        adj = defaultdict(set)
        for r in js["relations"]:
            adj[r["head"]].add(r["tail"])
            adj[r["tail"]].add(r["head"])
        seen = set()
        anchors = {p["id"] for p in persons_list}
        for start in anchors:
            q = deque([start])
            seen.add(start)
            while q:
                for nxt in adj[q.popleft()]:
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
        stray = [e["id"] for e in js["entities"] if e["id"] not in seen and not e.get("optional")]
        if stray:
            probs.append(f"entities anchored to no person: {stray}")
    if probs:
        problems_all.append({"bundle_id": bid, "problems": probs})
    for r in js["relations"]:
        rel_usage[r["relation"]] += 1
    bundles.append(js)
    pool_log[bid] = added

with open(a.out, "w", encoding="utf-8") as f:
    for b in bundles:
        f.write(json.dumps(b, ensure_ascii=False) + "\n")
nodes_cov = Counter(e["label"] for b in bundles for e in b["entities"])
report = {
    "bundles": len(bundles),
    "k": a.k,
    "problems": problems_all,
    "persons_per_bundle": dict(Counter(len(b["persons"]) for b in bundles)),
    "secondary_kinds": dict(Counter(p["kind"] for b in bundles for p in b["persons"] if p["kind"] != "subject")),
    "negatives": sum(len(b["relations_negative"]) for b in bundles),
    "facts_per_bundle": dict(Counter(len(b["relations"]) for b in bundles)),
    "entities_per_bundle": dict(Counter(len(b["entities"]) for b in bundles)),
    "distinct_nodes": len(nodes_cov),
    "distinct_relations": len(rel_usage),
    "node_counts": dict(nodes_cov.most_common()),
    "madde6_share": round(sum(b["madde6"] for b in bundles) / len(bundles), 2),
}
json.dump(
    report,
    open(os.path.join(os.path.dirname(a.out), "soundness_report.json"), "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
print(
    f"{len(bundles)} bundles -> {a.out} | problems {len(problems_all)} | facts/bundle {report['facts_per_bundle']} | "
    f"distinct nodes {report['distinct_nodes']}/118, relations {report['distinct_relations']}/105 | madde6 {report['madde6_share']}"
)
