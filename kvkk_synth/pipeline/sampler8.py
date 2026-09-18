#!/usr/bin/env python3
"""e08 sampler: 5,000 bundles from briefs5000.jsonl (fresh Nemotron briefs, ≤3 uses each), built on sampler7 with the
changes the e07 pipeline review asked for:

  * scores keyed by brief uid (a brief used three times gets its scores three times; e07 keyed by bundle_id and broke on reuse)
  * density profile conditioned on the brief's PII affinity (sum of its top-10 centred cosines + noise, then quantiles
    8 / 22 / 45 / 25 % zero / sparse / normal / dense) instead of a blind draw — e07's coherence loss came from dense
    profiles on briefs that carry no PII by nature
  * pool: at least one tier-A/C node per non-zero pool; a node above the label floor goes to the back of the candidate
    list; a rare node is promoted only where its own centred score is positive
  * relations: least-used relation per node instead of a random one; structural (non-person) relations added next to
    the person relation with p=.35 (card ↔ CVV, IBAN ↔ SWIFT, parcel ↔ address, street ↔ address …); a quota pass that
    lifts every relation type to a floor of 40 bundles where the brief supports it (§4.8 targets)
  * persons: 23 embedding-scored roles (15 + the 8 role relations of taxonomy-ext/16), threshold .02, and a quota pass
    that keeps at least 25 % of the non-zero bundles multi-person; every counterpart / bystander / role person carries
    ≥ 1 own entity, so every multi-person bundle has ≥ 1 negative
  * value generators for the 21 example-only tier-A nodes (values8.py) and a person-aware passport MRZ
Run from code/:  python sampler8.py --out ../bundles.jsonl
"""

import argparse
import copy
import json
import os
import random
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "v2"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "taxonomy"))  # [pkg]
import geoloc
import kinship_claims
import persons as pers
import sampler7 as s7  # noqa: E402  (BundleV3, relations_for, centred)
import tiers  # noqa: E402
import values8

ks, v1 = s7.ks, s7.v1
values8.install(ks)

PROFILE = s7.PROFILE
PROFILE_SHARES = [("zero", 0.08), ("sparse", 0.22), ("normal", 0.45), ("dense", 0.25)]
SHARES_PERSONS = [(1, 60), (2, 28), (3, 12)]
ROLE_KIND = {**s7.ROLE_KIND, **{r: "role" for r in pers.ROLE_SPECS}}
ROLE_THRESHOLD = 0.02
LABEL_FLOOR = 150  # bundles per label above which a label is only offered on rank (3 % of 5,000)
RELATION_FLOOR = 40  # bundles per relation type the quota pass lifts every type to
MULTI_PERSON_SHARE = 0.25
P_STRUCT = 0.35
MAX_RELS_PER_BUNDLE = 14
NAME_PARTS = s7.NAME_PARTS
STRUCT_ALLOWED_TIER_B = {
    "company_name"
}  # writer-owned kinds the sampler may instantiate as the other side of a structural relation
SEX_ONLY = {"maiden_name": "F"}  # [e08] fields that only make sense for one sex of subject
EXCLUSIVE = [
    {
        "bar_registry_number",
        "judge_prosecutor_registry_number",
        "expert_registry_number",
        "doctor_license_number",
    },  # one profession per subject
    {"national_id_number", "foreigner_id_number"},
]  # a TCKN or a foreigner ID, never both
v1.FACT_OVERRIDES["card_component_of"] = (
    "HEAD, TAIL numaralı kartın bileşenidir (güvenlik kodu veya son kullanma tarihi)."
)


class BundleV4(s7.BundleV3):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.loc2 = None  # a second locale: the address of a property (property_at) is not the residence

    def property_address(self):
        if self.loc2 is None:
            self.loc2 = geoloc.Locale(random)
        return self.loc2.full_address(with_zip=random.random() < 0.6)

    def value(self, node):
        if node == "age" and "age" not in self.ctx and "date_of_birth" not in self.ctx:
            # [e08] a randomly drawn age left date_of_birth free to contradict it later (6 documents in e08:
            # "47 yaşında" next to 22.05.1995). Fix the birth date first, then derive the age from it.
            self.value("date_of_birth")
        if node == "passport_mrz":  # [e08] consistent with the person: name, passport number, birth date, sex
            if node not in self.ctx:
                pn = (
                    self.value("passport_number")
                    if "passport_number" in self.meta
                    else ks.surface(self.meta, "passport_number")
                )
                self.ctx[node] = values8.mrz(self.p.last, self.p.first, pn, self.value("date_of_birth"), self.p.sex)
            return self.ctx[node]
        return super().value(node)


def instantiable(node):
    return node not in tiers.TIER_B or node in STRUCT_ALLOWED_TIER_B


def struct_relations(node, rels_meta):
    """Non-person relations that involve `node` and whose other side the sampler can instantiate."""
    out = []
    for r in rels_meta:
        if v1.person_side(r) is not None:
            continue
        if node in r["head_types"] and any(instantiable(t) for t in r["tail_types"]):
            out.append((r, "head"))
        elif node in r["tail_types"] and any(instantiable(t) for t in r["head_types"]):
            out.append((r, "tail"))
    return out


def add_struct(b, node_id, node, rel, side, usage, rel_usage):
    """Add the other side of structural relation `rel` next to entity `node_id` of type `node`."""
    other_types = [t for t in (rel["tail_types"] if side == "head" else rel["head_types"]) if instantiable(t)]
    if "middle_name" in other_types and not b.p.middle:
        other_types.remove("middle_name")
    if not other_types:
        return False
    ot = b.pick_type(other_types, usage)
    val = b.property_address() if (rel["id"] == "property_at" and ot == "full_address") else b.value(ot)
    if val is None:
        return False
    oid = b.ent(ot, val)
    usage[ot] += 1
    b.add_rel(*((node_id, rel["id"], oid) if side == "head" else (oid, rel["id"], node_id)))
    rel_usage[rel["id"]] += 1
    return True


def add_node(b, node, rels_meta, usage, rel_usage, rel=None, p_struct=P_STRUCT):
    """sampler7.add_node with least-used relation choice and an optional structural relation on top."""
    rels = [
        r for r in s7.relations_for(node, rels_meta) if r["id"] not in pers.ROLE_SPECS
    ]  # role relations belong to the persons axis
    if not rels or node == "full_name":
        return False
    if rel is None:
        m = min(rel_usage[r["id"]] for r in rels)
        rel = random.choice([r for r in rels if rel_usage[r["id"]] == m])
    snap = (copy.deepcopy(b.entities), list(b.order), list(b.relations), dict(b.ctx), b.loc2)
    try:
        val = b.value(node)
        if val is None:
            raise ValueError("no value")
        side = v1.person_side(rel)
        nid = b.ent(node, val)
        if side == "tail":
            b.add_rel(nid, rel["id"], b.pid)
        elif side == "head":
            b.add_rel(b.pid, rel["id"], nid)
        elif node in rel["head_types"]:
            tt = b.pick_type([t for t in rel["tail_types"] if instantiable(t)], usage)
            b.add_rel(nid, rel["id"], b.ent(tt, b.value(tt)))
        else:
            ht = b.pick_type([t for t in rel["head_types"] if instantiable(t)], usage)
            b.add_rel(b.ent(ht, b.value(ht)), rel["id"], nid)
        if not b.close(usage):
            raise ValueError("cannot anchor")
        usage[node] += 1
        rel_usage[rel["id"]] += 1
        if side is not None and random.random() < p_struct:
            opts = struct_relations(node, rels_meta)
            if opts:
                m = min(rel_usage[r["id"]] for r, _ in opts)
                r2, side2 = random.choice([o for o in opts if rel_usage[o[0]["id"]] == m])
                add_struct(b, nid, node, r2, side2, usage, rel_usage)
                if not b.close(usage):
                    raise ValueError("cannot anchor structural")
        return True
    except Exception:
        b.entities, b.order, b.relations, b.ctx, b.loc2 = snap
        return False


def assign_profiles(briefs, cscore, rng):
    """Density profile by PII affinity: sum of the top-10 centred cosines, plus noise (half its spread), ranked into quantiles."""
    aff = {}
    for br in briefs:
        cs = cscore(br["uid"])
        aff[br["bundle_id"]] = sum(sorted(cs.values(), reverse=True)[:10])
    sd = statistics.pstdev(aff.values())
    noisy = {bid: a + rng.gauss(0, 0.5 * sd) for bid, a in aff.items()}
    order = sorted(noisy, key=noisy.get)
    prof, i = {}, 0
    for name, share in PROFILE_SHARES:
        n = round(len(order) * share) if name != "dense" else len(order) - i
        for bid in order[i : i + n]:
            prof[bid] = name
        i += n
    return prof, aff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefs", default="briefs.jsonl")  # [pkg] run-folder relative
    ap.add_argument("--scores", default="scores-e5.json")  # [pkg]
    ap.add_argument("--roles", default="scores-roles.json")  # [pkg]
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="bundles.jsonl")  # [pkg]
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    random.seed(a.seed)

    ents_meta, rels_meta = ks.load_taxonomy()
    rmap = {r["id"]: r for r in rels_meta}
    briefs = [json.loads(l) for l in open(a.briefs, encoding="utf-8")]
    if a.limit:
        briefs = briefs[: a.limit]
    node_scores = json.load(open(a.scores, encoding="utf-8"))["scores"]  # keyed by uid
    role_scores = json.load(open(a.roles, encoding="utf-8"))["scores"]  # keyed by uid
    cscore, crole = s7.centred(node_scores), s7.centred(role_scores)
    profiles, affinity = assign_profiles(briefs, cscore, random.Random(a.seed + 1))
    usage, node_usage, rel_usage = Counter(), Counter(), Counter()
    built, out_zero, stats = [], {}, Counter()

    # ---- pass 1: pools, facts, persons ------------------------------------------------------------------------------
    for br in briefs:
        bid, uid = br["bundle_id"], br["uid"]
        prof = profiles[bid]
        brief = {
            "domain": br["domain"],
            "document_type": br["document_type"],
            "description": br["description"],
            "document_format": br["document_format"],
        }
        if prof == "zero":
            out_zero[bid] = {
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
                "brief": brief,
                "brief_uid": uid,
                "affinity": round(affinity[bid], 4),
                "pool": [],
                "persons": [],
                "relations_negative": [],
                "writer_allowed": [],
            }
            stats["zero"] += 1
            continue
        P = PROFILE[prof]
        cs = cscore(uid)
        ranked = sorted(cs, key=lambda n: -cs[n])
        cand = [n for n in ranked if n not in ("full_name", "national_id_number")][: 2 * P["k"]]
        # a label above the floor goes last; a rare label is promoted only where its own centred score is positive
        cand.sort(
            key=lambda n: (
                node_usage[n] > LABEL_FLOOR,
                -(cs[n] if cs[n] > 0 else -1e9) if node_usage[n] <= LABEL_FLOOR else -cs[n],
            )
        )
        chosen = cand[: P["k"]]
        if not any(n not in tiers.TIER_B for n in chosen):  # ≥ 1 tier-A/C node per pool
            ac = next(
                (n for n in ranked if n not in tiers.TIER_B and n not in ("full_name", "national_id_number")), None
            )
            if ac:
                chosen[-1] = ac
        pool = ["full_name"] + (["national_id_number"] if random.random() < P["tckn_p"] else []) + chosen
        if "foreigner_id_number" in chosen and "national_id_number" in pool:
            pool.remove("foreigner_id_number" if random.random() < 0.5 else "national_id_number")
        b = BundleV4(ents_meta, rels_meta, bid, scenario="pool", full_parts=bool(NAME_PARTS & set(pool)))
        added = ["full_name"]
        for node in pool[1:]:
            if node in tiers.TIER_B:
                continue
            if SEX_ONLY.get(node) not in (None, b.p.sex):  # [e08] "kızlık soyadı" only for a female subject
                continue
            if any(node in g and g & set(added) for g in EXCLUSIVE):
                continue
            if add_node(b, node, rels_meta, usage, rel_usage):
                added.append(node)
                node_usage[node] += 1
        # persons: shares as prior, role only where the brief supports it
        n = pers._wchoice(SHARES_PERSONS)
        if n >= 3:
            n = random.choice([3, 3, 4])
        rs = crole(uid)
        eligible = [r for r in rs if rs[r] >= ROLE_THRESHOLD and r in ROLE_KIND]
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
            role = random.choices(eligible, [max(rs[r], 1e-6) for r in eligible])[0]
            entry = pers.add_secondary(b, ROLE_KIND[role], rels_meta, usage, role=role)
            if entry:
                persons_list.append(entry)
                eligible = [r for r in eligible if r != role]
                rel_usage[role] += 1
        built.append(
            {
                "br": br,
                "b": b,
                "prof": prof,
                "added": added,
                "persons": persons_list,
                "rs": rs,
                "pool": pool,
                "cs": cs,
                "quota": 0,
            }
        )

    # ---- pass 2: relation quota — lift every relation type to RELATION_FLOOR where a brief supports it ----------------
    def rel_counts():
        c = Counter()
        for it in built:
            for _, r, _ in it["b"].relations:
                c[r] += 1
        return c

    counts = rel_counts()
    quota_log = {}
    floor = max(
        1, round(RELATION_FLOOR * len(briefs) / 5000)
    )  # the floor is per 5,000 briefs; a --limit test scales it

    def _all_b(types):
        return all(t in tiers.TIER_B and t not in STRUCT_ALLOWED_TIER_B for t in types if t != "full_name")

    writer_owned = {
        r["id"]
        for r in rels_meta
        if (
            v1.person_side(r) in ("head", "tail")
            and _all_b(r["head_types"] if v1.person_side(r) == "tail" else r["tail_types"])
        )
        or (v1.person_side(r) is None and (_all_b(r["head_types"]) or _all_b(r["tail_types"])))
    } | {"member_of"}  # member_of: memberships are writer-owned; a company is an employer, not a membership
    for rid in sorted(rmap, key=lambda r: counts[r]):
        if rid in writer_owned:
            continue  # tier-B relations come from the writer's own facts
        need = floor - counts[rid]
        if need <= 0:
            continue
        r = rmap[rid]
        side = v1.person_side(r)
        got = 0
        if rid in ROLE_KIND:  # kinship or role relation → a person
            cands = sorted(
                (
                    it
                    for it in built
                    if len(it["persons"]) <= 2 and it["quota"] < 2 and all(p["role"] != rid for p in it["persons"])
                ),
                key=lambda it: -it["rs"].get(rid, -1),
            )
            for it in cands:
                if got >= need:
                    break
                if it["rs"].get(rid, -1) < 0:
                    break  # never on a brief that scores below its mean
                entry = pers.add_secondary(it["b"], ROLE_KIND[rid], rels_meta, usage, role=rid)
                if entry:
                    it["persons"].append(entry)
                    it["quota"] += 1
                    got += 1
        elif side in ("head", "tail"):
            nodes = [
                t
                for t in (r["head_types"] if side == "tail" else r["tail_types"])
                if instantiable(t) and t != "full_name"
            ]
            if not nodes:
                continue
            cands = sorted(
                (it for it in built if it["quota"] < 2 and len(it["b"].relations) < MAX_RELS_PER_BUNDLE),
                key=lambda it: -max(it["cs"].get(t, -1) for t in nodes),
            )
            for it in cands:
                if got >= need:
                    break
                node = max(nodes, key=lambda t: it["cs"].get(t, -1))
                if it["cs"].get(node, -1) < 0:
                    break
                if any(n_ == node for (n_, v), i in it["b"].entities.items()):
                    continue
                if any(node in g and g & set(it["added"]) for g in EXCLUSIVE):
                    continue
                if SEX_ONLY.get(node) not in (None, it["b"].p.sex):
                    continue
                if add_node(it["b"], node, rels_meta, usage, rel_usage, rel=r, p_struct=0.0):
                    it["added"].append(node)
                    node_usage[node] += 1
                    it["quota"] += 1
                    got += 1
        else:  # structural: both sides
            heads = [t for t in r["head_types"] if instantiable(t)]
            tails = [t for t in r["tail_types"] if instantiable(t)]
            if not heads or not tails:
                continue
            cands = sorted(
                (it for it in built if it["quota"] < 2 and len(it["b"].relations) < MAX_RELS_PER_BUNDLE),
                key=lambda it: -max(it["cs"].get(t, -1) for t in heads + tails),
            )
            for it in cands:
                if got >= need:
                    break
                b = it["b"]
                anchor_node = max(
                    heads + tails, key=lambda t: it["cs"].get(t, -1)
                )  # the side the brief supports is added via its person relation first
                if it["cs"].get(anchor_node, -1) < 0:
                    break
                aid = next((i for (n_, v), i in b.entities.items() if n_ == anchor_node), None)
                if aid is None:
                    if not add_node(b, anchor_node, rels_meta, usage, rel_usage, p_struct=0.0):
                        continue
                    it["added"].append(anchor_node)
                    node_usage[anchor_node] += 1
                    aid = next(i for (n_, v), i in b.entities.items() if n_ == anchor_node)
                if any(rr == rid for _, rr, _ in b.relations):
                    continue
                if add_struct(
                    b, aid, anchor_node, r, "head" if anchor_node in heads else "tail", usage, rel_usage
                ) and b.close(usage):
                    it["quota"] += 1
                    got += 1
        quota_log[rid] = {"before": counts[rid], "added": got}

    # ---- pass 3: multi-person quota ------------------------------------------------------------------------------------
    multi = sum(1 for it in built if len(it["persons"]) > 1)
    target = int(MULTI_PERSON_SHARE * len(built))
    if multi < target:
        singles = sorted((it for it in built if len(it["persons"]) == 1), key=lambda it: -max(it["rs"].values()))
        for it in singles:
            if multi >= target:
                break
            roles = [r for r in it["rs"] if r in ROLE_KIND and it["rs"][r] > 0]
            if not roles:
                break
            role = random.choices(roles, [it["rs"][r] for r in roles])[0]
            entry = pers.add_secondary(it["b"], ROLE_KIND[role], rels_meta, usage, role=role)
            if entry:
                it["persons"].append(entry)
                it["quota"] += 1
                multi += 1
                rel_usage[role] += 1

    # ---- export ----------------------------------------------------------------------------------------------------------
    out = []
    for it in built:
        br, b, prof, added, persons_list, rs, pool, cs = (
            it["br"],
            it["b"],
            it["prof"],
            it["added"],
            it["persons"],
            it["rs"],
            it["pool"],
            it["cs"],
        )
        bid, uid = br["bundle_id"], br["uid"]
        b.add_name_parts()
        js = b.to_json(ents_meta, rels_meta, br["document_type"], br["document_format"], [])
        js["persons"] = persons_list
        kinship_claims.rewrite_bundle(js)  # [e08] sexed, scenario-free kinship hints
        lab = {e["id"]: e["label"] for e in js["entities"]}
        val = {e["id"]: e["value"] for e in js["entities"]}
        for x in js["relations"]:
            x["fact_tr"] = x["fact_tr"].replace("Av. Av. ", "Av. ").replace("Dr. Dr. ", "Dr. ")
            if (
                x["relation"] == "card_component_of" and lab.get(x["head"]) == "card_cvv"
            ):  # the taxonomy template assumes an expiry date
                x["fact_tr"] = f"Kart {val[x['tail']]}, güvenlik kodu (CVV) {val[x['head']]}"
        P = PROFILE[prof]
        js.update(
            brief={
                "domain": br["domain"],
                "document_type": br["document_type"],
                "description": br["description"],
                "document_format": br["document_format"],
            },
            brief_uid=uid,
            affinity=round(affinity[bid], 4),
            profile=prof,
            fact_target=random.randint(*P["target"]),
            pool=[
                {
                    "node": n_,
                    "cosine": round(node_scores[uid].get(n_, 0.0), 4),
                    "centred": round(cs.get(n_, 0.0), 4),
                    "instantiated": n_ in added,
                    "tier": "B" if n_ in tiers.TIER_B else ("C" if n_ in tiers.TIER_C else "A"),
                }
                for n_ in pool
            ]
            + [
                {
                    "node": n_,
                    "cosine": round(node_scores[uid].get(n_, 0.0), 4),
                    "centred": round(cs.get(n_, 0.0), 4),
                    "instantiated": True,
                    "tier": "A",
                    "quota": True,
                }
                for n_ in added
                if n_ not in pool
            ],
            writer_allowed=sorted(tiers.TIER_B),
            persons=persons_list,
            relations_negative=[x for p in persons_list for x in p["negatives"]],
            locale={"city": b.loc.city, "town": b.loc.town, "neighbourhood": b.loc.nb, "zip": b.loc.zip},
            role_scores={
                r: round(rs[r], 3) for r in rs if rs[r] >= ROLE_THRESHOLD or r in {p["role"] for p in persons_list[1:]}
            },
        )
        js["fact_target"] = min(js["fact_target"], len(js["relations"])) or 1
        out.append(js)
        stats[prof] += 1
        stats[f"persons={len(persons_list)}"] += 1
        for p in persons_list[1:]:
            stats[f"kind={p['kind']}"] += 1
    out += list(out_zero.values())
    out.sort(key=lambda js: js["bundle_id"])
    with open(a.out, "w", encoding="utf-8") as f:
        for js in out:
            f.write(json.dumps(js, ensure_ascii=False) + "\n")
    nodes_cov = Counter(e["label"] for js in out for e in js["entities"])
    rels_cov = Counter(x["relation"] for js in out for x in js["relations"])
    nonzero = [js for js in out if js["profile"] != "zero"]
    report = {
        "bundles": len(out),
        "profiles": {k: v for k, v in stats.items() if "=" not in k},
        "persons": {k: v for k, v in stats.items() if k.startswith("persons")},
        "multi_person_share_nonzero": round(
            sum(1 for js in nonzero if len(js["persons"]) > 1) / max(len(nonzero), 1), 3
        ),
        "secondary_kinds": {k: v for k, v in stats.items() if k.startswith("kind")},
        "negatives": sum(len(js["relations_negative"]) for js in out),
        "multi_person_without_negative": sum(
            1 for js in nonzero if len(js["persons"]) > 1 and not js["relations_negative"]
        ),
        "distinct_nodes_instantiated": len(nodes_cov),
        "nodes_never_instantiated": sorted(n for n in ents_meta if n not in nodes_cov and n not in tiers.TIER_B),
        "distinct_relations": f"{len(rels_cov)}/{len(rels_meta)}",
        "relation_floor": floor,
        "relations_below_floor": {r: rels_cov[r] for r in rmap if rels_cov[r] < floor and r not in writer_owned},
        "writer_owned_relations": sorted(writer_owned),
        "relation_counts": dict(rels_cov.most_common()),
        "quota": quota_log,
        "node_counts": dict(nodes_cov.most_common()),
        "facts_per_bundle": dict(sorted(Counter(len(js["relations"]) for js in out).items())),
        "affinity_by_profile": {
            p: round(statistics.mean(js["affinity"] for js in out if js["profile"] == p), 3)
            for p in ("zero", "sparse", "normal", "dense")
            if any(js["profile"] == p for js in out)
        },
        "tier_b_kinds_offered": sum(1 for js in out for p in js["pool"] if p["tier"] == "B"),
        "tier_c_instantiated": sum(1 for js in out for p in js["pool"] if p["tier"] == "C" and p["instantiated"]),
    }
    json.dump(report, open(a.report or "soundness_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("node_counts", "relation_counts", "quota")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
