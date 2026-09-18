#!/usr/bin/env python3
"""v2 parse/bind/accept — FIX C on top of the v1 parser.

Adds: optional name-part entities (surname-only mentions bind, are not required); extended
leak guards (unlisted long numbers, addresses, organizations, demographic fields); suffix
vowel-harmony post-fix (offset-safe); a cap on auto-repairs; wider meta-language filter.
Usage: python parse_facts.py --raw raw.parquet --bundles bundles-gated.jsonl [--prefix retry-]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PARENT, "..", "taxonomy"))  # [pkg]
import harmony  # noqa: E402  (v2 dir is sys.path[0] when run as a script)
import kvkk_sampler as ks  # noqa: E402

sys.path.insert(0, PARENT)
import tiers  # noqa: E402  [e05] writer-owned kinds


def _load(name, path):
    """Load a v1 module by file path so same-named v2 files do not shadow it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("sample_facts", os.path.join(PARENT, "sample_facts.py"))  # v1 pools, needed by v1 parse_facts
_v1 = _load("parse_facts_v1", os.path.join(PARENT, "parse_facts.py"))
parse_tagged, NAME_RE, NO_REPAIR = _v1.parse_tagged, _v1.NAME_RE, _v1.NO_REPAIR

# invented-person guard over the big name pools (names/*.csv), not the small v1 lists
_names_dir = os.path.join(HERE, "names")
if os.path.exists(os.path.join(_names_dir, "first_names.csv")):
    import csv

    _first = [r["name"] for r in csv.DictReader(open(os.path.join(_names_dir, "first_names.csv"), encoding="utf-8"))]
    _last = [r["name"] for r in csv.DictReader(open(os.path.join(_names_dir, "surnames.csv"), encoding="utf-8"))]
    NAME_RE = re.compile(
        r"(?<![\wçğıöşüÇĞİÖŞÜ])(?:"
        + "|".join(map(re.escape, sorted(_first, key=len, reverse=True)))
        + r") (?:"
        + "|".join(map(re.escape, sorted(_last, key=len, reverse=True)))
        + r")(?![\wçğıöşüÇĞİÖŞÜ])"
    )

META_RE = re.compile(
    r"coreference|Madde 6|atıf çözümü|DOC_SUBJECT|sampler|coarse mod|entity node|anotasyon|"
    r"simülasyon amaçlı|eğitim amaçlı örnek|fine-coarse|simetrik ilişki|tek yönlü olarak|\(rol:|etiket olarak",
    re.I,
)
NUMBER_RE = re.compile(
    r"(?<![\d.])(?:\d[ .\-]?){10,}(?![\d.])"
)  # 10+ digits: phones, TCKN, IBAN, cards — not 9-digit reference numbers
ADDRESS_RE = re.compile(r"\S+ (?:Mah\.|Mahallesi|Cad\.|Caddesi|Sok\.|Sokak|Bulvarı)[^\n]{0,40}?No:?\s?\d+")
IDENT_RE = re.compile(
    r"\b(?:[A-HJ-NPR-Z0-9]{17}|\d{2} [A-Z]{1,3} \d{2,4}|TR\d{2}(?: ?\d{4}){5} ?\d{2})\b"
)  # VIN, plate, IBAN
ORG_RE = re.compile(
    r"(?:[A-ZÇĞİÖŞÜ][\wçğıöşü&.-]*[ \t]){1,4}(?:A\.Ş\.|Ltd\. ?Şti\.|San\. Tic\.|Ltd\.)"
)  # [e05] no newline
DEMO_RE = re.compile(r"(Cinsiyet|Yaş|Medeni Durum|Uyruk|Doğum Tarihi|Kan Grubu|Din|Etnik Köken)\s*:\s*([^\n,;]+)")
MAX_REPAIRS = 4
FACTS_OPTIONAL = False
# words that only exist in the taxonomy's ASCII-folded descriptions, never in real Turkish text
FOLDED_RE = re.compile(
    r"\b(kisisin\w*|kisisi\b|kisisine|bileseni|parcasidir|numarasidir|sicilidir|isaretler|"
    r"baglanmistir|goruntulenmistir|kesilmistir|uyrugudur|tam adinin|tam adresinin)\b"
)


def _norm(s):
    return re.sub(r"[^\w ]", " ", s.lower()).split()


def tlow(s):  # [e08] Turkish-aware lowercase: "NİHAYET" -> "nihayet" (str.lower turns İ into i + combining dot)
    return s.replace("İ", "i").replace("I", "ı").lower()


def bind(clean, spans, bundle, all_labels):
    problems = []
    by_value = {}
    for e in bundle["entities"]:
        by_value.setdefault(tlow(e["value"]), []).append(e)
    ents_out, repaired, writer_added = [], 0, []
    known_all = {e["value"] for e in bundle["entities"]}
    for s in spans:
        cands = by_value.get(tlow(s["span"]))
        if not cands:
            ok, why = (
                tiers.validate(s["label"], s["span"], known_all, NAME_RE)
                if s["label"] in tiers.TIER_B
                else (False, "not a writer-owned kind")
            )  # [e05]
            if ok:
                wid = f"w{len(writer_added) + 1}"
                writer_added.append({"id": wid, "label": s["label"], "value": s["span"]})
                by_value.setdefault(tlow(s["span"]), []).append(writer_added[-1])
                ents_out.append(
                    {
                        "id": wid,
                        "label": s["label"],
                        "span": s["span"],
                        "start": s["start"],
                        "end": s["end"],
                        "source": "writer",
                    }
                )
            else:
                problems.append(f"unlisted PII span: [{s['span']}]{s['label']} ({why})")
            continue
        match = next((e for e in cands if e["label"] == s["label"]), None)
        if not match:
            problems.append(f"label mismatch: [{s['span']}]{s['label']} expected {'/'.join(e['label'] for e in cands)}")
            continue
        ents_out.append(
            {"id": match["id"], "label": match["label"], "span": s["span"], "start": s["start"], "end": s["end"]}
        )
    covered = [(s["start"], s["end"]) for s in spans]
    value_labels = {}
    for e in bundle["entities"]:
        value_labels.setdefault(tlow(e["value"]), set()).add(e["label"])
    for e in sorted(bundle["entities"], key=lambda e: -len(e["value"])):
        if len(e["value"]) < 4 or (e["value"].isdigit() and len(e["value"]) < 6):  # [e05] never auto-bind '12', '008'
            continue
        if len(value_labels[tlow(e["value"])]) > 1:  # [e08] same value under two labels: never auto-repair
            continue
        flags = (
            0 if e["label"] in ("gender", "marital_status", "nationality", "title_honorific") else re.IGNORECASE
        )  # [e08] "erkek kardeşi" is not the gender field
        pat = re.compile(r"(?<!\w)" + re.escape(e["value"]) + r"(?!\w)", flags)
        for m in pat.finditer(clean):
            if any(
                a < m.end() and m.start() < b for a, b in covered
            ):  # [e08] any overlap with a tagged span, not only containment
                continue
            if (
                e["label"] == "nationality" and e["value"].rstrip(".").upper() in ("T.C", "TC")
            ):  # [e08] an untagged "T.C." is a prefix (T.C. kimlik no, T.C. Sağlık Bakanlığı), never a nationality mention
                continue
            if e["label"] in NO_REPAIR:
                problems.append(f"untagged occurrence: {e['value']}")
                break
            ents_out.append(
                {"id": e["id"], "label": e["label"], "span": m.group(0), "start": m.start(), "end": m.end()}
            )
            covered.append((m.start(), m.end()))
            repaired += 1
    if repaired > MAX_REPAIRS:
        problems.append(f"too many auto-repairs: {repaired}")
    if len(writer_added) > tiers.MAX_WRITER_ADDED:  # [e05] -> [e07] a flag, not a reject
        problems.append(f"FLAG too many writer-added facts: {len(writer_added)}")
    ents_out.sort(key=lambda x: x["start"])
    mentioned = {x["id"] for x in ents_out}
    for e in bundle["entities"]:
        required = not e.get("optional") and (not FACTS_OPTIONAL or e["id"] == bundle["subject_id"])
        if e["id"] not in mentioned and required:
            problems.append(f"entity not mentioned: {e['id']} [{e['value']}]{e['label']}")
    # --- guards ---
    known = {e["value"] for e in bundle["entities"]} | {w["value"] for w in writer_added}  # [e05]
    for m in NAME_RE.finditer(clean):
        if not any(m.group(0) in v for v in known):
            problems.append(f"invented person: {m.group(0)}")
            break

    def inside(a, b):
        return any(x <= a and b <= y for x, y in covered)

    digits_known = {re.sub(r"\D", "", v) for v in known} - {""}
    for m in NUMBER_RE.finditer(clean):
        d = re.sub(r"\D", "", m.group(0))
        s0 = m.group(0).strip()
        if (
            re.match(r"\d{1,2}[./-]\d{1,2}[./-](19|20)\d{2}[ .\-]\d{1,2}\b", s0)
            or re.match(
                r"(19|20)\d{2}-\d{2}-\d{2}[ T]\d{1,2}\b", s0
            )  # [e08] "14.05.2025 09:30", "2025-02-14 16:00": a date and a time
            or re.fullmatch(r"\d{1,3}(\.\d{3}){3,}", s0)
            or re.fullmatch(r"(255|0)(\.(255|0)){3}", s0)
        ):  # [e08] "1.148.600.000" TL, "255.255.255.0"
            continue
        if not inside(m.start(), m.end()) and not any(d in k or k in d for k in digits_known):
            problems.append(f"unlisted number: {m.group(0).strip()}")
            break
    for m in IDENT_RE.finditer(
        clean
    ):  # invented VINs / plates / IBAN-like codes (alphanumeric, so NUMBER_RE misses them)
        if not inside(m.start(), m.end()) and not any(m.group(0) in v for v in known):
            problems.append(f"unlisted identifier: {m.group(0)}")
            break
    for m in ADDRESS_RE.finditer(clean):
        if not inside(m.start(), m.end()):
            problems.append(f"untagged address: {m.group(0)[:50]}")
            break
    for m in ORG_RE.finditer(clean):
        if not inside(m.start(), m.end()) and not any(m.group(0).strip() in v for v in known):
            problems.append(f"untagged organization: {m.group(0).strip()}")
            break
    for m in DEMO_RE.finditer(clean):
        vs = m.start(2)
        if not any(x <= vs < y for x, y in covered):
            problems.append(f"untagged demographic field: {m.group(0)[:40]}")
            break
    if "[" in clean or "]" in clean:
        problems.append("stray bracket")
    for s_ in spans:  # [e08] repeated word right before a tag: "Op. Dr. [Op. Dr. X]", "A Blok'taki [A Blok, ...]"
        head = clean[max(0, s_["start"] - 40) : s_["start"]].rstrip()
        first = s_["span"].split()[0] if s_["span"].split() else ""
        if len(first) >= 3 and head.endswith(first) and not head[: -len(first)].rstrip().endswith((":", "-")):
            problems.append(f"FLAG repeated word before tag: {head[-25:]!r} + [{s_['span'][:25]}]")
            break
    for s_ in spans:  # [e08] "T.C." tagged as nationality inside "T.C. Kimlik Numarası"
        if (
            s_["label"] == "nationality"
            and s_["span"].rstrip(".") in ("T.C", "TC")
            and re.match(r"\s*kimlik", clean[s_["end"] : s_["end"] + 10], re.I)
        ):
            problems.append("label mismatch: 'T.C.' inside 'T.C. Kimlik Numarası' is not a nationality mention")
            break
    for s_ in spans:  # [e07] residue of a copied label description right after a tag ("]maaş / ücret tutarı")
        tail = clean[s_["end"] : s_["end"] + 40]
        if re.match(r"\s*/\s*[a-zçğıöşü][a-zçğıöşü ]{2,30}", tail):
            problems.append(f"label description residue after [{s_['span'][:30]}]: {tail[:30].strip()!r}")
            break
    if META_RE.search(clean):
        problems.append(f"taxonomy meta-language: {META_RE.search(clean).group(0)}")
    if FOLDED_RE.search(clean.lower()):
        problems.append(f"ascii-folded template word: {FOLDED_RE.search(clean.lower()).group(0)}")

    # verbatim copying of a hint's boilerplate — entity values are removed first, otherwise a
    # 6-word address value would count as "copying"
    def strip_values(s):
        for v in sorted(known, key=len, reverse=True):
            s = s.replace(v, " ")
        return _norm(s)

    doc_words = " ".join(strip_values(clean))
    for r in bundle["relations"]:
        w = strip_values(r["fact_tr"])
        if len(w) >= 5 and any(" ".join(w[i : i + 5]) in doc_words for i in range(len(w) - 4)):
            problems.append(f"fact hint copied verbatim: {r['fact_tr'][:50]}")
            break
    leaked = [l for l in all_labels if re.search(rf"(?<![\w\[]){re.escape(l)}(?![\w\]])", clean)]
    if leaked:
        problems.append(f"label words leaked: {leaked}")
    return ents_out, problems, repaired, writer_added


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw.parquet")
    ap.add_argument("--bundles", default="bundles-gated.jsonl")
    ap.add_argument("--prefix", default="")
    ap.add_argument(
        "--facts-optional",
        action="store_true",
        help="facts were candidates: keep only relations whose entities the model actually used",
    )
    a = ap.parse_args()
    FACTS_OPTIONAL = a.facts_optional

    ents_meta, rels_meta = ks.load_taxonomy()
    all_labels = set(ents_meta)
    ALIAS = {}  # [e06] Turkish/English aliases -> node id, for writer-added tags like ]maaş
    for nid, e in ents_meta.items():
        for a_ in e["labels_en"] + e["labels_tr"]:
            ALIAS.setdefault(a_, nid)
            ALIAS.setdefault(a_.replace("_", " "), nid)
    ALIAS.update(
        {
            "maaş": "salary_info",
            "maas": "salary_info",
            "işyeri": "workplace_location",
            "isyeri": "workplace_location",
            "kredi": "risk_report",
        }
    )
    bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open(a.bundles, encoding="utf-8"))}
    cols = ["bundle_id", "scenario", "genre", "document_format", "text_tagged"]
    df = pq.read_table(a.raw, columns=cols).to_pandas(ignore_metadata=True)

    records, full, rejects, reasons, flagged = [], [], [], Counter(), []  # [e07]
    for _, r in df.iterrows():
        b = bundles[int(r["bundle_id"])]
        raw = str(r["text_tagged"]).strip()
        raw = re.sub(r"^```[a-z]*\n|\n```$", "", raw)
        problems = []
        if not raw.endswith("===SON==="):
            problems.append("truncated")
        raw = raw.removesuffix("===SON===").strip()
        clean, spans = parse_tagged(raw)
        if b.get("profile") == "zero":  # [e07] zero-PII document: no tags, no guard hits, no pool name
            probs0 = ["tagged span in a zero-PII document"] if spans else []
            if NAME_RE.search(clean):
                probs0.append(f"person name in a zero-PII document: {NAME_RE.search(clean).group(0)}")
            for rx, what in (
                (NUMBER_RE, "long number"),
                (IDENT_RE, "identifier"),
                (ADDRESS_RE, "address"),
                (DEMO_RE, "demographic field"),
            ):
                m_ = rx.search(clean)
                if m_:
                    probs0.append(f"{what} in a zero-PII document: {m_.group(0)[:40].strip()}")
            if "[" in clean or "]" in clean:
                probs0.append("stray bracket")
            problems += probs0
            rec = {"text": clean, "entities": [], "relations": [], "relations_negative": [], "negative": True}
            meta = {
                "bundle_id": b["bundle_id"],
                "scenario": "zero",
                "genre": r["genre"],
                "document_format": r["document_format"],
                "profile": "zero",
                "groups": [],
                "madde6": False,
                "repaired_mentions": 0,
                "harmony_fixes": 0,
                "facts_offered": 0,
                "facts_used": 0,
                "relations_omitted": [],
                "text_tagged": raw,
                "writer_added": [],
                "relations_writer": [],
                "persons": [],
                "persons_used": [],
            }
            if problems:
                for p in problems:
                    reasons[p.split(":")[0]] += 1
                rejects.append({**meta, "problems": problems})
            else:
                records.append(rec)
                full.append({**rec, **meta})
            continue
        for s_ in spans:  # [e06] map alias labels to node ids
            if s_["label"] not in all_labels and s_["label"] in ALIAS:
                s_["alias"] = s_["label"]
                s_["label"] = ALIAS[s_["label"]]
        ents_out, more, repaired, writer_added = bind(clean, spans, b, all_labels)
        problems += more
        clean, n_fix = harmony.fix_text(clean, ents_out)
        used = {x["id"] for x in ents_out}
        rels_all = [{"head": x["head"], "relation": x["relation"], "tail": x["tail"]} for x in b["relations"]]
        if FACTS_OPTIONAL:
            rels = [x for x in rels_all if x["head"] in used and x["tail"] in used]
            omitted = [x for x in rels_all if x not in rels]
            if not rels and not writer_added:  # [e05] a document may carry writer-owned facts only
                problems.append("no candidate fact used")
            # dependency closure on what the model used: every used entity must connect to the
            # subject through used relations (no CVV without its card, no mother without mother_of)
            from collections import deque

            adj = {}
            for x in rels:
                adj.setdefault(x["head"], set()).add(x["tail"])
                adj.setdefault(x["tail"], set()).add(x["head"])
            anchors = {p["id"] for p in b.get("persons", [])} or {
                b["subject_id"]
            }  # [e06] every named person anchors its own facts
            seen, q = set(anchors), deque(anchors)
            while q:
                for nxt in adj.get(q.popleft(), ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            optional_ids = {e["id"] for e in b["entities"] if e.get("optional")}
            dangling = [
                e for e in b["entities"] if e["id"] in used and e["id"] not in seen and e["id"] not in optional_ids
            ]
            if dangling:
                problems.append("dependency broken: " + ", ".join(f"{e['label']}={e['value']}" for e in dangling[:3]))
        else:
            rels, omitted = rels_all, []
        canon = tiers.canonical_relations(rels_meta)  # [e05] pending relations for writer-added facts
        rels_writer = []
        for w in writer_added:
            rid, side = canon[w["label"]]
            rels_writer.append(
                {
                    "head": w["id"] if side == "head" else b["subject_id"],
                    "relation": rid,
                    "tail": b["subject_id"] if side == "head" else w["id"],
                    "status": "pending",
                }
            )
        negs = [n for n in b.get("relations_negative", []) if n["head"] in used and n["tail"] in used]  # [e06]
        rec = {"text": clean, "entities": ents_out, "relations": rels, "relations_negative": negs}
        meta = {
            "bundle_id": b["bundle_id"],
            "scenario": b["scenario"],
            "genre": r["genre"],
            "document_format": r["document_format"],
            "groups": b["groups"],
            "madde6": b["madde6"],
            "repaired_mentions": repaired,
            "harmony_fixes": n_fix,
            "facts_offered": len(rels_all),
            "facts_used": len(rels),
            "relations_omitted": omitted,
            "text_tagged": raw,
            "writer_added": writer_added,
            "relations_writer": rels_writer,  # [e05]
            "persons": b.get("persons", []),
            "persons_used": [p["id"] for p in b.get("persons", []) if p["id"] in used],
        }  # [e06]
        meta["profile"] = b.get("profile")
        meta["fact_target"] = b.get("fact_target")  # [e07]
        hard = [p for p in problems if not p.startswith("FLAG")]
        if hard:
            for p in problems:
                reasons[p.split(":")[0]] += 1
            rejects.append({**meta, "problems": problems})
        elif problems:  # [e07] flags only -> flagged bucket (kept out of records until reviewed)
            for p in problems:
                reasons[p.split(":")[0]] += 1
            flagged.append({**rec, **meta, "flags": problems})
        else:
            records.append(rec)
            full.append({**rec, **meta})

    p = a.prefix
    for name, data in (
        ("records.jsonl", records),
        ("records_full.jsonl", full),
        ("rejects.jsonl", rejects),
        ("flagged.jsonl", flagged),
    ):  # [e07]
        with open(f"{p}{name}", "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{p}retry-bundles.jsonl", "w", encoding="utf-8") as f:
        for rec in rejects:
            f.write(json.dumps(bundles[rec["bundle_id"]], ensure_ascii=False) + "\n")
    n_ent = sum(len(r["entities"]) for r in records)
    print(
        f"parsed {len(df)} docs -> {len(records)} accepted, {len(flagged)} flagged, {len(rejects)} rejected | reasons {dict(reasons.most_common())}"
    )
    print(
        f"accepted: {n_ent} mentions, {sum(len(r['relations']) for r in records)} relations | "
        f"auto-repairs {sum(r['repaired_mentions'] for r in full)} | harmony fixes {sum(r['harmony_fixes'] for r in full)} | "
        f"avg chars {sum(len(r['text']) for r in records) / max(len(records), 1):.0f}"
    )
