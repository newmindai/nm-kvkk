#!/usr/bin/env python3
"""Parse generated documents, bind spans to the sampled entities, attach relations, validate.

Accept a document only if it is a faithful realization of its bundle:
  * ends with the ===SON=== marker (not truncated), tags parse, no stray brackets
  * every tagged span is one of the bundle's entities with the right label (no invented PII)
  * every bundle entity is mentioned at least once
  * no untagged occurrence of an entity value outside a longer span
  * no label word leaked into the text
Outputs the taxonomy's record schema (records.jsonl), the GLiNER conversion (gliner.jsonl),
rejects with reasons (rejects.jsonl) and a stats summary.

Usage: python parse_facts.py --raw raw.parquet [--bundles bundles.jsonl]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "taxonomy"))
import kvkk_sampler as ks  # noqa: E402

TAG_RE = re.compile(
    r"\[([^\[\]]+)\]([a-zçğıöşü][a-z0-9_çğıöşü]*)"
)  # [e06] Turkish label words are captured, then mapped
# annotation-guideline vocabulary that must never surface inside a document
META_RE = re.compile(
    r"coreference|Madde 6|atıf çözümü|DOC_SUBJECT|sampler|coarse mod|entity node|anotasyon|simülasyon amaçlı|eğitim amaçlı örnek",
    re.I,
)

import sample_facts as sf  # noqa: E402  (name pools, for the invented-person guard)

NAME_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, sf.FIRST_F + sf.FIRST_M)) + r") (" + "|".join(map(re.escape, sf.LAST)) + r")\b"
)


def parse_tagged(tagged):
    out, spans, pos, clean_len = [], [], 0, 0
    for m in TAG_RE.finditer(tagged):
        pre = tagged[pos : m.start()]
        out.append(pre)
        clean_len += len(pre)
        value, label = m.group(1), m.group(2)
        lead = len(value) - len(value.lstrip())
        core = value.strip()
        trail = len(value) - lead - len(core)  # [e07] "[ Ad Soyad ]"
        out.append(value[:lead])
        clean_len += lead
        spans.append({"start": clean_len, "end": clean_len + len(core), "span": core, "label": label})
        out.append(core)
        clean_len += len(core)
        out.append(value[len(value) - trail :] if trail else "")
        clean_len += trail
        pos = m.end()
    out.append(tagged[pos:])
    clean = "".join(out)
    for s in spans:
        assert clean[s["start"] : s["end"]] == s["span"]
    return clean, spans


# labels whose values are generic words/numbers: an untagged occurrence may be a different
# use of the word, so it is never auto-repaired (the document is rejected instead)
NO_REPAIR = {
    "gender",
    "marital_status",
    "nationality",
    "age",
    "place_of_birth",
    "city",
    "district",
    "title_honorific",
    "postal_code",
}


def bind(clean, spans, bundle, all_labels):
    problems = []
    by_value = {}
    for e in bundle["entities"]:
        by_value.setdefault(e["value"].lower(), []).append(e)  # case-insensitive: "Kemoterapi" at sentence start
    ents_out, repaired = [], 0
    for s in spans:
        cands = by_value.get(s["span"].lower())
        if not cands:
            problems.append(f"unlisted PII span: [{s['span']}]{s['label']}")
            continue
        match = next((e for e in cands if e["label"] == s["label"]), None)
        if not match:
            problems.append(f"label mismatch: [{s['span']}]{s['label']} expected {'/'.join(e['label'] for e in cands)}")
            continue
        ents_out.append(
            {"id": match["id"], "label": match["label"], "span": s["span"], "start": s["start"], "end": s["end"]}
        )
    # untagged occurrences of known values outside any tagged span: repair (longest values first,
    # so a first name inside an untagged full name is covered by the full-name repair) or reject
    covered = [(s["start"], s["end"]) for s in spans]
    for e in sorted(bundle["entities"], key=lambda e: -len(e["value"])):
        pat = re.compile(r"(?<!\w)" + re.escape(e["value"]) + r"(?!\w)", re.IGNORECASE)
        for m in pat.finditer(clean):
            if any(a <= m.start() and m.end() <= b for a, b in covered):
                continue
            if e["label"] in NO_REPAIR:
                problems.append(f"untagged occurrence: {e['value']}")
                break
            ents_out.append(
                {"id": e["id"], "label": e["label"], "span": m.group(0), "start": m.start(), "end": m.end()}
            )
            covered.append((m.start(), m.end()))
            repaired += 1
    ents_out.sort(key=lambda x: x["start"])
    mentioned = {x["id"] for x in ents_out}
    for e in bundle["entities"]:
        if e["id"] not in mentioned:
            problems.append(f"entity not mentioned: {e['id']} [{e['value']}]{e['label']}")
    bind.repaired = repaired
    # invented people: any "First Last" pair from the name pools that is not a bundle value
    known = {e["value"] for e in bundle["entities"]}
    for m in NAME_RE.finditer(clean):
        if not any(m.group(0) in v for v in known):
            problems.append(f"invented person: {m.group(0)}")
            break
    if "[" in clean or "]" in clean:
        problems.append("stray bracket")
    if META_RE.search(clean):
        problems.append(f"taxonomy meta-language: {META_RE.search(clean).group(0)}")
    leaked = [l for l in all_labels if re.search(rf"(?<![\w\[]){re.escape(l)}(?![\w\]])", clean)]
    if leaked:
        problems.append(f"label words leaked: {leaked}")
    return ents_out, problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw.parquet")
    ap.add_argument("--bundles", default="bundles.jsonl")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()

    ents_meta, rels_meta = ks.load_taxonomy()
    all_labels = set(ents_meta)
    bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open(a.bundles, encoding="utf-8"))}
    cols = ["bundle_id", "domain", "document_type", "document_format", "dominant_group", "text_tagged"]
    df = pq.read_table(a.raw, columns=cols).to_pandas(ignore_metadata=True)

    records, full, rejects, reasons = [], [], [], Counter()
    for _, r in df.iterrows():
        b = bundles[int(r["bundle_id"])]
        raw = str(r["text_tagged"]).strip()
        raw = re.sub(r"^```[a-z]*\n|\n```$", "", raw)
        problems = []
        if not raw.endswith("===SON==="):
            problems.append("truncated")
        raw = raw.removesuffix("===SON===").strip()
        clean, spans = parse_tagged(raw)
        ents_out, more = bind(clean, spans, b, all_labels)
        problems += more
        rec = {
            "text": clean,
            "entities": ents_out,
            "relations": [{"head": x["head"], "relation": x["relation"], "tail": x["tail"]} for x in b["relations"]],
        }
        meta = {
            "bundle_id": b["bundle_id"],
            "domain": r["domain"],
            "document_type": r["document_type"],
            "document_format": r["document_format"],
            "dominant_group": r["dominant_group"],
            "groups": b["groups"],
            "madde6": b["madde6"],
            "repaired_mentions": bind.repaired,
            "text_tagged": raw,
        }
        if problems:
            for p in problems:
                reasons[p.split(":")[0]] += 1
            rejects.append({**meta, "problems": problems})
        else:
            records.append(rec)
            full.append({**rec, **meta})

    p = a.prefix
    with open(f"{p}records.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{p}records_full.jsonl", "w", encoding="utf-8") as f:
        for rec in full:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{p}gliner.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(ks.to_gliner(rec), ensure_ascii=False) + "\n")
    with open(f"{p}rejects.jsonl", "w", encoding="utf-8") as f:
        for rec in rejects:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{p}retry-bundles.jsonl", "w", encoding="utf-8") as f:  # feed back into gen_facts.py
        for rec in rejects:
            f.write(json.dumps(bundles[rec["bundle_id"]], ensure_ascii=False) + "\n")

    n_ent = sum(len(r["entities"]) for r in records)
    n_rel = sum(len(r["relations"]) for r in records)
    labels = Counter(e["label"] for r in records for e in r["entities"])
    rels = Counter(x["relation"] for r in records for x in r["relations"])
    print(f"parsed {len(df)} docs -> {len(records)} accepted, {len(rejects)} rejected")
    print("reject reasons:", dict(reasons.most_common()))
    print(f"accepted: {n_ent} entity mentions ({len(labels)} labels), {n_rel} relations ({len(rels)} relation types)")
    print(
        f"avg mentions/doc {n_ent / max(len(records), 1):.1f} | avg chars {sum(len(r['text']) for r in records) / max(len(records), 1):.0f}"
        f" | auto-repaired mentions: {sum(r['repaired_mentions'] for r in full)} in {sum(1 for r in full if r['repaired_mentions'])} docs"
    )
