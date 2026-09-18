#!/usr/bin/env python3
"""Build viewer.html for one experiment: the pipeline flow explained + every document with its tags
highlighted, the offered facts marked used/omitted, and the parser / cheap-judge / Sonnet verdicts.
Run from the run folder: python code/build_viewer.py  -> viewer.html
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taxonomy"))  # [pkg]
import kvkk_sampler as ks  # noqa: E402

TAG = re.compile(r"\[([^\[\]]+)\]([a-z][a-z0-9_]*)")
GROUP_OF_FILE = {
    "01": "person",
    "02": "identity",
    "03": "contact",
    "04": "civil",
    "05": "financial",
    "06": "digital",
    "07": "location",
    "08": "location",
    "09": "sensitive",
    "10": "sensitive",
    "11": "employment",
    "12": "employment",
    "13": "legal",
    "15": "legal",
}
ents_meta, rels_meta = ks.load_taxonomy()
label_group = {n: GROUP_OF_FILE[e["_file"][:2]] for n, e in ents_meta.items()}
rel_tr = {r["id"]: r["labels_tr"][0] for r in rels_meta}

bundles = {b["bundle_id"]: b for b in (json.loads(l) for l in open("bundles.jsonl", encoding="utf-8"))}


def _load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


acc = {r["bundle_id"]: r for r in _load("dataset/records_full.jsonl")}
manifest = {m["bundle_id"]: m for m in _load("manifest.jsonl")}
others = {}
for p, bucket in (
    ("repair-flagged.jsonl", "flagged"),
    ("repair-rejects.jsonl", "discarded"),
    ("flagged.jsonl", "flagged"),
    ("rejects.jsonl", "discarded"),
):
    for r in _load(p):
        if r["bundle_id"] not in acc and r["bundle_id"] not in others:
            others[r["bundle_id"]] = {**r, "bucket": bucket}
import glob as _glob  # [e08] the coherence judge writes judge-<tag>.json

_jf = "judge.json" if os.path.exists("judge.json") else next(iter(sorted(_glob.glob("judge-*.json"))), None)
judge = {int(k): v for k, v in json.load(open(_jf, encoding="utf-8")).items()} if _jf else {}
sonnet = {}
for f in sorted(glob.glob("eval/out-*.json")):
    for x in json.load(open(f, encoding="utf-8"))["docs"]:
        sonnet[x["bundle_id"]] = x
summary = (
    json.load(open("eval/summary.json", encoding="utf-8"))
    if os.path.exists("eval/summary.json")
    else {"overall_mean": 0, "mean_scores": {}, "verdicts": {}, "docs_with_no_issues": 0, "parser_disagreements": []}
)

# [e08] 4,496 accepted documents render to 32 MB; --sample keeps the page publishable (16 MB limit) with a
# stratified pick: every reviewed document, then zero-PII / repaired / flagged / discarded / multi-person / by profile.
import argparse as _ap
import random as _rnd

_a = _ap.ArgumentParser()
_ap_ = _a.add_argument
_ap_("--sample", type=int, default=0)
_ap_("--out", default="viewer.html")
_args = _a.parse_args()
if _args.sample:
    _rnd.seed(11)
    keep, groups = set(sonnet), {}
    for bid in bundles:
        r = acc.get(bid) or others.get(bid)
        if r is None:
            continue
        g = (
            "discarded"
            if bid in others and others[bid]["bucket"] == "discarded"
            else "flagged"
            if bid in others
            else "zero"
            if r.get("negative")
            else "repair"
            if r.get("stage") in ("repair", "sexfix")
            else "multi"
            if len(r.get("persons_used", [])) > 1
            else "writer"
            if r.get("writer_added")
            else r.get("profile", "normal")
        )
        groups.setdefault(g, []).append(bid)
    per = max(1, (_args.sample - len(keep)) // max(len(groups), 1))
    for ids in groups.values():
        _rnd.shuffle(ids)
        keep.update(ids[:per])
    bundles = {bid: b for bid, b in bundles.items() if bid in keep}

docs = []
for bid in sorted(bundles):
    b = bundles[bid]
    r = acc.get(bid) or others.get(bid)
    if r is None:
        continue
    accepted = bid in acc
    brief = b.get("brief", {})
    byid = {e["id"]: e for e in b["entities"]}
    byid.update(
        {
            e["id"]: {"id": e["id"], "label": e["label"], "value": e["span"]}
            for e in r.get("entities", [])
            if e.get("source") == "writer"
        }
    )
    subj = byid[b["subject_id"]]["value"] if b.get("subject_id") else "—"
    tagged_values = {(m.group(1).strip().lower(), m.group(2)) for m in TAG.finditer(r["text_tagged"])}
    used_ids = {e["id"] for e in b["entities"] if (e["value"].lower(), e["label"]) in tagged_values}
    pool_cos = {p["node"]: p.get("centred", p.get("cosine")) for p in b.get("pool", [])}
    facts = [
        {
            "fact": x["fact_tr"],
            "relation": x["relation"],
            "relation_tr": rel_tr.get(x["relation"], x["relation"]),
            "used": x["head"] in used_ids and x["tail"] in used_ids,
        }
        for x in b["relations"]
    ]
    rels_kept = [
        {
            "head": byid.get(x["head"], {}).get("value", x["head"]),
            "relation": x["relation"],
            "tail": byid.get(x["tail"], {}).get("value", x["tail"]),
        }
        for x in (r.get("relations", []) if accepted else [])
    ]
    s_ = sonnet.get(bid, {})
    j = judge.get(bid, {})
    bucket = "accepted" if accepted else r.get("bucket", "discarded")
    docs.append(
        {
            "id": bid,
            "type": brief.get("document_type"),
            "domain": brief.get("domain"),
            "format": brief.get("document_format"),
            "description": brief.get("description"),
            "subject": subj,
            "scenario": b.get("profile", "pool"),
            "madde6": bool(b.get("madde6")),
            "profile": b.get("profile"),
            "stage": r.get("stage", manifest.get(bid, {}).get("stage", "gen")),
            "bucket": bucket,
            "chars": len(TAG.sub(r"\1", r["text_tagged"])),
            "parser": {"decision": bucket, "problems": r.get("problems", []) + r.get("flags", [])},
            "facts": facts,
            "entities": [
                {
                    "id": e["id"],
                    "label": e["label"],
                    "value": e["value"],
                    "optional": bool(e.get("optional")),
                    "used": e["id"] in used_ids,
                    "group": label_group.get(e["label"], "legal"),
                    "cosine": pool_cos.get(e["label"]),
                }
                for e in b["entities"]
            ],
            "relations_kept": rels_kept,
            "writer_added": [
                {
                    **w,
                    "status": next(
                        (
                            x.get("status", "pending")
                            for x in r.get("relations_writer", [])
                            if w["id"] in (x["head"], x["tail"])
                        ),
                        "pending",
                    ),
                }
                for w in r.get("writer_added", [])
            ],
            "locale": b.get("locale"),
            "persons": b.get("persons", []),
            "negatives": r.get("relations_negative", []) if accepted else [],
            "text_tagged": r["text_tagged"],
            "judge": {"score": j.get("coherence"), "reason": j.get("reason", "")},
            "sonnet": {
                "scores": s_.get("scores"),
                "verdict": s_.get("verdict"),
                "parser_agreement": s_.get("parser_agreement"),
                "facts_used_naturally": s_.get("facts_used_naturally"),
                "issues": s_.get("issues", []),
            },
        }
    )
stats = {
    "docs": len(docs),
    "accepted": len(acc),
    "rejected": len(others),
    "sonnet_overall": summary["overall_mean"],
    "sonnet_means": summary["mean_scores"],
    "verdicts": summary["verdicts"],
    "judge_mean": round(
        sum(v["coherence"] for v in judge.values() if v.get("coherence") is not None)
        / max(sum(1 for v in judge.values() if v.get("coherence") is not None), 1),
        2,
    ),
    "facts_used": sum(f["used"] for d in docs for f in d["facts"]),
    "facts_offered": sum(len(d["facts"]) for d in docs),
    "no_issue_docs": summary["docs_with_no_issues"],
    "parser_disagreements": len(summary["parser_disagreements"]),
}
payload = json.dumps({"docs": docs, "stats": stats, "label_group": label_group}, ensure_ascii=False).replace(
    "</", "<\\/"
)

HTML = r"""<title>__RUN__ Generation Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#F5F7F4; --surface:#FFFFFF; --surface-2:#EEF2EC; --ink:#1B2430; --ink-2:#4A5563; --muted:#7A8594; --line:#D7DDD5; --line-2:#C3CBC0;
  --accent:#0F6E56; --accent-ink:#FFFFFF; --accent-soft:#DDEFE7;
  --ok:#1E7B4A; --ok-soft:#DCEFE2; --warn:#9A6B08; --warn-soft:#F6EBCB; --bad:#B4362E; --bad-soft:#F6DCD9;
  --g-person:#A2660A; --g-identity:#BB4A1E; --g-contact:#1F63A8; --g-civil:#1F7A78; --g-financial:#6640B5; --g-digital:#4149B0; --g-location:#5E6E12; --g-sensitive:#B0307E; --g-employment:#7A5A34; --g-legal:#4B5A6E;
  --shadow:0 1px 2px rgba(20,30,25,.06), 0 6px 18px rgba(20,30,25,.06);
  --font-display:"Bricolage Grotesque","IBM Plex Sans",system-ui,sans-serif; --font-body:"IBM Plex Sans",system-ui,sans-serif; --font-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --ground:#151A18; --surface:#1D2321; --surface-2:#242C29; --ink:#E7ECE8; --ink-2:#B8C1BB; --muted:#8B958F; --line:#33403A; --line-2:#44524B;
  --accent:#4FC59F; --accent-ink:#0E1B16; --accent-soft:#1E3A30;
  --ok:#5CCB8A; --ok-soft:#1F3B2B; --warn:#E3B84A; --warn-soft:#3E3319; --bad:#F08078; --bad-soft:#442423;
  --g-person:#E5A94D; --g-identity:#F0906A; --g-contact:#77B2F0; --g-civil:#6BC8C4; --g-financial:#B69BF2; --g-digital:#9BA2F5; --g-location:#B9C95A; --g-sensitive:#EE8CC7; --g-employment:#D0AC85; --g-legal:#A9B7C9;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --ground:#151A18; --surface:#1D2321; --surface-2:#242C29; --ink:#E7ECE8; --ink-2:#B8C1BB; --muted:#8B958F; --line:#33403A; --line-2:#44524B;
  --accent:#4FC59F; --accent-ink:#0E1B16; --accent-soft:#1E3A30;
  --ok:#5CCB8A; --ok-soft:#1F3B2B; --warn:#E3B84A; --warn-soft:#3E3319; --bad:#F08078; --bad-soft:#442423;
  --g-person:#E5A94D; --g-identity:#F0906A; --g-contact:#77B2F0; --g-civil:#6BC8C4; --g-financial:#B69BF2; --g-digital:#9BA2F5; --g-location:#B9C95A; --g-sensitive:#EE8CC7; --g-employment:#D0AC85; --g-legal:#A9B7C9;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--font-body);font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
a{color:var(--accent)}
.wrap{max-width:1240px;margin:0 auto;padding:28px 24px 64px}
h1,h2,h3{font-family:var(--font-display);font-weight:700;letter-spacing:-.01em;text-wrap:balance;margin:0}
h1{font-size:34px;line-height:1.1}
h2{font-size:22px;margin:0 0 6px}
h3{font-size:15px;font-weight:600;font-family:var(--font-body)}
.eyebrow{font-family:var(--font-mono);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{max-width:66ch;color:var(--ink-2);margin:10px 0 0}
header.top{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end;padding-bottom:22px;border-bottom:1px solid var(--line)}
.stats{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px}
.stat{padding:10px 12px;background:var(--surface);border:1px solid var(--line);border-radius:6px}
.stat b{display:block;font-family:var(--font-display);font-size:22px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1}
.stat span{font-size:12px;color:var(--muted)}
section{margin-top:36px}
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:14px}
.section-head p{margin:4px 0 0;color:var(--ink-2);max-width:70ch}
figure{margin:0;padding:18px 18px 12px;background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow-x:auto}
figure svg{display:block;max-width:100%;height:auto;min-width:860px;color:var(--ink)}
figcaption{font-size:13px;color:var(--muted);margin-top:8px}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:14px}
.step{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;display:flex;flex-direction:column;gap:6px}
.step .n{font-family:var(--font-mono);font-size:11.5px;color:var(--accent);letter-spacing:.06em}
.step p{margin:0;color:var(--ink-2);font-size:14px}
.step .io{font-family:var(--font-mono);font-size:12px;color:var(--muted);margin-top:auto;padding-top:6px;border-top:1px dashed var(--line)}
.step .io b{color:var(--ink-2);font-weight:500}
.note{margin-top:14px;padding:12px 16px;border-left:3px solid var(--accent);background:var(--accent-soft);color:var(--ink-2);font-size:14px;border-radius:0 6px 6px 0}
.note b{color:var(--ink)}
/* reader */
.reader{display:grid;grid-template-columns:320px minmax(0,1fr);gap:18px;align-items:start}
.rail{position:sticky;top:12px;background:var(--surface);border:1px solid var(--line);border-radius:8px;display:flex;flex-direction:column;max-height:calc(100vh - 24px)}
.filters{display:flex;flex-wrap:wrap;gap:6px;padding:12px;border-bottom:1px solid var(--line)}
.chip{font-family:var(--font-mono);font-size:11.5px;padding:3px 9px;border-radius:999px;border:1px solid var(--line-2);background:transparent;color:var(--ink-2);cursor:pointer}
.chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.chip:focus-visible,.item:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.list{overflow:auto;padding:6px}
.item{display:grid;grid-template-columns:34px 1fr auto;gap:8px;align-items:center;width:100%;text-align:left;padding:8px 8px;border:0;background:transparent;border-radius:6px;color:var(--ink);cursor:pointer;font:inherit}
.item:hover{background:var(--surface-2)}
.item[aria-current="true"]{background:var(--accent-soft)}
.item .id{font-family:var(--font-mono);font-size:12px;color:var(--muted)}
.item .t{font-size:13.5px;line-height:1.3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item .t small{display:block;color:var(--muted);font-size:11.5px;font-family:var(--font-mono)}
.pill{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;padding:2px 7px;border-radius:4px;white-space:nowrap}
.pill.keep,.pill.accepted{background:var(--ok-soft);color:var(--ok)}
.pill.fix{background:var(--warn-soft);color:var(--warn)}
.pill.drop,.pill.rejected{background:var(--bad-soft);color:var(--bad)}
.pill.na{background:var(--surface-2);color:var(--muted)}
.detail{display:flex;flex-direction:column;gap:14px;min-width:0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.doc-head{display:flex;flex-direction:column;gap:8px}
.doc-head .row{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;color:var(--ink-2);font-size:13.5px}
.doc-head .row .k{font-family:var(--font-mono);font-size:11.5px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase;margin-right:4px}
.brief{font-size:13.5px;color:var(--ink-2);border-left:2px solid var(--line-2);padding-left:12px;max-width:80ch}
.nav{display:flex;gap:6px;margin-left:auto}
.nav button{font:inherit;font-size:13px;padding:4px 10px;border:1px solid var(--line-2);background:var(--surface);color:var(--ink);border-radius:5px;cursor:pointer}
.nav button:hover{background:var(--surface-2)}
.facts{display:grid;gap:6px;margin:0;padding:0;list-style:none}
.facts li{display:grid;grid-template-columns:22px 1fr auto;gap:8px;align-items:baseline;font-size:14px}
.facts .m{font-family:var(--font-mono);font-size:13px;text-align:center}
.facts li.used .m{color:var(--accent)} .facts li.omit{color:var(--muted)} .facts li.omit .m{color:var(--line-2)}
.facts .rel{font-family:var(--font-mono);font-size:11.5px;color:var(--muted)}
.paper{font-size:14.5px;line-height:1.7;white-space:pre-wrap;word-wrap:break-word;max-width:78ch}
mark.tag{background:color-mix(in srgb,var(--g) 20%,transparent);color:var(--ink);border-radius:3px;padding:1px 3px;box-decoration-break:clone;-webkit-box-decoration-break:clone;border-bottom:2px solid var(--g)}
mark.tag i{font-style:normal;font-family:var(--font-mono);font-size:10px;color:var(--g);margin-left:4px;letter-spacing:.02em;vertical-align:1px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-family:var(--font-mono);font-size:11.5px;color:var(--muted);margin-top:12px;padding-top:10px;border-top:1px dashed var(--line)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--g);margin-right:5px;vertical-align:-1px}
.checks{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.check{display:flex;flex-direction:column;gap:8px}
.check .who{display:flex;align-items:center;justify-content:space-between;gap:8px}
.check .who h3{margin:0}
.check p{margin:0;font-size:13.5px;color:var(--ink-2)}
.scores{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}
.score{display:flex;flex-direction:column;align-items:center;padding:6px 2px;background:var(--surface-2);border-radius:5px}
.score b{font-family:var(--font-display);font-size:18px;font-variant-numeric:tabular-nums;line-height:1}
.score span{font-family:var(--font-mono);font-size:10px;color:var(--muted);margin-top:4px;letter-spacing:.03em}
.big{font-family:var(--font-display);font-size:28px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.big small{font-family:var(--font-mono);font-size:11px;color:var(--muted);font-weight:400;margin-left:4px}
.issues{margin:0;padding:0;list-style:none;display:grid;gap:8px}
.issues li{font-size:13px;border-left:2px solid var(--line-2);padding-left:10px}
.issues li .type{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--warn)}
.issues li .type.leak,.issues li .type.label,.issues li .type.relation{color:var(--bad)}
.issues li q{font-family:var(--font-mono);font-size:12px;color:var(--ink);quotes:"“" "”"}
.problems{margin:0;padding-left:18px;font-family:var(--font-mono);font-size:12.5px;color:var(--bad)}
.ents{display:flex;flex-wrap:wrap;gap:6px}
.ent{font-family:var(--font-mono);font-size:12px;padding:3px 8px;border-radius:4px;border:1px solid color-mix(in srgb,var(--g) 45%,transparent);background:color-mix(in srgb,var(--g) 12%,transparent)}
.ent.omit{opacity:.5;border-style:dashed;background:transparent}
.ent b{font-weight:500;color:var(--g);margin-right:6px}
.rels{margin:8px 0 0;padding:0;list-style:none;font-size:13px;display:grid;gap:4px}
.rels code{font-family:var(--font-mono);font-size:12px;color:var(--accent)}
.kbd{font-family:var(--font-mono);font-size:11px;border:1px solid var(--line-2);border-radius:4px;padding:0 5px;color:var(--muted)}
@media (max-width: 860px){
  header.top{grid-template-columns:1fr}
  .stats{grid-template-columns:repeat(2,1fr)}
  .reader{grid-template-columns:1fr}
  .rail{position:static;max-height:none}
  .list{max-height:260px}
}
@media (prefers-reduced-motion: no-preference){.item,.chip{transition:background .12s ease}}
</style>

<div class="wrap">
<header class="top">
  <div>
    <div class="eyebrow">kvkk_synth · runs/__RUN__</div>
    <h1>__RUN__ Generation Review</h1>
    <p class="lede">One thousand briefs from the Nemotron-PII catalogue across 30 domains, each with a density profile (zero, sparse, normal, dense), a candidate pool chosen by a local embedding model with centred scores, values in three tiers (sampler, writer-owned free text, real geography), and secondary persons only where the brief supports the role. Documents that failed a check were flagged or repaired by the same writer; nothing was silently dropped. Filters on the left cover the buckets and profiles; a stratified sample of 48 was reviewed by independent Sonnet reviewers.</p>
  </div>
  <div class="stats" id="stats"></div>
</header>

<section id="flow">
  <div class="section-head">
    <div><h2>The flow</h2><p>Sampling decides every value and relation; the model only writes the text. That is why the annotation can be checked mechanically afterwards: a tag is either one of the offered values with the right label, or it is a leak.</p></div>
  </div>
  <figure>
    <svg viewBox="0 0 1180 300" role="img" aria-label="Pipeline: taxonomy and name pools feed the sampler, which writes fact bundles; a Nemotron brief is attached; the writer model produces tagged Turkish text; the parser binds tags to offered values and applies leak guards, splitting accepted from rejected; a cheap judge and independent reviewers score both; everything lands in the run folder." xmlns="http://www.w3.org/2000/svg">
      <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
      <g font-family="IBM Plex Sans, system-ui, sans-serif" font-size="12.5" fill="currentColor" stroke="currentColor" stroke-width="1.2">
        <!-- inputs -->
        <rect x="20" y="40" width="150" height="46" rx="6" fill="none"/>
        <text x="95" y="59" text-anchor="middle" stroke="none" font-weight="600">Taxonomy v2.2</text>
        <text x="95" y="76" text-anchor="middle" stroke="none" font-size="11" opacity=".75">118 nodes · 105 relations</text>
        <rect x="20" y="104" width="150" height="46" rx="6" fill="none"/>
        <text x="95" y="123" text-anchor="middle" stroke="none" font-weight="600">Name pools · heuristics</text>
        <text x="95" y="140" text-anchor="middle" stroke="none" font-size="11" opacity=".75">2,224 names · checksums</text>
        <rect x="20" y="182" width="150" height="46" rx="6" fill="none" stroke-dasharray="4 3"/>
        <text x="95" y="201" text-anchor="middle" stroke="none" font-weight="600">Nemotron-PII briefs</text>
        <text x="95" y="218" text-anchor="middle" stroke="none" font-size="11" opacity=".75">type · domain · description</text>
        <!-- sampler -->
        <rect x="235" y="72" width="140" height="56" rx="6" fill="none" stroke-width="1.6"/>
        <text x="305" y="95" text-anchor="middle" stroke="none" font-weight="600">1 · Pool sampler</text>
        <text x="305" y="112" text-anchor="middle" stroke="none" font-size="11" opacity=".75">e5 top-15 + name + TCKN → values</text>
        <line x1="170" y1="63" x2="235" y2="92" marker-end="url(#arr)"/>
        <line x1="170" y1="127" x2="235" y2="108" marker-end="url(#arr)"/>
        <!-- attach brief -->
        <rect x="435" y="72" width="140" height="56" rx="6" fill="none" stroke-width="1.6"/>
        <text x="505" y="95" text-anchor="middle" stroke="none" font-weight="600">2 · Brief (e01 set)</text>
        <text x="505" y="112" text-anchor="middle" stroke="none" font-size="11" opacity=".75">same 24 briefs as e01 / e02</text>
        <line x1="375" y1="100" x2="435" y2="100" marker-end="url(#arr)"/>
        <text x="405" y="92" text-anchor="middle" stroke="none" font-family="IBM Plex Mono, monospace" font-size="10.5">bundles.jsonl</text>
        <path d="M170,205 C 300,205 380,150 435,112" fill="none" marker-end="url(#arr)"/>
        <!-- writer -->
        <rect x="635" y="66" width="150" height="68" rx="6" fill="none" stroke-width="2"/>
        <text x="710" y="90" text-anchor="middle" stroke="none" font-weight="600">3 · Writer</text>
        <text x="710" y="106" text-anchor="middle" stroke="none" font-size="11" opacity=".75">GPT-5.6 Luna, flex, reasoning medium</text>
        <text x="710" y="121" text-anchor="middle" stroke="none" font-size="11" opacity=".75">chooses facts, writes [value]label</text>
        <line x1="575" y1="100" x2="635" y2="100" marker-end="url(#arr)"/>
        <text x="605" y="92" text-anchor="middle" stroke="none" font-family="IBM Plex Mono, monospace" font-size="10.5">prompt</text>
        <!-- parser -->
        <rect x="845" y="66" width="150" height="68" rx="6" fill="none" stroke-width="1.6"/>
        <text x="920" y="90" text-anchor="middle" stroke="none" font-weight="600">4 · Parser + guards</text>
        <text x="920" y="106" text-anchor="middle" stroke="none" font-size="11" opacity=".75">tags → offsets, bind to values</text>
        <text x="920" y="121" text-anchor="middle" stroke="none" font-size="11" opacity=".75">reject invented PII</text>
        <line x1="785" y1="100" x2="845" y2="100" marker-end="url(#arr)"/>
        <text x="815" y="92" text-anchor="middle" stroke="none" font-family="IBM Plex Mono, monospace" font-size="10.5">text_tagged</text>
        <!-- outcomes -->
        <rect x="1045" y="44" width="115" height="40" rx="6" fill="none"/>
        <text x="1102" y="69" text-anchor="middle" stroke="none" font-weight="600">accepted · 23</text>
        <rect x="1045" y="116" width="115" height="40" rx="6" fill="none"/>
        <text x="1102" y="141" text-anchor="middle" stroke="none" font-weight="600">rejected · 1</text>
        <line x1="995" y1="90" x2="1045" y2="66" marker-end="url(#arr)"/>
        <line x1="995" y1="110" x2="1045" y2="134" marker-end="url(#arr)"/>
        <!-- judges -->
        <rect x="635" y="196" width="150" height="56" rx="6" fill="none"/>
        <text x="710" y="219" text-anchor="middle" stroke="none" font-weight="600">5 · Cheap judge</text>
        <text x="710" y="236" text-anchor="middle" stroke="none" font-size="11" opacity=".75">DeepSeek · coherence 1–5</text>
        <rect x="845" y="196" width="150" height="56" rx="6" fill="none"/>
        <text x="920" y="219" text-anchor="middle" stroke="none" font-weight="600">6 · Independent review</text>
        <text x="920" y="236" text-anchor="middle" stroke="none" font-size="11" opacity=".75">4 Sonnet reviewers · rubric</text>
        <path d="M1102,156 C 1102,224 1050,224 995,224" fill="none" marker-end="url(#arr)"/>
        <path d="M1090,84 L1090,116" fill="none" stroke-dasharray="3 3"/>
        <line x1="845" y1="224" x2="785" y2="224" marker-end="url(#arr)"/>
        <text x="1030" y="216" text-anchor="middle" stroke="none" font-family="IBM Plex Mono, monospace" font-size="10.5">all 24</text>
        <!-- run folder -->
        <rect x="235" y="196" width="340" height="56" rx="6" fill="none" stroke-dasharray="4 3"/>
        <text x="405" y="219" text-anchor="middle" stroke="none" font-weight="600">7 · Run folder</text>
        <text x="405" y="236" text-anchor="middle" stroke="none" font-size="11" opacity=".75">bundles · raw · records · rejects · judge · eval · README · code</text>
        <line x1="635" y1="224" x2="575" y2="224" marker-end="url(#arr)"/>
      </g>
    </svg>
    <figcaption>Left to right: what goes in, what each stage produces, and where every artefact of the run ends up. Dashed boxes are inputs or storage, solid boxes are processing stages.</figcaption>
  </figure>

  <div class="steps">
    <div class="step"><span class="n">1 · POOL BY EMBEDDING</span><h3>Three tiers of values</h3><p>The brief still chooses the pool (e5 top 15 plus name and TCKN). Tier A types in the pool get checksum-valid values from the taxonomy recipes. Tier C types (addresses, birthplace, registered place) come from one real locale per bundle: city → district → neighbourhood → street, postal code from the neighbourhood, plate province from the city. Tier B types are not filled at all: the writer is told which 30 kinds it may add with values of its own.</p><div class="io"><b>out</b> bundles.jsonl · 24 bundles, 16–18 facts each</div></div>
    <div class="step"><span class="n">2 · THE BRIEF</span><h3>Same 24 briefs as e01 and e02</h3><p>Each bundle keeps the Nemotron brief its bundle number had in e01: domain, document type and an English description. Only the catalogue is used, never Nemotron's text or values. Keeping the briefs fixed makes the three runs comparable: only the pool and the writer change.</p><div class="io"><b>in</b> e01 bundles.jsonl · <b>out</b> brief attached</div></div>
    <div class="step"><span class="n">3 · WRITE THE DOCUMENT</span><h3>The model decides which facts belong, then writes</h3><p>The prompt gives the brief, the candidate facts as a dependency tree, the exact values with their labels, the label definitions and the tagging rules. GPT-5.6 Luna reasons about which facts a real document of that type would carry, omits the rest, and writes Turkish with inline tags like <span class="kbd">[Ankara]city'da</span>, ending with a marker that proves the output was not cut off.</p><div class="io"><b>out</b> raw.parquet · 24 documents</div></div>
    <div class="step"><span class="n">4 · PARSE, BIND, GUARD</span><h3>Turn tags into offsets and refuse what cannot be verified</h3><p>Tags become character offsets. Each tagged value must be one of the offered values with the right label. Untagged repeats of a known value are auto-tagged. Guards reject invented persons, long numbers, plates, VINs, addresses, organisations and demographic lines that were never offered, plus stray brackets and meta-language. Relations are kept only when both of their entities were used.</p><div class="io"><b>out</b> records_full.jsonl · rejects.jsonl</div></div>
    <div class="step"><span class="n">5 · CHEAP JUDGE</span><h3>A fast coherence score for the loop</h3><p>DeepSeek reads each document with its stated type and scores plausibility from 1 to 5 with a one-sentence reason. In earlier pilots this judge was calibrated against Sonnet; in e01 it lands within one point on all 24 documents.</p><div class="io"><b>out</b> judge.json</div></div>
    <div class="step"><span class="n">6 · INDEPENDENT REVIEW</span><h3>Four reviewers who did not build the pipeline</h3><p>Claude Sonnet reviewers score fluency, coherence, tag boundaries, relation expression and leaks, give a keep / fix / drop verdict, say whether they agree with the parser, and quote every issue verbatim. Quotes are checked against the text; a blank sheet is left for human verdicts.</p><div class="io"><b>out</b> eval/out-01..04.json · summary.json · review.md</div></div>
  </div>
  <p class="note"><b>Reading the verdicts.</b> Omitting an offered fact is fine, that is the writer's job. Using a value without tagging it is a leak. A document can be accepted by the parser and still be marked <em>fix</em> or <em>drop</em> by the reviewers when the problem is semantic: a health number repurposed as a construction file number, a receipt written as a thank-you message.</p>
</section>

<section id="docs">
  <div class="section-head">
    <div><h2>The 1,000 documents</h2><p>Pick a document on the left. Tags are highlighted with a small label; colour follows the taxonomy group. Facts marked ✓ were used, ○ were offered and omitted; the cosine next to each value is the embedding score that put it in the pool. Use <span class="kbd">←</span> <span class="kbd">→</span> to move.</p></div>
  </div>
  <div class="reader">
    <aside class="rail">
      <div class="filters" id="filters" role="group" aria-label="Filter documents"></div>
      <div class="list" id="list" role="listbox" aria-label="Documents"></div>
    </aside>
    <div class="detail" id="detail"></div>
  </div>
</section>
</div>

<script type="application/json" id="data">__PAYLOAD__</script>
<script>
(function(){
  const DATA = JSON.parse(document.getElementById('data').textContent);
  const docs = DATA.docs, stats = DATA.stats;
  const GROUP_NAMES = {person:'person name',identity:'identity number',contact:'contact',civil:'birth / civil status',financial:'financial',digital:'digital identifier',location:'location / vehicle',sensitive:'health / special category',employment:'employment / organisation',legal:'legal / customer / registry'};
  const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const TAG = /\[([^\[\]]+)\]([a-z][a-z0-9_]*)/g;

  // header stats
  const v = stats.verdicts;
  document.getElementById('stats').innerHTML = [
    ['<b>'+stats.accepted+' / '+stats.docs+'</b><span>accepted by the parser</span>'],
    ['<b>'+stats.sonnet_overall.toFixed(2)+'</b><span>Sonnet overall (1–5)</span>'],
    ['<b>'+(v.keep||0)+' · '+(v.fix||0)+' · '+(v.drop||0)+'</b><span>keep · fix · drop</span>'],
    ['<b>'+stats.facts_used+' / '+stats.facts_offered+'</b><span>facts used / offered</span>'],
  ].map(x => '<div class="stat">'+x[0]+'</div>').join('');

  // filters
  const FILTERS = [
    ['all','all · '+docs.length, d=>true],
    ['keep','keep · '+docs.filter(d=>d.sonnet.verdict==='keep').length, d=>d.sonnet.verdict==='keep'],
    ['fix','fix · '+docs.filter(d=>d.sonnet.verdict==='fix').length, d=>d.sonnet.verdict==='fix'],
    ['drop','drop · '+docs.filter(d=>d.sonnet.verdict==='drop').length, d=>d.sonnet.verdict==='drop'],
    ['rejected','discarded · '+docs.filter(d=>d.bucket==='discarded').length, d=>d.bucket==='discarded'],
    ['flagged','flagged · '+docs.filter(d=>d.bucket==='flagged').length, d=>d.bucket==='flagged'],
    ['multi','multi-person · '+docs.filter(d=>d.persons&&d.persons.length>1).length, d=>d.persons&&d.persons.length>1],
    ['writer','writer-added · '+docs.filter(d=>d.writer_added.length).length, d=>d.writer_added.length>0],
    ['dense','dense · '+docs.filter(d=>d.profile==='dense').length, d=>d.profile==='dense'],
    ['zero','zero-PII · '+docs.filter(d=>d.profile==='zero').length, d=>d.profile==='zero'],
    ['reviewed','reviewed by Sonnet · '+docs.filter(d=>d.sonnet.verdict).length, d=>!!d.sonnet.verdict],
    ['issues','with issues · '+docs.filter(d=>d.sonnet.issues.length).length, d=>d.sonnet.issues.length>0],
  ];
  let filter = 'all', current = null;
  try { const s = localStorage.getItem('e08.current'); if (s) current = parseInt(s,10); } catch(e){}
  if (!docs.some(d=>d.id===current)) current = docs[0].id;

  function renderFilters(){
    document.getElementById('filters').innerHTML = FILTERS.map(f =>
      '<button class="chip" type="button" data-f="'+f[0]+'" aria-pressed="'+(filter===f[0])+'">'+esc(f[1])+'</button>').join('');
  }
  function visible(){ const fn = FILTERS.find(f=>f[0]===filter)[2]; return docs.filter(fn); }
  function renderList(){
    const vis = visible();
    document.getElementById('list').innerHTML = vis.map(d =>
      '<button class="item" type="button" role="option" data-id="'+d.id+'" aria-current="'+(d.id===current)+'">'+
      '<span class="id">#'+d.id+'</span>'+
      '<span class="t">'+esc(d.type)+'<small>'+esc(d.domain)+' · '+esc(d.format)+'</small></span>'+
      '<span class="pill '+(d.bucket!=='accepted'?'rejected':esc(d.sonnet.verdict||'na'))+'">'+(d.bucket!=='accepted'?esc(d.bucket):esc(d.sonnet.verdict||d.profile))+'</span>'+
      '</button>').join('');
  }
  function highlight(text){
    let out = '', pos = 0, m; TAG.lastIndex = 0;
    while ((m = TAG.exec(text))) {
      out += esc(text.slice(pos, m.index));
      const g = DATA.label_group[m[2]] || 'legal';
      out += '<mark class="tag" style="--g:var(--g-'+g+')">'+esc(m[1])+'<i>'+esc(m[2])+'</i></mark>';
      pos = m.index + m[0].length;
    }
    return out + esc(text.slice(pos));
  }
  function renderDetail(){
    const d = docs.find(x=>x.id===current); if (!d) return;
    const s = d.sonnet, sc = s.scores || {};
    const groups = [...new Set(d.entities.filter(e=>e.used).map(e=>e.group))];
    const parserPill = '<span class="pill '+d.parser.decision+'">'+d.parser.decision+'</span>';
    const html = [
      '<div class="card doc-head">',
        '<div class="row"><span class="eyebrow">document #'+d.id+'</span>'+parserPill+(s.verdict?'<span class="pill '+esc(s.verdict)+'">Sonnet: '+esc(s.verdict)+'</span>':'')+
          '<span class="nav"><button type="button" data-nav="-1" aria-label="Previous document">← prev</button><button type="button" data-nav="1" aria-label="Next document">next →</button></span></div>',
        '<h2>'+esc(d.type)+'</h2>',
        '<div class="row"><span><span class="k">domain</span>'+esc(d.domain)+'</span><span><span class="k">format</span>'+esc(d.format)+'</span><span><span class="k">subject</span>'+esc(d.subject)+'</span><span><span class="k">scenario</span>'+esc(d.scenario)+(d.madde6?' · Madde 6':'')+'</span><span><span class="k">length</span>'+d.chars.toLocaleString('en')+' chars</span></div>',
        '<p class="brief">'+esc(d.description)+'</p>',
      '</div>',
      '<div class="card"><h3>Facts offered · '+d.facts.filter(f=>f.used).length+' of '+d.facts.length+' used</h3><ul class="facts">'+
        d.facts.map(f=>'<li class="'+(f.used?'used':'omit')+'"><span class="m">'+(f.used?'✓':'○')+'</span><span>'+esc(f.fact)+'</span><span class="rel">'+esc(f.relation)+'</span></li>').join('')+'</ul>'+
        (d.persons&&d.persons.length>1?'<div style="margin:10px 0 4px"><span class="eyebrow">named persons</span></div><ul class="rels">'+d.persons.map(p=>'<li>'+esc(p.name)+' <code>'+esc(p.kind)+'</code> '+esc(p.role_en)+'</li>').join('')+'</ul>':'')+'<div class="ents" style="margin-top:12px">'+(d.locale?'<span class="ent" style="--g:var(--g-contact)"><b>locale</b>'+esc(d.locale.city)+' / '+esc(d.locale.town)+' / '+esc(d.locale.neighbourhood)+' · '+esc(d.locale.zip)+'</span>':'')+d.entities.map(e=>'<span class="ent '+(e.used?'':'omit')+'" style="--g:var(--g-'+e.group+')"><b>'+esc(e.label)+'</b>'+esc(e.value)+(e.optional?' <small>(optional)</small>':'')+(e.cosine!=null?' <small>· '+e.cosine.toFixed(3)+'</small>':'')+'</span>').join('')+(d.writer_added.length?'<div style="margin-top:8px"><span class="eyebrow">writer-added, validated</span></div><div class="ents" style="margin-top:6px">'+d.writer_added.map(w=>'<span class="ent" style="--g:var(--g-'+(DATA.label_group[w.label]||'legal')+')"><b>'+esc(w.label)+'</b>'+esc(w.value)+' <small>· '+esc(w.status)+'</small></span>').join('')+'</div>':'')+'</div>'+
        (d.negatives.length?'<div style="margin-top:8px"><span class="eyebrow">negatives (not the subject\'s)</span></div><ul class="rels">'+d.negatives.map(n=>'<li>'+esc(n.head)+' <code>'+esc(n.relation)+'</code> '+esc(n.tail)+' <small>= false</small></li>').join('')+'</ul>':'')+(d.relations_kept.length?'<ul class="rels">'+d.relations_kept.map(r=>'<li>'+esc(r.head)+' <code>'+esc(r.relation)+'</code> '+esc(r.tail)+'</li>').join('')+'</ul>':'')+
      '</div>',
      '<div class="card"><h3>Document as written, tags highlighted</h3><div class="paper" style="margin-top:8px">'+highlight(d.text_tagged)+'</div>'+
        (groups.length?'<div class="legend">'+groups.map(g=>'<span style="--g:var(--g-'+g+')"><i></i>'+esc(GROUP_NAMES[g])+'</span>').join('')+'</div>':'')+
      '</div>',
      '<div class="checks">',
        '<div class="card check"><div class="who"><h3>Parser</h3>'+parserPill+'</div>'+
          (d.parser.problems.length?'<ul class="problems">'+d.parser.problems.map(p=>'<li>'+esc(p)+'</li>').join('')+'</ul>':'<p>Every tag bound to an offered value; no guard fired.</p>')+
          (s.parser_agreement?'<p><b style="color:var(--ink)">Reviewer '+(s.parser_agreement.agree?'agrees':'disagrees')+'.</b> '+esc(s.parser_agreement.note)+'</p>':'')+'</div>',
        '<div class="card check"><div class="who"><h3>Cheap judge · DeepSeek</h3><span class="big">'+(d.judge.score==null?'—':d.judge.score)+'<small>/ 5</small></span></div><p>'+esc(d.judge.reason)+'</p></div>',
        '<div class="card check"><div class="who"><h3>Sonnet reviewer</h3>'+(s.verdict?'<span class="pill '+esc(s.verdict)+'">'+esc(s.verdict)+'</span>':'')+'</div>'+
          (s.scores?'<div class="scores">'+[['fluency','flu'],['coherence','coh'],['tag_boundaries','tags'],['relation_expression','rel'],['completeness_and_leaks','leak']].map(k=>'<div class="score"><b>'+sc[k[0]]+'</b><span>'+k[1]+'</span></div>').join('')+'</div>':'')+
          (s.facts_used_naturally!=null?'<p>Facts used naturally: '+s.facts_used_naturally+'</p>':'')+
          (s.issues.length?'<ul class="issues">'+s.issues.map(i=>'<li><span class="type '+esc(i.type)+'">'+esc(i.type)+'</span><br><q>'+esc(i.quote)+'</q><br>'+esc(i.note)+'</li>').join('')+'</ul>':'<p>No issues reported.</p>')+
        '</div>',
      '</div>'
    ].join('');
    document.getElementById('detail').innerHTML = html;
  }
  function select(id, scroll){
    current = id; try { localStorage.setItem('e08.current', String(id)); } catch(e){}
    renderList(); renderDetail();
    if (scroll) { const el = document.getElementById('docs'); el && el.scrollIntoView({block:'start'}); }
    const cur = document.querySelector('.item[aria-current="true"]'); cur && cur.scrollIntoView({block:'nearest'});
  }
  function move(delta){
    const vis = visible(); if (!vis.length) return;
    let i = vis.findIndex(d=>d.id===current); i = (i + delta + vis.length) % vis.length;
    select(vis[i].id, false);
  }
  document.getElementById('filters').addEventListener('click', e => {
    const b = e.target.closest('.chip'); if (!b) return;
    filter = b.dataset.f; renderFilters();
    const vis = visible(); if (!vis.some(d=>d.id===current) && vis.length) current = vis[0].id;
    renderList(); renderDetail();
  });
  document.getElementById('list').addEventListener('click', e => { const b = e.target.closest('.item'); if (b) select(parseInt(b.dataset.id,10), false); });
  document.getElementById('detail').addEventListener('click', e => { const b = e.target.closest('[data-nav]'); if (b) move(parseInt(b.dataset.nav,10)); });
  document.addEventListener('keydown', e => {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    if (e.key === 'ArrowRight') { move(1); e.preventDefault(); } else if (e.key === 'ArrowLeft') { move(-1); e.preventDefault(); }
  });
  renderFilters(); renderList(); renderDetail();
})();
</script>
"""
_run_name = os.path.basename(os.getcwd())  # [pkg] the run folder the viewer is built in
open(_args.out, "w", encoding="utf-8").write(HTML.replace("__PAYLOAD__", payload).replace("__RUN__", _run_name))
print(_args.out, len(docs), "docs,", os.path.getsize(_args.out) // 1024, "KB")
