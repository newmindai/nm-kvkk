"""Consolidate the scored runs into results/benchmark/RESULTS.md (document-level headline tables, the
kvkk19 view, sentence-level rows, per-label breakdown, speed). Regenerate after every run.

  python scripts/make_results_md.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "results" / "benchmark"
MAIN = [
    "s2short-500",
    "nm6k-2stage",
    "nm6k-en-2000",
    "rele07tr",
    "mursit-fullrel",
    "kvkk-champion",
    "gliner25-zero-shot",
    "stack-new (ner-new + kvkk-v9-cus)",
    "stack-prod (ner-v1 + kvkk-v3)",
    "kvkk-v9-cus",
    "kvkk-v8",
    "kvkk-v3",
    "kvkk-v7",
    "kvkk-v8-cus",
    "ner-new",
    "ner-old",
    "ner-new-v2",
    "kvkk-v3-anony",
    "ner-v1-anony",
    "redact",
]
FAMILY = {
    "s2short-500": "ours",
    "nm6k-2stage": "ours",
    "nm6k-en-2000": "ours",
    "rele07tr": "ours (production)",
    "mursit-fullrel": "ours (Mursit)",
    "kvkk-champion": "ours (19-label entity model)",
    "gliner25-zero-shot": "open, zero-shot",
    "redact": "open",
    "cosmos-pii": "open (ModernBERT-TR PII)",
}


def fam(name):
    base = name.replace("@doc", "")
    if base in FAMILY:
        return FAMILY[base]
    if base.startswith("stack"):
        return "served stack"
    if base.startswith("kvkk"):
        return "served KVKK"
    if base.startswith("ner"):
        return "served NER"
    return ""


def table(rows, subset_label, unit=None, per_label=None, gold=None):
    rows = [r for r in rows if unit is None or r["unit"] == unit]
    rows.sort(key=lambda r: -r["strict"]["micro"]["f1"])
    out = "| model | family | strict P | R | **F1** | macro F1 | lenient F1 | lenient macro | n pred | ms/unit | s/doc |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        s, l, t = r["strict"], r["lenient"], r["timing"]
        out += (
            f"| {r['model']} | {fam(r['model'])} | {s['micro']['precision'] * 100:.1f} | {s['micro']['recall'] * 100:.1f} | **{s['micro']['f1'] * 100:.1f}** | "
            f"{s['macro']['f1'] * 100:.1f} | {l['micro']['f1'] * 100:.1f} | {l['macro']['f1'] * 100:.1f} | {r['n_pred']} | {t['ms_per_unit']:.0f} | {t['s_per_doc']:.2f} |\n"
        )
    if per_label:
        out += (
            "\nPer-label strict F1 (gold count):\n\n| model | "
            + " | ".join(f"{l} ({gold.get(l, 0)})" for l in per_label)
            + " |\n|---|"
            + "---:|" * len(per_label)
            + "\n"
        )
        for r in rows:
            out += (
                f"| {r['model']} | "
                + " | ".join(f"{r['strict']['per_label'].get(l, {}).get('f1', 0) * 100:.0f}" for l in per_label)
                + " |\n"
            )
    return out


def section(run_name: str, title: str, note: str, tag: str = "") -> str:
    run = RUNS / run_name
    sfx = f"_{tag}" if tag else ""
    if not (run / f"results_stack21{sfx}.json").exists():
        return f"## {title}\n\n*(not scored yet)*\n"
    s21 = json.loads((run / f"results_stack21{sfx}.json").read_text(encoding="utf-8"))
    k19 = json.loads((run / f"results_kvkk19{sfx}.json").read_text(encoding="utf-8"))
    gold_counts = {}
    for line in (run / "gold.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            for sp in json.loads(line)["spans"]:
                gold_counts[sp["label"]] = gold_counts.get(sp["label"], 0) + 1
    labels21 = [
        l
        for l in [
            "full_name",
            "national_id_number",
            "full_address",
            "company_name",
            "phone_number",
            "email_address",
            "iban",
            "date_of_birth",
            "place_of_birth",
            "card_number",
            "ip_address",
            "passport_number",
            "drivers_license_number",
            "license_plate",
            "id_document_serial",
            "gps_coordinates",
            "customer_number",
            "url_with_pii",
            "device_id",
            "social_media_handle",
            "audiovisual_record_reference",
        ]
        if gold_counts.get(l, 0) >= 3
    ]
    md = f"## {title}\n\n{note}\n\n"
    md += f"### {run_name} · document-level input · STACK-21 ({s21['n_gold_subset']} gold spans)\n\n" + table(
        s21["rows"], "stack21", unit="doc", per_label=labels21, gold=gold_counts
    )
    md += (
        f"\n### {run_name} · document-level input · KVKK-19 ({k19['n_gold_subset']} gold spans; NER-only models score 0 by construction)\n\n"
        + table([r for r in k19["rows"] if not r["model"].startswith("ner")], "kvkk19", unit="doc")
    )
    c14 = run / f"results_cosmos14{sfx}.json"
    if c14.exists():
        c = json.loads(c14.read_text(encoding="utf-8"))
        md += (
            f"\n### {run_name} · document-level input · COSMOS-14 ({c['n_gold_subset']} gold spans; the labels every model incl. cosmos-pii has)\n\n"
            + table(c["rows"], "cosmos14", unit="doc")
        )
    md += f"\n### {run_name} · sentence-level input (production splitter) · STACK-21\n\n" + table(
        s21["rows"], "stack21", unit="sentence"
    )
    return md


VEK_NOTE = """20 real notary documents (86–212 words), 209 gold spans, 188 inside STACK-21. **Document-level rows are
the fair comparison** on this set: the production sentencizer cuts 28 of the 37 gold addresses and 9 of
the 34 company names in two (they contain "Mah.", "No.", "Kat.", "A.Ş."), so at sentence level no
model can return those spans whole — every model's `full_address` F1 sits at 14–19 there and 86–89
at document level. The served models accept a whole document as one `sentence` (their API tokenises
internally), so the document-level rows use exactly the same input for everyone."""
NM6K_NOTE = """The nm-kvkk-pii-6K test split **minus the 52 documents rele07tr trained on** (`datasets/nm6k/
contaminated_for_rele07.json`) = 550 synthetic documents, gold surfaces expanded to every whole-word
occurrence. **In-distribution for our nm6k-trained models** (s2short-500, nm6k-2stage, nm6k-en-2000,
mursit-fullrel) — they have seen this generator; rele07tr, kvkk-champion, cosmos and the served models
have not. The STACK-21 gold here is 60 % `full_name`, so models without a name label (the KVKK
services alone) are capped near 40 whatever they do on their own labels — read the KVKK-19 table for
them. Only 68 gold spans cross a production sentence boundary, so sentence- and document-level rows
are comparable. `device_id` (MAC address in the served models) is an approximate mapping and scores
0 for them."""

md = """# Cross-model benchmark — results

Pipeline, subsets and scoring: `README.md`. Served models and label mapping: `EXTERNAL_MODELS.md`.
Every number: threshold 0.5 for GLiNER models, strict = exact character offsets + label, lenient =
overlapping offsets + same label, macro = mean F1 over subset labels with gold in the set, speed =
model time only (sentencizing excluded) — served models on the legacy encoder-NER pipeline (over the
LAN), our models on a Mac M4 (MPS), redact on the Mac CPU. Regenerate with `make_results_md.py`;
the raw tables per run are `results/benchmark/<set>/results_<subset>.md`; the prediction files under `predictions/` hold every
prediction behind these numbers.

"""
MIXED_NOTE = """Real-world dataset ground truth v2 (`datasets/gt/mixed_v2/gt.jsonl`): the 118 accepted documents (48 accept + 70
accept_with_fixes) — 100 new real documents across 18 groups plus the 20 vekaletname_v1 documents; 1,928
gold spans over 71 labels, 1,323 of them inside STACK-21. Recall-audited (est. 98.1 %) but audit
additions are not precision-checked, so some gold over-annotation is expected (the source README).
Broader and harder than vekaletname: `company_name` (288 spans) and `full_address` (168) dominate the
errors of every model. Document-level rows are the fair comparison for the same reason as in §1."""
md += (
    section("vekaletname", "1. Real documents — vekaletname_v1", VEK_NOTE)
    + "\n"
    + section("mixed_v2", "2. Real documents — real-world dataset ground truth v2 (mixed_v2, 118 docs)", MIXED_NOTE)
    + "\n"
    + section("nm6k", "3. Synthetic — nm6k test split (clean 550)", NM6K_NOTE, tag="clean550")
)
(RUNS / "RESULTS.md").write_text(md, encoding="utf-8")
print("wrote", RUNS / "RESULTS.md", len(md.splitlines()), "lines")
