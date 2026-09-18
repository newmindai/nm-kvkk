#!/usr/bin/env python3
"""v2 generation — realize scenario bundles as Turkish documents with inline tags.

Changes vs v1 (FIX B): genre comes from the bundle (scenario pool) instead of a Nemotron brief;
the "add a section if a fact does not fit" instruction is gone; no markdown; optional name-part
entities are allowed but not required; max_tokens 6144.
Usage: python gen_facts.py --bundles bundles-gated.jsonl --out raw.parquet [--regenre]
"""

import argparse
import json
import os
import random
import sys

import pandas as pd

# [pkg] NeMo Data Designer is imported lazily: only the legacy __main__ path uses it; gen_direct.py renders the prompts directly

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PARENT, "..", "taxonomy"))  # [pkg]
import kvkk_sampler as ks  # noqa: E402

sys.path.insert(0, PARENT)
import persons as pers  # noqa: E402  [e06]
import prompts  # [pkg] prompt files
import tiers  # noqa: E402  [e05] writer-owned kinds
from scenarios import GENRES  # noqa: E402

MODEL_ID = "deepseek/deepseek-v4-flash"

SYSTEM = prompts.load("writer_system.txt")  # [pkg] see kvkk_synth/prompts/

PROMPT = prompts.load("writer_required.j2")  # [pkg] see kvkk_synth/prompts/


# Optional-facts mode: a Nemotron brief sets the document; the model decides in its own reasoning
# which candidate facts a real document of that type would contain, and omits the rest.
PROMPT_OPTIONAL = prompts.load("writer_pool.j2")  # [pkg] see kvkk_synth/prompts/


def _extra_body(model_id):
    if model_id.startswith("deepseek/"):
        return {"reasoning": {"enabled": False}}
    if model_id.startswith("google/"):
        return {
            "provider": {"only": ["google-vertex/global/flex"], "allow_fallbacks": False},
            "reasoning": {"effort": "low"},
        }
    if model_id.startswith("openai/"):  # [e02] OpenAI flex tier, reasoning kept on (medium)
        return {"provider": {"only": ["openai/flex"], "allow_fallbacks": False}, "reasoning": {"effort": "medium"}}
    if model_id.startswith("meta/"):
        return {}  # Muse Spark: keep its reasoning on — it is meant to judge plausibility
    return {"reasoning": {"enabled": False}}


def assign_nemotron_briefs(bundles, seed_docs):
    """Attach a random Nemotron brief to each bundle (persisted so several models get the same brief)."""
    for b in bundles:
        if "brief" not in b:
            d = seed_docs.sample(1).iloc[0]
            b["brief"] = {
                "domain": d.domain,
                "document_type": d.document_type,
                "description": d.document_description,
                "document_format": d.document_format,
            }


def facts_tree(b):
    """Render candidate facts with their dependencies: a fact whose entities do not touch the
    subject is nested under the fact that anchors it (a CVV under its card, a verdict under its
    case), so the writer keeps them together or drops them together."""
    subj = b["subject_id"]
    top = [r for r in b["relations"] if subj in (r["head"], r["tail"])]
    dep = [r for r in b["relations"] if subj not in (r["head"], r["tail"])]
    lines = []
    for r in top:
        lines.append(f"- {r['fact_tr']}")
        for d in dep:
            if {d["head"], d["tail"]} & {r["head"], r["tail"]}:
                lines.append(f"    - (only together with the fact above) {d['fact_tr']}")
    placed = {id(d) for r in top for d in dep if {d["head"], d["tail"]} & {r["head"], r["tail"]}}
    lines += [f"- {d['fact_tr']}" for d in dep if id(d) not in placed]
    return "\n".join(lines)


def profile_block(b):
    """[e07] density profile -> what the writer is asked for."""
    prof, target = b.get("profile"), b.get("fact_target")
    if not prof or prof == "zero":
        return ""
    s = f"DENSITY. Use about {target} of the offered facts (the ones that genuinely belong; never fewer than 1 besides the name)."
    if prof == "dense":
        s += " Lay the document out as a form or record: field labels with values, one item per line, several sections, as a real filled-in form of this type would look."
    if prof == "sparse":
        s += (
            " This document mentions personal data only in passing: most of it is ordinary content. If the document is a form, "
            "include only the fields you actually fill — never write placeholder answers such as 'Belirtilmemiş' or 'Beyan edilmiştir'."
        )
    return s


PROMPT_ZERO = prompts.load("writer_zero.j2")  # [pkg] see kvkk_synth/prompts/


def build_seed(bundles, ents_meta, regenre=False, optional_facts=False):
    rows = []
    for b in bundles:
        genre, fmt = b["genre"], b["document_format"]
        domain, description = b["scenario"], ""
        if b.get("brief"):
            genre, fmt = b["brief"]["document_type"], b["brief"]["document_format"]
            domain, description = b["brief"]["domain"], b["brief"]["description"]
        elif regenre:
            others = [g for g in GENRES[b["scenario"]] if g[0] != genre]
            genre, fmt = random.choice(others) if others else (genre, fmt)
            b["genre"], b["document_format"] = genre, fmt
        subject = next(e for e in b["entities"] if e["id"] == b["subject_id"])
        required = [e for e in b["entities"] if not e.get("optional")]
        optional = [e for e in b["entities"] if e.get("optional")]
        used_nodes = sorted({e["label"] for e in b["entities"]})
        rows.append(
            {
                "bundle_id": b["bundle_id"],
                "scenario": b["scenario"],
                "genre": genre,
                "document_format": fmt,
                "domain": domain,
                "description": description,
                "subject": subject["value"],
                "facts_block": facts_tree(b)
                if optional_facts
                else "\n".join(f"- {r['fact_tr']}" for r in b["relations"]),
                "entity_block": "\n".join(f"- {e['id']}: [{e['value']}]{e['label']}" for e in required),
                "optional_block": (
                    "The person's name parts may also appear on their own (e.g. in a salutation or signature); "
                    "if they do, tag them: " + ", ".join(f"[{e['value']}]{e['label']}" for e in optional) + "\n"
                )
                if optional
                else "",
                "desc_block": "\n".join(f"- {n}: {ents_meta[n]['description']}" for n in used_nodes),
                "tierb_block": tiers.prompt_block(ents_meta) if optional_facts else "",  # [e05]
                "persons_block": pers.prompt_block(b["persons"])
                if b.get("persons") and len(b["persons"]) > 1
                else "",  # [e06]
                "profile_block": profile_block(b),  # [e07]
            }
        )
    return pd.DataFrame(rows)


def load_config_builder(model_id, seed_path, max_tokens, prompt=PROMPT):
    import data_designer.config as dd  # [pkg] lazy

    model = dd.ModelConfig(
        alias="gen",
        model=model_id,
        provider="openrouter",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.85, top_p=0.95, max_tokens=max_tokens, extra_body=_extra_body(model_id)
        ),
    )
    b = dd.DataDesignerConfigBuilder(model_configs=[model])
    b.with_seed_dataset(dd.LocalFileSeedSource(path=seed_path))
    b.add_column(dd.LLMTextColumnConfig(name="text_tagged", model_alias="gen", system_prompt=SYSTEM, prompt=prompt))
    return b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default="bundles-gated.jsonl")
    ap.add_argument("--out", default="raw.parquet")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--regenre", action="store_true", help="pick a different genre from the scenario pool (retries)")
    ap.add_argument(
        "--briefs",
        choices=["scenario", "nemotron"],
        default="scenario",
        help="nemotron: attach a random Nemotron-PII brief to each bundle (persisted for reuse across models)",
    )
    ap.add_argument("--facts-optional", action="store_true", help="the model chooses which candidate facts to include")
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    ents_meta, _ = ks.load_taxonomy()
    bundles = [json.loads(l) for l in open(a.bundles, encoding="utf-8")]
    if a.briefs == "nemotron":
        assign_nemotron_briefs(bundles, pd.read_parquet(os.path.join(PARENT, "seed-200.parquet")))
    seed = build_seed(bundles, ents_meta, regenre=a.regenre, optional_facts=a.facts_optional)
    if a.regenre or a.briefs == "nemotron":  # persist genres/briefs so parser, report and other models see them
        with open(a.bundles, "w", encoding="utf-8") as f:
            for b in bundles:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
    seed_path = a.out.replace(".parquet", "-seed.parquet")
    seed.to_parquet(seed_path, index=False)
    print(f"seed: {len(seed)} rows -> {seed_path}")

    from data_designer.interface import DataDesigner  # [pkg] lazy

    designer = DataDesigner(artifact_path=os.path.join(PARENT, "dd-artifacts"))
    builder = load_config_builder(a.model, seed_path, a.max_tokens, PROMPT_OPTIONAL if a.facts_optional else PROMPT)
    if a.preview:
        df = designer.preview(builder, num_records=a.preview).dataset
    else:
        res = designer.create(
            builder, num_records=len(seed), dataset_name="v2-" + os.path.basename(a.out).replace(".parquet", "")
        )
        df = res.load_dataset()
    df.to_parquet(a.out, index=False)
    print("WROTE", a.out, df.shape)
