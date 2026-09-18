import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQUIRED = [
    "pipeline/sampler8.py",
    "pipeline/sampler7.py",
    "pipeline/pool_sampler.py",
    "pipeline/persons.py",
    "pipeline/tiers.py",
    "pipeline/values8.py",
    "pipeline/geoloc.py",
    "pipeline/kinship_claims.py",
    "pipeline/embed_score.py",
    "pipeline/role_scores.py",
    "pipeline/gen_direct.py",
    "pipeline/verify_relations7.py",
    "pipeline/judge_writer_relations.py",
    "pipeline/judge_negatives.py",
    "pipeline/judge_coherence.py",
    "pipeline/scan_untagged.py",
    "pipeline/repair.py",
    "pipeline/repair_sex.py",
    "pipeline/audit_relations.py",
    "pipeline/build_dataset.py",
    "pipeline/select_review.py",
    "pipeline/aggregate_review.py",
    "pipeline/build_viewer7.py",
    "pipeline/analyse_distribution.py",
    "pipeline/sample_facts.py",
    "pipeline/parse_facts.py",
    "pipeline/v2/gen_facts.py",
    "pipeline/v2/parse_facts.py",
    "pipeline/v2/harmony.py",
    "pipeline/v2/scenarios.py",
    "pipeline/v2/sample_facts.py",
    "pipeline/v2/names/first_names.csv",
    "pipeline/v2/names/surnames.csv",
    "taxonomy/index.json",
    "taxonomy/kvkk_sampler.py",
    "taxonomy/tr_heuristics.py",
    "taxonomy/nodes.jsonl",
    "taxonomy/taxonomies/16_relations_roles.json",
    "geo/build_geo.py",
    "geo/README.md",
    "briefs/README.md",
    "briefs/nemotron_catalogue.parquet",
    "briefs/nemotron_catalogue_fresh_for_e08.parquet",
    "briefs/e07_types.json",
    "prompts/review/RUBRIC.md",
    "prompts/review/REVIEWER_PROMPT.md",
    "runs/README.md",
]


def test_required_files_exist():
    missing = [p for p in REQUIRED if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, missing


def test_no_run_artifacts_or_caches_copied():
    for dirpath, dirs, files in os.walk(os.path.join(ROOT, "pipeline")):
        assert "dd-artifacts" not in dirs, dirpath
        assert not any(f.endswith(".parquet") for f in files), (dirpath, files)


def test_taxonomy_has_16_files_and_113_relations():
    import sys

    tax = os.path.join(ROOT, "taxonomy", "taxonomies")
    assert len([f for f in os.listdir(tax) if f.endswith(".json")]) == 16
    sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
    sys.modules.pop("kvkk_sampler", None)
    import kvkk_sampler as ks

    ents, rels = ks.load_taxonomy()
    assert len(ents) == 118 and len(rels) == 113


def test_e07_types_list():
    t = json.load(open(os.path.join(ROOT, "briefs", "e07_types.json"), encoding="utf-8"))
    assert len(t) == 912 and all(x == x.lower() for x in t)
