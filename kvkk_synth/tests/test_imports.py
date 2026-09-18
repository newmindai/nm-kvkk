import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPE = os.path.join(ROOT, "pipeline")
SHADOW = (
    "kvkk_sampler",
    "tiers",
    "tr_heuristics",
    "kinship_claims",
    "persons",
    "values8",
    "harmony",
    "scenarios",
    "sample_facts",
    "gen_facts",
    "prompts",
    "config",
)


def _fresh(modname, path):
    for k in SHADOW:
        sys.modules.pop(k, None)
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_taxonomy_loads_118_nodes_113_relations():
    sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
    sys.modules.pop("kvkk_sampler", None)
    import kvkk_sampler as ks

    ents, rels = ks.load_taxonomy()
    assert len(ents) == 118 and len(rels) == 113


def test_tiers_and_kinship_import_from_pipeline():
    t = _fresh("tiers", os.path.join(PIPE, "tiers.py"))
    assert len(t.TIER_B) == 30
    k = _fresh("kinship_claims", os.path.join(PIPE, "kinship_claims.py"))
    assert k.sentence("child_of", "Ayşe", "Ali", "F") == "Ayşe, Ali'in kızıdır."


def test_gen_facts_imports_without_data_designer():
    sys.modules.pop("data_designer", None)
    sys.modules["data_designer"] = None  # importing it must not be attempted at module level
    try:
        g = _fresh("gen_facts", os.path.join(PIPE, "v2", "gen_facts.py"))
        assert "===SON===" in g.PROMPT_OPTIONAL
    finally:
        sys.modules.pop("data_designer", None)


def test_no_source_line_points_at_code_or_pilot_paths():
    bad = []
    for dp, _, fs in os.walk(PIPE):
        for f in fs:
            if not f.endswith(".py"):
                continue
            src = open(os.path.join(dp, f), encoding="utf-8").read()
            for needle in (
                '"code/taxonomy"',
                '"../../v2/names',
                'HERE, "geo"',
                'RUN, "briefs5000',
                'RUN, "scores-e5-cat',
            ):
                if needle in src:
                    bad.append((f, needle))
    assert not bad, bad
