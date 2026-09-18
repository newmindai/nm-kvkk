import ast
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E08 = os.path.abspath(
    os.path.join(ROOT, "..", "experiments", "kvkk_relations_pilot", "runs", "2026-09-09-e08-luna-5000", "code")
)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import prompts

MAP = {  # prompt file -> (e08 source file, constant name)
    "writer_system.txt": ("v2/gen_facts.py", "SYSTEM"),
    "writer_required.j2": ("v2/gen_facts.py", "PROMPT"),
    "writer_pool.j2": ("v2/gen_facts.py", "PROMPT_OPTIONAL"),
    "writer_zero.j2": ("v2/gen_facts.py", "PROMPT_ZERO"),
    "repair.txt": ("repair.py", "PROMPT"),
    "repair_sex.txt": ("repair_sex.py", "PROMPT"),
    "judge_relations.txt": ("verify_relations7.py", "PROMPT"),
    "judge_writer_relations.txt": ("judge_writer_relations.py", "PROMPT"),
    "judge_negatives.txt": ("judge_negatives.py", "PROMPT"),
    "judge_coherence.txt": ("judge_coherence.py", "PROMPT"),
    "scan_untagged.txt": ("scan_untagged.py", "PROMPT"),
}


def constant(src_file, name):
    tree = ast.parse(open(src_file, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


@pytest.mark.parametrize("fname,src", MAP.items())
def test_prompt_file_equals_e08_constant(fname, src):
    if not os.path.exists(E08):
        pytest.skip("e08 source tree not present")
    assert prompts.load(fname) == constant(os.path.join(E08, src[0]), src[1])


def test_loader_strips_exactly_one_trailing_newline():
    assert not prompts.load("writer_system.txt").endswith("\n")
    assert prompts.load("judge_relations.txt").count("{text}") == 1


def test_pipeline_scripts_have_no_inline_prompt_constants():
    for src, name in set(MAP.values()):
        tree = ast.parse(open(os.path.join(ROOT, "pipeline", src), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                assert not isinstance(node.value, ast.Constant), f"{src}:{name} still inline"
