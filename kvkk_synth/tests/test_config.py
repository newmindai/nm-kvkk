import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config


def test_defaults_are_the_e08_values():
    assert config.get("writer.model") == "openai/gpt-5.6-luna"
    assert config.get("writer.provider") == ["openai/flex"]
    assert config.get("writer.reasoning_effort") == "medium"
    assert config.get("writer.max_tokens") == 12288
    assert config.get("judge.model") == "deepseek/deepseek-v4-flash"
    assert config.get("run.block") == 2000
    assert config.get("run.review_n") == 96
    assert config.get("sampler.seed") == 8


def test_run_folder_snapshot_wins(tmp_path):
    (tmp_path / "config.yaml").write_text("writer:\n  model: test/model\n", encoding="utf-8")
    pipeline = os.path.join(ROOT, "pipeline")
    code = f"import sys; sys.path.insert(0, {pipeline!r}); import config; print(config.get('writer.model'), config.get('judge.model'))"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert out.stdout.strip() == "test/model deepseek/deepseek-v4-flash"


def test_scripts_read_models_from_config():
    for f, needle in [
        ("gen_direct.py", 'config.get("writer.model")'),
        ("repair.py", 'config.get("writer.model")'),
        ("repair_sex.py", 'config.get("writer.model")'),
        ("verify_relations7.py", 'config.get("judge.model")'),
        ("judge_writer_relations.py", 'config.get("judge.model")'),
        ("judge_negatives.py", 'config.get("judge.model")'),
        ("scan_untagged.py", 'config.get("judge.model")'),
        ("judge_coherence.py", 'config.get("judge.model")'),
    ]:
        src = open(os.path.join(ROOT, "pipeline", f), encoding="utf-8").read()
        assert needle in src, f
        assert '"openai/gpt-5.6-luna"' not in src and '"deepseek/deepseek-v4-flash"' not in src, (
            f"{f} still hard-codes a model id"
        )
        assert '["openai/flex"]' not in src, f"{f} still hard-codes the provider pin"
