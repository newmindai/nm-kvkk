"""train.py pieces that can be verified offline with the tiny fixtures from the GLiNER2 test suite."""

import json
import sys
from pathlib import Path

import torch

GLINER2 = Path(__file__).resolve().parents[1] / "GLiNER2"
sys.path.insert(0, str(GLINER2))
from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD  # noqa: E402
from tests.fixtures.tiny_encoder import build_tiny_encoder_config  # noqa: E402
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer  # noqa: E402
from train import DeviceTrainer, build_extractor_config, load_boundary_head, warm_start_head  # noqa: E402


def _tiny(seed, **overrides):
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor

    tokenizer = build_tiny_tokenizer()
    cfg = ExtractorConfig(
        model_name="tiny-bert-fixture", architecture="boundary", boundary_head=dict(TINY_BOUNDARY_HEAD), **overrides
    )
    torch.manual_seed(seed)
    return BoundaryExtractor(
        cfg, encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)), tokenizer=tokenizer
    )


def test_load_boundary_head_applies_overrides(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"boundary_head": {"enable_records": True, "pool_size": 192}}), encoding="utf-8"
    )
    head = load_boundary_head(tmp_path, {"enable_records": False})
    assert head["enable_records"] is False and head["pool_size"] == 192


def test_build_extractor_config_carries_the_pilot_settings(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"boundary_head": dict(TINY_BOUNDARY_HEAD)}), encoding="utf-8")
    cfg = build_extractor_config(
        {
            "model_name": "x",
            "boundary_head_from": str(tmp_path),
            "boundary_head_overrides": {"enable_relations": False},
            "token_pooling": "mean",
            "tokenization": "native",
            "add_bos": True,
            "attn_implementation": "sdpa",
            "max_len": 384,
        }
    )
    assert (cfg.architecture, cfg.token_pooling, cfg.tokenization, cfg.add_bos, cfg.max_len) == (
        "boundary",
        "mean",
        "native",
        True,
        384,
    )
    assert cfg.boundary_head["enable_relations"] is False


def test_warm_start_copies_head_tensors_but_not_the_encoder(tmp_path):
    source = _tiny(seed=1)
    source.save_pretrained(str(tmp_path / "src"))
    target = _tiny(seed=2)
    before_encoder = {k: v.clone() for k, v in target.state_dict().items() if k.startswith("encoder.")}
    loaded, skipped = warm_start_head(target, tmp_path / "src")
    src_state, tgt_state = source.state_dict(), target.state_dict()
    head_keys = [k for k in src_state if not k.startswith("encoder.")]
    assert loaded == len(head_keys) and skipped == 0
    assert all(torch.equal(src_state[k], tgt_state[k]) for k in head_keys)
    assert all(torch.equal(before_encoder[k], tgt_state[k]) for k in before_encoder)


def test_device_trainer_places_the_model_on_the_requested_device(tmp_path):
    from gliner2.training.trainer import TrainingConfig

    model = _tiny(seed=3)
    cfg = TrainingConfig(
        output_dir=str(tmp_path), max_steps=1, batch_size=1, eval_strategy="no", num_workers=0, pin_memory=False
    )
    trainer = DeviceTrainer(model, cfg, device="cpu")
    assert trainer.device.type == "cpu" and not trainer.config.fp16 and not trainer.config.bf16
    if torch.backends.mps.is_available():
        trainer = DeviceTrainer(_tiny(seed=3), cfg, device="mps")
        assert trainer.device.type == "mps" and next(trainer.model.parameters()).device.type == "mps"
        assert not trainer.config.fp16 and not trainer.config.bf16


def test_device_trainer_writes_metrics_jsonl(tmp_path):
    """Every _log_metrics call (train and eval) lands in <log_dir>/metrics.jsonl with the step."""
    from gliner2.training.trainer import TrainingConfig

    model = _tiny(seed=4)
    cfg = TrainingConfig(
        output_dir=str(tmp_path / "ckpt"),
        max_steps=2,
        batch_size=1,
        eval_strategy="no",
        num_workers=0,
        pin_memory=False,
        logging_steps=1,
        fp16=False,
        bf16=False,
        validate_data=False,
    )
    trainer = DeviceTrainer(model, cfg, device="cpu", log_dir=tmp_path / "logs")
    trainer.train(
        train_data=[
            {"input": "john works at acme .", "output": {"entities": {"person": ["john"], "company": ["acme"]}}}
        ]
        * 4
    )
    lines = [json.loads(l) for l in (tmp_path / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    train_lines = [l for l in lines if l["prefix"] == "train"]
    assert len(train_lines) >= 2 and all("loss" in l and "step" in l and "time" in l for l in train_lines)
    assert train_lines[-1]["step"] == 2


def test_guard_metrics_swallows_hook_errors_and_passes_results_through(caplog):
    import logging

    from train import guard_metrics

    def bad(model, ds):
        raise RuntimeError("boom")

    # on failure: the tracked metric gets the worst possible value, so the trainer
    # neither crashes nor mistakes the eval_loss fallback for a new best checkpoint
    with caplog.at_level(logging.ERROR):
        out = guard_metrics(bad, metric_for_best="exact_f1_micro", greater_is_better=True)(object(), object())
    assert out == {"exact_f1_micro": float("-inf")}
    assert any("metrics hook failed" in r.getMessage() for r in caplog.records)
    # trainer-computed metrics (eval_*) must NOT be clobbered by a sentinel: the real
    # value is already in the metrics dict and the .get() fallback handles the rest
    out = guard_metrics(bad, metric_for_best="eval_loss", greater_is_better=False)(object(), object())
    assert out == {}

    def good(model, ds):
        return {"exact_f1_micro": 0.5}

    guarded_good = guard_metrics(good)
    assert guarded_good(None, None) == {"exact_f1_micro": 0.5}
    assert guarded_good.failures == 0

    guarded_bad = guard_metrics(bad)
    guarded_bad(None, None)
    guarded_bad(None, None)
    assert guarded_bad.failures == 2
