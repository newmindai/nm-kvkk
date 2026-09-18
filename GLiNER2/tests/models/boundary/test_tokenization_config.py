"""tokenization / add_bos are model-level settings: validated by ExtractorConfig, handed to the
processor by BoundaryExtractor, and persisted through save_pretrained/from_pretrained."""
import pytest

from gliner2 import AutoExtractor, ExtractorConfig
from gliner2.inference.engine import BoundaryExtractor
from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
from tests.fixtures.tiny_encoder import build_tiny_encoder_config
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


def _build(**overrides):
    tokenizer = build_tiny_tokenizer()
    config = ExtractorConfig(model_name="tiny-bert-fixture", architecture="boundary",
                             boundary_head=dict(TINY_BOUNDARY_HEAD), **overrides)
    encoder_config = build_tiny_encoder_config(vocab_size=len(tokenizer))
    return BoundaryExtractor(config, encoder_config=encoder_config, tokenizer=tokenizer)


def test_config_defaults_to_per_word_without_bos():
    config = ExtractorConfig(model_name="x", architecture="boundary")
    assert config.tokenization == "per_word" and config.add_bos is False


def test_config_rejects_an_unknown_tokenization_mode():
    with pytest.raises(ValueError):
        ExtractorConfig(model_name="x", architecture="boundary", tokenization="whole_sentence")


def test_model_hands_the_flags_to_its_processor():
    model = _build(tokenization="native", add_bos=True)
    assert model.processor.tokenization == "native"
    assert model.processor.add_bos is True


def test_flags_survive_save_and_reload(tmp_path):
    _build(tokenization="native", add_bos=True).save_pretrained(str(tmp_path))
    reloaded = AutoExtractor.from_pretrained(str(tmp_path))
    assert reloaded.config.tokenization == "native" and reloaded.config.add_bos is True
    assert reloaded.processor.tokenization == "native" and reloaded.processor.add_bos is True


def test_validate_rechecks_tokenization_after_mutation():
    config = ExtractorConfig(model_name="x", architecture="boundary")
    config.tokenization = "bogus"
    with pytest.raises(ValueError):
        config.validate()
