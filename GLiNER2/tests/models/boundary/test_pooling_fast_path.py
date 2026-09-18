"""The boundary model's _encode_core must use the vectorised pooling for mean/max
instead of falling back to the per-subword Python loop, and produce the same states."""
import torch

from gliner2 import ExtractorConfig
from gliner2.inference.engine import BoundaryExtractor
from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
from tests.fixtures.tiny_encoder import build_tiny_encoder_config
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

TEXT = "apple acquired google ."
SCHEMA = {"entities": {"company": ""}}


def _model(pooling):
    tokenizer = build_tiny_tokenizer()
    config = ExtractorConfig(model_name="tiny-bert-fixture", architecture="boundary",
                             boundary_head=dict(TINY_BOUNDARY_HEAD), token_pooling=pooling)
    torch.manual_seed(17)
    model = BoundaryExtractor(config, encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)), tokenizer=tokenizer)
    return model.eval()


def test_encode_core_does_not_use_the_loop_for_mean_pooling(monkeypatch):
    model = _model("mean")
    def loop_must_not_run(*args, **kwargs):
        raise AssertionError("_encode_core fell back to the Python loop")
    monkeypatch.setattr(model.processor, "_extract_embeddings_loop", loop_must_not_run)
    model.extract_entities(TEXT, ["company"])


def test_encode_core_mean_states_equal_the_loop_reference():
    model = _model("mean")
    batch = model.processor.collate_fn_inference([(TEXT, SCHEMA)])
    with torch.no_grad():
        token_embeddings = model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
        loop_words, _ = model.processor._extract_embeddings_loop(token_embeddings, batch.input_ids, batch)
        fast = model._encode_core(batch)["text_states"]
    assert torch.allclose(fast[0, : loop_words[0].shape[0]], loop_words[0], atol=1e-6)
