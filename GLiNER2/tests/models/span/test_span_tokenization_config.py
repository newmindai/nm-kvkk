"""The span architecture must honour the same tokenization settings as the boundary one."""
import torch

from gliner2 import ExtractorConfig
from gliner2.inference.engine import GLiNER2
from tests.fixtures.tiny_encoder import build_tiny_encoder_config
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


def test_span_model_hands_tokenization_to_its_processor():
    tokenizer = build_tiny_tokenizer()
    config = ExtractorConfig(model_name="tiny-bert-fixture", max_width=8, counting_layer="count_lstm", tokenization="native")
    torch.manual_seed(13)
    model = GLiNER2(config, encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)), tokenizer=tokenizer)
    assert model.processor.tokenization == "native"
