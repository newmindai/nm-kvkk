"""mean / max token pooling through the vectorised (scatter) path.

The historical fast path only handles token_pooling="first" (one gather per sample);
mean/max fell back to a Python loop over every subword. With per-subword word ids in the
batch, mean/max become a segment-reduce and must equal the loop path exactly.
"""
import pytest
import torch
from transformers import AutoTokenizer

from gliner2.processor import SchemaTransformer

TEXTS = ["Ahmet Yılmaz 12 Mart 2024 tarihinde İstanbul ' da imzaladı .", "Dava reddedildi ."]
SCHEMA = {"entities": {"kişi": "", "tarih": ""}}


def _batch(pooling, tokenization="native"):
    try:
        tokenizer = AutoTokenizer.from_pretrained("newmindai/Mursit-Base", local_files_only=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Mursit tokenizer not in the local HF cache: {exc}")
    processor = SchemaTransformer(tokenizer=tokenizer, token_pooling=pooling, tokenization=tokenization)
    return processor, processor.collate_fn_inference([(text, SCHEMA) for text in TEXTS])


def _random_states(batch, hidden=16, requires_grad=False):
    torch.manual_seed(0)
    return torch.randn(*batch.input_ids.shape, hidden, requires_grad=requires_grad)


def test_batch_carries_padded_word_ids_per_subword():
    _, batch = _batch("mean")
    assert batch.text_word_ids.shape == batch.input_ids.shape
    for i, n_words in enumerate(batch.text_word_counts):
        ids = batch.text_word_ids[i]
        assert set(ids[ids >= 0].tolist()) == set(range(n_words))
        assert (ids[batch.attention_mask[i] == 0] == -1).all(), "padding is not a word"


@pytest.mark.parametrize("pooling", ["mean", "max"])
@pytest.mark.parametrize("tokenization", ["native", "per_word"])
def test_vectorised_pooling_matches_the_reference_loop(pooling, tokenization):
    processor, batch = _batch(pooling, tokenization)
    states = _random_states(batch)
    fast_words, fast_schema = processor._extract_embeddings_fast(states, batch)
    loop_words, loop_schema = processor._extract_embeddings_loop(states, batch.input_ids, batch)
    assert len(fast_words) == len(loop_words)
    for fast, loop in zip(fast_words, loop_words):
        assert fast.shape == loop.shape and torch.allclose(fast, loop, atol=1e-6)
    for fast_groups, loop_groups in zip(fast_schema, loop_schema):
        for fast_group, loop_group in zip(fast_groups, loop_groups):
            assert all(torch.equal(a, b) for a, b in zip(fast_group, loop_group))


def test_mean_pooling_backpropagates_to_every_subword_of_a_word():
    processor, batch = _batch("mean")
    states = _random_states(batch, requires_grad=True)
    words, _ = processor.extract_embeddings_from_batch(states, batch.input_ids, batch)
    sum(w.sum() for w in words).backward()
    for i in range(len(batch)):
        text_positions = batch.text_word_ids[i] >= 0
        assert (states.grad[i][text_positions].abs().sum(-1) > 0).all()


def test_mean_pooling_takes_the_fast_path(monkeypatch):
    processor, batch = _batch("mean")
    def loop_must_not_run(*args, **kwargs):
        raise AssertionError("mean pooling fell back to the Python loop")
    monkeypatch.setattr(processor, "_extract_embeddings_loop", loop_must_not_run)
    processor.extract_embeddings_from_batch(_random_states(batch), batch.input_ids, batch)


# --- review round ---------------------------------------------------------------------------

_BAD = ("hello world", {"classifications": [{"task": "t", "labels": ["a", "b"]}]})  # missing true_label
_GOOD = ("hello world", {"entities": {"company": "companies"}})


def test_fallback_records_pool_like_the_loop(tiny_tokenizer):
    processor = SchemaTransformer(tokenizer=tiny_tokenizer, token_pooling="mean")
    batch = processor.collate_fn_inference([_BAD, _GOOD], error_policy="fallback")
    assert len(batch) == 2 and batch.text_word_counts[0] == 1 and batch.schema_special_indices[0][0]
    states = _random_states(batch)
    fast_words, fast_schema = processor.extract_embeddings_from_batch(states, batch.input_ids, batch)
    loop_words, loop_schema = processor._extract_embeddings_loop(states, batch.input_ids, batch)
    for fast, loop in zip(fast_words, loop_words):
        assert torch.allclose(fast, loop, atol=1e-6)
    assert all(len(f) == len(l) for f, l in zip(fast_schema, loop_schema))


def test_pad_batch_rejects_inconsistent_word_ids(tiny_tokenizer):
    processor = SchemaTransformer(tokenizer=tiny_tokenizer, token_pooling="mean")
    record = processor._transform_record({"text": "hello world", "schema": {"entities": {"company": ""}}})
    record.text_word_ids = record.text_word_ids[:-1]
    with pytest.raises(ValueError):
        processor._pad_batch([record])


def test_word_ids_are_only_materialised_for_mean_and_max(tiny_tokenizer):
    first = SchemaTransformer(tokenizer=tiny_tokenizer, token_pooling="first").collate_fn_inference([_GOOD])
    mean = SchemaTransformer(tokenizer=tiny_tokenizer, token_pooling="mean").collate_fn_inference([_GOOD])
    assert first.text_word_ids is None and mean.text_word_ids is not None


def test_mean_accumulates_in_fp32_so_fp16_does_not_overflow():
    processor, batch = _batch("mean")
    states = torch.full((*batch.input_ids.shape, 16), 3.0e4, dtype=torch.float16)  # 3 pieces sum past fp16 max
    words, _ = processor.extract_embeddings_from_batch(states, batch.input_ids, batch)
    assert all(torch.isfinite(w).all() for w in words)
    assert torch.allclose(words[0].float(), torch.full_like(words[0].float(), 3.0e4))
