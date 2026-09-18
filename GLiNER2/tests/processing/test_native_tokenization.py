"""Native tokenization mode: each segment (schema group, text) is tokenized the way the
encoder's own tokenizer tokenizes running text, instead of word by word.

Byte-level BPE encoders (ModernBERT / newmindai/Mursit-Base) mark word starts with a
leading-space piece ("Ġword"); GLiNER2's per-word tokenize() never produces it because the
words it feeds the tokenizer carry no spaces. Native mode restores the pretraining format:
first word of a segment bare, every later word space-prefixed, markers atomic.
"""
import pytest
from transformers import AutoTokenizer

from gliner2.processor import SchemaTransformer

TEXT = "Ahmet Yılmaz 12 Mart 2024 tarihinde İstanbul ' da imzaladı ."  # cased, as in the data
SCHEMA = {"entities": {"kişi": "", "tarih": ""}}


@pytest.fixture  # function-scoped: SchemaTransformer mutates the tokenizer it is given
def mursit_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("newmindai/Mursit-Base", local_files_only=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Mursit tokenizer not in the local HF cache: {exc}")


def _segment_ids(record, seg_type):
    return [tid for tid, (seg, _, _) in zip(record.input_ids, record.mapped_indices) if seg == seg_type]


def _words(processor, text):
    """The words GLiNER2 actually feeds the encoder: regex-split, then lowercased."""
    return [w for w, _, _ in processor.word_splitter(text, lower=True)]


def test_text_segment_matches_the_tokenizers_native_encoding(mursit_tokenizer):
    processor = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native")
    record = processor._transform_record({"text": TEXT, "schema": SCHEMA})

    # the encoder must see the lowercased words exactly as it would see them in running text
    expected = processor.tokenizer(" ".join(_words(processor, TEXT)), add_special_tokens=False)["input_ids"]
    assert _segment_ids(record, "text") == expected


def test_schema_markers_are_atomic_and_label_words_are_space_prefixed(mursit_tokenizer):
    processor = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native")
    record = processor._transform_record({"text": TEXT, "schema": SCHEMA})

    pieces = processor.tokenizer.convert_ids_to_tokens(_segment_ids(record, "schema"))
    assert pieces.count("[E]") == 2 and pieces.count("[P]") == 1
    assert "Ġ" not in pieces, "no stray space piece before a marker"
    assert pieces[pieces.index("[E]") + 1].startswith("Ġ"), "label after a marker is a running-text word"
    assert pieces[0] == "(", "the first token of a segment is bare, like a sequence start"


def test_every_word_maps_to_its_first_native_piece(mursit_tokenizer):
    processor = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native")
    record = processor._transform_record({"text": TEXT, "schema": SCHEMA})

    words = _words(processor, TEXT)
    assert len(record.text_word_first_positions) == len(words)
    pieces = processor.tokenizer.convert_ids_to_tokens(record.input_ids)
    starts = [pieces[p] for p in record.text_word_first_positions]
    assert starts[0] == "ahmet"                      # first word: bare
    assert starts[1].startswith("Ġ")                 # later words: space-prefixed
    assert starts[words.index("12")] == "12"         # not the standalone space piece that precedes it
    # the piece sequence of a word is contiguous and covers the whole text segment
    word_ids = [w for (seg, _, _), w in zip(record.mapped_indices, record.text_word_ids) if seg == "text"]
    assert word_ids == sorted(word_ids) and set(word_ids) == set(range(len(words)))


def test_add_bos_prepends_the_encoders_bos_token(mursit_tokenizer):
    processor = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native", add_bos=True)
    record = processor._transform_record({"text": TEXT, "schema": SCHEMA})

    assert record.input_ids[0] == mursit_tokenizer.bos_token_id
    assert record.mapped_indices[0][0] == "bos"
    without = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native")._transform_record({"text": TEXT, "schema": SCHEMA})
    assert record.input_ids[1:] == without.input_ids
    assert record.text_word_first_positions == [p + 1 for p in without.text_word_first_positions]
    assert record.schema_special_positions == [[p + 1 for p in group] for group in without.schema_special_positions]


def test_per_word_mode_is_the_default_and_unchanged(mursit_tokenizer):
    default = SchemaTransformer(tokenizer=mursit_tokenizer)
    assert default.tokenization == "per_word"
    record = default._transform_record({"text": TEXT, "schema": SCHEMA})
    expected = [tid for word in _words(default, TEXT) for tid in mursit_tokenizer.convert_tokens_to_ids(mursit_tokenizer.tokenize(word))]
    assert _segment_ids(record, "text") == expected


def test_unknown_tokenization_mode_is_rejected(mursit_tokenizer):
    with pytest.raises(ValueError):
        SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="whole_sentence")


def test_multi_group_schema_routes_markers_like_per_word_mode(mursit_tokenizer):
    """Entities + a classification group: every group keeps one position per marker, and the
    positions point at the marker ids, exactly as the per-word path does."""
    schema = {
        "entities": {"kişi": "", "tarih": ""},
        "classifications": [{"task": "duygu", "labels": ["olumlu", "olumsuz"], "true_label": ["olumlu"]}],
    }
    # one tokenizer instance per processor: registering the markers mutates the tokenizer
    native = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native")
    per_word = SchemaTransformer(tokenizer=AutoTokenizer.from_pretrained("newmindai/Mursit-Base", local_files_only=True))
    rec_n = native._transform_record({"text": TEXT, "schema": schema})
    rec_p = per_word._transform_record({"text": TEXT, "schema": schema})

    assert [len(g) for g in rec_n.schema_special_positions] == [len(g) for g in rec_p.schema_special_positions]
    marker_ids = native._special_ids
    for group in rec_n.schema_special_positions:
        assert all(rec_n.input_ids[p] in marker_ids for p in group)
    assert rec_n.task_types == rec_p.task_types and rec_n.structure_labels == rec_p.structure_labels


def test_native_mode_is_robust_to_markers_pre_registered_as_plain_strings(mursit_tokenizer):
    """A tokenizer that already carries the markers without lstrip (e.g. one saved by a
    per-word checkpoint) must still yield atomic markers with no stray space piece."""
    mursit_tokenizer.add_special_tokens({"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS})
    processor = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native")
    record = processor._transform_record({"text": TEXT, "schema": SCHEMA})

    pieces = processor.tokenizer.convert_ids_to_tokens(_segment_ids(record, "schema"))
    assert "Ġ" not in pieces
    for group in record.schema_special_positions:
        assert all(record.input_ids[p] in processor._special_ids for p in group)


# --- review round: behaviours the first implementation got wrong ---------------------------

def test_add_bos_also_applies_in_per_word_mode(mursit_tokenizer):
    """add_bos is a property of the input format, not of the tokenization mode."""
    with_bos = SchemaTransformer(tokenizer=mursit_tokenizer, add_bos=True)._transform_record({"text": TEXT, "schema": SCHEMA})
    without = SchemaTransformer(tokenizer=AutoTokenizer.from_pretrained("newmindai/Mursit-Base", local_files_only=True))._transform_record({"text": TEXT, "schema": SCHEMA})
    assert with_bos.input_ids[0] == mursit_tokenizer.bos_token_id and with_bos.mapped_indices[0][0] == "bos"
    assert with_bos.input_ids[1:] == without.input_ids
    assert with_bos.text_word_first_positions == [p + 1 for p in without.text_word_first_positions]


def test_first_position_of_a_number_word_is_its_digit_piece_not_the_space(mursit_tokenizer):
    """Native pieces for ' 12' are ['Ġ', '12']: the space piece stays inside the word (mean/max see it),
    but 'first' pooling must represent the word by the first piece that carries a character."""
    processor = SchemaTransformer(tokenizer=mursit_tokenizer, tokenization="native")
    record = processor._transform_record({"text": TEXT, "schema": SCHEMA})
    words = _words(processor, TEXT)
    pieces = processor.tokenizer.convert_ids_to_tokens(record.input_ids)
    pos = record.text_word_first_positions[words.index("12")]
    assert pieces[pos] == "12" and pieces[pos - 1] == "Ġ"
    assert record.text_word_ids[pos - 1] == record.text_word_ids[pos] == words.index("12")


def test_native_mode_requires_a_fast_tokenizer():
    from unittest.mock import MagicMock
    with pytest.raises(ValueError):
        SchemaTransformer(tokenizer=MagicMock(is_fast=False), tokenization="native")


def test_add_bos_requires_a_bos_or_cls_token_at_construction(mursit_tokenizer):
    from transformers import PreTrainedTokenizerFast
    bare = PreTrainedTokenizerFast(tokenizer_object=mursit_tokenizer.backend_tokenizer)  # no special tokens
    assert bare.bos_token_id is None and bare.cls_token_id is None
    with pytest.raises(ValueError):
        SchemaTransformer(tokenizer=bare, tokenization="native", add_bos=True)
