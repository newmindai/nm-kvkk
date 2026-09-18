# Changes relative to upstream GLiNER2 (branch `nm-turkish`)

Base: `fastino-ai/GLiNER2` commit `3c913c7` (2026-08-24, package version 2.0.0). Five commits on top,
HEAD `46859c1` (`git describe`: `v2.0.0-5-g46859c1`). Diff: 12 files, +760 / −17; inference code paths
used by the shipped mDeBERTa models are unchanged.

| commit | area | what |
|---|---|---|
| `1f649dd` | `gliner2/processor.py` | **Native tokenization mode** (`tokenization="native"`): each schema group and the text are tokenized as one string with offset mapping, and every sub-word piece is mapped back to its word; adds `text_word_ids` (word index per piece) to records and batches. Default `"per_word"` is byte-identical to upstream. |
| `fca4c37` | `gliner2/configuration.py`, `models/boundary/model.py`, `models/span/model.py` | `ExtractorConfig(tokenization=..., add_bos=...)` — validated, persisted in `config.json`, restored by `from_pretrained`; handed to `SchemaTransformer` by both architectures. Old checkpoints load unchanged. |
| `8e1263b` | `gliner2/processor.py` | Vectorised `mean` / `max` word pooling (`SchemaTransformer.pool_text_states`, `scatter_add` / `scatter_reduce`) on the fast path; the Python loop stays as the reference. `first` pooling untouched. |
| `20c59c9` | processor / models / README | Review fixes: BOS in both modes, tokenizer suitability checked at construction, fallback records carry routing, a word's first position is its first character-bearing piece, fp32 accumulation for the mean, span-model plumbing, README section. |
| `46859c1` | `models/boundary/losses.py`, `training/trainer.py` | **bf16 fix**: `marginal_pair_consistency_loss` computed `log1p(-sigmoid(x).clamp(max=1-1e-6))`, whose clamp bound rounds to exactly 1.0 in bf16, so a saturated pair logit produced `log1p(-1)` — finite forward, NaN backward — and one step poisoned every parameter (observed as a run frozen at loss 0.0). Replaced by the identity `logsigmoid(-x)` in fp32. Trainer: a non-finite gradient norm now skips the optimizer step (after `clip_grad_norm_` has already spread the NaN) and logs a warning. |

## Why the tokenizer change exists

Upstream GLiNER2 passes **every word and every schema marker to the encoder tokenizer on its own**
(`tokenizer.tokenize(word)`) and concatenates the pieces. For SentencePiece encoders (mDeBERTa, the
encoder of gliner2.5-multi-v1 and of every model in this bundle) that is exact: the tokenizer adds its
own word-start marker `▁` to any string. For byte-level BPE encoders (the ModernBERT family, including
newmind's Turkish `Mursit-Base`) the word start is a leading *space* merged into the piece
(`Ġsözleşme`), which a bare word never carries — the encoder saw `söz` + `leşme` instead of
`Ġsözleşme` for 41–44 % of words, with 22 % more pieces than in pre-training.

Native mode fixes this without touching the training logic: segments are tokenized whole, the
first word of a segment is bare, later words are space-prefixed, the schema markers are registered
as `AddedToken(lstrip=True)` so they stay atomic, `add_bos=True` prepends the encoder's BOS/CLS token,
and all downstream structures (`mapped_indices`, `text_word_first_positions`, entity matching on
words) are produced exactly as before. Mean/max pooling were vectorised because with byte-level
pieces the "first sub-word" is often a fragment.

The mDeBERTa-based models (`gliner2.5-kvkk-tr-v1`, `gliner2.5-kvkk-tr-v2`) use the upstream defaults
(`per_word`, no BOS, `first` pooling); the tokenizer options have no effect for them. The Mursit-based model
(`gliner2.5-mursit-kvkk-tr-v1`) needs native tokenization and does not work correctly with the upstream package.
The bf16 loss fix and the gradient guard were active during the training of all three models on H100 (bf16).

## Tests

31 new tests: `tests/processing/test_native_tokenization.py`, `tests/processing/test_pooling_vectorized.py`,
`tests/models/boundary/test_tokenization_config.py`, `tests/models/boundary/test_pooling_fast_path.py`,
`tests/models/span/test_span_tokenization_config.py`.

## Release preparation for nm-kvkk

- `gliner2/__init__.py`: adds `__nm_kvkk__ = True`, so code can check that this copy is installed.
- `gliner2/training/lora.py`: one comment reworded.
- Every changed source file starts with a notice that names this file.
- `pyproject.toml`: distribution name `gliner2-nm-kvkk` (import name stays `gliner2`), Fastino AI as upstream author,
  NewMind AI as maintainer, project URLs point to nm-kvkk and upstream.
- `README.md`: notice that this is a modified copy, installation from this folder, and the tokenization section.
