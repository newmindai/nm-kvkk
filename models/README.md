# models/ — plain model folders, no Hugging Face cache

Every model is a self-contained directory `models/<name>/` (config + tokenizer + `model.safetensors`) and every
config refers to it by that relative path. Nothing is committed here. Training clusters usually have no internet:
download on a machine that has it and rsync the folder.

```bash
python scripts/download_models.py            # the bases: gliner2.5-multi-v1, Mursit-Base (+ Mursit-Base-4k)
python scripts/download_models.py --all      # + Mursit-Large, Mursit-Base-TR-Retrieval, the PII-filter baseline
```

## Bases (open weights)

| folder | source | what |
|---|---|---|
| `gliner2.5-multi-v1` | `fastino/gliner2.5-multi-v1` | multilingual GLiNER2.5, boundary architecture, mDeBERTa-v3-base encoder, 287M params. **The base of every shipped model.** Head and encoder were pre-trained together on schema-diverse data, which is why it generalises over label wordings and why swapping the encoder costs that ability. |
| `Mursit-Base` | `newmindai/Mursit-Base` | Turkish ModernBERT (22 layers, hidden 768, 1,024 positions, 165M with the head). |
| `Mursit-Base-4k` | built by `download_models.py` | Mursit-Base with `max_position_embeddings` 4096 and NTK-scaled global RoPE (weights symlinked, byte-identical). Required by the Mursit recipes: the nm6k schema prompt alone exceeds 1,024 tokens. |
| `Mursit-Large`, `Mursit-Large-4k` | `newmindai/Mursit-Large` | 28 layers / hidden 1024; only 93 of 136 head tensors can be warm-started. |
| `gliner2-privacy-filter-PII-multi` | `fastino/gliner2-privacy-filter-PII-multi` | zero-shot PII baseline (span architecture, max span width 8). |

## Fine-tuned release checkpoints

Placed here by hand (or downloaded once the Hugging Face repositories are public) as `models/<name>/`. The model
cards are committed (`models/<name>/README.md`, identical to the Hugging Face cards): recipe, benchmark results,
and **every entity label and relation type the model understands**, in the query names it was trained on.

| folder | what | label names |
|---|---|---|
| [`gliner2.5-kvkk-tr-v2`](https://huggingface.co/newmindai/gliner2.5-kvkk-tr-v2) | GLiNER2.5-multi → MIX100k (real KVKK+NER sentences) → nm-kvkk-pii-6K, checkpoint at step 500 (entities + relations). **Start here.** | Turkish |
| [`gliner2.5-kvkk-tr-v1`](https://huggingface.co/newmindai/gliner2.5-kvkk-tr-v1) | GLiNER2.5-multi → KVKK50k → e07 (entities + relations); best on the 19 KVKK fields | Turkish |
| [`gliner2.5-mursit-kvkk-tr-v1`](https://huggingface.co/newmindai/gliner2.5-mursit-kvkk-tr-v1) | Mursit-Base encoder → MIX100k → nm-kvkk-pii-6K with the full relation schema; query with bare relation names; needs this repository's GLiNER2 | Turkish |


The recipes and `configs/benchmark_models.json` refer to the checkpoints by their internal tags; this is the mapping:

| released model | tag | recipe | what it is |
|---|---|---|---|
| `gliner2.5-kvkk-tr-v2` | `s2short-500` | 3 | curriculum stage 1 (100k real KVKK+NER sentences) → 500 steps of nm-kvkk-pii-6K; best on real documents and on relations |
| `gliner2.5-kvkk-tr-v1` | `rele07tr` | 2 | the reference model: entities (118-node taxonomy) + 92 Turkish-named relations in one pass; best on the 19 KVKK fields |
| `gliner2.5-mursit-kvkk-tr-v1` | `mursit-fullrel` | 4 | the same curriculum on the Mursit-Base encoder with the full relation schema in every record (bare relation names) |

Load any of them with the vendored library:

```python
import sys

sys.path.insert(0, "GLiNER2")
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("models/gliner2.5-kvkk-tr-v2", map_location="cpu")  # or "cuda" / "mps"
schema = (
    model.create_schema()
    .entities(["ad soyad", "telefon numarası", "IBAN"])
    .relations({"telefon numarası sahibi": "Telefon numarası kişiye aittir"})
)
out = model.extract("Ahmet Yılmaz'ın telefonu 0532 111 22 33.", schema, threshold=0.5)
out["entities"], out["relation_extraction"]
```

Which relation naming and prompt shape each model expects is a per-model fact;
`configs/benchmark_models.json` records the vocabulary (`taxonomy_tr`, `nm6k_tr`, `nm6k_en`) per model.
