---
language:
- tr
license: apache-2.0
library_name: gliner2
pipeline_tag: token-classification
base_model: fastino/gliner2.5-multi-v1
tags:
- gliner2
- ner
- relation-extraction
- pii
- kvkk
- turkish
---
# gliner2.5-kvkk-tr-v2

GLiNER2.5-multi trained in two stages — first on 100,000 pseudonymised real Turkish legal and KVKK sentences (30 labels, entities only), then on nm-kvkk-pii-6K with Turkish label and relation names — and taken at **step 500** of the second stage: half an epoch of synthetic data, enough to bind the full 118-label vocabulary and the relation head without fitting the generator. The best of our models on the real notary documents and on relation extraction, and within a point of the best on the broad real set.

Part of the **nm-kvkk** release (https://github.com/newmindai/nm-kvkk): training recipes, the patched GLiNER2 library, the synthetic-data
generator and the evaluators behind every number below.

## Model details

| | |
|---|---|
| Architecture | GLiNER2 boundary extractor: schema-prompted entities + relations in one pass |
| Encoder | mDeBERTa-v3-base (GLiNER2.5-multi, 287M) |
| Label language | Turkish label names |
| Taxonomy | KVKK personal-data taxonomy v2.2 — 118 entity types; this model's schema has 113 relation types (vocabulary files in the repository) |
| Licence | Apache-2.0 |

### Training recipe

| | |
|---|---|
| Starting point | fastino/gliner2.5-multi-v1 |
| Stage 1 | MIX100k: 100,000 pseudonymised real Turkish legal + KVKK sentences (real text, personal values masked and re-populated with synthetic entities), 30 labels, entities only; 13,000 steps, batch 4 on 1×H100, lr encoder 1e-5 / heads 5e-4, constant with 20 warm-up steps |
| Stage 2 | nm-kvkk-pii-6K with Turkish entity and relation names; run of 4,000 steps, same optimiser |
| Checkpoint | step 500 (c500). Single seed; neighbouring checkpoints differ by several points — the early region (500–2,000 steps) is consistently better on real documents than the end of the run |
| Precision | bf16; weight decay 0.01; seed 42 |
| Label language | Turkish entity and relation names |
| Inference prompt | entity labels + relation types as bare names (its best shape on real documents); descriptions also work, at a few points less |

Stopping the second stage early is deliberate: the final checkpoint of the same run fits the synthetic generator more and loses ground on real documents with the whole schema; this checkpoint is the one that beats the v1 reference on both entities and relations on the notary set.

## How to use

This checkpoint loads and runs with the upstream `gliner2` package (tested with gliner2 2.0.0 from PyPI). The patched library in the nm-kvkk repository is only needed to reproduce training (bf16 loss fix, non-finite-gradient guard) or to run the Mursit-based sibling model.

**Out of the box** — `extract_kvkk.py` ships in this repository: text in, JSON keyed by taxonomy ids out, the
label names handled for you.

```bash
pip install gliner2 huggingface_hub
python extract_kvkk.py --model newmindai/gliner2.5-kvkk-tr-v2 document.txt                   # 19 KVKK fields + person and company names
python extract_kvkk.py --model newmindai/gliner2.5-kvkk-tr-v2 --relations all document.txt   # ... plus every relation type
python extract_kvkk.py --model newmindai/gliner2.5-kvkk-tr-v2 --labels all --relations all document.txt   # the whole taxonomy
python extract_kvkk.py --model newmindai/gliner2.5-kvkk-tr-v2 --text "Ahmet Yılmaz, TC kimlik numarası 12345678901, ..."
```

(get the script with `huggingface_hub.hf_hub_download("newmindai/gliner2.5-kvkk-tr-v2", "extract_kvkk.py")` or from the repository.)

**In your own code:**

```python
import json
from gliner2 import AutoExtractor
from huggingface_hub import hf_hub_download

repo = "newmindai/gliner2.5-kvkk-tr-v2"
model = AutoExtractor.from_pretrained(repo)

# kvkk_schema.json ships with the model: every entity label and relation type, with the exact query
# names this model was trained on, so you never need to know whether it expects Turkish or English names.
S = json.load(open(hf_hub_download(repo, "kvkk_schema.json"), encoding="utf-8"))
ent_name = {e["id"]: e["name"] for e in S["entities"]}  # taxonomy id -> query name
rel_name = {r["id"]: r["name"] for r in S["relations"]}
rel_desc = {r["name"]: r["description"] for r in S["relations"]}

# ask for what you need — a shorter prompt is faster and, for these models, more accurate
wanted = S["subsets"]["stack21"]  # or S["subsets"]["kvkk19"], or any list of ids
relations = ["national_id_of", "phone_of", "email_of"]  # ids from S["relations"]

schema = model.create_schema().entities([ent_name[i] for i in wanted])
if S["relation_prompt"] == "desc":
    schema = schema.relations({rel_name[r]: rel_desc[rel_name[r]] for r in relations})
else:  # this model works best with bare relation names
    schema = schema.relations([rel_name[r] for r in relations])

out = model.extract(text, schema, threshold=0.5, include_confidence=True, include_spans=True)
out["entities"]  # {query name: [{text, start, end, confidence}]}
out["relation_extraction"]  # {query name: [{head, tail}]}

# map the query names in the output back to taxonomy ids
name_to_id = {v: k for k, v in ent_name.items()}
entities = {name_to_id[n]: spans for n, spans in out["entities"].items()}
```

`kvkk_schema.json` fields: `entities[] = {id, name, group, out_of_scope, trained}`, `relations[] = {id, name, description}`,
`subsets` (the 19 KVKK fields and the 21-label set used in the benchmarks), `label_language` and `relation_prompt`
(the prompt shape this model works best with). The same ids are used by every model of the release, so switching
model only changes the names the file maps them to. The full listing is in the section below.


## Benchmark results

All models below were run through the same pipeline and scored by the same code (see the repository). Scores are
**strict** character-offset precision / recall / F1 (start, end and label must all match), micro over spans;
**macro F1** averages the per-label F1 over the labels present; **lenient F1** accepts an overlapping span with the
right label (boundary conventions differ between word-level taggers and span models). Every model is queried with
the 21 labels all of them can express (19 KVKK fields + person name + company name); gold is restricted to those
labels. Threshold 0.5 for the GLiNER2 models.

**mixed_v2 — 118 real documents, 19 document families** (1323 gold spans in the 21 shared labels, whole-document input)

| model | P | R | F1 | macro F1 | lenient F1 |
|---|---:|---:|---:|---:|---:|
| **gliner2.5-kvkk-tr-v2** | **76.6** | **76.9** | **76.8** | **61.3** | **81.3** |
| gliner2.5-mursit-kvkk-tr-v1 | 84.8 | 59.2 | 69.7 | 54.6 | 73.6 |
| YTU-ModernBERT-TR-PII | 61.2 | 78.8 | 68.9 | 47.8 | 78.6 |
| gliner2.5-kvkk-tr-v1 | 74.8 | 62.7 | 68.2 | 54.6 | 73.3 |
| GLiNER2.5-multi (untuned) | 65.7 | 55.3 | 60.0 | 44.4 | 66.5 |
| redact (LiteRT) | 36.0 | 22.0 | 27.3 | 14.1 | 39.2 |

**vekaletname — 20 real notary documents** (188 gold spans in the 21 shared labels, whole-document input)

| model | P | R | F1 | macro F1 | lenient F1 |
|---|---:|---:|---:|---:|---:|
| **gliner2.5-kvkk-tr-v2** | **91.3** | **89.4** | **90.3** | **93.9** | **93.0** |
| gliner2.5-kvkk-tr-v1 | 90.4 | 80.3 | 85.1 | 86.4 | 89.6 |
| gliner2.5-mursit-kvkk-tr-v1 | 91.2 | 66.0 | 76.5 | 80.6 | 79.6 |
| YTU-ModernBERT-TR-PII | 68.8 | 79.8 | 73.9 | 74.4 | 81.8 |
| GLiNER2.5-multi (untuned) | 73.5 | 53.2 | 61.7 | 60.5 | 76.5 |
| redact (LiteRT) | 28.9 | 11.7 | 16.7 | 27.9 | 28.8 |

**nm-kvkk-pii-6K test — 550 synthetic documents (in-distribution for models trained on nm-kvkk-pii-6K, marked *)** (2289 gold spans in the 21 shared labels, whole-document input)

| model | P | R | F1 | macro F1 | lenient F1 |
|---|---:|---:|---:|---:|---:|
| gliner2.5-mursit-kvkk-tr-v1\* | 92.9 | 88.4 | 90.6 | 92.1 | 96.3 |
| **gliner2.5-kvkk-tr-v2\*** | **85.2** | **93.4** | **89.1** | **84.0** | **94.9** |
| gliner2.5-kvkk-tr-v1 | 83.9 | 92.0 | 87.7 | 82.6 | 93.6 |
| GLiNER2.5-multi (untuned) | 67.9 | 65.0 | 66.4 | 51.3 | 76.7 |
| YTU-ModernBERT-TR-PII | 60.4 | 69.1 | 64.5 | 48.3 | 86.7 |
| redact (LiteRT) | 40.7 | 25.3 | 31.2 | 19.2 | 58.7 |

### Full taxonomy and relation extraction

The same model queried with every label of the taxonomy it was trained with and every relation type, whole
document in one pass — the way it is meant to be deployed. Relation F1 is strict on type and both end-points,
coreference respected; the type-filtered column applies training-derived type constraints as a post-filter.

| set | entity F1 (P / R) | lenient | relation F1 (P / R) | relation F1, type-filtered |
|---|---|---:|---|---:|
| vekaletname (20 real notary documents) | 82.2 (82.6 / 81.8) | 83.7 | 37.9 (25.1 / 77.2) | 44.8 |
| mixed_v2 (118 real documents) | 61.8 (74.4 / 52.9) | 64.4 | 31.0 (24.2 / 43.1) | 34.9 |

## Limitations

- Trained on synthetic Turkish documents (plus, for the MIX100k models, real sentences); scores on real documents
  are 10–25 points below the synthetic test split. The real-document sets used here are small (20 and 118 documents).
- Company and institution names are the hardest label for every model in this family.
- Sentence-level input hurts: feed whole documents or paragraphs, not individual sentences.
- Pseudonymised real data was used for evaluation only; no real personal data was used in training.

## Citation

NewMind AI, 2026 — *nm-kvkk: Turkish KVKK personal-data extraction with GLiNER2.5*, https://github.com/newmindai/nm-kvkk.
