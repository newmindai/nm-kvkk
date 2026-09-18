# briefs — Nemotron-PII document-type catalogue

Source: NVIDIA `nvidia/Nemotron-PII` on Hugging Face, licence **CC-BY-4.0** (attribution: NVIDIA Corporation,
Nemotron-PII dataset, https://huggingface.co/datasets/nvidia/Nemotron-PII). Only five columns are used,
`uid, domain, document_type, document_description, document_format`, never Nemotron's text or values.

`download_seeds.py` downloads `data/train-00000-of-00001.parquet` and `data/test-00000-of-00001.parquet`
into `raw/` (gitignored) and rebuilds the two catalogues shipped here:

| file | rows | rule |
|---|---|---|
| `nemotron_catalogue.parquet` | 1,655 | train file; one row per (domain, type_norm); `type_norm` = whitespace-collapsed lower-case `document_type`; `uid` and `document_type` from the group's first row in file order; `document_description` and `document_format` = the most frequent value; `n` = group size |
| `nemotron_catalogue_fresh_for_e08.parquet` | 1,789 | train rows whose type e07 did not use, plus test rows whose type is neither in the train catalogue nor used by e07; `e07_types.json` is that list (912 types) |

`download_seeds.py --check` rebuilds both and asserts equality with the shipped files on every column.
A run samples its briefs from a catalogue with `pipeline/sample_briefs.py` (balanced over domains, each brief
used at most three times). A brief decides what a plausible document looks like; it says nothing about
personal data, which the sampler decides.
