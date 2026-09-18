---
pretty_name: nm-kvkk-pii-6K
language:
  - tr
task_categories:
  - token-classification
tags:
  - pii
  - kvkk
  - named-entity-recognition
  - relation-extraction
  - synthetic
size_categories:
  - 1K<n<10K
configs:
  - config_name: spans
    data_files:
      - split: train
        path: spans/train.jsonl
      - split: test
        path: spans/test.jsonl
      - split: test_hard
        path: spans/test_hard.jsonl
      - split: test_easy
        path: spans/test_easy.jsonl
  - config_name: bio
    data_files:
      - split: train
        path: bio/train.jsonl
      - split: test
        path: bio/test.jsonl
      - split: test_hard
        path: bio/test_hard.jsonl
      - split: test_easy
        path: bio/test_easy.jsonl
  - config_name: gliner2-tr
    data_files:
      - split: train
        path: tr/train.jsonl
      - split: test
        path: tr/test.jsonl
      - split: test_hard
        path: tr/test_hard.jsonl
      - split: test_easy
        path: tr/test_easy.jsonl
  - config_name: gliner2-en
    data_files:
      - split: train
        path: en/train.jsonl
      - split: test
        path: en/test.jsonl
      - split: test_hard
        path: en/test_hard.jsonl
      - split: test_easy
        path: en/test_easy.jsonl
---

# nm-kvkk-pii-6K — Turkish PII entities + relations

5,780 synthetic Turkish documents annotated with KVKK-taxonomy PII spans and the relations between them.

The corpus ships in **three forms over the same documents and the same split**: `spans/` (character offsets — the
canonical form), `bio/` (token + tag sequences) and the GLiNER2 training format in two label vocabularies, `tr/`
and `en/`. The document text is Turkish throughout; only the *label* language differs between `tr/` and `en/`,
and neither mixes the two.

Built from two runs of the `kvkk_synth` generator in the nm-kvkk repository (https://github.com/newmindai/nm-kvkk):
e07 (1,000 briefs) and e08 (5,000 briefs).

## Layout

```
nm-kvkk-pii-6K/
├── spans/                 CANONICAL — char offsets, entity ids, relations as a graph over them
│   ├── train.jsonl        5,178 records · 25,052 entities · 13,461 relations
│   ├── test.jsonl           602 records ·  3,709 entities ·  2,074 relations
│   ├── test_hard.jsonl      241 records — documents naming more than one person
│   └── test_easy.jsonl      361 records — single-person documents
├── bio/                   DERIVED — tokens, BIO tags and token offsets; same four splits
├── tr/                    GLiNER2 training format, Turkish label vocabulary
│   ├── train.jsonl        5,178 records (4,752 documents + 426 zero-PII negatives)
│   ├── test.jsonl           602 records (with "id")
│   ├── test_hard.jsonl      241 · test_easy.jsonl 361
│   └── labels.json        115 entity labels, 113 relation names, relation descriptions
├── en/                    same files, same documents, English label vocabulary
├── labels.json            taxonomy id → Turkish / English name (+ relation descriptions)
├── label_stats.json       per label: mentions and documents, train / test, group, sensitivity
├── id_map.json            record id ↔ generation-run id, both ways
├── split_report.json      strata, counts, warnings, seed
├── spans_report.json      verification of the spans / BIO build
└── README.md
```

`test_hard` and `test_easy` partition `test`, in every form.

## Which form to use

| you want to | use |
|---|---|
| train or evaluate GLiNER2 | `tr/` or `en/` |
| anything else — token classification, your own tokenizer, span or relation models, inspection | `spans/` |
| a ready-made token/tag sequence | `bio/` |

`spans/` is canonical: it is the only form that records **where** each mention is and **which
occurrence** was annotated when a string repeats. The GLiNER2 form cannot — the library takes no
offsets and locates a mention by whole-word string search — and BIO is tied to token boundaries.


## Record ids

Every document has one id, the same in all four forms:

```
id         nmkvkk-00001 … nmkvkk-05780
source_id  e07-1, e08-4471, e08-neg-16 — the generation-run id it was built from
```

Ids are assigned once over the whole corpus, in a fixed order (run, documents before zero-PII
negatives, bundle number), so they are stable across rebuilds. `id_map.json` holds the mapping both
ways. Ids are **not** contiguous within a split: `train` and `test` interleave, because the split is
brief-disjoint rather than sequential.

## Record format — `spans/` (canonical)

```json
{"id": "nmkvkk-00951", "source": "e08",
 "text": "İŞLETİM VE KULLANIM LİSANSI\n\nİşbu lisans, … lisans alan sıfatıyla Orhan Yeter Kaşıkçı …",
 "entities": [
   {"id": "e1", "label": "full_name",          "span": "Orhan Yeter Kaşıkçı", "start": 189, "end": 208},
   {"id": "e2", "label": "national_id_number", "span": "58676141176",         "start": 232, "end": 243}],
 "relations":          [{"head": "e2",  "relation": "national_id_of",  "tail": "e1"}],
 "relations_negative": [{"head": "e12", "relation": "bar_registry_of", "tail": "e1"}],
 "meta": {"profile": "normal", "persons": 2, "hard": true, "zero_pii": false,
          "domain": "Legal", "document_type": "License", "madde6": false}}
```

- `text[start:end] == span` for **all 28,761 mentions** — verified at build time, 0 mismatches.
- `label` and `relation` are **taxonomy ids**, language-neutral; `labels.json` maps each to its
  Turkish and English display name, so one copy of the data serves both languages.
- `relations_negative` are judge-verified statements that a relation does **not** hold — typically a
  second person's value that must not be attributed to the subject. 796 of them.
- `meta.hard` is the multi-person flag `test_hard` is built on.

## Record format — `bio/` (derived)

```json
{"id": "nmkvkk-00090", "source_id": "e07-100", "source": "e07",
 "text":         "PAKETLEME LİSTESİ\n\nBelge Türü: Tüketici Ürünleri Sevkiyat Paketleme Listesi\n…",
 "tokens":       ["PAKETLEME", "LİSTESİ", "Belge", "Türü", ":", …],
 "ner_tags":     ["O", "O", "O", "O", "O", …, "B-full_name", "I-full_name", …],
 "token_starts": [0, 10, 19, 25, 30, …],
 "token_ends":   [9, 17, 24, 29, 31, …]}
```

The document `text` is carried alongside the tokens, and the token offsets index into it, so the
tagging stays reversible and can be re-aligned to any other tokenizer. Tokenized with GLiNER2's
whitespace word splitter (original case). Relations are not
representable in BIO — use `spans/`.

**Boundary behaviour.** BIO tags are flat and tied to token boundaries. 28,753 of the 28,761 spans
round-trip out of the tags byte-identically; for the remaining 8 (in 7 documents) the tag covers the
touching tokens while the exact boundary differs — nested spans, and spans that sit inside a single
token such as `008` in `SIP-008` or a URL glued to a following `.`. `spans_report.json` names them.
Use `spans/` when exact boundaries matter.

## Record format — `tr/` and `en/` (GLiNER2 training format)

The format the nm-kvkk recipes train on, produced by `build_record` from the repository's
`scripts/convert_relations_train.py`, so records are byte-compatible with the training builds:

```json
{"input": "KAN TAHLİLİ RAPORU\n\nHasta: Yaşar Ergin\nT.C. Kimlik No: 49738174980\n...",
 "output": {
   "entities": {"ad soyad": ["Yaşar Ergin", "Dr. Berkan Ertürk"],
                "kimlik numarası": ["49738174980"],
                "kan grubu": ["A Rh+"],
                "araç plakası": []},
   "relations": [{"kimlik numarası sahibi": {"head": "49738174980", "tail": "Yaşar Ergin"}},
                 {"kan grubu sahibi":      {"head": "A Rh+",       "tail": "Yaşar Ergin"}},
                 {"lakap sahibi":          {"head": "",            "tail": ""}}]}}
```

- **Every** label is declared in every record; an empty list is explicit negative supervision for that label.
- Surfaces are verbatim slices of `input` (22,393 checked, 0 mismatches).
- Each record also declares 4 absent relation types with empty head and tail, the verified negative recipe.
  Train carries 13,473 positive relation instances and 20,712 declared negatives. Those negatives are
  sampled at build time for training; `spans/` instead carries the 796 relations a judge actually
  confirmed do **not** hold, and declares nothing empty — regenerate either from `labels.json`.
- Every record in every split carries `id` and `source_id`, so predictions can be traced back.

## Splits

**Hardness.** A document is *hard* when it names more than one person: a relative, a counterpart such as a
physician or lawyer, an unrelated bystander, or one of the role parties (vekil, kefil, mirasçı, kiracı, şirket
imza yetkilisi, tanık). Those documents carry the attribution problem worth measuring — several people and several
values in one text, plus explicit negatives saying which value is *not* the subject's. Train keeps 943 hard
documents, test 241, and `test_hard.jsonl` isolates them so entity and relation scores can be reported separately
from the single-person case.

**Disjointness.** Documents are grouped by the brief they were written from, and whole briefs go to one side, so
no document type in test was seen in training. Test was allocated over source × density profile × hardness by
largest remainder (seed 42), with a fixed quota of 240 hard documents. There is no text overlap between the two
splits (checked by hash). 15 zero-PII documents were dropped from train because their brief landed in test.

| | train | test | test_hard | test_easy |
|---|---|---|---|---|
| records | 5,178 | 602 | 241 | 361 |
| of which zero-PII negatives | 426 | 0 | 0 | 0 |
| hard (multi-person) | 943 | 241 | 241 | 0 |

## Label vocabularies

| | tr | en |
|---|---|---|
| entity labels | 115 | 118 |
| relation names | 113 | 113 |

The Turkish entity vocabulary is 115 rather than 118 because three pairs of taxonomy nodes share one canonical
Turkish name (for example `email_address` and `kep_address` are both "e-posta adresi"). English names are derived
from the taxonomy ids with underscores replaced by spaces ("full name", "represented by"). Turkish names are the
113 verified names of the e07 build plus 5 entity and 21 relation names written here in the same
convention: a relation name is a short noun phrase naming the **tail** argument of the gold head→tail direction
("kimlik numarası sahibi" = the person the ID belongs to; "kefil olduğu kişi" = the debtor the head vouches for).
The builder asserts that no entity name equals a relation name in either language.

`labels.json` in each variant carries `entity_labels` (taxonomy id → name), `relation_names` (id → name) and
`relation_descriptions` (name → one-sentence description in that language) for use as schema prompts.

## Label distribution

118 entity labels; **113 carry at least one mention**, 5 none, and 12 more appear in fewer than 10
documents. 28,761 mentions in total. Per-label counts (mentions and documents, split by train/test,
with taxonomy group and KVKK sensitivity) are in `label_stats.json`; relation counts are there too.

| taxonomy group | mentions |
|---|---:|
| person_name | 12,061 |
| identity_numbers | 2,834 |
| financial | 2,042 |
| digital_identifiers | 1,956 |
| health | 1,798 |
| contact | 1,593 |
| legal_customer_records | 1,592 |
| birth_civil_status | 1,198 |
| organization (out of KVKK scope) | 1,183 |
| employment | 903 |
| vehicle | 680 |
| registry_records | 474 |
| location | 345 |
| special_categories | 102 |

Most frequent labels:

| label | Turkish name | mentions | documents |
|---|---|---:|---:|
| `full_name` | ad soyad | 11,521 | 5,354 |
| `national_id_number` | kimlik numarası | 2,098 | 1,971 |
| `company_name` | şirket adı | 733 | 489 |
| `insurance_policy_number` | sigorta poliçe numarası | 497 | 426 |
| `username` | kullanıcı adı | 461 | 361 |
| `phone_number` | telefon numarası | 432 | 383 |
| `email_address` | e-posta adresi | 426 | 376 |
| `customer_number` | müşteri numarası | 415 | 382 |
| `full_address` | adres | 389 | 368 |
| `health_report_id` | sağlık raporu numarası | 384 | 333 |

## Relation distribution

**15,535 positive relation instances**, 2.7 per document on average; 569 documents carry none
(the zero-PII negatives and documents whose values belong to no one in particular). 108 of the 113
relation types have support; 5 never occur (`conviction_of`, `depicted_in`, `military_status_of`,
`nickname_of`, `verdict_in_case`) and 9 more have fewer than 10 instances. Per-type counts are in
`label_stats.json`.

| relation | Turkish name | instances |
|---|---|---:|
| `national_id_of` | kimlik numarası sahibi | 2,011 |
| `account_of` | kullanıcı hesabı sahibi | 566 |
| `email_of` | e-posta adresi sahibi | 514 |
| `phone_of` | telefon numarası sahibi | 436 |
| `insurance_policy_of` | sigorta poliçesi sahibi | 424 |
| `customer_number_of` | müşteri numarası sahibi | 380 |
| `iban_of` | banka hesabı sahibi | 358 |
| `employer_of` | işvereni olduğu kişi | 354 |
| `position_of` | iş unvanı sahibi | 328 |
| `health_file_of` | sağlık dosyası sahibi | 328 |

Relation names are written to denote the **tail** argument of the gold head→tail direction
("kimlik numarası sahibi" = the person the ID belongs to), so the pair reads (ID number, person).

### Negative relations

`spans/` carries **796 negative relations across 638 documents — every one of them multi-person.**
They are judge-verified statements that a pairing is **false**: the same head value attached to the
*other* person named in the document, with the true pairing present as a positive alongside it.

```
nmkvkk-00405   (2 persons, Customer Order Form)
  NOT  '5752' (employee_id)  --employee_id_of-->  'Kamer Değirmenci'
  but  '5752'                --employee_id_of-->  'Arzu Arat'          ← the true one
```

| negative relation | Turkish name | count |
|---|---|---:|
| `phone_of` | telefon numarası sahibi | 140 |
| `email_of` | e-posta adresi sahibi | 132 |
| `prescribed_by` | reçeteyi yazan doktor | 113 |
| `authorised_signatory_of` | imza yetkilisi olduğu şirket | 87 |
| `employee_id_of` | personel numarası sahibi | 68 |
| `national_id_of` | kimlik numarası sahibi | 67 |

They exist because attaching an attribute to the wrong person is the characteristic failure of
relation extraction on documents with several people in them, and a positive-only corpus gives a
model nothing to push against.

**Do not confuse these with the `tr/` and `en/` "declared negatives."** Those are 4 absent relation
*types* sampled per record at build time (20,712 in train), with empty head and tail — they teach
"this type does not occur here". The 796 above are about *who owns what*, and only they are
annotation.

## Sources and provenance

| run | documents | zero-PII | notes |
|---|---|---|---|
| e07 | 885 | 65 | 1,000 Nemotron briefs, density profile drawn blind |
| e08 | 4,469 | 376 | 5,000 fresh briefs (no document type shared with e07), density conditioned on the brief's PII affinity, 8 role relations, structural relations |

Annotation is correct by construction: a sampler owned the facts and their relations, a reasoning writer wrote the
document with inline tags, and a parser bound the tags back to the offered values, rejecting anything invented. A
relation judge confirmed each relation is actually expressed, a negatives judge checked no secondary person's value
is attributed to the subject, and an untagged-PII scan flagged leaks. 96 documents of e08 were read by 16
independent reviewers: tag boundaries 4.93/5, zero attribution errors, zero sex errors. Full numbers in each run's
`README.md` and `dataset/stats.json`.

## Rebuilding

`recipes/data_synth_to_gliner.sh <run folders>` in the nm-kvkk repository rebuilds `tr/` and `en/` from generator
runs; `kvkk_synth/dataset/build_6k.py` decides the split (600 test, 240 hard, seed 42) and writes `split_report.json`.
The `spans/` and `bio/` forms hold the same documents in the same places: test membership is taken from
`tr/test*.jsonl` by id and train by text hash, offsets and the BIO round-trip are re-verified, and the result is
recorded in `spans_report.json`.
