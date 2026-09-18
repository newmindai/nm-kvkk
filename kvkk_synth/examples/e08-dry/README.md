# examples/e08-dry — the 50-document dry run that preceded e08

A complete miniature run, kept as the offline reference for the parser:

| file | what |
|---|---|
| `bundles.jsonl` | the first 50 e08 bundles (sampler output: subject, entities, relations, persons, negatives, brief) |
| `raw-dry.parquet` | the writer's tagged output for them (GPT-5.6 Luna, flex, $0.048) |
| `dry2-records_full.jsonl`, `dry2-records.jsonl` | the 46 accepted documents as parsed by the final e08 parser (offsets, relations, negatives) |
| `dry2-rejects.jsonl` | the 4 rejected ones with their reasons (a generic "Suriye" untagged in an ethnic-group analysis, a name-guard false positive, a heading in square brackets, a report number in a zero-PII document) |
| `dry2-retry-bundles.jsonl` | the bundles the repair pass would take back to the writer |

Re-parse it offline and compare:

```bash
cd examples/e08-dry
python ../../pipeline/v2/parse_facts.py --raw raw-dry.parquet --bundles bundles.jsonl --facts-optional --prefix check-
diff <(cut -c1-200 check-records.jsonl) <(cut -c1-200 dry2-records.jsonl) && echo same
```

`tests/test_units.py::test_parser_reproduces_the_e08_dry_run` does exactly this and asserts every span offset and
every reject count is identical. The full e08 run (5,000 documents) is summarised next door in `examples/e08/`.
