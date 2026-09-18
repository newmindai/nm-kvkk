# Reviewer instructions — e08 (given verbatim to each Sonnet reviewer subagent, one chunk each)

You are an independent reviewer of synthetic Turkish PII training documents. Read the rubric first:
`prompts/review/RUBRIC.md` (five dimensions 1–5, verdict keep / fix / drop, exact quotes
for every issue). Label definitions: `runs/<date>-<id>/code/taxonomy/taxonomies/*.json`.

Your input: `runs/<date>-<id>/eval/chunk-NN.jsonl` — one JSON row per document. Fields beyond the
rubric's: `profile` (sparse / normal / dense / zero) and `fact_target` (how many offered facts the writer was asked
to use), `named_persons` (every person that had to be named, with the role the writer was told: the subject, a
relative with a sexed role word, a counterpart such as a physician or lawyer, a bystander, or a **role person**:
vekil (attorney-in-fact), kefil (guarantor), mirasçı / muris (heir / deceased), kiracı / kiraya veren (tenant /
landlord), şirket imza yetkilisi / sahibi (company signatory / owner), tanık (witness)), `facts_offered` (every
candidate fact with `used` = whether the writer used it), `entities_offered`, `writer_added` (free-text facts the
writer added on its own, of the allowed kinds) with `relations_writer`, `relations_negative_used` (values that
belong to a secondary person and must NOT be attributed to the subject), `relations_kept` (the cheap judge kept
these as expressed) and `relations_unexpressed` (dropped), `stage` (gen = first pass, repair = rewritten once),
`zero_pii_document` (must contain no personal data at all), `locale` (the real Turkish il / ilçe / mahalle the
addresses come from).

Evaluate EVERY row. For each document, besides the rubric's `scores`, `verdict` and `issues`, report:

- `parser_agreement`: `{"agree": true|false, "note": "..."}` — do the tags in `text_tagged` and `entities_bound`
  agree with what you see (spans exact, labels right)?
- `facts_used_naturally` / `facts_forced`: of the facts marked `used`, how many sit naturally in this document
  type and how many are forced in (a diagnosis in a bond certificate, a crypto wallet in a school form)?
- `persons_natural` / `persons_forced`: of the secondary named persons, how many belong in this document in the
  stated role and how many are forced (a company signatory in a public PIN notice)?
- `attribution_errors`: values presented as the wrong person's (a relative's TCKN written as the subject's, the
  subject shown as the vekil instead of the vekalet veren, the kefil and the borçlu swapped).
- `sex_errors`: a relative or role person described with words that contradict the stated sex (kızı for a son,
  "hanım" for a man) or a name that reads as the other sex.
- `writer_added_ok` / `writer_added_bad`: writer-added values that are realistic and correctly tagged vs. invented,
  placeholder-like or mis-tagged.
- `tierb_misses`: free-text personal facts of the allowed kinds (diagnosis, medication, job title, salary,
  membership, belief, education, work history …) that the writer WROTE but did not tag.
- `structural_ok` / `structural_bad`: relations between two non-person values (card ↔ CVV or expiry, IBAN ↔ SWIFT,
  plate ↔ VIN, parcel ↔ address, company ↔ MERSİS / trade registry / tax number): correctly co-located and
  attributable vs. wrong.
- `value_realism`: 1–5 — do the values look like real Turkish data of that kind (registry numbers, policy numbers,
  parcel notation, MRZ lines, addresses)?
- for a `zero_pii_document`: `leaked_pii` = list of any personal data you find (should be empty).

Issue `type` values: boundary | label | relation | leak | fluency | coherence | judge_error | attribution | sex |
value_realism | structural | other. Quote exact text from `text_tagged` for every issue (the aggregator verifies
each quote as a substring). Be strict: this data trains a model.

Write ONE JSON file: `runs/<date>-<id>/eval/out-NN.json` with
`{"docs": [...], "global_observations": [...], "recommendations": [...]}`. Do not modify any other file. Report
only the path you wrote and a one-line count (docs, keep / fix / drop).
