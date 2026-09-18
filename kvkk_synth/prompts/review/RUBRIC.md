# Evaluation rubric — synthetic Turkish PII documents with inline tags and relations

You are evaluating machine-generated Turkish documents intended as training data for
PII named-entity recognition (GLiNER-style, label names given as text) and relation
extraction (GLiREL-style). Each input row contains:

- `document_type`, `domain`, `document_format` — the brief the generator was given
- `facts_that_must_be_expressed` — Turkish fact sentences the document had to convey
- `entities` — the exact values that had to appear, with their taxonomy label (node id)
- `relations_kept_by_judge` / `relations_dropped_by_judge` — a cheap LLM judge's verdict on
  whether each relation is expressed in the text (you are the second opinion)
- `text_tagged` — the document with inline tags `[value]label`; the tag encloses exactly
  the span that becomes a labeled entity; Turkish apostrophe suffixes must stay OUTSIDE the
  tag (`[Ankara]city'da` is correct, `[Ankara'da]city` is wrong)

Label definitions (Turkish) are in `taxonomy/taxonomies/*.json` (field `description` per node)
and span rules in `taxonomy/POLICY.md` — consult them when a boundary question arises.

## Score every document on five dimensions, 1 (bad) to 5 (excellent)

1. **fluency** — grammatical, natural Turkish; correct case/possessive suffixes around tagged
   values; register appropriate to the document type. Penalize translationese, agreement
   errors ("Kaya'in"), robotic repetition.
2. **coherence** — do the facts sit naturally in this document type, or are they forced
   ("Hamile fizik tedavi uygulandığına dair rapor tahvil dosyasında saklanmaktadır" in a bond
   certificate)? Would a Turkish reader find the document plausible?
3. **tag_boundaries** — is every tagged span exactly the entity (no missing/extra words, suffix
   outside, no nested or merged entities)? Is the label the right node for that span?
4. **relation_expression** — is each relation in `relations_kept_by_judge` actually stated or
   unambiguously implied (not mere co-occurrence)? Did the judge wrongly drop any in
   `relations_dropped_by_judge`? Is attribution to the right person clear?
5. **completeness_and_leaks** — any personal data left UNTAGGED (other people's names,
   organization names, numbers, dates of birth, addresses)? Any annotation/taxonomy
   meta-language in the text ("Madde 6", "coreference", "etiket")? Placeholders, markdown
   artifacts, truncation, disclaimers like "eğitim amaçlı"?

Then give a **verdict**: `keep` (usable as is), `fix` (usable after a small mechanical fix you
name), or `drop` (misleading as training data).

## Output

Write ONE JSON file at the path you were given, with this exact shape:

```json
{
  "docs": [
    {
      "bundle_id": 3,
      "scores": {"fluency": 4, "coherence": 3, "tag_boundaries": 5, "relation_expression": 4, "completeness_and_leaks": 5},
      "verdict": "keep",
      "issues": [
        {"type": "boundary|label|relation|leak|fluency|coherence|judge_error|other",
         "quote": "exact quote from text_tagged", "note": "what is wrong and what it should be"}
      ]
    }
  ],
  "global_observations": ["..."],
  "recommendations": ["..."]
}
```

Rules: evaluate EVERY row in your input file (do not sample); quote exact text for every
issue; be strict — this data trains a model, so a subtly wrong span is worse than an
awkward sentence; do not modify any file other than your output file.
