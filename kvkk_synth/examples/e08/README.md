# e08 — 5,000 documents, GPT-5.6 Luna (flex), fresh Nemotron briefs, relation targets

Scale-up of e07 (1,000 documents) to 5,000, on the fix list of the e07 pipeline review
(the e07 run, the synthetic-v2 dataset) and the relations plan of the v3 design. Writer, judges, tiers, geography, parser and
repair loop are those of e07; the inputs and the sampler changed.

## Inputs

| input | file | what |
|---|---|---|
| briefs | `briefs5000.jsonl` | 5,000 documents over 1,787 Nemotron-PII briefs (`briefs/nemotron_catalogue_fresh_for_e08.parquet`, 1,789 rows), each brief used at most 3 times, 57 domains, **no document type in common with e07** (user: completely different samples). `briefs_catalogue.jsonl` = the 1,789 briefs with their embedding text. |
| taxonomy | `code/taxonomy/` | v2.2 (118 entity nodes, 105 relations) **+ `taxonomies/16_relations_roles.json`** (8 role relations proposed from the vekaletname ground truth: represented_by, authorised_signatory_of, owner_of_company, guarantor_of, heir_of, tenant_of, witness_of, property_at) = 113 relations, validated (0 errors). |
| entity scores | `scores-e5-cat.json` | multilingual-e5-large-instruct cosine brief × node (local, conda `onto`), keyed by brief uid |
| role scores | `scores-roles.json` | same model, brief × 23 role descriptions (15 of e07 + the 8 new roles), keyed by uid |
| geography | `code/geo/` | NVİ address hierarchy parquet (81 il, 975 ilçe, 74,659 mahalle, 1.1 M streets) |

## What changed since e07 (code/, every change marked `[e08]`)

Sampler `sampler8.py` (on top of `sampler7.py`):
- scores keyed by brief **uid** (e07 keyed by bundle_id and broke when a brief was reused);
- **density by affinity**: a brief's PII affinity = sum of its top-10 centred cosines; with noise (half the spread) the
  5,000 documents are ranked and cut 8 / 22 / 45 / 25 % into zero / sparse / normal / dense. Mean affinity per
  profile: zero 0.065, sparse 0.115, normal 0.183, dense 0.282. e07 drew the profile blind, which forced facts onto
  briefs that carry no PII by nature (its coherence loss);
- pool: ≥ 1 tier-A/C node per non-zero pool; a label above the floor (150 bundles) goes to the back of the candidate
  list; a rare label is promoted only where its own centred score is positive; one profession registry per subject
  (bar / judge / expert / doctor); a TCKN or a foreigner ID, never both; the role relations are never drawn for the
  subject (the subject is a party, not a witness of their own record);
- relations: **least-used relation** per node; **structural relations** (card ↔ CVV/expiry, IBAN ↔ SWIFT, plate ↔
  VIN, street/district/zip ↔ address, parcel ↔ property address, MERSİS / trade registry / tax number ↔ company)
  added next to the person relation with p = .35 — e07 never produced them for nodes that also had a person
  relation; **quota pass** lifting every sampler-owned relation type to ≥ 40 bundles on the briefs that score it
  highest (26 relation types are writer-owned — memberships, health, beliefs, salary, job title … — and come from the
  writer's tier-B facts; `member_of` is writer-owned too: a company is an employer, never a membership);
- persons: 23 embedding-scored roles, threshold .02, ≥ 25 % of the non-zero bundles multi-person (quota pass on the
  best-scoring single-person bundles); the 8 role relations are wired in `persons.py::ROLE_SPECS` with the side the
  subject takes (vekalet veren → vekil; kefil → borçlu; heir ↔ deceased and tenant ↔ landlord in both directions;
  signatory / owner → the company in the document; witness → the subject's case or notarial act) and the words the
  writer sees; every counterpart / bystander / role person and 85 % of relatives carry ≥ 1 own datum, each an
  explicit negative against the subject;
- values: `values8.py` gives format-shaped generators to the 32 nodes that v2.2 leaves with a single example (e07:
  453 of 4,375 mentions were the example string) and a person-consistent ICAO passport MRZ.

Parser / generator / judges:
- `v2/parse_facts.py`: auto-repair skips any overlap with a tagged span (not only containment) and any value that
  the bundle lists under two labels; an untagged "T.C." is a prefix (T.C. kimlik no, T.C. Sağlık Bakanlığı) and
  never an untagged-occurrence reject, and "[T.C.]nationality" right before "kimlik" is a label mismatch (found by
  the dry run: the first version of this guard was case-sensitive and rejected document 3); a repeated word right
  before a tag is a flag;
- `gen_direct.py`: transport-error crash fixed (`r = None`), empty outputs are regenerated on resume;
- `verify_relations7.py` / `scan_untagged.py`: a judge failure marks the record (`judge_failed`, `FLAG …`) and
  `build_dataset.py` moves it to the flagged bucket — nothing unverified is accepted by default; the relation judge
  prompt carries the field co-location rule;
- `judge_negatives.py`: claim wording for the role relations (X is the signatory of «company»);
- `tiers.canonical_relations`: explicit priority `company_name → employer_of` (e07's 27 `member_of` writer relations
  are relabelled in the e07 dataset);
- `audit_relations.py`: gate after the first 2,000 documents — the chain stops if a relation type offered ≥ 16 times
  is expressed in < 25 % of them or first-pass acceptance falls below 80 %.

## Bundles (`bundles.jsonl`, `soundness_report.json`)

5,000 bundles: 400 zero / 1,100 sparse / 2,250 normal / 1,250 dense. 4,600 non-zero bundles carry 1 / 2 / 3 / 4
persons = 3,450 / 931 / 172 / 47 (25.0% multi-person; secondary kinds: related 431, counterpart 375, bystander 161,
role 449), 1,917 negatives, 48 multi-person bundles without a negative. All 89 tier-A/C nodes instantiated (rarest
mothers_maiden_name 182 bundles); all 86 sampler-owned relation types ≥ 40 bundles (the quota pass added 19 relations in 3 types: heir_of, mother_of, owner_of_company).

## Chain

`run_e08.sh` — sampler → block 1 (2,000) generate / parse / relation judge / audit gate → rest → parse → writer
judges → untagged scan → repair → judges on repaired → relation judge → coherence → dataset → sex check → review
sample → viewer. A 50-document dry run preceded the run (`raw-dry.parquet`, $0.048; `dry-*.jsonl` = parse + all four
judges before the "T.C." fix: 45 accepted, relation judge 137 kept / 9 unexpressed, writer relations 3/3 confirmed,
negatives 13 checked / 0 violated, secondary-person facts 13/13 attributed, untagged scan 0 findings; `dry2-*.jsonl` =
re-parse after the fix: 46 accepted, 4 rejects = a generic "Suriye" untagged in an ethnic-group analysis, the
name-guard false positive "Ekolojik Temel Durum", a heading in square brackets, a report number in a zero-PII
document — all of the kind the repair pass handles). The chain was started at 18:32 and restarted at 18:33 after a
working-directory bug in the sex-check step was fixed (nothing generated was reused).

## Block 1 and the audit stop (2,000 documents, 18:33–19:55)

Generation: 2,000 documents, $1.97, 63 min, 0 empty outputs. Parse: 1,783 accepted / 5 flagged / 212 rejected
(88.6 %). Relation judge: 4,633 kept, 268 unexpressed (5.5 %), $0.27.

**The audit gate stopped the chain**, and it was right to for one reason and wrong for another:
- *Wrong*: it treated pool relations expressed in < 25 % of the bundles that offered them as broken. In a pool design
  the writer picks ~1–10 of 6–15 candidates, so 3–40 % is the expected range (crypto wallets 3 %, MRZ 7 %, address
  parts 4 % — the last cannot be expressed at all when the address is one tagged span). The gate now flags a pool
  relation only when it is offered ≥ 50 times and expressed in < 2 % of them.
- *Right*: `child_of` was expressed in 1 of 13 accepted documents that named the child, `spouse_of` in 15 of 27.
  Cause: the claim shown to the relation judge (and the hint shown to the writer) was the taxonomy's scenario
  template, "{tail}'in oğlu {head} okula kaydedildi." — it hard-codes *oğlu* (son) whatever the relative's sex and
  adds school enrolment, so for a daughter in a visa file the judge correctly answered "not expressed". Fix
  (`code/kinship_claims.py`): neutral, sex-correct sentences ("X, Y'in kızıdır.") for the seven kinship relations,
  used by the judge (`verify_relations7.py`), by the sampler at export and applied in place to the frozen bundles
  (431 hints rewritten; `bundles.before-kinship-hints.jsonl` keeps the originals). Block 1 was generated with the
  old hints, but the writer followed the sexed persons block; the sex-check stage catches any *oğlu* written for a
  daughter. Re-judging block 1's 404 multi-person documents with the new claims: 1,660 kept / 72 unexpressed
  (4.2 %), and the gate's person check — the relation must be expressed in ≥ 50 % of the accepted documents that
  name the person — passes for every kinship and role relation (child 13/13, spouse 27/27, mother 13/13, father
  14/14, sibling 12/13, relative 12/12, emergency contact 40/40, vekil 13/14, kefil 37/37, heir 17/17, tenant 21/22,
  signatory 28/28, owner 21/21; witness 10/20 because the witnessed case number is often not written).
  Counterparts and bystanders have no relation by design and are excluded from the check. Report: `audit-2000.json`.

Fixed while block 1 was being judged, in force from the full parse (step 6) on:
- tier-B validators loosened on the block-1 evidence (94 refused spans, most of them legitimate): disability
  statuses without the word *engel* ("sürekli kısmi iş göremezlik"), organisation names without a corporate
  suffix ("Marmara Hukuk Danışmanlık", any capitalised 2–7-word name), "İslam", any ≥ 3-word work history, longer
  diagnoses / procedures / titles. Height, weight and blood pressure tagged as `biometric_data_reference` stay
  refused (wrong label). Effect on block 1: 1,838 accepted instead of 1,783.
- parser false positives: uppercase names now bind through a Turkish-aware lowercase ("NİHAYET SEVİNÇ"); date + time
  strings ("14.05.2025 09:30", "2025-02-14 16:00"), dot-grouped amounts and subnet masks are not "unlisted numbers";
  demographic values match case-sensitively ("erkek kardeşi" is not the gender field).

The chain continues from step 5 with `run_e08_cont.sh` (same steps as `run_e08.sh`; relation judge at 12 workers).

## Results

Cost $9.03 in all (generation $6.84, judges $1.82, repair $0.30, sex fix $0.01), 18:33 to 01:35.

### Dataset (`dataset/`, `dataset/stats.json`)

| | |
|---|---|
| training documents | **4,469** |
| zero-PII documents, kept apart in `negatives.jsonl` | 376 |
| tagged mentions | 24,386 on **110/118** labels |
| relations | **13,255** on **105/113** types |
| explicit negative pairs | 774 |
| multi-person documents | 1,111 |
| writer-added free-text facts | 1,107 |
| mentions per document | 5.5 (non-name 3.2) |
| buckets | accepted 4,845 · flagged 53 · discarded 102 |
| stages | first pass 4,503 · repaired 320 · sex-fixed 22 |

Files: `records.jsonl` (text + offsets + relations), `records_full.jsonl` (everything, including judge evidence),
`gliner.jsonl`, `tr_pii_relations_e08.parquet`, `negatives.jsonl`, `discarded.jsonl`, `stats.json`; `manifest.jsonl`
says what happened to each of the 5,000 bundles. Distribution analysis: `analysis/entity_distribution.md`.

Relations, which were the point of this build: 13,255 instances over 105 types,
806 of them person relations (kinship and the 8 role relations) and 860
structural (value ↔ value: card ↔ CVV, IBAN ↔ SWIFT, plate ↔ VIN, parcel ↔ address, company ↔ registry numbers).
The vekaletname axis the user asked for is present throughout: power of attorney 50,
guarantor 88, company signatory 87,
company owner 39, tenant 66,
heir 44, witness 36,
property at an address 107.

### Independent review (16 Sonnet reviewers, 96 documents, every quote verified)

| dimension | e08 | e07 |
|---|---|---|
| fluency | 4.78 | 4.73 |
| **coherence** | **4.05** | 3.83 |
| tag boundaries | 4.93 | 4.98 |
| relation expression | 4.67 | 4.56 |
| completeness and leaks | 4.74 | 4.88 |
| overall | **4.63** | 4.60 |

| | e08 | e07 |
|---|---|---|
| verdicts | keep 59 · fix 37 · **drop 0** | keep 27 · fix 20 · drop 1 |
| documents with no issue | 33 / 96 | 16 / 48 |
| **attribution errors** | **0** | 1 |
| **sex errors** | **0** | (not scored) |
| persons natural / forced | 45 / 12 | 54 / 6 |
| facts used naturally / forced | 261 / 29 | – / 27 |
| writer-added ok / bad · tier-B misses | 39 / 0 · 16 | 17 / 0 · 0 |
| structural relations ok / bad | 24 / 0 | (none produced) |
| value realism | 4.73 | (not scored) |
| leaks in zero-PII documents | 0 | 0 |
| cheap judge vs reviewers | within 1 point on 76 / 96; means 4.4 vs 4.05 | 41 / 48; 4.27 vs 3.83 |

Reviewer coherence by group: dense 4.53 · sparse 4.20 · repaired 4.19 · normal 3.95 · **multi-person 3.74**.

### Reading

1. **Conditioning density on brief affinity worked.** Coherence 3.83 → 4.05, and the profile ordering inverted:
   in e07 dense documents were the worst (3.87) because density was drawn blind; in e08 they are the best (4.53),
   because a dense profile now only lands on a brief whose own embedding says it carries personal data. Forced
   facts fell to 29 in 290 used facts (10 %).
2. **The annotation layer is sound at 5,000 documents.** Tag boundaries 4.93, zero attribution errors across 57
   secondary persons, zero sex errors, zero bad writer-added values, zero wrong structural relations, no leaks in
   any zero-PII document, and every one of the 24,386 spans matches the text at its recorded offsets
   (checked exhaustively, not sampled). Four reviewer "parser disagreements" are the auto-repair binding an untagged
   repeat of a known value: correct behaviour that the tagged view simply does not show.
3. **Multi-person documents are now the weak axis** (coherence 3.74; 12 of 57 secondary persons judged forced).
   The role scoring puts the right *kind* of person in, but the 0.02 threshold still admits a company signatory
   into a public PIN notice. Next run: raise the role threshold, or require the role's centred score to be in the
   brief's own top few.
4. **Remaining smaller defects**, all found by the reviewers and now guarded in `build_dataset.py` (documents moved
   to the flagged bucket) and fixed in the sampler for later runs: a male subject carrying a "kızlık soyadı"
   (19 documents; `SEX_ONLY` in `sampler8.py`), the given name "Kadın" reading as the common noun "woman"
   (3 documents; the sampler now excludes it), and an age contradicting the birth date because the age was drawn
   before the birth date existed (6 documents; the sampler now fixes the birth date first). 16 tier-B misses remain:
   free-text personal facts the writer wrote but did not tag.
5. **Coverage**: 101 of 118 labels appear in 10 or more documents (e07: 71).
   17 labels stay below that floor, nearly all special-category kinds the writer only adds when a document calls for
   them (attire, sexual life, philosophical belief, union membership) plus three rare registry numbers.
