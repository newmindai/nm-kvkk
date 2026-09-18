"""Score a GLiNER2 checkpoint's zero-shot relation extraction on the KVKK
relations pilot (datasets/kvkk_relations/v2/relations_gold.json).

For every record the model runs ONE combined forward pass — the 56 Turkish
entity labels of entities_eval.jsonl plus all 57 English relation types (each
with a short English description phrased in the gold head->tail direction) —
via ``model.extract(text, schema, threshold, include_confidence=True,
include_spans=True)``. Only the top-level ``relation_extraction`` output is
scored here (entity scoring lives in scripts/eval_checkpoint.py; in the
standard runtime the relation decode is independent of the entity queries).

Matching (coreference-aware, per the gold file's deduped mention lists):
a predicted (head_surface, relation, tail_surface) is a true positive iff the
relation type matches and the head surface equals ANY gold head mention and
the tail surface ANY gold tail mention of one still-uncredited gold triple of
that record — surfaces compared casefolded after whitespace normalization
(the same normalization as scripts/span_metrics.py). Matching is greedy
one-to-one: each gold triple is creditable once and each prediction consumes
at most one gold triple. A secondary "lenient direction" score also accepts a
prediction whose head/tail surfaces are swapped relative to gold, because the
pristine model's learned argument order is reversed for some types (e.g. it
emits person->file for depicted_in where gold is file->person).

One native extraction pass runs per requested threshold (the decode gates raw
pair scores, so native passes are exact where offline confidence filtering
could interact with edge deduplication). The FIRST threshold is primary.

Usage:
  python scripts/eval_relations.py \\
      --model models/gliner2.5-multi-v1 --device mps \\
      [--gold datasets/kvkk_relations/v2/relations_gold.json] \\
      [--entities datasets/kvkk_relations/v2/entities_eval.jsonl] \\
      [--thresholds 0.3,0.5] [--out report.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "GLiNER2"))  # local clone wins over any installed gliner2

logger = logging.getLogger("eval_relations")

# One description per gold relation type, phrased head-first / tail-second in
# the gold direction (derived from the head_label_raw/tail_label_raw pairs of
# relations_gold.json). GLiNER2 prompts these as "name: description".
RELATION_DESCRIPTIONS: dict[str, str] = {
    "verdict_in_case": "Criminal verdict issued in the court case",
    "depicted_in": "Photo or recording file depicts the person",
    "residence_of": "Address is the residence of the person",
    "conviction_of": "Criminal conviction belongs to the person",
    "party_in_case": "Person is a party in the court case",
    "workplace_of": "Workplace location where the person works",
    "coordinates_of": "GPS coordinates locate the person",
    "work_history_of": "Work history information belongs to the person",
    "credential_of": "Password or session token belongs to the person",
    "attire_of": "Clothing description of the person",
    "father_of": "First person is the father of the second person",
    "vehicle_registration_of": "Vehicle registration document number belongs to the person",
    "same_vehicle": "License plate and chassis number belong to the same vehicle",
    "foreigner_id_of": "Foreigner identity number belongs to the person",
    "vehicle_plate_of": "Vehicle license plate belongs to the person",
    "vehicle_vin_of": "Vehicle chassis (VIN) number belongs to the person",
    "tax_id_of": "Tax identification number belongs to the person",
    "subscription_of": "Subscription number belongs to the person",
    "health_file_of": "Health report or file number belongs to the person",
    "medication_of": "Medication is used by the person",
    "blood_type_of": "Blood type of the person",
    "diagnosis_of": "Medical or mental condition diagnosed in the person",
    "phone_of": "Phone or fax number belongs to the person",
    "email_of": "E-mail or KEP address belongs to the person",
    "part_of_address": "Street line or city is part of the full address",
    "salary_of": "Salary information of the person",
    "position_of": "Job title held by the person",
    "education_of": "Education information of the person",
    "iban_of": "IBAN or bank account number belongs to the person",
    "passport_of": "Passport number belongs to the person",
    "nationality_of": "Nationality of the person",
    "disability_of": "Disability status of the person",
    "treatment_of": "Medical procedure or treatment applied to the person",
    "relative_of": "First person is a relative of the second person",
    "age_of": "Age of the person",
    "card_component_of": "Card expiry date belongs to the payment card number",
    "card_of": "Payment card number belongs to the person",
    "bank_of_account": "SWIFT or BIC code is the bank of the account number",
    "account_of": "Username or account belongs to the person",
    "marital_status_of": "Marital status of the person",
    "order_of": "Order reference belongs to the person",
    "ip_used_by": "IP address is used by the person",
    "profile_url_of": "Profile URL belongs to the person",
    "child_of": "First person is the child of the second person",
    "sibling_of": "First person is a sibling of the second person",
    "spouse_of": "First person is the spouse of the second person",
    "mother_of": "First person is the mother of the second person",
    "date_of_birth_of": "Date of birth of the person",
    "customer_number_of": "Customer number belongs to the person",
    "genetic_of": "Genetic data reference belongs to the person",
    "sexual_life_of": "Sexual life information about the person",
    "employer_of": "Company is the employer of the person",
    "national_id_of": "National identity number belongs to the person",
    "invoice_recipient": "Invoice document is addressed to the person",
    "political_view_of": "Political opinion of the person",
    "ethnicity_of": "Ethnic origin of the person",
    "employee_id_of": "Employee ID number belongs to the person",
}


def norm(value: Any) -> str:
    """span_metrics normalization + casefold for case-insensitive comparison."""
    if isinstance(value, dict):
        value = value.get("text", "")
    return " ".join(str(value).split()).casefold()


def load_model(checkpoint: Path, device: str):
    import torch

    from gliner2 import AutoExtractor

    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is not available")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    model = AutoExtractor.from_pretrained(str(checkpoint), map_location=device)
    model.eval()
    return model


def parse_pair(item: Any) -> tuple[str, str, float | None]:
    """(head_surface, tail_surface, confidence) from either output shape."""
    if isinstance(item, dict):
        head, tail = item.get("head") or {}, item.get("tail") or {}
        confidence = head.get("confidence") if isinstance(head, dict) else None
        return norm(head), norm(tail), confidence
    head, tail = item  # plain (head, tail) tuple/list
    return norm(head), norm(tail), None


def match_records(
    gold_records: list[dict[str, Any]],
    predictions: dict[str, dict[str, list[Any]]],
    relation_types: list[str],
    lenient_direction: bool,
) -> dict[str, Any]:
    """Greedy one-to-one matching of predicted pairs against gold triples."""
    counts = {rel: [0, 0, 0, 0] for rel in relation_types}  # tp, fp, fn, support
    details: list[dict[str, Any]] = []
    for record in gold_records:
        record_id = record["id"]
        rel_pred = predictions.get(record_id, {})
        gold_by_type: dict[str, list[dict[str, Any]]] = {}
        for triple in record["triples"]:
            gold_by_type.setdefault(triple["relation"], []).append(triple)
        for rel in relation_types:
            gold_triples = gold_by_type.get(rel, [])
            credited = [False] * len(gold_triples)
            counts[rel][3] += len(gold_triples)
            for item in rel_pred.get(rel, []):
                head, tail, confidence = parse_pair(item)
                hit = None
                for j, triple in enumerate(gold_triples):
                    if credited[j]:
                        continue
                    heads = {norm(m) for m in triple["head_mentions"]}
                    tails = {norm(m) for m in triple["tail_mentions"]}
                    forward = head in heads and tail in tails
                    swapped = lenient_direction and head in tails and tail in heads
                    if forward or swapped:
                        hit = (j, "swapped" if not forward else "forward")
                        break
                if hit is not None:
                    credited[hit[0]] = True
                    counts[rel][0] += 1
                    verdict = "TP" if hit[1] == "forward" else "TP(swapped)"
                else:
                    counts[rel][1] += 1
                    verdict = "FP"
                details.append(
                    {
                        "record": record_id,
                        "relation": rel,
                        "verdict": verdict,
                        "head": head,
                        "tail": tail,
                        "confidence": confidence,
                    }
                )
            for j, triple in enumerate(gold_triples):
                if not credited[j]:
                    counts[rel][2] += 1
                    details.append(
                        {
                            "record": record_id,
                            "relation": rel,
                            "verdict": "FN",
                            "head": " | ".join(triple["head_mentions"]),
                            "tail": " | ".join(triple["tail_mentions"]),
                            "confidence": None,
                        }
                    )
    per_type = {}
    for rel in relation_types:
        tp, fp, fn, support = counts[rel]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_type[rel] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    micro = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return {"micro": micro, "per_type": per_type, "details": details}


def print_report(threshold: float, strict: dict[str, Any], lenient: dict[str, Any]) -> None:
    supported = {rel: s for rel, s in strict["per_type"].items() if s["support"] > 0 or s["fp"] > 0}
    width = max(len("relation"), *(len(rel) for rel in supported))
    print(f"\n=== threshold {threshold} — strict direction (per type; only rows with gold support or predictions) ===")
    print(f"{'relation':<{width}}  {'P':>6}  {'R':>6}  {'F1':>6}  {'tp':>3} {'fp':>3} {'fn':>3}  {'support':>7}")
    for rel, s in sorted(supported.items(), key=lambda kv: -kv[1]["support"]):
        print(
            f"{rel:<{width}}  {s['precision']:>6.3f}  {s['recall']:>6.3f}  {s['f1']:>6.3f}  "
            f"{s['tp']:>3} {s['fp']:>3} {s['fn']:>3}  {s['support']:>7}"
        )
    for name, report in (("strict", strict), ("lenient-direction", lenient)):
        m = report["micro"]
        print(
            f"micro [{name:>17}]  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
            f"(tp {m['tp']} fp {m['fp']} fn {m['fn']})"
        )


def run(args) -> dict[str, Any]:
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    relation_types = gold["relation_types"]
    entity_labels = gold["eval_labels"]
    texts: dict[str, str] = {}
    with Path(args.entities).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                texts[record["id"]] = record["input"]
    missing = [r["id"] for r in gold["records"] if r["id"] not in texts]
    if missing:
        raise SystemExit(f"records missing from {args.entities}: {missing}")
    # Descriptions: the gold file's own relation_descriptions (e07+) take
    # precedence; the built-in v2 dict is the fallback (v2 gold has no such key,
    # so v2 behavior is unchanged).
    descriptions = {**RELATION_DESCRIPTIONS, **gold.get("relation_descriptions", {})}
    unknown = [rel for rel in relation_types if rel not in descriptions]
    if unknown:
        raise SystemExit(f"no description for relation types: {unknown}")

    model = load_model(args.model, args.device)
    logger.info("loaded %s from %s on %s", type(model).__name__, args.model, args.device)
    schema = model.create_schema().entities(entity_labels).relations({rel: descriptions[rel] for rel in relation_types})

    thresholds = [float(t) for t in args.thresholds.split(",")]
    result: dict[str, Any] = {
        "model": str(args.model),
        "gold": str(args.gold),
        "entities": str(args.entities),
        "device": args.device,
        "thresholds": thresholds,
        "primary_threshold": thresholds[0],
        "n_records": len(gold["records"]),
        "n_gold_triples": sum(len(r["triples"]) for r in gold["records"]),
        "relation_descriptions": {rel: descriptions[rel] for rel in relation_types},
        "by_threshold": {},
    }
    for threshold in thresholds:
        started = time.time()
        predictions: dict[str, dict[str, list[Any]]] = {}
        entity_outputs: dict[str, Any] = {}
        for record in gold["records"]:
            out = model.extract(
                texts[record["id"]], schema, threshold=threshold, include_confidence=True, include_spans=True
            )
            predictions[record["id"]] = out.get("relation_extraction") or {}
            entity_outputs[record["id"]] = out.get("entities") or {}
        elapsed = time.time() - started
        strict = match_records(gold["records"], predictions, relation_types, lenient_direction=False)
        lenient = match_records(gold["records"], predictions, relation_types, lenient_direction=True)
        print_report(threshold, strict, lenient)
        print(f"{len(gold['records'])} records | {elapsed:.1f} s | threshold {threshold} | {args.device}")
        result["by_threshold"][str(threshold)] = {
            "elapsed_s": round(elapsed, 1),
            "strict": {"micro": strict["micro"], "per_type": strict["per_type"]},
            "lenient_direction": {"micro": lenient["micro"], "per_type": lenient["per_type"]},
            "details": strict["details"],
            "lenient_details": lenient["details"],
            "raw_predictions": predictions,
            "entity_outputs": entity_outputs,
        }
    result["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="checkpoint dir (config.json + model.safetensors)")
    parser.add_argument("--gold", type=Path, default=PROJECT_ROOT / "datasets/kvkk_relations/v2/relations_gold.json")
    parser.add_argument(
        "--entities",
        type=Path,
        default=PROJECT_ROOT / "datasets/kvkk_relations/v2/entities_eval.jsonl",
        help="entity eval JSONL (source of record ids + input texts)",
    )
    parser.add_argument("--thresholds", default="0.3,0.5", help="comma list; first is primary, one native pass each")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--out", type=Path, default=None, help="write full report JSON")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
