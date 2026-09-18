"""Preflight an nm-kvkk-pii-6K training file against GLiNER2's *actual* gold-matching
rule and drop the records the boundary collator would reject.

Why this exists: `validate_data` in the trainer uses a substring check, but the
collator locates a mention by matching its word-token sequence as a whole-word
sublist of the tokenized text — and `SchemaTransformer._transform_record`
(processor.py:566-567) appends "." to any text that does not already end in
`.`/`!`/`?`. The whitespace splitter keeps "<url>." as ONE token, so a document that ends in a
bare URL gets that period glued onto its final token
("https://linkedin.com/in/x" -> "https://linkedin.com/in/x."), the declared surface
no longer matches, and training dies with
`ValueError: entity '<label>' was not found in sample N` at the first such batch.

The check here replicates exactly that rule (terminal period + the model's own word
splitter, lowercased), so it is deterministic — unlike running the collator, whose
label augmentation renames labels at random.

Usage:
  python scripts/preflight_nm6k_train.py --data datasets/nm6k/tr/train.jsonl \\
      --out datasets/nm6k/tr/train_clean.jsonl [--report datasets/nm6k/tr/preflight_tr.json]
  python scripts/preflight_nm6k_train.py --data ... --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "GLiNER2"))

from gliner2.processing.word_splitter import resolve_word_splitter  # noqa: E402

_SPLITTER = resolve_word_splitter(None)


def tokens(text: str) -> list[str]:
    return [t for t, _, _ in _SPLITTER(text, lower=True)]


def text_tokens(text: str) -> list[str]:
    """The token sequence the processor actually matches against (processor.py:566)."""
    if text and not text.endswith((".", "!", "?")):
        text = text + "."
    return tokens(text)


def find(sub: list[str], seq: list[str]) -> bool:
    if not sub or all(t == "" for t in sub):
        return False
    n = len(sub)
    return any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


def unmatchable(record: dict[str, Any]) -> list[tuple[str, str]]:
    """[(label, surface)] the collator would fail to locate in this record."""
    seq = text_tokens(record["input"])
    out = []
    for label, surfaces in record["output"].get("entities", {}).items():
        for surface in surfaces:
            if not find(tokens(surface), seq):
                out.append((label, surface))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=None, help="cleaned JSONL (records with an unmatchable surface dropped)"
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad, kept = [], []
    for i, record in enumerate(records):
        problems = unmatchable(record)
        if problems:
            bad.append({"index": i, "id": record.get("id"), "problems": problems, "text_tail": record["input"][-80:]})
        else:
            kept.append(record)

    report = {
        "data": str(args.data),
        "n_records": len(records),
        "n_dropped": len(bad),
        "n_kept": len(kept),
        "dropped": bad,
    }
    print(
        f"{args.data}: {len(records)} records · {len(bad)} unmatchable ({100 * len(bad) / max(len(records), 1):.2f} %)"
    )
    for entry in bad[:10]:
        labels = ", ".join(f"{lab}={surf!r}" for lab, surf in entry["problems"])
        print(f"   drop {entry['id']}: {labels}")
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.out and not args.check_only:
        with args.out.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}: {len(kept)} records")
    if args.check_only and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
