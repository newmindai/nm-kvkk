#!/usr/bin/env python
"""Extract KVKK personal-data entities (and the relations that attach them to persons) from Turkish text
with a newmindai GLiNER2.5 KVKK model — out of the box, no label knowledge needed.

    pip install gliner2 huggingface_hub
    python extract_kvkk.py --model newmindai/gliner2.5-kvkk-tr-v2 document.txt
    python extract_kvkk.py --model newmindai/gliner2.5-kvkk-tr-v2 --text "Ahmet Yılmaz, TC 12345678901 ..."
    python extract_kvkk.py --model ... --labels kvkk19 --relations none *.txt        # only the 19 KVKK fields, no relations
    python extract_kvkk.py --model ... --labels all --relations all document.txt     # the whole taxonomy (slower)

The model repository ships `kvkk_schema.json`: every entity label and relation type with the exact query name
this model was trained on (Turkish or English). The script reads it, builds the prompt, runs the model on the
whole text in one pass, and prints JSON keyed by taxonomy ids — the same ids for every model of the release.
Output: {"entities": [{id, name, text, start, end, confidence}], "relations": [{id, name, head: {...}, tail: {...}}]}.
Offsets are character offsets into the input text (end exclusive).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_schema(model: str) -> dict:
    p = Path(model) / "kvkk_schema.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    if Path(model).is_dir():
        sys.exit(
            f"{model} has no kvkk_schema.json — copy it from the model's Hugging Face repository next to config.json"
        )
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model, "kvkk_schema.json")).read_text(encoding="utf-8"))


def check_library(model: str) -> None:
    """The Mursit-based model needs the GLiNER2 library from github.com/newmindai/nm-kvkk; the PyPI package loads it
    without error but ignores its native-tokenization settings and returns wrong output."""
    cfg_path = Path(model) / "config.json"
    if not cfg_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            cfg_path = Path(hf_hub_download(model, "config.json"))
        except Exception:
            return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("tokenization") == "native" or cfg.get("token_pooling") == "mean":
        import gliner2

        if not hasattr(gliner2, "__nm_kvkk__"):
            sys.stderr.write(
                "WARNING: this checkpoint needs the GLiNER2 library from https://github.com/newmindai/nm-kvkk "
                "(export PYTHONPATH=<repo>/GLiNER2); with the PyPI gliner2 package it loads but produces wrong output.\n"
            )


def build_schema(model, S: dict, labels: str, relations: str):
    ent_name = {e["id"]: e["name"] for e in S["entities"]}
    rel = {r["id"]: r for r in S["relations"]}
    if labels == "all":
        wanted = [e["id"] for e in S["entities"] if not e["out_of_scope"]]
    elif labels in S["subsets"]:
        wanted = S["subsets"][labels]
    else:
        wanted = [x.strip() for x in labels.split(",") if x.strip()]
        unknown = [x for x in wanted if x not in ent_name]
        if unknown:
            sys.exit(f"unknown entity ids: {unknown} (see kvkk_schema.json)")
    schema = model.create_schema().entities([ent_name[i] for i in wanted])
    rel_ids: list = []
    if relations == "all":
        rel_ids = list(rel)
    elif relations != "none":
        rel_ids = [x.strip() for x in relations.split(",") if x.strip()]
        unknown = [x for x in rel_ids if x not in rel]
        if unknown:
            sys.exit(f"unknown relation ids: {unknown} (see kvkk_schema.json)")
    if rel_ids:
        if S.get("relation_prompt") == "desc":
            schema = schema.relations({rel[r]["name"]: rel[r]["description"] for r in rel_ids})
        else:  # this model works best with bare relation names
            schema = schema.relations([rel[r]["name"] for r in rel_ids])
    return schema, {v: k for k, v in ent_name.items()}, {r["name"]: r["id"] for r in S["relations"]}


def to_ids(out: dict, ent_id: dict, rel_id: dict) -> dict:
    entities = []
    for name, spans in (out.get("entities") or {}).items():
        for s in spans:
            entities.append(
                {
                    "id": ent_id.get(name, name),
                    "name": name,
                    "text": s["text"],
                    "start": s["start"],
                    "end": s["end"],
                    "confidence": round(float(s.get("confidence", 0)), 4),
                }
            )
    entities.sort(key=lambda e: (e["start"], e["end"]))
    relations = []
    for name, insts in (out.get("relation_extraction") or {}).items():
        for inst in insts:
            head, tail = (inst["head"], inst["tail"]) if isinstance(inst, dict) else inst
            ends = lambda x: (
                {"text": x["text"], "start": x["start"], "end": x["end"]} if isinstance(x, dict) else {"text": x}
            )
            relations.append({"id": rel_id.get(name, name), "name": name, "head": ends(head), "tail": ends(tail)})
    return {"entities": entities, "relations": relations}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="text files (UTF-8); omit to read stdin, or use --text")
    ap.add_argument(
        "--model", required=True, help="Hugging Face repo id or local folder of a newmindai GLiNER2.5 KVKK model"
    )
    ap.add_argument("--text", help="text to process instead of files")
    ap.add_argument(
        "--labels",
        default="stack21",
        help="kvkk19 | stack21 | all | comma-separated taxonomy ids (default stack21: 19 KVKK fields + person and company names)",
    )
    ap.add_argument(
        "--relations",
        default="none",
        help="none | all | comma-separated relation ids (needs the persons among the labels)",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = ap.parse_args(argv)

    if args.text:
        docs = [("text", args.text)]
    elif args.inputs:
        docs = [(p, Path(p).read_text(encoding="utf-8")) for p in args.inputs]
    else:
        docs = [("stdin", sys.stdin.read())]

    check_library(args.model)
    from gliner2 import AutoExtractor

    kw = {"map_location": args.device} if args.device else {}
    model = AutoExtractor.from_pretrained(args.model, **kw)
    S = load_schema(args.model)
    schema, ent_id, rel_id = build_schema(model, S, args.labels, args.relations)

    results = {}
    for name, text in docs:
        out = model.extract(text, schema, threshold=args.threshold, include_confidence=True, include_spans=True)
        results[name] = to_ids(out, ent_id, rel_id)
    payload = results[docs[0][0]] if len(docs) == 1 else results  # one document -> flat; several files -> keyed by path
    js = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(js, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()
