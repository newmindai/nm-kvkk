"""Build configs/labels/taxonomy_labels_tr.json — the bridge between the KVKK PII
taxonomy (118 entity nodes) and the natural Turkish label names the models
were trained on.

Sources, in priority order:
  1. configs/labels/e07_mapping.json `entity_label_mapping` — the
     names the rele07/e07 models saw in training (113 nodes, 3 pairs merged);
  2. NEW_NAMES below — hand-written names for the nodes e07 never had
     (5 nodes) and for the second member of each merged pair, so every node
     gets its own query name (un-merging: KEP != e-posta, YKN != TCKN,
     faks != telefon).

Output entry per node: {"id", "tr", "group", "group_tr", "sensitivity",
"out_of_scope", "source": "e07"|"new"}. Every "tr" is unique.

Usage:
  python scripts/build_taxonomy_labels.py [--taxonomy DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = PROJECT_ROOT / "kvkk_synth/taxonomy"  # index.json + taxonomies/*.json (v2.2 + roles)
DEFAULT_E07_MAPPING = PROJECT_ROOT / "configs/labels/e07_mapping.json"
DEFAULT_OUT = PROJECT_ROOT / "configs/labels/taxonomy_labels_tr.json"

# Nodes without an e07 name of their own. Style follows e07's natural
# Turkish names ("kimlik seri numarası", "sosyal güvenlik numarası", ...).
NEW_NAMES = {
    "kep_address": "KEP adresi",
    "foreigner_id_number": "yabancı kimlik numarası",
    "fax_number": "faks numarası",
    "nickname": "lakap",
    "religious_belief": "dini inanç",
    "philosophical_belief": "felsefi inanç",
    "union_membership": "sendika üyeliği",
    "military_service_status": "askerlik durumu",
}


def build(taxonomy_dir: Path, e07_mapping: Path) -> list[dict]:
    index = json.loads((taxonomy_dir / "index.json").read_text(encoding="utf-8"))
    e07 = json.loads(e07_mapping.read_text(encoding="utf-8"))["entity_label_mapping"]

    # A merged e07 name belongs to the FIRST node that claims it; the other
    # members of the pair must come from NEW_NAMES.
    claimed: dict[str, str] = {}
    entries: list[dict] = []
    for concept in index["concepts"]:
        for node in concept["nodes"]:
            node_id = node["id"]
            e07_name = e07.get(node_id)
            if node_id in NEW_NAMES:
                name, source = NEW_NAMES[node_id], "new"
            elif e07_name and e07_name not in claimed:
                name, source = e07_name, "e07"
            else:
                raise SystemExit(f"no unique name for {node_id!r} (e07={e07_name!r}) — add it to NEW_NAMES")
            if name in claimed:
                raise SystemExit(f"duplicate Turkish name {name!r}: {claimed[name]} and {node_id}")
            claimed[name] = node_id
            entries.append(
                {
                    "id": node_id,
                    "tr": name,
                    "group": concept["concept"]["en"],
                    "group_tr": concept["concept"]["tr"],
                    "sensitivity": concept["kvkk_sensitivity"],
                    "out_of_scope": bool(concept.get("out_of_kvkk_scope", False)),
                    "source": source,
                }
            )
    if len(entries) != index["total_nodes"]:
        raise SystemExit(f"built {len(entries)} entries, taxonomy declares {index['total_nodes']}")
    return entries


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--e07-mapping", type=Path, default=DEFAULT_E07_MAPPING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    entries = build(args.taxonomy, args.e07_mapping)
    args.out.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    new = sum(1 for e in entries if e["source"] == "new")
    print(
        f"wrote {args.out}: {len(entries)} nodes ({len(entries) - new} e07 names, {new} new), "
        f"{len({e['group'] for e in entries})} groups"
    )


if __name__ == "__main__":
    main()
