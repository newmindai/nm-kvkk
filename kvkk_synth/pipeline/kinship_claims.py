"""[e08] Neutral, sex-correct Turkish sentences for the kinship relations of the persons axis.

The taxonomy's templates_tr for kinship are scenario hints ("{tail}'in oğlu {head} okula kaydedildi."): they hard-code
'oğlu' (son) whatever the relative's sex and add scenario content (school, hearing). Used as the fact hint they
contradict the sexed persons block the writer also sees, and used as the judge's claim they make the judge reject a
correctly written daughter (block 1 of e08: child_of expressed 1 / 13, spouse_of 15 / 27). These sentences replace
them for both uses. Direction as in persons.py: the relative is the head, the subject the tail.
"""

SENT = {
    "child_of": {"F": "{head}, {tail}'in kızıdır.", "M": "{head}, {tail}'in oğludur."},
    "sibling_of": {"F": "{head}, {tail}'in kız kardeşidir.", "M": "{head}, {tail}'in erkek kardeşidir."},
    "spouse_of": {"F": "{head}, {tail}'in eşidir.", "M": "{head}, {tail}'in eşidir."},
    "mother_of": {"F": "{head}, {tail}'in annesidir.", "M": "{head}, {tail}'in annesidir."},
    "father_of": {"F": "{head}, {tail}'in babasıdır.", "M": "{head}, {tail}'in babasıdır."},
    "relative_of": {"F": "{head}, {tail}'in akrabasıdır.", "M": "{head}, {tail}'in akrabasıdır."},
    "emergency_contact_of": {
        "F": "{head}, {tail}'in acil durumda ulaşılacak kişisidir.",
        "M": "{head}, {tail}'in acil durumda ulaşılacak kişisidir.",
    },
}


def sentence(relation, head_name, tail_name, sex):
    tpl = SENT.get(relation)
    if not tpl:
        return None
    return tpl.get(sex or "M", tpl["M"]).format(head=head_name, tail=tail_name)


def rewrite_bundle(js):
    """Replace the fact_tr of every kinship relation that involves a secondary person; returns the number changed."""
    persons = {p["id"]: p for p in js.get("persons", [])}
    val = {e["id"]: e["value"] for e in js["entities"]}
    n = 0
    for x in js["relations"]:
        if x["relation"] not in SENT:
            continue
        p = (
            persons.get(x["head"])
            if x["head"] in persons and persons[x["head"]]["kind"] != "subject"
            else persons.get(x["tail"])
        )
        if p is None or p.get("kind") == "subject":
            continue
        s = sentence(x["relation"], val[x["head"]], val[x["tail"]], p.get("sex"))
        if s and s != x["fact_tr"]:
            x["fact_tr"] = s
            n += 1
    return n


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "bundles.jsonl"
    bs = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = sum(rewrite_bundle(b) for b in bs)
    with open(path, "w", encoding="utf-8") as f:
        for b in bs:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"kinship hints rewritten: {n} relations in {path}")
