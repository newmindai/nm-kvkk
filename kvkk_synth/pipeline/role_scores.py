#!/usr/bin/env python3
"""Role affinity for the persons axis: cosine between each brief and a description of each secondary-person role,
with the same local e5 model and query instruction as the entity pool. Writes scores-roles.json {bundle_id: {role: cosine}}.
Run from the run folder: python code/role_scores.py
"""

import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

ROLES = {
    "spouse_of": "the subject's spouse or partner is named: marriage, household, family, beneficiary, next of kin, joint application",
    "mother_of": "the subject's mother is named: parent, family record, birth, civil registry, school or minor's form",
    "father_of": "the subject's father is named: parent, family record, birth, civil registry, school or minor's form",
    "child_of": "the subject's child is named: dependant, minor, school, family insurance, custody",
    "sibling_of": "the subject's sibling is named: family, inheritance, household",
    "relative_of": "a relative of the subject is named: family, next of kin, inheritance, guardian",
    "emergency_contact_of": "an emergency contact person is named: employment file, enrolment, membership, medical form, travel",
    "physician": "the treating physician or doctor is named: prescription, medical report, hospital, clinic, health insurance",
    "lawyer": "the subject's lawyer or attorney is named: court, lawsuit, legal notice, power of attorney, contract dispute",
    "expert": "a court-appointed expert witness is named: litigation, appraisal, damage assessment, technical report for a court",
    "judge": "the judge or prosecutor handling the case is named: court decision, indictment, hearing, enforcement file",
    "hr_contact": "the responsible officer, HR contact, account manager or customer-relations agent is named with contact details: company letter, support ticket, onboarding, customer service",
    "witness": "a witness is named: incident report, accident, police statement, signature of witnesses on a contract",
    "other_customer": "another customer, applicant or participant is listed in the same document: list, roster, registration sheet, multi-party form",
    "colleague": "a colleague or co-worker of the subject is mentioned: internal memo, project note, team report",
    # [e08] role relations from the taxonomy extension
    "represented_by": "the subject grants power of attorney to a named attorney or representative (vekil): vekaletname, notary deed, authorisation to act on one's behalf, court representation",
    "authorised_signatory_of": "a named person signs for a company as its authorised signatory or officer: company letter, contract, invoice, tender, board decision",
    "owner_of_company": "a named person is the owner, founder or shareholder of the company: trade registry, partnership, sole proprietorship, business licence",
    "guarantor_of": "a named guarantor (kefil) vouches for the subject's debt or contract: loan, lease, credit application, surety",
    "heir_of": "a named heir or the deceased is mentioned: inheritance, succession certificate, estate, will, probate",
    "tenant_of": "a tenant and a landlord are both named: lease, rental contract, eviction, deposit, rent receipt",
    "witness_of": "named witnesses sign or attend a legal act: notary deed, contract signature, incident report, court hearing",
    "property_at": "a property is identified by parcel or unit number together with its address: title deed, cadastral record, condominium, real-estate sale or lease",
}
TASK = "identify which people, besides the subject, would be named inside a document of this type"
briefs = [json.loads(l) for l in open("briefs_catalogue.jsonl", encoding="utf-8")]
t0 = time.time()
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", token=os.environ.get("HF_TOKEN"))
q = model.encode([f"Instruct: {TASK.capitalize()}\nQuery: {b['text']}" for b in briefs], normalize_embeddings=True)
d = model.encode(list(ROLES.values()), normalize_embeddings=True)
sims = np.asarray(q) @ np.asarray(d).T
roles = list(ROLES)
out = {str(b["uid"]): {r: float(sims[i, j]) for j, r in enumerate(roles)} for i, b in enumerate(briefs)}
json.dump(
    {"model": "intfloat/multilingual-e5-large-instruct", "roles": ROLES, "scores": out},
    open("scores-roles.json", "w"),
    indent=0,
)
print(f"role scores for {len(briefs)} briefs × {len(roles)} roles in {round(time.time() - t0)}s -> scores-roles.json")
