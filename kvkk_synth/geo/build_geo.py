#!/usr/bin/env python3
"""Turn the NVİ-sourced address dumps (geo/raw/*.sql, MIT, github.com/berkanumutlu/php-turkiye-il-ilce-adres,
source adres.nvi.gov.tr) into compact parquet tables the sampler can load in a second.

  geo/cities.parquet          id, name, plate
  geo/towns.parquet           id, city_id, name                       (ilçe)
  geo/neighbourhoods.parquet  id, city_id, town_id, district_id, name_raw, name, zip, rural, n_streets   (mahalle / köy)
  geo/streets.parquet         id, neighbourhood_id, name_raw, name, kind                                 (sokak / cadde / bulvar / küme evleri / other)
  geo/README.md               provenance + counts
Run: python geo/build_geo.py   (from kvkk_synth/)
"""

import os
import re
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def parse_tuple(line):
    """Parse one MySQL VALUES tuple `( ... ),` into a list of python values (strings raw, NULL -> None)."""
    vals, i, n = [], 1, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c == "'":
            j = i + 1
            buf = []
            while j < n:
                ch = line[j]
                if ch == "\\":
                    buf.append(line[j + 1])
                    j += 2
                    continue
                if ch == "'":
                    if j + 1 < n and line[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(ch)
                j += 1
            vals.append("".join(buf))
            i = j + 1
        elif line.startswith("NULL", i):
            vals.append(None)
            i += 4
        elif line.startswith("0x", i):
            j = i + 2
            while j < n and line[j] in "0123456789abcdefABCDEF":
                j += 1
            vals.append("<bin>")
            i = j
        else:
            j = i
            while j < n and line[j] not in ",)":
                j += 1
            tok = line[i:j].strip()
            vals.append(tok)
            i = j
        # skip to next value
        while i < n and line[i] not in ",)":
            i += 1
        if i < n and line[i] == ")":
            break
        i += 1
    return vals


def rows(fname, ncols):
    out = []
    with open(os.path.join(RAW, fname), encoding="utf-8") as f:
        for line in f:
            if line.startswith("("):
                v = parse_tuple(line.rstrip("\n").rstrip(";").rstrip(","))
                if len(v) >= ncols:
                    out.append(v[:ncols])
    return out


def up_tr(ch):
    return {"i": "İ", "ı": "I"}.get(ch, ch.upper())


def low_tr(s):
    return "".join({"I": "ı", "İ": "i"}.get(ch, ch.lower()) for ch in s)


def title_tr(s):
    return " ".join(up_tr(w[:1]) + low_tr(w[1:]) if w else w for w in s.split())


ABBR = {
    "mah": "Mah.",
    "mahallesi": "Mahallesi",
    "sokak": "Sokak",
    "sokağı": "Sokağı",
    "caddesi": "Caddesi",
    "cadde": "Cadde",
    "bulvarı": "Bulvarı",
    "bulvar": "Bulvar",
    "küme": "Küme",
    "evleri": "Evleri",
    "köyü": "Köyü",
    "mevkii": "Mevkii",
    "sitesi": "Sitesi",
}


def clean_nbhd(name):
    base = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()  # drop "(dedeli köyü)"
    rural = bool(re.search(r"köy|kume|küme|belde|mezra|yayla", name, re.I)) and "mah" not in base.lower().split()[-1:]
    base = re.sub(r"\s+mah$", " Mah.", base, flags=re.I)
    return title_tr(base).replace(" Mah.", " Mah."), rural


def clean_street(name):
    n = name.strip()
    kind = "other"
    low = n.lower()
    if low.endswith(("sokak", "sokağı", "sk", "sk.")):
        kind = "sokak"
    elif low.endswith(("caddesi", "cadde", "cad", "cad.")):
        kind = "cadde"
    elif low.endswith(("bulvarı", "bulvar", "blv", "blv.")):
        kind = "bulvar"
    elif "küme evleri" in low or "kume evleri" in low:
        kind = "kume"
    elif low.startswith("("):
        kind = "village"
    return title_tr(n), kind


t0 = time.time()
cities = pd.DataFrame([{"id": int(v[0]), "name": v[1], "plate": v[2]} for v in rows("cities.sql", 3)])
towns = pd.DataFrame([{"id": int(v[0]), "city_id": int(v[1]), "name": v[2]} for v in rows("towns.sql", 3)])
nb = []
for v in rows("neighbourhoods.sql", 7):
    name, rural = clean_nbhd(v[4])
    nb.append(
        {
            "id": int(v[0]),
            "city_id": int(v[1]),
            "town_id": int(v[2]),
            "district_id": int(v[3]),
            "name_raw": v[4],
            "name": name,
            "zip": str(v[5]).zfill(5),
            "rural": rural,
            "n_streets": v[6].count('"id"'),
        }
    )
neighbourhoods = pd.DataFrame(nb)
st = []
for v in rows("streets.sql", 6):
    name, kind = clean_street(v[5])
    st.append({"id": int(v[0]), "neighbourhood_id": int(v[4]), "name_raw": v[5], "name": name, "kind": kind})
streets = pd.DataFrame(st)
for name, df in (("cities", cities), ("towns", towns), ("neighbourhoods", neighbourhoods), ("streets", streets)):
    df.to_parquet(os.path.join(HERE, f"{name}.parquet"), index=False)
kinds = streets["kind"].value_counts().to_dict()
readme = f"""# geo — real Turkish address hierarchy for the sampler

Source: `raw/*.sql` from https://github.com/berkanumutlu/php-turkiye-il-ilce-adres (MIT, © 2024 Berkan Ümütlü),
whose stated data source is the civil registry address query https://adres.nvi.gov.tr/VatandasIslemleri/AdresSorgu
(dump dated 2024-04/05). Cross-check list: `raw/turkiye_mahalleleri.cc0.csv` (Hugging Face `ceyyyh/turkiye_mahalleleri`, CC0,
30,778 neighbourhood rows). Raw dumps are gitignored; the parquet tables are built by `build_geo.py` in {time.time() - t0:.0f} s.

| table | rows | note |
|---|---|---|
| cities | {len(cities)} | il, with plate code |
| towns | {len(towns)} | ilçe |
| neighbourhoods | {len(neighbourhoods)} | mahalle / köy with 5-digit postal code; {int(neighbourhoods.rural.sum())} flagged rural; street count per neighbourhood |
| streets | {len(streets)} | kinds: {kinds} |

Names are Title-cased with Turkish rules; village qualifiers in parentheses are dropped from the neighbourhood name
(kept in `name_raw`). The sampler draws city → ilçe → mahalle (weighted by street count, i.e. urban) → street, and
takes the postal code from the mahalle, so every address component agrees.
"""
open(os.path.join(HERE, "README.md"), "w", encoding="utf-8").write(readme)
print(readme)
