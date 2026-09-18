"""One consistent place per bundle, drawn from the real NVİ-sourced hierarchy (geo/*.parquet):
city → ilçe → mahalle (weighted by street count, i.e. urban) → street, postal code from the mahalle.
Gives every tier-C value: full_address, street_line, district, city, postal_code, place_of_birth,
registered_place, tracked_location — all agreeing with each other. Surface shapes vary per bundle.
"""

import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "..", "geo")  # [pkg] kvkk_synth/geo, built by download_seeds.py
sys.path.insert(0, os.path.join(HERE, "..", "taxonomy"))  # [pkg]
try:
    from tr_heuristics import PLATE_PROV  # TÜİK-weighted province codes

    PLATE_W = {f"{p:02d}": w for p, w in PLATE_PROV}
except Exception:
    PLATE_W = {}

_cities = pd.read_parquet(os.path.join(GEO, "cities.parquet"))
_towns = pd.read_parquet(os.path.join(GEO, "towns.parquet"))
_nb = pd.read_parquet(os.path.join(GEO, "neighbourhoods.parquet"))
_st = pd.read_parquet(os.path.join(GEO, "streets.parquet"))
_st = _st[_st.kind.isin(["sokak", "cadde", "bulvar"])]
_nb_urban = _nb[(~_nb.rural) & (_nb.n_streets >= 5)]
_towns_by_city = {c: g for c, g in _towns.groupby("city_id")}
_nb_by_town = {t: g for t, g in _nb_urban.groupby("town_id")}
_st_by_nb = {n: g for n, g in _st.groupby("neighbourhood_id")}
CITY_ROWS = _cities.to_dict("records")
CITY_WEIGHTS = [PLATE_W.get(c["plate"], 1) for c in CITY_ROWS]

ABBR = {"sokak": ["Sokak", "Sk.", "Sok."], "cadde": ["Caddesi", "Cad.", "Cd."], "bulvar": ["Bulvarı", "Blv.", "Bulv."]}


def _street_display(row, abbr):
    base = row["name"]
    kind = row["kind"]
    words = base.split()
    stem = (
        " ".join(words[:-1])
        if words and words[-1].lower() in ("sokak", "sokağı", "caddesi", "cadde", "bulvarı", "bulvar")
        else base
    )
    return f"{stem} {abbr[kind]}"


class Locale:
    def __init__(self, rng=random):
        for _ in range(50):
            city = rng.choices(CITY_ROWS, CITY_WEIGHTS)[0]
            towns = _towns_by_city.get(city["id"])
            if towns is None:
                continue
            town = towns.sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
            nbs = _nb_by_town.get(int(town["id"]))
            if nbs is None or nbs.empty:
                continue
            nb = nbs.sample(1, weights=nbs.n_streets, random_state=rng.randint(0, 1 << 30)).iloc[0]
            sts = _st_by_nb.get(int(nb["id"]))
            if sts is None or sts.empty:
                continue
            street = sts.sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
            break
        else:
            raise RuntimeError("no urban locale found")
        self.city, self.town, self.nb, self.street = city["name"], str(town["name"]), str(nb["name"]), street
        self.plate = str(city["plate"])
        self.zip = str(nb["zip"])
        self.no, self.daire, self.kat = rng.randint(1, 120), rng.randint(1, 24), rng.randint(1, 8)
        self.abbr = {k: rng.choice(v) for k, v in ABBR.items()}
        self.mah = self.nb if rng.random() < 0.7 else self.nb.replace(" Mah.", " Mahallesi")
        self.shape = rng.choices([1, 2, 3, 4, 5], [35, 25, 15, 10, 15])[0]
        self.rng = rng

    @property
    def street_text(self):
        return _street_display(self.street, self.abbr)

    def street_line(self):
        s = self.street_text
        return {
            1: f"{self.mah} {s} No:{self.no} D:{self.daire}",
            2: f"{self.mah} {s} No:{self.no} Daire:{self.daire}",
            3: f"{s} No:{self.no}/{self.daire}, {self.mah}",
            4: f"{self.mah}, {s}, No: {self.no}, Kat: {self.kat}, Daire: {self.daire}",
            5: f"{self.mah} {s} {self.no}/{self.daire}",
        }[self.shape]

    def full_address(self, with_zip):
        sl = self.street_line()
        if self.shape == 3:
            return f"{sl}, {self.town}, {self.city}" + (f" {self.zip}" if with_zip else "")
        if self.shape == 4:
            return f"{sl}, {self.zip + ' ' if with_zip else ''}{self.town} / {self.city}"
        if self.shape == 5:
            return f"{sl} {self.town} {self.city}"
        return f"{sl} {self.zip + ' ' if with_zip else ''}{self.town}/{self.city}"

    def place_of_birth(self):
        if self.rng.random() < 0.4:
            return self.city
        return self.rng.choices(CITY_ROWS, CITY_WEIGHTS)[0]["name"]

    def registered_place(self):
        r = self.rng.random()
        if r < 0.5:
            return f"{self.city}/{self.town}"
        if r < 0.8:
            return f"{self.city}/{self.town}/{self.nb.replace(' Mah.', '')}"
        return f"{self.city} {self.town}"

    def tracked_location(self):
        return self.rng.choice(
            [
                f"{self.city} {self.town}",
                f"{self.town}, {self.city}",
                f"{self.nb}, {self.town}",
                f"{self.city}/{self.town}",
            ]
        )


if __name__ == "__main__":
    random.seed(3)
    for _ in range(6):
        L = Locale()
        print(L.full_address(with_zip=True), "|", L.registered_place(), "|", L.place_of_birth())
