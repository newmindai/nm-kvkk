#!/usr/bin/env python3
"""
tr_heuristics.py — Arastirma raporuna dayali Turkce PII uretim heuristikleri.
Her deger [SABIT | AGIRLIKLI-SECIM | RASTGELE | HESAPLANAN] ayrisimiyla uretilir.
kvkk_sampler.py bu modulu bulursa GEN tablosunu HGEN ile gunceller.
Kaynak: HEURISTICS.md (TCMB banka kodlari, BTK operator paylari, TUIK ad/plaka
agirliklari, MOD-97 / MERNIS / VKN / Luhn algoritmalari).
"""

import random


def W(pairs):
    vals, wts = zip(*pairs)
    return random.choices(vals, weights=wts, k=1)[0]


def _d(n):
    return "".join(str(random.randint(0, 9)) for _ in range(n))


# ---------------- kod tablolari (agirlikli) ----------------
BANK_CODES = [
    ("0010", 18),
    ("0064", 11),
    ("0062", 10),
    ("0046", 9),
    ("0067", 9),
    ("0012", 8),
    ("0015", 8),
    ("0111", 6),
    ("0134", 5),
    ("0032", 4),
    ("0205", 4),
    ("0206", 2),
    ("0203", 2),
    ("0099", 2),
    ("0103", 1),
    ("0059", 1),
    ("0146", 1),
]  # TCMB EFT
SWIFTS = [
    "TCZBTR2A",
    "ISBKTRIS",
    "TGBATRIS",
    "AKBKTRIS",
    "YAPITRIS",
    "TVBATR2A",
    "TRHBTR2A",
    "FNNBTRIS",
    "DENITRIS",
    "TEBUTRIS",
    "KTEFTRIS",
    "AFKBTRIS",
    "INGBTRIS",
]
CARD_BINS = [
    ("979204", 10),
    ("454360", 9),
    ("540668", 9),
    ("435508", 8),
    ("552608", 8),
    ("415565", 6),
    ("557829", 6),
    ("402940", 5),
    ("543771", 4),
    ("222110", 3),
]  # TROY+Visa+MC
MOBILE_PREFIX = (
    [(p, 39) for p in range(530, 540)]
    + [(p, 31) for p in list(range(501, 510)) + list(range(551, 560))]
    + [(p, 29) for p in range(540, 550)]
)  # Turkcell/TT/Vodafone paylari (BTK 2025)
AREA_CODES = [
    (212, 20),
    (216, 16),
    (312, 12),
    (232, 9),
    (224, 6),
    (242, 6),
    (322, 5),
    (332, 4),
    (342, 4),
    (262, 3),
    (462, 2),
    (482, 1),
]
PLATE_PROV = [(34, 60), (6, 29), (35, 20), (16, 12), (7, 12), (1, 8), (42, 8), (41, 6), (38, 5), (27, 5)] + [
    (p, 1) for p in range(1, 82) if p not in (34, 6, 35, 16, 7, 1, 42, 41, 38, 27)
]
PLATE_LETTERS = "ABCDEFGHIJKLMNOPRSTUVYZ"  # Q,W,X ve TR ozel harfleri yok
PLATE_FORMATS = [(1, 4), (2, 3), (2, 4), (3, 2), (3, 3)]
VIN_WMI = ["NM0", "NLE", "NMC", "VF1", "WVW", "WDB", "WBA", "VSS"]
VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # I,O,Q haric
IMEI_TACS = ["35325010", "35692803", "35404906", "35152420", "86723605"]

FIRST_M = [
    ("Mehmet", 127),
    ("Mustafa", 106),
    ("Ahmet", 86),
    ("Ali", 80),
    ("Hüseyin", 70),
    ("Hasan", 65),
    ("İbrahim", 60),
    ("Murat", 55),
    ("İsmail", 52),
    ("Yusuf", 50),
    ("Emre", 40),
    ("Ömer", 38),
    ("Osman", 35),
    ("Fatih", 33),
    ("Hakan", 30),
    ("Baran", 5),
    ("Serhat", 4),
    ("Welat", 1),
]
FIRST_F = [
    ("Fatma", 115),
    ("Ayşe", 110),
    ("Emine", 100),
    ("Hatice", 90),
    ("Zeynep", 80),
    ("Elif", 70),
    ("Meryem", 60),
    ("Merve", 50),
    ("Esra", 45),
    ("Zehra", 45),
    ("Selin", 20),
    ("Deniz", 15),
    ("Dilan", 5),
    ("Rojin", 3),
    ("Berfin", 3),
]
SURNAMES = [
    ("Yılmaz", 118),
    ("Kaya", 90),
    ("Demir", 88),
    ("Çelik", 72),
    ("Yıldız", 70),
    ("Şahin", 69),
    ("Yıldırım", 67),
    ("Öztürk", 59),
    ("Aydın", 58),
    ("Özdemir", 56),
    ("Arslan", 50),
    ("Doğan", 48),
    ("Kılıç", 45),
    ("Çetin", 42),
    ("Kara", 40),
    ("Koç", 38),
    ("Kurt", 37),
    ("Polat", 35),
    ("Şimşek", 34),
]
DOUBLE_F = ["Nur", "Sultan", "Naz"]
DOUBLE_M = ["Ali", "Can", "Efe"]
TITLES = [
    ("Dr.", 4),
    ("Av.", 3),
    ("Prof. Dr.", 1),
    ("Doç. Dr.", 1),
    ("Uzm. Dr.", 1),
    ("Op. Dr.", 1),
    ("Ecz.", 1),
    ("Dt.", 1),
    ("Müh.", 2),
]
MAH = [
    ("Cumhuriyet", 30),
    ("Yeni", 28),
    ("Fatih", 26),
    ("Atatürk", 24),
    ("Yıldıztepe", 5),
    ("Bahçelievler", 8),
    ("Esentepe", 6),
    ("Çamlık", 5),
]
SOK = [
    ("Atatürk Cad.", 36),
    ("Cumhuriyet Cad.", 30),
    ("Fatih Sok.", 14),
    ("Gül Sok.", 13),
    ("Okul Sok.", 12),
    ("Karanfil Sok.", 12),
    ("Lale Sok.", 12),
    ("Menekşe Sok.", 12),
    ("İnönü Cad.", 11),
    ("İstiklal Cad.", 11),
]
CITY_W = [
    ("İstanbul", 60),
    ("Ankara", 29),
    ("İzmir", 20),
    ("Bursa", 12),
    ("Antalya", 12),
    ("Adana", 8),
    ("Konya", 8),
    ("Kocaeli", 6),
    ("Kayseri", 5),
    ("Gaziantep", 5),
    ("Trabzon", 3),
    ("Sivas", 2),
    ("Mardin", 2),
]
DIST_W = [
    ("Kadıköy", 10),
    ("Esenyurt", 10),
    ("Çankaya", 9),
    ("Bornova", 6),
    ("Osmangazi", 5),
    ("Muratpaşa", 5),
    ("Selçuklu", 4),
    ("Nilüfer", 4),
    ("Ortahisar", 2),
]
EMAIL_DOM = [
    ("gmail.com", 50),
    ("hotmail.com", 25),
    ("outlook.com", 10),
    ("yandex.com.tr", 8),
    ("yahoo.com", 5),
    ("mynet.com", 2),
]
FINDEKS_BANDS = [((1, 969), 10), ((970, 1149), 15), ((1150, 1469), 30), ((1470, 1719), 28), ((1720, 1900), 17)]


def ascii_tr(s):
    return (
        s.lower()
        .replace("ı", "i")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ğ", "g")
        .replace("â", "a")
    )


# ---------------- checksum algoritmalari ----------------
def luhn_check(digits):
    s = 0
    for i, x in enumerate(reversed(digits)):
        x = x * 2 if i % 2 == 0 else x
        s += x - 9 if x > 9 else x
    return (10 - s % 10) % 10


def gen_tckn():  # MERNIS 2 adim
    d = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
    d10 = ((sum(d[0::2]) * 7) - sum(d[1::2])) % 10
    d11 = (sum(d) + d10) % 10
    return "".join(map(str, d)) + str(d10) + str(d11)


def gen_vkn():  # GIB kontrol hanesi
    d = [random.randint(0, 9) for _ in range(9)]
    total = 0
    for i, digit in enumerate(d):
        tmp = (digit + (9 - i)) % 10
        v = (tmp * (2 ** (9 - i))) % 9
        if tmp != 0 and v == 0:
            v = 9
        total += v
    check = (10 - (total % 10)) % 10
    return "".join(map(str, d)) + str(check)


def gen_iban(grouped=None):  # MOD-97, gercek banka kodu agirlikli
    bank5 = "0" + W(BANK_CODES)
    bban = bank5 + "0" + _d(16)
    check = 98 - (int(bban + "292700") % 97)
    body = f"TR{check:02d}{bban}"
    if grouped is None:
        grouped = random.random() < 0.7
    return " ".join(body[i : i + 4] for i in range(0, 26, 4)) if grouped else body


def gen_card():
    bin6 = W(CARD_BINS)
    d = [int(c) for c in bin6] + [random.randint(0, 9) for _ in range(9)]
    d.append(luhn_check(d))
    s = "".join(map(str, d))
    return " ".join(s[i : i + 4] for i in range(0, 16, 4)) if random.random() < 0.7 else s


def gen_phone():
    pfx = W(MOBILE_PREFIX)
    rest = _d(7)
    style = random.random()
    if style < 0.35:
        return f"+90 {pfx} {rest[:3]} {rest[3:5]} {rest[5:]}"
    if style < 0.65:
        return f"0{pfx} {rest[:3]} {rest[3:5]} {rest[5:]}"
    if style < 0.85:
        return f"0{pfx}{rest}"
    return f"({pfx}) {rest[:3]} {rest[3:5]} {rest[5:]}"


def gen_landline():
    ac = W(AREA_CODES)
    return f"0{ac} {_d(3)} {_d(2)} {_d(2)}"


def gen_plate():
    prov = W(PLATE_PROV)
    nl, nd = random.choice(PLATE_FORMATS)
    letters = "".join(random.choice(PLATE_LETTERS) for _ in range(nl))
    p = f"{prov:02d} {letters} {_d(nd)}"
    return p.replace(" ", "").lower() if random.random() < 0.12 else p


def gen_postal():
    return f"{W(PLATE_PROV):02d}{_d(3)}"  # ilk 2 hane = il plaka kodu


def gen_vin():
    body = "".join(random.choice(VIN_CHARS) for _ in range(14))
    return random.choice(VIN_WMI) + body


def gen_imei():
    d = [int(c) for c in random.choice(IMEI_TACS)] + [random.randint(0, 9) for _ in range(6)]
    return "".join(map(str, d)) + str(luhn_check(d))


def gen_mersis():
    return "0" + gen_vkn() + random.choice(["00015", "00016", "00017", "00019"])


def gen_uets():
    body = str(W([(1, 80), (2, 10), (3, 10)])) + _d(14)  # ilk hane kisi tipi
    return f"{body[:5]}-{body[5:10]}-{body[10:]}"


def gen_first(gender=None):
    if gender is None:
        gender = random.random() < 0.5
    name = W(FIRST_F if gender else FIRST_M)
    if random.random() < 0.15:
        name += " " + random.choice(DOUBLE_F if gender else DOUBLE_M)
    return name


def gen_full_name():
    n = f"{gen_first()} {W(SURNAMES)}"
    if random.random() < 0.15:
        n = f"{W(TITLES)} {n}"
    if random.random() < 0.10:
        n = n.lower()
    return n


def gen_street():
    return f"{W(MAH)} Mah. {W(SOK)} No:{random.randint(1, 120)}" + (
        f" D:{random.randint(1, 30)}" if random.random() < 0.6 else ""
    )


def gen_address():
    return f"{gen_street()} {W(DIST_W)}/{W(CITY_W)}"


def gen_email():
    f, l = ascii_tr(gen_first().split()[0]), ascii_tr(W(SURNAMES))
    sep = random.choice([".", "_", ""])
    tail = str(random.randint(1, 99)) if random.random() < 0.4 else ""
    return f"{f}{sep}{l}{tail}@{W(EMAIL_DOM)}"


def gen_findeks():
    lo, hi = W(FINDEKS_BANDS)
    return str(random.randint(lo, hi))


def gen_dob():  # ortanca yas ~35 etrafinda
    year = int(random.gauss(1990, 14))
    year = max(1940, min(2007, year))
    return f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{year}"


# ---------------- dogrulayicilar (selftest icin) ----------------
def valid_tckn(s):
    if len(s) != 11 or not s.isdigit() or s[0] == "0":
        return False
    d = [int(c) for c in s]
    return d[9] == ((sum(d[0:9:2]) * 7) - sum(d[1:9:2])) % 10 and d[10] == sum(d[:10]) % 10


def valid_vkn(s):
    if len(s) != 10 or not s.isdigit():
        return False
    return gen_check_vkn(s[:9]) == int(s[9])


def gen_check_vkn(first9):
    total = 0
    for i, ch in enumerate(first9):
        tmp = (int(ch) + (9 - i)) % 10
        v = (tmp * (2 ** (9 - i))) % 9
        if tmp != 0 and v == 0:
            v = 9
        total += v
    return (10 - (total % 10)) % 10


def valid_iban(s):
    s = s.replace(" ", "").upper()
    if len(s) != 26 or not s.startswith("TR"):
        return False
    return int(s[4:] + "2927" + s[2:4]) % 97 == 1


def valid_luhn(s):
    d = [int(c) for c in s.replace(" ", "")]
    return luhn_check(d[:-1]) == d[-1]


# ---------------- GEN gecersiz kilma tablosu ----------------
HGEN = {
    "full_name": gen_full_name,
    "first_name": lambda: gen_first().split()[0],
    "last_name": lambda: W(SURNAMES),
    "maiden_name": lambda: W(SURNAMES),
    "mothers_maiden_name": lambda: W(SURNAMES),
    "national_id_number": gen_tckn,
    "tax_id_number": gen_vkn,
    "iban": gen_iban,
    "card_number": gen_card,
    "phone_number": gen_phone,
    "fax_number": gen_landline,
    "postal_code": gen_postal,
    "license_plate": gen_plate,
    "vin_chassis_number": gen_vin,
    "device_id": gen_imei,
    "mersis_number": gen_mersis,
    "company_tax_number": gen_vkn,
    "uets_address": gen_uets,
    "credit_score": gen_findeks,
    "swift_bic": lambda: random.choice(SWIFTS),
    "email_address": gen_email,
    "street_line": gen_street,
    "full_address": gen_address,
    "city": lambda: W(CITY_W),
    "district": lambda: W(DIST_W),
    "date_of_birth": gen_dob,
    "place_of_birth": lambda: W(CITY_W),
    "sim_identifier": lambda: "286" + random.choice(["01", "02", "03"]) + _d(10),
    "advertising_id": lambda: "-".join([_hex(8), _hex(4), "4" + _hex(3), random.choice("89ab") + _hex(3), _hex(12)]),
}


def _hex(n):
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def selftest(n=300):
    for _ in range(n):
        assert valid_tckn(gen_tckn())
        assert valid_vkn(gen_vkn())
        assert valid_iban(gen_iban())
        assert valid_luhn(gen_card())
        assert valid_luhn(gen_imei())
        assert gen_postal()[:2].isdigit() and 1 <= int(gen_postal()[:2]) <= 81
    return True


if __name__ == "__main__":
    print("selftest:", "OK" if selftest() else "FAIL")
    for f in [
        gen_full_name,
        gen_iban,
        gen_card,
        gen_phone,
        gen_plate,
        gen_vkn,
        gen_mersis,
        gen_uets,
        gen_email,
        gen_findeks,
    ]:
        print(" ", f())
