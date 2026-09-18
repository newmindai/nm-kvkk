"""Format-shaped generators for the tier-A nodes that taxonomy v2.2 leaves with a single example (design §3):
every instance in e07 was the example string (453 of 4,375 mentions). Registered into kvkk_sampler.GEN at import,
so the pool sampler, the secondary persons and the repair all draw fresh values. MRZ is person-aware (see mrz()).
"""

import random
import string


def _d(n, first_nonzero=False):
    s = "".join(random.choice(string.digits) for _ in range(n))
    return (random.choice("123456789") + s[1:]) if first_nonzero and n else s


def _alnum(n, alphabet=string.ascii_uppercase + string.digits):
    return "".join(random.choice(alphabet) for _ in range(n))


B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def gen_crypto_wallet():
    r = random.random()
    if r < 0.4:
        return random.choice("13") + "".join(random.choice(B58) for _ in range(random.randint(32, 33)))
    if r < 0.7:
        return "bc1q" + "".join(random.choice("023456789acdefghjklmnpqrstuvwxyz") for _ in range(38))
    return "0x" + "".join(random.choice("0123456789abcdef") for _ in range(40))


def gen_cookie():
    r = random.random()
    if r < 0.5:
        return f"_ga=GA1.2.{_d(9, True)}.{_d(10, True)}"
    if r < 0.75:
        return "PHPSESSID=" + _alnum(26, string.ascii_lowercase + string.digits)
    return "JSESSIONID=" + _alnum(32, "0123456789ABCDEF")


def gen_risk_report():
    return random.choice(
        [
            f"{random.randint(1, 4)} gecikmiş ödeme",
            f"son {random.choice([6, 12, 24])} ayda {random.randint(1, 3)} gecikme kaydı",
            "kanuni takip kaydı bulunmamaktadır",
            f"limit aşımı: {random.randint(1, 40)}.{_d(3)} TL",
            "yasal takipte 1 kredi kartı borcu",
            f"toplam kredi riski {random.randint(20, 900)}.{_d(3)} TL",
            "kapatılmış takip kaydı (1 adet)",
        ]
    )


def gen_land_parcel():
    ada, parsel = random.randint(1, 9999), random.randint(1, 999)
    return random.choice(
        [
            f"Ada {ada}, Parsel {parsel}",
            f"{ada} ada {parsel} parsel",
            f"Ada: {ada} Parsel: {parsel}",
            f"Pafta {random.randint(1, 60)}, Ada {ada}, Parsel {parsel}",
        ]
    )


def gen_independent_section():
    blok = random.choice("ABCDEFG")
    no = random.randint(1, 60)
    return random.choice(
        [
            f"{blok} Blok, Bağımsız Bölüm {no}",
            f"{blok} Blok Daire {no}",
            f"Bağımsız Bölüm No: {no}",
            f"{blok}-{no} numaralı bağımsız bölüm",
        ]
    )


VGEN = {
    "yupass_number": lambda: _d(9, True),
    "crypto_wallet_address": gen_crypto_wallet,
    "insurance_policy_number": lambda: random.choice(
        [f"{_d(4, True)}-{_d(6)}", f"POL-{random.randint(2019, 2026)}-{_d(6)}", _d(10, True)]
    ),
    "cookie_id": gen_cookie,
    "risk_report": gen_risk_report,
    "medula_tracking_number": lambda: f"{random.randint(1, 9)}P{_d(7)}",
    "provision_number": lambda: _d(8, True),
    "eprescription_number": lambda: _alnum(6, "ABCDEFGHJKLMNPRSTUVYZ0123456789"),
    "doctor_license_number": lambda: _d(6, True),
    "student_number": lambda: f"{random.randint(2014, 2025)}{_d(6)}",
    "investigation_number": lambda: f"{random.randint(2019, 2026)}/{random.randint(100, 99999)}",
    "notification_barcode": lambda: _d(13, True),
    "notary_record_number": lambda: str(random.randint(100, 99999)).zfill(random.choice([4, 5])),
    "bar_registry_number": lambda: str(random.randint(100, 99999)),
    "expert_registry_number": lambda: str(random.randint(100, 99999)),
    "judge_prosecutor_registry_number": lambda: str(random.randint(10000, 999999)),
    "enforcement_file_number": lambda: f"{random.randint(2018, 2026)}/{random.randint(100, 99999)} E.",
    "loyalty_card_number": lambda: _d(random.choice([10, 12, 16]), True),
    "cargo_tracking_number": lambda: _d(random.choice([12, 13]), True),
    "civil_registry_volume_number": lambda: str(random.randint(10, 9999)),
    "family_order_number": lambda: str(random.randint(1, 99999)).zfill(5),
    "individual_order_number": lambda: str(random.randint(1, 999)).zfill(3),
    "land_parcel_number": gen_land_parcel,
    "independent_section_number": gen_independent_section,
    "drivers_license_number": lambda: random.choice(
        [_d(6, True), f"{random.choice('ABCDEFGHJKLMNPRSTUVYZ')}{_d(2)}{random.choice('ABCDEFGHJKLMNPRSTUVYZ')}{_d(5)}"]
    ),
    "social_security_number": lambda: _d(13, True),
    "bank_account_number": lambda: _d(random.choice([8, 9, 10]), True),
    "health_report_id": lambda: f"{random.randint(2019, 2026)}/{_d(random.choice([4, 5, 6]), True)}",
    "customer_number": lambda: _d(random.choice([7, 8, 9, 10]), True),
    "subscription_number": lambda: _d(random.choice([7, 8, 10]), True),
    "employee_id": lambda: _d(random.choice([4, 5, 6]), True),
    "card_cvv": lambda: _d(3),
}

TR2ASCII = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")


def _mrz_check(s):
    w = [7, 3, 1]
    tot = 0
    for i, ch in enumerate(s):
        v = int(ch) if ch.isdigit() else (0 if ch == "<" else ord(ch) - 55)
        tot += v * w[i % 3]
    return str(tot % 10)


def mrz(surname, given, passport_no, dob_ddmmyyyy, sex, nationality="TUR", expiry_year=None):
    """ICAO 9303 TD3 (two lines of 44) consistent with the person; dob 'dd.mm.yyyy'."""
    sn = surname.upper().translate(TR2ASCII).replace(" ", "<")
    gn = given.upper().translate(TR2ASCII).replace(" ", "<")
    l1 = ("P<" + nationality + sn + "<<" + gn)[:44].ljust(44, "<")
    d, m, y = dob_ddmmyyyy.split(".")
    dob = y[2:] + m + d
    exp = (
        f"{(expiry_year or random.randint(2027, 2034)) % 100:02d}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
    )
    pn = passport_no.upper().translate(TR2ASCII).replace(" ", "")[:9].ljust(9, "<")
    l2 = (
        pn
        + _mrz_check(pn)
        + nationality
        + dob
        + _mrz_check(dob)
        + ("F" if sex == "F" else "M")
        + exp
        + _mrz_check(exp)
        + "<" * 14
        + "<"
    )
    l2 = l2 + _mrz_check(l2[0:10] + l2[13:20] + l2[21:43])
    return l1 + "\n" + l2


def install(ks):
    """Register into kvkk_sampler.GEN (does not override recipes tr_heuristics already provides)."""
    for k, f in VGEN.items():
        if k not in ks.GEN or k in (
            "drivers_license_number",
            "social_security_number",
            "bank_account_number",
            "health_report_id",
            "customer_number",
            "subscription_number",
            "employee_id",
            "card_cvv",
        ):
            ks.GEN[k] = f
    return sorted(VGEN)
