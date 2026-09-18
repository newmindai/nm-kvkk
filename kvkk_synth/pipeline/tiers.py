"""Who generates which value (design §4.6).

TIER_B: nodes the writer may add with values of its own; each has a format check. A tagged tier-B span that is
not in the bundle is accepted when validate() says so; the node's canonical relation to the subject is attached
as pending until the relation judge confirms it.
TIER_C: nodes whose values come from the real geographic data (locale.py). Everything else is tier A.
"""

import re

PLACEHOLDER_RE = re.compile(
    r"(^|\s)(x|y|z|abc|xyz|örnek|ornek|lorem|n/?a|test|xxx+|\?{2,}|\.{3,})(\s|$)|firma adı|şirket adı|kurum adı|\[[^\]]*\]|<[^>]*>",
    re.I,
)
SLUR_RE = re.compile(r"\b(ibne|top|nonoş|sapık)\b", re.I)
TR = "A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû"

_ORG_KW = re.compile(
    "|".join(
        [
            r"A\.Ş\.?",
            r"A\.S\.?",
            "Ltd",
            "Şti",
            r"San\.",
            r"Tic\.",
            "Holding",
            "Vakfı",
            "Derneği",
            "Belediyesi",
            "Üniversitesi",
            "Hastanesi",
            "Bankası",
            "Sigorta",
            "Kooperatif",
            "Müdürlüğü",
            "Bakanlığı",
            "Başkanlığı",
            "Okulu",
            "Lisesi",
            "Kliniği",
            "Eczanesi",
            "Market",
            "Şirketi",
            "Grup",
            "Group",
            "Inc",
            "GmbH",
            "Limited",
            "Anonim",
            "Enerji",
            "Lojistik",
            "Teknoloji",
            "Yazılım",
            "İnşaat",
            "Tekstil",
            "Gıda",
            "Otomotiv",
            "Merkezi",
            "Danışmanlık",
            "Laboratuvar",
            "Hizmetleri",
            "Sanayi",
            "Medya",
            "Çözümleri",
            "Ajans",
            "Kurumu",
            "Ofisi",
            "Bürosu",
        ]
    ),
    re.I,
)


def _kw(*words):
    return re.compile("|".join(words), re.I)


TIER_B = {
    "nickname": (lambda v: 2 <= len(v) <= 30 and len(v.split()) <= 3, "lakap"),
    "salary_info": (
        re.compile(r"\d{1,3}(\.\d{3})+\s?(TL|₺|lira|TRY)|\d{4,6}\s?(TL|₺|lira|TRY)|\d{1,3}\.\d{3}", re.I),
        "maaş / ücret tutarı",
    ),
    "risk_report": (
        _kw("gecik", "takip", "limit", "borç", "kayıt", "risk", "kredi", "ödeme"),
        "kredi/risk raporu ifadesi",
    ),
    "workplace_location": (
        lambda v: 2 <= len(v) <= 60 and len(v.split()) <= 7 and "No:" not in v,
        "işyeri konumu (semt/plaza)",
    ),  # [e08] loosened
    "medical_condition": (
        lambda v: 3 <= len(v) <= 120 and v.lower() not in ("hastalık", "rahatsızlık"),
        "hastalık / tanı",
    ),  # [e08] 80 -> 120
    "mental_health_info": (lambda v: 3 <= len(v) <= 120, "ruh sağlığı bilgisi"),
    "medication": (lambda v: 3 <= len(v) <= 60 and v[0].isalpha(), "ilaç"),
    "medical_procedure": (lambda v: 4 <= len(v) <= 120, "tıbbi işlem / tedavi"),
    "disability_status": (
        _kw(
            "engel",
            r"%\s?\d{1,3}",
            "yüzde",
            "iş göremez",
            "iş gücü",
            "işgücü",
            "çalışma güc",
            "kısıtl",
            "kayıp",
            "kayb",
            "malul",
            "sakat",
            "özürlü",
            "rapor",
        ),
        "engellilik durumu",
    ),  # [e08] block 1: 25 refused statuses
    "ethnicity": (
        _kw(
            "kökenli",
            "asıllı",
            "kürt",
            "türk",
            "çerkes",
            "arap",
            "laz",
            "roman",
            "ermeni",
            "rum",
            "boşnak",
            "arnavut",
            "gürcü",
            "zaza",
            "etnik",
        ),
        "etnik köken",
    ),
    "political_opinion": (
        _kw(
            "parti",
            "üyesi",
            "sempatizan",
            "görüş",
            "siyasi",
            "siyas",
            "seçmen",
            "destek",
            "oy ver",
            "demokrat",
            "liberal",
            "muhafazak",
            "milliyetçi",
            "aday",
            "ideolo",
            "politik",
            "iktidar",
            "muhalefet",
            r"\bsol\b",
            r"\bsağ\b",
            "sosyal adalet",
            "şeffaflık",
        ),
        "siyasi düşünce / parti",
    ),  # [e08]
    "religious_belief": (
        _kw(
            "inanc",
            "inanç",
            "mezhep",
            "mensup",
            "sünni",
            "alevi",
            "hanefi",
            "şafi",
            "hristiyan",
            "hıristiyan",
            "müslüman",
            "yahudi",
            "musevi",
            "ortodoks",
            "katolik",
            "protestan",
            "dini",
            r"\bdin\b",
            "islam",
            "islâm",
            "hindu",
            "budist",
            "ibadet",
            "namaz",
            "oruç",
            "kilise",
            "cami",
            "sinagog",
            "cemevi",
        ),
        "dini inanç",
    ),  # [e08]
    "philosophical_belief": (
        _kw(
            "ateist",
            "deist",
            "agnostik",
            "hümanist",
            "felsef",
            "laik",
            "seküler",
            "inançsız",
            "panteist",
            "nihilist",
            "stoacı",
        ),
        "felsefi inanç",
    ),
    "attire_info": (
        _kw("başörtü", "türban", "çarşaf", "peçe", "sarık", "cübbe", "takke", "tesettür", "kıyafet", "giyim", "örtü"),
        "kılık kıyafet",
    ),
    "union_membership": (
        _kw("sendika", r"-sen\b", r"\bsen\b", r"-iş\b", "türk-iş", "disk", "hak-iş", "kamu-sen", "memur-sen", "birlik"),
        "sendika üyeliği",
    ),
    "association_foundation_membership": (
        _kw("derne", "vakf", "kulüb", r"\boda", "birliğ", "cemiyet", "federasyon", "üyesi", "gönüllü"),
        "dernek / vakıf üyeliği",
    ),
    "sexual_life_info": (
        _kw("cinsel", "eşcinsel", "heteroseksüel", "biseksüel", "lgbt", "yönelim", "partner", "ilişki"),
        "cinsel hayat bilgisi",
    ),
    "criminal_record": (
        _kw("hapis", "ceza", "hüküm", "mahkum", "mahkûm", "denetimli", "sabıka", "beraat", "suç", "adli"),
        "ceza mahkumiyeti / güvenlik tedbiri",
    ),
    "biometric_data_reference": (
        _kw(
            "parmak izi",
            "yüz tanıma",
            "iris",
            "retina",
            "biyometri",
            "FP-",
            "BIO-",
            "BT-",
            "ses izi",
            "avuç",
            r"^[A-Z]{2,4}-\d",
        ),
        "biyometrik veri referansı",
    ),
    "genetic_data_reference": (
        _kw(r"\bgen", "mutasyon", "varyant", "taşıyıc", "DNA", "kromozom", "BRCA", "MTHFR", "HLA", "genetik"),
        "genetik veri referansı",
    ),
    "signature_handwriting": (_kw("imza", "el yazısı", "paraf"), "imza / el yazısı referansı"),
    "voice_record": (_kw("ses kayd", "çağrı kayd", "sesli", "kayıt", r"\d{1,2}:\d{2}"), "ses kaydı referansı"),
    "audiovisual_record_reference": (
        _kw("kamera", "kayıt", "görüntü", "video", "fotoğraf", r"\.jpe?g", r"\.png", r"\.mp4", "IMG_", "KAM-", "CAM"),
        "görsel / işitsel kayıt referansı",
    ),
    "job_title": (
        lambda v: (
            3 <= len(v) <= 80
            and len(v.split()) <= 8
            and re.fullmatch(rf"[{TR}.\-/ ]+", v) is not None
            and v.lower() not in ("çalışan", "personel")
        ),
        "meslek / unvan",
    ),  # [e08] 6 -> 8 words
    "work_history": (
        lambda v: (
            10 <= len(v) <= 160
            and (
                len(v.split()) >= 3
                or re.search(r"(19|20)\d{2}|\byıl\b|\bay\b|süre|arasında|olarak çalış|görev|deneyim", v, re.I)
                is not None
            )
        ),
        "iş geçmişi",
    ),  # [e08] any ≥3-word description
    "performance_disciplinary": (
        _kw(
            "kınama",
            "uyarı",
            "ihtar",
            "ceza",
            "takdir",
            "ödül",
            "disiplin",
            "başarı",
            "performans",
            "görev",
            "değerlendir",
            "yeterli",
            "yetersiz",
            "prosedür",
        ),
        "disiplin / performans kaydı",
    ),  # [e08]
    "education_info": (
        _kw(
            "üniversite",
            "fakülte",
            "lise",
            "bölüm",
            "meslek",
            "yüksekokul",
            "doktora",
            "lisans",
            "mezun",
            "okul",
            "akademi",
            "enstitü",
            r"(19|20)\d{2}",
            "eğitim",
            "mühendislik",
            "öğrenim",
            "kurs",
            "sertifika",
            "diploma",
        ),
        "eğitim bilgisi",
    ),  # [e08]
    "military_service_status": (
        _kw("tecil", "muaf", "terhis", "yapıldı", "yapmadı", "yapılmadı", "bedelli", "askerlik", "yedek", "asker"),
        "askerlik durumu",
    ),  # [e08]
    "company_name": (
        lambda v: (
            (
                _ORG_KW.search(v) is not None
                or (
                    2 <= len(v.split()) <= 7
                    and len(v) <= 70
                    and all(w[0].isupper() or w in ("ve", "&") for w in v.split())
                )
            )
            and v.lower() != v
        ),
        "kurum / şirket adı",
    ),  # [e08] keyword OR a capitalised multi-word name
    "tax_office_name": (re.compile(r"vergi dairesi\s*$", re.I), "vergi dairesi adı"),
}
TIER_C = {
    "full_address",
    "street_line",
    "district",
    "city",
    "postal_code",
    "place_of_birth",
    "tracked_location",
    "registered_place",
}
MAX_WRITER_ADDED = 6


def validate(node, value, known_values, name_re):
    """(ok, reason) for a writer-added tagged value."""
    v = value.strip()
    if node not in TIER_B:
        return False, "not a tier-B label"
    if len(v) > 160 or not v:
        return False, "length"
    if PLACEHOLDER_RE.search(v):
        return False, "placeholder"
    if v.isdigit():
        return False, "digits only"
    if node == "sexual_life_info" and SLUR_RE.search(v):
        return False, "slur"
    if any(v.lower() == k.lower() or (len(v) > 3 and v.lower() in k.lower()) for k in known_values):
        return False, "collides with an offered value"
    if node != "company_name" and name_re.search(v):
        return False, "contains a person name"
    check, _ = TIER_B[node]
    ok = check(v) if callable(check) else check.search(v) is not None
    return (True, "ok") if ok else (False, "format")


CANONICAL_PRIORITY = {
    "company_name": "employer_of",
    "tax_office_name": "tax_office_of",
    "workplace_location": "workplace_of",
    "nickname": "nickname_of",
    "audiovisual_record_reference": "depicted_in",
}  # [e08] never member_of for a company (pipeline review finding 3)


def canonical_relations(rels_meta):
    """node -> (relation_id, 'head'|'tail' = side the node takes; the subject takes the other side)."""
    out = {}
    for node, rid in CANONICAL_PRIORITY.items():
        r = next((x for x in rels_meta if x["id"] == rid), None)
        if r and node in TIER_B:
            out[node] = (rid, "head" if node in r["head_types"] else "tail")
    for node in TIER_B:
        if node in out:
            continue
        for r in rels_meta:
            if node in r["head_types"] and "full_name" in r["tail_types"]:
                out[node] = (r["id"], "head")
                break
            if node in r["tail_types"] and "full_name" in r["head_types"]:
                out[node] = (r["id"], "tail")
                break
    missing = [n for n in TIER_B if n not in out]
    assert not missing, f"no person relation for {missing}"
    return out


def prompt_block(ents_meta):
    """The tier-B paragraph for the writer prompt."""
    lines = [f"- {node}: {desc}" for node, (_, desc) in TIER_B.items()]
    return (
        "You MAY additionally add personal facts about the subject of ONLY the following kinds, with realistic values "
        "of your own (real-sounding institutions, diagnoses, job titles, memberships — never placeholders like X, ABC or Örnek), "
        "when a real document of this type would contain them. Every such fact you write MUST be tagged inline exactly like the listed ones, "
        "using EXACTLY the label id on the left (ASCII, e.g. ]salary_info — never the Turkish word), and attributed clearly to the subject:\n"
        + "\n".join(lines)
        + "\nNever add personal data of any other kind (no extra names, identity or account numbers, phone numbers, e-mails, "
        "addresses, birth dates, plates or codes beyond the listed values)."
    )
