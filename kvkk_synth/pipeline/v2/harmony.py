"""Turkish suffix-harmony post-fix for apostrophe suffixes attached to tagged values.

The generator often writes the suffix before "seeing" the value (Yılmaz'in → 'ın, Zeynep'de →
'te). Suffixes sit outside spans, and the fix only swaps vowels or d↔t, so string length and
every character offset stay unchanged.
"""

import re

BACK, ROUND = set("aıou"), set("oöuü")
VOWELS = set("aeıioöuü")
VOICELESS = set("pçtkfsşh")
# last spoken digit -> (vowel of its Turkish name, name ends voiceless)
DIGIT = {
    "1": ("i", False),
    "2": ("i", False),
    "3": ("ü", True),
    "4": ("ö", True),
    "5": ("e", True),
    "6": ("ı", False),
    "7": ("i", False),
    "8": ("i", False),
    "9": ("u", False),
}
TENS = {
    "10": ("o", False),
    "20": ("i", False),
    "30": ("u", False),
    "40": ("ı", True),
    "50": ("i", False),
    "60": ("ı", True),
    "70": ("i", True),
    "80": ("e", False),
    "90": ("a", False),
}
LETTER_NAME = {"A": "a", "I": "ı", "İ": "i", "O": "o", "Ö": "ö", "U": "u", "Ü": "ü", "E": "e"}  # others: "…e"
SUFFIX_RE = re.compile(r"['’]([a-zçğıöşü]{1,5})(?![a-zçğıöşü])")


def phonology(value):
    """(last vowel, ends voiceless) of how the value is read aloud; None if unknown."""
    v = value.rstrip(" .)]\"'’")
    if not v:
        return None
    last = v[-1]
    if last.isdigit():
        num = re.search(r"\d+$", v).group(0)
        if num.endswith("000"):
            return "i", False  # bin
        if num.endswith("00"):
            return "ü", False  # yüz
        if num.endswith("0"):
            return TENS.get(num[-2:], ("a", False))  # on/yirmi/…
        return DIGIT[last]
    token = re.split(r"[ .]", v)[-1]
    if (
        token
        and token == token.upper()
        and last.isalpha()
        and (len(token) <= 3 or not any(c in VOWELS for c in token.lower()))
    ):
        return LETTER_NAME.get(last, "e"), False  # abbreviation (TL, A.Ş., TCKN) read letter by letter
    lower = v.lower()
    for ch in reversed(lower):
        if ch in VOWELS:
            return ch, lower[-1] in VOICELESS
    return None


def correct(suffix, vowel, voiceless):
    back, rnd = vowel in BACK, vowel in ROUND
    two = "a" if back else "e"
    four = ("u" if rnd else "ı") if back else ("ü" if rnd else "i")
    d = "t" if voiceless else "d"
    rules = [
        (r"^[dt]([ae])(n|ki)?$", lambda m: d + two + (m.group(2) or "")),  # 'da 'den 'daki
        (r"^n[dt]([ae])(n|ki)?$", lambda m: "nd" + two + (m.group(2) or "")),  # 'nda 'nden (after possessive)
        (r"^(n?)[ıiuü]n$", lambda m: m.group(1) + four + "n"),  # 'ın 'nin
        (r"^(y?)[ae]$", lambda m: m.group(1) + two),  # 'a 'ye
        (r"^(y?)[ıiuü]$", lambda m: m.group(1) + four),  # 'ı 'yi
        (r"^n[ıiuü]$", lambda m: "n" + four),  # 'nı
        (r"^[dt][ıiuü]r$", lambda m: d + four + "r"),  # 'dır 'tir
        (r"^(y?)l[ae]$", lambda m: m.group(1) + "l" + two),  # 'la 'yle
        (r"^l[ae]r$", lambda m: "l" + two + "r"),  # 'lar
        (r"^[dt][ae]n[ıiuü]$", lambda m: d + two + "n" + four),  # 'danı (rare)
    ]
    for pat, fn in rules:
        m = re.match(pat, suffix)
        if m:
            return fn(m)
    return suffix


def fix_text(text, spans):
    """Rewrite wrong apostrophe suffixes right after each span. Returns (text, n_fixes)."""
    out, n = list(text), 0
    for s in spans:
        end = s["end"]
        m = SUFFIX_RE.match(text, end)
        if not m:
            continue
        ph = phonology(s["span"])
        if not ph:
            continue
        new = correct(m.group(1), *ph)
        if new != m.group(1) and len(new) == len(m.group(1)):
            out[m.start(1) : m.end(1)] = list(new)
            n += 1
    return "".join(out), n
