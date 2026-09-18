"""Scenario layer: which relation groups may co-occur in one document, and which Turkish
document genres can host each scenario. This is what makes bundles plausible by
construction instead of drawing relations from all 75 at random."""

# primary group -> compatible neighbour groups (in order of preference)
COMPAT = {
    "health": ["identity_documents", "civil_attributes", "contact", "kinship"],
    "financial": ["identity_documents", "contact", "organization", "digital"],
    "employment": ["identity_documents", "contact", "organization", "financial", "special_attribution"],
    "legal_customer": ["identity_documents", "contact", "organization", "financial"],
    "identity_documents": ["civil_attributes", "contact", "kinship", "name_variants", "registry"],  # [e01]
    "kinship": ["civil_attributes", "identity_documents", "contact", "health", "registry"],  # [e01]
    "civil_attributes": ["identity_documents", "kinship", "contact", "employment"],
    "name_variants": ["identity_documents", "kinship", "digital", "contact"],
    "contact": ["identity_documents", "location", "legal_customer", "financial"],
    "location": ["contact", "vehicle", "digital", "employment"],
    "digital": ["contact", "identity_documents", "financial", "legal_customer"],
    "vehicle": ["identity_documents", "contact", "financial", "legal_customer"],
    "special_attribution": ["health", "employment", "legal_customer", "identity_documents", "kinship"],
    "organization": ["employment", "financial", "legal_customer", "contact"],
    "registry": ["identity_documents", "kinship", "civil_attributes", "contact"],  # [e01] v2.2 group
}

# neighbour relations that only make sense next to specific primaries are filtered here
# (a union membership belongs in an HR / association context, not in a vehicle sale)
NEIGHBOUR_ALLOW = {
    ("employment", "special_attribution"): {"member_of", "disability_of"},
    ("employment", "organization"): {"mersis_of", "trade_registry_of"},
    ("health", "kinship"): {"spouse_of", "child_of", "mother_of", "father_of", "relative_of"},
    ("kinship", "health"): {"diagnosis_of", "disability_of", "medication_of"},
    ("employment", "financial"): {"iban_of", "salary_of"},
    ("legal_customer", "financial"): {"iban_of", "card_of", "salary_of"},
    ("financial", "digital"): {"account_of", "credential_of", "ip_used_by"},
    ("vehicle", "legal_customer"): {"customer_number_of", "order_of", "subscription_of"},
    ("vehicle", "financial"): {"iban_of", "card_of"},
    ("digital", "legal_customer"): {"customer_number_of", "subscription_of", "order_of"},
    ("digital", "identity_documents"): {"national_id_of", "passport_of"},
    ("contact", "location"): {"coordinates_of", "workplace_of"},
    ("contact", "legal_customer"): {"customer_number_of", "subscription_of", "order_of"},
    ("location", "vehicle"): {"vehicle_plate_of"},
    ("location", "digital"): {"device_used_by", "ip_used_by"},
    ("location", "employment"): {"employer_of", "position_of", "employee_id_of"},
}

# primary group -> (genre, format) pool. Genres are Turkish document types a reader would
# expect to carry that scenario's personal data.
GENRES = {
    "health": [
        ("hasta epikriz raporu", "unstructured"),
        ("poliklinik muayene notu", "unstructured"),
        ("hastane yatış kabul formu", "structured"),
        ("sağlık kurulu raporu", "structured"),
        ("hasta bilgilendirme ve onam formu", "structured"),
        ("aile hekimi sevk yazısı", "unstructured"),
    ],
    "financial": [
        ("kredi başvuru formu", "structured"),
        ("banka hesap bilgilendirme yazısı", "unstructured"),
        ("kredi kartı ekstre bildirimi", "structured"),
        ("havale/EFT talimat formu", "structured"),
        ("kira sözleşmesi ödeme maddesi", "unstructured"),
        ("sigorta poliçesi", "structured"),
    ],
    "employment": [
        ("belirsiz süreli iş sözleşmesi", "unstructured"),
        ("personel özlük dosyası formu", "structured"),
        ("işe giriş bildirgesi", "structured"),
        ("performans değerlendirme yazısı", "unstructured"),
        ("disiplin tutanağı", "unstructured"),
        ("çalışma belgesi / referans mektubu", "unstructured"),
    ],
    "legal_customer": [
        ("dava dilekçesi", "unstructured"),
        ("mahkeme tebligatı", "unstructured"),
        ("abonelik sözleşmesi", "structured"),
        ("müşteri şikayet kaydı", "structured"),
        ("sipariş teyit e-postası", "unstructured"),
        ("icra takip yazısı", "unstructured"),
    ],
    "identity_documents": [
        ("nüfus kayıt örneği", "structured"),
        ("kimlik doğrulama formu", "structured"),
        ("pasaport başvuru formu", "structured"),
        ("e-Devlet kayıt çıktısı", "structured"),
        ("noter vekaletname", "unstructured"),
        ("SGK hizmet dökümü", "structured"),
    ],
    "kinship": [
        ("okul veli bilgi formu", "structured"),
        ("aile hekimi kayıt formu", "structured"),
        ("vukuatlı nüfus kayıt örneği", "structured"),
        ("veraset ilamı dilekçesi", "unstructured"),
        ("acil durum iletişim formu", "structured"),
        ("evlilik başvuru beyanı", "unstructured"),
    ],
    "civil_attributes": [
        ("üyelik başvuru formu", "structured"),
        ("anket katılımcı kaydı", "structured"),
        ("burs başvuru formu", "structured"),
        ("kurs kayıt formu", "structured"),
        ("kan bağışı kayıt formu", "structured"),
        ("dernek üye bilgi formu", "structured"),
    ],
    "name_variants": [
        ("isim/soyadı değişikliği dilekçesi", "unstructured"),
        ("evlilik sonrası soyadı bildirimi", "unstructured"),
        ("kayıt düzeltme talebi", "unstructured"),
        ("okul mezuniyet kaydı düzeltme yazısı", "unstructured"),
        ("banka müşteri bilgi güncelleme formu", "structured"),
    ],
    "contact": [
        ("adres değişikliği bildirimi", "unstructured"),
        ("kargo teslimat formu", "structured"),
        ("abonelik adres güncelleme talebi", "unstructured"),
        ("tebligat adresi beyanı", "unstructured"),
        ("randevu teyit e-postası", "unstructured"),
        ("iletişim bilgileri güncelleme formu", "structured"),
    ],
    "location": [
        ("saha personeli konum raporu", "unstructured"),
        ("kargo takip kaydı", "structured"),
        ("araç takip sistemi raporu", "structured"),
        ("bina giriş-çıkış tutanağı", "structured"),
        ("iş seyahati görev raporu", "unstructured"),
    ],
    "digital": [
        ("BT güvenlik olay raporu", "unstructured"),
        ("hesap kurtarma talebi", "unstructured"),
        ("kullanıcı destek talebi", "unstructured"),
        ("erişim izni formu", "structured"),
        ("veri ihlali bildirimi", "unstructured"),
        ("hesap kapatma talebi", "unstructured"),
    ],
    "vehicle": [
        ("araç satış sözleşmesi", "unstructured"),
        ("trafik sigortası poliçesi", "structured"),
        ("kaza tespit tutanağı", "structured"),
        ("araç muayene raporu", "structured"),
        ("otopark abonelik formu", "structured"),
        ("araç kiralama sözleşmesi", "unstructured"),
    ],
    "special_attribution": [
        ("dernek/sendika üyelik başvurusu", "structured"),
        ("sosyal hizmet inceleme raporu", "unstructured"),
        ("denetimli serbestlik yazısı", "unstructured"),
        ("engelli sağlık kurulu raporu", "structured"),
        ("adli sicil kaydı yazısı", "structured"),
        ("insan kaynakları eşit fırsat anketi", "structured"),
        ("cezaevi ziyaret başvuru formu", "structured"),
    ],
    "registry": [  # [e01] v2.2 group
        ("nüfus kayıt örneği", "structured"),
        ("tapu kayıt belgesi", "structured"),
        ("veraset ilamı", "unstructured"),
        ("kat mülkiyeti bilgi yazısı", "unstructured"),
        ("yerleşim yeri ve adres beyanı", "structured"),
    ],
    "organization": [
        ("e-fatura", "structured"),
        ("ticaret sicil gazetesi ilanı", "unstructured"),
        ("tedarikçi sözleşmesi", "unstructured"),
        ("şirket imza sirküleri", "unstructured"),
        ("hizmet alım sözleşmesi", "unstructured"),
    ],
}

# scenario weights: Madde 6 primaries are kept modest so the overall Madde 6 share lands
# near 35% (they also enter as neighbours of health/employment/legal scenarios)
WEIGHTS = {g: 1.0 for g in COMPAT}
WEIGHTS.update(
    {"health": 1.9, "special_attribution": 1.6, "organization": 0.6, "name_variants": 0.7, "registry": 0.8}
)  # [e01]
