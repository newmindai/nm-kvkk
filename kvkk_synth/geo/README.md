# geo — real Turkish address hierarchy for the sampler

Source: `raw/*.sql` from https://github.com/berkanumutlu/php-turkiye-il-ilce-adres (MIT, © 2024 Berkan Ümütlü),
whose stated data source is the civil registry address query https://adres.nvi.gov.tr/VatandasIslemleri/AdresSorgu
(dump dated 2024-04/05). Cross-check list: `raw/turkiye_mahalleleri.cc0.csv` (Hugging Face `ceyyyh/turkiye_mahalleleri`, CC0,
30,778 neighbourhood rows). Raw dumps are gitignored; the parquet tables are built by `build_geo.py` in 18 s.

| table | rows | note |
|---|---|---|
| cities | 81 | il, with plate code |
| towns | 975 | ilçe |
| neighbourhoods | 74659 | mahalle / köy with 5-digit postal code; 12767 flagged rural; street count per neighbourhood |
| streets | 1097149 | kinds: {'sokak': 887723, 'cadde': 117925, 'kume': 63900, 'village': 13753, 'bulvar': 8820, 'other': 5028} |

Names are Title-cased with Turkish rules; village qualifiers in parentheses are dropped from the neighbourhood name
(kept in `name_raw`). The sampler draws city → ilçe → mahalle (weighted by street count, i.e. urban) → street, and
takes the postal code from the mahalle, so every address component agrees.
