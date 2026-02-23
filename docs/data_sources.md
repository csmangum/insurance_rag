# Data sources

Medicare RAG downloads from CMS and (optionally) CDC. This document centralizes URLs, formats, and update expectations.

## Overview

| Source | Provider | Content | Update frequency |
|--------|----------|---------|-------------------|
| IOM | CMS | Medicare manuals (100-02, 100-03, 100-04) | As CMS updates index/chapters |
| MCD | CMS | LCDs, NCDs, Articles | As CMS publishes exports |
| HCPCS | CMS | HCPCS Level II alpha-numeric file | Quarterly |
| ICD-10-CM | CMS / CDC | ICD-10-CM tabular (optional) | Annual (code year) |

All downloads go under `data/raw/` (or `RAW_DIR` from `.env`). Each source has a `manifest.json` with source URL, download date, and file list (optional SHA-256).

---

## IOM (Internet-Only Manuals)

- **Index URL:**  
  [https://www.cms.gov/medicare/regulations-guidance/manuals/internet-only-manuals-ioms](https://www.cms.gov/medicare/regulations-guidance/manuals/internet-only-manuals-ioms)

- **Manuals used:** 100-02 (Benefit Policy), 100-03 (NCD), 100-04 (Claims Processing). The downloader scrapes the index for these three, then follows each manual’s page to collect chapter PDF links.

- **Format:** Chapter PDFs per manual (e.g. `data/raw/iom/100-02/...pdf`).

- **Pipeline:** `download/iom.py` → `ingest/extract.py` (pdfplumber, optional unstructured fallback) → chunking → embed → store.

- **Notes:** If CMS restructures the IOM index or link text, the scraper may need updates (e.g. `TARGET_MANUALS` or link selectors in `iom.py`).

---

## MCD (Medicare Coverage Database)

- **Bulk export URL:**  
  [https://downloads.cms.gov/medicare-coverage-database/downloads/exports/all_data.zip](https://downloads.cms.gov/medicare-coverage-database/downloads/exports/all_data.zip)

- **Content:** Single ZIP containing inner ZIPs and/or CSVs for:
  - **LCD** — Local Coverage Determinations: `current_lcd.zip`, `all_lcd.zip` (each with CSV data, e.g. `lcd.csv` with policy text).
  - **NCD** — National Coverage Determinations: `ncd.zip`.
  - **Article** — Coverage articles: `current_article.zip`, `all_article.zip`.

- **Format:** Extracted to `data/raw/mcd/`. Inner ZIPs are extracted; CSVs are parsed with configurable `CSV_FIELD_SIZE_LIMIT` (default 10 MB) so large policy/HTML fields are not truncated.

- **Pipeline:** `download/mcd.py` → `ingest/extract.py` (CSV parsing, HTML stripping in cells) → MCD-specific chunking (larger `LCD_CHUNK_SIZE`/`LCD_CHUNK_OVERLAP`) → embed → store.

- **Notes:** LCD policy text can be very long; ensure `CSV_FIELD_SIZE_LIMIT` is sufficient if you see truncation.

---

## HCPCS (Level II codes)

- **Quarterly update page:**  
  [https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/quarterly-update](https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/quarterly-update)

- **Format:** The downloader finds the latest “Alpha-Numeric HCPCS File (ZIP)” link on that page and downloads it to `data/raw/codes/hcpcs/`. Content is a ZIP containing fixed-width (320-char) record files; the ingest pipeline parses them and enriches with semantic labels (see `ingest/enrich.py`).

- **Update frequency:** Quarterly. Re-run `download_all.py --source codes` and re-ingest when you want the latest codes.

- **Pipeline:** `download/codes.py` → `ingest/extract.py` (fixed-width parser) → `ingest/enrich.py` (HCPCS category/synonyms) → one chunk per code → embed → store.

---

## ICD-10-CM (optional)

- **URL:** Not hardcoded. Set `ICD10_CM_ZIP_URL` in `.env` to a CMS or CDC ZIP that contains ICD-10-CM tabular (XML) data.  
  Example (update year as needed):  
  `https://www.cms.gov/files/zip/2025-code-tables-tabular-and-index.zip`  
  See [CDC ICD-10-CM](https://www.cdc.gov/nchs/icd/icd-10-cm.htm) or `.env.example` for alternatives.

- **Format:** ZIP containing XML (e.g. tabular and index). Ingest uses `defusedxml` when available for safe parsing; extracts code + description for embedding.

- **Update frequency:** Typically annual (code year). Set the URL for the code year you want and re-download/re-ingest when updating.

- **Pipeline:** `download/codes.py` (only if `ICD10_CM_ZIP_URL` is set) → `ingest/extract.py` (XML) → `ingest/enrich.py` (ICD-10-CM chapter/synonyms) → one chunk per code → embed → store.

---

## Download commands

```bash
# All sources
python scripts/download_all.py --source all

# Single source
python scripts/download_all.py --source iom
python scripts/download_all.py --source mcd
python scripts/download_all.py --source codes
```

Use `--force` to re-download and overwrite; otherwise the pipeline skips when a valid manifest and files already exist.
