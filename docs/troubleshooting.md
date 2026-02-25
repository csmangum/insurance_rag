# Troubleshooting

Common issues and how to fix them.

## Embedding dimension mismatch

**Symptom:** Streamlit app or retrieval fails with an error about embedding dimension (e.g. expected 384, got 768).

**Cause:** The embedding model was changed (e.g. via `EMBEDDING_MODEL`) but the ChromaDB collection was built with a different model. Chroma stores vectors with a fixed dimension per collection.

**Fix:**

1. Either re-ingest so the index is rebuilt with the current model:
   ```bash
   python scripts/ingest_all.py --source all --force
   ```
2. Or set `EMBEDDING_MODEL` back to the value used when the index was created (default: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions).

---

## ChromaDB unavailable / import errors

**Symptom:** Tests or app fail with ChromaDB import or pydantic errors (e.g. on Python 3.14+).

**Cause:** ChromaDB (or its dependency pydantic) may be incompatible with some Python versions.

**Fix:**

- Index-related tests in `test_index.py` are skipped automatically when Chroma is unavailable; the rest of the suite can still run.
- For local use, use a supported Python version (3.11 or 3.12). See `pyproject.toml` for `requires-python`.

---

## Empty index / no results

**Symptom:** Query or Streamlit search returns no documents, or "collection is empty".

**Cause:** The pipeline has not been run, or data directories are empty.

**Fix:**

1. Download data: `python scripts/download_all.py --source all`
2. Ingest (extract, chunk, embed, store): `python scripts/ingest_all.py --source all`
3. Ensure `DATA_DIR` (default: repo `data/`) is where you ran download/ingest and contains `chroma/` with the `insurance_rag` collection.

---

## PDF extraction yields little or no text

**Symptom:** IOM (or other) PDFs produce very short or empty extracted text.

**Cause:** pdfplumber works best on text-based PDFs. Image-only or scanned PDFs have little extractable text.

**Fix:**

- Install the optional unstructured extra: `pip install -e ".[unstructured]"`
- The extractor falls back to `unstructured` when a PDF page yields fewer than 50 characters with pdfplumber, which can improve extraction for some scanned/image-heavy PDFs.

---

## LCD / coverage-determination retrieval is weak

**Symptom:** Queries about LCDs, NCDs, or MAC coverage return few or no relevant chunks; eval category `lcd_policy` has low hit rate.

**Cause:** LCD content is long and policy-dense; chunk boundaries and retrieval settings may need tuning.

**Fix:**

- Increase LCD retrieval k: set `LCD_RETRIEVAL_K` (default 12) higher in `.env`.
- Ensure full LCD policy text is ingested: the pipeline uses `CSV_FIELD_SIZE_LIMIT` (default 10 MB) so large policy fields in MCD CSVs are not truncated; increase if needed.
- Use hybrid retrieval for better recall: `pip install -e ".[dev]"` or `pip install -e ".[hybrid]"` so the hybrid retriever (semantic + BM25) is used by default.

---

## ICD-10-CM queries fail or return no codes

**Symptom:** Queries about ICD-10-CM codes (e.g. hypertension, diabetes) do not find code documents.

**Cause:** ICD-10-CM data is optional and not downloaded unless a URL is configured.

**Fix:**

1. Set `ICD10_CM_ZIP_URL` in `.env` (see `.env.example` for a commented example; use the current year’s CMS or CDC tabular ZIP URL).
2. Download codes: `python scripts/download_all.py --source codes`
3. Re-ingest: `python scripts/ingest_all.py --source codes` (or `--source all`).

---

## Download fails (timeout, 404, or connection error)

**Symptom:** `download_all.py` fails with httpx timeout or HTTP errors.

**Cause:** Network issues, CMS/CDC URL changes, or rate limiting.

**Fix:**

- Increase timeout: set `DOWNLOAD_TIMEOUT` in `.env` (default 60 seconds).
- Check that CMS.gov/CDC.gov pages are up and URLs in the code (or `.env` for ICD-10-CM) are still valid; CMS may change quarterly HCPCS or IOM index structure.
- Re-run with `--force` only after fixing the cause, so you do not keep retrying bad URLs.

---

## Validation or eval script errors

**Symptom:** `validate_and_eval.py` fails (e.g. missing keys in eval JSON, or "no questions").

**Cause:** Eval question file schema is wrong or path is incorrect.

**Fix:**

- Use the canonical eval file: `scripts/eval_questions.json`
- If using a custom file, ensure each question has required fields: `id`, `query`, `expected_keywords`, `expected_sources`, `category`, `difficulty`. See [docs/eval_questions.md](eval_questions.md) for the full schema.
