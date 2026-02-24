# Medicare RAG

A **Retrieval-Augmented Generation (RAG)** proof-of-concept for Medicare Revenue Cycle Management. It ingests CMS manuals, coverage determinations, and coding files; embeds them in a vector store; and answers natural-language questions with cited sources. Everything runs locally—no API keys required.

## What it does

- **Download** — IOM manuals (100-02, 100-03, 100-04), MCD bulk data, HCPCS and optional ICD-10-CM code files into `data/raw/`.
- **Ingest** — Extract text (PDF and structured sources), enrich HCPCS/ICD-10 documents with semantic labels and related terms, chunk with LangChain splitters, embed with sentence-transformers, and upsert into ChromaDB with incremental updates by content hash.
- **Query** — Interactive REPL and RAG chain: retrieve relevant chunks, then generate answers using a local Hugging Face model (e.g. TinyLlama) with citations.
- **Validate & evaluate** — Index validation (metadata, sources, embedding dimension) and retrieval evaluation (hit rate, MRR) against a Medicare-focused question set.
- **Embedding search UI** — Optional Streamlit app for interactive semantic search over the index with filters and quick-check questions.

## Requirements

- **Python 3.11+**
- **No API keys** — Embeddings and LLM run locally via sentence-transformers and Hugging Face.

## Quick start

```bash
# Create venv and install
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Optional: copy .env.example to .env for paths and model overrides
cp .env.example .env

# Download data (IOM, MCD, codes)
python scripts/download_all.py --source all

# Extract, chunk, embed, and store
python scripts/ingest_all.py --source all

# Ask questions (RAG with local LLM)
python scripts/query.py
```

## Pipeline in detail

### 1. Download

```bash
python scripts/download_all.py [--source iom|mcd|codes|all] [--force]
```

- **Sources:** `iom` (IOM manuals), `mcd` (MCD bulk ZIP), `codes` (HCPCS + optional ICD-10-CM), or `all`.
- **Idempotent:** Skips when manifest and files exist; use `--force` to re-download.
- Output: `data/raw/<source>/` plus a `manifest.json` per source (URL, date, file list, optional SHA-256). Set `ICD10_CM_ZIP_URL` in `.env` if you want ICD-10-CM (see [CDC ICD-10-CM](https://www.cdc.gov/nchs/icd/icd-10-cm/index.html) or the ZIP URL in `.env.example`).

### 2. Ingest (extract → chunk → embed → store)

```bash
python scripts/ingest_all.py [--source iom|mcd|codes|all] [--force] [--skip-extract] [--skip-index] [--no-summaries]
```

- **Extract:** PDFs (pdfplumber; optional `unstructured` for image-heavy PDFs), MCD/codes from structured files. HCPCS and ICD-10-CM documents are automatically enriched with category labels, synonyms, and related terms (e.g., E-codes get "Durable Medical Equipment: wheelchair, hospital bed, oxygen equipment...") to improve semantic retrieval.
- **Chunk:** LangChain text splitters; MCD/LCD documents use larger chunks (`LCD_CHUNK_SIZE=1500`) to preserve policy context. Metadata (source, manual, jurisdiction, etc.) is preserved.
- **Topic summaries:** By default, document-level and topic-cluster summaries are generated (extractive, no LLM needed) and indexed alongside regular chunks. These act as stable retrieval anchors for fragmented topics. Disable with `--no-summaries`.
- **Embed & store:** sentence-transformers (default `all-MiniLM-L6-v2`) and ChromaDB at `data/chroma/` (collection `medicare_rag`). Only new or changed chunks (by content hash) are re-embedded and upserted.
- Use `--skip-extract` to skip extraction and only run chunking on existing processed files.
- Use `--skip-index` to run only extract and chunk (no embedding or vector store).

### 3. Query (RAG)

```bash
python scripts/query.py [--filter-source iom|mcd|codes] [--filter-manual 100-02] [--filter-jurisdiction JL] [-k 8]
```

- Retrieves top-k chunks by similarity, then generates an answer with the local LLM and prints cited sources. With `pip install -e ".[hybrid]"` (or `pip install -e ".[dev]"`, both of which add `rank-bm25`), the default retriever is **hybrid** (semantic + BM25 via Reciprocal Rank Fusion, cross-source query expansion, source diversification, topic-summary boosting). Without `rank-bm25`, the LCD-aware semantic retriever is used instead.
- **Env:** `LOCAL_LLM_MODEL`, `LOCAL_LLM_DEVICE` (e.g. `cpu` or `auto`), `LOCAL_LLM_MAX_NEW_TOKENS`, `LOCAL_LLM_REPETITION_PENALTY`. Use `CUDA_VISIBLE_DEVICES=""` for CPU-only.

### 4. Validate and evaluate

```bash
python scripts/validate_and_eval.py                    # validate index + run retrieval eval
python scripts/validate_and_eval.py --validate-only   # index only
python scripts/validate_and_eval.py --eval-only -k 10  # retrieval eval only
python scripts/validate_and_eval.py --eval-only --json # metrics as JSON (stdout; redirect to save)
python scripts/validate_and_eval.py --report data/eval_report.md  # write markdown report (requires eval, not --validate-only)
```

- **Validation:** Checks Chroma collection, document count, sample metadata (`doc_id`, `content_hash`), and that similarity search runs.
- **Evaluation:** Uses `scripts/eval_questions.json` (Medicare queries with expected keywords/sources). Reports **hit rate** (relevant doc in top-k) and **MRR** (mean reciprocal rank). Edit `eval_questions.json` to extend the set. With `--json`, metrics are printed to stdout; redirect to save, e.g. `... --json > scripts/eval_metrics.json`.

**Full-RAG eval (answer quality):** Run the RAG chain on the eval set and write a report for manual review:

```bash
python scripts/run_rag_eval.py [--eval-file scripts/eval_questions.json] [--out data/rag_eval_report.md] [-k 8]
```

The latest report is written to `data/rag_eval_report.md` and includes answer-quality heuristics for manual assessment.

**Results and analysis:** See **[EVAL_RESULTS.md](EVAL_RESULTS.md)** for the full evaluation suite (unit tests, index validation, retrieval metrics, RAG answer quality) and recommendations. High-level snapshot from a typical run: **393** unit tests passing; retrieval **82% hit rate**, **0.58 MRR**, **0.91 NDCG@5** on 79 eval questions; index validation passes except when the `codes` source is not ingested. RAG answer quality (keyword coverage, citations) is limited by the default small local LLM; retrieval is strong and improvements are best focused on prompt/model for generation and citing.

### 5. Embedding search UI (optional)

```bash
pip install -e ".[ui]"
streamlit run app.py
```

- Semantic search over the index with a search bar and quick-check question buttons.
- Filters: source, manual, jurisdiction.
- Options: top-k, distance threshold, full chunk content.
- Styled result cards with similarity scores and metadata.

## Configuration

Copy `.env.example` to `.env` and override as needed:

| Variable | Purpose |
|----------|----------|
| `DATA_DIR` | Root for `raw/`, `processed/`, `chroma/` (default: repo `data/`) |
| `EMBEDDING_MODEL` | sentence-transformers model (default: `all-MiniLM-L6-v2`). Changing it changes vector dimension; re-ingest or match the model used at index time. |
| `LOCAL_LLM_MODEL` | Hugging Face model (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) |
| `LOCAL_LLM_DEVICE` | `auto`, `cpu`, or device map |
| `LOCAL_LLM_MAX_NEW_TOKENS` | Max tokens generated (default: 512). Invalid values fall back to default with a warning. |
| `LOCAL_LLM_REPETITION_PENALTY` | Repetition penalty (default: 1.05). Invalid values fall back to default with a warning. |
| `ICD10_CM_ZIP_URL` | Optional; for ICD-10-CM code download |
| `DOWNLOAD_TIMEOUT` | HTTP timeout in seconds for downloads (default: 60) |
| `CSV_FIELD_SIZE_LIMIT` | Max CSV field size in bytes for MCD ingestion (default: 10 MB). Increase if very large policy fields are truncated. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Standard text splitter settings (1000 / 200). |
| `LCD_CHUNK_SIZE`, `LCD_CHUNK_OVERLAP` | MCD/LCD-specific chunking (1500 / 300). Larger to preserve policy context. |
| `LCD_RETRIEVAL_K` | Higher k for LCD/coverage-determination queries (default: 12). |
| `ENABLE_TOPIC_SUMMARIES` | Generate topic-cluster and document-level summaries at ingest time (default: `1`). |
| `HYBRID_SEMANTIC_WEIGHT`, `HYBRID_KEYWORD_WEIGHT` | Fusion weights for hybrid retriever (0.6 / 0.4). |
| `RRF_K` | Reciprocal Rank Fusion smoothing parameter (default: 60). |
| `CROSS_SOURCE_MIN_PER_SOURCE` | Minimum docs per source type in diversified results (default: 2). |

## Testing

Install the dev optional dependency (includes pytest, ruff, and rank-bm25), then run:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

- **Config:** `tests/test_config.py` — safe env var parsing for numeric settings.
- **Download:** `tests/test_download.py` — mocked HTTP, idempotency, zip-slip and URL sanitization.
- **Ingest:** `tests/test_ingest.py` — extraction and chunking (including enrichment integration).
- **Enrichment:** `tests/test_enrich.py` — HCPCS/ICD-10-CM semantic enrichment (category labels, synonyms, edge cases).
- **Clustering:** `tests/test_cluster.py` — topic definition loading, assignment, clustering, and tagging.
- **Summarization:** `tests/test_summarize.py` — document-level and topic-cluster summary generation.
- **Index:** `tests/test_index.py` — Chroma and embeddings (skipped when Chroma unavailable, e.g. some Python 3.14+ setups).
- **Query:** `tests/test_query.py` — LCD query detection, query expansion, `LCDAwareRetriever`.
- **Hybrid retrieval:** `tests/test_hybrid.py` — BM25 index, RRF fusion, source diversification, `HybridRetriever`.
- **Summary boosting:** `tests/test_retriever_boost.py` — topic-summary injection and boosting in retrieval results.
- **Validation/eval:** `tests/test_search_validation.py` — validation and eval question schema.
- **UI helpers:** `tests/test_app.py` — Streamlit app helpers (requires `.[ui]`).

No network or real downloads needed for the core suite; mocks are used for HTTP and external deps.

## Optional extras

- **`pip install -e ".[ui]"`** — Streamlit for the embedding search UI.
- **`pip install -e ".[dev]"`** — pytest (test suite), ruff (linting/formatting), and rank-bm25 (hybrid retrieval). Required to run tests.
- **`pip install -e ".[unstructured]"`** — Fallback extractor for image-heavy PDFs when pdfplumber yields little text.

## Project layout

- **`src/medicare_rag/`** — Main package:
  - `config.py` — centralized configuration.
  - `download/` — IOM, MCD, and HCPCS/ICD-10-CM downloaders.
  - `ingest/` — extraction (`extract.py`), enrichment (`enrich.py`), chunking (`chunk.py`), topic clustering (`cluster.py`), and summarization (`summarize.py`).
  - `index/` — embedding (`embed.py`) and ChromaDB vector store (`store.py`).
  - `query/` — LCD-aware retriever (`retriever.py`), hybrid retriever (`hybrid.py`), cross-source query expansion (`expand.py`), and RAG chain (`chain.py`).
- **`scripts/`** — CLI: `download_all.py`, `ingest_all.py`, `validate_and_eval.py`, `query.py`, `run_rag_eval.py`, `eval_questions.json`.
- **`tests/`** — Pytest suite (see Testing section).
- **`data/`** — Runtime data (gitignored): `raw/`, `processed/`, `chroma/`.

See **AGENTS.md** for detailed layout, conventions, and patterns.

## Documentation

- **[EVAL_RESULTS.md](EVAL_RESULTS.md)** — Full evaluation suite: unit tests, index validation, retrieval metrics (hit rate, MRR, NDCG), RAG answer quality summary, and analysis with recommendations.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to run tests, ruff, and contribute.
- **[docs/troubleshooting.md](docs/troubleshooting.md)** — Common issues and fixes.
- **[docs/eval_questions.md](docs/eval_questions.md)** — Eval question schema and how to run evaluation.
- **[docs/topic_definitions.md](docs/topic_definitions.md)** — Topic definitions for clustering and summary boosting.
- **[docs/data_sources.md](docs/data_sources.md)** — External data URLs and formats.
