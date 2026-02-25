# Insurance RAG

A **multi-domain Retrieval-Augmented Generation (RAG)** system for US insurance industries. It uses a domain-plugin architecture where each insurance line (Medicare, Auto, Property, etc.) registers its own data sources, extractors, query patterns, and system prompts. Shared infrastructure handles embeddings, vector storage, chunking, hybrid retrieval, and LLM generation. Everything runs locally—no API keys required.

**Implemented domains:** **Medicare** (fully functional) and **Auto Insurance** (scaffolded).

## What it does

- **Download** — Domain-specific data into `data/<domain>/raw/` (e.g. IOM manuals, MCD bulk data, HCPCS/ICD-10-CM for Medicare; regulations, forms, claims, rates for Auto).
- **Ingest** — Extract text (PDF and structured sources), enrich where applicable (e.g. HCPCS/ICD-10 semantic labels), chunk with LangChain splitters, embed with sentence-transformers, and upsert into ChromaDB with per-domain collections and incremental updates by content hash.
- **Query** — Interactive REPL and RAG chain: retrieve relevant chunks, then generate answers using a local Hugging Face model (e.g. TinyLlama) with citations. Domain-aware query expansion and specialized retrieval (e.g. LCD/coverage for Medicare, coverage/limits for Auto).
- **Validate & evaluate** — Index validation (metadata, sources, embedding dimension) and retrieval evaluation (hit rate, MRR) against domain-specific question sets.
- **Embedding search UI** — Optional Streamlit app with domain selector for interactive semantic search over the index with filters and quick-check questions.

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

# Download data for a domain (default: medicare)
python scripts/download_all.py --domain medicare --source all

# Extract, chunk, embed, and store
python scripts/ingest_all.py --domain medicare --source all

# Ask questions (RAG with local LLM)
python scripts/query.py --domain medicare
```

## Domain plugin architecture

Each domain is a self-contained plugin that provides:

- **Data sources** — Downloaders and extractors for raw data
- **Topic definitions** — For clustering and summary boosting
- **Query patterns** — Specialized query detection (e.g. LCD/coverage for Medicare)
- **Source patterns** — Cross-source relevance and expansion
- **System prompt** — LLM instructions tailored to the domain

Domains are discovered via `@register_domain` and selected with `--domain`. Data is stored under `data/<domain>/raw/` and `data/<domain>/processed/`; each domain has its own ChromaDB collection.

See **AGENTS.md** for the full `InsuranceDomain` interface and how to add new domains.

## Pipeline in detail

### 1. Download

```bash
python scripts/download_all.py [--domain medicare|auto|all] [--source <source-kind>|all] [--force]
```

- **Domains:** `medicare`, `auto`, or `all` (runs all registered domains).
- **Sources:** Domain-specific (e.g. `iom`, `mcd`, `codes` for Medicare; `regulations`, `forms`, `claims`, `rates` for Auto).
- **Idempotent:** Skips when manifest and files exist; use `--force` to re-download.
- Output: `data/<domain>/raw/<source>/` plus a `manifest.json` per source. Set `ICD10_CM_ZIP_URL` in `.env` for Medicare ICD-10-CM (see [CDC ICD-10-CM](https://www.cdc.gov/nchs/icd/icd-10-cm/index.html) or `.env.example`).

### 2. Ingest (extract → chunk → embed → store)

```bash
python scripts/ingest_all.py [--domain medicare|auto|all] [--source <source-kind>|all] [--force] [--skip-extract] [--skip-index] [--no-summaries]
```

- **Extract:** Domain-specific extractors (PDFs via pdfplumber; optional `unstructured` for image-heavy PDFs; structured MCD/codes for Medicare). HCPCS and ICD-10-CM (Medicare) are enriched with category labels and related terms.
- **Chunk:** LangChain text splitters; domain-specific overrides (e.g. MCD/LCD uses larger chunks for Medicare). Metadata (source, manual, jurisdiction, etc.) is preserved.
- **Topic summaries:** By default, document-level and topic-cluster summaries are generated (extractive, no LLM needed) and indexed. Disable with `--no-summaries`.
- **Embed & store:** sentence-transformers (default `all-MiniLM-L6-v2`) and ChromaDB at `data/chroma/` with per-domain collections (e.g. `medicare`, `auto_insurance`). Only new or changed chunks (by content hash) are re-embedded and upserted.
- Use `--skip-extract` to skip extraction and only run chunking on existing processed files.
- Use `--skip-index` to run only extract and chunk (no embedding or vector store).

### 3. Query (RAG)

```bash
python scripts/query.py [--domain medicare|auto] [--filter-source <source>] [--filter-manual 100-02] [--filter-jurisdiction JL] [--filter-state CA] [-k 8]
```

- Retrieves top-k chunks by similarity, then generates an answer with the local LLM and prints cited sources. With `pip install -e ".[hybrid]"` (or `pip install -e ".[dev]"`), the default retriever is **hybrid** (semantic + BM25 via Reciprocal Rank Fusion, cross-source query expansion, source diversification, topic-summary boosting). Without `rank-bm25`, the domain-aware semantic retriever is used instead.
- **Env:** `LOCAL_LLM_MODEL`, `LOCAL_LLM_DEVICE` (e.g. `cpu` or `auto`), `LOCAL_LLM_MAX_NEW_TOKENS`, `LOCAL_LLM_REPETITION_PENALTY`. Use `CUDA_VISIBLE_DEVICES=""` for CPU-only.

### 4. Validate and evaluate

```bash
python scripts/validate_and_eval.py [--domain medicare|auto]                    # validate index + run retrieval eval
python scripts/validate_and_eval.py --domain medicare --validate-only   # index only
python scripts/validate_and_eval.py --domain medicare --eval-only -k 10  # retrieval eval only
python scripts/validate_and_eval.py --eval-only --json # metrics as JSON (stdout; redirect to save)
python scripts/validate_and_eval.py --report data/eval_report.md  # write markdown report
```

- **Validation:** Checks Chroma collection for the domain, document count, sample metadata (`doc_id`, `content_hash`), and that similarity search runs.
- **Evaluation:** Uses `scripts/eval_questions.json` (domain-specific queries with expected keywords/sources). Reports **hit rate** (relevant doc in top-k) and **MRR** (mean reciprocal rank).

**Full-RAG eval (answer quality):**

```bash
python scripts/run_rag_eval.py [--domain medicare] [--eval-file scripts/eval_questions.json] [--out data/rag_eval_report.md] [-k 8]
```

### 5. Embedding search UI (optional)

```bash
pip install -e ".[ui]"
streamlit run app.py
```

- Domain selector, semantic search bar, and quick-check question buttons.
- Filters: source, manual, jurisdiction (Medicare), state (Auto).
- Options: top-k, distance threshold, full chunk content.
- Styled result cards with similarity scores and metadata.

## Configuration

Copy `.env.example` to `.env` and override as needed:

| Variable | Purpose |
|----------|---------|
| `DATA_DIR` | Root for domain data (default: repo `data/`). Paths are `data/<domain>/raw/`, `data/<domain>/processed/`, `data/chroma/`. |
| `ACTIVE_DOMAINS` | Comma-separated domain names to discover (default: all built-in). |
| `DEFAULT_DOMAIN` | Domain used when none specified (default: `medicare`). |
| `EMBEDDING_MODEL` | sentence-transformers model (default: `all-MiniLM-L6-v2`). Changing it changes vector dimension; re-ingest or match the model used at index time. |
| `LOCAL_LLM_MODEL` | Hugging Face model (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) |
| `LOCAL_LLM_DEVICE` | `auto`, `cpu`, or device map |
| `LOCAL_LLM_MAX_NEW_TOKENS` | Max tokens generated (default: 512). Invalid values fall back to default with a warning. |
| `LOCAL_LLM_REPETITION_PENALTY` | Repetition penalty (default: 1.05). Invalid values fall back to default with a warning. |
| `ICD10_CM_ZIP_URL` | Optional; for Medicare ICD-10-CM code download |
| `DOWNLOAD_TIMEOUT` | HTTP timeout in seconds for downloads (default: 60) |
| `CSV_FIELD_SIZE_LIMIT` | Max CSV field size in bytes for MCD ingestion (default: 10 MB). |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Standard text splitter settings (1000 / 200). |
| `LCD_CHUNK_SIZE`, `LCD_CHUNK_OVERLAP` | MCD/LCD-specific chunking for Medicare (1500 / 300). |
| `LCD_RETRIEVAL_K` | Higher k for specialized/coverage queries (default: 12). |
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
- **Domains:** `tests/test_domains.py` — domain registry, interface compliance, domain-specific tests.
- **Download:** `tests/test_download.py` — mocked HTTP, idempotency, zip-slip and URL sanitization.
- **Ingest:** `tests/test_ingest.py` — extraction and chunking (including enrichment integration).
- **Enrichment:** `tests/test_enrich.py` — HCPCS/ICD-10-CM semantic enrichment (Medicare).
- **Clustering:** `tests/test_cluster.py` — topic definition loading, assignment, clustering, and tagging.
- **Summarization:** `tests/test_summarize.py` — document-level and topic-cluster summary generation.
- **Index:** `tests/test_index.py` — Chroma and embeddings (skipped when Chroma unavailable).
- **Query:** `tests/test_query.py` — specialized query detection, query expansion, domain-aware retriever.
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

- **`src/insurance_rag/`** — Main package:
  - `config.py` — centralized configuration, domain paths, multi-domain support.
  - `domains/` — Domain plugin system (`base.py`, `medicare/`, `auto/`).
  - `download/` — Domain-specific downloaders (IOM, MCD, HCPCS/ICD-10-CM for Medicare).
  - `ingest/` — extraction (`extract.py`), enrichment (`enrich.py`), chunking (`chunk.py`), topic clustering (`cluster.py`), summarization (`summarize.py`).
  - `index/` — embedding (`embed.py`) and ChromaDB vector store (`store.py`).
  - `query/` — domain-aware retriever (`retriever.py`), hybrid retriever (`hybrid.py`), cross-source query expansion (`expand.py`), RAG chain (`chain.py`).
- **`scripts/`** — CLI: `download_all.py`, `ingest_all.py`, `validate_and_eval.py`, `query.py`, `run_rag_eval.py`, `eval_questions.json`.
- **`tests/`** — Pytest suite (see Testing section).
- **`data/`** — Runtime data (gitignored): `data/<domain>/raw/`, `data/<domain>/processed/`, `data/chroma/`.

See **AGENTS.md** for detailed layout, conventions, and patterns.

## Documentation

- **[EVAL_RESULTS.md](EVAL_RESULTS.md)** — Full evaluation suite: unit tests, index validation, retrieval metrics (hit rate, MRR, NDCG), RAG answer quality summary, and analysis.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to run tests, ruff, and contribute.
- **[docs/troubleshooting.md](docs/troubleshooting.md)** — Common issues and fixes.
- **[docs/eval_questions.md](docs/eval_questions.md)** — Eval question schema and how to run evaluation.
- **[docs/topic_definitions.md](docs/topic_definitions.md)** — Topic definitions for clustering and summary boosting.
- **[docs/data_sources.md](docs/data_sources.md)** — External data URLs and formats.
