# Contributing to Medicare RAG

Thanks for your interest in contributing. This document covers how to run tests, lint, and what we expect from pull requests.

## Setup

1. Create and activate a virtual environment (see [README.md](README.md#quick-start)).
2. Install the package in editable mode with dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

   This installs pytest, ruff, and rank-bm25 so you can run the full test suite and hybrid retrieval.

## Running tests

From the repository root:

```bash
pytest tests/ -v
```

- Tests use mocks for HTTP and external dependencies; no network or real downloads are required.
- Some index tests are skipped automatically when ChromaDB is unavailable (e.g. on some Python 3.14+ setups).
- A shared `conftest.py` fixture resets the BM25 index after each test for isolation.

Run a specific test file or test:

```bash
pytest tests/test_config.py -v
pytest tests/test_ingest.py -k "chunk" -v
```

## Linting and formatting (Ruff)

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- **Lint:** `ruff check src/ tests/`
- **Format:** `ruff format src/ tests/`

Configuration is in `pyproject.toml`: `target-version = "py311"`, `line-length = 100`, rules `E, F, W, I, B, UP`.

## Pull request expectations

1. **Tests pass** — Run `pytest tests/ -v` before submitting.
2. **Ruff clean** — Run `ruff check src/ tests/` and `ruff format src/ tests/`.
3. **Add tests for new behavior** — New features or bug fixes should include or extend tests in `tests/`.
4. **Conventions** — See [AGENTS.md](AGENTS.md) for layout, configuration patterns, and code style. We do not require backwards compatibility; refactor as needed.

## Adding tests for new features

- Place new tests in the appropriate `tests/test_*.py` module (e.g. `test_ingest.py` for ingest changes).
- Use `tmp_path` for file I/O and `unittest.mock.patch` for HTTP and heavy dependencies.
- For retriever/index tests, use the existing patterns in `test_query.py`, `test_hybrid.py`, or `test_index.py` (mocked store/embeddings).

## More information

- **Project layout and conventions:** [AGENTS.md](AGENTS.md)
- **Architecture and data flow:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Configuration:** [README.md](README.md#configuration) and `.env.example`
