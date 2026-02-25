# Adding a New Insurance Domain

This guide walks through adding a new insurance domain (e.g., Property/Fire, Workers' Comp, Health) to the Insurance RAG platform. Each domain is a self-contained plugin that plugs into shared infrastructure for embeddings, vector storage, hybrid retrieval, and LLM generation.

No changes to core infrastructure are required to add a new domain.

## Quick reference

| File you create | Purpose |
| --- | --- |
| `domains/<name>/__init__.py` | Domain class with `@register_domain` |
| `domains/<name>/patterns.py` | Query patterns, synonyms, source expansions, system prompt |
| `domains/<name>/data/topics.json` | Topic definitions for clustering |
| `domains/<name>/states.py` | (Optional) State-specific configuration |

After creating these files, register the domain in the discovery function and you're done.

---

## Step 1: Create the domain directory

```
src/insurance_rag/domains/<name>/
    __init__.py
    patterns.py
    data/
        topics.json
    states.py          # optional, for state-specific domains
```

## Step 2: Define query patterns (`patterns.py`)

This file holds all the domain-specific text matching data that drives query expansion, source detection, and synonym enrichment.

### Required exports

```python
"""<Domain> query patterns, synonyms, and expansion data."""
from __future__ import annotations

import re

# 1. SPECIALIZED_QUERY_PATTERNS
# Regexes that identify queries specific to this domain.
# When matched, the retriever applies domain-specific query expansion.
SPECIALIZED_QUERY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\byour pattern here\b",
    ]
]

# 2. SPECIALIZED_TOPIC_PATTERNS
# Each tuple is (regex, expansion_string). When the regex matches a query,
# the expansion string is appended to broaden retrieval.
SPECIALIZED_TOPIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), expansion)
    for p, expansion in [
        (r"\btopic keyword\b", "related terms for expansion"),
    ]
]

# 3. STRIP_NOISE / STRIP_FILLER
# Regexes for removing domain boilerplate and filler words from queries
# before concept extraction.
STRIP_NOISE = re.compile(
    r"\b(?:domain|specific|noise|words)\b",
    re.IGNORECASE,
)
STRIP_FILLER = re.compile(
    r"\b(?:does|have|has|an|the|for|is|are|what|which)\b",
    re.IGNORECASE,
)

# 4. SOURCE_PATTERNS
# Map of source-kind -> list of regexes. Used to detect which source type
# a query is most relevant to, so retrieval can be weighted accordingly.
SOURCE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "source_kind_a": [
        re.compile(r"\bkeyword\b", re.IGNORECASE),
    ],
    "source_kind_b": [
        re.compile(r"\bother keyword\b", re.IGNORECASE),
    ],
}

# 5. SOURCE_EXPANSIONS
# When a source kind is detected as relevant, this string is added to
# the expanded query to improve recall.
SOURCE_EXPANSIONS: dict[str, str] = {
    "source_kind_a": "expansion terms for source A",
    "source_kind_b": "expansion terms for source B",
}

# 6. SYNONYM_MAP
# List of (compiled_regex, expansion_string). When a term matches,
# the expansion is appended to the query for broader retrieval.
SYNONYM_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), expansion)
    for p, expansion in [
        (r"\bdomain term\b", "synonym1 synonym2 related-concept"),
    ]
]

# 7. SYSTEM_PROMPT
# The LLM system prompt for RAG generation. Tailor this to the domain's
# subject matter and expected answer style.
SYSTEM_PROMPT = (
    "You are a <domain> specialist. "
    "Answer the user's question using ONLY the provided context. "
    "Cite sources using [1], [2], etc. "
    "If the context is insufficient to answer, say so explicitly."
)

# 8. DEFAULT_SOURCE_RELEVANCE
# Fallback weights when no source patterns match a query.
# Values should roughly sum to 1.0.
DEFAULT_SOURCE_RELEVANCE: dict[str, float] = {
    "source_kind_a": 0.5,
    "source_kind_b": 0.5,
}

# 9. QUICK_QUESTIONS
# Example questions shown in the Streamlit UI for this domain.
QUICK_QUESTIONS: list[str] = [
    "What is the coverage requirement for ...?",
    "How does ... work in California?",
]
```

## Step 3: Define topic clusters (`data/topics.json`)

Topic definitions drive clustering during ingestion and topic-summary boosting during retrieval. Each topic has patterns that match against chunk text.

```json
[
  {
    "name": "snake_case_topic_id",
    "label": "Human-Readable Topic Name",
    "patterns": [
      "\\bregex pattern\\b",
      "\\banother pattern\\b"
    ],
    "summary_prefix": "Topic Name: ",
    "min_pattern_matches": 1
  }
]
```

Fields:

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Unique snake_case identifier |
| `label` | Yes | Display name |
| `patterns` | Yes | List of regex strings (not compiled) matched against chunk text |
| `summary_prefix` | Yes | Prepended to generated topic summaries |
| `min_pattern_matches` | Yes | Minimum regex hits for a chunk to be assigned to this topic |

Aim for 5-15 topics per domain. Too few and summaries are too broad; too many and clusters become sparse.

## Step 4: (Optional) State-specific configuration (`states.py`)

If your domain has state-level variation (regulations, requirements, limits), create a `states.py` with a dataclass for state config and a list of top markets.

```python
"""State-specific configuration for <Domain>."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class State<Domain>Config:
    """Regulatory profile for a US state."""

    code: str
    name: str
    # Add domain-specific fields:
    # min_coverage: str
    # special_requirements: bool
    notes: str = ""


TOP_MARKETS: list[str] = ["CA", "TX", "FL", "NY", "IL", "PA", "OH"]

STATE_CONFIGS: dict[str, State<Domain>Config] = {
    "CA": State<Domain>Config(
        code="CA",
        name="California",
        notes="...",
    ),
    # ... more states
}
```

State codes are attached as metadata on ingested documents, enabling state-filtered retrieval in the CLI (`--filter-state CA`) and the Streamlit UI.

## Step 5: Implement the domain class (`__init__.py`)

This is the central file that wires everything together. Decorate your class with `@register_domain` and implement the `InsuranceDomain` abstract interface.

```python
"""<Domain> insurance domain plugin."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from insurance_rag.domains import register_domain
from insurance_rag.domains.base import InsuranceDomain

logger = logging.getLogger(__name__)


@register_domain
class <Domain>InsuranceDomain(InsuranceDomain):
    """<Domain> insurance domain.

    Brief description of what this domain covers.
    """

    # -- Identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "<name>"  # short id, used in CLI --domain flag

    @property
    def display_name(self) -> str:
        return "<Display Name>"

    @property
    def collection_name(self) -> str:
        return "<name>_insurance"  # unique ChromaDB collection

    @property
    def source_kinds(self) -> list[str]:
        return ["source_a", "source_b"]

    # -- Data pipeline ---------------------------------------------------

    def get_downloaders(self) -> dict[str, Any]:
        # Return real download functions when data sources are available.
        # Placeholder pattern for scaffolding:
        def _download_placeholder(raw_dir: Path, *, force: bool = False) -> None:
            logger.info("<Domain> downloader placeholder")
            raw_dir.mkdir(parents=True, exist_ok=True)

        return {sk: _download_placeholder for sk in self.source_kinds}

    def get_extractors(self) -> dict[str, Any]:
        # Return real extractor functions when parsers are implemented.
        def _extract_placeholder(
            processed_dir: Path, raw_dir: Path, *, force: bool = False
        ) -> list[tuple[Path, Path]]:
            logger.info("<Domain> extractor placeholder")
            return []

        return {sk: _extract_placeholder for sk in self.source_kinds}

    def get_enricher(self) -> Any | None:
        return None  # implement if you have semantic enrichment

    # -- Topics ----------------------------------------------------------

    def get_topic_definitions_path(self) -> Path:
        return Path(__file__).parent / "data" / "topics.json"

    # -- Query / retrieval -----------------------------------------------

    def get_query_patterns(self) -> dict[str, Any]:
        from insurance_rag.domains.<name>.patterns import (
            SPECIALIZED_QUERY_PATTERNS,
            SPECIALIZED_TOPIC_PATTERNS,
            STRIP_FILLER,
            STRIP_NOISE,
        )

        return {
            "specialized_query_patterns": SPECIALIZED_QUERY_PATTERNS,
            "specialized_topic_patterns": SPECIALIZED_TOPIC_PATTERNS,
            "strip_noise_pattern": STRIP_NOISE,
            "strip_filler_pattern": STRIP_FILLER,
        }

    def get_source_patterns(self) -> dict[str, list[Any]]:
        from insurance_rag.domains.<name>.patterns import SOURCE_PATTERNS

        return SOURCE_PATTERNS

    def get_source_expansions(self) -> dict[str, str]:
        from insurance_rag.domains.<name>.patterns import SOURCE_EXPANSIONS

        return SOURCE_EXPANSIONS

    def get_synonym_map(self) -> list[tuple[Any, str]]:
        from insurance_rag.domains.<name>.patterns import SYNONYM_MAP

        return SYNONYM_MAP

    def get_system_prompt(self) -> str:
        from insurance_rag.domains.<name>.patterns import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def get_default_source_relevance(self) -> dict[str, float]:
        from insurance_rag.domains.<name>.patterns import DEFAULT_SOURCE_RELEVANCE

        return DEFAULT_SOURCE_RELEVANCE

    # -- Optional overrides ----------------------------------------------

    def get_states(self) -> list[str] | None:
        # Return None for federal-level domains (like Medicare).
        # For state-specific domains:
        # from insurance_rag.domains.<name>.states import TOP_MARKETS
        # return TOP_MARKETS
        return None

    def get_chunk_overrides(self) -> dict[str, dict[str, int]] | None:
        # Override chunk sizes for source kinds with unusually long
        # or dense documents:
        # return {"regulations": {"chunk_size": 1500, "chunk_overlap": 300}}
        return None

    def get_quick_questions(self) -> list[str]:
        from insurance_rag.domains.<name>.patterns import QUICK_QUESTIONS

        return QUICK_QUESTIONS
```

## Step 6: Register the domain for discovery

Add your domain's module path to the `_discover_domains` function in `src/insurance_rag/domains/__init__.py`:

```python
def _discover_domains() -> None:
    """Import built-in domain packages so they self-register."""
    import importlib

    for mod_name in (
        "insurance_rag.domains.medicare",
        "insurance_rag.domains.auto",
        "insurance_rag.domains.<name>",        # <-- add this line
    ):
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass
```

## Step 7: Download and ingest data

```bash
# Download raw data
python scripts/download_all.py --domain <name>

# Extract, chunk, embed, and index
python scripts/ingest_all.py --domain <name>

# Query interactively
python scripts/query.py --domain <name>

# Or use the Streamlit UI (select domain from sidebar)
streamlit run app.py
```

Data is stored at `data/<name>/raw/` and `data/<name>/processed/`. The ChromaDB collection is created automatically with the name returned by `collection_name`.

---

## Interface reference

All abstract methods must be implemented. Optional methods have sensible defaults.

| Method | Abstract? | Return type | Purpose |
| --- | --- | --- | --- |
| `name` | Yes | `str` | Short identifier (CLI flag value) |
| `display_name` | Yes | `str` | Human-readable label (UI title) |
| `collection_name` | Yes | `str` | Unique ChromaDB collection name |
| `source_kinds` | Yes | `list[str]` | Source type identifiers |
| `get_downloaders()` | Yes | `dict[str, callable]` | Source -> download function |
| `get_extractors()` | Yes | `dict[str, callable]` | Source -> extract function |
| `get_enricher()` | No | `callable \| None` | Semantic enrichment (default: None) |
| `get_topic_definitions_path()` | Yes | `Path` | Path to `topics.json` |
| `get_query_patterns()` | Yes | `dict[str, Any]` | Domain-specific query detection |
| `get_source_patterns()` | Yes | `dict[str, list]` | Source relevance detection regexes |
| `get_source_expansions()` | Yes | `dict[str, str]` | Source-targeted expansion strings |
| `get_synonym_map()` | Yes | `list[tuple]` | Synonym expansion pairs |
| `get_system_prompt()` | Yes | `str` | LLM system prompt |
| `get_default_source_relevance()` | No | `dict[str, float]` | Fallback relevance weights |
| `get_states()` | No | `list[str] \| None` | US state codes (default: None) |
| `get_chunk_overrides()` | No | `dict \| None` | Per-source chunk size overrides |
| `get_quick_questions()` | No | `list[str]` | Example questions for UI |

---

## Checklist

- [ ] `domains/<name>/__init__.py` — domain class with `@register_domain`
- [ ] `domains/<name>/patterns.py` — all 9 required exports
- [ ] `domains/<name>/data/topics.json` — topic definitions array
- [ ] `domains/<name>/states.py` — state config (if state-specific)
- [ ] `domains/__init__.py` — module path added to `_discover_domains`
- [ ] `name` and `collection_name` are unique across all domains
- [ ] Tests pass: `pytest tests/test_domains.py -v`
- [ ] Linter passes: `ruff check src/insurance_rag/domains/<name>/`

## Example domains

- **Medicare** (`domains/medicare/`): Federal-level domain. No state config. Three source kinds (IOM manuals, MCD coverage determinations, HCPCS/ICD codes). Rich query patterns for LCD/NCD detection. HCPCS/ICD-10 semantic enrichment.

- **Auto Insurance** (`domains/auto/`): State-specific domain with 10 state configurations. Four source kinds (regulations, forms, claims, rates). Coverage-focused query patterns. Currently scaffolded with placeholder downloaders/extractors.
