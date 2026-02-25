"""Base class for insurance domain plugins."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class InsuranceDomain(ABC):
    """Abstract base for all insurance domain plugins.

    Each domain provides its own data sources, extractors, enrichment
    rules, topic definitions, query patterns, and system prompt.  The
    core infrastructure (embeddings, vector store, chunking, hybrid
    retrieval) is shared across all domains.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"medicare"``, ``"auto"``."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label, e.g. ``"Medicare"``, ``"Auto Insurance"``."""

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """ChromaDB collection name (unique per domain)."""

    @property
    @abstractmethod
    def source_kinds(self) -> list[str]:
        """Valid source-kind identifiers for this domain."""

    # ------------------------------------------------------------------
    # Data pipeline hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def get_downloaders(self) -> dict[str, Any]:
        """Map of source-kind -> download callable(raw_dir, *, force)."""

    @abstractmethod
    def get_extractors(self) -> dict[str, Any]:
        """Map of source-kind -> extract callable(raw_dir, processed_dir, *, force)."""

    def get_enricher(self) -> Any | None:
        """Optional enrichment callable applied during extraction."""
        return None

    # ------------------------------------------------------------------
    # Topic clustering
    # ------------------------------------------------------------------

    @abstractmethod
    def get_topic_definitions_path(self) -> Path:
        """Path to this domain's ``topics.json``."""

    # ------------------------------------------------------------------
    # Query / retrieval
    # ------------------------------------------------------------------

    @abstractmethod
    def get_query_patterns(self) -> dict[str, Any]:
        """Domain-specific query detection patterns.

        Expected keys (all optional):
            ``"specialized_query_patterns"`` — list of compiled regexes
            ``"specialized_topic_patterns"`` — list of (regex, expansion) tuples
            ``"strip_noise_pattern"`` — compiled regex for stripping domain jargon
        """

    @abstractmethod
    def get_source_patterns(self) -> dict[str, list[Any]]:
        """Map of source-kind -> list of compiled regexes for source detection."""

    @abstractmethod
    def get_source_expansions(self) -> dict[str, str]:
        """Map of source-kind -> expansion suffix string."""

    @abstractmethod
    def get_synonym_map(self) -> list[tuple[Any, str]]:
        """List of (compiled_regex, expansion_string) for domain synonyms."""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """System prompt for the RAG chain LLM."""

    def get_default_source_relevance(self) -> dict[str, float]:
        """Fallback source relevance scores when no patterns match."""
        n = len(self.source_kinds)
        if n == 0:
            return {}
        base = round(1.0 / n, 2)
        return {sk: base for sk in self.source_kinds}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def get_states(self) -> list[str] | None:
        """US state codes relevant to this domain, or None if federal."""
        return None

    def get_chunk_overrides(self) -> dict[str, dict[str, int]] | None:
        """Per-source-kind chunk size overrides.

        Return e.g. ``{"mcd": {"chunk_size": 1500, "chunk_overlap": 300}}``
        to use larger chunks for a specific source kind.
        """
        return None

    def get_quick_questions(self) -> list[str]:
        """Example questions for the Streamlit UI."""
        return []
