"""Auto insurance domain plugin."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from insurance_rag.domains import register_domain
from insurance_rag.domains.base import InsuranceDomain

logger = logging.getLogger(__name__)


@register_domain
class AutoInsuranceDomain(InsuranceDomain):
    """US auto insurance domain.

    Covers state regulations, policy forms, claims handling,
    and rate filing data for the top US auto insurance markets.
    """

    @property
    def name(self) -> str:
        return "auto"

    @property
    def display_name(self) -> str:
        return "Auto Insurance"

    @property
    def collection_name(self) -> str:
        return "auto_insurance"

    @property
    def source_kinds(self) -> list[str]:
        return ["regulations", "forms", "claims", "rates"]

    def get_downloaders(self) -> dict[str, Any]:
        def _download_placeholder(raw_dir: Path, *, force: bool = False) -> None:
            logger.info(
                "Auto insurance downloader placeholder — "
                "implement data source connectors for state DOI regulations, "
                "NAIC model laws, ISO forms, and rate filings."
            )
            raw_dir.mkdir(parents=True, exist_ok=True)

        return {sk: _download_placeholder for sk in self.source_kinds}

    def get_extractors(self) -> dict[str, Any]:
        def _extract_placeholder(
            processed_dir: Path, raw_dir: Path, *, force: bool = False
        ) -> list[tuple[Path, Path]]:
            logger.info(
                "Auto insurance extractor placeholder — "
                "implement parsing for auto insurance document formats."
            )
            return []

        return {sk: _extract_placeholder for sk in self.source_kinds}

    def get_enricher(self) -> Any | None:
        return None

    def get_topic_definitions_path(self) -> Path:
        return Path(__file__).parent / "data" / "topics.json"

    def get_query_patterns(self) -> dict[str, Any]:
        from insurance_rag.domains.auto.patterns import (
            COVERAGE_QUERY_PATTERNS,
            COVERAGE_TOPIC_PATTERNS,
            STRIP_COVERAGE_NOISE,
            STRIP_FILLER,
        )

        return {
            "specialized_query_patterns": COVERAGE_QUERY_PATTERNS,
            "specialized_topic_patterns": COVERAGE_TOPIC_PATTERNS,
            "strip_noise_pattern": STRIP_COVERAGE_NOISE,
            "strip_filler_pattern": STRIP_FILLER,
        }

    def get_source_patterns(self) -> dict[str, list[Any]]:
        from insurance_rag.domains.auto.patterns import SOURCE_PATTERNS

        return SOURCE_PATTERNS

    def get_source_expansions(self) -> dict[str, str]:
        from insurance_rag.domains.auto.patterns import SOURCE_EXPANSIONS

        return SOURCE_EXPANSIONS

    def get_synonym_map(self) -> list[tuple[Any, str]]:
        from insurance_rag.domains.auto.patterns import SYNONYM_MAP

        return SYNONYM_MAP

    def get_system_prompt(self) -> str:
        from insurance_rag.domains.auto.patterns import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def get_default_source_relevance(self) -> dict[str, float]:
        from insurance_rag.domains.auto.patterns import DEFAULT_SOURCE_RELEVANCE

        return DEFAULT_SOURCE_RELEVANCE

    def get_states(self) -> list[str]:
        from insurance_rag.domains.auto.states import TOP_MARKETS

        return TOP_MARKETS

    def get_chunk_overrides(self) -> dict[str, dict[str, int]] | None:
        return {"regulations": {"chunk_size": 1500, "chunk_overlap": 300}}

    def get_quick_questions(self) -> list[str]:
        from insurance_rag.domains.auto.patterns import QUICK_QUESTIONS

        return QUICK_QUESTIONS

    def get_specialized_source_filter(self) -> dict[str, str] | None:
        return {"source": "regulations"}
