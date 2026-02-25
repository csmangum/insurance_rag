"""Medicare insurance domain plugin."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from insurance_rag.domains import register_domain
from insurance_rag.domains.base import InsuranceDomain


@register_domain
class MedicareDomain(InsuranceDomain):
    """Medicare Revenue Cycle Management domain.

    Covers IOM manuals, MCD (LCD/NCD) coverage determinations, and
    HCPCS / ICD-10-CM code files.  Operates at the federal level
    (no state-specific partitioning).
    """

    @property
    def name(self) -> str:
        return "medicare"

    @property
    def display_name(self) -> str:
        return "Medicare"

    @property
    def collection_name(self) -> str:
        return "medicare"

    @property
    def source_kinds(self) -> list[str]:
        return ["iom", "mcd", "codes"]

    def get_downloaders(self) -> dict[str, Any]:
        from insurance_rag.download.codes import download_codes
        from insurance_rag.download.iom import download_iom
        from insurance_rag.download.mcd import download_mcd

        return {"iom": download_iom, "mcd": download_mcd, "codes": download_codes}

    def get_extractors(self) -> dict[str, Any]:
        from insurance_rag.ingest.extract import (
            extract_hcpcs,
            extract_icd10cm,
            extract_iom,
            extract_mcd,
        )

        def _extract_codes(
            processed_dir: Path, raw_dir: Path, *, force: bool = False
        ) -> list[tuple[Path, Path]]:
            return extract_hcpcs(processed_dir, raw_dir, force=force) + extract_icd10cm(
                processed_dir, raw_dir, force=force
            )

        return {"iom": extract_iom, "mcd": extract_mcd, "codes": _extract_codes}

    def get_enricher(self) -> Any | None:
        from insurance_rag.ingest.enrich import enrich_hcpcs_text, enrich_icd10_text

        return {"hcpcs": enrich_hcpcs_text, "icd10": enrich_icd10_text}

    def get_topic_definitions_path(self) -> Path:
        return Path(__file__).parent / "data" / "topics.json"

    def get_query_patterns(self) -> dict[str, Any]:
        from insurance_rag.domains.medicare.patterns import (
            LCD_QUERY_PATTERNS,
            LCD_TOPIC_PATTERNS,
            STRIP_FILLER,
            STRIP_LCD_NOISE,
        )

        return {
            "specialized_query_patterns": LCD_QUERY_PATTERNS,
            "specialized_topic_patterns": LCD_TOPIC_PATTERNS,
            "strip_noise_pattern": STRIP_LCD_NOISE,
            "strip_filler_pattern": STRIP_FILLER,
        }

    def get_source_patterns(self) -> dict[str, list[Any]]:
        from insurance_rag.domains.medicare.patterns import SOURCE_PATTERNS

        return SOURCE_PATTERNS

    def get_source_expansions(self) -> dict[str, str]:
        from insurance_rag.domains.medicare.patterns import SOURCE_EXPANSIONS

        return SOURCE_EXPANSIONS

    def get_synonym_map(self) -> list[tuple[Any, str]]:
        from insurance_rag.domains.medicare.patterns import SYNONYM_MAP

        return SYNONYM_MAP

    def get_system_prompt(self) -> str:
        from insurance_rag.domains.medicare.patterns import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def get_default_source_relevance(self) -> dict[str, float]:
        from insurance_rag.domains.medicare.patterns import DEFAULT_SOURCE_RELEVANCE

        return DEFAULT_SOURCE_RELEVANCE

    def get_states(self) -> list[str] | None:
        return None

    def get_chunk_overrides(self) -> dict[str, dict[str, int]] | None:
        from insurance_rag.config import LCD_CHUNK_OVERLAP, LCD_CHUNK_SIZE

        return {"mcd": {"chunk_size": LCD_CHUNK_SIZE, "chunk_overlap": LCD_CHUNK_OVERLAP}}

    def get_quick_questions(self) -> list[str]:
        from insurance_rag.domains.medicare.patterns import QUICK_QUESTIONS

        return QUICK_QUESTIONS

    def get_specialized_source_filter(self) -> dict[str, str] | None:
        return {"source": "mcd"}
