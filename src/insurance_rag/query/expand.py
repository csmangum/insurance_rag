"""Cross-source query expansion for insurance RAG retrieval.

Detects which source types are relevant to a query and generates
additional query variants that target each source's vocabulary,
improving recall for questions that span multiple sources.

Domain-specific patterns are loaded from the active domain plugin.
"""
from __future__ import annotations

import re
from typing import Any


def _get_domain_patterns() -> (
    tuple[dict[str, list[Any]], dict[str, str], list[tuple[Any, str]], dict[str, float]]
):
    """Load source patterns, expansions, synonym map, and default relevance
    from the active domain."""
    try:
        from insurance_rag.config import DEFAULT_DOMAIN
        from insurance_rag.domains import get_domain

        domain = get_domain(DEFAULT_DOMAIN)
        return (
            domain.get_source_patterns(),
            domain.get_source_expansions(),
            domain.get_synonym_map(),
            domain.get_default_source_relevance(),
        )
    except (KeyError, ImportError):
        return {}, {}, [], {}


def detect_source_relevance(
    query: str,
    source_patterns: dict[str, list[re.Pattern[str]]] | None = None,
    default_relevance: dict[str, float] | None = None,
) -> dict[str, float]:
    """Score each source type's relevance to the query on a 0.0-1.0 scale.

    When no specific source signals are detected, returns moderate scores
    for all sources so cross-source retrieval still casts a wide net.
    """
    if source_patterns is None or default_relevance is None:
        _sp, _se, _sm, _dr = _get_domain_patterns()
        if source_patterns is None:
            source_patterns = _sp
        if default_relevance is None:
            default_relevance = _dr

    scores: dict[str, float] = {}
    for name, patterns in source_patterns.items():
        threshold = max(1, len(patterns) // 3)
        matches = sum(1 for p in patterns if p.search(query))
        scores[name] = min(1.0, matches / threshold)

    if all(v == 0 for v in scores.values()):
        return dict(default_relevance)
    return scores


def _apply_synonyms(
    query: str,
    synonym_map: list[tuple[re.Pattern[str], str]] | None = None,
) -> str:
    """Expand a query with domain synonyms."""
    if synonym_map is None:
        _, _, synonym_map, _ = _get_domain_patterns()

    additions: list[str] = []
    for pattern, expansion in synonym_map:
        if pattern.search(query):
            additions.append(expansion)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def expand_cross_source_query(
    query: str,
    source_patterns: dict[str, list[re.Pattern[str]]] | None = None,
    source_expansions: dict[str, str] | None = None,
    synonym_map: list[tuple[re.Pattern[str], str]] | None = None,
    default_relevance: dict[str, float] | None = None,
) -> list[str]:
    """Expand a query into multiple variants optimized for different sources.

    Returns a list where the first element is always the original query,
    followed by source-specific variants for each relevant source, and
    optionally a synonym-expanded variant.
    """
    if source_patterns is None or source_expansions is None or synonym_map is None:
        _sp, _se, _sm, _dr = _get_domain_patterns()
        if source_patterns is None:
            source_patterns = _sp
        if source_expansions is None:
            source_expansions = _se
        if synonym_map is None:
            synonym_map = _sm
        if default_relevance is None:
            default_relevance = _dr

    variants = [query]
    relevance = detect_source_relevance(query, source_patterns, default_relevance)

    for source, score in relevance.items():
        if score > 0 and source in source_expansions:
            expansion = source_expansions[source]
            variants.append(f"{query} {expansion}")

    synonym_expanded = _apply_synonyms(query, synonym_map)
    if synonym_expanded != query:
        variants.append(synonym_expanded)

    return variants
