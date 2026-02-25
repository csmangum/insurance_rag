"""Tests for the domain plugin system."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from insurance_rag.domains import get_domain, list_domains
from insurance_rag.domains.base import InsuranceDomain


class TestDomainRegistry:
    """Domain registration and lookup."""

    def test_list_domains_includes_medicare(self):
        assert "medicare" in list_domains()

    def test_list_domains_includes_auto(self):
        assert "auto" in list_domains()

    def test_get_domain_returns_instance(self):
        domain = get_domain("medicare")
        assert isinstance(domain, InsuranceDomain)

    def test_get_unknown_domain_raises(self):
        with pytest.raises(KeyError, match="Unknown domain"):
            get_domain("nonexistent_domain_xyz")

    def test_list_domains_is_sorted(self):
        domains = list_domains()
        assert domains == sorted(domains)


class TestInsuranceDomainInterface:
    """Ensure all registered domains implement the full interface."""

    @pytest.fixture(params=["medicare", "auto"])
    def domain(self, request) -> InsuranceDomain:
        return get_domain(request.param)

    def test_name_is_string(self, domain: InsuranceDomain):
        assert isinstance(domain.name, str)
        assert len(domain.name) > 0

    def test_display_name_is_string(self, domain: InsuranceDomain):
        assert isinstance(domain.display_name, str)
        assert len(domain.display_name) > 0

    def test_collection_name_is_string(self, domain: InsuranceDomain):
        assert isinstance(domain.collection_name, str)
        assert len(domain.collection_name) > 0

    def test_source_kinds_is_list(self, domain: InsuranceDomain):
        kinds = domain.source_kinds
        assert isinstance(kinds, list)
        assert len(kinds) > 0
        assert all(isinstance(k, str) for k in kinds)

    def test_get_downloaders_returns_dict(self, domain: InsuranceDomain):
        downloaders = domain.get_downloaders()
        assert isinstance(downloaders, dict)
        assert len(downloaders) > 0
        for key in downloaders:
            assert key in domain.source_kinds

    def test_get_extractors_returns_dict(self, domain: InsuranceDomain):
        extractors = domain.get_extractors()
        assert isinstance(extractors, dict)
        assert len(extractors) > 0
        for key in extractors:
            assert key in domain.source_kinds

    def test_get_topic_definitions_path(self, domain: InsuranceDomain):
        path = domain.get_topic_definitions_path()
        assert isinstance(path, Path)
        assert path.exists(), f"Topics file not found: {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0
        for topic in data:
            assert "name" in topic
            assert "patterns" in topic

    def test_get_query_patterns(self, domain: InsuranceDomain):
        patterns = domain.get_query_patterns()
        assert isinstance(patterns, dict)
        assert "specialized_query_patterns" in patterns
        assert isinstance(patterns["specialized_query_patterns"], list)
        assert all(
            isinstance(p, re.Pattern) for p in patterns["specialized_query_patterns"]
        )

    def test_get_source_patterns(self, domain: InsuranceDomain):
        sp = domain.get_source_patterns()
        assert isinstance(sp, dict)
        for _key, patterns in sp.items():
            assert isinstance(patterns, list)
            assert all(isinstance(p, re.Pattern) for p in patterns)

    def test_get_source_expansions(self, domain: InsuranceDomain):
        se = domain.get_source_expansions()
        assert isinstance(se, dict)
        for _key, expansion in se.items():
            assert isinstance(expansion, str)
            assert len(expansion) > 0

    def test_get_synonym_map(self, domain: InsuranceDomain):
        sm = domain.get_synonym_map()
        assert isinstance(sm, list)
        for pattern, expansion in sm:
            assert isinstance(pattern, re.Pattern)
            assert isinstance(expansion, str)

    def test_get_system_prompt(self, domain: InsuranceDomain):
        prompt = domain.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_get_default_source_relevance(self, domain: InsuranceDomain):
        relevance = domain.get_default_source_relevance()
        assert isinstance(relevance, dict)
        assert len(relevance) > 0
        assert all(isinstance(v, float) for v in relevance.values())

    def test_get_quick_questions(self, domain: InsuranceDomain):
        questions = domain.get_quick_questions()
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

    def test_get_specialized_source_filter(self, domain: InsuranceDomain):
        result = domain.get_specialized_source_filter()
        if result is not None:
            assert isinstance(result, dict)
            assert "source" in result
            assert result["source"] in domain.source_kinds


class TestMedicareDomain:
    """Medicare-specific domain tests."""

    def test_medicare_collection_name(self):
        domain = get_domain("medicare")
        assert domain.collection_name == "medicare"

    def test_medicare_source_kinds(self):
        domain = get_domain("medicare")
        assert set(domain.source_kinds) == {"iom", "mcd", "codes"}

    def test_medicare_has_no_states(self):
        domain = get_domain("medicare")
        assert domain.get_states() is None

    def test_medicare_has_lcd_chunk_overrides(self):
        domain = get_domain("medicare")
        overrides = domain.get_chunk_overrides()
        assert overrides is not None
        assert "mcd" in overrides
        assert "chunk_size" in overrides["mcd"]

    def test_medicare_enricher_present(self):
        domain = get_domain("medicare")
        enricher = domain.get_enricher()
        assert enricher is not None

    def test_medicare_system_prompt_mentions_medicare(self):
        domain = get_domain("medicare")
        assert "Medicare" in domain.get_system_prompt()


class TestAutoDomain:
    """Auto insurance domain tests."""

    def test_auto_collection_name(self):
        domain = get_domain("auto")
        assert domain.collection_name == "auto_insurance"

    def test_auto_source_kinds(self):
        domain = get_domain("auto")
        assert set(domain.source_kinds) == {"regulations", "forms", "claims", "rates"}

    def test_auto_has_states(self):
        domain = get_domain("auto")
        states = domain.get_states()
        assert states is not None
        assert "CA" in states
        assert "FL" in states
        assert "NY" in states

    def test_auto_has_regulation_chunk_overrides(self):
        domain = get_domain("auto")
        overrides = domain.get_chunk_overrides()
        assert overrides is not None
        assert "regulations" in overrides

    def test_auto_system_prompt_mentions_auto(self):
        domain = get_domain("auto")
        assert "auto" in domain.get_system_prompt().lower()

    def test_auto_state_configs(self):
        from insurance_rag.domains.auto.states import STATE_CONFIGS, TOP_MARKETS

        for state_code in TOP_MARKETS:
            assert state_code in STATE_CONFIGS
            config = STATE_CONFIGS[state_code]
            assert config.code == state_code
            assert config.tort_system in ("tort", "no-fault", "choice")
            assert config.min_liability  # non-empty string

    def test_auto_topics_valid_json(self):
        domain = get_domain("auto")
        path = domain.get_topic_definitions_path()
        data = json.loads(path.read_text(encoding="utf-8"))
        topic_names = [t["name"] for t in data]
        assert "liability_coverage" in topic_names
        assert "um_uim" in topic_names
        assert "claims_handling" in topic_names


class TestDomainIsolation:
    """Verify domains are independent."""

    def test_collection_names_unique(self):
        domains = [get_domain(name) for name in list_domains()]
        names = [d.collection_name for d in domains]
        assert len(names) == len(set(names)), "Duplicate collection names across domains"

    def test_domain_names_unique(self):
        domains = [get_domain(name) for name in list_domains()]
        names = [d.name for d in domains]
        assert len(names) == len(set(names))
