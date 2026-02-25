"""Retrieval and RAG chain (Phase 4).

Submodules:
    retriever — Domain-aware retriever with query expansion and topic-summary boosting.
    hybrid    — Hybrid retriever (semantic + BM25, RRF, cross-source diversification).
    expand    — Cross-source query expansion with domain-configurable synonym mapping.
    chain     — RAG chain wiring (retriever + local LLM + domain-configurable prompt).
"""

from insurance_rag.query import expand, hybrid
