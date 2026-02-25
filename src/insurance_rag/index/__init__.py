"""Embedding and vector store (Phase 3)."""
from insurance_rag.index.embed import get_embeddings
from insurance_rag.index.store import get_or_create_chroma, upsert_documents

__all__ = ["get_embeddings", "get_or_create_chroma", "upsert_documents"]
