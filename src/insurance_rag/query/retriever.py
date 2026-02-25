"""Domain-aware VectorStoreRetriever (Phase 4).

Provides a retriever that detects domain-specific specialized queries
(e.g. LCD/coverage-determination queries in Medicare) and applies query
expansion plus source-filtered multi-query retrieval.

Summary documents (``doc_type`` = ``document_summary`` or ``topic_summary``)
are boosted in retrieval results to provide stable "anchor" chunks that
match consistently regardless of query phrasing.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from insurance_rag.config import LCD_RETRIEVAL_K
from insurance_rag.index import get_embeddings, get_or_create_chroma
from insurance_rag.index.store import get_raw_collection

logger = logging.getLogger(__name__)


def _get_domain_query_patterns() -> dict[str, Any]:
    """Load specialized query patterns from the active domain."""
    try:
        from insurance_rag.config import DEFAULT_DOMAIN
        from insurance_rag.domains import get_domain

        return get_domain(DEFAULT_DOMAIN).get_query_patterns()
    except (KeyError, ImportError):
        return {}


def is_lcd_query(query: str) -> bool:
    """Return True if the query matches the active domain's specialized query patterns.

    For Medicare, this detects LCD/coverage-determination queries.
    Kept as ``is_lcd_query`` for backward compatibility.
    """
    patterns = _get_domain_query_patterns().get("specialized_query_patterns", [])
    return any(p.search(query) for p in patterns)


def expand_lcd_query(query: str) -> list[str]:
    """Return a list of expanded/reformulated queries for specialized retrieval.

    Produces up to three variants:
      1. The original query (unchanged).
      2. Original + topic-specific expansion terms.
      3. A stripped concept query (domain jargon removed) so the embedding
         focuses on the core topic.
    """
    domain_patterns = _get_domain_query_patterns()
    topic_patterns = domain_patterns.get("specialized_topic_patterns", [])
    strip_noise = domain_patterns.get("strip_noise_pattern")
    strip_filler = domain_patterns.get("strip_filler_pattern")

    queries = [query]

    topic_expansions = [exp for pat, exp in topic_patterns if pat.search(query)]

    if topic_expansions:
        queries.append(f"{query} {' '.join(topic_expansions)}")
    else:
        queries.append(
            f"{query} Local Coverage Determination LCD policy coverage criteria"
        )

    concept = _strip_to_concept(query, strip_noise, strip_filler)
    if concept and concept.lower() != query.lower():
        queries.append(concept)

    return queries


def _strip_to_medical_concept(query: str) -> str:
    """Backward-compatible wrapper for _strip_to_concept using domain patterns."""
    patterns = _get_domain_query_patterns()
    return _strip_to_concept(
        query,
        patterns.get("strip_noise_pattern"),
        patterns.get("strip_filler_pattern"),
    )


def _strip_to_concept(
    query: str,
    strip_noise: re.Pattern[str] | None,
    strip_filler: re.Pattern[str] | None,
) -> str:
    """Remove domain jargon and filler words to isolate the core concept."""
    cleaned = query
    if strip_noise:
        cleaned = strip_noise.sub("", cleaned)
    if strip_filler:
        cleaned = strip_filler.sub("", cleaned)
    cleaned = re.sub(r"[()]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ?.,;:")
    return cleaned


def detect_query_topics(query: str) -> list[str]:
    """Return the list of topic cluster names relevant to the query."""
    from insurance_rag.ingest.cluster import assign_topics

    return assign_topics(Document(page_content=query, metadata={}))


def boost_summaries(
    docs: list[Document],
    query_topics: list[str],
    max_k: int,
) -> list[Document]:
    """Re-rank *docs* so that topic/document summaries matching the query
    topics appear near the top of the result list.

    Summary documents act as stable anchors: they consolidate fragmented
    content and match consistently regardless of how the question is phrased.
    """
    if not query_topics or not docs:
        return docs[:max_k]

    topic_set = set(query_topics)
    boosted: list[Document] = []
    rest: list[Document] = []

    for doc in docs:
        doc_type = doc.metadata.get("doc_type", "")
        topic_cluster = doc.metadata.get("topic_cluster", "")
        topic_clusters = doc.metadata.get("topic_clusters", "")

        is_relevant_summary = False
        if doc_type in ("topic_summary", "document_summary"):
            if topic_cluster and topic_cluster in topic_set:
                is_relevant_summary = True
            elif topic_clusters:
                doc_topics = set(topic_clusters.split(","))
                if doc_topics & topic_set:
                    is_relevant_summary = True

        if is_relevant_summary:
            boosted.append(doc)
        else:
            rest.append(doc)

    return (boosted + rest)[:max_k]


def inject_topic_summaries(
    store: Any,
    docs: list[Document],
    query_topics: list[str],
    max_k: int,
) -> list[Document]:
    """Prepend topic summary docs for detected topics when not already in docs.

    Ensures stable anchor docs are always present in the candidate set before
    boosting, fixing the fragmented content consistency gap when topic summaries
    don't rank in top-k by similarity.
    """
    if not query_topics:
        return docs[:max_k]

    ids = [f"topic_{t}" for t in query_topics]
    collection = get_raw_collection(store)
    result = collection.get(ids=ids, include=["documents", "metadatas"])

    returned_ids = result.get("ids") or []
    texts = result.get("documents") or []
    metas = result.get("metadatas") or []

    injected: list[Document] = []
    for i, _cid in enumerate(returned_ids):
        text = texts[i] if i < len(texts) else ""
        meta = (metas[i] if i < len(metas) else None) or {}
        injected.append(Document(page_content=text or "", metadata=dict(meta)))

    existing_ids = {d.metadata.get("doc_id", "") for d in docs}
    new_injected = [d for d in injected if d.metadata.get("doc_id", "") not in existing_ids]
    if new_injected:
        logger.debug(
            "Injected %d topic summaries for query topics: %s",
            len(new_injected),
            ", ".join(query_topics),
        )
    combined = new_injected + docs
    return combined[:max_k]


def apply_topic_summary_boost(
    store: Any,
    docs: list[Document],
    query: str,
    max_k: int,
) -> list[Document]:
    """Run topic detection, inject topic summaries if needed, boost them,
    return up to max_k docs."""
    query_topics = detect_query_topics(query)
    if query_topics:
        docs = inject_topic_summaries(store, docs, query_topics, max_k)
        docs = boost_summaries(docs, query_topics, max_k)
    return docs[:max_k]


def _deduplicate_docs(
    doc_lists: list[list[Document]],
    max_k: int,
) -> list[Document]:
    """Merge doc lists via round-robin interleaving, deduplicating by
    doc_id+chunk_index."""
    seen: set[str] = set()
    merged: list[Document] = []
    max_len = max((len(dl) for dl in doc_lists), default=0)
    for pos in range(max_len):
        for dl in doc_lists:
            if pos >= len(dl):
                continue
            doc = dl[pos]
            key = (
                f"{doc.metadata.get('doc_id', '')}"
                f"\x00{doc.metadata.get('chunk_index', 0)}"
            )
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                if len(merged) >= max_k:
                    return merged
    return merged


class LCDAwareRetriever(BaseRetriever):
    """Retriever that boosts specialized retrieval via query expansion
    and source-filtered search.

    For non-specialized queries, delegates to standard similarity search.
    For specialized queries (e.g. LCD in Medicare), runs multi-variant
    source-filtered searches and merges results.
    """

    model_config = {"arbitrary_types_allowed": True}

    store: Any
    k: int = 8
    lcd_k: int = LCD_RETRIEVAL_K
    metadata_filter: dict | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if is_lcd_query(query):
            return self._lcd_retrieve(query)
        search_kwargs: dict = {"k": self.k}
        if self.metadata_filter is not None:
            search_kwargs["filter"] = self.metadata_filter
        docs = self.store.similarity_search(query, **search_kwargs)
        docs = apply_topic_summary_boost(self.store, docs, query, self.k)
        return docs

    def _lcd_retrieve(self, query: str) -> list[Document]:
        if (
            self.metadata_filter is not None
            and self.metadata_filter.get("source") not in (None, "mcd")
        ):
            search_kwargs = {"k": self.k, "filter": self.metadata_filter}
            return self.store.similarity_search(query, **search_kwargs)

        mcd_filter = {"source": "mcd"}
        if self.metadata_filter is not None:
            mcd_filter = {**self.metadata_filter, "source": "mcd"}

        per_variant = max(4, self.lcd_k // 3)

        mcd_docs = self.store.similarity_search(
            query, k=per_variant, filter=mcd_filter
        )

        expanded_queries = expand_lcd_query(query)
        variant_results: list[list[Document]] = []
        for eq in expanded_queries[1:]:
            variant_results.append(
                self.store.similarity_search(
                    eq, k=per_variant, filter=mcd_filter,
                )
            )

        base_kwargs: dict = {"k": per_variant}
        if self.metadata_filter is not None:
            base_kwargs["filter"] = self.metadata_filter
        base_docs = self.store.similarity_search(query, **base_kwargs)

        doc_lists = [mcd_docs] + variant_results + [base_docs]
        merged = _deduplicate_docs(doc_lists, max_k=self.lcd_k)
        merged = apply_topic_summary_boost(self.store, merged, query, self.lcd_k)
        return merged


def get_retriever(
    k: int = 8,
    metadata_filter: dict | None = None,
    embeddings: Any = None,
    store: Any = None,
) -> BaseRetriever:
    """Return a hybrid retriever combining semantic and keyword search.

    Falls back to :class:`LCDAwareRetriever` when ``rank-bm25`` is
    unavailable.
    """
    try:
        from insurance_rag.query.hybrid import get_hybrid_retriever

        return get_hybrid_retriever(
            k=k, metadata_filter=metadata_filter, embeddings=embeddings, store=store
        )
    except ImportError:
        pass

    if embeddings is None:
        embeddings = get_embeddings()
    if store is None:
        store = get_or_create_chroma(embeddings)
    return LCDAwareRetriever(
        store=store,
        k=k,
        lcd_k=max(k, LCD_RETRIEVAL_K),
        metadata_filter=metadata_filter,
    )
