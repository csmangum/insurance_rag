"""Topic clustering for fragmented insurance content.

Groups chunks by clinical/policy topic so that related content scattered
across source documents can be consolidated into topic-level summaries.
This improves retrieval stability when users rephrase the same question
in different ways.

Each topic is defined by a set of keyword patterns.  A chunk may belong
to multiple topics.

Topic definitions are loaded from the active domain's ``topics.json``
when available; otherwise from ``DATA_DIR/topic_definitions.json`` or the
package default (``insurance_rag/data/topic_definitions.json``).
"""

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from insurance_rag.config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicDef:
    """Immutable definition of a topic cluster."""

    name: str
    label: str
    patterns: tuple[re.Pattern[str], ...]
    summary_prefix: str = ""
    min_pattern_matches: int = 1


def _compile(raw: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in raw)


def _load_topic_definitions() -> list[TopicDef]:
    """Load topic definitions from the active domain, DATA_DIR, or package default."""
    raw: str | None = None

    # 1. Try the active domain's topic definitions
    try:
        from insurance_rag.config import DEFAULT_DOMAIN
        from insurance_rag.domains import get_domain

        domain = get_domain(DEFAULT_DOMAIN)
        domain_path = domain.get_topic_definitions_path()
        if domain_path.exists():
            raw = domain_path.read_text(encoding="utf-8")
    except (KeyError, ImportError, OSError) as e:
        logger.debug("Domain topic definitions not available: %s", e)

    # 2. Fallback: DATA_DIR/topic_definitions.json
    if raw is None:
        path = DATA_DIR / "topic_definitions.json"
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Could not read %s: %s; using package default", path, e)

    # 3. Fallback: package default
    if raw is None:
        from importlib.resources import files

        pkg_path = files("insurance_rag") / "data" / "topic_definitions.json"
        try:
            raw = pkg_path.read_text(encoding="utf-8")
        except Exception as e:
            raise FileNotFoundError(
                f"Topic definitions not found in domain, DATA_DIR, or package default: {e}"
            ) from e

    data = json.loads(raw)
    out: list[TopicDef] = []
    for item in data:
        name = item.get("name", "")
        label = item.get("label", name)
        patterns_raw = item.get("patterns") or []
        summary_prefix = item.get("summary_prefix", "")
        min_pattern_matches = max(1, int(item.get("min_pattern_matches", 1)))
        out.append(
            TopicDef(
                name=name,
                label=label,
                patterns=_compile(patterns_raw),
                summary_prefix=summary_prefix,
                min_pattern_matches=min_pattern_matches,
            )
        )
    return out


TOPIC_DEFINITIONS: list[TopicDef] = _load_topic_definitions()
_TOPIC_DEF_MAP: dict[str, TopicDef] = {td.name: td for td in TOPIC_DEFINITIONS}


def assign_topics(doc: Document) -> list[str]:
    """Return the list of topic names that match the document content."""
    text = doc.page_content
    topics: list[str] = []
    for topic_def in TOPIC_DEFINITIONS:
        matches = sum(1 for p in topic_def.patterns if p.search(text))
        if matches >= topic_def.min_pattern_matches:
            topics.append(topic_def.name)
    return topics


def cluster_documents(documents: list[Document]) -> dict[str, list[Document]]:
    """Group documents by topic cluster.

    Returns a mapping from topic name to the list of documents that
    belong to that cluster.  Documents may appear in multiple clusters.
    """
    clusters: dict[str, list[Document]] = {}
    for doc in documents:
        topics = assign_topics(doc)
        for topic in topics:
            clusters.setdefault(topic, []).append(doc)
    return clusters


def get_topic_def(name: str) -> TopicDef | None:
    """Look up a topic definition by name."""
    return _TOPIC_DEF_MAP.get(name)


def tag_documents_with_topics(documents: list[Document]) -> list[Document]:
    """Add ``topic_clusters`` metadata to each document.

    Returns new Document instances (original list is not mutated).
    """
    tagged: list[Document] = []
    for doc in documents:
        topics = assign_topics(doc)
        if topics:
            meta = dict(doc.metadata)
            meta["topic_clusters"] = ",".join(topics)
            tagged.append(Document(page_content=doc.page_content, metadata=meta))
        else:
            tagged.append(doc)
    return tagged
