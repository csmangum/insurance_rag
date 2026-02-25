#!/usr/bin/env python3
"""Interactive REPL for Insurance RAG queries.

Usage:
  python scripts/query.py
  python scripts/query.py --domain auto
  python scripts/query.py --domain medicare --filter-source iom
"""
import argparse
import sys
from pathlib import Path

from insurance_rag.config import CHROMA_DIR
from insurance_rag.domains import get_domain, list_domains
from insurance_rag.query.chain import build_rag_chain

try:
    import readline

    _HISTORY_PATH = Path.home() / ".insurance_rag_query_history"
    _READLINE_AVAILABLE = True
except ImportError:
    _READLINE_AVAILABLE = False
    _HISTORY_PATH = None


SOURCE_META_KEYS = ("source", "manual", "chapter", "doc_id", "jurisdiction", "title", "state")


def _check_index_has_docs(domain_name: str) -> bool:
    try:
        from insurance_rag.index import get_embeddings, get_or_create_chroma

        domain = get_domain(domain_name)
        embeddings = get_embeddings()
        store = get_or_create_chroma(embeddings, collection_name=domain.collection_name)
        n = store._collection.count()
        return n > 0
    except Exception:
        return False


def main() -> int:
    available = list_domains()
    parser = argparse.ArgumentParser(description="Insurance RAG query REPL")
    parser.add_argument(
        "--domain",
        choices=available,
        default="medicare",
        help=f"Domain to query (default: medicare). Available: {', '.join(available)}",
    )
    parser.add_argument(
        "--filter-source", type=str, help="Filter by source kind"
    )
    parser.add_argument(
        "--filter-manual", type=str, help="Filter by manual (Medicare)"
    )
    parser.add_argument(
        "--filter-jurisdiction", type=str, help="Filter by jurisdiction"
    )
    parser.add_argument(
        "--filter-state", type=str, help="Filter by state code (e.g. CA, FL)"
    )
    parser.add_argument(
        "-k", type=int, default=8, help="Number of chunks to retrieve (default 8)"
    )
    args = parser.parse_args()

    domain = get_domain(args.domain)

    metadata_filter: dict | None = None
    filter_args = {
        "source": args.filter_source,
        "manual": args.filter_manual,
        "jurisdiction": args.filter_jurisdiction,
        "state": args.filter_state,
    }
    active_filters = {k: v for k, v in filter_args.items() if v}
    if active_filters:
        metadata_filter = active_filters

    if not CHROMA_DIR.exists():
        print(
            f"Error: Chroma index not found at {CHROMA_DIR}. "
            "Run ingestion first (scripts/ingest_all.py).",
            file=sys.stderr,
        )
        return 1

    if not _check_index_has_docs(args.domain):
        print(
            f"Error: Collection for {domain.display_name} is empty. "
            f"Run ingestion first: python scripts/ingest_all.py --domain {args.domain}",
            file=sys.stderr,
        )
        return 1

    if _READLINE_AVAILABLE and _HISTORY_PATH is not None:
        try:
            readline.read_history_file(_HISTORY_PATH)
        except OSError:
            pass
        try:
            readline.set_history_length(500)
        except (AttributeError, TypeError):
            pass

    print(f"{domain.display_name} RAG query (blank line to quit)")
    print("---")

    from insurance_rag.index import get_embeddings, get_or_create_chroma

    embeddings = get_embeddings()
    store = get_or_create_chroma(embeddings, collection_name=domain.collection_name)

    chain = build_rag_chain(
        k=args.k,
        metadata_filter=metadata_filter,
        domain_name=args.domain,
        store=store,
        embeddings=embeddings,
        system_prompt=domain.get_system_prompt(),
    )

    try:
        _repl_loop(chain)
    finally:
        if _READLINE_AVAILABLE and _HISTORY_PATH is not None:
            try:
                readline.write_history_file(_HISTORY_PATH)
            except OSError:
                pass

    print("Bye.")
    return 0


def _repl_loop(chain) -> None:
    while True:
        try:
            question = input("Question (blank to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        try:
            result = chain({"question": question})
            answer = result["answer"]
            source_docs = result["source_documents"]
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        print()
        print(answer)
        print()
        print("Sources:")
        for i, doc in enumerate(source_docs, start=1):
            meta = doc.metadata
            parts = [f"  [{i}]"]
            for key in SOURCE_META_KEYS:
                if key in meta and meta[key] is not None:
                    parts.append(f"{key}={meta[key]}")
            print(" ".join(parts))
        print("---")


if __name__ == "__main__":
    sys.exit(main())
