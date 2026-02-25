#!/usr/bin/env python3
"""CLI entry point for extraction, chunking, and indexing.

Supports domain selection via ``--domain``.  Each domain provides
extractors and an optional enricher.
"""
import argparse
import logging
import sys

from insurance_rag.config import domain_processed_dir, domain_raw_dir
from insurance_rag.domains import get_domain, list_domains
from insurance_rag.ingest.chunk import chunk_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    available = list_domains()
    parser = argparse.ArgumentParser(
        description="Extract, chunk, and index insurance data."
    )
    parser.add_argument(
        "--domain",
        choices=available + ["all"],
        default="medicare",
        help=f"Domain to ingest (default: medicare). Available: {', '.join(available)}",
    )
    parser.add_argument(
        "--source",
        default="all",
        help="Source kind within the domain (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output files exist",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; only run chunking on existing processed dir",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip embed and vector store; only run extract and chunk",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Disable topic clustering and summary generation",
    )
    args = parser.parse_args()

    domains_to_run = available if args.domain == "all" else [args.domain]

    try:
        for domain_name in domains_to_run:
            domain = get_domain(domain_name)
            raw_dir = domain_raw_dir(domain_name)
            processed_dir = domain_processed_dir(domain_name)
            processed_dir.mkdir(parents=True, exist_ok=True)

            extractors = domain.get_extractors()
            if args.source == "all":
                sources = list(extractors.keys())
            elif args.source in extractors:
                sources = [args.source]
            else:
                logger.error(
                    "Unknown source %r for domain %s. Available: %s",
                    args.source,
                    domain_name,
                    ", ".join(extractors.keys()),
                )
                return 1

            if not args.skip_extract:
                total_written = 0
                for source in sources:
                    logger.info(
                        "[%s] Extracting %s...", domain.display_name, source
                    )
                    written = extractors[source](
                        processed_dir, raw_dir, force=args.force
                    )
                    total_written += len(written)
                logger.info(
                    "[%s] Extraction: %d documents",
                    domain.display_name,
                    total_written,
                )
            else:
                logger.info("[%s] Skipping extraction (--skip-extract)", domain.display_name)

            docs = chunk_documents(
                processed_dir,
                source=args.source if args.source != "all" else "all",
                enable_summaries=not args.no_summaries,
            )
            logger.info("[%s] Chunking: %d chunks", domain.display_name, len(docs))

            if not args.skip_index:
                from insurance_rag.index import (
                    get_embeddings,
                    get_or_create_chroma,
                    upsert_documents,
                )

                embeddings = get_embeddings()
                store = get_or_create_chroma(
                    embeddings, collection_name=domain.collection_name
                )
                n_upserted, n_skipped = upsert_documents(store, docs, embeddings)
                logger.info(
                    "[%s] Indexed %d new/updated, %d unchanged",
                    domain.display_name,
                    n_upserted,
                    n_skipped,
                )

    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
