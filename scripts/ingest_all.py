#!/usr/bin/env python3
"""CLI entry point for extraction and chunking (Phase 2). Optionally embed+store in Phase 3."""
import argparse
import logging
import sys

from medicare_rag.config import PROCESSED_DIR, RAW_DIR
from medicare_rag.ingest.chunk import chunk_documents
from medicare_rag.ingest.extract import extract_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCES = ("iom", "mcd", "codes", "all")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and chunk Medicare RAG data (Phase 2)."
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="all",
        help="Source to process: iom, mcd, codes, or all (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output .txt + .meta.json exist",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; only run chunking on existing processed dir",
    )
    args = parser.parse_args()

    processed_dir = PROCESSED_DIR
    raw_dir = RAW_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_extract:
            written = extract_all(processed_dir, raw_dir, source=args.source, force=args.force)
            logger.info("Extraction: %d documents available", len(written))
        else:
            logger.info("Skipping extraction (--skip-extract)")

        docs = chunk_documents(processed_dir, source=args.source)
        logger.info("Chunking: %d chunks produced", len(docs))
        print(f"Documents (chunks): {len(docs)}")
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
