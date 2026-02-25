#!/usr/bin/env python3
"""CLI entry point for downloading insurance data sources.

Supports domain selection via ``--domain``.  Each domain registers its
own downloaders and source kinds.
"""
import argparse
import logging
import sys

import httpx

from insurance_rag.config import domain_raw_dir
from insurance_rag.domains import get_domain, list_domains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    available = list_domains()
    parser = argparse.ArgumentParser(
        description="Download insurance RAG data sources."
    )
    parser.add_argument(
        "--domain",
        choices=available + ["all"],
        default="medicare",
        help=f"Domain to download data for (default: medicare). Available: {', '.join(available)}",
    )
    parser.add_argument(
        "--source",
        default="all",
        help="Source kind within the domain (default: all). "
        "Depends on domain; use --help after choosing a domain to see valid kinds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    domains_to_run = available if args.domain == "all" else [args.domain]

    try:
        for domain_name in domains_to_run:
            domain = get_domain(domain_name)
            raw_dir = domain_raw_dir(domain_name)
            raw_dir.mkdir(parents=True, exist_ok=True)
            downloaders = domain.get_downloaders()

            if args.source == "all":
                sources = list(downloaders.keys())
            elif args.source in downloaders:
                sources = [args.source]
            else:
                logger.error(
                    "Unknown source %r for domain %s. Available: %s",
                    args.source,
                    domain_name,
                    ", ".join(downloaders.keys()),
                )
                return 1

            for source in sources:
                logger.info(
                    "[%s] Downloading %s data...", domain.display_name, source
                )
                downloaders[source](raw_dir, force=args.force)

    except httpx.HTTPError as e:
        logger.error("HTTP error during download: %s", e)
        return 1
    except OSError as e:
        logger.error("File or I/O error during download: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during download: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
