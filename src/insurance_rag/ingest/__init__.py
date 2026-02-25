"""Extraction and chunking (Phase 2)."""
from typing import Literal

# Legacy Medicare source kinds (kept for backward compatibility).
# New domains define their own source kinds via InsuranceDomain.source_kinds.
SourceKind = Literal["iom", "mcd", "codes", "all"]
