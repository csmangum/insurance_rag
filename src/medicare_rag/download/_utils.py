"""Shared utilities for download scripts."""
from urllib.parse import urlparse

# Shared timeout for all download HTTP requests (seconds)
DOWNLOAD_TIMEOUT = 60.0


def sanitize_filename_from_url(url: str, default_basename: str) -> str:
    """Extract a safe filename from a URL (no path traversal).

    Uses only the last path segment and strips query string. Returns default_basename
    if the result would be empty or contain path traversal (e.g. "..").
    """
    path = urlparse(url).path or ""
    name = path.rstrip("/").split("/")[-1].split("?")[0].strip()
    if not name or ".." in name or "/" in name or "\\" in name:
        return default_basename
    return name
