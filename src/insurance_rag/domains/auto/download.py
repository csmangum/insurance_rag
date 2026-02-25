"""Auto insurance data downloaders: NAIC model laws, state consumer guides, claims, rates."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from insurance_rag.download._manifest import file_sha256, write_manifest
from insurance_rag.download._utils import (
    DOWNLOAD_TIMEOUT,
    sanitize_filename_from_url,
    stream_download,
)

logger = logging.getLogger(__name__)

# Browser-like headers so NAIC and state DOI sites don't return 403 for script requests.
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://content.naic.org/",
}

NAIC_BASE = "https://content.naic.org/sites/default/files"

# Regulations: NAIC model laws and state adoption charts (PDFs).
REGULATIONS_URLS: dict[str, str] = {
    "mo-710": f"{NAIC_BASE}/model-law-710.pdf",
    "mo-680": f"{NAIC_BASE}/model-law-680.pdf",
    "mo-720": f"{NAIC_BASE}/model-law-720.pdf",
    "mo-725": f"{NAIC_BASE}/model-law-725.pdf",
    "mo-745": f"{NAIC_BASE}/model-law-745.pdf",
    "mo-751": f"{NAIC_BASE}/model-law-751.pdf",
    "mo-777": f"{NAIC_BASE}/model-law-777.pdf",
    "chart-pa-30": f"{NAIC_BASE}/model-law-chart-pa-30-compulsory-motor-vehicle-insurance.pdf",
    "chart-pa-10": f"{NAIC_BASE}/model-law-chart-pa-10-rate-filing-methods-for-property-casualty-insurance-workers-comp-title.pdf",
    "chart-pa-15": f"{NAIC_BASE}/model-law-chart-pa-15-form-filing-methods-for-property-casualty.pdf",
}

# Forms: state consumer guides (PDF or HTML). Key = state code or "naic".
# Keys that are HTML pages (not PDF).
FORMS_HTML_KEYS: frozenset[str] = frozenset({"TX", "IL", "FL"})

FORMS_URLS: dict[str, str] = {
    "CA": "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/upload/IG-Auto-Insurance-Updated-092123.pdf",
    "TX": "https://www.tdi.texas.gov/pubs/consumer/cb020.html",
    "NY": "https://www.dfs.ny.gov/system/files/documents/2025/03/Automobile-Insurance_2024.pdf",
    "IL": "https://idoi.illinois.gov/consumers/consumerinsurance/auto-insurance-shopping-guide.html",
    "PA": "https://www.pa.gov/content/dam/copapwp-pagov/en/insurance/documents/consumer-help-center/learn-about-insurance/auto-insurance-guide.pdf",
    "OH": "https://insurance.ohio.gov/wps/wcm/connect/gov/e992ae26-da3a-4bf3-8d4a-1c548eaac41c/AutomobileInsuranceGuide.pdf?MOD=AJPERES&CONVERT_TO=url&CACHEID=ROOTWORKSPACE.Z18_79GCH8013HMOA06A2E16IV2082-e992ae26-da3a-4bf3-8d4a-1c548eaac41c-peTe7gj",
    "NJ": "https://www.nj.gov/dobi/division_consumers/pdf/autoguide2023.pdf",
    "MI": "https://www.michigan.gov/-/media/Project/Websites/difs/Publication/Auto/Auto_Insurance_Guide.pdf?rev=5e970c4aed3c4599bdd08a4706a44cbd",
    "GA": "https://oci.georgia.gov/document/document/guide-auto-insurance/download",
    "FL": "https://floir.com/property-casualty/automobile-insurance",
    "naic": f"{NAIC_BASE}/publication-aut-pp-consumer-auto.pdf",
}

# Claims: limited public guides (PDF or HTML).
CLAIMS_URLS: dict[str, str] = {
    "ga_claim_tips": "https://oci.georgia.gov/insurance-resources/auto/auto-claim-tips",
}

# Rates: limited public data (e.g. consolidated rules with rate info).
RATES_URLS: dict[str, str] = {
    "ny_auto_consolidated": "https://www.dfs.ny.gov/system/files/documents/2020/11/rf_auto_consolidated_txt.pdf",
}


def _download_url_to_file(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    force: bool,
    binary: bool = True,
) -> bool:
    """Download url to dest. Return True if downloaded or already present, False on 404/skip."""
    if dest.exists() and not force:
        logger.debug("Skipping (exists): %s", dest)
        return True
    try:
        if binary:
            stream_download(client, url, dest)
        else:
            resp = client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(resp.text, encoding="utf-8", errors="replace")
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning("Skipping (404): %s", url)
            return False
        if e.response.status_code == 403:
            logger.warning(
                "Skipping (403 Forbidden): %s — site may block script requests",
                url,
            )
            return False
        raise


def download_regulations(raw_dir: Path, *, force: bool = False) -> None:
    """Download NAIC model law PDFs and state adoption charts to raw_dir/regulations/naic/."""
    out_base = raw_dir / "regulations" / "naic"
    out_base.mkdir(parents=True, exist_ok=True)
    files_with_hashes: list[tuple[Path, str | None]] = []

    with httpx.Client(
        timeout=DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        headers=DOWNLOAD_HEADERS,
    ) as client:
        for key, url in REGULATIONS_URLS.items():
            name = f"{key}.pdf"
            dest = out_base / name
            logger.info("Downloading %s -> %s", url, dest)
            ok = _download_url_to_file(client, url, dest, force=force, binary=True)
            if ok:
                try:
                    h = file_sha256(dest)
                except OSError:
                    h = None
                files_with_hashes.append((dest, h))

    manifest_path = out_base / "manifest.json"
    write_manifest(
        manifest_path,
        NAIC_BASE,
        files_with_hashes,
        base_dir=out_base,
        sources=list(REGULATIONS_URLS.values()),
    )
    logger.info("Wrote manifest to %s", manifest_path)


def download_forms(raw_dir: Path, *, force: bool = False) -> None:
    """Download state consumer guides (PDF or HTML) to raw_dir/forms/{state}/ and raw_dir/forms/naic/."""
    out_base = raw_dir / "forms"
    files_with_hashes: list[tuple[Path, str | None]] = []

    with httpx.Client(
        timeout=DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        headers=DOWNLOAD_HEADERS,
    ) as client:
        for key, url in FORMS_URLS.items():
            key_lower = key.lower()
            if key_lower == "naic":
                subdir = out_base / "naic"
            else:
                subdir = out_base / key
            subdir.mkdir(parents=True, exist_ok=True)

            is_html = key in FORMS_HTML_KEYS
            if is_html:
                ext = ".html"
                default_name = "guide.html"
            else:
                ext = ".pdf"
                default_name = "guide.pdf"
            name = sanitize_filename_from_url(url, default_name)
            if not name.lower().endswith(ext.lstrip(".")):
                name = name.rsplit(".", 1)[0] + ext if "." in name else name + ext
            dest = subdir / name

            if dest.exists() and not force:
                logger.debug("Skipping (exists): %s", dest)
                try:
                    h = file_sha256(dest)
                except OSError:
                    h = None
                files_with_hashes.append((dest, h))
                continue
            try:
                logger.info("Downloading %s -> %s", url, dest)
                ok = _download_url_to_file(client, url, dest, force=force, binary=not is_html)
                if ok:
                    try:
                        h = file_sha256(dest)
                    except OSError:
                        h = None
                    files_with_hashes.append((dest, h))
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning("Skipping (404): %s", url)
                else:
                    raise

    manifest_path = out_base / "manifest.json"
    write_manifest(
        manifest_path,
        "state_doi_and_naic",
        files_with_hashes,
        base_dir=out_base,
        sources=list(FORMS_URLS.values()),
    )
    logger.info("Wrote manifest to %s", manifest_path)


def download_claims(raw_dir: Path, *, force: bool = False) -> None:
    """Download claims process guides to raw_dir/claims/. Limited public sources."""
    out_base = raw_dir / "claims"
    out_base.mkdir(parents=True, exist_ok=True)
    files_with_hashes: list[tuple[Path, str | None]] = []

    with httpx.Client(
        timeout=DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        headers=DOWNLOAD_HEADERS,
    ) as client:
        for key, url in CLAIMS_URLS.items():
            is_html = ".html" in url or not url.rstrip("/").lower().endswith(".pdf")
            ext = ".html" if is_html else ".pdf"
            name = sanitize_filename_from_url(url, f"{key}{ext}")
            if not name.lower().endswith(ext.lstrip(".")):
                name = name.rsplit(".", 1)[0] + ext if "." in name else name + ext
            dest = out_base / name
            if dest.exists() and not force:
                logger.debug("Skipping (exists): %s", dest)
                try:
                    h = file_sha256(dest)
                except OSError:
                    h = None
                files_with_hashes.append((dest, h))
                continue
            try:
                logger.info("Downloading %s -> %s", url, dest)
                ok = _download_url_to_file(client, url, dest, force=force, binary=not is_html)
                if ok:
                    try:
                        h = file_sha256(dest)
                    except OSError:
                        h = None
                    files_with_hashes.append((dest, h))
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning("Skipping (404): %s", url)
                else:
                    raise

    manifest_path = out_base / "manifest.json"
    write_manifest(
        manifest_path,
        "state_doi_claims",
        files_with_hashes,
        base_dir=out_base,
        sources=list(CLAIMS_URLS.values()),
    )
    logger.info("Wrote manifest to %s", manifest_path)


def download_rates(raw_dir: Path, *, force: bool = False) -> None:
    """Download rate-related docs (e.g. NY consolidated auto rules) to raw_dir/rates/. Limited public sources."""
    out_base = raw_dir / "rates"
    out_base.mkdir(parents=True, exist_ok=True)
    files_with_hashes: list[tuple[Path, str | None]] = []

    with httpx.Client(
        timeout=DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        headers=DOWNLOAD_HEADERS,
    ) as client:
        for key, url in RATES_URLS.items():
            name = sanitize_filename_from_url(url, f"{key}.pdf")
            if not name.lower().endswith(".pdf"):
                # Ensure a .pdf extension while preserving the base name:
                # - If there is no dot in the name, just append ".pdf".
                # - If there is a dot, replace only the final extension with ".pdf"
                #   (e.g., "file.name.txt" -> "file.name.pdf", not "file.pdf").
                name = name + ".pdf" if "." not in name else name.rsplit(".", 1)[0] + ".pdf"
            dest = out_base / name
            logger.info("Downloading %s -> %s", url, dest)
            ok = _download_url_to_file(client, url, dest, force=force, binary=True)
            if ok:
                try:
                    h = file_sha256(dest)
                except OSError:
                    h = None
                files_with_hashes.append((dest, h))

    manifest_path = out_base / "manifest.json"
    write_manifest(
        manifest_path,
        "state_doi_rates",
        files_with_hashes,
        base_dir=out_base,
        sources=list(RATES_URLS.values()),
    )
    logger.info("Wrote manifest to %s", manifest_path)
