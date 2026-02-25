"""Auto insurance extraction: PDF and HTML from regulations, forms, claims, rates."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF via pdfplumber."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _extract_html_text(html_path: Path) -> str:
    """Extract main text from an HTML file."""
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _write_doc(
    processed_dir: Path, subdir: str, doc_id: str, text: str, meta: dict
) -> tuple[Path, Path]:
    """Write .txt and .meta.json under processed_dir/subdir. Returns (txt_path, meta_path)."""
    out_dir = processed_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{doc_id}.txt"
    meta_path = out_dir / f"{doc_id}.meta.json"
    txt_path.write_text(text, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return txt_path, meta_path


def _extract_from_dir(
    processed_dir: Path,
    raw_subdir: Path,
    processed_subdir: str,
    source_kind: str,
    *,
    force: bool,
) -> list[tuple[Path, Path]]:
    """Walk raw_subdir for PDFs and HTML, extract text, write to processed_subdir."""
    if not raw_subdir.exists():
        logger.warning("Raw subdir not found: %s", raw_subdir)
        return []
    written: list[tuple[Path, Path]] = []
    for path in sorted(raw_subdir.rglob("*")):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf == ".pdf":
            try:
                text = _extract_pdf_text(path)
            except OSError as e:
                logger.warning("Extract failed for %s: %s", path, e)
                continue
        elif suf in (".html", ".htm"):
            try:
                text = _extract_html_text(path)
            except OSError as e:
                logger.warning("Extract failed for %s: %s", path, e)
                continue
        else:
            continue
        if not text.strip():
            logger.warning("No text recovered for %s; skipping", path)
            continue
        doc_id = path.stem
        if not doc_id.replace("-", "").replace("_", "").isalnum():
            doc_id = path.name.replace(path.suffix, "").replace(" ", "_")
        out_txt = processed_dir / processed_subdir / f"{doc_id}.txt"
        out_meta = processed_dir / processed_subdir / f"{doc_id}.meta.json"
        if not force and out_txt.exists() and out_meta.exists():
            written.append((out_txt, out_meta))
            continue
        meta = {
            "source": source_kind,
            "doc_id": f"{source_kind}_{doc_id}",
            "source_file": path.name,
        }
        txt_path, meta_path = _write_doc(
            processed_dir, processed_subdir, doc_id, text, meta
        )
        written.append((txt_path, meta_path))
        logger.info("Wrote %s (%d chars)", txt_path, len(text))
    return written


def extract_regulations(
    processed_dir: Path, raw_dir: Path, *, force: bool = False
) -> list[tuple[Path, Path]]:
    """Extract NAIC model laws and charts from raw_dir/regulations/naic/."""
    return _extract_from_dir(
        processed_dir,
        raw_dir / "regulations" / "naic",
        "regulations",
        "regulations",
        force=force,
    )


def extract_forms(
    processed_dir: Path, raw_dir: Path, *, force: bool = False
) -> list[tuple[Path, Path]]:
    """Extract state consumer guides from raw_dir/forms/ (PDF and HTML)."""
    forms_raw = raw_dir / "forms"
    if not forms_raw.exists():
        logger.warning("Forms raw dir not found: %s", forms_raw)
        return []
    written: list[tuple[Path, Path]] = []
    for subdir in sorted(forms_raw.iterdir()):
        if not subdir.is_dir():
            continue
        written.extend(
            _extract_from_dir(
                processed_dir,
                subdir,
                f"forms/{subdir.name}",
                "forms",
                force=force,
            )
        )
    return written


def extract_claims(
    processed_dir: Path, raw_dir: Path, *, force: bool = False
) -> list[tuple[Path, Path]]:
    """Extract claims guides from raw_dir/claims/."""
    return _extract_from_dir(
        processed_dir,
        raw_dir / "claims",
        "claims",
        "claims",
        force=force,
    )


def extract_rates(
    processed_dir: Path, raw_dir: Path, *, force: bool = False
) -> list[tuple[Path, Path]]:
    """Extract rate-related docs from raw_dir/rates/."""
    return _extract_from_dir(
        processed_dir,
        raw_dir / "rates",
        "rates",
        "rates",
        force=force,
    )
