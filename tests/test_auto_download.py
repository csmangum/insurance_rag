"""Tests for auto insurance download scripts."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insurance_rag.domains.auto.download import (
    CLAIMS_URLS,
    FORMS_URLS,
    RATES_URLS,
    REGULATIONS_URLS,
    download_claims,
    download_forms,
    download_regulations,
    download_rates,
)
from insurance_rag.domains.auto.extract import (
    extract_claims,
    extract_forms,
    extract_rates,
    extract_regulations,
)
from insurance_rag.download._utils import sanitize_filename_from_url

PDF_CONTENT = b"%PDF-1.4 fake"
HTML_CONTENT = "<html><body><p>Auto insurance guide</p></body></html>"


@pytest.fixture
def tmp_raw(tmp_path: Path) -> Path:
    return tmp_path / "raw"


def _make_stream_cm(content: bytes = PDF_CONTENT):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.iter_bytes = MagicMock(return_value=iter([content]))
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=r)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_download_regulations_naic_models(tmp_raw: Path) -> None:
    """download_regulations downloads NAIC model law PDFs and writes manifest."""
    with patch("insurance_rag.domains.auto.download.httpx") as mock_httpx:
        mock_client = MagicMock()
        mock_client.stream.side_effect = lambda *a, **k: _make_stream_cm()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client

        download_regulations(tmp_raw, force=True)

    naic_dir = tmp_raw / "regulations" / "naic"
    assert naic_dir.exists()
    pdfs = list(naic_dir.glob("*.pdf"))
    assert len(pdfs) == len(REGULATIONS_URLS)
    manifest = naic_dir / "manifest.json"
    assert manifest.exists()
    manifest_text = manifest.read_text()
    assert "source_url" in manifest_text
    assert "naic" in manifest_text.lower() or "content.naic" in manifest_text


def test_regulations_idempotency(tmp_raw: Path) -> None:
    """When all regulation files exist and force=False, download_regulations skips re-download."""
    naic_dir = tmp_raw / "regulations" / "naic"
    naic_dir.mkdir(parents=True)
    for key in REGULATIONS_URLS:
        (naic_dir / f"{key}.pdf").write_bytes(PDF_CONTENT)
    (naic_dir / "manifest.json").write_text(
        '{"source_url": "x", "files": [{"path": "mo-725.pdf", "file_hash": null}]}'
    )

    stream_calls: list = []

    def track_stream(*args, **kwargs):
        stream_calls.append(args)
        return _make_stream_cm()

    with patch("insurance_rag.domains.auto.download.httpx") as mock_httpx:
        mock_client = MagicMock()
        mock_client.stream.side_effect = track_stream
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client

        download_regulations(tmp_raw, force=False)

    assert len(stream_calls) == 0, "Should not re-download when all files exist"


def test_download_forms_state_guides(tmp_raw: Path) -> None:
    """download_forms downloads state/NAIC guides (PDF and HTML) and writes manifest."""
    def fake_stream(*args, **kwargs):
        return _make_stream_cm(PDF_CONTENT)

    def fake_get(url, **kwargs):
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.text = HTML_CONTENT
        return r

    with patch("insurance_rag.domains.auto.download.httpx") as mock_httpx:
        mock_client = MagicMock()
        mock_client.stream.side_effect = fake_stream
        mock_client.get.side_effect = fake_get
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client

        download_forms(tmp_raw, force=True)

    forms_dir = tmp_raw / "forms"
    assert forms_dir.exists()
    assert (forms_dir / "naic").exists()
    assert (forms_dir / "CA").exists()
    assert (forms_dir / "TX").exists()
    manifest = forms_dir / "manifest.json"
    assert manifest.exists()
    manifest_text = manifest.read_text()
    assert "source_url" in manifest_text
    pdf_count = len(list(forms_dir.rglob("*.pdf")))
    html_count = len(list(forms_dir.rglob("*.html")))
    assert pdf_count + html_count == len(FORMS_URLS)


def test_forms_idempotency(tmp_raw: Path) -> None:
    """When a form file exists and force=False, that file is not overwritten."""
    ca_dir = tmp_raw / "forms" / "CA"
    ca_dir.mkdir(parents=True)
    ca_filename = sanitize_filename_from_url(FORMS_URLS["CA"], "guide.pdf")
    ca_file = ca_dir / ca_filename
    ca_file.write_bytes(PDF_CONTENT)
    original_size = ca_file.stat().st_size

    def fake_stream(*args, **kwargs):
        return _make_stream_cm(b"%PDF other content")

    def fake_get(url, **kwargs):
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.text = HTML_CONTENT
        return r

    with patch("insurance_rag.domains.auto.download.httpx") as mock_httpx:
        mock_client = MagicMock()
        mock_client.stream.side_effect = fake_stream
        mock_client.get.side_effect = fake_get
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client

        download_forms(tmp_raw, force=False)

    assert ca_file.exists()
    assert ca_file.read_bytes() == PDF_CONTENT, "Existing file should not be overwritten when force=False"


def test_download_claims_and_rates(tmp_raw: Path) -> None:
    """download_claims and download_rates run and write manifests (thin sources)."""
    def fake_get(url, **kwargs):
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.text = HTML_CONTENT
        return r

    with patch("insurance_rag.domains.auto.download.httpx") as mock_httpx:
        mock_client = MagicMock()
        mock_client.get.side_effect = fake_get
        mock_client.stream.side_effect = lambda *a, **k: _make_stream_cm()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client

        download_claims(tmp_raw, force=True)
        download_rates(tmp_raw, force=True)

    claims_dir = tmp_raw / "claims"
    assert claims_dir.exists()
    assert (claims_dir / "manifest.json").exists()
    if len(CLAIMS_URLS) > 0:
        assert len(list(claims_dir.glob("*"))) >= 2

    rates_dir = tmp_raw / "rates"
    assert rates_dir.exists()
    assert (rates_dir / "manifest.json").exists()
    assert len(list(rates_dir.glob("*.pdf"))) == len(RATES_URLS)


# --- Extract function tests ---


def _mock_pdfplumber_open(text: str = "Auto insurance content."):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = text
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    return mock_pdf


def test_extract_regulations_writes_txt_and_meta(tmp_path: Path) -> None:
    """extract_regulations extracts PDFs from raw/regulations/naic/ and writes txt + meta."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    naic_dir = raw / "regulations" / "naic"
    naic_dir.mkdir(parents=True)
    (naic_dir / "mo-710.pdf").write_bytes(b"%PDF-1.4 minimal")

    with patch("insurance_rag.domains.auto.extract.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdfplumber_open("Model law 710 content.")
        written = extract_regulations(processed, raw, force=True)

    assert len(written) == 1
    txt_path, meta_path = written[0]
    assert txt_path.exists()
    assert meta_path.exists()
    assert "Model law 710 content." in txt_path.read_text()
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "regulations"


def test_extract_forms_writes_txt_and_meta(tmp_path: Path) -> None:
    """extract_forms extracts PDFs and HTML from raw/forms/{state}/ subdirs."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    ca_dir = raw / "forms" / "CA"
    ca_dir.mkdir(parents=True)
    (ca_dir / "auto-guide.pdf").write_bytes(b"%PDF-1.4 minimal")
    tx_dir = raw / "forms" / "TX"
    tx_dir.mkdir(parents=True)
    (tx_dir / "guide.html").write_text(
        "<html><body><p>TX auto insurance guide</p></body></html>", encoding="utf-8"
    )

    with patch("insurance_rag.domains.auto.extract.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdfplumber_open("CA auto guide content.")
        written = extract_forms(processed, raw, force=True)

    assert len(written) == 2
    paths = {p.name for p, _ in written}
    assert "auto-guide.txt" in paths
    assert "guide.txt" in paths
    for txt_path, meta_path in written:
        meta = json.loads(meta_path.read_text())
        assert meta["source"] == "forms"


def test_extract_forms_skips_non_directory(tmp_path: Path) -> None:
    """extract_forms skips non-directory files (e.g. manifest.json) at the forms root."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    forms_dir = raw / "forms"
    forms_dir.mkdir(parents=True)
    (forms_dir / "manifest.json").write_text('{"files": []}', encoding="utf-8")
    ca_dir = forms_dir / "CA"
    ca_dir.mkdir()
    (ca_dir / "guide.html").write_text(
        "<html><body><p>CA guide</p></body></html>", encoding="utf-8"
    )

    written = extract_forms(processed, raw, force=True)

    assert len(written) == 1


def test_extract_claims_writes_txt_and_meta(tmp_path: Path) -> None:
    """extract_claims extracts HTML files from raw/claims/."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    claims_dir = raw / "claims"
    claims_dir.mkdir(parents=True)
    (claims_dir / "claim-tips.html").write_text(
        "<html><body><p>GA auto claim tips content.</p></body></html>", encoding="utf-8"
    )

    written = extract_claims(processed, raw, force=True)

    assert len(written) == 1
    txt_path, meta_path = written[0]
    assert txt_path.exists()
    assert "claim tips content" in txt_path.read_text()
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "claims"


def test_extract_rates_writes_txt_and_meta(tmp_path: Path) -> None:
    """extract_rates extracts PDFs from raw/rates/."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    rates_dir = raw / "rates"
    rates_dir.mkdir(parents=True)
    (rates_dir / "ny-consolidated.pdf").write_bytes(b"%PDF-1.4 minimal")

    with patch("insurance_rag.domains.auto.extract.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdfplumber_open("NY rate rules content.")
        written = extract_rates(processed, raw, force=True)

    assert len(written) == 1
    txt_path, meta_path = written[0]
    assert txt_path.exists()
    assert "NY rate rules content." in txt_path.read_text()
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "rates"


def test_extract_idempotency(tmp_path: Path) -> None:
    """When processed files exist and force=False, extraction is skipped (output not overwritten)."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    naic_dir = raw / "regulations" / "naic"
    naic_dir.mkdir(parents=True)
    (naic_dir / "mo-710.pdf").write_bytes(b"%PDF-1.4 minimal")
    out_dir = processed / "regulations"
    out_dir.mkdir(parents=True)
    (out_dir / "mo-710.txt").write_text("existing content", encoding="utf-8")
    (out_dir / "mo-710.meta.json").write_text('{"source": "regulations"}', encoding="utf-8")

    with patch("insurance_rag.domains.auto.extract.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdfplumber_open("NEW content.")
        written = extract_regulations(processed, raw, force=False)

    assert len(written) == 1
    assert written[0][0].read_text() == "existing content", "Existing file should not be overwritten when force=False"
