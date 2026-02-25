"""Tests for auto insurance download scripts."""
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
    assert pdf_count + html_count >= len(FORMS_URLS)


def test_forms_idempotency(tmp_raw: Path) -> None:
    """When a form file exists and force=False, that file is not overwritten."""
    ca_dir = tmp_raw / "forms" / "CA"
    ca_dir.mkdir(parents=True)
    ca_file = ca_dir / "IG-Auto-Insurance-Updated-092123.pdf"
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
