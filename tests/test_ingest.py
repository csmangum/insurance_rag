"""Tests for extraction and chunking (Phase 2)."""
import csv
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medicare_rag.ingest.chunk import _is_code_doc, chunk_documents
from medicare_rag.ingest.extract import (
    _format_date_yyyymmdd,
    _html_to_text,
    _meta_schema,
    _parse_hcpcs_line,
    extract_all,
    extract_hcpcs,
    extract_icd10cm,
    extract_iom,
    extract_mcd,
)


# --- Extraction helpers ---


def test_meta_schema() -> None:
    meta = _meta_schema(source="iom", manual="100-02", chapter="6")
    assert meta["source"] == "iom"
    assert meta["manual"] == "100-02"
    assert meta["chapter"] == "6"
    assert meta["title"] is None
    meta2 = _meta_schema(source="codes", hcpcs_code="A1001")
    assert meta2["hcpcs_code"] == "A1001"


def test_html_to_text() -> None:
    assert _html_to_text("<p>Hello</p>") == "Hello"
    result = _html_to_text("<div><b>Foo</b> bar</div>")
    assert "Foo" in result and "bar" in result
    assert _html_to_text("") == ""


def test_format_date_yyyymmdd() -> None:
    assert _format_date_yyyymmdd("20020701") == "2002-07-01"
    assert _format_date_yyyymmdd("") is None
    assert _format_date_yyyymmdd("123") is None


def test_parse_hcpcs_line() -> None:
    # 320-char fixed width: code 1-5, RIC 11, long 12-91, short 92-119
    line = (
        "A1001"  # 1-5
        + "00100"  # 6-10 seq
        + "7"  # 11 RIC (7 = first modifier)
        + "Dressing for one wound".ljust(80)  # 12-91
        + "Dressing for one wound".ljust(28)  # 92-119
        + " " * (277 - 120)
        + "20020701"  # 277-284
        + "20020701"  # 285-292
        + " " * (320 - 292)
    )
    rec = _parse_hcpcs_line(line)
    assert rec is not None
    assert rec["code"] == "A1001"
    assert rec["ric"] == "7"
    assert "Dressing" in rec["long_desc"]
    assert rec["effective_date"] == "20020701"


# --- IOM extraction (mocked PDF) ---


@pytest.fixture
def tmp_iom_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "iom"
    (raw / "100-02").mkdir(parents=True)
    # Create a minimal file so the path exists; we mock pdfplumber
    (raw / "100-02" / "bp102c06.pdf").write_bytes(b"%PDF-1.4 minimal")
    return tmp_path / "raw"


def test_extract_iom_writes_txt_and_meta(tmp_iom_raw: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Chapter 6 content here."
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("medicare_rag.ingest.extract.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = mock_pdf
        written = extract_iom(processed, tmp_iom_raw, force=True)

    assert len(written) == 1
    txt_path, meta_path = written[0]
    assert txt_path.exists()
    assert meta_path.exists()
    assert "Chapter 6 content" in txt_path.read_text()
    meta = __import__("json").loads(meta_path.read_text())
    assert meta["source"] == "iom"
    assert meta["manual"] == "100-02"
    assert meta["chapter"] == "6"


# --- MCD extraction (CSV with HTML) ---


@pytest.fixture
def tmp_mcd_raw(tmp_path: Path) -> Path:
    mcd = tmp_path / "raw" / "mcd"
    mcd.mkdir(parents=True)
    # One inner zip containing a CSV with HTML column
    sub = mcd / "current_lcd"
    sub.mkdir(parents=True)
    csv_path = sub / "LCD.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["LCD_ID", "Title", "Body"],
        )
        w.writeheader()
        w.writerow({
            "LCD_ID": "L12345",
            "Title": "Test LCD",
            "Body": "<p>Coverage criteria for <b>test</b>.</p>",
        })
    return tmp_path / "raw"


def test_extract_mcd_writes_txt_and_meta(tmp_mcd_raw: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    written = extract_mcd(processed, tmp_mcd_raw, force=True)
    assert len(written) >= 1
    txt_path, meta_path = written[0]
    assert txt_path.exists()
    assert meta_path.exists()
    text = txt_path.read_text()
    assert "Coverage criteria" in text
    assert "<p>" not in text
    meta = __import__("json").loads(meta_path.read_text())
    assert meta["source"] == "mcd"
    assert meta.get("lcd_id") or "L12345" in str(meta.get("doc_id", ""))


# --- HCPCS extraction (fixed-width lines) ---


@pytest.fixture
def tmp_hcpcs_raw(tmp_path: Path) -> Path:
    codes = tmp_path / "raw" / "codes" / "hcpcs" / "sample"
    codes.mkdir(parents=True)
    # Two records: first line RIC 3 (procedure), second RIC 4 (continuation)
    line1 = (
        "A1001"
        + "00100"
        + "3"
        + "Dressing for one wound".ljust(80)
        + "Dressing for one wound".ljust(28)
        + " " * (277 - 120)
        + "20020701"
        + "20020701"
        + " " * (320 - 292)
    )
    line2 = (
        "A1001"
        + "00200"
        + "4"
        + " (continued description)".ljust(80)
        + "".ljust(28)
        + " " * (320 - 92)
    )
    (codes / "HCPC_sample.txt").write_text(line1 + "\n" + line2 + "\n", encoding="utf-8")
    return tmp_path / "raw"


def test_extract_hcpcs_writes_txt_and_meta(tmp_hcpcs_raw: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    written = extract_hcpcs(processed, tmp_hcpcs_raw, force=True)
    assert len(written) >= 1
    txt_path, meta_path = written[0]
    assert txt_path.exists()
    assert meta_path.exists()
    text = txt_path.read_text()
    assert "A1001" in text
    assert "Dressing" in text
    assert "continued description" in text
    meta = __import__("json").loads(meta_path.read_text())
    assert meta["source"] == "codes"
    assert meta.get("hcpcs_code") == "A1001"


# --- extract_all ---


def test_extract_all_respects_source(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    (raw / "codes" / "hcpcs" / "x").mkdir(parents=True)
    line = (
        "A1001" + "00100" + "3"
        + "Dressing for one wound".ljust(80)
        + "Short".ljust(28)
        + " " * (277 - 120) + "20020701" + "20020701" + " " * (320 - 292)
    )
    (raw / "codes" / "hcpcs" / "x" / "HCPC_x.txt").write_text(line)
    written = extract_all(processed, raw, source="codes", force=True)
    assert len(written) >= 1


# --- ICD-10-CM extraction (ZIP + XML) ---


@pytest.fixture
def tmp_icd10cm_raw(tmp_path: Path) -> Path:
    """Minimal ICD-10-CM ZIP with one tabular-style XML for code/description pairs."""
    icd_dir = tmp_path / "raw" / "codes" / "icd10-cm"
    icd_dir.mkdir(parents=True)
    zip_path = icd_dir / "icd10cm_sample.zip"
    # Minimal XML: root with elements that have <code> and <desc> (or description) children
    xml_content = """<?xml version="1.0"?>
<root>
  <row><code>A00.0</code><desc>Cholera due to Vibrio cholerae 01, biovar cholerae</desc></row>
  <row><code>A00.1</code><description>Cholera due to Vibrio cholerae 01, biovar el tor</description></row>
</root>
"""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("tabular.xml", xml_content.encode("utf-8"))
    return tmp_path / "raw"


def test_extract_icd10cm_writes_txt_and_meta(tmp_icd10cm_raw: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    written = extract_icd10cm(processed, tmp_icd10cm_raw, force=True)
    assert len(written) >= 2
    by_code = {}
    for txt_path, meta_path in written:
        assert txt_path.exists()
        assert meta_path.exists()
        text = txt_path.read_text()
        assert "Code:" in text and "Description:" in text
        meta = __import__("json").loads(meta_path.read_text())
        assert meta["source"] == "codes"
        assert "icd10_code" in meta
        by_code[meta["icd10_code"]] = (text, meta)
    assert "A00.0" in by_code
    assert "A00.1" in by_code
    assert "Cholera" in by_code["A00.0"][0]


# --- Chunking ---


def test_is_code_doc() -> None:
    assert _is_code_doc({"source": "codes"}) is True
    assert _is_code_doc({"source": "iom"}) is False


def test_chunk_documents_attaches_metadata(tmp_path: Path) -> None:
    (tmp_path / "iom" / "100-02").mkdir(parents=True)
    (tmp_path / "iom" / "100-02" / "ch6.txt").write_text(
        "First paragraph.\n\nSecond paragraph.\n\nThird paragraph with more content. " * 50
    )
    (tmp_path / "iom" / "100-02" / "ch6.meta.json").write_text(
        '{"source": "iom", "manual": "100-02", "chapter": "6", "doc_id": "iom_100-02_ch6"}'
    )
    docs = chunk_documents(tmp_path, source="iom", chunk_size=100, chunk_overlap=20)
    assert len(docs) >= 2
    for d in docs:
        assert d.metadata.get("source") == "iom"
        assert "chunk_index" in d.metadata
        assert "total_chunks" in d.metadata


def test_chunk_documents_code_one_chunk_per_doc(tmp_path: Path) -> None:
    (tmp_path / "codes" / "hcpcs").mkdir(parents=True)
    (tmp_path / "codes" / "hcpcs" / "A1001.txt").write_text("Code: A1001\n\nLong description.\n\nShort.")
    (tmp_path / "codes" / "hcpcs" / "A1001.meta.json").write_text(
        '{"source": "codes", "doc_id": "hcpcs_A1001", "hcpcs_code": "A1001"}'
    )
    docs = chunk_documents(tmp_path, source="codes")
    assert len(docs) == 1
    assert docs[0].page_content.strip().startswith("Code:")
    assert docs[0].metadata["source"] == "codes"
    assert "chunk_index" not in docs[0].metadata
