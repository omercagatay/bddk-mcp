"""Explicitly provisioned GPU integration tests for LightOnOCR.

Run only with the offline GPU/OCR preconditions documented in the test
strategy. The retained fixture prevents this test from contacting a live
regulatory source.
"""

from pathlib import Path

import pytest

from bddk_mcp.ocr.base import LightOCRBackend

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("provisioned_gpu_ocr_lane")]

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "mevzuat_42628_sample.pdf"


@pytest.fixture(scope="module")
def backend() -> LightOCRBackend:
    backend = LightOCRBackend()
    assert backend.is_available() is True
    return backend


@pytest.fixture(scope="module")
def pdf_42628() -> bytes:
    assert FIXTURE_PDF.is_file(), f"retained fixture missing: {FIXTURE_PDF.name}"
    pdf_bytes = FIXTURE_PDF.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    return pdf_bytes


def test_42628_ek2_formulas_extracted(backend: LightOCRBackend, pdf_42628: bytes):
    """EK-2'de formul sembollerinden en az biri gorunmeli."""
    markdown = backend.extract(pdf_42628)
    assert markdown is not None, "LightOCR returned None"
    assert len(markdown) > 5000, f"Output too short: {len(markdown)} chars"

    lower = markdown.lower()
    formula_markers = ["$", "\\frac", "t_0", "t0", "paralel yukarı"]
    hits = [m for m in formula_markers if m in lower]
    assert hits, f"No formula markers found in output. First 2000 chars:\n{markdown[:2000]}"


def test_turkish_chars_preserved(backend: LightOCRBackend, pdf_42628: bytes):
    """Turkce karakterler (c, g, i, s, o, u) bozulmamali."""
    markdown = backend.extract(pdf_42628)
    assert markdown is not None
    assert "Yönetmelik" in markdown or "yönetmelik" in markdown.lower()
    assert any(c in markdown for c in "çğıöşü")
