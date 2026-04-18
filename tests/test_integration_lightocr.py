"""GPU integration tests for LightOnOCR on real BDDK documents.

Skipped in CI (no GPU). Run locally: pytest tests/test_integration_lightocr.py -m gpu
"""

import pytest

from ocr_backends import LightOCRBackend

pytestmark = pytest.mark.gpu


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def backend() -> LightOCRBackend:
    if not _cuda_available():
        pytest.skip("CUDA not available")
    return LightOCRBackend()


@pytest.fixture(scope="module")
def pdf_42628() -> bytes:
    """Fetch mevzuat_42628 PDF once per test module.

    Tries GeneratePdf first (preserves formulas), falls back to the
    static /MevzuatMetin/yonetmelik/7.5.42628.pdf URL.
    """
    import httpx

    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    with httpx.Client(timeout=180.0, follow_redirects=True, headers={"User-Agent": ua}) as client:
        client.get(
            "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5"
        )
        resp = client.get(
            "https://www.mevzuat.gov.tr/File/GeneratePdf?"
            "mevzuatNo=42628&mevzuatTur=Yonetmelik&mevzuatTertip=5"
        )
        if resp.status_code != 200 or resp.content[:5] != b"%PDF-":
            resp = client.get(
                "https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628.pdf"
            )
    assert resp.status_code == 200, f"PDF fetch failed: {resp.status_code}"
    assert resp.content[:5] == b"%PDF-", f"Not a PDF: {resp.content[:20]!r}"
    return resp.content


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
