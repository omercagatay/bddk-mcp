"""GPU-gated integration test for Chandra2 end-to-end.

Skipped unless CUDA is available. Loads the chandra-ocr HF model in-process
and runs a real mevzuat PDF through it. Runtime: ~2 minutes on first run
(model download + inference).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


FIXTURE_PDF = Path(__file__).parent / "fixtures" / "mevzuat_42628_sample.pdf"


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda(), reason="CUDA not available")
def test_chandra_end_to_end_on_real_fixture():
    if not FIXTURE_PDF.exists():
        pytest.skip(f"fixture missing: {FIXTURE_PDF}")

    from ocr_backends_chandra import ChandraBackend

    pdf_bytes = FIXTURE_PDF.read_bytes()

    backend = ChandraBackend()
    assert backend.is_available() is True
    output = backend.extract(pdf_bytes)

    assert output is not None
    assert len(output) > 100
    # sanity: Turkish diacritics should survive OCR
    assert any(c in output for c in "çğışöüÇĞİŞÖÜ")
