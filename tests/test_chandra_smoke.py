"""Explicitly provisioned GPU integration test for Chandra2 end-to-end.

Loads a local chandra-ocr model in-process and runs a retained mevzuat fixture
through it. The shared preflight forbids model downloads. Runtime depends on
the provisioned accelerator and model cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "mevzuat_42628_sample.pdf"


@pytest.mark.gpu
@pytest.mark.usefixtures("provisioned_gpu_ocr_lane")
def test_chandra_end_to_end_on_real_fixture():
    assert FIXTURE_PDF.is_file(), f"retained fixture missing: {FIXTURE_PDF.name}"

    from bddk_mcp.ocr.chandra import ChandraBackend

    pdf_bytes = FIXTURE_PDF.read_bytes()

    backend = ChandraBackend()
    assert backend.is_available() is True
    output = backend.extract(pdf_bytes)

    assert output is not None
    assert len(output) > 100
    # sanity: Turkish diacritics should survive OCR
    assert any(c in output for c in "çğışöüÇĞİŞÖÜ")
