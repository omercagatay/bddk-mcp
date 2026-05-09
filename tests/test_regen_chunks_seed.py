"""Tests for scripts/regen_chunks_seed.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from regen_chunks_seed import build_chunk_records  # noqa: E402


def test_build_chunk_records_includes_section_metadata():
    records = build_chunk_records(
        {
            "document_id": "mevzuat_22599",
            "title": "Karşılık Yönetmeliği",
            "markdown_content": "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm.",
        }
    )

    assert records
    assert records[0]["section_type"] == "madde"
    assert records[0]["section_ref"] == "9"
    assert records[0]["section_start_char"] == 0
    assert records[0]["section_end_char"] > records[0]["section_start_char"]
    assert len(records[0]["section_content_hash"]) == 64
