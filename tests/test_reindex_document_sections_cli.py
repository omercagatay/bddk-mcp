"""Tests for scripts/reindex_document_sections.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from reindex_document_sections import (  # noqa: E402
    DocumentSectionReindexCandidate,
    reindex_document_rows,
    render_summary,
)


class FakeSectionStore:
    def __init__(self) -> None:
        self.replacements: list[tuple[str, int]] = []

    async def replace_document_sections(self, doc_id: str, sections: list) -> int:
        self.replacements.append((doc_id, len(sections)))
        return len(sections)


@pytest.mark.asyncio
async def test_reindex_document_rows_indexes_parsed_sections():
    rows = [
        DocumentSectionReindexCandidate(
            document_id="mevzuat_22599",
            markdown_content="MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm.",
        ),
        DocumentSectionReindexCandidate(document_id="blank", markdown_content=""),
    ]
    store = FakeSectionStore()

    stats = await reindex_document_rows(rows, store=store, dry_run=False)

    assert stats.scanned_documents == 2
    assert stats.documents_with_sections == 1
    assert stats.sections_indexed == 2
    assert store.replacements == [("mevzuat_22599", 2), ("blank", 0)]


@pytest.mark.asyncio
async def test_reindex_document_rows_dry_run_does_not_write():
    rows = [DocumentSectionReindexCandidate(document_id="943", markdown_content="İlke 5\nModel validasyonu.")]
    store = FakeSectionStore()

    stats = await reindex_document_rows(rows, store=store, dry_run=True)

    assert stats.scanned_documents == 1
    assert stats.documents_with_sections == 1
    assert stats.sections_indexed == 1
    assert store.replacements == []


def test_render_summary_explains_dry_run():
    summary = render_summary(
        scanned_documents=2,
        documents_with_sections=1,
        sections_indexed=3,
        dry_run=True,
    )

    assert "Documents scanned: 2" in summary
    assert "Documents with sections: 1" in summary
    assert "Sections parsed: 3" in summary
    assert "Dry run" in summary
