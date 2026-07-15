"""Tests for scripts/reindex_document_sections.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from reindex_document_sections import (  # noqa: E402
    DocumentSectionReindexCandidate,
    _resolve_ingestion_database_url,
    execute_reindex,
    reindex_document_rows,
    render_summary,
)


class FakeSectionStore:
    def __init__(self) -> None:
        self.replacements: list[tuple[str, int, str]] = []

    async def replace_document_sections(self, doc_id: str, sections: list, *, source_content_hash: str) -> int:
        self.replacements.append((doc_id, len(sections), source_content_hash))
        return len(sections)


@pytest.mark.asyncio
async def test_reindex_document_rows_indexes_parsed_sections():
    rows = [
        DocumentSectionReindexCandidate(
            document_id="mevzuat_22599",
            markdown_content="MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm.",
            content_hash="a" * 64,
        ),
        DocumentSectionReindexCandidate(document_id="blank", markdown_content="", content_hash=""),
    ]
    store = FakeSectionStore()

    stats = await reindex_document_rows(rows, store=store, dry_run=False)

    assert stats.scanned_documents == 2
    assert stats.documents_with_sections == 1
    assert stats.sections_indexed == 2
    assert store.replacements == [("mevzuat_22599", 2, "a" * 64), ("blank", 0, "")]


@pytest.mark.asyncio
async def test_reindex_document_rows_dry_run_does_not_write():
    rows = [
        DocumentSectionReindexCandidate(
            document_id="943",
            markdown_content="İlke 5\nModel validasyonu.",
            content_hash="b" * 64,
        )
    ]
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


def test_reindex_database_override_must_match_ingestion_identity(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "reindex_document_sections.require_database_url",
        lambda profile: "postgresql://ingestion.invalid/bddk" if profile == "ingestion" else None,
    )

    with pytest.raises(RuntimeError, match="BDDK_INGESTION_DATABASE_URL"):
        _resolve_ingestion_database_url("postgresql://public.invalid/bddk")

    assert _resolve_ingestion_database_url(None) == "postgresql://ingestion.invalid/bddk"


@pytest.mark.asyncio
async def test_reindex_verifies_actual_ingestion_login_before_loading_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = MagicMock()
    pool.close = AsyncMock()
    identity = AsyncMock(side_effect=RuntimeError("identity rejected"))
    monkeypatch.setattr(
        "reindex_document_sections.require_database_url",
        lambda profile: "postgresql://ingestion.invalid/bddk" if profile == "ingestion" else None,
    )

    with (
        patch("reindex_document_sections.asyncpg.create_pool", new=AsyncMock(return_value=pool)),
        patch("reindex_document_sections.assert_database_ready", new=AsyncMock()),
        patch("reindex_document_sections.assert_database_identity", new=identity),
        pytest.raises(RuntimeError, match="identity rejected"),
    ):
        await execute_reindex(dry_run=True)

    identity.assert_awaited_once_with(pool, "ingestion")
    pool.fetch.assert_not_called()
    pool.close.assert_awaited_once_with()
