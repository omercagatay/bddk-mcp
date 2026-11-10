"""Tests for scripts/reindex_document_sections.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from reindex_document_sections import (  # noqa: E402
    DocumentSectionReindexCandidate,
    _build_parser,
    execute_reindex,
    load_document_rows,
    reindex_document_rows,
    render_summary,
)

from bddk_mcp.core.config import CORPUS_PATH  # noqa: E402
from bddk_mcp.store.corpus_db import CorpusDB  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument  # noqa: E402

MADDE_BODY = "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm."


class FakeSectionStore:
    def __init__(self) -> None:
        self.replacements: list[tuple[str, int, str]] = []

    async def replace_document_sections(self, doc_id: str, sections: list, *, source_content_hash: str) -> int:
        self.replacements.append((doc_id, len(sections), source_content_hash))
        return len(sections)


async def test_reindex_document_rows_indexes_parsed_sections():
    rows = [
        DocumentSectionReindexCandidate(
            document_id="mevzuat_22599", markdown_content=MADDE_BODY, content_hash="a" * 64
        ),
        DocumentSectionReindexCandidate(document_id="blank", markdown_content="", content_hash=""),
    ]
    store = FakeSectionStore()

    stats = await reindex_document_rows(rows, store=store, dry_run=False)

    assert stats.scanned_documents == 2
    assert stats.documents_with_sections == 1
    assert stats.sections_indexed == 2
    assert store.replacements == [("mevzuat_22599", 2, "a" * 64), ("blank", 0, "")]


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


async def test_load_document_rows_filters_by_doc_id_and_limit(tmp_path):
    db = CorpusDB.open(tmp_path / "c.sqlite")
    try:
        store = DocumentStore(db)
        await store.store_document(StoredDocument(document_id="943", title="İlke", markdown_content="İlke 5\nMetin."))
        await store.store_document(
            StoredDocument(document_id="mevzuat_22599", title="Madde", markdown_content=MADDE_BODY)
        )
        await store.store_document(StoredDocument(document_id="blank", title="Boş", markdown_content=""))

        rows = await load_document_rows(db)
        assert [row.document_id for row in rows] == ["943", "mevzuat_22599"]  # empty bodies are skipped
        assert rows[1].markdown_content == MADDE_BODY
        assert rows[1].content_hash == (await store.get_document("mevzuat_22599")).content_hash

        assert [row.document_id for row in await load_document_rows(db, doc_id="943")] == ["943"]
        assert [row.document_id for row in await load_document_rows(db, doc_id="blank")] == []
        assert [row.document_id for row in await load_document_rows(db, limit=1)] == ["943"]
    finally:
        db.close()


async def test_execute_reindex_rebuilds_sections_in_the_corpus_file(tmp_path):
    corpus = tmp_path / "c.sqlite"
    db = CorpusDB.open(corpus)
    try:
        store = DocumentStore(db)
        await store.store_document(
            StoredDocument(document_id="mevzuat_22599", title="Madde", markdown_content=MADDE_BODY)
        )
        # Simulate a stale index: the parsed sections are gone
        await db.transaction(lambda conn: conn.execute("DELETE FROM document_sections"))

        dry = await execute_reindex(corpus_path=str(corpus), dry_run=True)
        assert (dry.scanned_documents, dry.documents_with_sections, dry.sections_indexed) == (1, 1, 2)
        assert await store.get_document_section("mevzuat_22599") == []

        stats = await execute_reindex(corpus_path=str(corpus), dry_run=False)
        assert (stats.scanned_documents, stats.documents_with_sections, stats.sections_indexed) == (1, 1, 2)
        sections = await store.get_document_section("mevzuat_22599")
        assert len(sections) == 2
        assert all(section.source_content_hash == sections[0].source_content_hash for section in sections)
    finally:
        db.close()


def test_parser_defaults_to_the_configured_corpus_path():
    assert _build_parser().parse_args([]).corpus == CORPUS_PATH
    assert _build_parser().parse_args(["--corpus", "other.sqlite", "--execute"]).corpus == "other.sqlite"
