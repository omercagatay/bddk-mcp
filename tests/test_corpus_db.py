"""Tests for the corpus file itself: opening modes, schema guard, readiness."""

import sqlite3

import pytest

from bddk_mcp.store.corpus_db import SCHEMA_VERSION, CorpusDB, CorpusNotReadyError
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument


def test_writer_creates_the_schema_and_records_its_version(tmp_path):
    db = CorpusDB.open(tmp_path / "c.sqlite")
    try:
        tables = {
            row[0]
            for row in db.run_sync(lambda conn: conn.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        }
        assert {"documents", "document_sections", "chunks", "decision_cache", "sync_failures", "meta"} <= tables
        version = db.run_sync(lambda conn: conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone())
        assert version[0] == str(SCHEMA_VERSION)
    finally:
        db.close()


def test_read_only_open_requires_an_existing_file_and_blocks_writes(tmp_path):
    with pytest.raises(CorpusNotReadyError, match="not found"):
        CorpusDB.open(tmp_path / "missing.sqlite", readonly=True)

    CorpusDB.open(tmp_path / "c.sqlite").close()
    db = CorpusDB.open(tmp_path / "c.sqlite", readonly=True)
    try:
        with pytest.raises(CorpusNotReadyError, match="read-only"):
            db.transaction_sync(lambda conn: conn.execute("DELETE FROM meta"))
        with pytest.raises(sqlite3.OperationalError):  # sqlite itself refuses too, even outside transaction_sync
            db.run_sync(lambda conn: conn.execute("DELETE FROM meta"))
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_file_that_is_not_a_corpus_is_not_ready_with_the_usual_hint(tmp_path):
    empty = tmp_path / "empty.sqlite"
    empty.write_bytes(b"")
    db = CorpusDB.open(empty, readonly=True)
    try:
        with pytest.raises(CorpusNotReadyError, match=r"not a readable corpus.*bddk-mcp build"):
            await db.assert_ready(require_corpus=True)
    finally:
        db.close()


def test_repair_scripts_can_refuse_a_missing_file(tmp_path):
    with pytest.raises(CorpusNotReadyError, match="not found"):
        CorpusDB.open(tmp_path / "missing.sqlite", create=False)


def test_an_old_schema_version_is_refused_instead_of_silently_upgraded(tmp_path):
    db = CorpusDB.open(tmp_path / "c.sqlite")
    db.transaction_sync(lambda conn: conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'"))
    db.close()

    with pytest.raises(CorpusNotReadyError, match="schema version 0"):
        CorpusDB.open(tmp_path / "c.sqlite")


@pytest.mark.asyncio
async def test_readiness_reports_every_gap_with_an_actionable_hint(corpus_db):
    empty = corpus_db.readiness_sync(require_corpus=True)
    assert {
        "the corpus holds no documents",
        "the corpus holds no retrieval chunks",
        "the catalog (decision_cache) is empty",
    } <= set(empty.issues)
    assert corpus_db.readiness_sync(require_corpus=False).issues == []

    store = DocumentStore(corpus_db)
    await store.store_document(StoredDocument(document_id="1", title="T", markdown_content="MADDE 1\nMetin."))
    partial = corpus_db.readiness_sync(require_corpus=True)
    assert "a document with content has no current chunks (rebuild the corpus)" in partial.issues
    with pytest.raises(CorpusNotReadyError, match="bddk-mcp build"):
        await corpus_db.assert_ready(require_corpus=True)


def test_change_token_moves_after_a_write(tmp_path):
    db = CorpusDB.open(tmp_path / "c.sqlite")
    try:
        before = db.change_token()
        db.transaction_sync(lambda conn: conn.execute("INSERT INTO meta (key, value) VALUES ('x', 'y')"))
        assert db.change_token() != before
        assert db.change_token() == db.change_token()
    finally:
        db.close()
