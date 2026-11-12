"""Tests for the corpus file itself: opening modes, schema guard, readiness."""

import hashlib
import sqlite3

import pytest

from bddk_mcp.core.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from bddk_mcp.store.corpus_db import SCHEMA_VERSION, CorpusDB, CorpusNotReadyError, corpus_digest
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

    before = (tmp_path / "c.sqlite").read_bytes()

    with pytest.raises(CorpusNotReadyError, match="schema version 0"):
        CorpusDB.open(tmp_path / "c.sqlite")
    assert (tmp_path / "c.sqlite").read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["c.sqlite"]


def test_a_writer_refuses_an_sqlite_database_that_is_not_a_corpus_without_touching_it(tmp_path):
    other = tmp_path / "other.sqlite"
    connection = sqlite3.connect(other)
    connection.execute("CREATE TABLE findings (id INTEGER PRIMARY KEY, note TEXT)")
    connection.execute("INSERT INTO findings (note) VALUES ('keep me')")
    connection.commit()
    connection.close()
    before = other.read_bytes()

    with pytest.raises(CorpusNotReadyError, match="not a corpus file"):
        CorpusDB.open(other)

    assert other.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["other.sqlite"]


def test_a_writer_refuses_a_file_that_is_not_sqlite_without_touching_it(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_bytes(b"not a database, " * 64)

    with pytest.raises(CorpusNotReadyError, match="left unchanged"):
        CorpusDB.open(other)

    assert other.read_bytes() == b"not a database, " * 64


def test_a_writer_gives_an_empty_file_the_schema(tmp_path):
    empty = tmp_path / "c.sqlite"
    empty.write_bytes(b"")

    db = CorpusDB.open(empty)
    try:
        version = db.run_sync(lambda conn: conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone())
        assert version[0] == str(SCHEMA_VERSION)
    finally:
        db.close()


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
    with pytest.raises(CorpusNotReadyError, match="bddk-mcp build") as raised:
        await corpus_db.assert_ready(require_corpus=True)
    # sync re-extracts text from live sources; it is not the remedy for a file without chunks.
    assert "sync" not in str(raised.value)


def test_change_token_moves_after_a_write(tmp_path):
    db = CorpusDB.open(tmp_path / "c.sqlite")
    try:
        before = db.change_token()
        db.transaction_sync(lambda conn: conn.execute("INSERT INTO meta (key, value) VALUES ('x', 'y')"))
        assert db.change_token() != before
        assert db.change_token() == db.change_token()
    finally:
        db.close()


def _header_journal_bytes(path) -> tuple[int, int]:
    header = path.read_bytes()[:20]
    return header[18], header[19]


def test_a_closed_writer_leaves_a_self_contained_rollback_journal_file(tmp_path):
    corpus = tmp_path / "c.sqlite"
    db = CorpusDB.open(corpus)
    db.transaction_sync(lambda conn: conn.execute("INSERT INTO meta (key, value) VALUES ('x', 'y')"))
    assert db.run_sync(lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0]) == "wal"
    db.close()

    assert _header_journal_bytes(corpus) == (1, 1)  # legacy (rollback journal), not WAL (2, 2)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["c.sqlite"]


@pytest.mark.asyncio
async def test_a_read_only_open_with_readiness_creates_no_side_files(tmp_path):
    corpus = tmp_path / "c.sqlite"
    CorpusDB.open(corpus).close()

    db = CorpusDB.open(corpus, readonly=True)
    try:
        db.readiness_sync(require_corpus=False)
        await db.assert_ready(require_corpus=False)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["c.sqlite"]
    finally:
        db.close()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["c.sqlite"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message", ["attempt to write a readonly database", "unable to open database file", "Permission denied"]
)
async def test_an_access_failure_is_reported_as_permissions_not_as_a_broken_corpus(tmp_path, message):
    CorpusDB.open(tmp_path / "c.sqlite").close()
    db = CorpusDB.open(tmp_path / "c.sqlite", readonly=True)

    def refuse(*, require_corpus: bool = True):
        raise sqlite3.OperationalError(message)

    db.readiness_sync = refuse
    try:
        with pytest.raises(CorpusNotReadyError) as raised:
            await db.assert_ready(require_corpus=True)
    finally:
        db.close()
    text = str(raised.value)
    assert "cannot open the corpus file for reading" in text and "permissions" in text
    assert "bddk-mcp build" not in text and "not a readable corpus" not in text


@pytest.mark.asyncio
async def test_readiness_digest_identifies_the_document_texts(corpus_db):
    store = DocumentStore(corpus_db)
    await store.store_document(StoredDocument(document_id="1", title="T", markdown_content="MADDE 1\nMetin."))
    await store.store_document(StoredDocument(document_id="2", title="U", markdown_content="MADDE 2\nBaşka."))
    first = corpus_db.readiness_sync(require_corpus=False)
    hashes = sorted(
        f"{doc.document_id}:{doc.content_hash}"
        for doc in [await store.get_document("1"), await store.get_document("2")]
    )

    assert first.digest == hashlib.sha256("\n".join(hashes).encode()).hexdigest()[:12]
    assert first.digest == corpus_digest([("2", hashes[1].split(":")[1]), ("1", hashes[0].split(":")[1])])
    assert first.schema_version == str(SCHEMA_VERSION)

    await store.store_document(StoredDocument(document_id="1", title="Other title", markdown_content="MADDE 1\nMetin."))
    assert corpus_db.readiness_sync(require_corpus=False).digest == first.digest  # metadata is not part of it

    await store.store_document(StoredDocument(document_id="1", title="T", markdown_content="MADDE 1\nDeğişti."))
    assert corpus_db.readiness_sync(require_corpus=False).digest != first.digest


async def _corpus_with_one_chunk(corpus_db):
    """A corpus whose only document has a matching chunk and a matching section."""
    store = DocumentStore(corpus_db)
    await store.store_document(StoredDocument(document_id="1", title="T", markdown_content="MADDE 1\nMetin."))
    content_hash = corpus_db.run_sync(lambda conn: conn.execute("SELECT content_hash FROM documents").fetchone()[0])

    def add_chunk(conn):
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, chunk_text, start_char, end_char, content_hash, embedding) "
            "VALUES ('1', 0, 'MADDE 1', 0, 7, ?, ?)",
            (content_hash, b"\x00\x00"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_model', ?), ('embedding_dimension', ?)",
            (EMBEDDING_MODEL_NAME, str(EMBEDDING_DIMENSION)),
        )

    corpus_db.transaction_sync(add_chunk)
    return content_hash


@pytest.mark.asyncio
async def test_a_consistent_corpus_with_a_chunk_is_ready(corpus_db):
    await _corpus_with_one_chunk(corpus_db)

    assert (await corpus_db.assert_ready(require_corpus=False)).issues == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("embedding_model", "other/model", "different model"),
        ("embedding_dimension", "384", "embedding dimension"),
    ],
)
async def test_readiness_refuses_a_corpus_embedded_with_another_model_or_dimension(corpus_db, key, value, message):
    await _corpus_with_one_chunk(corpus_db)
    corpus_db.transaction_sync(lambda conn: conn.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key)))

    with pytest.raises(CorpusNotReadyError, match=message):
        await corpus_db.assert_ready(require_corpus=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("orphan", ["missing_document", "changed_document_hash"])
async def test_readiness_refuses_a_chunk_that_matches_no_document(corpus_db, orphan):
    content_hash = await _corpus_with_one_chunk(corpus_db)
    if orphan == "missing_document":
        corpus_db.transaction_sync(lambda conn: conn.execute("UPDATE chunks SET doc_id = '999'"))
    else:
        corpus_db.transaction_sync(lambda conn: conn.execute("UPDATE chunks SET content_hash = 'stale'"))
    assert content_hash != "stale"

    with pytest.raises(CorpusNotReadyError, match="chunk belongs to a missing or changed document"):
        await corpus_db.assert_ready(require_corpus=False)


@pytest.mark.asyncio
async def test_readiness_refuses_a_section_with_a_stale_source_hash(corpus_db):
    await _corpus_with_one_chunk(corpus_db)
    count = corpus_db.run_sync(lambda conn: conn.execute("SELECT COUNT(*) FROM document_sections").fetchone()[0])
    assert count > 0
    corpus_db.transaction_sync(lambda conn: conn.execute("UPDATE document_sections SET source_content_hash = 'stale'"))

    with pytest.raises(CorpusNotReadyError, match="section belongs to a missing or changed document"):
        await corpus_db.assert_ready(require_corpus=False)
