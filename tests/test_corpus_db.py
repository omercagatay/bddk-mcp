"""Tests for the corpus file itself: opening modes, schema guard, readiness."""

import hashlib
import sqlite3

import pytest

from bddk_mcp.core.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from bddk_mcp.store.corpus_db import (
    NOT_READY_HINT,
    SCHEMA_VERSION,
    CorpusDB,
    CorpusNotReadyError,
    corpus_digest,
)
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

    with pytest.raises(CorpusNotReadyError, match=message) as raised:
        await corpus_db.assert_ready(require_corpus=False)
    # A plain build and --force both refuse such a file; only a fresh file built with this model fixes it.
    assert "Delete the corpus file and run `bddk-mcp build`" in str(raised.value)
    assert "refuse to switch an existing file to another embedding model" in str(raised.value)
    assert "skips" not in str(raised.value)
    assert NOT_READY_HINT not in str(raised.value)


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


# -- A leftover write-ahead log beside a replaced file -----------------------------

_KILLED_WRITER = """
import os, sys
from bddk_mcp.store.corpus_db import CorpusDB
db = CorpusDB.open(sys.argv[1])
db.transaction_sync(lambda conn: conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('who', 'OLD')"))
os._exit(0)  # killed before close(): the committed frames stay in the -wal file
"""


def _killed_writer(path) -> None:
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", _KILLED_WRITER, str(path)], check=True)


def _who(db) -> str:
    return db.run_sync(lambda conn: conn.execute("SELECT value FROM meta WHERE key = 'who'").fetchone()[0])


def _replaced_after_a_killed_writer(tmp_path):
    corpus = tmp_path / "bddk.sqlite"
    _killed_writer(corpus)
    wal = tmp_path / "bddk.sqlite-wal"
    assert wal.stat().st_size > 0 and _header_journal_bytes(corpus) == (2, 2)
    new = tmp_path / "new.sqlite"
    db = CorpusDB.open(new)
    db.transaction_sync(lambda conn: conn.execute("INSERT INTO meta (key, value) VALUES ('who', 'NEW')"))
    db.close()
    new.replace(corpus)
    assert _header_journal_bytes(corpus) == (1, 1)
    return corpus


@pytest.mark.parametrize("readonly", [True, False])
def test_a_stale_wal_beside_a_replaced_file_is_refused_without_touching_it(tmp_path, readonly):
    corpus = _replaced_after_a_killed_writer(tmp_path)
    before = corpus.read_bytes()
    wal_before = (tmp_path / "bddk.sqlite-wal").read_bytes()

    with pytest.raises(CorpusNotReadyError) as raised:
        CorpusDB.open(corpus, readonly=readonly)

    text = str(raised.value)
    assert "bddk.sqlite-wal" in text and "bddk.sqlite-shm" in text
    assert "earlier file" in text and "on top of" in text
    assert "move or delete" in text.lower()
    assert corpus.read_bytes() == before
    assert (tmp_path / "bddk.sqlite-wal").read_bytes() == wal_before


def test_a_rollback_journal_file_with_an_empty_or_missing_wal_opens_normally(tmp_path):
    corpus = tmp_path / "c.sqlite"
    db = CorpusDB.open(corpus)
    db.transaction_sync(lambda conn: conn.execute("INSERT INTO meta (key, value) VALUES ('who', 'NEW')"))
    db.close()
    assert _header_journal_bytes(corpus) == (1, 1)

    reader = CorpusDB.open(corpus, readonly=True)
    assert _who(reader) == "NEW"
    reader.close()

    (tmp_path / "c.sqlite-wal").write_bytes(b"")
    reader = CorpusDB.open(corpus, readonly=True)
    assert _who(reader) == "NEW"
    reader.close()
    CorpusDB.open(corpus).close()


def test_a_wal_mode_file_with_its_own_wal_still_recovers_after_a_killed_writer(tmp_path):
    corpus = tmp_path / "bddk.sqlite"
    _killed_writer(corpus)
    assert (tmp_path / "bddk.sqlite-wal").stat().st_size > 0

    reader = CorpusDB.open(corpus, readonly=True)
    try:
        assert _who(reader) == "OLD"
    finally:
        reader.close()
    writer = CorpusDB.open(corpus)
    try:
        assert _who(writer) == "OLD"
    finally:
        writer.close()


def test_a_writer_refuses_a_foreign_database_with_a_key_value_meta_table_without_touching_it(tmp_path):
    other = tmp_path / "History"
    connection = sqlite3.connect(other)
    connection.execute("CREATE TABLE meta (key LONGVARCHAR NOT NULL UNIQUE PRIMARY KEY, value LONGVARCHAR)")
    connection.executemany("INSERT INTO meta VALUES (?, ?)", [("version", "58"), ("last_compatible_version", "16")])
    connection.execute("CREATE TABLE urls (id INTEGER PRIMARY KEY, url LONGVARCHAR)")
    connection.execute("INSERT INTO urls (url) VALUES ('https://example.invalid/')")
    connection.commit()
    connection.close()
    before = other.read_bytes()

    with pytest.raises(CorpusNotReadyError, match=r"not a corpus file.*schema_version.*left unchanged"):
        CorpusDB.open(other)

    assert other.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["History"]


def test_a_writer_accepts_a_half_initialised_corpus_without_a_schema_version_row(tmp_path):
    corpus = tmp_path / "c.sqlite"
    CorpusDB.open(corpus).close()
    connection = sqlite3.connect(corpus)
    connection.execute("DELETE FROM meta")
    connection.commit()
    connection.close()

    db = CorpusDB.open(corpus)
    try:
        version = db.run_sync(lambda conn: conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone())
        assert version[0] == str(SCHEMA_VERSION)
    finally:
        db.close()


def _refusing_connect(real_connect):
    def connect(database, *args, **kwargs):
        if kwargs.get("uri") and "mode=ro" in str(database):
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(database, *args, **kwargs)

    return connect


def test_a_file_the_reader_may_not_open_gives_the_unreadable_message(tmp_path, monkeypatch):
    from bddk_mcp.store.corpus_db import UNREADABLE_MESSAGE

    CorpusDB.open(tmp_path / "c.sqlite").close()
    monkeypatch.setattr(sqlite3, "connect", _refusing_connect(sqlite3.connect))

    with pytest.raises(CorpusNotReadyError) as raised:
        CorpusDB.open(tmp_path / "c.sqlite", readonly=True)

    assert str(raised.value) == UNREADABLE_MESSAGE


@pytest.mark.asyncio
async def test_readiness_counts_sync_failures_by_category(corpus_db):
    assert corpus_db.readiness_sync(require_corpus=False).sync_failures == {}

    store = DocumentStore(corpus_db)
    await store.record_sync_failure("1", "timeout", "download")
    await store.record_sync_failure("1", "timeout", "download")  # a retry of the same document counts once
    await store.record_sync_failure("2", "timeout", "download")
    await store.record_sync_failure("3", "empty", "extraction")

    readiness = corpus_db.readiness_sync(require_corpus=False)
    assert readiness.sync_failures == {"download": 2, "extraction": 1}
    assert readiness.issues == []  # failures are reported, they do not make the corpus unready


def _store_fingerprint(corpus_db, value) -> None:
    from bddk_mcp.store import fingerprint

    corpus_db.transaction_sync(
        lambda conn: conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (fingerprint.FINGERPRINT_META_KEY, value)
        )
    )


@pytest.mark.asyncio
async def test_readiness_reports_the_build_fingerprint_and_its_stale_parts_as_a_warning_only(corpus_db):
    from bddk_mcp.store import fingerprint

    await _corpus_with_one_chunk(corpus_db)
    absent = await corpus_db.assert_ready(require_corpus=False)
    assert (absent.build_fingerprint, absent.stale_fingerprint_parts) == (None, [])

    current = {"seed": "0" * 16, **fingerprint.runtime_parts()}
    _store_fingerprint(corpus_db, fingerprint.encode(current))
    ready = await corpus_db.assert_ready(require_corpus=False)
    assert ready.build_fingerprint == current and ready.stale_fingerprint_parts == []

    _store_fingerprint(corpus_db, fingerprint.encode({**current, "parser": "1" * 16, "chunker": "2" * 16}))
    stale = await corpus_db.assert_ready(require_corpus=False)  # a stale fingerprint does not fail readiness
    assert stale.stale_fingerprint_parts == ["parser", "chunker"]
    assert stale.issues == []


def test_readiness_never_lists_the_seed_part_and_skips_unknown_parts(corpus_db, monkeypatch):
    from bddk_mcp.store import fingerprint

    current = fingerprint.runtime_parts()
    _store_fingerprint(corpus_db, fingerprint.encode({"seed": "x", **current, "chunker": fingerprint.UNKNOWN}))
    assert corpus_db.readiness_sync(require_corpus=False).stale_fingerprint_parts == []

    monkeypatch.setattr(fingerprint, "_module_source", lambda name: None)
    _store_fingerprint(corpus_db, fingerprint.encode({"seed": "x", **current, "parser": "1" * 16}))
    assert corpus_db.readiness_sync(require_corpus=False).stale_fingerprint_parts == []
