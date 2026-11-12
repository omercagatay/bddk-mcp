"""The corpus is one SQLite file: schema, connection handling and readiness.

Serving opens the file read-only (``mode=ro``), which is the airlock: a
serving process cannot write the corpus even by mistake.  ``bddk-mcp build``
and ``bddk-mcp sync`` open it read-write and are the only writers.

Full-text search uses FTS5 over normalized shadow columns (``*_norm``): text
is lowercased with Turkish rules and stripped of diacritics in Python before
it is indexed and before a query is matched, so ranking does not depend on
SQLite's own case folding.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from bddk_mcp.core.config import CORPUS_PATH, EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
NOT_READY_HINT = "Run `bddk-mcp build` (seed_data -> corpus file) or `bddk-mcp sync` to prepare the corpus."

_T = TypeVar("_T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    decision_date TEXT NOT NULL DEFAULT '',
    decision_number TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    pdf_blob BLOB,
    markdown_content TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    downloaded_at REAL,
    extracted_at REAL,
    extraction_method TEXT NOT NULL DEFAULT '',
    total_pages INTEGER NOT NULL DEFAULT 1,
    file_size INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents (category);
CREATE TABLE IF NOT EXISTS document_versions (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    markdown_content TEXT NOT NULL DEFAULT '',
    synced_at REAL NOT NULL,
    PRIMARY KEY (document_id, version)
);
CREATE TABLE IF NOT EXISTS document_sections (
    doc_id TEXT NOT NULL,
    section_type TEXT NOT NULL,
    section_ref TEXT NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    source_content_hash TEXT NOT NULL,
    heading_norm TEXT NOT NULL DEFAULT '',
    content_norm TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON document_sections (doc_id, start_char);
CREATE INDEX IF NOT EXISTS idx_sections_ref ON document_sections (section_type, section_ref);
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
    heading_norm, content_norm,
    content='document_sections', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS document_sections_ai AFTER INSERT ON document_sections BEGIN
    INSERT INTO sections_fts(rowid, heading_norm, content_norm)
    VALUES (new.rowid, new.heading_norm, new.content_norm);
END;
CREATE TRIGGER IF NOT EXISTS document_sections_ad AFTER DELETE ON document_sections BEGIN
    INSERT INTO sections_fts(sections_fts, rowid, heading_norm, content_norm)
    VALUES ('delete', old.rowid, old.heading_norm, old.content_norm);
END;
CREATE TABLE IF NOT EXISTS chunks (
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    chunk_text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    section_type TEXT NOT NULL DEFAULT '',
    section_ref TEXT NOT NULL DEFAULT '',
    section_start_char INTEGER,
    section_end_char INTEGER,
    section_content_hash TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    title_norm TEXT NOT NULL DEFAULT '',
    chunk_text_norm TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (doc_id, chunk_index)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title_norm, chunk_text_norm,
    content='chunks', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title_norm, chunk_text_norm)
    VALUES (new.rowid, new.title_norm, new.chunk_text_norm);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title_norm, chunk_text_norm)
    VALUES ('delete', old.rowid, old.title_norm, old.chunk_text_norm);
END;
CREATE TABLE IF NOT EXISTS decision_cache (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    decision_date TEXT NOT NULL DEFAULT '',
    decision_number TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    cached_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sync_failures (
    document_id TEXT PRIMARY KEY,
    error TEXT NOT NULL,
    error_category TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    retryable INTEGER NOT NULL DEFAULT 1,
    attempts INTEGER NOT NULL DEFAULT 1,
    first_failed_at REAL NOT NULL,
    last_failed_at REAL NOT NULL
);
"""

_READINESS_CHECKS: tuple[tuple[str, str], ...] = (
    (
        "a document with content has no current chunks (rebuild the corpus)",
        "SELECT 1 FROM documents AS d WHERE d.markdown_content <> '' AND NOT EXISTS ("
        "SELECT 1 FROM chunks AS c WHERE c.doc_id = d.document_id AND c.content_hash = d.content_hash) LIMIT 1",
    ),
    (
        "a chunk belongs to a missing or changed document (rebuild the corpus)",
        "SELECT 1 FROM chunks AS c WHERE NOT EXISTS ("
        "SELECT 1 FROM documents AS d WHERE d.document_id = c.doc_id AND d.content_hash = c.content_hash) LIMIT 1",
    ),
    (
        "a section belongs to a missing or changed document (rebuild the corpus)",
        "SELECT 1 FROM document_sections AS s WHERE NOT EXISTS ("
        "SELECT 1 FROM documents AS d WHERE d.document_id = s.doc_id AND d.content_hash = s.source_content_hash) "
        "LIMIT 1",
    ),
)


class CorpusNotReadyError(RuntimeError):
    """The corpus file is missing, incomplete or built with another embedding model."""


@dataclass
class CorpusReadiness:
    documents: int = 0
    chunks: int = 0
    catalog: int = 0
    embedding_model: str = ""
    issues: list[str] = field(default_factory=list)


class CorpusDB:
    """One SQLite connection, serialized behind a lock, driven from asyncio threads.

    ponytail: a single shared connection and a global lock keep this trivial;
    per-thread read connections are the upgrade if query throughput ever matters.
    """

    def __init__(self, connection: sqlite3.Connection, *, path: str, readonly: bool) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._write_epoch = 0
        self.path = path
        self.readonly = readonly

    @classmethod
    def open(cls, path: str | Path | None = None, *, readonly: bool = False, create: bool = True) -> CorpusDB:
        """Open the corpus file.

        Writers create or upgrade the schema; ``create=False`` makes a writer
        refuse a missing file instead of silently starting an empty corpus
        (operator scripts that repair an existing corpus use it).
        """
        target = Path(path or CORPUS_PATH)
        if readonly or not create:
            if not target.is_file():
                raise CorpusNotReadyError(f"Corpus file not found: {target}. {NOT_READY_HINT}")
        if readonly:
            uri = f"{target.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(target), check_same_thread=False, isolation_level=None)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            stored = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if stored is not None and stored[0] != str(SCHEMA_VERSION):
                # CREATE TABLE IF NOT EXISTS never alters an existing layout: an
                # old file is rebuilt from seed data, not upgraded in place.
                connection.close()
                raise CorpusNotReadyError(
                    f"Corpus file {target} has schema version {stored[0]}, this runtime expects {SCHEMA_VERSION}. "
                    "Delete the file and run `bddk-mcp build`."
                )
            connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        connection.row_factory = sqlite3.Row
        logger.info("Corpus opened (%s)", "read-only" if readonly else "read-write")
        return cls(connection, path=str(target), readonly=readonly)

    # -- execution ---------------------------------------------------------------

    def run_sync(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run read-only SQL work on the calling thread under the connection lock."""
        with self._lock:
            return fn(self._conn)

    async def run(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run read-only SQL work on a worker thread so the event loop stays free."""
        return await asyncio.to_thread(self.run_sync, fn)

    def transaction_sync(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run ``fn`` inside one immediate write transaction."""
        if self.readonly:
            raise CorpusNotReadyError("The corpus was opened read-only; writes belong to `bddk-mcp build`/`sync`.")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(self._conn)
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")
            self._write_epoch += 1
            return result

    async def transaction(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        return await asyncio.to_thread(self.transaction_sync, fn)

    def change_token(self) -> tuple[int, int]:
        """Cheap token that changes after any write by this or another connection."""
        with self._lock:
            data_version = self._conn.execute("PRAGMA data_version").fetchone()[0]
            return self._write_epoch, int(data_version)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- readiness ---------------------------------------------------------------

    def readiness_sync(self, *, require_corpus: bool = True) -> CorpusReadiness:
        with self._lock:
            conn = self._conn
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
            readiness = CorpusReadiness(embedding_model=meta.get("embedding_model", ""))
            if meta.get("schema_version") != str(SCHEMA_VERSION):
                readiness.issues.append("the corpus file has no recognised schema")
                return readiness
            readiness.documents = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE markdown_content <> ''"
            ).fetchone()[0]
            readiness.chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            readiness.catalog = conn.execute("SELECT COUNT(*) FROM decision_cache").fetchone()[0]
            if readiness.chunks:
                if readiness.embedding_model != EMBEDDING_MODEL_NAME:
                    readiness.issues.append("the corpus was embedded with a different model than BDDK_EMBEDDING_MODEL")
                if meta.get("embedding_dimension") != str(EMBEDDING_DIMENSION):
                    readiness.issues.append("the corpus embedding dimension does not match this runtime")
            for message, sql in _READINESS_CHECKS:
                if conn.execute(sql).fetchone() is not None:
                    readiness.issues.append(message)
            if require_corpus:
                if not readiness.documents:
                    readiness.issues.append("the corpus holds no documents")
                if not readiness.chunks:
                    readiness.issues.append("the corpus holds no retrieval chunks")
                if not readiness.catalog:
                    readiness.issues.append("the catalog (decision_cache) is empty")
            return readiness

    async def assert_ready(self, *, require_corpus: bool = True) -> CorpusReadiness:
        """Raise an actionable error unless the corpus can be served."""
        try:
            readiness = await asyncio.to_thread(self.readiness_sync, require_corpus=require_corpus)
        except sqlite3.DatabaseError as exc:
            raise CorpusNotReadyError(
                f"Corpus is not ready: the file is not a readable corpus. {NOT_READY_HINT}"
            ) from exc
        if readiness.issues:
            raise CorpusNotReadyError("Corpus is not ready: " + "; ".join(readiness.issues) + f". {NOT_READY_HINT}")
        return readiness
