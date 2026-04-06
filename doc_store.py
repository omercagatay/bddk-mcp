"""
SQLite + FTS5 document store for BDDK regulatory documents.

Provides local storage, full-text search, and pagination for BDDK decisions,
regulations, and mevzuat.gov.tr documents. Designed for offline-first usage
with optional sync to Railway deployment.

Schema includes a nullable `embedding` column for future vector search
(sqlite-vec / TCMB / banka-ici document expansion).
"""

import hashlib
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 5000
_DEFAULT_DB_PATH = Path(__file__).parent / "bddk_docs.db"

# ── Pydantic models ──────────────────────────────────────────────────────────


class StoredDocument(BaseModel):
    """A document stored in the local SQLite database."""

    document_id: str
    title: str
    category: str = ""
    decision_date: str = ""
    decision_number: str = ""
    source_url: str = ""
    pdf_bytes: Optional[bytes] = None
    markdown_content: str = ""
    content_hash: str = ""
    extraction_method: str = "markitdown"
    total_pages: int = 1
    file_size: int = 0

    model_config = {"arbitrary_types_allowed": True}


class DocumentPage(BaseModel):
    """A single page of a paginated document."""

    document_id: str
    title: str
    markdown_content: str
    page_number: int = 1
    total_pages: int = 1
    extraction_method: str = ""
    category: str = ""


class SearchHit(BaseModel):
    """A search result from FTS5 full-text search."""

    document_id: str
    title: str
    snippet: str = ""
    category: str = ""
    rank: float = 0.0
    decision_date: str = ""


class StoreStats(BaseModel):
    """Statistics about the document store."""

    total_documents: int = 0
    total_size_mb: float = 0.0
    categories: dict[str, int] = Field(default_factory=dict)
    extraction_methods: dict[str, int] = Field(default_factory=dict)
    oldest_document: Optional[str] = None
    newest_document: Optional[str] = None
    documents_needing_refresh: int = 0


# ── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    category          TEXT DEFAULT '',
    decision_date     TEXT DEFAULT '',
    decision_number   TEXT DEFAULT '',
    source_url        TEXT DEFAULT '',
    pdf_blob          BLOB,
    markdown_content  TEXT DEFAULT '',
    content_hash      TEXT DEFAULT '',
    downloaded_at     REAL,
    extracted_at      REAL,
    extraction_method TEXT DEFAULT 'markitdown',
    total_pages       INTEGER DEFAULT 1,
    file_size         INTEGER DEFAULT 0,
    embedding         BLOB
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(decision_date);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    markdown_content,
    category,
    content='documents',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

-- Triggers to keep FTS index in sync with documents table
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, markdown_content, category)
    VALUES (new.rowid, new.title, new.markdown_content, new.category);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, markdown_content, category)
    VALUES ('delete', old.rowid, old.title, old.markdown_content, old.category);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, markdown_content, category)
    VALUES ('delete', old.rowid, old.title, old.markdown_content, old.category);
    INSERT INTO documents_fts(rowid, title, markdown_content, category)
    VALUES (new.rowid, new.title, new.markdown_content, new.category);
END;
"""


def _content_hash(content: str) -> str:
    """SHA-256 hash of document content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ── DocumentStore ────────────────────────────────────────────────────────────


class DocumentStore:
    """
    Async SQLite + FTS5 document store.

    Usage::

        store = DocumentStore()
        await store.initialize()
        await store.store_document(doc)
        page = await store.get_document_page("1291", page=1)
        hits = await store.search_content("sermaye yeterliliği")
        await store.close()
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Open DB connection and create schema if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.commit()
        logger.info("DocumentStore initialized: %s", self._db_path)

    async def close(self) -> None:
        """Close DB connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("DocumentStore closed")

    async def __aenter__(self) -> "DocumentStore":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _ensure_open(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("DocumentStore not initialized. Call initialize() first.")
        return self._db

    # ── CRUD ─────────────────────────────────────────────────────────────

    async def store_document(self, doc: StoredDocument) -> None:
        """Insert or replace a document in the store."""
        db = self._ensure_open()
        now = time.time()
        content_hash = _content_hash(doc.markdown_content) if doc.markdown_content else ""
        total_pages = max(1, math.ceil(len(doc.markdown_content) / _CHUNK_SIZE)) if doc.markdown_content else 1

        await db.execute(
            """\
            INSERT INTO documents (
                document_id, title, category, decision_date, decision_number,
                source_url, pdf_blob, markdown_content, content_hash,
                downloaded_at, extracted_at, extraction_method, total_pages, file_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title,
                category=excluded.category,
                decision_date=excluded.decision_date,
                decision_number=excluded.decision_number,
                source_url=excluded.source_url,
                pdf_blob=COALESCE(excluded.pdf_blob, documents.pdf_blob),
                markdown_content=excluded.markdown_content,
                content_hash=excluded.content_hash,
                downloaded_at=excluded.downloaded_at,
                extracted_at=excluded.extracted_at,
                extraction_method=excluded.extraction_method,
                total_pages=excluded.total_pages,
                file_size=excluded.file_size
            """,
            (
                doc.document_id,
                doc.title,
                doc.category,
                doc.decision_date,
                doc.decision_number,
                doc.source_url,
                doc.pdf_bytes,
                doc.markdown_content,
                content_hash,
                now,
                now if doc.markdown_content else None,
                doc.extraction_method,
                total_pages,
                doc.file_size or (len(doc.pdf_bytes) if doc.pdf_bytes else 0),
            ),
        )
        await db.commit()
        logger.debug("Stored document %s (%s)", doc.document_id, doc.title[:60])

    async def get_document(self, doc_id: str) -> Optional[StoredDocument]:
        """Retrieve a full document by ID."""
        db = self._ensure_open()
        async with db.execute(
            "SELECT * FROM documents WHERE document_id = ?", (doc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return StoredDocument(
                document_id=row["document_id"],
                title=row["title"],
                category=row["category"] or "",
                decision_date=row["decision_date"] or "",
                decision_number=row["decision_number"] or "",
                source_url=row["source_url"] or "",
                pdf_bytes=row["pdf_blob"],
                markdown_content=row["markdown_content"] or "",
                content_hash=row["content_hash"] or "",
                extraction_method=row["extraction_method"] or "markitdown",
                total_pages=row["total_pages"] or 1,
                file_size=row["file_size"] or 0,
            )

    async def get_document_page(
        self, doc_id: str, page: int = 1
    ) -> Optional[DocumentPage]:
        """Retrieve a single paginated page of a document's markdown content."""
        db = self._ensure_open()
        async with db.execute(
            "SELECT document_id, title, markdown_content, extraction_method, category "
            "FROM documents WHERE document_id = ?",
            (doc_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

        md = row["markdown_content"] or ""
        total_pages = max(1, math.ceil(len(md) / _CHUNK_SIZE))

        if page < 1 or page > total_pages:
            return DocumentPage(
                document_id=doc_id,
                title=row["title"],
                markdown_content=f"Invalid page {page}. Document has {total_pages} page(s).",
                page_number=page,
                total_pages=total_pages,
                extraction_method=row["extraction_method"] or "",
                category=row["category"] or "",
            )

        start = (page - 1) * _CHUNK_SIZE
        chunk = md[start : start + _CHUNK_SIZE]

        return DocumentPage(
            document_id=doc_id,
            title=row["title"],
            markdown_content=chunk,
            page_number=page,
            total_pages=total_pages,
            extraction_method=row["extraction_method"] or "",
            category=row["category"] or "",
        )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted."""
        db = self._ensure_open()
        cursor = await db.execute(
            "DELETE FROM documents WHERE document_id = ?", (doc_id,)
        )
        await db.commit()
        return cursor.rowcount > 0

    # ── Search ───────────────────────────────────────────────────────────

    async def search_content(
        self, query: str, limit: int = 20, category: Optional[str] = None
    ) -> list[SearchHit]:
        """Full-text search across document titles and content using FTS5."""
        db = self._ensure_open()

        # Build FTS5 query: quote each term for safety, join with AND
        terms = query.strip().split()
        if not terms:
            return []
        fts_query = " AND ".join(f'"{t}"' for t in terms)

        sql = """\
            SELECT
                d.document_id,
                d.title,
                snippet(documents_fts, 1, '>>>', '<<<', '...', 40) AS snippet,
                d.category,
                d.decision_date,
                rank
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
        """
        params: list = [fts_query]

        if category:
            sql += " AND d.category = ?"
            params.append(category)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        hits: list[SearchHit] = []
        async with db.execute(sql, params) as cursor:
            async for row in cursor:
                hits.append(
                    SearchHit(
                        document_id=row["document_id"],
                        title=row["title"],
                        snippet=row["snippet"] or "",
                        category=row["category"] or "",
                        rank=row["rank"] or 0.0,
                        decision_date=row["decision_date"] or "",
                    )
                )

        logger.info("FTS search '%s': %d hits", query, len(hits))
        return hits

    # ── Utilities ────────────────────────────────────────────────────────

    async def needs_refresh(self, doc_id: str, max_age_days: int = 30) -> bool:
        """Check if a document needs to be re-downloaded/re-extracted."""
        db = self._ensure_open()
        async with db.execute(
            "SELECT downloaded_at, markdown_content FROM documents WHERE document_id = ?",
            (doc_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return True  # not in store at all
            if not row["markdown_content"]:
                return True  # downloaded but not extracted
            age_days = (time.time() - (row["downloaded_at"] or 0)) / 86400
            return age_days > max_age_days

    async def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store (with content)."""
        db = self._ensure_open()
        async with db.execute(
            "SELECT 1 FROM documents WHERE document_id = ? AND markdown_content != ''",
            (doc_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def import_from_cache(self, cache_items: list[dict]) -> int:
        """
        Import document metadata from the existing BddkApiClient cache.

        Only creates entries for documents not already in the store.
        Does NOT download content — use doc_sync for that.
        Returns the number of new entries created.
        """
        db = self._ensure_open()
        imported = 0
        for item in cache_items:
            doc_id = item.get("document_id", "")
            if not doc_id:
                continue
            # Skip if already exists
            async with db.execute(
                "SELECT 1 FROM documents WHERE document_id = ?", (doc_id,)
            ) as cursor:
                if await cursor.fetchone():
                    continue

            await db.execute(
                """\
                INSERT INTO documents (document_id, title, category, decision_date,
                    decision_number, source_url, downloaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    item.get("title", ""),
                    item.get("category", ""),
                    item.get("decision_date", ""),
                    item.get("decision_number", ""),
                    item.get("source_url", ""),
                    time.time(),
                ),
            )
            imported += 1

        await db.commit()
        logger.info("Imported %d items from cache", imported)
        return imported

    async def list_documents(
        self, category: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """List documents with basic metadata (no content)."""
        db = self._ensure_open()
        sql = """\
            SELECT document_id, title, category, decision_date,
                   extraction_method, total_pages, file_size,
                   downloaded_at, extracted_at
            FROM documents
        """
        params: list = []
        if category:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY downloaded_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        results = []
        async with db.execute(sql, params) as cursor:
            async for row in cursor:
                results.append(dict(row))
        return results

    async def stats(self) -> StoreStats:
        """Return statistics about the document store."""
        db = self._ensure_open()

        # Total docs and size
        async with db.execute(
            "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM documents"
        ) as cursor:
            row = await cursor.fetchone()
            total = row[0]
            total_size = row[1]

        # Categories
        categories: dict[str, int] = {}
        async with db.execute(
            "SELECT COALESCE(category, 'Unknown') AS cat, COUNT(*) FROM documents GROUP BY cat ORDER BY cat"
        ) as cursor:
            async for row in cursor:
                categories[row[0]] = row[1]

        # Extraction methods
        methods: dict[str, int] = {}
        async with db.execute(
            "SELECT COALESCE(extraction_method, 'none') AS m, COUNT(*) "
            "FROM documents WHERE markdown_content != '' GROUP BY m"
        ) as cursor:
            async for row in cursor:
                methods[row[0]] = row[1]

        # Date range
        async with db.execute(
            "SELECT MIN(downloaded_at), MAX(downloaded_at) FROM documents"
        ) as cursor:
            row = await cursor.fetchone()
            oldest = time.strftime("%Y-%m-%d", time.localtime(row[0])) if row[0] else None
            newest = time.strftime("%Y-%m-%d", time.localtime(row[1])) if row[1] else None

        # Needing refresh (no content or older than 30 days)
        threshold = time.time() - (30 * 86400)
        async with db.execute(
            "SELECT COUNT(*) FROM documents WHERE markdown_content = '' OR downloaded_at < ?",
            (threshold,),
        ) as cursor:
            needs_refresh = (await cursor.fetchone())[0]

        return StoreStats(
            total_documents=total,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            categories=categories,
            extraction_methods=methods,
            oldest_document=oldest,
            newest_document=newest,
            documents_needing_refresh=needs_refresh,
        )
