"""
PostgreSQL + tsvector document store for BDDK regulatory documents.

Provides storage, full-text search (Turkish tsvector), pagination, and
versioning for BDDK decisions, regulations, and mevzuat.gov.tr documents.

Requires: asyncpg, PostgreSQL 14+ with unaccent extension.
"""

import hashlib
import logging
import math
import re
import time

import asyncpg
from pydantic import BaseModel, Field

from config import FTS_RANK_THRESHOLD, PAGE_SIZE

logger = logging.getLogger(__name__)

# -- Pydantic models ----------------------------------------------------------


class StoredDocument(BaseModel):
    """A document stored in the PostgreSQL database."""

    document_id: str
    title: str
    category: str = ""
    decision_date: str = ""
    decision_number: str = ""
    source_url: str = ""
    pdf_bytes: bytes | None = None
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
    """A search result from full-text search."""

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
    oldest_document: str | None = None
    newest_document: str | None = None
    documents_needing_refresh: int = 0


# -- Schema -------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Make unaccent() usable in immutable contexts (triggers, indexes)
CREATE OR REPLACE FUNCTION immutable_unaccent(text)
RETURNS text AS $$
    SELECT unaccent($1);
$$ LANGUAGE sql IMMUTABLE PARALLEL SAFE;

CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    category          TEXT DEFAULT '',
    decision_date     TEXT DEFAULT '',
    decision_number   TEXT DEFAULT '',
    source_url        TEXT DEFAULT '',
    pdf_blob          BYTEA,
    markdown_content  TEXT DEFAULT '',
    content_hash      TEXT DEFAULT '',
    downloaded_at     DOUBLE PRECISION,
    extracted_at      DOUBLE PRECISION,
    extraction_method TEXT DEFAULT 'markitdown',
    total_pages       INTEGER DEFAULT 1,
    file_size         INTEGER DEFAULT 0,
    tsv               tsvector
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(decision_date);
CREATE INDEX IF NOT EXISTS idx_documents_tsv ON documents USING GIN(tsv);

-- Trigger to keep tsv column in sync
CREATE OR REPLACE FUNCTION documents_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv :=
        to_tsvector('simple', immutable_unaccent(coalesce(NEW.title, '')))
        || to_tsvector('simple', immutable_unaccent(coalesce(NEW.markdown_content, '')))
        || to_tsvector('simple', immutable_unaccent(coalesce(NEW.category, '')));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_tsv ON documents;
CREATE TRIGGER trg_documents_tsv
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_tsv_trigger();

CREATE TABLE IF NOT EXISTS document_versions (
    id                SERIAL PRIMARY KEY,
    document_id       TEXT NOT NULL,
    version           INTEGER NOT NULL DEFAULT 1,
    content_hash      TEXT NOT NULL,
    markdown_content  TEXT DEFAULT '',
    synced_at         DOUBLE PRECISION NOT NULL,
    UNIQUE(document_id, version)
);

CREATE INDEX IF NOT EXISTS idx_versions_doc_id ON document_versions(document_id);

CREATE TABLE IF NOT EXISTS sync_metadata (
    document_id       TEXT PRIMARY KEY,
    etag              TEXT DEFAULT '',
    last_modified     TEXT DEFAULT '',
    last_sync_at      DOUBLE PRECISION,
    sync_count        INTEGER DEFAULT 0
);
"""


def _content_hash(content: str) -> str:
    """SHA-256 hash of document content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# -- DocumentStore ------------------------------------------------------------


class DocumentStore:
    """
    Async PostgreSQL document store with tsvector full-text search.

    Usage::

        store = DocumentStore(pool)
        await store.initialize()
        await store.store_document(doc)
        page = await store.get_document_page("1291", page=1)
        hits = await store.search_content("sermaye yeterliliği")
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        """Create schema if needed."""
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)
        logger.info("DocumentStore initialized (PostgreSQL)")

    async def close(self) -> None:
        """No-op — pool lifecycle is managed externally."""
        logger.info("DocumentStore closed")

    async def __aenter__(self) -> "DocumentStore":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- CRUD -----------------------------------------------------------------

    async def store_document(self, doc: StoredDocument) -> None:
        """Insert or replace a document in the store."""
        now = time.time()
        content_hash = _content_hash(doc.markdown_content) if doc.markdown_content else ""
        total_pages = max(1, math.ceil(len(doc.markdown_content) / PAGE_SIZE)) if doc.markdown_content else 1

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Archive previous version if content changed
                if content_hash and doc.markdown_content:
                    existing = await conn.fetchrow(
                        "SELECT content_hash, markdown_content FROM documents WHERE document_id = $1",
                        doc.document_id,
                    )
                    if existing and existing["content_hash"] and existing["content_hash"] != content_hash:
                        max_ver = await conn.fetchval(
                            "SELECT COALESCE(MAX(version), 0) FROM document_versions WHERE document_id = $1",
                            doc.document_id,
                        )
                        await conn.execute(
                            "INSERT INTO document_versions (document_id, version, content_hash, markdown_content, synced_at) "
                            "VALUES ($1, $2, $3, $4, $5)",
                            doc.document_id,
                            max_ver + 1,
                            existing["content_hash"],
                            existing["markdown_content"],
                            now,
                        )

                await conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, title, category, decision_date, decision_number,
                        source_url, pdf_blob, markdown_content, content_hash,
                        downloaded_at, extracted_at, extraction_method, total_pages, file_size
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT(document_id) DO UPDATE SET
                        title=EXCLUDED.title,
                        category=EXCLUDED.category,
                        decision_date=EXCLUDED.decision_date,
                        decision_number=EXCLUDED.decision_number,
                        source_url=EXCLUDED.source_url,
                        pdf_blob=COALESCE(EXCLUDED.pdf_blob, documents.pdf_blob),
                        markdown_content=EXCLUDED.markdown_content,
                        content_hash=EXCLUDED.content_hash,
                        downloaded_at=EXCLUDED.downloaded_at,
                        extracted_at=EXCLUDED.extracted_at,
                        extraction_method=EXCLUDED.extraction_method,
                        total_pages=EXCLUDED.total_pages,
                        file_size=EXCLUDED.file_size
                    """,
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
                )

        logger.debug("Stored document %s (%s)", doc.document_id, doc.title[:60])

    async def get_document(self, doc_id: str) -> StoredDocument | None:
        """Retrieve a full document by ID."""
        row = await self._pool.fetchrow("SELECT * FROM documents WHERE document_id = $1", doc_id)
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

    async def get_document_page(self, doc_id: str, page: int = 1) -> DocumentPage | None:
        """Retrieve a single paginated page of a document's markdown content."""
        row = await self._pool.fetchrow(
            "SELECT document_id, title, markdown_content, extraction_method, category "
            "FROM documents WHERE document_id = $1",
            doc_id,
        )
        if not row:
            return None

        md = row["markdown_content"] or ""
        total_pages = max(1, math.ceil(len(md) / PAGE_SIZE))

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

        start = (page - 1) * PAGE_SIZE
        chunk = md[start : start + PAGE_SIZE]

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
        result = await self._pool.execute("DELETE FROM documents WHERE document_id = $1", doc_id)
        return result == "DELETE 1"

    # -- Search ---------------------------------------------------------------

    @staticmethod
    def _sanitize_fts_term(term: str) -> str:
        """Sanitize a single term for safe use in tsquery."""
        sanitized = re.sub(r'["\*\(\)\^\+\-\!\&\|\<\>:]', "", term)
        if sanitized.upper() in ("AND", "OR", "NOT", "NEAR"):
            return ""
        return sanitized.strip()

    async def search_content(self, query: str, limit: int = 20, category: str | None = None) -> list[SearchHit]:
        """Full-text search across document titles and content using tsvector."""
        terms = [self._sanitize_fts_term(t) for t in query.strip().split()]
        terms = [t for t in terms if t]
        if not terms:
            return []

        # Build tsquery: each term joined with &
        tsquery = " & ".join(f"unaccent('{t}')" for t in terms)
        tsquery_expr = f"to_tsquery('simple', {tsquery})"

        # Use plainto_tsquery for safety, with unaccent on the query
        safe_query = " ".join(terms)

        sql = """
            SELECT
                d.document_id,
                d.title,
                ts_headline('simple', d.markdown_content,
                    plainto_tsquery('simple', immutable_unaccent($1)),
                    'StartSel=>>>, StopSel=<<<, MaxWords=40, MinWords=20'
                ) AS snippet,
                d.category,
                d.decision_date,
                ts_rank_cd(d.tsv, plainto_tsquery('simple', immutable_unaccent($1))) AS rank
            FROM documents d
            WHERE d.tsv @@ plainto_tsquery('simple', immutable_unaccent($1))
        """
        params: list = [safe_query]

        if category:
            sql += " AND d.category = $2"
            params.append(category)

        sql += " ORDER BY rank DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)

        rows = await self._pool.fetch(sql, *params)
        hits = [
            SearchHit(
                document_id=row["document_id"],
                title=row["title"],
                snippet=row["snippet"] or "",
                category=row["category"] or "",
                rank=row["rank"] or 0.0,
                decision_date=row["decision_date"] or "",
            )
            for row in rows
            if (row["rank"] or 0.0) >= FTS_RANK_THRESHOLD
        ]

        logger.info("FTS search '%s': %d hits", query, len(hits))
        return hits

    # -- Utilities ------------------------------------------------------------

    async def needs_refresh(self, doc_id: str, max_age_days: int = 30) -> bool:
        """Check if a document needs to be re-downloaded/re-extracted."""
        row = await self._pool.fetchrow(
            "SELECT downloaded_at, markdown_content FROM documents WHERE document_id = $1",
            doc_id,
        )
        if not row:
            return True
        if not row["markdown_content"]:
            return True
        age_days = (time.time() - (row["downloaded_at"] or 0)) / 86400
        return age_days > max_age_days

    async def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store (with content)."""
        row = await self._pool.fetchval(
            "SELECT 1 FROM documents WHERE document_id = $1 AND markdown_content != ''",
            doc_id,
        )
        return row is not None

    async def import_from_cache(self, cache_items: list[dict]) -> int:
        """Import document metadata from BddkApiClient cache.

        Only creates entries for documents not already in the store.
        Does NOT download content -- use doc_sync for that.
        """
        imported = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for item in cache_items:
                    doc_id = item.get("document_id", "")
                    if not doc_id:
                        continue
                    existing = await conn.fetchval(
                        "SELECT 1 FROM documents WHERE document_id = $1", doc_id
                    )
                    if existing:
                        continue

                    await conn.execute(
                        """
                        INSERT INTO documents (document_id, title, category, decision_date,
                            decision_number, source_url, downloaded_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        doc_id,
                        item.get("title", ""),
                        item.get("category", ""),
                        item.get("decision_date", ""),
                        item.get("decision_number", ""),
                        item.get("source_url", ""),
                        time.time(),
                    )
                    imported += 1

        logger.info("Imported %d items from cache", imported)
        return imported

    async def list_documents(self, category: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        """List documents with basic metadata (no content)."""
        sql = """
            SELECT document_id, title, category, decision_date,
                   extraction_method, total_pages, file_size,
                   downloaded_at, extracted_at
            FROM documents
        """
        params: list = []
        if category:
            sql += " WHERE category = $1"
            params.append(category)
        sql += f" ORDER BY downloaded_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.extend([limit, offset])

        rows = await self._pool.fetch(sql, *params)
        return [dict(row) for row in rows]

    async def stats(self) -> StoreStats:
        """Return statistics about the document store."""
        row = await self._pool.fetchrow("SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM documents")
        total = row[0]
        total_size = row[1]

        categories: dict[str, int] = {}
        rows = await self._pool.fetch(
            "SELECT COALESCE(category, 'Unknown') AS cat, COUNT(*) AS cnt FROM documents GROUP BY cat ORDER BY cat"
        )
        for r in rows:
            categories[r["cat"]] = r["cnt"]

        methods: dict[str, int] = {}
        rows = await self._pool.fetch(
            "SELECT COALESCE(extraction_method, 'none') AS m, COUNT(*) AS cnt "
            "FROM documents WHERE markdown_content != '' GROUP BY m"
        )
        for r in rows:
            methods[r["m"]] = r["cnt"]

        row = await self._pool.fetchrow("SELECT MIN(downloaded_at), MAX(downloaded_at) FROM documents")
        oldest = time.strftime("%Y-%m-%d", time.localtime(row[0])) if row[0] else None
        newest = time.strftime("%Y-%m-%d", time.localtime(row[1])) if row[1] else None

        threshold = time.time() - (30 * 86400)
        needs_refresh = await self._pool.fetchval(
            "SELECT COUNT(*) FROM documents WHERE markdown_content = '' OR downloaded_at < $1",
            threshold,
        )

        return StoreStats(
            total_documents=total,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            categories=categories,
            extraction_methods=methods,
            oldest_document=oldest,
            newest_document=newest,
            documents_needing_refresh=needs_refresh,
        )

    # -- Document Versioning --------------------------------------------------

    async def get_document_history(self, doc_id: str) -> list[dict]:
        """Get version history for a document."""
        rows = await self._pool.fetch(
            "SELECT version, content_hash, markdown_content, synced_at "
            "FROM document_versions WHERE document_id = $1 ORDER BY version DESC",
            doc_id,
        )
        return [
            {
                "version": row["version"],
                "content_hash": row["content_hash"],
                "synced_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(row["synced_at"])),
                "content_length": len(row["markdown_content"] or ""),
            }
            for row in rows
        ]

    # -- Incremental Sync Metadata --------------------------------------------

    async def get_sync_metadata(self, doc_id: str) -> dict | None:
        """Get sync metadata for incremental sync."""
        row = await self._pool.fetchrow(
            "SELECT etag, last_modified, last_sync_at, sync_count FROM sync_metadata WHERE document_id = $1",
            doc_id,
        )
        if not row:
            return None
        return {
            "etag": row["etag"],
            "last_modified": row["last_modified"],
            "last_sync_at": row["last_sync_at"],
            "sync_count": row["sync_count"],
        }

    async def update_sync_metadata(self, doc_id: str, etag: str = "", last_modified: str = "") -> None:
        """Update sync metadata after a successful sync."""
        now = time.time()
        await self._pool.execute(
            """
            INSERT INTO sync_metadata (document_id, etag, last_modified, last_sync_at, sync_count)
            VALUES ($1, $2, $3, $4, 1)
            ON CONFLICT(document_id) DO UPDATE SET
                etag=EXCLUDED.etag,
                last_modified=EXCLUDED.last_modified,
                last_sync_at=EXCLUDED.last_sync_at,
                sync_count=sync_metadata.sync_count + 1
            """,
            doc_id, etag, last_modified, now,
        )
