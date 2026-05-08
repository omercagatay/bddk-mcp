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


class StoredDocumentSection(BaseModel):
    """A structural section persisted for a document."""

    doc_id: str
    section_type: str
    section_ref: str
    heading: str = ""
    start_char: int
    end_char: int
    content: str
    content_hash: str
    page_start: int | None = None
    page_end: int | None = None


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

CREATE TABLE IF NOT EXISTS document_sections (
    id            SERIAL PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    section_type  TEXT NOT NULL,
    section_ref   TEXT NOT NULL,
    heading       TEXT DEFAULT '',
    start_char    INTEGER NOT NULL,
    end_char      INTEGER NOT NULL,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    page_start    INTEGER,
    page_end      INTEGER,
    tsv           tsvector,
    UNIQUE(doc_id, section_type, section_ref, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_document_sections_doc_id ON document_sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_sections_ref ON document_sections(section_type, section_ref);
CREATE INDEX IF NOT EXISTS idx_document_sections_tsv ON document_sections USING GIN(tsv);

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

CREATE OR REPLACE FUNCTION document_sections_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv :=
        to_tsvector('simple', immutable_unaccent(coalesce(NEW.heading, '')))
        || to_tsvector('simple', immutable_unaccent(coalesce(NEW.content, '')));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_sections_tsv ON document_sections;
CREATE TRIGGER trg_document_sections_tsv
    BEFORE INSERT OR UPDATE ON document_sections
    FOR EACH ROW EXECUTE FUNCTION document_sections_tsv_trigger();

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

CREATE TABLE IF NOT EXISTS sync_failures (
    document_id       TEXT PRIMARY KEY,
    error             TEXT NOT NULL,
    error_category    TEXT NOT NULL DEFAULT 'unknown',
    source_url        TEXT DEFAULT '',
    retryable         BOOLEAN DEFAULT true,
    attempts          INTEGER DEFAULT 1,
    first_failed_at   DOUBLE PRECISION,
    last_failed_at    DOUBLE PRECISION
);
"""


def _content_hash(content: str) -> str:
    """SHA-256 hash of document content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _section_from_row(row) -> StoredDocumentSection:
    """Convert an asyncpg row into a StoredDocumentSection."""
    return StoredDocumentSection(
        doc_id=row["doc_id"],
        section_type=row["section_type"],
        section_ref=row["section_ref"],
        heading=row["heading"] or "",
        start_char=row["start_char"],
        end_char=row["end_char"],
        content=row["content"] or "",
        content_hash=row["content_hash"] or "",
        page_start=row["page_start"],
        page_end=row["page_end"],
    )


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

    async def get_pdf_bytes(self, document_id: str) -> bytes | None:
        """Return cached PDF bytes for a document, or None if absent."""
        row = await self._pool.fetchrow(
            "SELECT pdf_blob FROM documents WHERE document_id = $1",
            document_id,
        )
        if row is None or row["pdf_blob"] is None:
            return None
        return bytes(row["pdf_blob"])

    async def get_extraction_method(self, document_id: str) -> str | None:
        """Return the extraction_method recorded for a document, or None if absent."""
        row = await self._pool.fetchrow(
            "SELECT extraction_method FROM documents WHERE document_id = $1",
            document_id,
        )
        if row is None:
            return None
        return row["extraction_method"] or ""

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
        await self.delete_document_sections(doc_id)
        result = await self._pool.execute("DELETE FROM documents WHERE document_id = $1", doc_id)
        return result == "DELETE 1"

    # -- Sections -------------------------------------------------------------

    async def replace_document_sections(self, doc_id: str, sections: list) -> int:
        """Replace all indexed structural sections for a document."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM document_sections WHERE doc_id = $1", doc_id)
                if not sections:
                    return 0

                args = [
                    (
                        getattr(section, "doc_id", doc_id),
                        section.section_type,
                        section.section_ref,
                        section.heading,
                        section.start_char,
                        section.end_char,
                        section.content,
                        section.content_hash,
                        section.page_start,
                        section.page_end,
                    )
                    for section in sections
                ]
                await conn.executemany(
                    """
                    INSERT INTO document_sections (
                        doc_id, section_type, section_ref, heading, start_char, end_char,
                        content, content_hash, page_start, page_end
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    ON CONFLICT(doc_id, section_type, section_ref, content_hash) DO UPDATE SET
                        heading=EXCLUDED.heading,
                        start_char=EXCLUDED.start_char,
                        end_char=EXCLUDED.end_char,
                        content=EXCLUDED.content,
                        page_start=EXCLUDED.page_start,
                        page_end=EXCLUDED.page_end
                    """,
                    args,
                )
        return len(args)

    async def delete_document_sections(self, doc_id: str) -> bool:
        """Delete all structural sections for a document."""
        result = await self._pool.execute("DELETE FROM document_sections WHERE doc_id = $1", doc_id)
        return result != "DELETE 0"

    async def get_document_section(
        self,
        doc_id: str,
        *,
        section_type: str | None = None,
        section_ref: str | None = None,
        heading: str | None = None,
    ) -> list[StoredDocumentSection]:
        """Fetch structural sections by document ID and optional exact refs."""
        where = ["doc_id = $1"]
        params: list = [doc_id]
        if section_type:
            params.append(section_type)
            where.append(f"section_type = ${len(params)}")
        if section_ref:
            params.append(section_ref)
            where.append(f"section_ref = ${len(params)}")
        if heading:
            params.append(f"%{heading}%")
            where.append(f"heading ILIKE ${len(params)}")

        rows = await self._pool.fetch(
            f"""
            SELECT doc_id, section_type, section_ref, heading, start_char, end_char,
                   content, content_hash, page_start, page_end
            FROM document_sections
            WHERE {" AND ".join(where)}
            ORDER BY start_char
            """,
            *params,
        )
        return [_section_from_row(row) for row in rows]

    async def search_document_sections(
        self,
        query: str,
        *,
        document_id: str | None = None,
        section_type: str | None = None,
        limit: int = 10,
    ) -> list[StoredDocumentSection]:
        """Full-text search over structural sections."""
        where = ["tsv @@ plainto_tsquery('simple', immutable_unaccent($1))"]
        params: list = [query]
        if document_id:
            params.append(document_id)
            where.append(f"doc_id = ${len(params)}")
        if section_type:
            params.append(section_type)
            where.append(f"section_type = ${len(params)}")
        params.append(limit)

        rows = await self._pool.fetch(
            f"""
            SELECT doc_id, section_type, section_ref, heading, start_char, end_char,
                   content, content_hash, page_start, page_end,
                   ts_rank_cd(tsv, plainto_tsquery('simple', immutable_unaccent($1))) AS rank
            FROM document_sections
            WHERE {" AND ".join(where)}
            ORDER BY rank DESC, start_char
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_section_from_row(row) for row in rows]

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

    # -- Sync failure tracking ---------------------------------------------------

    async def record_sync_failure(
        self,
        doc_id: str,
        error: str,
        category: str,
        source_url: str = "",
        retryable: bool = True,
    ) -> None:
        """Record or update a sync failure for a document."""
        now = time.time()
        await self._pool.execute(
            """
            INSERT INTO sync_failures (document_id, error, error_category, source_url, retryable, attempts, first_failed_at, last_failed_at)
            VALUES ($1, $2, $3, $4, $5, 1, $6, $6)
            ON CONFLICT(document_id) DO UPDATE SET
                error = EXCLUDED.error,
                error_category = EXCLUDED.error_category,
                retryable = EXCLUDED.retryable,
                attempts = sync_failures.attempts + 1,
                last_failed_at = EXCLUDED.last_failed_at
            """,
            doc_id,
            error,
            category,
            source_url,
            retryable,
            now,
        )

    async def clear_sync_failure(self, doc_id: str) -> None:
        """Remove a sync failure record after successful sync."""
        await self._pool.execute("DELETE FROM sync_failures WHERE document_id = $1", doc_id)

    async def get_sync_failures(self, retryable_only: bool = False) -> list[dict]:
        """Get all current sync failures."""
        query = "SELECT * FROM sync_failures"
        if retryable_only:
            query += " WHERE retryable = true"
        query += " ORDER BY last_failed_at DESC"
        rows = await self._pool.fetch(query)
        return [dict(r) for r in rows]

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
                    existing = await conn.fetchval("SELECT 1 FROM documents WHERE document_id = $1", doc_id)
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
            "SELECT version, content_hash, LENGTH(COALESCE(markdown_content, '')) AS content_length, synced_at "
            "FROM document_versions WHERE document_id = $1 ORDER BY version DESC",
            doc_id,
        )
        return [
            {
                "version": row["version"],
                "content_hash": row["content_hash"],
                "synced_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(row["synced_at"])),
                "content_length": row["content_length"],
            }
            for row in rows
        ]

    async def get_version_count(self, doc_id: str) -> tuple[int, str | None]:
        """Get version count and latest sync time for a document (lightweight)."""
        row = await self._pool.fetchrow(
            "SELECT COUNT(*) AS cnt, MAX(synced_at) AS latest FROM document_versions WHERE document_id = $1",
            doc_id,
        )
        if not row or row["cnt"] == 0:
            return 0, None
        latest = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["latest"]))
        return row["cnt"], latest

    async def get_version_counts(self, doc_ids: list[str]) -> dict[str, tuple[int, str | None]]:
        """Batch version count for multiple documents. Returns {doc_id: (count, latest_date)}."""
        if not doc_ids:
            return {}
        rows = await self._pool.fetch(
            "SELECT document_id, COUNT(*) AS cnt, MAX(synced_at) AS latest "
            "FROM document_versions WHERE document_id = ANY($1) "
            "GROUP BY document_id",
            doc_ids,
        )
        result = {}
        for row in rows:
            latest = None
            if row["latest"]:
                latest = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["latest"]))
            result[row["document_id"]] = (row["cnt"], latest)
        return result

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
            doc_id,
            etag,
            last_modified,
            now,
        )
