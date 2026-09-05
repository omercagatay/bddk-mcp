"""
PostgreSQL + tsvector document store for BDDK regulatory documents.

Provides storage, full-text search (Turkish tsvector), pagination, and
versioning for BDDK decisions, regulations, and mevzuat.gov.tr documents.

Requires: asyncpg, PostgreSQL 14+ with unaccent extension.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from datetime import datetime

import asyncpg
from pydantic import BaseModel, Field

from bddk_mcp.core.config import FTS_RANK_THRESHOLD, PAGE_SIZE
from bddk_mcp.corpus_coordination import acquire_corpus_mutation_lock
from bddk_mcp.quality.markdown_quality import prepare_markdown_for_storage
from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    artifact_id_for,
    blob_id_for,
    evidence_id_for,
    instrument_id_for,
    legal_version_id_for,
    provision_id_for,
)
from bddk_mcp.store.bulk_write import insert_document_metadata_rows, insert_document_section_rows
from bddk_mcp.store.section_index import extract_document_sections

logger = logging.getLogger(__name__)
DOCUMENT_STORE_SEARCH_PROFILE_VERSION = "document-store-simple-fts-v2"
_SAFE_SYNC_FAILURE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def _safe_sync_failure_fields(error: str, category: str) -> tuple[str, str]:
    """Return bounded codes suitable for durable operational metadata."""

    safe_category = category if _SAFE_SYNC_FAILURE_TOKEN_RE.fullmatch(category) else "unknown"
    if _SAFE_SYNC_FAILURE_TOKEN_RE.fullmatch(error):
        return error, safe_category
    return f"sync_{safe_category}_failed", safe_category


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


class StoredSectionCitationMapping(BaseModel):
    """Validated database mapping needed to construct Citation v1."""

    instrument_id: str
    instrument_jurisdiction: str
    instrument_authority_code: str
    instrument_identity_key: str
    legal_version_id: str
    legal_version_key: str
    legal_validation_record_sha256: str
    provision_validation_record_sha256: str
    artifact_id: str
    artifact_blob_id: str
    artifact_sha256: str
    source_url: str
    artifact_retrieved_at: datetime
    evidence_id: str
    evidence_locator: str
    evidence_statement_sha256: str
    provision_id: str
    provision_kind: str
    provision_path: str


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
    normalized_source_range: str = ""
    source_content_hash: str = ""
    citation_mapping: StoredSectionCitationMapping | None = None
    rank: float | None = None
    """FTS match rank (ts_rank_cd, length-normalized). Only set by search paths;
    comparable within one query's result set, not across queries."""


class StoreStats(BaseModel):
    """Statistics about the document store."""

    total_documents: int = 0
    total_size_mb: float = 0.0
    categories: dict[str, int] = Field(default_factory=dict)
    extraction_methods: dict[str, int] = Field(default_factory=dict)
    oldest_document: str | None = None
    newest_document: str | None = None
    documents_needing_refresh: int = 0


def _content_hash(content: str) -> str:
    """SHA-256 hash of document content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _citation_identities_match(row) -> bool:
    """Recheck content-derived identities before exposing a trusted mapping."""

    statement_sha256 = row["citation_evidence_statement_sha256"]
    return (
        row["citation_instrument_id"]
        == instrument_id_for(
            jurisdiction=row["citation_instrument_jurisdiction"],
            authority_code=row["citation_instrument_authority_code"],
            identity_key=row["citation_instrument_identity_key"],
        )
        and row["citation_legal_version_id"]
        == legal_version_id_for(
            instrument_id=row["citation_instrument_id"],
            version_key=row["citation_legal_version_key"],
            legal_text_sha256=row["source_content_hash"],
        )
        and row["citation_artifact_blob_id"] == blob_id_for(content_sha256=row["citation_artifact_sha256"])
        and row["citation_artifact_id"]
        == artifact_id_for(
            blob_id=row["citation_artifact_blob_id"],
            canonical_uri=row["citation_source_url"],
            retrieved_at=row["citation_artifact_retrieved_at"],
        )
        and statement_sha256 == row["content_hash"]
        and row["citation_evidence_id"]
        == evidence_id_for(
            artifact_id=row["citation_artifact_id"],
            locator=row["citation_evidence_locator"],
            statement_sha256=statement_sha256,
            authority_level=AuthorityLevel.AUTHORITATIVE,
        )
        and row["citation_provision_id"]
        == provision_id_for(
            instrument_id=row["citation_instrument_id"],
            kind=row["citation_provision_kind"],
            canonical_path=row["citation_provision_path"],
        )
    )


def _section_from_row(row) -> StoredDocumentSection:
    """Convert an asyncpg row into a StoredDocumentSection."""
    keys = set(row.keys())
    citation_mapping = None
    if "citation_instrument_id" in keys and row["citation_instrument_id"] is not None:
        try:
            if not _citation_identities_match(row):
                raise ValueError("citation identity mismatch")
            citation_mapping = StoredSectionCitationMapping(
                instrument_id=row["citation_instrument_id"],
                instrument_jurisdiction=row["citation_instrument_jurisdiction"],
                instrument_authority_code=row["citation_instrument_authority_code"],
                instrument_identity_key=row["citation_instrument_identity_key"],
                legal_version_id=row["citation_legal_version_id"],
                legal_version_key=row["citation_legal_version_key"],
                legal_validation_record_sha256=row["citation_legal_validation_record_sha256"],
                provision_validation_record_sha256=row["citation_provision_validation_record_sha256"],
                artifact_id=row["citation_artifact_id"],
                artifact_blob_id=row["citation_artifact_blob_id"],
                artifact_sha256=row["citation_artifact_sha256"],
                source_url=row["citation_source_url"],
                artifact_retrieved_at=row["citation_artifact_retrieved_at"],
                evidence_id=row["citation_evidence_id"],
                evidence_locator=row["citation_evidence_locator"],
                evidence_statement_sha256=row["citation_evidence_statement_sha256"],
                provision_id=row["citation_provision_id"],
                provision_kind=row["citation_provision_kind"],
                provision_path=row["citation_provision_path"],
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Validated citation view returned a noncanonical identity; mapping omitted")
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
        normalized_source_range=row["normalized_source_range"] if "normalized_source_range" in keys else "",
        source_content_hash=row["source_content_hash"] if "source_content_hash" in keys else "",
        citation_mapping=citation_mapping,
        rank=row["rank"] if "rank" in keys else None,
    )


# -- DocumentStore ------------------------------------------------------------


class DocumentStore:
    """
    Async PostgreSQL document store with tsvector full-text search.

    Usage::

        # First run ``bddk-mcp migrate`` with schema-owner credentials.
        store = DocumentStore(pool)
        await store.store_document(doc)
        page = await store.get_document_page("1291", page=1)
        hits = await store.search_content("sermaye yeterliliği")
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        """Deprecated SELECT-only compatibility readiness check.

        Schema creation is available only through ``bddk-mcp migrate``.
        """
        from bddk_mcp.db_lifecycle import assert_database_ready

        await assert_database_ready(pool=self._pool, require_corpus=False)

    async def close(self) -> None:
        """No-op — pool lifecycle is managed externally."""
        logger.info("DocumentStore closed")

    async def __aenter__(self) -> DocumentStore:
        # Context entry is a runtime path and must not acquire DDL privileges.
        # Use the shared SELECT-only catalog check so a missing migration fails
        # with an actionable error before callers attempt document DML.
        from bddk_mcp.db_lifecycle import assert_database_ready

        await assert_database_ready(pool=self._pool, require_corpus=False)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- CRUD -----------------------------------------------------------------

    async def store_document(self, doc: StoredDocument) -> None:
        """Atomically replace a document and its derived structural sections."""
        now = time.time()
        if doc.markdown_content:
            doc.markdown_content = prepare_markdown_for_storage(doc.markdown_content)
        content_hash = _content_hash(doc.markdown_content) if doc.markdown_content else ""
        total_pages = max(1, math.ceil(len(doc.markdown_content) / PAGE_SIZE)) if doc.markdown_content else 1
        sections = extract_document_sections(doc.document_id, doc.markdown_content) if doc.markdown_content else []

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await acquire_corpus_mutation_lock(conn)
                # Serialize every canonical/derived write for this document,
                # including concurrent first inserts where no row exists yet.
                await conn.fetchval(
                    "SELECT pg_catalog.pg_advisory_xact_lock("
                    "pg_catalog.hashtextextended($1::pg_catalog.text, 1095652431))",
                    doc.document_id,
                )
                # Archive previous version if content changed
                existing = await conn.fetchrow(
                    "SELECT content_hash, markdown_content FROM public.documents WHERE document_id = $1 FOR UPDATE",
                    doc.document_id,
                )
                if existing and existing["content_hash"] and existing["content_hash"] != content_hash:
                    max_ver = await conn.fetchval(
                        "SELECT COALESCE(MAX(version), 0) FROM public.document_versions WHERE document_id = $1",
                        doc.document_id,
                    )
                    await conn.execute(
                        "INSERT INTO public.document_versions "
                        "(document_id, version, content_hash, markdown_content, synced_at) "
                        "VALUES ($1, $2, $3, $4, $5)",
                        doc.document_id,
                        max_ver + 1,
                        existing["content_hash"],
                        existing["markdown_content"],
                        now,
                    )

                await conn.execute(
                    """
                    INSERT INTO public.documents (
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
                await self._replace_document_sections_on_connection(
                    conn,
                    doc.document_id,
                    sections,
                    source_content_hash=content_hash,
                )

        logger.debug(
            "Stored document (content_chars=%d, sections=%d)",
            len(doc.markdown_content),
            len(sections),
        )

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
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await acquire_corpus_mutation_lock(conn)
                await conn.fetchval(
                    "SELECT pg_catalog.pg_advisory_xact_lock("
                    "pg_catalog.hashtextextended($1::pg_catalog.text, 1095652431))",
                    doc_id,
                )
                # Current canonical foreign keys cascade sections, chunks, and
                # retrieval publication from this one parent mutation.  A
                # pre-v3 schema is not a supported writable state.
                result = await conn.execute("DELETE FROM public.documents WHERE document_id = $1", doc_id)
                return result == "DELETE 1"

    # -- Sections -------------------------------------------------------------

    async def replace_document_sections(
        self,
        doc_id: str,
        sections: list,
        *,
        source_content_hash: str,
    ) -> int:
        """Replace sections only when they were parsed from the current body."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await acquire_corpus_mutation_lock(conn)
                await conn.fetchval(
                    "SELECT pg_catalog.pg_advisory_xact_lock("
                    "pg_catalog.hashtextextended($1::pg_catalog.text, 1095652431))",
                    doc_id,
                )
                stored_hash = await conn.fetchval(
                    "SELECT content_hash FROM public.documents WHERE document_id = $1 FOR UPDATE",
                    doc_id,
                )
                if stored_hash is None or stored_hash != source_content_hash:
                    raise ValueError("sections were not published because the source document changed")
                return await self._replace_document_sections_on_connection(
                    conn,
                    doc_id,
                    sections,
                    source_content_hash=source_content_hash,
                )

    @staticmethod
    async def _replace_document_sections_on_connection(
        conn,
        doc_id: str,
        sections: list,
        *,
        source_content_hash: str,
    ) -> int:
        """Replace derived sections using the caller's publication transaction."""

        return await DocumentStore._replace_document_sections_many_on_connection(
            conn,
            {doc_id: sections},
            source_content_hashes={doc_id: source_content_hash},
        )

    @staticmethod
    async def _replace_document_sections_many_on_connection(
        conn,
        sections_by_document: dict[str, list],
        *,
        source_content_hashes: dict[str, str],
    ) -> int:
        """Replace sections for many documents with two bounded statements.

        The caller owns the encompassing transaction and corpus mutation lock.
        Documents with no parsed sections are included in the delete membership
        so their stale derived rows are still removed.
        """

        document_ids = sorted(sections_by_document)
        if set(document_ids) != set(source_content_hashes):
            raise ValueError("section sources must exactly match replacement document membership")
        if any(not doc_id or not isinstance(source_content_hashes[doc_id], str) for doc_id in document_ids):
            raise ValueError("section replacement document identities and hashes must be valid strings")
        rows: list[tuple] = []
        for doc_id in document_ids:
            sections = sections_by_document[doc_id]
            if any(getattr(section, "doc_id", doc_id) != doc_id for section in sections):
                raise ValueError("section document identity does not match the publication target")
            rows.extend(
                (
                    doc_id,
                    section.section_type,
                    section.section_ref,
                    section.heading,
                    section.start_char,
                    section.end_char,
                    section.content,
                    section.content_hash,
                    section.page_start,
                    section.page_end,
                    source_content_hashes[doc_id],
                )
                for section in sections
            )
        if document_ids:
            await conn.execute(
                "DELETE FROM public.document_sections WHERE doc_id = ANY($1::pg_catalog.text[])",
                document_ids,
            )
        return await insert_document_section_rows(conn, rows)

    async def delete_document_sections(self, doc_id: str) -> bool:
        """Delete all structural sections for a document."""
        async with self._pool.acquire() as conn, conn.transaction():
            await acquire_corpus_mutation_lock(conn)
            await conn.fetchval(
                "SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended($1::pg_catalog.text, 1095652431))",
                doc_id,
            )
            result = await conn.execute("DELETE FROM public.document_sections WHERE doc_id = $1", doc_id)
            return result != "DELETE 0"

    async def get_document_section(
        self,
        doc_id: str,
        *,
        section_type: str | None = None,
        section_ref: str | None = None,
        heading: str | None = None,
        limit: int | None = None,
    ) -> list[StoredDocumentSection]:
        """Fetch structural sections by document ID and optional exact refs."""
        if limit is not None and (isinstance(limit, bool) or not 1 <= limit <= 1_000):
            raise ValueError("section limit must be between 1 and 1000")
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

        limit_clause = ""
        if limit is not None:
            params.append(limit)
            limit_clause = f"LIMIT ${len(params)}"

        where = [f"section.{condition}" for condition in where]
        rows = await self._pool.fetch(
            f"""
            SELECT section.doc_id, section.section_type, section.section_ref, section.heading,
                   section.start_char, section.end_char, section.content, section.content_hash,
                   section.page_start, section.page_end,
                   pg_catalog.substr(
                       document.markdown_content,
                       section.start_char + 1,
                       section.end_char - section.start_char
                   ) AS normalized_source_range,
                   document.content_hash AS source_content_hash,
                   citation.instrument_id AS citation_instrument_id,
                   citation.instrument_jurisdiction AS citation_instrument_jurisdiction,
                   citation.instrument_authority_code AS citation_instrument_authority_code,
                   citation.instrument_identity_key AS citation_instrument_identity_key,
                   citation.legal_version_id AS citation_legal_version_id,
                   citation.legal_version_key AS citation_legal_version_key,
                   citation.review_record_sha256 AS citation_legal_validation_record_sha256,
                   citation.provision_review_record_sha256 AS citation_provision_validation_record_sha256,
                   citation.artifact_id AS citation_artifact_id,
                   citation.artifact_blob_id AS citation_artifact_blob_id,
                   citation.artifact_sha256 AS citation_artifact_sha256,
                   citation.source_url AS citation_source_url,
                   citation.artifact_retrieved_at AS citation_artifact_retrieved_at,
                   citation.evidence_id AS citation_evidence_id,
                   citation.evidence_locator AS citation_evidence_locator,
                   citation.evidence_statement_sha256 AS citation_evidence_statement_sha256,
                   citation.provision_id AS citation_provision_id,
                   citation.provision_kind AS citation_provision_kind,
                   citation.provision_path AS citation_provision_path
            FROM public.document_sections AS section
            JOIN public.documents AS document
              ON document.document_id = section.doc_id
             AND document.content_hash = section.source_content_hash
            LEFT JOIN LATERAL (
                SELECT mapping.instrument_id,
                       mapping.instrument_jurisdiction,
                       mapping.instrument_authority_code,
                       mapping.instrument_identity_key,
                       mapping.legal_version_id,
                       mapping.legal_version_key,
                       mapping.review_record_sha256,
                       mapping.provision_review_record_sha256,
                       mapping.artifact_id,
                       mapping.artifact_blob_id,
                       mapping.artifact_sha256,
                       mapping.source_url,
                       mapping.artifact_retrieved_at,
                       mapping.evidence_id,
                       mapping.evidence_locator,
                       mapping.evidence_statement_sha256,
                       mapping.provision_id,
                       mapping.provision_kind,
                       mapping.provision_path
                FROM public.regulatory_validated_section_citations AS mapping
                WHERE mapping.document_section_id = section.id
                  AND mapping.source_document_id = section.doc_id
                  AND mapping.normalized_document_sha256 = document.content_hash
                  AND mapping.normalized_section_sha256 = section.content_hash
                  AND mapping.legal_text_sha256 = document.content_hash
                  AND mapping.provision_text_sha256 = section.content_hash
            ) AS citation ON true
            WHERE {" AND ".join(where)}
            ORDER BY section.start_char
            {limit_clause}
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
        where = ["section.tsv @@ pg_catalog.plainto_tsquery('simple', public.immutable_unaccent($1))"]
        params: list = [query]
        if document_id:
            params.append(document_id)
            where.append(f"section.doc_id = ${len(params)}")
        if section_type:
            params.append(section_type)
            where.append(f"section.section_type = ${len(params)}")
        else:
            # Nested fıkra/bent rows duplicate a parent madde; PDF wraps turn
            # them into fragments that outrank the article that holds the limit.
            # govde remainder is unparsed leftover (headings, footnotes), not a
            # provision identity — same opt-in as fıkra/bent via section_type.
            where.append("section.section_type NOT IN ('fikra', 'bent', 'govde')")
        params.append(limit)

        rows = await self._pool.fetch(
            f"""
            SELECT section.doc_id, section.section_type, section.section_ref, section.heading,
                   section.start_char, section.end_char, section.content, section.content_hash,
                   section.page_start, section.page_end,
                   pg_catalog.substr(
                       document.markdown_content,
                       section.start_char + 1,
                       section.end_char - section.start_char
                   ) AS normalized_source_range,
                   document.content_hash AS source_content_hash,
                   citation.instrument_id AS citation_instrument_id,
                   citation.instrument_jurisdiction AS citation_instrument_jurisdiction,
                   citation.instrument_authority_code AS citation_instrument_authority_code,
                   citation.instrument_identity_key AS citation_instrument_identity_key,
                   citation.legal_version_id AS citation_legal_version_id,
                   citation.legal_version_key AS citation_legal_version_key,
                   citation.review_record_sha256 AS citation_legal_validation_record_sha256,
                   citation.provision_review_record_sha256 AS citation_provision_validation_record_sha256,
                   citation.artifact_id AS citation_artifact_id,
                   citation.artifact_blob_id AS citation_artifact_blob_id,
                   citation.artifact_sha256 AS citation_artifact_sha256,
                   citation.source_url AS citation_source_url,
                   citation.artifact_retrieved_at AS citation_artifact_retrieved_at,
                   citation.evidence_id AS citation_evidence_id,
                   citation.evidence_locator AS citation_evidence_locator,
                   citation.evidence_statement_sha256 AS citation_evidence_statement_sha256,
                   citation.provision_id AS citation_provision_id,
                   citation.provision_kind AS citation_provision_kind,
                   citation.provision_path AS citation_provision_path,
                   -- normalization flag 1 divides by 1+log(length): without it,
                   -- jumbo boilerplate sections outrank on-point short maddeler
                   pg_catalog.ts_rank_cd(
                       section.tsv,
                       pg_catalog.plainto_tsquery('simple', public.immutable_unaccent($1)),
                       1
                   ) AS rank
            FROM public.document_sections AS section
            JOIN public.documents AS document
              ON document.document_id = section.doc_id
             AND document.content_hash = section.source_content_hash
            LEFT JOIN LATERAL (
                SELECT mapping.instrument_id,
                       mapping.instrument_jurisdiction,
                       mapping.instrument_authority_code,
                       mapping.instrument_identity_key,
                       mapping.legal_version_id,
                       mapping.legal_version_key,
                       mapping.review_record_sha256,
                       mapping.provision_review_record_sha256,
                       mapping.artifact_id,
                       mapping.artifact_blob_id,
                       mapping.artifact_sha256,
                       mapping.source_url,
                       mapping.artifact_retrieved_at,
                       mapping.evidence_id,
                       mapping.evidence_locator,
                       mapping.evidence_statement_sha256,
                       mapping.provision_id,
                       mapping.provision_kind,
                       mapping.provision_path
                FROM public.regulatory_validated_section_citations AS mapping
                WHERE mapping.document_section_id = section.id
                  AND mapping.source_document_id = section.doc_id
                  AND mapping.normalized_document_sha256 = document.content_hash
                  AND mapping.normalized_section_sha256 = section.content_hash
                  AND mapping.legal_text_sha256 = document.content_hash
                  AND mapping.provision_text_sha256 = section.content_hash
            ) AS citation ON true
            WHERE {" AND ".join(where)}
            ORDER BY rank DESC, section.start_char
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

        logger.info("FTS search completed: query_chars=%d hits=%d", len(query), len(hits))
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
        """Record a privacy-safe failure code without exception text or URLs."""
        now = time.time()
        safe_error, safe_category = _safe_sync_failure_fields(error, category)
        await self._pool.execute(
            """
            INSERT INTO sync_failures (document_id, error, error_category, source_url, retryable, attempts, first_failed_at, last_failed_at)
            VALUES ($1, $2, $3, $4, $5, 1, $6, $6)
            ON CONFLICT(document_id) DO UPDATE SET
                error = EXCLUDED.error,
                error_category = EXCLUDED.error_category,
                source_url = EXCLUDED.source_url,
                retryable = EXCLUDED.retryable,
                attempts = sync_failures.attempts + 1,
                last_failed_at = EXCLUDED.last_failed_at
            """,
            doc_id,
            safe_error,
            safe_category,
            "",
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
        # Collapse duplicate cache entries deterministically before acquiring
        # the global writer lock.  The last entry is the same behavior callers
        # would observe from a later catalog refresh.
        by_id: dict[str, dict] = {}
        for item in cache_items:
            doc_id = item.get("document_id", "")
            if isinstance(doc_id, str) and doc_id:
                by_id[doc_id] = item
        if not by_id:
            return 0

        async with self._pool.acquire() as conn, conn.transaction():
            await acquire_corpus_mutation_lock(conn)
            candidate_ids = sorted(by_id)
            existing_rows = await conn.fetch(
                "SELECT document_id FROM public.documents WHERE document_id = ANY($1::pg_catalog.text[])",
                candidate_ids,
            )
            existing_ids = {str(row["document_id"]) for row in existing_rows}
            now = time.time()
            rows = [
                (
                    doc_id,
                    by_id[doc_id].get("title", ""),
                    by_id[doc_id].get("category", ""),
                    by_id[doc_id].get("decision_date", ""),
                    by_id[doc_id].get("decision_number", ""),
                    by_id[doc_id].get("source_url", ""),
                    now,
                )
                for doc_id in candidate_ids
                if doc_id not in existing_ids
            ]
            # Empty input is a true no-op: no INSERT statement means no corpus
            # epoch trigger fires on repeated imports.
            imported = await insert_document_metadata_rows(conn, rows)

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
