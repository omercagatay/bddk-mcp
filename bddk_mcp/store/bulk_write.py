"""Set-based PostgreSQL writers for corpus publication hot paths.

Every helper executes at most one data-changing statement and deliberately
does nothing for an empty batch.  Callers are responsible for opening a
transaction and acquiring :func:`acquire_corpus_mutation_lock` before calling
these helpers; keeping lock ownership at the workflow boundary makes it
possible to publish several related entity types atomically.

PostgreSQL's multi-argument ``unnest`` form is not used here.  ``ROWS FROM``
zips explicitly typed, one-dimensional arrays, while Python validates equal
row widths and unique logical keys before SQL is sent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _columns(
    rows: Sequence[Sequence[Any]],
    *,
    width: int,
    key_positions: tuple[int, ...],
    row_kind: str,
) -> tuple[list[Any], ...]:
    """Validate and transpose a batch without silently truncating ragged rows."""

    columns: list[list[Any]] = [[] for _ in range(width)]
    keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        if isinstance(row, str | bytes | bytearray) or not isinstance(row, Sequence) or len(row) != width:
            raise ValueError(f"{row_kind} row {index} must contain exactly {width} values")
        key = tuple(row[position] for position in key_positions)
        if any(value is None or (isinstance(value, str) and not value) for value in key):
            raise ValueError(f"{row_kind} row {index} has an invalid logical key")
        if key in keys:
            raise ValueError(f"{row_kind} batch contains a duplicate logical key")
        keys.add(key)
        for position, value in enumerate(row):
            columns[position].append(value)
    return tuple(columns)


def _count(value: Any, *, maximum: int, row_kind: str, exact: bool) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{row_kind} bulk write returned an invalid row count") from None
    if count < 0 or count > maximum or (exact and count != maximum):
        raise RuntimeError(f"{row_kind} bulk write returned an unexpected row count")
    return count


async def upsert_decision_cache_rows(connection: Any, rows: Sequence[Sequence[Any]]) -> int:
    """Upsert a complete decision-cache batch in one statement."""

    if not rows:
        return 0
    values = _columns(rows, width=8, key_positions=(0,), row_kind="decision cache")
    count = await connection.fetchval(
        """
        WITH persisted AS (
            INSERT INTO public.decision_cache AS existing (
                document_id, title, content, decision_date, decision_number,
                category, source_url, cached_at
            )
            SELECT incoming.document_id, incoming.title, incoming.content,
                   incoming.decision_date, incoming.decision_number,
                   incoming.category, incoming.source_url, incoming.cached_at
            FROM ROWS FROM (
                pg_catalog.unnest($1::pg_catalog.text[]),
                pg_catalog.unnest($2::pg_catalog.text[]),
                pg_catalog.unnest($3::pg_catalog.text[]),
                pg_catalog.unnest($4::pg_catalog.text[]),
                pg_catalog.unnest($5::pg_catalog.text[]),
                pg_catalog.unnest($6::pg_catalog.text[]),
                pg_catalog.unnest($7::pg_catalog.text[]),
                pg_catalog.unnest($8::pg_catalog.float8[])
            ) AS incoming (
                document_id, title, content, decision_date, decision_number,
                category, source_url, cached_at
            )
            ON CONFLICT (document_id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                decision_date = EXCLUDED.decision_date,
                decision_number = EXCLUDED.decision_number,
                category = EXCLUDED.category,
                source_url = EXCLUDED.source_url,
                cached_at = EXCLUDED.cached_at
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
        """,
        *values,
    )
    return _count(count, maximum=len(rows), row_kind="decision cache", exact=True)


async def insert_document_metadata_rows(connection: Any, rows: Sequence[Sequence[Any]]) -> int:
    """Insert absent document metadata rows and return the exact insert count."""

    if not rows:
        return 0
    values = _columns(rows, width=7, key_positions=(0,), row_kind="document metadata")
    count = await connection.fetchval(
        """
        WITH persisted AS (
            INSERT INTO public.documents (
                document_id, title, category, decision_date,
                decision_number, source_url, downloaded_at
            )
            SELECT incoming.document_id, incoming.title, incoming.category,
                   incoming.decision_date, incoming.decision_number,
                   incoming.source_url, incoming.downloaded_at
            FROM ROWS FROM (
                pg_catalog.unnest($1::pg_catalog.text[]),
                pg_catalog.unnest($2::pg_catalog.text[]),
                pg_catalog.unnest($3::pg_catalog.text[]),
                pg_catalog.unnest($4::pg_catalog.text[]),
                pg_catalog.unnest($5::pg_catalog.text[]),
                pg_catalog.unnest($6::pg_catalog.text[]),
                pg_catalog.unnest($7::pg_catalog.float8[])
            ) AS incoming (
                document_id, title, category, decision_date,
                decision_number, source_url, downloaded_at
            )
            ON CONFLICT (document_id) DO NOTHING
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
        """,
        *values,
    )
    return _count(count, maximum=len(rows), row_kind="document metadata", exact=False)


async def upsert_document_rows(connection: Any, rows: Sequence[Sequence[Any]]) -> int:
    """Upsert canonical seed documents in one statement."""

    if not rows:
        return 0
    values = _columns(rows, width=13, key_positions=(0,), row_kind="document")
    count = await connection.fetchval(
        """
        WITH persisted AS (
            INSERT INTO public.documents AS existing (
                document_id, title, category, decision_date, decision_number,
                source_url, markdown_content, content_hash, downloaded_at,
                extracted_at, extraction_method, total_pages, file_size
            )
            SELECT incoming.document_id, incoming.title, incoming.category,
                   incoming.decision_date, incoming.decision_number,
                   incoming.source_url, incoming.markdown_content,
                   incoming.content_hash, incoming.downloaded_at,
                   incoming.extracted_at, incoming.extraction_method,
                   incoming.total_pages, incoming.file_size
            FROM ROWS FROM (
                pg_catalog.unnest($1::pg_catalog.text[]),
                pg_catalog.unnest($2::pg_catalog.text[]),
                pg_catalog.unnest($3::pg_catalog.text[]),
                pg_catalog.unnest($4::pg_catalog.text[]),
                pg_catalog.unnest($5::pg_catalog.text[]),
                pg_catalog.unnest($6::pg_catalog.text[]),
                pg_catalog.unnest($7::pg_catalog.text[]),
                pg_catalog.unnest($8::pg_catalog.text[]),
                pg_catalog.unnest($9::pg_catalog.float8[]),
                pg_catalog.unnest($10::pg_catalog.float8[]),
                pg_catalog.unnest($11::pg_catalog.text[]),
                pg_catalog.unnest($12::pg_catalog.int4[]),
                pg_catalog.unnest($13::pg_catalog.int4[])
            ) AS incoming (
                document_id, title, category, decision_date, decision_number,
                source_url, markdown_content, content_hash, downloaded_at,
                extracted_at, extraction_method, total_pages, file_size
            )
            ON CONFLICT (document_id) DO UPDATE SET
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                decision_date = EXCLUDED.decision_date,
                decision_number = EXCLUDED.decision_number,
                source_url = EXCLUDED.source_url,
                markdown_content = EXCLUDED.markdown_content,
                content_hash = EXCLUDED.content_hash,
                downloaded_at = EXCLUDED.downloaded_at,
                extracted_at = EXCLUDED.extracted_at,
                extraction_method = EXCLUDED.extraction_method,
                total_pages = EXCLUDED.total_pages,
                file_size = EXCLUDED.file_size
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
        """,
        *values,
    )
    return _count(count, maximum=len(rows), row_kind="document", exact=True)


async def insert_document_version_rows(connection: Any, rows: Sequence[Sequence[Any]]) -> int:
    """Insert document-history snapshots in one statement."""

    if not rows:
        return 0
    values = _columns(rows, width=5, key_positions=(0,), row_kind="document version")
    # A document may legitimately contribute only one version per publication
    # transaction, so document_id is the logical batch key here.
    count = await connection.fetchval(
        """
        WITH persisted AS (
            INSERT INTO public.document_versions (
                document_id, version, content_hash, markdown_content, synced_at
            )
            SELECT incoming.document_id, incoming.version, incoming.content_hash,
                   incoming.markdown_content, incoming.synced_at
            FROM ROWS FROM (
                pg_catalog.unnest($1::pg_catalog.text[]),
                pg_catalog.unnest($2::pg_catalog.int4[]),
                pg_catalog.unnest($3::pg_catalog.text[]),
                pg_catalog.unnest($4::pg_catalog.text[]),
                pg_catalog.unnest($5::pg_catalog.float8[])
            ) AS incoming (document_id, version, content_hash, markdown_content, synced_at)
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
        """,
        *values,
    )
    return _count(count, maximum=len(rows), row_kind="document version", exact=True)


async def insert_document_section_rows(connection: Any, rows: Sequence[Sequence[Any]]) -> int:
    """Insert or reconcile derived document sections in one statement."""

    if not rows:
        return 0
    values = _columns(rows, width=11, key_positions=(0, 1, 2, 7), row_kind="document section")
    count = await connection.fetchval(
        """
        WITH persisted AS (
            INSERT INTO public.document_sections AS existing (
                doc_id, section_type, section_ref, heading, start_char, end_char,
                content, content_hash, page_start, page_end, source_content_hash
            )
            SELECT incoming.doc_id, incoming.section_type, incoming.section_ref,
                   incoming.heading, incoming.start_char, incoming.end_char,
                   incoming.content, incoming.content_hash, incoming.page_start,
                   incoming.page_end, incoming.source_content_hash
            FROM ROWS FROM (
                pg_catalog.unnest($1::pg_catalog.text[]),
                pg_catalog.unnest($2::pg_catalog.text[]),
                pg_catalog.unnest($3::pg_catalog.text[]),
                pg_catalog.unnest($4::pg_catalog.text[]),
                pg_catalog.unnest($5::pg_catalog.int4[]),
                pg_catalog.unnest($6::pg_catalog.int4[]),
                pg_catalog.unnest($7::pg_catalog.text[]),
                pg_catalog.unnest($8::pg_catalog.text[]),
                pg_catalog.unnest($9::pg_catalog.int4[]),
                pg_catalog.unnest($10::pg_catalog.int4[]),
                pg_catalog.unnest($11::pg_catalog.text[])
            ) AS incoming (
                doc_id, section_type, section_ref, heading, start_char, end_char,
                content, content_hash, page_start, page_end, source_content_hash
            )
            ON CONFLICT (doc_id, section_type, section_ref, content_hash) DO UPDATE SET
                heading = EXCLUDED.heading,
                start_char = EXCLUDED.start_char,
                end_char = EXCLUDED.end_char,
                content = EXCLUDED.content,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end,
                source_content_hash = EXCLUDED.source_content_hash
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
        """,
        *values,
    )
    return _count(count, maximum=len(rows), row_kind="document section", exact=True)


async def insert_document_chunk_rows(connection: Any, rows: Sequence[Sequence[Any]]) -> int:
    """Insert an exact chunk/embedding batch in one statement."""

    if not rows:
        return 0
    values = _columns(rows, width=19, key_positions=(0, 1), row_kind="document chunk")
    for index, row in enumerate(rows):
        chunk_index = row[1]
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError(f"document chunk row {index} has an invalid chunk index")
        if not isinstance(row[18], str) or not row[18].startswith("[") or not row[18].endswith("]"):
            raise ValueError(f"document chunk row {index} has an invalid vector representation")
    count = await connection.fetchval(
        """
        WITH persisted AS (
            INSERT INTO public.document_chunks (
                doc_id, chunk_index, title, category, decision_date,
                decision_number, source_url, total_chunks, total_pages,
                content_hash, chunk_start_char, chunk_end_char, section_type,
                section_ref, section_start_char, section_end_char,
                section_content_hash, chunk_text, embedding
            )
            SELECT incoming.doc_id, incoming.chunk_index, incoming.title,
                   incoming.category, incoming.decision_date,
                   incoming.decision_number, incoming.source_url,
                   incoming.total_chunks, incoming.total_pages,
                   incoming.content_hash, incoming.chunk_start_char,
                   incoming.chunk_end_char, incoming.section_type,
                   incoming.section_ref, incoming.section_start_char,
                   incoming.section_end_char, incoming.section_content_hash,
                   incoming.chunk_text, incoming.embedding::public.vector
            FROM ROWS FROM (
                pg_catalog.unnest($1::pg_catalog.text[]),
                pg_catalog.unnest($2::pg_catalog.int4[]),
                pg_catalog.unnest($3::pg_catalog.text[]),
                pg_catalog.unnest($4::pg_catalog.text[]),
                pg_catalog.unnest($5::pg_catalog.text[]),
                pg_catalog.unnest($6::pg_catalog.text[]),
                pg_catalog.unnest($7::pg_catalog.text[]),
                pg_catalog.unnest($8::pg_catalog.int4[]),
                pg_catalog.unnest($9::pg_catalog.int4[]),
                pg_catalog.unnest($10::pg_catalog.text[]),
                pg_catalog.unnest($11::pg_catalog.int4[]),
                pg_catalog.unnest($12::pg_catalog.int4[]),
                pg_catalog.unnest($13::pg_catalog.text[]),
                pg_catalog.unnest($14::pg_catalog.text[]),
                pg_catalog.unnest($15::pg_catalog.int4[]),
                pg_catalog.unnest($16::pg_catalog.int4[]),
                pg_catalog.unnest($17::pg_catalog.text[]),
                pg_catalog.unnest($18::pg_catalog.text[]),
                pg_catalog.unnest($19::pg_catalog.text[])
            ) AS incoming (
                doc_id, chunk_index, title, category, decision_date,
                decision_number, source_url, total_chunks, total_pages,
                content_hash, chunk_start_char, chunk_end_char, section_type,
                section_ref, section_start_char, section_end_char,
                section_content_hash, chunk_text, embedding
            )
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
        """,
        *values,
    )
    return _count(count, maximum=len(rows), row_kind="document chunk", exact=True)


async def upsert_document_retrieval_publication_rows(
    connection: Any,
    rows: Sequence[Sequence[Any]],
) -> int:
    """Publish validated per-document retrieval memberships in one statement."""

    if not rows:
        return 0
    values = _columns(rows, width=4, key_positions=(0,), row_kind="retrieval publication")
    count = await connection.fetchval(
        """
        WITH persisted AS (
            INSERT INTO public.document_retrieval_publications AS existing (
                doc_id, content_hash, retrieval_profile_hash, expected_chunks, published_at
            )
            SELECT incoming.doc_id, incoming.content_hash,
                   incoming.retrieval_profile_hash, incoming.expected_chunks,
                   CURRENT_TIMESTAMP
            FROM ROWS FROM (
                pg_catalog.unnest($1::pg_catalog.text[]),
                pg_catalog.unnest($2::pg_catalog.text[]),
                pg_catalog.unnest($3::pg_catalog.text[]),
                pg_catalog.unnest($4::pg_catalog.int4[])
            ) AS incoming (doc_id, content_hash, retrieval_profile_hash, expected_chunks)
            ON CONFLICT (doc_id) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                retrieval_profile_hash = EXCLUDED.retrieval_profile_hash,
                expected_chunks = EXCLUDED.expected_chunks,
                published_at = EXCLUDED.published_at
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
        """,
        *values,
    )
    return _count(count, maximum=len(rows), row_kind="retrieval publication", exact=True)


__all__ = (
    "insert_document_chunk_rows",
    "insert_document_metadata_rows",
    "insert_document_section_rows",
    "insert_document_version_rows",
    "upsert_decision_cache_rows",
    "upsert_document_retrieval_publication_rows",
    "upsert_document_rows",
)
