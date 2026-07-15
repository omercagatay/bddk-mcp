"""Tests for seed.py — focus on the import-skip logic."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bddk_mcp.db_lifecycle import DatabaseNotReadyError
from bddk_mcp.ingest import seed


@pytest.fixture
def temp_seed_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point seed.SEED_DIR at a tmp dir for the duration of one test."""
    monkeypatch.setattr(seed, "SEED_DIR", tmp_path)
    return tmp_path


@pytest.fixture
async def clean_pool(pg_pool):
    """Provide pg_pool with documents/chunks/decision_cache truncated before
    and after the test, so seed import tests don't pollute other tests."""

    async def _truncate(conn):
        await conn.execute(
            """
            TRUNCATE TABLE
                public.document_retrieval_publications,
                public.document_chunks,
                public.document_sections,
                public.document_versions,
                public.documents,
                public.decision_cache
            RESTART IDENTITY CASCADE
            """
        )

    async with pg_pool.acquire() as conn:
        await _truncate(conn)
    yield pg_pool
    async with pg_pool.acquire() as conn:
        await _truncate(conn)


def _write_seed_files(
    seed_dir: Path,
    *,
    docs: list[dict],
    chunks: list[dict] | None = None,
    cache: list[dict] | None = None,
) -> None:
    normalized_docs = json.loads(json.dumps(docs, ensure_ascii=False))
    hashes: dict[str, str] = {}
    for document in normalized_docs:
        content_hash = hashlib.sha256(document.get("markdown_content", "").encode()).hexdigest()
        document["content_hash"] = content_hash
        hashes[document["document_id"]] = content_hash
    normalized_chunks = json.loads(json.dumps(chunks or [], ensure_ascii=False))
    for chunk in normalized_chunks:
        if chunk.get("doc_id") in hashes:
            chunk["content_hash"] = hashes[chunk["doc_id"]]
    (seed_dir / "documents.json").write_text(
        json.dumps(normalized_docs, ensure_ascii=False),
        encoding="utf-8",
    )
    (seed_dir / "chunks.json").write_text(
        json.dumps(normalized_chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    (seed_dir / "decision_cache.json").write_text(json.dumps(cache or [], ensure_ascii=False), encoding="utf-8")


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def test_seed_document_validation_rejects_a_copied_or_tampered_hash() -> None:
    with pytest.raises(RuntimeError, match="content hash validation failed"):
        seed._validate_seed_documents(
            [
                {
                    "document_id": "tampered",
                    "markdown_content": "canonical text",
                    "content_hash": "0" * 64,
                }
            ]
        )


class _SelectOnlyBootstrapPool:
    """Minimal pool that makes any unexpected mutating SQL fail the test."""

    def __init__(self) -> None:
        self.selects: list[str] = []

    @asynccontextmanager
    async def acquire(self):
        yield self

    async def fetchval(self, query: str, *_args):
        normalized = query.strip()
        assert normalized.upper().startswith("SELECT"), normalized
        self.selects.append(normalized)
        return 0

    async def execute(self, *_args, **_kwargs):
        raise AssertionError("empty bootstrap must not issue DML or DDL")


@pytest.mark.asyncio
async def test_import_requires_prepared_schema_and_never_calls_initializers(temp_seed_dir):
    """Bootstrap validates first and cannot silently upgrade the database."""
    pool = _SelectOnlyBootstrapPool()
    readiness = AsyncMock()
    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    vector_store = MagicMock()
    vector_store.initialize = AsyncMock()

    with (
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=readiness),
        patch("bddk_mcp.store.doc_store.DocumentStore", return_value=document_store),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=vector_store),
    ):
        result = await seed.import_seed(pool=pool, force=True)

    assert result["embedded"] == 0
    readiness.assert_awaited_once_with(pool=pool, require_corpus=False)
    document_store.initialize.assert_not_awaited()
    vector_store.initialize.assert_not_awaited()
    assert all(statement.upper().startswith("SELECT") for statement in pool.selects)


@pytest.mark.asyncio
async def test_import_fails_readiness_before_opening_a_dml_connection(temp_seed_dir):
    pool = MagicMock()
    readiness = AsyncMock(side_effect=DatabaseNotReadyError("migration required"))
    document_store_class = MagicMock()
    vector_store_class = MagicMock()

    with (
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=readiness),
        patch("bddk_mcp.store.doc_store.DocumentStore", new=document_store_class),
        patch("bddk_mcp.store.vector_store.VectorStore", new=vector_store_class),
        pytest.raises(DatabaseNotReadyError, match="migration required"),
    ):
        await seed.import_seed(pool=pool, force=True)

    readiness.assert_awaited_once_with(pool=pool, require_corpus=False)
    pool.acquire.assert_not_called()
    document_store_class.assert_not_called()
    vector_store_class.assert_not_called()


@pytest.mark.asyncio
async def test_owned_bootstrap_pool_requires_exact_ingestion_identity_before_dml(temp_seed_dir):
    pool = MagicMock()
    pool.close = AsyncMock()
    identity = AsyncMock(side_effect=RuntimeError("identity rejected"))

    with (
        patch("bddk_mcp.ingest.seed.assert_database_transport", side_effect=lambda value: value),
        patch("bddk_mcp.ingest.seed.asyncpg.create_pool", new=AsyncMock(return_value=pool)),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch("bddk_mcp.ingest.seed.assert_database_identity", new=identity),
        pytest.raises(RuntimeError, match="identity rejected"),
    ):
        await seed.import_seed(dsn="postgresql://different-text-same-login", force=True)

    identity.assert_awaited_once_with(pool, "ingestion")
    pool.acquire.assert_not_called()
    pool.execute.assert_not_called()
    pool.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_import_backfills_embeddings_for_new_chunks(clean_pool, temp_seed_dir):
    """Seed import must not leave chunks with NULL embeddings — otherwise
    semantic search is silently broken for most queries (see #62: the main
    production DB had 96% NULL embeddings after a seed import)."""
    docs = [
        {
            "document_id": "test_embed_1",
            "title": "Embed Test",
            "markdown_content": "test body for embedding",
            "content_hash": "h1",
        }
    ]
    chunks = [
        {
            "doc_id": "test_embed_1",
            "chunk_index": 0,
            "chunk_text": "kredi riski ve sermaye yeterliliği",
            "content_hash": "h1",
        },
        {
            "doc_id": "test_embed_1",
            "chunk_index": 1,
            "chunk_text": "temerrüt halinde kayıp tahmini",
            "content_hash": "h1",
        },
    ]
    _write_seed_files(temp_seed_dir, docs=docs, chunks=chunks)

    result = await seed.import_seed(pool=clean_pool, force=True)
    assert result["chunks"] >= 1
    assert result["embedded"] == result["chunks"]
    stored_text = await clean_pool.fetchval(
        "SELECT string_agg(chunk_text, '') FROM document_chunks WHERE doc_id = $1",
        "test_embed_1",
    )
    assert "test body for embedding" in stored_text
    assert "temerrüt halinde" not in stored_text

    # Verify no NULL embeddings remain for this doc.
    null_count = await clean_pool.fetchval(
        "SELECT COUNT(*) FROM document_chunks WHERE doc_id = $1 AND embedding IS NULL",
        "test_embed_1",
    )
    assert null_count == 0

    # Idempotency: a second import without --force should skip (content
    # already matches), leaving the embeddings from the first run intact.
    result2 = await seed.import_seed(pool=clean_pool, force=False)
    assert result2["skipped"] is True
    null_count_after = await clean_pool.fetchval(
        "SELECT COUNT(*) FROM document_chunks WHERE doc_id = $1 AND embedding IS NULL",
        "test_embed_1",
    )
    assert null_count_after == 0


@pytest.mark.asyncio
async def test_import_skips_when_db_matches_seed(clean_pool, temp_seed_dir):
    """Baseline: matching counts AND matching content hashes → skip."""
    docs = [
        {
            "document_id": "test_match_1",
            "title": "Test",
            "markdown_content": "clean content",
            "content_hash": "hash_clean",
        }
    ]
    chunks = [
        {
            "doc_id": "test_match_1",
            "chunk_index": 0,
            "chunk_text": "clean content",
            "content_hash": "hash_clean",
        }
    ]
    _write_seed_files(temp_seed_dir, docs=docs, chunks=chunks)

    # Pre-populate DB with same content
    await seed.import_seed(pool=clean_pool, force=True)

    # Second call with same seed → should skip
    result = await seed.import_seed(pool=clean_pool, force=False)
    assert result["skipped"] is True


@pytest.mark.asyncio
async def test_import_does_not_skip_when_seed_content_differs(clean_pool, temp_seed_dir):
    """REGRESSION: counts match but content_hash differs — must re-import.

    This was the prod deploy bug (2026-04-17): clean docs in seed_data/ never
    reached the DB because count-only check declared 'DB up-to-date'.
    """
    # First import: seed says doc has corrupted content
    _write_seed_files(
        temp_seed_dir,
        docs=[
            {
                "document_id": "test_drift_1",
                "title": "Test",
                "markdown_content": "corrupted \ufffd text",
                "content_hash": "hash_old_corrupted",
            }
        ],
        chunks=[
            {
                "doc_id": "test_drift_1",
                "chunk_index": 0,
                "chunk_text": "corrupted \ufffd text",
                "content_hash": "hash_old_corrupted",
            }
        ],
    )
    await seed.import_seed(pool=clean_pool, force=True)

    # Now seed file is updated with clean content (same doc_id, new hash)
    _write_seed_files(
        temp_seed_dir,
        docs=[
            {
                "document_id": "test_drift_1",
                "title": "Test",
                "markdown_content": "clean text",
                "content_hash": "hash_new_clean",
            }
        ],
        chunks=[
            {
                "doc_id": "test_drift_1",
                "chunk_index": 0,
                "chunk_text": "clean text",
                "content_hash": "hash_new_clean",
            }
        ],
    )
    with pytest.raises(RuntimeError, match="rerun with --force"):
        await seed.import_seed(pool=clean_pool, force=False)

    result = await seed.import_seed(pool=clean_pool, force=True)
    assert result["skipped"] is False
    assert result["documents"] >= 1

    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT markdown_content, content_hash FROM documents WHERE document_id = $1",
            "test_drift_1",
        )
    assert row is not None
    assert row["markdown_content"] == "clean text"
    assert row["content_hash"] == _content_hash("clean text")


@pytest.mark.asyncio
async def test_import_ignores_untrusted_committed_chunks_and_regenerates_from_documents(
    clean_pool,
    temp_seed_dir,
):
    """A copied document hash cannot publish arbitrary committed chunk text."""
    docs = [
        {
            "document_id": "test_chunk_drift_1",
            "title": "T",
            "markdown_content": "new clean content",
            "content_hash": "doc_hash_new",
        }
    ]
    chunks_old = [
        {
            "doc_id": "test_chunk_drift_1",
            "chunk_index": 0,
            "chunk_text": "old corrupted \ufffd content",
            "content_hash": "chunk_hash_old",
        }
    ]
    _write_seed_files(temp_seed_dir, docs=docs, chunks=chunks_old)
    await seed.import_seed(pool=clean_pool, force=True)

    result = await seed.import_seed(pool=clean_pool, force=False)

    assert result["skipped"] is True
    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT chunk_text, content_hash FROM document_chunks WHERE doc_id = $1",
            "test_chunk_drift_1",
        )
    assert row is not None
    assert row["chunk_text"] == "new clean content"
    assert row["content_hash"] == _content_hash("new clean content")


@pytest.mark.asyncio
async def test_import_removes_stale_chunks_when_doc_reextracted_to_fewer_chunks(clean_pool, temp_seed_dir):
    """A republish replaces every stale tail row before publishing again."""
    docs = [
        {
            "document_id": "test_shrinking_doc",
            "title": "Shrinking",
            "markdown_content": "new shorter content",
            "content_hash": "doc_hash_new",
        }
    ]
    _write_seed_files(temp_seed_dir, docs=docs)
    await seed.import_seed(pool=clean_pool, force=True)

    async with clean_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO public.document_chunks (
                doc_id, chunk_index, title, total_chunks, total_pages,
                content_hash, chunk_text, embedding
            )
            SELECT original.doc_id,
                   generated.chunk_index,
                   original.title,
                   5,
                   original.total_pages,
                   original.content_hash,
                   'stale tail ' || generated.chunk_index::pg_catalog.text,
                   original.embedding
            FROM public.document_chunks AS original
            CROSS JOIN pg_catalog.generate_series(1, 4) AS generated(chunk_index)
            WHERE original.doc_id = $1 AND original.chunk_index = 0
            """,
            "test_shrinking_doc",
        )
        count_before = await conn.fetchval(
            "SELECT COUNT(*) FROM document_chunks WHERE doc_id = $1",
            "test_shrinking_doc",
        )
    assert count_before == 5

    result = await seed.import_seed(pool=clean_pool, force=False)

    assert result["skipped"] is False
    async with clean_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chunk_index, chunk_text, content_hash FROM document_chunks WHERE doc_id = $1 ORDER BY chunk_index",
            "test_shrinking_doc",
        )
    assert [r["chunk_index"] for r in rows] == [0]
    assert rows[0]["chunk_text"] == "new shorter content"
    assert rows[0]["content_hash"] == _content_hash("new shorter content")


class TestStripDocsDumpHeader:
    """Cover both header-present and header-absent inputs."""

    def test_strips_header_when_separator_present(self):
        text = "# Kredi Riski\n- Document ID: mevzuat_20029\n- Decision Date: N/A\n---\nbody line one\nbody line two\n"
        assert seed._strip_docs_dump_header(text) == "body line one\nbody line two\n"

    def test_passes_through_when_no_separator(self):
        text = "body line one\nbody line two\n"
        assert seed._strip_docs_dump_header(text) == text

    def test_splits_only_on_first_separator(self):
        text = "header\n---\nbody with\n---\nembedded separator\n"
        assert seed._strip_docs_dump_header(text) == "body with\n---\nembedded separator\n"


def test_expected_seed_sections_parses_article_references():
    sections = seed._expected_seed_sections(
        [
            {
                "document_id": "mevzuat_22599",
                "markdown_content": "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n",
            }
        ]
    )

    assert [(item.section_type, item.section_ref) for item in sections["mevzuat_22599"]] == [("madde", "9")]


def test_reviewed_seed_contains_required_exact_section_fixtures():
    documents = json.loads((seed.SEED_DIR / "documents.json").read_text(encoding="utf-8"))
    targets = {"943", "mevzuat_22599"}
    sections = seed._expected_seed_sections(
        [document for document in documents if document.get("document_id") in targets]
    )

    assert any(item.section_type == "ilke" and item.section_ref == "5" for item in sections["943"])
    assert any(item.section_type == "madde" and item.section_ref == "9" for item in sections["mevzuat_22599"])


@pytest.mark.asyncio
async def test_import_restores_missing_section_index(clean_pool, temp_seed_dir):
    """Matching documents and chunks must not hide a missing section index."""
    docs = [
        {
            "document_id": "mevzuat_22599",
            "title": "TFRS 9 Test",
            "markdown_content": "MADDE 9 - Karşılıklar\nBankalar beklenen kredi zararı hesaplar.\n",
            "content_hash": "section_hash",
        }
    ]
    chunks = [
        {
            "doc_id": "mevzuat_22599",
            "chunk_index": 0,
            "chunk_text": docs[0]["markdown_content"],
            "content_hash": "section_hash",
        }
    ]
    _write_seed_files(temp_seed_dir, docs=docs, chunks=chunks)

    first = await seed.import_seed(pool=clean_pool, force=True)
    assert first["sections"] == 1
    await clean_pool.execute(
        "DELETE FROM document_sections WHERE doc_id = $1",
        "mevzuat_22599",
    )

    repaired = await seed.import_seed(pool=clean_pool, force=False)
    assert repaired["skipped"] is False
    assert repaired["sections"] == 1
    row = await clean_pool.fetchrow(
        """SELECT section_type, section_ref
           FROM document_sections
           WHERE doc_id = $1""",
        "mevzuat_22599",
    )
    assert (row["section_type"], row["section_ref"]) == ("madde", "9")

    current = await seed.import_seed(pool=clean_pool, force=False)
    assert current["skipped"] is True


@pytest.mark.asyncio
async def test_reindex_existing_is_complete_profile_aware_and_resumable(clean_pool):
    from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
    from bddk_mcp.store.vector_store import VectorStore

    document_store = DocumentStore(clean_pool)
    for doc_id, content in (("reindex-a", "Kredi riski"), ("reindex-b", "Likidite riski")):
        await document_store.store_document(
            StoredDocument(
                document_id=doc_id,
                title=doc_id,
                markdown_content=content,
            )
        )

    vector_store = VectorStore(clean_pool)
    first = await seed.reindex_existing_documents(clean_pool, vs=vector_store)
    second = await seed.reindex_existing_documents(clean_pool, vs=vector_store)

    assert first == {
        "reindex_scanned": 2,
        "reindex_published": 2,
        "reindex_current": 0,
    }
    assert second == {
        "reindex_scanned": 2,
        "reindex_published": 0,
        "reindex_current": 2,
    }

    await clean_pool.execute(
        "UPDATE public.document_retrieval_publications SET retrieval_profile_hash = $2 WHERE doc_id = $1",
        "reindex-a",
        "0" * 64,
    )
    reconciled = await seed.reindex_existing_documents(clean_pool, vs=vector_store)

    assert reconciled == {
        "reindex_scanned": 2,
        "reindex_published": 1,
        "reindex_current": 1,
    }
    assert await clean_pool.fetchval(
        "SELECT retrieval_profile_hash = $2 FROM public.document_retrieval_publications WHERE doc_id = $1",
        "reindex-a",
        vector_store.retrieval_profile_hash,
    )
