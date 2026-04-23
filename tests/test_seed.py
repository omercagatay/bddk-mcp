"""Tests for seed.py — focus on the import-skip logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import seed


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
        for tbl in ("documents", "document_chunks", "decision_cache", "document_versions"):
            try:
                await conn.execute(f"TRUNCATE {tbl} RESTART IDENTITY")
            except Exception:
                pass

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
    (seed_dir / "documents.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    (seed_dir / "chunks.json").write_text(json.dumps(chunks or [], ensure_ascii=False), encoding="utf-8")
    (seed_dir / "decision_cache.json").write_text(json.dumps(cache or [], ensure_ascii=False), encoding="utf-8")


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
    result = await seed.import_seed(pool=clean_pool, force=False)

    assert result["skipped"] is False, "must re-import when content hash drifts"
    assert result["documents"] >= 1

    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT markdown_content, content_hash FROM documents WHERE document_id = $1",
            "test_drift_1",
        )
    assert row is not None
    assert row["markdown_content"] == "clean text"
    assert row["content_hash"] == "hash_new_clean"


@pytest.mark.asyncio
async def test_import_does_not_skip_when_chunks_out_of_sync_with_docs(clean_pool, temp_seed_dir):
    """REGRESSION: docs were re-imported but chunks have stale content_hash.

    This was the second prod symptom (2026-04-17): documents.markdown_content
    was clean but document_chunks.chunk_text still corrupted because chunks were
    regenerated separately and their content_hash never matched the new docs.
    """
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

    # Now seed has chunks regenerated to match docs
    chunks_new = [
        {
            "doc_id": "test_chunk_drift_1",
            "chunk_index": 0,
            "chunk_text": "new clean content",
            "content_hash": "doc_hash_new",
        }
    ]
    _write_seed_files(temp_seed_dir, docs=docs, chunks=chunks_new)
    result = await seed.import_seed(pool=clean_pool, force=False)

    assert result["skipped"] is False, "must re-import when chunk hash drifts from doc hash"
    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT chunk_text, content_hash FROM document_chunks WHERE doc_id = $1",
            "test_chunk_drift_1",
        )
    assert row is not None
    assert row["chunk_text"] == "new clean content"
    assert row["content_hash"] == "doc_hash_new"


@pytest.mark.asyncio
async def test_import_removes_stale_chunks_when_doc_reextracted_to_fewer_chunks(clean_pool, temp_seed_dir):
    """REGRESSION: re-extracted doc has fewer chunks than before — old tail must be deleted.

    Observed on 2026-04-21 when mevzuat_21193 was re-ingested from a manual PDF
    (fewer chunks than the html_parser version). Old chunks with index >=
    new_max remained in document_chunks under the same doc_id because the
    import used INSERT...ON CONFLICT UPDATE (upsert only). Those leftover
    rows still carried the old chunk_text and a stale pgvector embedding, so
    semantic search could surface pre-fix content.
    """
    docs = [
        {
            "document_id": "test_shrinking_doc",
            "title": "Shrinking",
            "markdown_content": "new shorter content",
            "content_hash": "doc_hash_new",
        }
    ]
    chunks_before = [
        {
            "doc_id": "test_shrinking_doc",
            "chunk_index": i,
            "chunk_text": f"old chunk {i}",
            "content_hash": "doc_hash_old",
        }
        for i in range(5)
    ]
    _write_seed_files(
        temp_seed_dir,
        docs=[{**docs[0], "content_hash": "doc_hash_old"}],
        chunks=chunks_before,
    )
    await seed.import_seed(pool=clean_pool, force=True)

    async with clean_pool.acquire() as conn:
        count_before = await conn.fetchval(
            "SELECT COUNT(*) FROM document_chunks WHERE doc_id = $1",
            "test_shrinking_doc",
        )
    assert count_before == 5

    chunks_after = [
        {
            "doc_id": "test_shrinking_doc",
            "chunk_index": 0,
            "chunk_text": "new chunk 0",
            "content_hash": "doc_hash_new",
        },
        {
            "doc_id": "test_shrinking_doc",
            "chunk_index": 1,
            "chunk_text": "new chunk 1",
            "content_hash": "doc_hash_new",
        },
    ]
    _write_seed_files(temp_seed_dir, docs=docs, chunks=chunks_after)
    result = await seed.import_seed(pool=clean_pool, force=False)

    assert result["skipped"] is False
    async with clean_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chunk_index, chunk_text, content_hash FROM document_chunks WHERE doc_id = $1 ORDER BY chunk_index",
            "test_shrinking_doc",
        )
    assert [r["chunk_index"] for r in rows] == [0, 1], "old chunks 2-4 must be deleted"
    assert all(r["chunk_text"].startswith("new chunk") for r in rows)
    assert all(r["content_hash"] == "doc_hash_new" for r in rows)


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
