"""Tests for scripts/patch_doc.py — patch_document() behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import patch_doc  # type: ignore  # noqa: E402

from doc_store import StoredDocument  # noqa: E402


def _stored_doc(doc_id: str, *, markdown: str = "old body", content_hash: str = "deadbeef") -> StoredDocument:
    """StoredDocument instance shaped like DocumentStore.get_document returns."""
    return StoredDocument(
        document_id=doc_id,
        title=f"Title of {doc_id}",
        category="Sermaye Yeterliliği",
        decision_date="",
        decision_number="",
        source_url=f"https://example.org/{doc_id}",
        markdown_content=markdown,
        content_hash=content_hash,
        extraction_method="markitdown_degraded",
        total_pages=1,
        file_size=len(markdown),
    )


def _seed_doc_entry(doc_id: str, *, markdown: str = "old body", content_hash: str = "deadbeef") -> dict:
    """Dict shaped like seed_data/documents.json entries (adds downloaded_at / extracted_at)."""
    return {
        "document_id": doc_id,
        "title": f"Title of {doc_id}",
        "category": "Sermaye Yeterliliği",
        "decision_date": "",
        "decision_number": "",
        "source_url": f"https://example.org/{doc_id}",
        "markdown_content": markdown,
        "content_hash": content_hash,
        "downloaded_at": 1_700_000_000,
        "extracted_at": 1_700_000_000,
        "extraction_method": "markitdown_degraded",
        "total_pages": 1,
        "file_size": len(markdown),
    }


def _write_seed_files(seed_dir: Path, *, docs: list[dict], chunks: list[dict]) -> None:
    (seed_dir / "documents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (seed_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_doc_store(current_doc: StoredDocument | None) -> MagicMock:
    """Build a DocumentStore mock that returns current_doc from get_document."""
    store = MagicMock()
    store.get_document = AsyncMock(return_value=current_doc)
    store.store_document = AsyncMock()
    return store


def _mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.add_document = AsyncMock(return_value=1)
    return vs


@pytest.mark.asyncio
async def test_patch_document_aborts_when_markdown_file_missing(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    missing_md = tmp_path / "does_not_exist.md"

    with pytest.raises(FileNotFoundError):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=missing_md,
            extraction_method="html_parser+manual_latex",
            doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_patch_document_aborts_when_doc_missing_from_db(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("new body\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="not found in DB"):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=md,
            extraction_method="html_parser+manual_latex",
            doc_store=_mock_doc_store(None),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_patch_document_aborts_when_doc_missing_from_seed(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_99999")], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("new body\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="not found in seed_data"):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=md,
            extraction_method="html_parser+manual_latex",
            doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_dry_run_performs_no_writes(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    original_docs = [_seed_doc_entry("mevzuat_20029", markdown="old body", content_hash="oldhash")]
    original_chunks = [
        {
            "doc_id": "mevzuat_20029",
            "chunk_index": 0,
            "title": "Title of mevzuat_20029",
            "category": "",
            "decision_date": "",
            "decision_number": "",
            "source_url": "",
            "total_chunks": 1,
            "total_pages": 1,
            "content_hash": "oldhash",
            "chunk_text": "old body",
        }
    ]
    _write_seed_files(seed_dir, docs=original_docs, chunks=original_chunks)
    md = tmp_path / "body.md"
    md.write_text("brand new body content\n", encoding="utf-8")

    ds = _mock_doc_store(_stored_doc("mevzuat_20029"))
    vs = _mock_vector_store()

    result = await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method="html_parser+manual_latex",
        doc_store=ds,
        vector_store=vs,
        seed_dir=seed_dir,
        dry_run=True,
    )

    # No DB writes
    ds.store_document.assert_not_awaited()
    vs.add_document.assert_not_awaited()
    # No FS writes — JSON files byte-identical to what we seeded
    assert json.loads((seed_dir / "documents.json").read_text(encoding="utf-8")) == original_docs
    assert json.loads((seed_dir / "chunks.json").read_text(encoding="utf-8")) == original_chunks
    # Result carries the planned numbers
    assert result["dry_run"] is True
    assert result["new_hash"] != "oldhash"
    assert result["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_strips_docs_dump_header_before_hashing(tmp_path):
    """When markdown has a docs_dump header, the hash is of the body only."""
    import hashlib

    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    body = "the actual body\n"
    md.write_text(
        "# Kredi Riski\n- Document ID: mevzuat_20029\n- Decision Date: N/A\n---\n" + body,
        encoding="utf-8",
    )

    result = await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method="html_parser+manual_latex",
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=True,
    )

    expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert result["new_hash"] == expected_hash
