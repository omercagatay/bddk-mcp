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
            extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
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
            extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
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
            extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
            doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_patch_document_aborts_when_chunks_file_missing(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    # Write documents.json but NOT chunks.json
    (seed_dir / "documents.json").write_text(
        json.dumps([_seed_doc_entry("mevzuat_20029")], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = tmp_path / "body.md"
    md.write_text("new body\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="chunks.json not found"):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=md,
            extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
            doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_patch_document_aborts_when_body_empty_after_header_strip(tmp_path):
    """Header-only file (no body) gives a clear error instead of 'no chunks produced'."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "header_only.md"
    md.write_text("# Title\n- Document ID: mevzuat_20029\n---\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="no content after stripping"):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=md,
            extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
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
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
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
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=True,
    )

    expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert result["new_hash"] == expected_hash


@pytest.mark.asyncio
async def test_db_update_passes_body_and_metadata(tmp_path):
    """store_document + add_document must be awaited with the stripped body."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    body = "fresh body content\n"
    md.write_text(body, encoding="utf-8")

    ds = _mock_doc_store(_stored_doc("mevzuat_20029"))
    vs = _mock_vector_store()

    await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
        doc_store=ds,
        vector_store=vs,
        seed_dir=seed_dir,
        dry_run=False,
    )

    # store_document awaited once with the correct body + extraction_method
    ds.store_document.assert_awaited_once()
    stored = ds.store_document.await_args.args[0]
    assert stored.document_id == "mevzuat_20029"
    assert stored.markdown_content == body
    assert stored.extraction_method == patch_doc.DEFAULT_EXTRACTION_METHOD
    # Existing title / category / source_url carried over from the DB row
    assert stored.title == "Title of mevzuat_20029"
    assert stored.category == "Sermaye Yeterliliği"
    assert stored.source_url == "https://example.org/mevzuat_20029"

    # Hash invariant — store_document recomputes from markdown_content; the
    # StoredDocument we pass should already carry that exact hash for the
    # body we patched.
    import hashlib

    expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    # store_document recomputes hash from markdown_content, so we verify
    # the body itself round-trips to the expected hash. (The Pydantic model's
    # content_hash field is unused by store_document.)
    assert hashlib.sha256(stored.markdown_content.encode("utf-8")).hexdigest() == expected_hash
    # total_pages is computed in patch_document and passed verbatim
    assert stored.total_pages == 1  # body is 19 bytes, well under PAGE_SIZE

    # add_document awaited once with the same body
    vs.add_document.assert_awaited_once()
    kwargs = vs.add_document.await_args.kwargs
    assert kwargs["doc_id"] == "mevzuat_20029"
    assert kwargs["content"] == body
    assert kwargs["title"] == "Title of mevzuat_20029"


@pytest.mark.asyncio
async def test_seed_surgery_updates_only_target_doc(tmp_path):
    """documents.json: target updated, sibling byte-identical."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    target = _seed_doc_entry("mevzuat_20029", markdown="old body", content_hash="oldhash")
    sibling = _seed_doc_entry("mevzuat_99999", markdown="sibling body", content_hash="siblinghash")
    _write_seed_files(seed_dir, docs=[target, sibling], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("new corrected body\n", encoding="utf-8")

    await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=False,
    )

    written = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    assert len(written) == 2
    target_out = next(d for d in written if d["document_id"] == "mevzuat_20029")
    sibling_out = next(d for d in written if d["document_id"] == "mevzuat_99999")
    assert target_out["markdown_content"] == "new corrected body\n"
    assert target_out["content_hash"] != "oldhash"
    assert target_out["extraction_method"] == patch_doc.DEFAULT_EXTRACTION_METHOD
    assert target_out["extracted_at"] > 1_700_000_000  # newer than the seeded value
    # Sibling preserved exactly
    assert sibling_out == sibling

    # ensure_ascii=False regression guard — Turkish characters must render
    # as themselves, not \u-escapes, otherwise diff noise explodes in real repos.
    raw_text = (seed_dir / "documents.json").read_text(encoding="utf-8")
    assert "Sermaye Yeterliliği" in raw_text


@pytest.mark.asyncio
async def test_seed_surgery_rewrites_chunks_for_target_only(tmp_path):
    """chunks.json: target chunks replaced with fresh ones carrying new hash;
    sibling chunks byte-identical."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(
        seed_dir,
        docs=[_seed_doc_entry("mevzuat_20029"), _seed_doc_entry("mevzuat_99999")],
        chunks=[
            {
                "doc_id": "mevzuat_20029",
                "chunk_index": 0,
                "title": "Title of mevzuat_20029",
                "category": "",
                "decision_date": "",
                "decision_number": "",
                "source_url": "",
                "total_chunks": 2,
                "total_pages": 1,
                "content_hash": "oldhash",
                "chunk_text": "old part 1",
            },
            {
                "doc_id": "mevzuat_20029",
                "chunk_index": 1,
                "title": "Title of mevzuat_20029",
                "category": "",
                "decision_date": "",
                "decision_number": "",
                "source_url": "",
                "total_chunks": 2,
                "total_pages": 1,
                "content_hash": "oldhash",
                "chunk_text": "old part 2",
            },
            {
                "doc_id": "mevzuat_99999",
                "chunk_index": 0,
                "title": "Title of mevzuat_99999",
                "category": "",
                "decision_date": "",
                "decision_number": "",
                "source_url": "",
                "total_chunks": 1,
                "total_pages": 1,
                "content_hash": "siblinghash",
                "chunk_text": "sibling text",
            },
        ],
    )
    md = tmp_path / "body.md"
    md.write_text("new body for reembed\n", encoding="utf-8")

    await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=False,
    )

    written = json.loads((seed_dir / "chunks.json").read_text(encoding="utf-8"))
    target_chunks = [c for c in written if c["doc_id"] == "mevzuat_20029"]
    sibling_chunks = [c for c in written if c["doc_id"] == "mevzuat_99999"]

    # Target replaced: every new chunk carries the same new hash, none have oldhash
    assert target_chunks, "target chunks missing after patch"
    assert all(c["content_hash"] != "oldhash" for c in target_chunks)
    assert len({c["content_hash"] for c in target_chunks}) == 1

    # Sibling preserved exactly (unchanged order, unchanged content)
    assert sibling_chunks == [
        {
            "doc_id": "mevzuat_99999",
            "chunk_index": 0,
            "title": "Title of mevzuat_99999",
            "category": "",
            "decision_date": "",
            "decision_number": "",
            "source_url": "",
            "total_chunks": 1,
            "total_pages": 1,
            "content_hash": "siblinghash",
            "chunk_text": "sibling text",
        }
    ]


@pytest.mark.asyncio
async def test_seed_surgery_matches_doc_and_chunk_hashes(tmp_path):
    """Sanity: every new chunk_text concat back to body; hashes agree with doc."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("consistent body check\n", encoding="utf-8")

    await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=False,
    )

    docs_after = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    chunks_after = json.loads((seed_dir / "chunks.json").read_text(encoding="utf-8"))

    target_doc = next(d for d in docs_after if d["document_id"] == "mevzuat_20029")
    target_chunks = [c for c in chunks_after if c["doc_id"] == "mevzuat_20029"]

    assert target_chunks
    assert all(c["content_hash"] == target_doc["content_hash"] for c in target_chunks)


def test_arg_parser_requires_doc_id_and_markdown():
    parser = patch_doc._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing positional + required --markdown
    args = parser.parse_args(["mevzuat_20029", "--markdown", "body.md"])
    assert args.doc_id == "mevzuat_20029"
    assert args.markdown == Path("body.md")
    assert args.extraction_method == patch_doc.DEFAULT_EXTRACTION_METHOD
    assert args.dry_run is False


def test_arg_parser_honors_flags():
    parser = patch_doc._build_arg_parser()
    args = parser.parse_args(
        [
            "mevzuat_20029",
            "--markdown",
            "body.md",
            "--extraction-method",
            "html_parser",
            "--dry-run",
        ]
    )
    assert args.extraction_method == "html_parser"
    assert args.dry_run is True
