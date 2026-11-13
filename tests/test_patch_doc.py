"""Tests for scripts/patch_doc.py — patch_document() against a temporary corpus file."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import patch_doc  # type: ignore  # noqa: E402

from bddk_mcp.core.config import CORPUS_PATH  # noqa: E402
from bddk_mcp.store.corpus_db import CorpusDB  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument  # noqa: E402
from bddk_mcp.store.vector_store import VectorStore  # noqa: E402

DOC_ID = "mevzuat_20029"


def _embed(texts: list[str]) -> np.ndarray:
    """One-hot unit vectors (dimension 768) stand in for the embedding model."""
    vectors = np.zeros((len(texts), 768), dtype=np.float32)
    for row, text in enumerate(texts):
        vectors[row, int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 768] = 1.0
    return vectors


class _Tokenizer:
    """Whitespace tokens stand in for the embedding model's tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


@pytest.fixture
def stores(tmp_path):
    """A real DocumentStore and VectorStore over a fresh corpus file."""
    db = CorpusDB.open(tmp_path / "c.sqlite")
    yield DocumentStore(db), VectorStore(db, embed_fn=_embed, tokenizer=_Tokenizer())
    db.close()


def _stored_doc(doc_id: str, *, markdown: str = "old body") -> StoredDocument:
    """The document as it sits in the corpus before the patch."""
    return StoredDocument(
        document_id=doc_id,
        title=f"Title of {doc_id}",
        category="Sermaye Yeterliliği",
        source_url=f"https://example.org/{doc_id}",
        markdown_content=markdown,
        extraction_method="markitdown_degraded",
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


def _write_seed_dir(tmp_path: Path, docs: list[dict]) -> Path:
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    (seed_dir / "documents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    return seed_dir


async def _patch(stores, seed_dir: Path, markdown_path: Path, **kwargs):
    doc_store, vector_store = stores
    return await patch_doc.patch_document(
        doc_id=DOC_ID,
        markdown_path=markdown_path,
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
        doc_store=doc_store,
        vector_store=vector_store,
        seed_dir=seed_dir,
        **kwargs,
    )


async def test_patch_document_aborts_when_markdown_file_missing(tmp_path, stores):
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry(DOC_ID)])
    await stores[0].store_document(_stored_doc(DOC_ID))

    with pytest.raises(FileNotFoundError):
        await _patch(stores, seed_dir, tmp_path / "does_not_exist.md", dry_run=True)


async def test_patch_document_aborts_when_doc_missing_from_db(tmp_path, stores):
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry(DOC_ID)])
    md = tmp_path / "body.md"
    md.write_text("new body\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="not found in DB"):
        await _patch(stores, seed_dir, md, dry_run=True)


async def test_patch_document_aborts_when_doc_missing_from_seed(tmp_path, stores):
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry("mevzuat_99999")])
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    md.write_text("new body\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="not found in .*documents.json"):
        await _patch(stores, seed_dir, md, dry_run=True)


async def test_patch_document_aborts_when_body_empty_after_header_strip(tmp_path, stores):
    """Header-only file (no body) gives a clear error instead of 'no chunks produced'."""
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry(DOC_ID)])
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "header_only.md"
    md.write_text("# Title\n- Document ID: mevzuat_20029\n---\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="no content after stripping"):
        await _patch(stores, seed_dir, md, dry_run=True)


async def test_patch_document_rejects_unbalanced_latex(tmp_path, stores):
    """An unclosed `$$` block in the corrected markdown should fail fast,
    before touching the corpus or seed_data."""
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry(DOC_ID)])
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "bad_latex.md"
    md.write_text(
        "opens but never closes: $$RA = \\max\\{0;\n\nmore text below.\n",
        encoding="utf-8",
    )

    with pytest.raises(patch_doc.PatchError, match="LaTeX validation failed"):
        await _patch(stores, seed_dir, md, dry_run=True)


async def test_patch_document_skip_latex_check_bypasses_validator(tmp_path, stores):
    """--skip-latex-check lets a body with unbalanced $ through — useful for
    documents that legitimately contain currency figures or other bare `$`."""
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry(DOC_ID)])
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "bare_dollar.md"
    md.write_text("price $100 and $200 — total $300", encoding="utf-8")

    # Doesn't raise — bypass path is honored.
    result = await _patch(stores, seed_dir, md, dry_run=True, skip_latex_check=True)
    assert result["dry_run"] is True


async def test_dry_run_performs_no_writes(tmp_path, stores):
    doc_store, vector_store = stores
    original_docs = [_seed_doc_entry(DOC_ID, markdown="old body", content_hash="oldhash")]
    seed_dir = _write_seed_dir(tmp_path, original_docs)
    await doc_store.store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    md.write_text("brand new body content\n", encoding="utf-8")

    result = await _patch(stores, seed_dir, md, dry_run=True)

    # No corpus writes: the stored row keeps its body and hash, nothing was embedded
    current = await doc_store.get_document(DOC_ID)
    assert current.markdown_content == "old body"
    assert current.content_hash == result["old_hash"]
    assert await vector_store.chunk_count() == 0
    # No FS writes — JSON files byte-identical to what we seeded
    assert json.loads((seed_dir / "documents.json").read_text(encoding="utf-8")) == original_docs
    # Result carries the planned numbers
    assert result["dry_run"] is True
    assert result["new_hash"] != "oldhash"
    assert result["chunk_count"] >= 1


async def test_strips_docs_dump_header_before_hashing(tmp_path, stores):
    """When markdown has a docs_dump header, the hash is of the body only."""
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry(DOC_ID)])
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    body = "the actual body\n"
    md.write_text(
        "# Kredi Riski\n- Document ID: mevzuat_20029\n- Decision Date: N/A\n---\n" + body,
        encoding="utf-8",
    )

    result = await _patch(stores, seed_dir, md, dry_run=True)

    expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert result["new_hash"] == expected_hash


async def test_db_update_passes_body_and_metadata(tmp_path, stores):
    """The corpus row carries the stripped body and its chunks are republished for it."""
    doc_store, vector_store = stores
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry(DOC_ID)])
    await doc_store.store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    body = "fresh body content\n"
    md.write_text(body, encoding="utf-8")

    result = await _patch(stores, seed_dir, md, dry_run=False)

    stored = await doc_store.get_document(DOC_ID)
    assert stored.markdown_content == body
    assert stored.extraction_method == patch_doc.DEFAULT_EXTRACTION_METHOD
    # Existing title / category / source_url carried over from the corpus row
    assert stored.title == f"Title of {DOC_ID}"
    assert stored.category == "Sermaye Yeterliliği"
    assert stored.source_url == f"https://example.org/{DOC_ID}"
    # Hash invariant — the stored row hashes the body we patched
    assert stored.content_hash == hashlib.sha256(body.encode("utf-8")).hexdigest() == result["new_hash"]
    assert stored.total_pages == 1  # body is 19 bytes, well under PAGE_SIZE
    # The previous body was archived as a version
    assert len(await doc_store.get_document_history(DOC_ID)) == 1

    # The vector index was republished against the new hash
    assert await vector_store.has_document(DOC_ID)
    assert await vector_store.chunk_count() == result["chunk_count"] >= 1


async def test_seed_surgery_updates_only_target_doc(tmp_path, stores):
    """documents.json: target updated, sibling byte-identical."""
    target = _seed_doc_entry(DOC_ID, markdown="old body", content_hash="oldhash")
    sibling = _seed_doc_entry("mevzuat_99999", markdown="sibling body", content_hash="siblinghash")
    seed_dir = _write_seed_dir(tmp_path, [target, sibling])
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    md.write_text("new corrected body\n", encoding="utf-8")

    await _patch(stores, seed_dir, md, dry_run=False)

    written = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    assert len(written) == 2
    target_out = next(d for d in written if d["document_id"] == DOC_ID)
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


def test_arg_parser_requires_doc_id_and_markdown():
    parser = patch_doc._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing positional + required --markdown
    args = parser.parse_args(["mevzuat_20029", "--markdown", "body.md"])
    assert args.doc_id == "mevzuat_20029"
    assert args.markdown == Path("body.md")
    assert args.corpus == CORPUS_PATH
    assert args.extraction_method == patch_doc.DEFAULT_EXTRACTION_METHOD
    assert args.dry_run is False


def test_arg_parser_honors_flags():
    parser = patch_doc._build_arg_parser()
    args = parser.parse_args(
        [
            "mevzuat_20029",
            "--markdown",
            "body.md",
            "--corpus",
            "other.sqlite",
            "--extraction-method",
            "html_parser",
            "--dry-run",
        ]
    )
    assert args.corpus == "other.sqlite"
    assert args.extraction_method == "html_parser"
    assert args.dry_run is True


async def test_storage_markers_raise_before_anything_is_written(tmp_path, stores):
    original_docs = [_seed_doc_entry(DOC_ID)]
    seed_dir = _write_seed_dir(tmp_path, original_docs)
    before = (seed_dir / "documents.json").read_bytes()
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    md.write_text("Madde 1 - metin ****** devam\n", encoding="utf-8")

    for dry_run in (True, False):
        with pytest.raises(patch_doc.PatchError, match="storage markers"):
            await _patch(stores, seed_dir, md, dry_run=dry_run)
    assert (seed_dir / "documents.json").read_bytes() == before


async def test_seed_is_written_in_canonical_format(tmp_path, stores):
    seed_dir = _write_seed_dir(tmp_path, [_seed_doc_entry("mevzuat_99999"), _seed_doc_entry(DOC_ID)])
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    md.write_text("new corrected body\n", encoding="utf-8")

    result = await _patch(stores, seed_dir, md, dry_run=False)

    assert result["seed_written"] is True
    raw = (seed_dir / "documents.json").read_text(encoding="utf-8")
    assert raw.endswith("]\n")
    assert [d["document_id"] for d in json.loads(raw)] == sorted([DOC_ID, "mevzuat_99999"])


async def test_no_seed_dir_leaves_seed_untouched(tmp_path, stores):
    await stores[0].store_document(_stored_doc(DOC_ID))
    md = tmp_path / "body.md"
    md.write_text("new corrected body\n", encoding="utf-8")
    doc_store, vector_store = stores

    result = await patch_doc.patch_document(
        doc_id=DOC_ID,
        markdown_path=md,
        extraction_method=patch_doc.DEFAULT_EXTRACTION_METHOD,
        doc_store=doc_store,
        vector_store=vector_store,
        seed_dir=None,
    )

    assert result["seed_written"] is False
    assert (await doc_store.get_document(DOC_ID)).markdown_content == "new corrected body\n"


def test_resolve_seed_dir(tmp_path):
    assert patch_doc._resolve_seed_dir(tmp_path, "other.sqlite") == tmp_path
    assert patch_doc._resolve_seed_dir(None, "other.sqlite") is None
    assert patch_doc._resolve_seed_dir(None, patch_doc.ROOT / "corpus" / "bddk.sqlite") == patch_doc.ROOT / "seed_data"


def test_resolve_seed_dir_ignores_a_scratch_corpus_set_through_the_environment(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch.sqlite"
    monkeypatch.setattr(patch_doc, "CORPUS_PATH", str(scratch))

    assert patch_doc._resolve_seed_dir(None, patch_doc.CORPUS_PATH) is None
    args = patch_doc._build_arg_parser().parse_args(["d", "--markdown", "b.md", "--seed-dir", str(tmp_path)])
    assert args.seed_dir == tmp_path
