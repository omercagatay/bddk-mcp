"""Tests for seed.py — building the corpus file from seed JSON and exporting it back."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bddk_mcp.ingest import seed
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore
from bddk_mcp.store.section_index import extract_document_sections
from bddk_mcp.store.vector_store import VectorStore
from tests.conftest import FakeTokenizer, fake_embed


def _fake_vector_store(db: CorpusDB) -> VectorStore:
    return VectorStore(db, embed_fn=fake_embed, tokenizer=FakeTokenizer())


def _write_seed_files(seed_dir: Path, *, docs: list[dict], cache: list[dict] | None = None) -> None:
    normalized_docs = json.loads(json.dumps(docs, ensure_ascii=False))
    for document in normalized_docs:
        document["content_hash"] = hashlib.sha256(document.get("markdown_content", "").encode()).hexdigest()
    (seed_dir / "documents.json").write_text(json.dumps(normalized_docs, ensure_ascii=False), encoding="utf-8")
    (seed_dir / "decision_cache.json").write_text(json.dumps(cache or [], ensure_ascii=False), encoding="utf-8")


DOCS = [
    {
        "document_id": "943",
        "title": "TFRS 9 Rehberi",
        "category": "Rehber",
        "markdown_content": "# Rehber\n\nİLKE 5\nBankalar beklenen kredi zararı karşılığı ayırır.\n",
        "extraction_method": "markitdown",
    },
    {
        "document_id": "mevzuat_22599",
        "title": "Karşılıklar Yönetmeliği",
        "category": "Yönetmelik",
        "markdown_content": "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n",
        "extraction_method": "html_parser",
    },
]
CACHE = [{"document_id": "943", "title": "TFRS 9 Rehberi", "category": "Rehber", "content": "Rehber"}]


async def _build(tmp_path: Path, **kwargs) -> dict:
    return await seed.build_corpus(
        corpus_path=tmp_path / "corpus.sqlite",
        seed_dir=tmp_path / "seed",
        vector_store_factory=_fake_vector_store,
        **kwargs,
    )


@pytest.fixture
def seed_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "seed"
    directory.mkdir()
    _write_seed_files(directory, docs=DOCS, cache=CACHE)
    return directory


def test_seed_document_validation_rejects_a_copied_or_tampered_hash() -> None:
    with pytest.raises(RuntimeError, match="content hash validation failed"):
        seed._validate_seed_documents(
            [{"document_id": "tampered", "markdown_content": "canonical text", "content_hash": "0" * 64}]
        )


def test_load_records_rejects_non_object_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    path.write_text('{"document_id": "not-a-list"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="object records"):
        seed._load_records(path)
    assert seed._load_records(tmp_path / "missing.json") == []


@pytest.mark.asyncio
async def test_build_creates_a_ready_corpus_from_seed_json(tmp_path: Path, seed_dir: Path):
    result = await _build(tmp_path)

    assert result == {"decision_cache": 1, "documents": 2, "sections": 2, "chunks": 3, "skipped": False}
    db = CorpusDB.open(tmp_path / "corpus.sqlite", readonly=True)
    try:
        readiness = await db.assert_ready(require_corpus=True)
        assert (readiness.documents, readiness.chunks, readiness.catalog) == (2, 3, 1)
        store = DocumentStore(db)
        stored = await store.get_document("943")
        assert stored is not None
        assert stored.markdown_content == DOCS[0]["markdown_content"]
        assert stored.content_hash == hashlib.sha256(DOCS[0]["markdown_content"].encode()).hexdigest()
        sections = await store.get_document_section("mevzuat_22599")
        assert [(s.section_type, s.section_ref) for s in sections] == [("madde", "9")]
        assert await store.load_catalog() and (await store.load_catalog())[0]["document_id"] == "943"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_build_skips_when_the_corpus_already_matches(tmp_path: Path, seed_dir: Path):
    await _build(tmp_path)

    result = await _build(tmp_path)

    assert result["skipped"] is True
    assert result["documents"] == 0


@pytest.mark.asyncio
async def test_build_rewrites_changed_documents_and_archives_the_previous_body(tmp_path: Path, seed_dir: Path):
    await _build(tmp_path)
    changed = json.loads(json.dumps(DOCS))
    changed[1]["markdown_content"] = "MADDE 9 - TFRS 9 karşılık\nBankalar güncel karşılık ayırır.\n"
    _write_seed_files(seed_dir, docs=changed, cache=CACHE)

    result = await _build(tmp_path)

    assert result["skipped"] is False
    assert result["documents"] == 1
    assert result["chunks"] >= 1
    db = CorpusDB.open(tmp_path / "corpus.sqlite", readonly=True)
    try:
        assert (await db.assert_ready()).issues == []
        store = DocumentStore(db)
        history = await store.get_document_history("mevzuat_22599")
        assert len(history) == 1
        assert await store.get_document_history("943") == []
        assert await _fake_vector_store(db).has_document("mevzuat_22599")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_build_force_rewrites_and_reembeds_everything(tmp_path: Path, seed_dir: Path):
    await _build(tmp_path)

    result = await _build(tmp_path, force=True)

    assert result["skipped"] is False
    assert result["documents"] == 2
    assert result["chunks"] == 3


@pytest.mark.asyncio
async def test_build_refuses_a_tampered_seed(tmp_path: Path, seed_dir: Path):
    documents = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    documents[0]["content_hash"] = "0" * 64
    (seed_dir / "documents.json").write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="content hash validation failed"):
        await _build(tmp_path)
    assert not (tmp_path / "corpus.sqlite").exists()


@pytest.mark.asyncio
async def test_export_round_trips_the_seed(tmp_path: Path, seed_dir: Path):
    await _build(tmp_path)
    out_dir = tmp_path / "out"

    documents, catalog = await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=out_dir)

    assert (documents, catalog) == (2, 1)
    exported = json.loads((out_dir / "documents.json").read_text(encoding="utf-8"))
    original = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    for key in ("document_id", "title", "category", "markdown_content", "content_hash", "extraction_method"):
        assert [d[key] for d in exported] == [d[key] for d in original]
    assert json.loads((out_dir / "decision_cache.json").read_text(encoding="utf-8"))[0]["document_id"] == "943"


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


def test_reviewed_seed_contains_required_exact_section_fixtures():
    documents = json.loads((seed.SEED_DIR / "documents.json").read_text(encoding="utf-8"))
    by_id = {document["document_id"]: document for document in documents}
    sections_943 = extract_document_sections("943", by_id["943"]["markdown_content"])
    sections_22599 = extract_document_sections("mevzuat_22599", by_id["mevzuat_22599"]["markdown_content"])

    assert any(item.section_type == "ilke" and item.section_ref == "5" for item in sections_943)
    assert any(item.section_type == "madde" and item.section_ref == "9" for item in sections_22599)
