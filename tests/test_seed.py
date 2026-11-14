"""Tests for seed.py — building the corpus file from seed JSON and exporting it back."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bddk_mcp import cli
from bddk_mcp.core.config import EMBEDDING_MODEL_NAME
from bddk_mcp.ingest import seed
from bddk_mcp.store import vector_store as vector_store_module
from bddk_mcp.store.corpus_db import CorpusDB, CorpusNotReadyError
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
from bddk_mcp.store.section_index import extract_document_sections
from bddk_mcp.store.vector_store import RetrievalProfileError, VectorStore
from tests.conftest import FakeTokenizer, fake_embed


def _fake_vector_store(db: CorpusDB) -> VectorStore:
    return VectorStore(db, embed_fn=fake_embed, tokenizer=FakeTokenizer())


def _forbidden_embed(texts: list[str]) -> np.ndarray:
    raise AssertionError("the embedding model must not run when the seed cache covers every chunk")


def _chunk_blobs(corpus: Path) -> list[tuple[str, int, bytes]]:
    db = CorpusDB.open(corpus, readonly=True)
    try:
        return db.run_sync(
            lambda conn: [
                (row["doc_id"], row["chunk_index"], row["embedding"])
                for row in conn.execute("SELECT doc_id, chunk_index, embedding FROM chunks ORDER BY 1, 2")
            ]
        )
    finally:
        db.close()


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

    assert result == {
        "decision_cache": 1,
        "documents": 2,
        "sections": 2,
        # The "# Rehber" caption opens the İLKE 5 chunk instead of being a chunk of its own.
        "chunks": 2,
        "chunks_cached": 0,
        "skipped": False,
    }
    db = CorpusDB.open(tmp_path / "corpus.sqlite", readonly=True)
    try:
        readiness = await db.assert_ready(require_corpus=True)
        assert (readiness.documents, readiness.chunks, readiness.catalog) == (2, 2, 1)
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
    assert result["chunks"] == 2


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
    stale = out_dir / "embeddings" / "gone.npy"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    documents, catalog, embedded = await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=out_dir)

    assert (documents, catalog, embedded) == (2, 1, 2)
    manifest = json.loads((out_dir / "embeddings" / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["embedding_model"], manifest["embedding_dimension"]) == (EMBEDDING_MODEL_NAME, 768)
    assert set(manifest["documents"]) == {"943", "mevzuat_22599"}
    assert sorted(path.name for path in (out_dir / "embeddings").glob("*.npy")) == ["943.npy", "mevzuat_22599.npy"]
    vectors = np.load(out_dir / "embeddings" / "943.npy", allow_pickle=False)
    assert vectors.dtype == np.float16 and vectors.shape[1] == 768
    exported = json.loads((out_dir / "documents.json").read_text(encoding="utf-8"))
    original = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    for key in ("document_id", "title", "category", "markdown_content", "content_hash", "extraction_method"):
        assert [d[key] for d in exported] == [d[key] for d in original]
    assert json.loads((out_dir / "decision_cache.json").read_text(encoding="utf-8"))[0]["document_id"] == "943"


def _tree_bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()
    }


async def _seed_export_and_partial_corpus(tmp_path: Path, seed_dir: Path) -> Path:
    """Export a ready corpus into ``seed_dir``, then return a corpus with documents and catalog but no chunks."""
    await _build(tmp_path)
    await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir)
    partial = tmp_path / "partial.sqlite"
    db = CorpusDB.open(partial)
    try:
        store = DocumentStore(db)
        await store.store_document(StoredDocument(document_id="943", title="T", markdown_content="MADDE 1\nMetin."))
        await store.replace_catalog(CACHE)
    finally:
        db.close()
    return partial


@pytest.mark.asyncio
async def test_export_refuses_a_corpus_without_chunks_and_leaves_the_seed_untouched(tmp_path: Path, seed_dir: Path):
    partial = await _seed_export_and_partial_corpus(tmp_path, seed_dir)
    before = _tree_bytes(seed_dir)
    assert "embeddings/943.npy" in before

    with pytest.raises(CorpusNotReadyError, match="no retrieval chunks"):
        await seed.export_corpus(corpus_path=partial, seed_dir=seed_dir)

    assert _tree_bytes(seed_dir) == before


def test_export_command_exits_1_on_a_corpus_that_is_not_ready(tmp_path: Path, seed_dir: Path, capsys):
    partial = asyncio.run(_seed_export_and_partial_corpus(tmp_path, seed_dir))
    before = _tree_bytes(seed_dir)

    with pytest.raises(SystemExit) as exited:
        cli.main(["export", "--corpus", str(partial), "--seed-dir", str(seed_dir)])

    assert exited.value.code == 1
    # Other tests may leave logging handlers behind, so look for the error line itself.
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("bddk-mcp export: error:")]
    assert len(lines) == 1 and "Corpus is not ready:" in lines[0] and lines[0].endswith("to prepare the corpus.")
    assert _tree_bytes(seed_dir) == before


@pytest.mark.asyncio
async def test_an_empty_body_document_survives_build_skip_export_and_rebuild(tmp_path: Path, seed_dir: Path):
    empty = {"document_id": "empty_1", "title": "Boş", "category": "Rehber", "markdown_content": ""}
    _write_seed_files(seed_dir, docs=[*DOCS, empty], cache=CACHE)

    first = await _build(tmp_path)
    again = await _build(tmp_path)

    assert first["documents"] == 3
    assert again["skipped"] is True
    db = CorpusDB.open(tmp_path / "corpus.sqlite", readonly=True)
    try:
        stored = await DocumentStore(db).get_document("empty_1")
        assert stored is not None and stored.content_hash == hashlib.sha256(b"").hexdigest()
        assert (await db.assert_ready(require_corpus=True)).documents == 2
    finally:
        db.close()

    out_dir = tmp_path / "out"
    await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=out_dir)
    rebuilt = await seed.build_corpus(
        corpus_path=tmp_path / "fresh.sqlite", seed_dir=out_dir, vector_store_factory=_fake_vector_store
    )

    assert (rebuilt["documents"], rebuilt["skipped"]) == (3, False)
    assert rebuilt["chunks_cached"] == rebuilt["chunks"] == first["chunks"]


def _offline_vector_store(db: CorpusDB) -> VectorStore:
    """A vector store whose chunk tokenizer cannot load, as on a host without the model files."""
    return VectorStore(db, embed_fn=_forbidden_embed)


@pytest.fixture
def offline_tokenizer(monkeypatch):
    def refuse(_model: str):
        raise RetrievalProfileError("The embedding tokenizer could not be loaded.")

    monkeypatch.setattr(vector_store_module, "_load_embedding_tokenizer", refuse)


@pytest.mark.asyncio
async def test_a_build_without_the_tokenizer_fails_before_creating_a_corpus_file(
    tmp_path: Path, seed_dir: Path, offline_tokenizer
):
    with pytest.raises(RetrievalProfileError, match="tokenizer"):
        await seed.build_corpus(
            corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir, vector_store_factory=_offline_vector_store
        )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["seed"]


@pytest.mark.asyncio
async def test_a_build_without_the_tokenizer_leaves_an_existing_corpus_unchanged(
    tmp_path: Path, seed_dir: Path, offline_tokenizer
):
    await _build(tmp_path)
    changed = json.loads(json.dumps(DOCS))
    changed[1]["markdown_content"] = "MADDE 9 - TFRS 9 karşılık\nBankalar güncel karşılık ayırır.\n"
    _write_seed_files(seed_dir, docs=changed, cache=[*CACHE, {"document_id": "x", "title": "X"}])

    with pytest.raises(RetrievalProfileError, match="tokenizer"):
        await seed.build_corpus(
            corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir, vector_store_factory=_offline_vector_store
        )

    db = CorpusDB.open(tmp_path / "corpus.sqlite", readonly=True)
    try:
        readiness = await db.assert_ready(require_corpus=True)
        assert readiness.catalog == 1
        stored = await DocumentStore(db).get_document("mevzuat_22599")
        assert stored is not None and stored.markdown_content == DOCS[1]["markdown_content"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_build_with_nothing_to_chunk_does_not_need_the_tokenizer(tmp_path: Path, offline_tokenizer):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[{"document_id": "empty_1", "title": "Boş", "markdown_content": ""}], cache=CACHE)

    result = await seed.build_corpus(
        corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir, vector_store_factory=_offline_vector_store
    )

    assert (result["documents"], result["decision_cache"], result["chunks"]) == (1, 1, 0)


@pytest.mark.asyncio
async def test_build_reuses_exported_embeddings_without_the_model(tmp_path: Path, seed_dir: Path):
    await _build(tmp_path)
    await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir)

    result = await seed.build_corpus(
        corpus_path=tmp_path / "cached.sqlite",
        seed_dir=seed_dir,
        vector_store_factory=lambda db: VectorStore(db, embed_fn=_forbidden_embed, tokenizer=FakeTokenizer()),
    )

    assert (result["chunks"], result["chunks_cached"]) == (2, 2)
    assert _chunk_blobs(tmp_path / "cached.sqlite") == _chunk_blobs(tmp_path / "corpus.sqlite")
    cached = CorpusDB.open(tmp_path / "cached.sqlite", readonly=True)
    try:
        assert (await cached.assert_ready(require_corpus=True)).issues == []
    finally:
        cached.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("spoil", ["content", "corrupt_file", "model", "force"])
async def test_build_embeds_again_when_the_cache_does_not_match(tmp_path: Path, seed_dir: Path, spoil: str):
    await _build(tmp_path)
    await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir)
    embeddings = seed_dir / "embeddings"
    kwargs = {}
    if spoil == "content":
        changed = json.loads(json.dumps(DOCS))
        changed[1]["markdown_content"] += "Ek fıkra eklendi.\n"
        _write_seed_files(seed_dir, docs=changed, cache=CACHE)
    elif spoil == "corrupt_file":
        (embeddings / "mevzuat_22599.npy").write_bytes(b"not a numpy file")
    elif spoil == "model":
        manifest = json.loads((embeddings / "manifest.json").read_text(encoding="utf-8"))
        manifest["embedding_model"] = "other/model"
        (embeddings / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    else:
        kwargs["force"] = True
    embedded: list[int] = []

    def counting_embed(texts: list[str]) -> np.ndarray:
        embedded.append(len(texts))
        return fake_embed(texts)

    result = await seed.build_corpus(
        corpus_path=tmp_path / "again.sqlite",
        seed_dir=seed_dir,
        vector_store_factory=lambda db: VectorStore(db, embed_fn=counting_embed, tokenizer=FakeTokenizer()),
        **kwargs,
    )

    assert result["chunks_cached"] == result["chunks"] - sum(embedded)
    if spoil in {"content", "corrupt_file"}:
        assert 0 < sum(embedded) < result["chunks"]  # only the spoiled document was embedded again
    else:
        assert (sum(embedded), result["chunks_cached"]) == (result["chunks"], 0)


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


def _canonical_json(records: list[dict]) -> bytes:
    ordered = sorted(records, key=lambda record: record["document_id"])
    return (json.dumps(ordered, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_export_after_build_gives_back_the_seed_bytes(tmp_path: Path):
    """Build from a canonical seed, export, and get the same bytes: timestamps and format survive."""
    documents = []
    for index, document in enumerate(reversed(DOCS)):
        content = document["markdown_content"]
        documents.append(
            {
                "document_id": document["document_id"],
                "title": document["title"],
                "category": document["category"],
                "decision_date": "",
                "decision_number": "",
                "source_url": f"https://www.bddk.org.tr/Mevzuat/DokumanGetir/{index}",
                "markdown_content": content,
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "downloaded_at": 1776786536.134837 + index,
                "extracted_at": 1778524399.189718 + index,
                "extraction_method": document["extraction_method"],
                "total_pages": 1,
                "file_size": 1234 + index,
            }
        )
    catalog = [
        {
            "document_id": document["document_id"],
            "title": document["title"],
            "content": document["title"],
            "decision_date": "",
            "decision_number": "",
            "category": document["category"],
            "source_url": document["source_url"],
        }
        for document in documents
    ]
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "documents.json").write_bytes(_canonical_json(documents))
    (seed_dir / "decision_cache.json").write_bytes(_canonical_json(catalog))

    await _build(tmp_path)
    out_dir = tmp_path / "out"
    await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=out_dir)

    for name in ("documents.json", "decision_cache.json"):
        assert (out_dir / name).read_bytes() == (seed_dir / name).read_bytes(), name


@pytest.mark.asyncio
async def test_build_against_a_corpus_of_another_model_says_to_delete_the_file(tmp_path: Path, seed_dir: Path):
    await _build(tmp_path)

    with pytest.raises(RuntimeError, match=r"different embedding model.*[Dd]elete"):
        await seed.build_corpus(
            corpus_path=tmp_path / "corpus.sqlite",
            seed_dir=seed_dir,
            force=True,
            vector_store_factory=lambda db: VectorStore(
                db, embedding_model="other/model", embed_fn=fake_embed, tokenizer=FakeTokenizer()
            ),
        )


def test_committed_embedding_cache_matches_the_seed_documents():
    """seed_data/embeddings must cover exactly the seed documents; after editing documents.json, run export."""
    documents = json.loads((seed.SEED_DIR / "documents.json").read_text(encoding="utf-8"))
    directory = seed.SEED_DIR / seed.EMBEDDINGS_SUBDIR
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    doc_ids = {document["document_id"] for document in documents}

    assert set(manifest["documents"]) == doc_ids, "stale embedding cache: run `bddk-mcp export` after a seed edit"
    assert {path.stem for path in directory.glob("*.npy")} == doc_ids
    for doc_id in sorted(doc_ids):
        vectors = np.load(directory / f"{doc_id}.npy", allow_pickle=False)
        assert vectors.dtype == np.float16 and vectors.ndim == 2 and vectors.shape[1] == 768, doc_id
        assert np.isfinite(vectors).all(), doc_id


def test_reviewed_seed_contains_required_exact_section_fixtures():
    documents = json.loads((seed.SEED_DIR / "documents.json").read_text(encoding="utf-8"))
    by_id = {document["document_id"]: document for document in documents}
    sections_943 = extract_document_sections("943", by_id["943"]["markdown_content"])
    sections_22599 = extract_document_sections("mevzuat_22599", by_id["mevzuat_22599"]["markdown_content"])

    assert any(item.section_type == "ilke" and item.section_ref == "5" for item in sections_943)
    assert any(item.section_type == "madde" and item.section_ref == "9" for item in sections_22599)


def _seed_documents_and_cache():
    documents = json.loads((seed.SEED_DIR / "documents.json").read_text(encoding="utf-8"))
    cache = {d["document_id"]: d for d in json.loads((seed.SEED_DIR / "decision_cache.json").read_text("utf-8"))}
    return documents, cache


def test_seed_decision_metadata_agrees_with_decision_cache():
    """An empty date or number in one file is a mismatch too: that was the defect."""
    documents, cache = _seed_documents_and_cache()
    assert {d["document_id"] for d in documents} == set(cache)
    mismatches = [
        (d["document_id"], field, d[field], cache[d["document_id"]][field])
        for d in documents
        for field in ("title", "decision_date", "decision_number")
        if d[field] != cache[d["document_id"]][field]
    ]
    assert not mismatches


def test_seed_titles_do_not_start_with_a_parenthesised_date():
    documents, _ = _seed_documents_and_cache()
    prefixed = [d["document_id"] for d in documents if re.match(r"\(\d{2}\.\d{2}\.\d{4}", d["title"])]
    assert not prefixed, prefixed


_PRINTED_DECISION_NUMBER = re.compile(r"Karar Sayısı\s*:\s*(\S+)", re.IGNORECASE)
_PRINTED_DECISION_DATE = re.compile(r"Karar Tarihi\s*:\s*(\d{1,2})[./](\d{1,2})[./](\d{4})", re.IGNORECASE)


def test_seed_decision_number_and_date_equal_the_values_the_decision_prints():
    """Where a Kurul Kararı prints its number and date at the top, the stored fields repeat them."""
    records = json.loads((seed.SEED_DIR / "documents.json").read_text(encoding="utf-8"))
    checked = 0
    mismatches = []
    for record in records:
        if record.get("category") != "Kurul Kararı":
            continue
        head = record["markdown_content"][:1500]
        number, date = _PRINTED_DECISION_NUMBER.search(head), _PRINTED_DECISION_DATE.search(head)
        if number is None or date is None:
            continue
        checked += 1
        printed_date = f"{int(date.group(1)):02d}.{int(date.group(2)):02d}.{date.group(3)}"
        stored = (record["decision_number"], record["decision_date"])
        if stored != (number.group(1), printed_date):
            mismatches.append((record["document_id"], stored, (number.group(1), printed_date)))

    assert checked >= 180
    assert mismatches == []


# -- Build fingerprint ----------------------------------------------------------


def _stored_meta(corpus: Path) -> dict[str, str]:
    db = CorpusDB.open(corpus, readonly=True)
    try:
        return db.run_sync(
            lambda conn: {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
        )
    finally:
        db.close()


def _logical_content(corpus: Path) -> list[str]:
    """Every row of the file as SQL (a writer open and close only bumps the header's change counter)."""
    import sqlite3

    connection = sqlite3.connect(corpus)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def _counting_store(embedded: list[int]):
    def counting_embed(texts: list[str]) -> np.ndarray:
        embedded.append(len(texts))
        return fake_embed(texts)

    return lambda db: VectorStore(db, embed_fn=counting_embed, tokenizer=FakeTokenizer())


@pytest.mark.asyncio
async def test_build_records_the_fingerprint_and_skips_an_unchanged_file(tmp_path: Path, seed_dir: Path):
    from bddk_mcp.store import fingerprint

    await _build(tmp_path)
    stored = fingerprint.decode(_stored_meta(tmp_path / "corpus.sqlite")[fingerprint.FINGERPRINT_META_KEY])
    docs = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    assert stored == fingerprint.build_fingerprint(docs, CACHE)

    assert (await _build(tmp_path))["skipped"] is True


@pytest.mark.asyncio
async def test_a_metadata_only_seed_edit_rewrites_every_document(tmp_path: Path, seed_dir: Path, caplog):
    await _build(tmp_path)
    changed = json.loads(json.dumps(DOCS))
    changed[0]["title"] = "TFRS 9 Rehberi (güncel)"
    _write_seed_files(seed_dir, docs=changed, cache=CACHE)

    with caplog.at_level("WARNING", logger=seed.logger.name):
        result = await _build(tmp_path)

    assert (result["skipped"], result["documents"]) == (False, 2)
    notes = [record.getMessage() for record in caplog.records if "fingerprint" in record.getMessage()]
    assert notes == ["Build fingerprint differs (seed); every document is rewritten"]
    db = CorpusDB.open(tmp_path / "corpus.sqlite", readonly=True)
    try:
        assert (await DocumentStore(db).get_document("943")).title == "TFRS 9 Rehberi (güncel)"
        assert (await db.assert_ready()).issues == []
    finally:
        db.close()
    assert (await _build(tmp_path))["skipped"] is True


@pytest.mark.asyncio
async def test_a_file_without_a_fingerprint_is_rewritten_once_and_then_skipped(tmp_path: Path, seed_dir: Path, caplog):
    from bddk_mcp.store import fingerprint

    await _build(tmp_path)
    db = CorpusDB.open(tmp_path / "corpus.sqlite")
    db.transaction_sync(
        lambda conn: conn.execute("DELETE FROM meta WHERE key = ?", (fingerprint.FINGERPRINT_META_KEY,))
    )
    db.close()

    with caplog.at_level("WARNING", logger=seed.logger.name):
        result = await _build(tmp_path)

    assert (result["skipped"], result["documents"]) == (False, 2)
    assert "No build fingerprint: file predates fingerprinting; every document is rewritten" in caplog.messages
    assert fingerprint.FINGERPRINT_META_KEY in _stored_meta(tmp_path / "corpus.sqlite")
    assert (await _build(tmp_path))["skipped"] is True


@pytest.mark.asyncio
async def test_a_new_file_is_built_without_a_fingerprint_note(tmp_path: Path, seed_dir: Path, caplog):
    with caplog.at_level("WARNING", logger=seed.logger.name):
        await _build(tmp_path)
    assert not [message for message in caplog.messages if "fingerprint" in message]


@pytest.mark.asyncio
async def test_a_parser_source_change_rewrites_every_document_from_the_cached_vectors(
    tmp_path: Path, seed_dir: Path, monkeypatch, caplog
):
    from bddk_mcp.store import fingerprint

    await _build(tmp_path)
    await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir)
    await _build(tmp_path)  # export wrote the stored metadata back into the seed: settle that seed change first
    assert (await _build(tmp_path))["skipped"] is True
    before = _chunk_blobs(tmp_path / "corpus.sqlite")
    real = fingerprint._module_source
    monkeypatch.setattr(
        fingerprint,
        "_module_source",
        lambda name: real(name) + "\n# parser changed\n" if name == "bddk_mcp.store.section_index" else real(name),
    )
    embedded: list[int] = []

    with caplog.at_level("WARNING", logger=seed.logger.name):
        result = await seed.build_corpus(
            corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir, vector_store_factory=_counting_store(embedded)
        )

    assert (result["skipped"], result["documents"]) == (False, 2)
    assert embedded == []  # chunk texts are unchanged, so every vector comes from the seed cache
    assert result["chunks_cached"] == result["chunks"] == 2
    assert "Build fingerprint differs (parser); every document is rewritten" in caplog.messages
    assert _chunk_blobs(tmp_path / "corpus.sqlite") == before
    stored = fingerprint.decode(_stored_meta(tmp_path / "corpus.sqlite")[fingerprint.FINGERPRINT_META_KEY])
    assert stored["parser"] == fingerprint.parser_part()


@pytest.mark.asyncio
async def test_the_fingerprint_is_not_written_when_a_document_fails(tmp_path: Path, seed_dir: Path):
    from bddk_mcp.store import fingerprint

    def failing_embed(texts: list[str]) -> np.ndarray:
        raise RuntimeError("embedding failed")

    with pytest.raises(Exception, match="unavailable|embedding failed"):
        await seed.build_corpus(
            corpus_path=tmp_path / "corpus.sqlite",
            seed_dir=seed_dir,
            vector_store_factory=lambda db: VectorStore(db, embed_fn=failing_embed, tokenizer=FakeTokenizer()),
        )

    assert fingerprint.FINGERPRINT_META_KEY not in _stored_meta(tmp_path / "corpus.sqlite")


@pytest.mark.asyncio
async def test_a_model_revision_only_difference_is_refused_with_delete_and_force(
    tmp_path: Path, seed_dir: Path, monkeypatch
):
    from bddk_mcp.core import config

    await _build(tmp_path)
    before = _logical_content(tmp_path / "corpus.sqlite")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_REVISION", "0" * 40)

    with pytest.raises(RuntimeError, match=r"revision.*[Dd]elete the corpus file.*bddk-mcp build --force"):
        await _build(tmp_path)
    assert _logical_content(tmp_path / "corpus.sqlite") == before

    result = await _build(tmp_path, force=True)  # --force behaves as today and records the new fingerprint
    assert (result["skipped"], result["documents"]) == (False, 2)
    assert (await _build(tmp_path))["skipped"] is True


@pytest.mark.asyncio
async def test_a_passage_prefix_change_is_refused_and_force_embeds_every_document_again(
    tmp_path: Path, seed_dir: Path, monkeypatch
):
    from bddk_mcp.store import fingerprint

    await _build(tmp_path)
    await seed.export_corpus(corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir)
    await _build(tmp_path)  # export wrote the stored metadata back into the seed: settle that seed change first
    assert (await _build(tmp_path))["skipped"] is True
    stored_before = _stored_meta(tmp_path / "corpus.sqlite")[fingerprint.FINGERPRINT_META_KEY]
    before = _logical_content(tmp_path / "corpus.sqlite")
    real = fingerprint._module_source
    monkeypatch.setattr(
        fingerprint,
        "_module_source",
        lambda name: (
            real(name).replace('_PASSAGE_PREFIX = "passage"', '_PASSAGE_PREFIX = "doc"', 1)
            if name == "bddk_mcp.store.vector_store"
            else real(name)
        ),
    )
    embedded: list[int] = []

    with pytest.raises(RuntimeError, match=r"passage prefix.*[Dd]elete the corpus file.*bddk-mcp build --force"):
        await seed.build_corpus(
            corpus_path=tmp_path / "corpus.sqlite", seed_dir=seed_dir, vector_store_factory=_counting_store(embedded)
        )
    assert embedded == []
    assert _logical_content(tmp_path / "corpus.sqlite") == before
    assert _stored_meta(tmp_path / "corpus.sqlite")[fingerprint.FINGERPRINT_META_KEY] == stored_before

    result = await seed.build_corpus(
        corpus_path=tmp_path / "corpus.sqlite",
        seed_dir=seed_dir,
        force=True,
        vector_store_factory=_counting_store(embedded),
    )
    assert (result["skipped"], result["documents"]) == (False, 2)
    assert sum(embedded) == result["chunks"] == 2  # the seed cache is ignored: every chunk is embedded again
    assert result["chunks_cached"] == 0
    stored = fingerprint.decode(_stored_meta(tmp_path / "corpus.sqlite")[fingerprint.FINGERPRINT_META_KEY])
    assert stored["model"] == fingerprint.model_part()
    db = CorpusDB.open(tmp_path / "corpus.sqlite", readonly=True)
    try:
        assert db.readiness_sync(require_corpus=True).stale_fingerprint_parts == []
    finally:
        db.close()
    assert (await _build(tmp_path))["skipped"] is True


@pytest.mark.asyncio
async def test_a_corpus_of_another_model_is_refused_before_any_write(tmp_path: Path, seed_dir: Path):
    await _build(tmp_path)
    db = CorpusDB.open(tmp_path / "corpus.sqlite")
    db.transaction_sync(
        lambda conn: conn.execute("UPDATE meta SET value = 'other/model' WHERE key = 'embedding_model'")
    )
    db.close()
    before = _logical_content(tmp_path / "corpus.sqlite")

    with pytest.raises(RetrievalProfileError, match=r"different embedding model.*delete"):
        await _build(tmp_path)
    assert _logical_content(tmp_path / "corpus.sqlite") == before
