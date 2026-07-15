"""Tests for seed.py — focus on the import-skip logic."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bddk_mcp.corpus_manifest import canonical_manifest_payload, canonical_manifest_sha256
from bddk_mcp.db_lifecycle import DatabaseNotReadyError
from bddk_mcp.ingest import seed
from bddk_mcp.store import vector_store


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
    observed_at = datetime(2026, 1, 1, tzinfo=UTC)
    hashes: dict[str, str] = {}
    for document in normalized_docs:
        content_hash = hashlib.sha256(document.get("markdown_content", "").encode()).hexdigest()
        document["content_hash"] = content_hash
        document.setdefault("downloaded_at", observed_at.timestamp())
        document.setdefault("extracted_at", observed_at.timestamp())
        hashes[document["document_id"]] = content_hash
    normalized_chunks = json.loads(json.dumps(chunks or [], ensure_ascii=False))
    for chunk in normalized_chunks:
        if chunk.get("doc_id") in hashes:
            chunk["content_hash"] = hashes[chunk["doc_id"]]
    artifact_values = {
        "documents.json": normalized_docs,
        "chunks.json": normalized_chunks,
        "decision_cache.json": cache or [],
    }
    (seed_dir / "documents.json").write_text(
        json.dumps(normalized_docs, ensure_ascii=False),
        encoding="utf-8",
    )
    (seed_dir / "chunks.json").write_text(
        json.dumps(normalized_chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    (seed_dir / "decision_cache.json").write_text(json.dumps(cache or [], ensure_ascii=False), encoding="utf-8")
    raw_manifest = {
        "schema_version": 1,
        "manifest_id": "test-seed-corpus-v1",
        "selection_owner": "test-suite",
        "purpose": "Test-only seed import corpus.",
        "exhaustive": False,
        "included_source_classes": ["test-fixtures"],
        "excluded_source_classes": ["all-production-sources"],
        "known_gaps": ["not-authoritative"],
        "freshness": {
            "source_observed_start": observed_at.isoformat(),
            "source_observed_end": observed_at.isoformat(),
            "corpus_built_at": observed_at.isoformat(),
            "scope_reviewed_at": observed_at.isoformat(),
            "business_expectation": "test-only",
            "source_detection_slo_seconds": None,
            "publication_slo_seconds": None,
            "max_manifest_age_seconds": None,
        },
        "artifacts": [
            {
                "role": role,
                "path": name,
                "sha256": hashlib.sha256((seed_dir / name).read_bytes()).hexdigest(),
                "bytes": (seed_dir / name).stat().st_size,
                "records": len(value),
            }
            for role, name, value in (
                ("documents", "documents.json", artifact_values["documents.json"]),
                ("chunks", "chunks.json", artifact_values["chunks.json"]),
                ("decision_cache", "decision_cache.json", artifact_values["decision_cache.json"]),
            )
        ],
        "integrity": {
            "manifest_sha256": "0" * 64,
            "signature_status": "not_configured",
            "signature_reference": None,
        },
    }
    raw_manifest["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw_manifest)
    (seed_dir / "corpus_scope.yml").write_text(
        yaml.safe_dump(raw_manifest, sort_keys=False),
        encoding="utf-8",
    )


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


def _complete_chunk_row(*, text: str = "canonical", index: int = 0) -> dict:
    return {
        "doc_id": "doc-1",
        "chunk_index": index,
        "title": "Title",
        "category": "Test",
        "decision_date": "2026-01-01",
        "decision_number": "1",
        "source_url": "https://example.invalid/document",
        "total_chunks": 1,
        "total_pages": 1,
        "content_hash": _content_hash(text),
        "chunk_start_char": 0,
        "chunk_end_char": len(text),
        "section_type": "",
        "section_ref": "",
        "section_start_char": None,
        "section_end_char": None,
        "section_content_hash": "",
        "chunk_text": text,
    }


def test_chunk_artifact_comparison_is_exact_order_independent_and_duplicate_safe() -> None:
    first = _complete_chunk_row(text="first")
    first["total_chunks"] = 2
    second = _complete_chunk_row(text="second", index=1)
    second["total_chunks"] = 2
    second["chunk_start_char"] = len("first")
    second["chunk_end_char"] = len("firstsecond")

    assert seed._chunk_artifact_matches_generated([second, first], [first, second])

    changed = json.loads(json.dumps([first, second]))
    changed[1]["chunk_text"] = "unsigned replacement"
    assert not seed._chunk_artifact_matches_generated(changed, [first, second])
    assert not seed._chunk_artifact_matches_generated([first, first], [first, second])

    partial = json.loads(json.dumps([first, second]))
    partial[0].pop("chunk_end_char")
    assert not seed._chunk_artifact_matches_generated(partial, [first, second])

    extra = json.loads(json.dumps([first, second]))
    extra[0]["ignored"] = "signed-but-unused"
    assert not seed._chunk_artifact_matches_generated(extra, [first, second])


def test_strict_chunk_drift_is_rejected_but_local_drift_is_explicit() -> None:
    generated = [_complete_chunk_row()]
    reviewed = [_complete_chunk_row(text="different")]
    local_result = {"chunk_artifact_match": None, "corpus_scope_warnings": []}

    seed._record_chunk_artifact_match(
        local_result,
        reviewed_chunks=reviewed,
        generated_chunks=generated,
        strict_release=False,
    )

    assert local_result["chunk_artifact_match"] is False
    assert local_result["corpus_scope_warnings"] == ["chunk_artifact_does_not_match_current_retrieval_profile"]
    with pytest.raises(RuntimeError, match="signed chunk artifact does not exactly match"):
        seed._record_chunk_artifact_match(
            {"chunk_artifact_match": None, "corpus_scope_warnings": []},
            reviewed_chunks=reviewed,
            generated_chunks=generated,
            strict_release=True,
        )


def test_strict_seed_shapes_reject_signed_but_ignored_fields() -> None:
    observed_at = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    document = {
        "document_id": "doc-1",
        "title": "Title",
        "category": "Test",
        "decision_date": "2026-01-01",
        "decision_number": "1",
        "source_url": "https://example.invalid/document",
        "markdown_content": "canonical",
        "content_hash": _content_hash("canonical"),
        "downloaded_at": observed_at,
        "extracted_at": observed_at,
        "extraction_method": "test",
        "total_pages": 1,
        "file_size": 9,
        "authoritative_published_at": observed_at,
        "source_detected_at": observed_at,
        "retrieval_published_at": observed_at,
    }
    cache = {
        "document_id": "doc-1",
        "title": "Title",
        "content": "Summary",
        "decision_date": "2026-01-01",
        "decision_number": "1",
        "category": "Test",
        "source_url": "https://example.invalid/document",
    }

    seed._validate_strict_seed_artifact_shapes([document], [cache])

    extra_document = dict(document, ignored_signed_field="not-persisted")
    with pytest.raises(RuntimeError, match="document artifact schema"):
        seed._validate_strict_seed_artifact_shapes([extra_document], [cache])

    incomplete_cache = dict(cache)
    incomplete_cache.pop("content")
    with pytest.raises(RuntimeError, match="decision-cache artifact schema"):
        seed._validate_strict_seed_artifact_shapes([document], [incomplete_cache])


@pytest.mark.asyncio
async def test_strict_release_checks_membership_after_publisher_locks_in_same_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Connection:
        @asynccontextmanager
        async def transaction(self):
            events.append("transaction_started")
            yield
            events.append("transaction_committed")

    connection = Connection()

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            events.append("connection_acquired")
            yield connection

    identity = SimpleNamespace(release_id="release-1", safe_dict=lambda: {"release_id": "release-1"})

    async def publish(selected_connection, *_args, **_kwargs):
        assert selected_connection is connection
        events.append("publisher_locks_held")
        return identity

    async def assert_membership(selected_connection, **_kwargs):
        assert selected_connection is connection
        events.append("membership_checked")

    readiness = SimpleNamespace(active_corpus_release=identity)
    monkeypatch.setattr("bddk_mcp.db_lifecycle.assert_database_ready", AsyncMock(return_value=readiness))
    monkeypatch.setattr(seed, "publish_strict_corpus_release", publish)
    monkeypatch.setattr(seed, "_assert_strict_seed_membership", assert_membership)

    result = await seed._activate_strict_release(
        Pool(),
        validation=SimpleNamespace(),
        retrieval_profile_sha256="7" * 64,
        expected_documents=[],
        expected_cache=[],
        expected_chunks=[],
        expected_embeddings=[],
        expected_sections={},
        require_quantified_freshness=True,
        require_measured_freshness=True,
        require_verified_signature=True,
    )

    assert result == {"release_id": "release-1"}
    assert events == [
        "connection_acquired",
        "transaction_started",
        "publisher_locks_held",
        "membership_checked",
        "transaction_committed",
    ]


@pytest.mark.asyncio
async def test_strict_membership_rejects_unsigned_derived_or_legal_rows() -> None:
    class Connection:
        def __init__(self) -> None:
            self.sections: list[dict] = []
            self.publications: list[dict] = []
            self.legal_rows = 0

        async def fetch(self, query: str):
            if "FROM public.document_sections" in query:
                return self.sections
            if "FROM public.document_retrieval_publications" in query:
                return self.publications
            return []

        async def fetchval(self, query: str):
            if "regulatory_instruments" in query:
                return self.legal_rows
            return 0

    connection = Connection()
    arguments = {
        "expected_documents": [],
        "expected_cache": [],
        "expected_chunks": [],
        "expected_embeddings": [],
        "expected_sections": {},
        "retrieval_profile_sha256": "7" * 64,
    }

    await seed._assert_strict_seed_membership(connection, **arguments)

    connection.sections = [{"doc_id": "unrepresented"}]
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await seed._assert_strict_seed_membership(connection, **arguments)

    connection.sections = []
    connection.publications = [{"doc_id": "unrepresented"}]
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await seed._assert_strict_seed_membership(connection, **arguments)

    connection.publications = []
    connection.legal_rows = 1
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await seed._assert_strict_seed_membership(connection, **arguments)


def test_strict_embedding_membership_rejects_tampering_and_malformed_vectors() -> None:
    dimension = vector_store.EMBEDDING_DIMENSION
    component = 1.0 / math.sqrt(dimension)
    expected = [component] * dimension
    chunk = {"doc_id": "d1", "chunk_index": 0}

    assert seed._stored_embeddings_match_regeneration(
        [chunk],
        [expected],
        [{**chunk, "embedding": json.dumps(expected)}],
    )

    excessive_error = expected.copy()
    excessive_error[0] += vector_store.PUBLICATION_EMBEDDING_MAX_ABS_ERROR * 2
    assert not seed._stored_embeddings_match_regeneration(
        [chunk],
        [expected],
        [{**chunk, "embedding": json.dumps(excessive_error)}],
    )

    # Every component remains within the absolute tolerance, while accumulated
    # alternating drift violates the independently enforced cosine floor.
    cosine_drift = [value + (0.0009 if index % 2 else -0.0009) for index, value in enumerate(expected)]
    assert max(abs(left - right) for left, right in zip(expected, cosine_drift, strict=True)) < 0.001
    assert not seed._stored_embeddings_match_regeneration(
        [chunk],
        [expected],
        [{**chunk, "embedding": json.dumps(cosine_drift)}],
    )

    for malformed in (None, "not-a-vector", [0.0] * dimension, [float("nan")] * dimension, [1.0]):
        assert not seed._stored_embeddings_match_regeneration(
            [chunk],
            [expected],
            [{**chunk, "embedding": malformed}],
        )


@pytest.mark.asyncio
async def test_strict_embedding_regeneration_covers_every_chunk() -> None:
    dimension = vector_store.EMBEDDING_DIMENSION

    class Store:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def _embed(self, texts, *, prefix):
            assert prefix == "passage"
            self.calls.append(texts)
            return [[1.0, *([0.0] * (dimension - 1))] for _ in texts]

    store = Store()
    chunks = [{"chunk_text": f"chunk {index}"} for index in range(5)]

    vectors = await seed._regenerate_seed_embedding_vectors(store, chunks, batch_size=2)

    assert len(vectors) == len(chunks)
    assert store.calls == [["chunk 0", "chunk 1"], ["chunk 2", "chunk 3"], ["chunk 4"]]


@pytest.mark.asyncio
async def test_strict_membership_fetches_and_rejects_tampered_stored_embedding() -> None:
    document = {
        "document_id": "strict-vector-doc",
        "title": "Strict vector",
        "category": "test",
        "decision_date": "",
        "decision_number": "",
        "source_url": "",
        "markdown_content": "MADDE",
        "content_hash": hashlib.sha256(b"MADDE").hexdigest(),
        "downloaded_at": 1.0,
        "extracted_at": 1.0,
        "extraction_method": "markitdown",
        "total_pages": 1,
        "file_size": 0,
    }
    cache = {
        "document_id": document["document_id"],
        "title": document["title"],
        "content": "",
        "decision_date": "",
        "decision_number": "",
        "category": "test",
        "source_url": "",
    }
    chunk = {
        "doc_id": document["document_id"],
        "chunk_index": 0,
        "title": document["title"],
        "category": "test",
        "decision_date": "",
        "decision_number": "",
        "source_url": "",
        "total_chunks": 1,
        "total_pages": 1,
        "content_hash": document["content_hash"],
        "chunk_start_char": 0,
        "chunk_end_char": 5,
        "section_type": "",
        "section_ref": "",
        "section_start_char": None,
        "section_end_char": None,
        "section_content_hash": "",
        "chunk_text": "MADDE",
    }
    vector = [1.0, *([0.0] * (vector_store.EMBEDDING_DIMENSION - 1))]

    class Connection:
        def __init__(self) -> None:
            self.embedding = json.dumps(vector)

        async def fetch(self, query: str):
            if "FROM public.documents" in query:
                return [document]
            if "FROM public.decision_cache" in query:
                return [cache]
            if "FROM public.document_chunks" in query:
                return [{**chunk, "embedding": self.embedding}]
            if "FROM public.document_sections" in query:
                return []
            if "FROM public.document_retrieval_publications" in query:
                return [
                    {
                        "doc_id": document["document_id"],
                        "content_hash": document["content_hash"],
                        "retrieval_profile_hash": "7" * 64,
                        "expected_chunks": 1,
                    }
                ]
            raise AssertionError("unexpected strict membership query")

        async def fetchval(self, _query: str):
            return 0

    connection = Connection()
    arguments = {
        "expected_documents": [document],
        "expected_cache": [cache],
        "expected_chunks": [chunk],
        "expected_embeddings": [vector],
        "expected_sections": {document["document_id"]: []},
        "retrieval_profile_sha256": "7" * 64,
    }

    await seed._assert_strict_seed_membership(connection, **arguments)
    tampered = vector.copy()
    tampered[0] += 0.002
    connection.embedding = json.dumps(tampered)
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await seed._assert_strict_seed_membership(connection, **arguments)


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
async def test_nonempty_import_requires_a_verified_corpus_manifest_before_dml(temp_seed_dir):
    (temp_seed_dir / "documents.json").write_text("[]", encoding="utf-8")
    (temp_seed_dir / "decision_cache.json").write_text('[{"document_id":"doc-1"}]', encoding="utf-8")
    pool = _SelectOnlyBootstrapPool()

    with (
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch("bddk_mcp.store.doc_store.DocumentStore", return_value=MagicMock()),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=MagicMock()),
        pytest.raises(RuntimeError, match="required corpus manifest is missing"),
    ):
        await seed.import_seed(pool=pool, force=True)

    assert pool.selects == []


@pytest.mark.asyncio
async def test_import_rejects_artifact_tampering_before_dml(temp_seed_dir):
    docs = [{"document_id": "doc-1", "markdown_content": "reviewed"}]
    _write_seed_files(temp_seed_dir, docs=docs)
    (temp_seed_dir / "documents.json").write_text('[{"document_id":"doc-1"}]', encoding="utf-8")
    pool = _SelectOnlyBootstrapPool()

    with (
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch("bddk_mcp.store.doc_store.DocumentStore", return_value=MagicMock()),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=MagicMock()),
        pytest.raises(RuntimeError, match="artifact size differs"),
    ):
        await seed.import_seed(pool=pool, force=True)

    assert pool.selects == []


@pytest.mark.asyncio
async def test_import_reads_documents_from_the_validated_manifest_role_path(temp_seed_dir):
    _write_seed_files(temp_seed_dir, docs=[{"document_id": "doc-1", "markdown_content": "reviewed"}])
    source = temp_seed_dir / "documents.json"
    renamed = temp_seed_dir / "reviewed-documents.json"
    source.rename(renamed)
    manifest_path = temp_seed_dir / "corpus_scope.yml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    documents = next(item for item in raw["artifacts"] if item["role"] == "documents")
    documents["path"] = renamed.name
    documents["sha256"] = hashlib.sha256(renamed.read_bytes()).hexdigest()
    documents["bytes"] = renamed.stat().st_size
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    pool = _SelectOnlyBootstrapPool()

    with (
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch("bddk_mcp.store.doc_store.DocumentStore", return_value=MagicMock()),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=MagicMock()),
        patch.object(seed, "_validate_seed_documents", side_effect=RuntimeError("renamed role was loaded")),
        pytest.raises(RuntimeError, match="renamed role was loaded"),
    ):
        await seed.import_seed(pool=pool, force=True)


@pytest.mark.asyncio
async def test_import_rejects_an_undeclared_reserved_cache_before_database_use(temp_seed_dir):
    _write_seed_files(temp_seed_dir, docs=[{"document_id": "doc-1", "markdown_content": "reviewed"}])
    manifest_path = temp_seed_dir / "corpus_scope.yml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["artifacts"] = [item for item in raw["artifacts"] if item["role"] != "decision_cache"]
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    pool = _SelectOnlyBootstrapPool()

    with pytest.raises(RuntimeError, match="undeclared reserved seed artifact"):
        await seed.import_seed(pool=pool, force=True)

    assert pool.selects == []


@pytest.mark.asyncio
async def test_signed_corpus_bootstrap_uses_a_separate_trust_key(temp_seed_dir):
    _write_seed_files(temp_seed_dir, docs=[{"document_id": "doc-1", "markdown_content": "reviewed"}])
    manifest_path = temp_seed_dir / "corpus_scope.yml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    trusted_key = temp_seed_dir / "trusted-corpus-key.pem"
    trusted_key.write_bytes(public_key)
    raw["integrity"].update(
        signature_status="verified",
        signature_algorithm="ed25519",
        signature_reference="corpus_scope.sig",
        signature_public_key_sha256=hashlib.sha256(public_key).hexdigest(),
    )
    (temp_seed_dir / "corpus_scope.sig").write_bytes(private_key.sign(canonical_manifest_payload(raw)))
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    pool = _SelectOnlyBootstrapPool()

    with pytest.raises(RuntimeError, match="separately supplied trusted public key"):
        await seed.import_seed(pool=pool, force=True, require_verified_signature=True)

    with (
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch("bddk_mcp.store.doc_store.DocumentStore", return_value=MagicMock()),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=MagicMock()),
        patch.object(seed, "_validate_seed_documents", side_effect=RuntimeError("signed corpus was loaded")),
        pytest.raises(RuntimeError, match="signed corpus was loaded"),
    ):
        await seed.import_seed(
            pool=pool,
            force=True,
            require_verified_signature=True,
            trusted_signing_key=trusted_key,
        )


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
async def test_strict_import_reports_separate_publication_without_activating(temp_seed_dir):
    validation = SimpleNamespace(
        warnings=(), manifest=SimpleNamespace(manifest_id="strict-test"), manifest_sha256="a" * 64
    )
    artifacts = {name: object() for name in ("documents", "chunks", "decision_cache")}
    pool = MagicMock()
    vector_store = MagicMock()
    activate = AsyncMock(side_effect=AssertionError("ingestion must not activate a release"))

    with (
        patch.object(seed, "_manifest_seed_artifacts", return_value=(validation, artifacts)),
        patch.object(seed, "_load_manifest_bound_records", return_value=[]),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch("bddk_mcp.store.doc_store.DocumentStore"),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=vector_store),
        patch.object(seed, "_activate_strict_release", new=activate),
    ):
        result = await seed.import_seed(
            pool=pool,
            force=True,
            require_quantified_freshness=True,
            require_measured_freshness=True,
            require_verified_signature=True,
            trusted_signing_key=Path("/separately-mounted/trust.pem"),
        )

    assert result["release_publication_required"] is True
    assert result["active_corpus_release"] is None
    activate.assert_not_awaited()


@pytest.mark.asyncio
async def test_owned_release_publication_pool_requires_exact_publisher_identity(temp_seed_dir):
    validation = SimpleNamespace(
        warnings=(), manifest=SimpleNamespace(manifest_id="strict-test"), manifest_sha256="a" * 64
    )
    artifacts = {name: object() for name in ("documents", "chunks", "decision_cache")}
    pool = MagicMock()
    pool.close = AsyncMock()
    identity = AsyncMock(side_effect=RuntimeError("publisher identity rejected"))

    with (
        patch.object(seed, "_manifest_seed_artifacts", return_value=(validation, artifacts)),
        patch.object(seed, "_load_manifest_bound_records", return_value=[]),
        patch("bddk_mcp.ingest.seed.assert_database_transport", side_effect=lambda value: value),
        patch("bddk_mcp.ingest.seed.asyncpg.create_pool", new=AsyncMock(return_value=pool)) as create_pool,
        patch("bddk_mcp.ingest.seed.assert_database_identity", new=identity),
        pytest.raises(RuntimeError, match="publisher identity rejected"),
    ):
        await seed.publish_seed_release(
            dsn="postgresql://release-publisher",
            trusted_signing_key=Path("/separately-mounted/trust.pem"),
        )

    create_pool.assert_awaited_once()
    assert create_pool.await_args.kwargs["init"].keywords == {"profile": "release-publisher"}
    identity.assert_awaited_once_with(pool, "release-publisher")
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
    first = await seed.import_seed(pool=clean_pool, force=True)
    assert first["chunk_artifact_match"] is False
    assert "chunk_artifact_does_not_match_current_retrieval_profile" in first["corpus_scope_warnings"]

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
