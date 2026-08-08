"""Verified-corpus publication, tamper detection, and append-only evidence tests."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import asyncpg
import pytest

from bddk_mcp import corpus_publication
from bddk_mcp.corpus_generations import (
    CorpusGenerationError,
    collect_generation_storage_evidence,
    inspect_release_retention,
    retain_active_corpus_generation,
)
from bddk_mcp.corpus_publication import (
    CorpusPublicationError,
    activate_staged_corpus_release,
    assert_release_publication_ready,
    inspect_active_corpus_release,
    publish_strict_corpus_release,
    stage_strict_corpus_release,
    strict_verification_evidence_sha256,
)
from bddk_mcp.migrations.v0007_retained_corpus_generations import RETAINED_CORPUS_RELATIONS

_PROFILE_SHA256 = "7" * 64
_ZERO_VECTOR = "[" + ",".join("0" for _ in range(768)) + "]"


def _strict_validation() -> SimpleNamespace:
    return SimpleNamespace(
        manifest_sha256="8" * 64,
        manifest=SimpleNamespace(
            manifest_id="release-test-001",
            artifacts=[
                SimpleNamespace(
                    role="documents",
                    path="documents.json",
                    sha256="3" * 64,
                    bytes=123,
                    records=2,
                )
            ],
            freshness=SimpleNamespace(
                source_detection_slo_seconds=60,
                publication_slo_seconds=120,
                max_manifest_age_seconds=3600,
                slo_evidence_status="measured",
            ),
            integrity=SimpleNamespace(
                signature_status="verified",
                signature_public_key_sha256="9" * 64,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_application_publication_contract_supplies_exact_sql_arguments() -> None:
    expected_completed_at = datetime(2026, 1, 2, tzinfo=UTC)
    returned_row = {
        "release_id": "corpus_release_sha256_" + "1" * 64,
        "manifest_id": "release-test-001",
        "manifest_sha256": "8" * 64,
        "signer_key_sha256": "9" * 64,
        "freshness_policy_result": "quantified_measured_signature_verified_pass",
        "source_detection_slo_seconds": 60,
        "publication_slo_seconds": 120,
        "max_manifest_age_seconds": 3600,
        "retrieval_profile_sha256": _PROFILE_SHA256,
        "corpus_state_sha256": "2" * 64,
        "completed_at": expected_completed_at,
    }

    class RecordingConnection:
        def __init__(self) -> None:
            self.query: str | None = None
            self.arguments: tuple[object, ...] | None = None

        async def fetchrow(self, query: str, *arguments: object):
            self.query = query
            self.arguments = arguments
            return returned_row

    connection = RecordingConnection()
    validation = SimpleNamespace(
        manifest_sha256="8" * 64,
        manifest=SimpleNamespace(
            manifest_id="release-test-001",
            freshness=SimpleNamespace(
                source_detection_slo_seconds=60,
                publication_slo_seconds=120,
                max_manifest_age_seconds=3600,
                slo_evidence_status="measured",
            ),
            integrity=SimpleNamespace(
                signature_status="verified",
                signature_public_key_sha256="9" * 64,
            ),
        ),
    )

    identity = await publish_strict_corpus_release(
        connection,
        validation,
        retrieval_profile_sha256=_PROFILE_SHA256,
        require_quantified_freshness=True,
        require_measured_freshness=True,
        require_verified_signature=True,
    )

    assert connection.arguments == (
        "release-test-001",
        "8" * 64,
        "9" * 64,
        60,
        120,
        3600,
        _PROFILE_SHA256,
    )
    assert connection.query is not None
    assert "WITH canonical_publication_inputs AS MATERIALIZED" in connection.query
    for name, value in (
        ("TimeZone", "UTC"),
        ("DateStyle", "ISO, YMD"),
        ("IntervalStyle", "postgres"),
        ("bytea_output", "hex"),
        ("extra_float_digits", "3"),
    ):
        assert f"pg_catalog.set_config('{name}', '{value}', true)" in connection.query
    assert "CROSS JOIN LATERAL bddk_meta.publish_verified_corpus_release(" in connection.query
    assert "inputs.retrieval_profile_sha256" in connection.query
    assert identity.completed_at == expected_completed_at


@pytest.mark.asyncio
async def test_staging_contract_supplies_exact_sql_arguments_without_activation() -> None:
    staged_at = datetime(2026, 1, 2, tzinfo=UTC)
    expires_at = datetime(2026, 1, 2, 0, 15, tzinfo=UTC)
    returned_row = {
        "request_id": "corpus_release_request_sha256_" + "a" * 64,
        "release_id": "corpus_release_sha256_" + "b" * 64,
        "corpus_state_sha256": "2" * 64,
        "corpus_epoch": 7,
        "staged_at": staged_at,
        "verification_expires_at": expires_at,
    }
    connection = SimpleNamespace(fetchrow=AsyncMock(return_value=returned_row))

    request = await stage_strict_corpus_release(
        connection,
        _strict_validation(),
        signature_sha256="4" * 64,
        verification_evidence_sha256="5" * 64,
        retrieval_profile_sha256=_PROFILE_SHA256,
        verifier_revision_sha256="6" * 64,
        verifier_image_digest="sha256:" + "7" * 64,
        valid_for_seconds=900,
    )

    query, *arguments = connection.fetchrow.await_args.args
    assert "bddk_meta.stage_verified_corpus_release(" in query
    assert "activate_staged_corpus_release" not in query
    assert arguments == [
        "release-test-001",
        "8" * 64,
        "4" * 64,
        "9" * 64,
        "5" * 64,
        60,
        120,
        3600,
        _PROFILE_SHA256,
        "6" * 64,
        "sha256:" + "7" * 64,
        900,
    ]
    assert request.request_id == returned_row["request_id"]
    assert request.release_id == returned_row["release_id"]
    assert request.verification_expires_at == expires_at


@pytest.mark.asyncio
async def test_activation_contract_accepts_only_request_identity() -> None:
    completed_at = datetime(2026, 1, 2, tzinfo=UTC)
    request_id = "corpus_release_request_sha256_" + "a" * 64
    returned_row = {
        "request_id": request_id,
        "release_id": "corpus_release_sha256_" + "1" * 64,
        "manifest_id": "release-test-001",
        "manifest_sha256": "8" * 64,
        "signer_key_sha256": "9" * 64,
        "freshness_policy_result": "quantified_measured_signature_verified_pass",
        "source_detection_slo_seconds": 60,
        "publication_slo_seconds": 120,
        "max_manifest_age_seconds": 3600,
        "retrieval_profile_sha256": _PROFILE_SHA256,
        "corpus_state_sha256": "2" * 64,
        "activation_sequence": 3,
        "completed_at": completed_at,
    }
    connection = SimpleNamespace(fetchrow=AsyncMock(return_value=returned_row))

    receipt = await activate_staged_corpus_release(connection, request_id=request_id)

    query, argument = connection.fetchrow.await_args.args
    assert "activate_staged_corpus_release($1" in query
    assert "stage_verified_corpus_release" not in query
    assert argument == request_id
    assert receipt.request_id == request_id
    assert receipt.activation_sequence == 3
    assert receipt.release.release_id == returned_row["release_id"]


def test_verification_evidence_binds_verifier_and_fresh_run() -> None:
    arguments = {
        "signature_sha256": "4" * 64,
        "retrieval_profile_sha256": _PROFILE_SHA256,
        "verifier_revision_sha256": "6" * 64,
        "verifier_image_digest": "sha256:" + "7" * 64,
        "verification_run_sha256": "a" * 64,
    }

    first = strict_verification_evidence_sha256(_strict_validation(), **arguments)
    repeated = strict_verification_evidence_sha256(_strict_validation(), **arguments)
    second_run = strict_verification_evidence_sha256(
        _strict_validation(),
        **{**arguments, "verification_run_sha256": "b" * 64},
    )

    assert first == repeated
    assert first != second_run
    assert len(first) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", (5, 6, 7))
async def test_publication_readiness_is_exactly_versioned_without_weakening_serving(
    monkeypatch,
    schema_version: int,
) -> None:
    completed_at = datetime(2026, 1, 2, tzinfo=UTC)
    active_row = {
        "release_id": "corpus_release_sha256_" + "1" * 64,
        "manifest_id": "release-test-001",
        "manifest_sha256": "8" * 64,
        "signer_key_sha256": "9" * 64,
        "freshness_policy_result": "quantified_measured_signature_verified_pass",
        "source_detection_slo_seconds": 60,
        "publication_slo_seconds": 120,
        "max_manifest_age_seconds": 3600,
        "retrieval_profile_sha256": _PROFILE_SHA256,
        "corpus_state_sha256": "2" * 64,
        "completed_at": completed_at,
    }

    class Pool:
        async def fetchval(self, query: str, profile: str):
            assert query == corpus_publication._CORPUS_PUBLICATION_READY_SQL
            assert profile == _PROFILE_SHA256
            return True

        async def fetchrow(self, query: str):
            assert query == corpus_publication._ACTIVE_RELEASE_SQL
            return active_row

    inspect_catalog = AsyncMock(return_value=SimpleNamespace(valid=True))
    monkeypatch.setattr(
        corpus_publication,
        "inspect_migration_state",
        AsyncMock(return_value=SimpleNamespace(current_version=schema_version)),
    )
    monkeypatch.setattr(corpus_publication, "inspect_catalog_integrity", inspect_catalog)

    active = await assert_release_publication_ready(
        Pool(),
        retrieval_profile_sha256=_PROFILE_SHA256,
        require_active_release=True,
    )

    assert active is not None
    assert active.release_id == active_row["release_id"]
    inspect_catalog.assert_awaited_once_with(
        ANY,
        expected_schema_version=schema_version,
    )


@pytest.mark.asyncio
async def test_publication_readiness_rejects_unreviewed_schema_versions(monkeypatch) -> None:
    catalog = AsyncMock()
    monkeypatch.setattr(
        corpus_publication,
        "inspect_migration_state",
        AsyncMock(return_value=SimpleNamespace(current_version=4)),
    )
    monkeypatch.setattr(corpus_publication, "inspect_catalog_integrity", catalog)

    with pytest.raises(CorpusPublicationError, match="version 5, 6, or 7"):
        await assert_release_publication_ready(
            SimpleNamespace(),
            retrieval_profile_sha256=_PROFILE_SHA256,
            require_active_release=False,
        )

    catalog.assert_not_awaited()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _ensure_release_publisher_role(connection: asyncpg.Connection) -> None:
    await connection.execute(
        """
        DO $role$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'bddk_release_publisher'
            ) THEN
                CREATE ROLE bddk_release_publisher NOLOGIN;
            END IF;
        END
        $role$
        """
    )


async def _insert_ready_corpus(connection: asyncpg.Connection, document_id: str) -> str:
    content = "# Madde 1\nDoğrulanmış düzenleme metni."
    content_hash = _sha256(content)
    section_hash = _sha256(content)
    await connection.execute(
        """
        INSERT INTO public.decision_cache (
            document_id, title, content, decision_date, decision_number,
            category, source_url, cached_at
        ) VALUES ($1, 'Kanıt', $2, '2026-01-01', '1', 'test',
                  'https://example.invalid/regulation', 1.0)
        """,
        document_id,
        content,
    )
    await connection.execute(
        """
        INSERT INTO public.documents (
            document_id, title, category, decision_date, decision_number,
            source_url, markdown_content, content_hash, total_pages, file_size
        ) VALUES ($1, 'Kanıt', 'test', '2026-01-01', '1',
                  'https://example.invalid/regulation', $2, $3, 1, $4)
        """,
        document_id,
        content,
        content_hash,
        len(content.encode("utf-8")),
    )
    await connection.execute(
        """
        INSERT INTO public.document_sections (
            doc_id, section_type, section_ref, heading, start_char, end_char,
            content, content_hash, source_content_hash
        ) VALUES ($1, 'article', '1', 'Madde 1', 0, $2, $3, $4, $5)
        """,
        document_id,
        len(content),
        content,
        section_hash,
        content_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.document_chunks (
            doc_id, chunk_index, title, category, decision_date,
            decision_number, source_url, total_chunks, total_pages,
            content_hash, chunk_start_char, chunk_end_char, section_type,
            section_ref, section_start_char, section_end_char,
            section_content_hash, chunk_text, embedding
        ) VALUES (
            $1, 0, 'Kanıt', 'test', '2026-01-01', '1',
            'https://example.invalid/regulation', 1, 1, $2, 0, $3,
            'article', '1', 0, $3, $4, $5, $6::public.vector
        )
        """,
        document_id,
        content_hash,
        len(content),
        section_hash,
        content,
        _ZERO_VECTOR,
    )
    await connection.execute(
        """
        INSERT INTO public.document_retrieval_publications (
            doc_id, content_hash, retrieval_profile_hash, expected_chunks
        ) VALUES ($1, $2, $3, 1)
        """,
        document_id,
        content_hash,
        _PROFILE_SHA256,
    )
    return content_hash


async def _insert_canonical_legal_state(
    connection: asyncpg.Connection,
    *,
    document_id: str,
    content_hash: str,
) -> None:
    instrument_id = "inst_sha256_" + "1" * 64
    bundle_id = "family_sha256_" + "2" * 64
    blob_id = "blob_sha256_" + "3" * 64
    artifact_id = "art_sha256_" + "4" * 64
    evidence_id = "evid_sha256_" + "5" * 64
    version_id = "ver_sha256_" + "6" * 64
    event_id = "event_sha256_" + "7" * 64
    assertion_id = "status_sha256_" + "8" * 64
    provision_id = "prov_sha256_" + "9" * 64
    review_hash = "a" * 64
    section_id = await connection.fetchval(
        "SELECT id FROM public.document_sections WHERE doc_id = $1",
        document_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_instruments (
            instrument_id, jurisdiction, authority_code, identity_key,
            canonical_title, instrument_type
        ) VALUES ($1, 'TR', 'BDDK', 'test-regulation', 'Test Düzenlemesi', 'regulation')
        """,
        instrument_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_family_imports (
            bundle_id, bundle_sha256, instrument_id, schema_version,
            fixture_only, imported_by, member_manifest
        ) VALUES ($1, $2, $3, 1, false, 'reviewer',
                  '{"z":1,"documents":["proof"]}'::pg_catalog.jsonb)
        """,
        bundle_id,
        "b" * 64,
        instrument_id,
    )
    await connection.execute(
        "INSERT INTO public.regulatory_source_blobs (blob_id, content_sha256) VALUES ($1, $2)",
        blob_id,
        content_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_source_artifacts (
            artifact_id, blob_id, canonical_uri, source_authority, media_type,
            retrieved_at, repository_document_id, fixture_only
        ) VALUES ($1, $2, 'https://example.invalid/regulation', 'BDDK',
                  'text/markdown', '2026-01-01T00:00:00Z', $3, false)
        """,
        artifact_id,
        blob_id,
        document_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_evidence (
            evidence_id, artifact_id, locator, statement_sha256, authority_level
        ) VALUES ($1, $2, 'article:1', $3, 'authoritative')
        """,
        evidence_id,
        artifact_id,
        content_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_versions (
            legal_version_id, instrument_id, version_key, legal_text_sha256,
            consolidation_state, validation_state, validated_by, validated_at,
            validation_method, review_record_sha256
        ) VALUES ($1, $2, '2026-original', $3, 'original', 'validated',
                  'reviewer', '2026-01-02T00:00:00Z', 'four-eyes', $4)
        """,
        version_id,
        instrument_id,
        content_hash,
        review_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_version_artifacts (
            legal_version_id, artifact_id, source_role
        ) VALUES ($1, $2, 'legal_text')
        """,
        version_id,
        artifact_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_events (
            event_id, legal_version_id, event_type, event_date, evidence_id,
            validation_state, validated_by, validated_at, validation_method,
            review_record_sha256
        ) VALUES ($1, $2, 'publication', '2026-01-01', $3, 'validated',
                  'reviewer', '2026-01-02T00:00:00Z', 'four-eyes', $4)
        """,
        event_id,
        version_id,
        evidence_id,
        review_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_status_assertions (
            assertion_id, legal_version_id, legal_status, valid_from,
            valid_through, evidence_id, validation_state, validated_by,
            validated_at, validation_method, review_record_sha256
        ) VALUES ($1, $2, 'effective', '2026-01-01', '9999-12-31', $3,
                  'validated', 'reviewer', '2026-01-02T00:00:00Z',
                  'four-eyes', $4)
        """,
        assertion_id,
        version_id,
        evidence_id,
        review_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_provisions (
            provision_id, instrument_id, provision_kind, canonical_path
        ) VALUES ($1, $2, 'article', 'article/1')
        """,
        provision_id,
        instrument_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_version_provisions (
            legal_version_id, provision_id, provision_text_sha256,
            document_section_id, evidence_id, validation_state, validated_by,
            validated_at, validation_method, review_record_sha256
        ) VALUES ($1, $2, $3, $4, $5, 'validated', 'reviewer',
                  '2026-01-02T00:00:00Z', 'four-eyes', $6)
        """,
        version_id,
        provision_id,
        content_hash,
        section_id,
        evidence_id,
        review_hash,
    )


async def _publish(
    connection: asyncpg.Connection,
    *,
    manifest_id: str = "release-test-001",
    manifest_sha256: str = "8" * 64,
):
    return await connection.fetchrow(
        """
        SELECT *
        FROM bddk_meta.publish_verified_corpus_release(
            $1, $2, $3, 60, 120, 3600, $4
        )
        """,
        manifest_id,
        manifest_sha256,
        "9" * 64,
        _PROFILE_SHA256,
    )


@asynccontextmanager
async def _rollback_savepoint(connection: asyncpg.Connection):
    savepoint = connection.transaction()
    await savepoint.start()
    try:
        yield
    finally:
        await savepoint.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_release_is_append_only_and_false_ready_mutations_invalidate_active_view(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    document_id = "corpus-release-integrity"
    try:
        await _ensure_release_publisher_role(connection)
        content_hash = await _insert_ready_corpus(connection, document_id)
        await _insert_canonical_legal_state(
            connection,
            document_id=document_id,
            content_hash=content_hash,
        )
        assert await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)

        # Privileged pre-release tampering cannot smuggle a replacement FTS
        # vector into an otherwise self-consistent corpus.
        for table, trigger, predicate in (
            ("document_sections", "trg_document_sections_tsv", "doc_id = $1"),
            ("document_chunks", "chunks_tsv_update", "doc_id = $1"),
        ):
            async with _rollback_savepoint(connection):
                await connection.execute(f"ALTER TABLE public.{table} DISABLE TRIGGER {trigger}")
                await connection.execute(
                    f"UPDATE public.{table} SET tsv = pg_catalog.to_tsvector('simple', 'tampered') WHERE {predicate}",
                    document_id,
                )
                await connection.execute(f"ALTER TABLE public.{table} ENABLE TRIGGER {trigger}")
                assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)

        published = await _publish(connection)
        repeated = await _publish(connection)
        assert published is not None
        assert repeated["release_id"] == published["release_id"]
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_releases") == 1
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_release_activations") == 1
        assert (
            await connection.fetchval("SELECT release_id FROM bddk_meta.active_corpus_release")
            == published["release_id"]
        )

        # CURRENT_TIMESTAMP is transaction-stable, so these two activations
        # deliberately tie on completed_at.  The monotonic sequence—not wall
        # clock ordering—must select the second release.
        replacement = await _publish(
            connection,
            manifest_id="release-test-002",
            manifest_sha256="a" * 64,
        )
        assert replacement["release_id"] != published["release_id"]
        activation_times = await connection.fetch(
            "SELECT completed_at FROM bddk_meta.corpus_release_activations ORDER BY activation_sequence"
        )
        assert len(activation_times) == 2
        assert activation_times[0]["completed_at"] == activation_times[1]["completed_at"]
        assert (
            await connection.fetchval("SELECT release_id FROM bddk_meta.active_corpus_release")
            == replacement["release_id"]
        )

        for statement in (
            "UPDATE bddk_meta.corpus_releases SET manifest_id = 'changed'",
            "DELETE FROM bddk_meta.corpus_release_activations",
        ):
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError):
                    await connection.execute(statement)

        # Every base table behind validated Citation v1 output is part of the
        # active release identity.  Curator changes cannot retain the previous
        # release ID merely because vector retrieval rows were untouched.
        for statement in (
            "UPDATE public.regulatory_evidence SET locator = 'article:1-mutated'",
            "UPDATE public.regulatory_legal_status_assertions SET legal_status = 'unknown'",
            "UPDATE public.regulatory_provisions SET canonical_path = 'article/1-mutated'",
        ):
            async with _rollback_savepoint(connection):
                await connection.execute(statement)
                assert await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
                assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        # An empty or unpublished document must not be hidden by one healthy
        # document elsewhere in the corpus.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "INSERT INTO public.documents (document_id, title) VALUES ('unpublished-empty', 'Empty')"
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        # Section coordinates and content must resolve to the canonical source
        # range, not merely carry a self-consistent hash.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.document_sections SET end_char = end_char - 1 WHERE doc_id = $1",
                document_id,
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        # Chunk text, ranges, document metadata, and section lineage are all
        # checked against canonical rows.  The chunk trigger also removes the
        # now-stale current-profile publication.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.document_chunks SET chunk_end_char = chunk_end_char - 1 WHERE doc_id = $1",
                document_id,
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.document_chunks SET section_content_hash = repeat('a', 64) WHERE doc_id = $1",
                document_id,
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)

        # The state digest also binds non-retrieval release inputs.  A decision
        # cache timestamp mutation remains retrieval-ready but invalidates the
        # exact release identity immediately.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.decision_cache SET cached_at = cached_at + 1 WHERE document_id = $1",
                document_id,
            )
            assert await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_concurrent_publication_serializes_to_one_activation(pg_pool) -> None:
    """Two ingestion workers cannot race duplicate latest activations."""

    document_id = "corpus-release-concurrent"
    release_id: str | None = None
    role_existed = bool(
        await pg_pool.fetchval("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bddk_release_publisher')")
    )
    try:
        async with pg_pool.acquire() as connection, connection.transaction():
            await _ensure_release_publisher_role(connection)
            await _insert_ready_corpus(connection, document_id)

        first, second = await asyncio.gather(
            _publish(pg_pool, manifest_id="release-concurrent-001"),
            _publish(pg_pool, manifest_id="release-concurrent-001"),
        )
        assert first is not None and second is not None
        release_id = str(first["release_id"])
        assert second["release_id"] == release_id
        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM bddk_meta.corpus_release_activations WHERE release_id = $1",
                release_id,
            )
            == 1
        )
    finally:
        # Append-only triggers protect production evidence.  This explicit
        # superuser-only teardown is confined to the disposable PG test DB.
        async with pg_pool.acquire() as connection, connection.transaction():
            cleanup_release_ids = await connection.fetch(
                "SELECT release_id FROM bddk_meta.corpus_releases WHERE manifest_id = 'release-concurrent-001'"
            )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_release_activations "
                "DISABLE TRIGGER reject_corpus_release_activation_update_delete"
            )
            for cleanup_release in cleanup_release_ids:
                await connection.execute(
                    "DELETE FROM bddk_meta.corpus_release_activations WHERE release_id = $1",
                    cleanup_release["release_id"],
                )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_release_activations "
                "ENABLE TRIGGER reject_corpus_release_activation_update_delete"
            )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_releases DISABLE TRIGGER reject_corpus_release_update_delete"
            )
            for cleanup_release in cleanup_release_ids:
                await connection.execute(
                    "DELETE FROM bddk_meta.corpus_releases WHERE release_id = $1",
                    cleanup_release["release_id"],
                )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_releases ENABLE TRIGGER reject_corpus_release_update_delete"
            )
            await connection.execute("DELETE FROM public.documents WHERE document_id = $1", document_id)
            await connection.execute("DELETE FROM public.decision_cache WHERE document_id = $1", document_id)
            if not role_existed:
                await connection.execute("DROP ROLE IF EXISTS bddk_release_publisher")


@pytest.mark.asyncio
async def test_active_release_reader_sanitizes_database_failures() -> None:
    class FailingPool:
        async def fetchrow(self, *_args):
            raise RuntimeError("private path and principal")

    with pytest.raises(CorpusPublicationError) as exc_info:
        await inspect_active_corpus_release(FailingPool())

    assert "private path" not in str(exc_info.value)
    assert "principal" not in str(exc_info.value)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_active_release_is_atomically_retained_sealed_and_costed(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    document_id = "corpus-generation-complete"
    try:
        await _ensure_release_publisher_role(connection)
        content_hash = await _insert_ready_corpus(connection, document_id)
        await connection.execute(
            """
            INSERT INTO public.document_versions (
                document_id, version, content_hash, markdown_content, synced_at
            ) SELECT document_id, 1, content_hash, markdown_content, 1.0
              FROM public.documents WHERE document_id = $1
            """,
            document_id,
        )
        await _insert_canonical_legal_state(
            connection,
            document_id=document_id,
            content_hash=content_hash,
        )
        await connection.execute(
            """
            SELECT pg_catalog.set_config('TimeZone', 'Europe/Istanbul', true),
                   pg_catalog.set_config('DateStyle', 'German', true),
                   pg_catalog.set_config('IntervalStyle', 'sql_standard', true),
                   pg_catalog.set_config('bytea_output', 'escape', true),
                   pg_catalog.set_config('extra_float_digits', '0', true)
            """
        )
        published = await _publish(connection, manifest_id="retained-release-001")
        release_id = str(published["release_id"])
        active_before = dict(await connection.fetchrow("SELECT * FROM bddk_meta.active_corpus_release"))

        await connection.execute(
            """
            SELECT pg_catalog.set_config('TimeZone', 'America/New_York', true),
                   pg_catalog.set_config('DateStyle', 'SQL, DMY', true),
                   pg_catalog.set_config('IntervalStyle', 'iso_8601', true),
                   pg_catalog.set_config('bytea_output', 'hex', true),
                   pg_catalog.set_config('extra_float_digits', '1', true)
            """
        )

        legacy = await inspect_release_retention(connection, release_id=release_id)
        assert legacy is not None
        assert legacy.retention_status == "legacy_v5_unretained"
        assert legacy.generation_id is None

        receipt = await retain_active_corpus_generation(connection, expected_release_id=release_id)
        retained_hash_before = await connection.fetchval(
            "SELECT bddk_meta.retained_row_sha256(member, false) "
            "FROM bddk_retained.documents AS member WHERE generation_id = $1",
            receipt.generation_id,
        )
        await connection.execute(
            """
            SELECT pg_catalog.set_config('TimeZone', 'Pacific/Auckland', true),
                   pg_catalog.set_config('DateStyle', 'ISO, MDY', true),
                   pg_catalog.set_config('IntervalStyle', 'postgres_verbose', true),
                   pg_catalog.set_config('bytea_output', 'escape', true),
                   pg_catalog.set_config('extra_float_digits', '2', true)
            """
        )
        assert (
            await connection.fetchval(
                "SELECT bddk_meta.retained_row_sha256(member, false) "
                "FROM bddk_retained.documents AS member WHERE generation_id = $1",
                receipt.generation_id,
            )
            == retained_hash_before
        )
        repeated = await retain_active_corpus_generation(connection, expected_release_id=release_id)
        assert repeated == receipt

        # Idempotency still re-verifies the physical generation. A privileged
        # restore/owner mutation must never turn an already-bound release into
        # an apparently successful integrity receipt.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "ALTER TABLE bddk_retained.documents DISABLE TRIGGER guard_retained_generation_member"
            )
            await connection.execute(
                "UPDATE bddk_retained.documents SET title = 'private retained tamper' WHERE generation_id = $1",
                receipt.generation_id,
            )
            await connection.execute(
                "ALTER TABLE bddk_retained.documents ENABLE TRIGGER guard_retained_generation_member"
            )
            with pytest.raises(CorpusGenerationError) as captured:
                await retain_active_corpus_generation(
                    connection,
                    expected_release_id=release_id,
                )
            assert captured.value.__cause__ is None
            assert "private retained tamper" not in str(captured.value)
        assert (
            await retain_active_corpus_generation(
                connection,
                expected_release_id=release_id,
            )
            == receipt
        )

        assert receipt.relation_count == len(RETAINED_CORPUS_RELATIONS) == 17
        assert receipt.row_count == 17
        assert receipt.corpus_state_sha256 == published["corpus_state_sha256"]
        assert (
            await connection.fetchval(
                "SELECT bddk_meta.retained_corpus_state_sha256($1, $2)",
                receipt.generation_id,
                _PROFILE_SHA256,
            )
            == receipt.corpus_state_sha256
        )
        assert dict(await connection.fetchrow("SELECT * FROM bddk_meta.active_corpus_release")) == active_before
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generation_seals") == 1
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_retained_releases") == 1
        for relation in RETAINED_CORPUS_RELATIONS:
            assert (
                await connection.fetchval(
                    f"SELECT count(*) FROM bddk_retained.{relation} WHERE generation_id = $1",
                    receipt.generation_id,
                )
                == 1
            )

        retained = await inspect_release_retention(connection, release_id=release_id)
        assert retained is not None
        assert retained.retention_status == "retained"
        assert retained.generation_id == receipt.generation_id
        storage = await collect_generation_storage_evidence(
            connection,
            generation_id=receipt.generation_id,
        )
        assert storage.row_count == receipt.row_count
        assert storage.generation_logical_bytes > 0
        assert storage.retained_store_total_bytes == (
            storage.retained_store_heap_main_bytes
            + storage.retained_store_heap_auxiliary_bytes
            + storage.retained_store_toast_bytes
            + storage.retained_store_index_bytes
        )

        for relation in RETAINED_CORPUS_RELATIONS:
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError):
                    await connection.execute(
                        f"DELETE FROM bddk_retained.{relation} WHERE generation_id = $1",
                        receipt.generation_id,
                    )

        for statement, arguments in (
            ("UPDATE bddk_retained.documents SET title = 'changed' WHERE generation_id = $1", (receipt.generation_id,)),
            ("DELETE FROM bddk_retained.documents WHERE generation_id = $1", (receipt.generation_id,)),
            (
                "INSERT INTO bddk_retained.documents (generation_id, document_id, title) VALUES ($1, 'new', 'new')",
                (receipt.generation_id,),
            ),
            ("TRUNCATE bddk_retained.documents CASCADE", ()),
            (
                "UPDATE bddk_meta.corpus_generation_seals SET row_count = row_count + 1 WHERE generation_id = $1",
                (receipt.generation_id,),
            ),
            (
                "DELETE FROM bddk_meta.corpus_retained_releases WHERE generation_id = $1",
                (receipt.generation_id,),
            ),
            (
                "UPDATE bddk_meta.corpus_generations SET generation_schema_version = 2 WHERE generation_id = $1",
                (receipt.generation_id,),
            ),
            (
                "UPDATE bddk_meta.corpus_generation_relation_inventory "
                "SET row_count = row_count + 1 WHERE generation_id = $1",
                (receipt.generation_id,),
            ),
            ("TRUNCATE bddk_meta.corpus_generation_relation_inventory CASCADE", ()),
        ):
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError):
                    await connection.execute(statement, *arguments)

        await connection.execute(
            "UPDATE public.decision_cache SET cached_at = cached_at + 1 WHERE document_id = $1",
            document_id,
        )
        replacement = await _publish(
            connection,
            manifest_id="retained-release-002",
            manifest_sha256="b" * 64,
        )
        replacement_activation_sequence = await connection.fetchval(
            "SELECT activation_sequence FROM bddk_meta.active_corpus_release"
        )
        second = await retain_active_corpus_generation(
            connection,
            expected_release_id=str(replacement["release_id"]),
        )
        assert second.generation_id != receipt.generation_id
        assert second.corpus_state_sha256 != receipt.corpus_state_sha256
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generation_seals") == 2
        assert (
            await connection.fetchval(
                "SELECT bddk_meta.retained_corpus_state_sha256($1, $2)",
                receipt.generation_id,
                _PROFILE_SHA256,
            )
            == receipt.corpus_state_sha256
        )

        # V5 may append a later activation for an already governed release
        # when the mutable corpus returns to that exact state.  Retention is
        # release-idempotent and must reuse the original physical generation,
        # not attempt a conflicting second release binding.
        await connection.execute(
            "UPDATE public.decision_cache SET cached_at = cached_at - 1 WHERE document_id = $1",
            document_id,
        )
        reactivated = await _publish(connection, manifest_id="retained-release-001")
        assert reactivated["release_id"] == release_id
        assert (
            await connection.fetchval("SELECT activation_sequence FROM bddk_meta.active_corpus_release")
            > replacement_activation_sequence
        )
        reused = await retain_active_corpus_generation(
            connection,
            expected_release_id=release_id,
        )
        assert reused == receipt
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generations") == 2
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generation_seals") == 2

        resigned = await _publish(
            connection,
            manifest_id="retained-release-003",
            manifest_sha256="c" * 64,
        )
        assert resigned["release_id"] not in {release_id, replacement["release_id"]}

        # A newly governed release may share an existing physical generation
        # only after the retained bytes and inventory are freshly verified.
        # Simulate privileged restore/owner tampering, then prove the binding
        # is refused and sanitized before rolling the tamper back.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "ALTER TABLE bddk_retained.documents DISABLE TRIGGER guard_retained_generation_member"
            )
            await connection.execute(
                "UPDATE bddk_retained.documents SET title = 'private retained tamper' WHERE generation_id = $1",
                receipt.generation_id,
            )
            await connection.execute(
                "ALTER TABLE bddk_retained.documents ENABLE TRIGGER guard_retained_generation_member"
            )
            async with _rollback_savepoint(connection):
                with pytest.raises(CorpusGenerationError) as captured:
                    await retain_active_corpus_generation(
                        connection,
                        expected_release_id=str(resigned["release_id"]),
                    )
            assert captured.value.__cause__ is None
            assert "private retained tamper" not in str(captured.value)
            assert not await connection.fetchval(
                "SELECT EXISTS (SELECT 1 FROM bddk_meta.corpus_retained_releases WHERE release_id = $1)",
                resigned["release_id"],
            )

        shared = await retain_active_corpus_generation(
            connection,
            expected_release_id=str(resigned["release_id"]),
        )
        assert shared.release_id == resigned["release_id"]
        assert shared.generation_id == receipt.generation_id
        assert shared.seal_id == receipt.seal_id
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generations") == 2
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generation_seals") == 2
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_retained_releases") == 3
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_relation",
    (
        *(f"bddk_retained.{relation}" for relation in RETAINED_CORPUS_RELATIONS),
        "bddk_meta.corpus_generation_relation_inventory",
        "bddk_meta.corpus_generation_seals",
        "bddk_meta.corpus_retained_releases",
    ),
)
async def test_retention_failure_at_each_durable_stage_rolls_back_without_changing_active_release(
    pg_pool,
    failure_relation: str,
) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    document_id = "corpus-generation-failure"
    try:
        await _ensure_release_publisher_role(connection)
        content_hash = await _insert_ready_corpus(connection, document_id)
        await connection.execute(
            """
            INSERT INTO public.document_versions (
                document_id, version, content_hash, markdown_content, synced_at
            ) SELECT document_id, 1, content_hash, markdown_content, 1.0
              FROM public.documents WHERE document_id = $1
            """,
            document_id,
        )
        await _insert_canonical_legal_state(
            connection,
            document_id=document_id,
            content_hash=content_hash,
        )
        published = await _publish(connection, manifest_id="retention-failure-001")
        release_id = str(published["release_id"])
        active_before = dict(await connection.fetchrow("SELECT * FROM bddk_meta.active_corpus_release"))
        await connection.execute(
            """
            CREATE FUNCTION pg_temp.fail_generation_copy()
            RETURNS trigger LANGUAGE plpgsql AS $function$
            BEGIN
                RAISE EXCEPTION 'private retained text' USING ERRCODE = '55000';
            END
            $function$
            """
        )
        await connection.execute(
            f"""
            CREATE TRIGGER fail_generation_stage
            BEFORE INSERT ON {failure_relation}
            FOR EACH ROW EXECUTE FUNCTION pg_temp.fail_generation_copy()
            """  # identifiers come only from the closed parameter list above
        )

        async with _rollback_savepoint(connection):
            with pytest.raises(CorpusGenerationError) as captured:
                await retain_active_corpus_generation(connection, expected_release_id=release_id)

        assert captured.value.__cause__ is None
        assert "private retained text" not in str(captured.value)
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generations") == 0
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_generation_seals") == 0
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_retained_releases") == 0
        assert dict(await connection.fetchrow("SELECT * FROM bddk_meta.active_corpus_release")) == active_before
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_concurrent_member_insert_cannot_append_after_generation_seal(pg_pool) -> None:
    """Close the insert-before-seal/FK-wait race with a generation row lock."""

    token = _sha256(f"retained-generation-race-{id(pg_pool)}-{datetime.now(tz=UTC).isoformat()}")
    release_id = "corpus_release_sha256_" + token
    generation_id = "corpus_generation_sha256_" + _sha256("generation-" + token)
    seal_id = "corpus_generation_seal_sha256_" + _sha256("seal-" + token)
    setup = await pg_pool.acquire()
    writer = await pg_pool.acquire()
    appender = await pg_pool.acquire()
    writer_transaction = writer.transaction()
    writer_started = False
    append_task: asyncio.Task[str] | None = None
    activation_sequence: int | None = None
    try:
        activation_sequence = int(
            await setup.fetchval(
                """
                WITH release AS (
                    INSERT INTO bddk_meta.corpus_releases (
                        release_id, manifest_id, manifest_sha256,
                        signer_key_sha256, freshness_policy_result,
                        source_detection_slo_seconds, publication_slo_seconds,
                        max_manifest_age_seconds, retrieval_profile_sha256,
                        corpus_state_sha256
                    ) VALUES (
                        $1, $2, $3, $4,
                        'quantified_measured_signature_verified_pass',
                        60, 120, 3600, $5, $6
                    ) RETURNING release_id
                )
                INSERT INTO bddk_meta.corpus_release_activations (
                    release_id, corpus_epoch, actor_fingerprint_sha256
                ) SELECT release_id, 0, $7 FROM release
                RETURNING activation_sequence
                """,
                release_id,
                "race-" + token[:24],
                _sha256("manifest-" + token),
                _sha256("signer-" + token),
                _sha256("profile-" + token),
                _sha256("state-" + token),
                _sha256("actor-" + token),
            )
        )
        await writer_transaction.start()
        writer_started = True
        await writer.execute(
            """
            INSERT INTO bddk_meta.corpus_generations (
                generation_id, generation_schema_version,
                source_activation_sequence, source_release_id,
                corpus_state_sha256, retrieval_profile_sha256,
                staged_by_fingerprint_sha256
            )
            SELECT $1, 1, activation_sequence, release_id,
                   corpus_state_sha256, retrieval_profile_sha256, $2
            FROM bddk_meta.corpus_release_activations AS activation
            JOIN bddk_meta.corpus_releases AS release USING (release_id)
            WHERE activation_sequence = $3
            """,
            generation_id,
            _sha256("stager-" + token),
            activation_sequence,
        )

        append_task = asyncio.create_task(
            appender.execute(
                "INSERT /* retained-generation-seal-race */ "
                "INTO bddk_retained.documents (generation_id, document_id, title) "
                "VALUES ($1, $2, 'race proof')",
                generation_id,
                "race-" + token[:32],
            )
        )
        await asyncio.sleep(0.1)
        await writer.execute(
            """
            INSERT INTO bddk_meta.corpus_generation_seals (
                seal_id, generation_id, corpus_state_sha256,
                retrieval_profile_sha256, inventory_sha256,
                relation_count, row_count, sealed_by_fingerprint_sha256
            )
            SELECT $1, generation_id, corpus_state_sha256,
                   retrieval_profile_sha256, $2, 17, 0, $3
            FROM bddk_meta.corpus_generations
            WHERE generation_id = $4
            """,
            seal_id,
            _sha256("inventory-" + token),
            _sha256("sealer-" + token),
            generation_id,
        )
        await writer_transaction.commit()
        writer_started = False

        with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError):
            await asyncio.wait_for(append_task, timeout=5)
        assert not await setup.fetchval(
            "SELECT EXISTS (SELECT 1 FROM bddk_retained.documents WHERE generation_id = $1)",
            generation_id,
        )
    finally:
        if append_task is not None and not append_task.done():
            append_task.cancel()
            await asyncio.gather(append_task, return_exceptions=True)
        if writer_started:
            await writer_transaction.rollback()
        if activation_sequence is not None:
            cleanup = setup.transaction()
            await cleanup.start()
            try:
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_generation_seals "
                    "DISABLE TRIGGER reject_corpus_generation_seals_update_delete"
                )
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_generations DISABLE TRIGGER reject_corpus_generations_update_delete"
                )
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_release_activations "
                    "DISABLE TRIGGER reject_corpus_release_activation_update_delete"
                )
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_releases DISABLE TRIGGER reject_corpus_release_update_delete"
                )
                await setup.execute(
                    "DELETE FROM bddk_meta.corpus_generation_seals WHERE generation_id = $1",
                    generation_id,
                )
                await setup.execute(
                    "DELETE FROM bddk_meta.corpus_generations WHERE generation_id = $1",
                    generation_id,
                )
                await setup.execute(
                    "DELETE FROM bddk_meta.corpus_release_activations WHERE activation_sequence = $1",
                    activation_sequence,
                )
                await setup.execute(
                    "DELETE FROM bddk_meta.corpus_releases WHERE release_id = $1",
                    release_id,
                )
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_releases ENABLE TRIGGER reject_corpus_release_update_delete"
                )
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_release_activations "
                    "ENABLE TRIGGER reject_corpus_release_activation_update_delete"
                )
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_generations ENABLE TRIGGER reject_corpus_generations_update_delete"
                )
                await setup.execute(
                    "ALTER TABLE bddk_meta.corpus_generation_seals "
                    "ENABLE TRIGGER reject_corpus_generation_seals_update_delete"
                )
                await cleanup.commit()
            except Exception:
                await cleanup.rollback()
                raise
        await pg_pool.release(appender)
        await pg_pool.release(writer)
        await pg_pool.release(setup)
