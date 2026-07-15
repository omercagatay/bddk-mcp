"""Canonical legal-version identity, import, and abstention regression tests."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp.citations import CitationV1, TrustedCitationContext, verify_normalized_range_citation
from bddk_mcp.core.deps import Dependencies
from bddk_mcp.migrations.v0004_canonical_legal_versions import V0004_CANONICAL_LEGAL_VERSIONS
from bddk_mcp.regulatory import (
    LegalVersionBundle,
    LegalVersionBundleError,
    LegalVersionPersistenceError,
    ResolutionReason,
    artifact_id_for,
    blob_id_for,
    canonical_bundle_sha256,
    event_id_for,
    evidence_id_for,
    import_legal_version_bundle,
    legal_version_id_for,
    load_legal_version_bundle,
    resolve_as_of,
    resolve_current,
    status_assertion_id_for,
    validate_canonical_source_uri,
)
from bddk_mcp.store.doc_store import DocumentStore

FIXTURE = Path(__file__).parent / "fixtures" / "legal_versions" / "synthetic_one_family.json"


def _mapping() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _write_bundle(tmp_path: Path, mapping: dict) -> Path:
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    path = tmp_path / "legal-version-family.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _trusted_test_bundle() -> LegalVersionBundle:
    """Return synthetic data with fixture guards disabled only inside this test module."""

    mapping = _mapping()
    mapping["fixture_only"] = False
    for artifact in mapping["artifacts"]:
        artifact["fixture_only"] = False
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    return LegalVersionBundle.model_validate(mapping)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trusted_citation_mapping(*, document_id: str, section_content: str) -> dict:
    """Build a one-version, non-fixture bundle whose normalized hashes are reconstructable."""

    mapping = _mapping()
    mapping["fixture_only"] = False
    version = deepcopy(mapping["versions"][0])
    artifact = deepcopy(
        next(item for item in mapping["artifacts"] if item["artifact_id"] in version["source_artifact_ids"])
    )
    artifact["fixture_only"] = False
    artifact["repository_document_id"] = document_id
    mapping["artifacts"] = [artifact]
    mapping["blobs"] = [next(item for item in mapping["blobs"] if item["blob_id"] == artifact["blob_id"])]
    version["events"]["supersession"] = None
    normalized_document_sha256 = _sha256(section_content)
    version["legal_text_sha256"] = normalized_document_sha256
    version["legal_version_id"] = legal_version_id_for(
        instrument_id=version["instrument_id"],
        version_key=version["version_key"],
        legal_text_sha256=normalized_document_sha256,
    )
    for event in version["events"].values():
        if event is None:
            continue
        event["legal_version_id"] = version["legal_version_id"]
        event["event_id"] = event_id_for(
            legal_version_id=version["legal_version_id"],
            event_type=event["event_type"],
            event_date=date.fromisoformat(event["event_date"]),
            evidence_id=event["evidence"]["evidence_id"],
            target_legal_version_id=event["target_legal_version_id"],
        )
    for assertion in version["status_assertions"]:
        assertion["legal_version_id"] = version["legal_version_id"]
        assertion["assertion_id"] = status_assertion_id_for(
            legal_version_id=version["legal_version_id"],
            status=assertion["status"],
            valid_from=date.fromisoformat(assertion["valid_from"]),
            valid_through=date.fromisoformat(assertion["valid_through"]),
            evidence_id=assertion["evidence"]["evidence_id"],
        )
    occurrence = version["provisions"][0]
    occurrence["legal_version_id"] = version["legal_version_id"]
    occurrence["provision_text_sha256"] = _sha256(section_content)
    occurrence["evidence"]["statement_sha256"] = occurrence["provision_text_sha256"]
    occurrence["evidence"]["evidence_id"] = evidence_id_for(
        artifact_id=occurrence["evidence"]["artifact_id"],
        locator=occurrence["evidence"]["locator"],
        statement_sha256=occurrence["evidence"]["statement_sha256"],
        authority_level=occurrence["evidence"]["authority_level"],
    )
    mapping["versions"] = [version]
    return mapping


async def _attach_repository_section(
    connection,
    mapping: dict,
    *,
    document_id: str,
    section_content: str = "Article 1",
    section_hash: str | None = None,
) -> int:
    version = next(item for item in mapping["versions"] if item["predecessor_version_id"] is None)
    occurrence = version["provisions"][0]
    artifact_id = occurrence["evidence"]["artifact_id"]
    artifact = next(item for item in mapping["artifacts"] if item["artifact_id"] == artifact_id)
    artifact["repository_document_id"] = document_id
    await connection.execute(
        """
        INSERT INTO public.documents (
            document_id, title, markdown_content, content_hash
        ) VALUES ($1, 'Synthetic fixture', $2, $3)
        """,
        document_id,
        section_content,
        version["legal_text_sha256"],
    )
    section_id = await connection.fetchval(
        """
        INSERT INTO public.document_sections (
            doc_id, section_type, section_ref, start_char, end_char,
            content, content_hash, source_content_hash
        ) VALUES ($1, 'article', '1', 0, $2, $3, $4, $5)
        RETURNING id
        """,
        document_id,
        len(section_content),
        section_content,
        section_hash or occurrence["provision_text_sha256"],
        version["legal_text_sha256"],
    )
    occurrence["document_section_id"] = section_id
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    return section_id


class _PinnedPool:
    def __init__(self, connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection

    async def fetch(self, query, *args):
        return await self.connection.fetch(query, *args)


class _ForbiddenPool:
    def acquire(self):
        raise AssertionError("fixture guard must run before acquiring a database connection")


def test_synthetic_family_import_is_deterministic_and_keeps_identity_layers_distinct() -> None:
    first = load_legal_version_bundle(FIXTURE)
    second = load_legal_version_bundle(FIXTURE)

    assert first == second
    assert first.bundle_sha256 == canonical_bundle_sha256(first)
    assert first.fixture_only
    assert first.instrument.jurisdiction == "ZZ"
    assert first.instrument.authority_code == "SYNTHETIC_TEST_AUTHORITY"
    assert "Not Legal Authority" in first.instrument.canonical_title
    assert len(first.blobs) == 2
    assert len(first.artifacts) == 2
    assert len(first.versions) == 2
    assert first.versions[0].legal_version_id != first.artifacts[0].artifact_id
    assert first.versions[1].predecessor_version_id == first.versions[0].legal_version_id
    assert first.versions[0].events.supersession is not None
    assert first.versions[0].events.supersession.target_legal_version_id == first.versions[1].legal_version_id


def test_logical_provision_identity_is_stable_across_the_amendment_chain() -> None:
    bundle = load_legal_version_bundle(FIXTURE)
    logical = bundle.provisions[0]

    assert bundle.versions[0].provisions[0].provision_id == logical.provision_id
    assert bundle.versions[1].provisions[0].provision_id == logical.provision_id
    assert bundle.versions[0].provisions[0].provision_text_sha256 != (
        bundle.versions[1].provisions[0].provision_text_sha256
    )


def test_fixture_bundle_always_abstains_even_when_its_test_claims_are_marked_validated() -> None:
    bundle = load_legal_version_bundle(FIXTURE)
    result = resolve_as_of(
        bundle,
        instrument_id=bundle.instrument.instrument_id,
        as_of=date(2024, 6, 30),
    )

    assert not result.resolved
    assert result.reason is ResolutionReason.FIXTURE_ONLY_DATA


def test_fixture_artifact_guard_still_abstains_if_bundle_marker_is_bypassed() -> None:
    bundle = load_legal_version_bundle(FIXTURE).model_copy(update={"fixture_only": False})

    result = resolve_as_of(
        bundle,
        instrument_id=bundle.instrument.instrument_id,
        as_of=date(2024, 6, 30),
    )

    assert not result.resolved
    assert result.reason is ResolutionReason.FIXTURE_ONLY_DATA


def test_as_of_resolution_returns_only_an_explicitly_validated_test_version() -> None:
    bundle = _trusted_test_bundle()
    result = resolve_as_of(
        bundle,
        instrument_id=bundle.instrument.instrument_id,
        as_of=date(2024, 6, 30),
    )

    assert result.resolved
    assert result.reason is ResolutionReason.RESOLVED
    assert result.legal_version_id == bundle.versions[0].legal_version_id
    assert len(result.evidence_ids) == 3
    assert set(result.evidence_ids) == {
        bundle.versions[0].events.publication.evidence.evidence_id,
        bundle.versions[0].events.effective.evidence.evidence_id,
        bundle.versions[0].status_assertions[0].evidence.evidence_id,
    }


@pytest.mark.parametrize(
    ("as_of", "reason"),
    [
        (date(2024, 1, 31), ResolutionReason.STATUS_NOT_VALIDATED_FOR_DATE),
        (date(2025, 1, 1), ResolutionReason.STATUS_NOT_VALIDATED_FOR_DATE),
        (date(2026, 7, 15), ResolutionReason.STATUS_NOT_VALIDATED_FOR_DATE),
    ],
)
def test_resolver_never_extrapolates_status_or_promotes_an_unvalidated_amendment(
    as_of: date,
    reason: ResolutionReason,
) -> None:
    bundle = _trusted_test_bundle()

    result = resolve_current(
        bundle,
        instrument_id=bundle.instrument.instrument_id,
        current_date=as_of,
    )

    assert not result.resolved
    assert result.reason is reason
    assert result.legal_version_id is None
    assert result.evidence_ids == ()


def test_unknown_instrument_abstains_without_falling_back_to_a_document_match() -> None:
    bundle = _trusted_test_bundle()

    result = resolve_as_of(
        bundle,
        instrument_id="inst_sha256_" + "0" * 64,
        as_of=date(2024, 6, 30),
    )

    assert result.reason is ResolutionReason.INSTRUMENT_NOT_FOUND
    assert not result.resolved


def test_unvalidated_terminal_signal_for_the_date_forces_conflict_abstention(tmp_path: Path) -> None:
    mapping = _mapping()
    mapping["fixture_only"] = False
    for artifact in mapping["artifacts"]:
        artifact["fixture_only"] = False
    version = mapping["versions"][0]
    evidence = deepcopy(version["events"]["effective"]["evidence"])
    evidence["locator"] = "synthetic-metadata/unreviewed-repeal-signal"
    evidence["authority_level"] = "repository_fixture"
    evidence["evidence_id"] = evidence_id_for(
        artifact_id=evidence["artifact_id"],
        locator=evidence["locator"],
        statement_sha256=evidence["statement_sha256"],
        authority_level=evidence["authority_level"],
    )
    version["events"]["repeal"] = {
        "event_id": event_id_for(
            legal_version_id=version["legal_version_id"],
            event_type="repeal",
            event_date=date(2024, 5, 1),
            evidence_id=evidence["evidence_id"],
        ),
        "legal_version_id": version["legal_version_id"],
        "event_type": "repeal",
        "event_date": "2024-05-01",
        "evidence": evidence,
        "validation": {
            "state": "unvalidated",
            "validated_by": None,
            "validated_at": None,
            "method": None,
            "review_record_sha256": None,
        },
        "target_legal_version_id": None,
    }
    bundle = load_legal_version_bundle(_write_bundle(tmp_path, mapping))

    result = resolve_as_of(
        bundle,
        instrument_id=bundle.instrument.instrument_id,
        as_of=date(2024, 6, 30),
    )

    assert result.reason is ResolutionReason.CONFLICTING_STATUS_EVIDENCE
    assert not result.resolved


def test_unvalidated_covering_status_signal_forces_conflict_abstention() -> None:
    mapping = _mapping()
    mapping["fixture_only"] = False
    for artifact in mapping["artifacts"]:
        artifact["fixture_only"] = False
    assertion_validation = mapping["versions"][0]["status_assertions"][0]["validation"]
    assertion_validation.update(
        state="unvalidated",
        validated_by=None,
        validated_at=None,
        method=None,
        review_record_sha256=None,
    )
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    bundle = LegalVersionBundle.model_validate(mapping)

    result = resolve_as_of(
        bundle,
        instrument_id=bundle.instrument.instrument_id,
        as_of=date(2024, 6, 30),
    )

    assert result.reason is ResolutionReason.CONFLICTING_STATUS_EVIDENCE
    assert not result.resolved


def test_changing_mutable_title_does_not_change_instrument_identity(tmp_path: Path) -> None:
    mapping = _mapping()
    original_id = mapping["instrument"]["instrument_id"]
    mapping["instrument"]["canonical_title"] = "Renamed Synthetic Fixture — Still Not Legal Authority"

    bundle = load_legal_version_bundle(_write_bundle(tmp_path, mapping))

    assert bundle.instrument.instrument_id == original_id


def test_changed_blob_hash_cannot_reuse_an_existing_blob_identity(tmp_path: Path) -> None:
    mapping = _mapping()
    mapping["blobs"][0]["content_sha256"] = "f" * 64

    with pytest.raises(LegalVersionBundleError, match="schema validation") as exc_info:
        load_legal_version_bundle(_write_bundle(tmp_path, mapping))

    assert "canonical-version-pilot" not in str(exc_info.value)
    assert "https://" not in str(exc_info.value)


def test_identical_bytes_reuse_blob_but_distinct_acquisitions_do_not_collide() -> None:
    content_sha256 = _sha256("identical synthetic artifact bytes")
    blob_id = blob_id_for(content_sha256=content_sha256)
    first_time = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    first = artifact_id_for(
        blob_id=blob_id,
        canonical_uri="https://authority.example.test/instrument.pdf",
        retrieved_at=first_time,
    )
    later = artifact_id_for(
        blob_id=blob_id,
        canonical_uri="https://authority.example.test/instrument.pdf",
        retrieved_at=first_time + timedelta(seconds=1),
    )
    alternate_uri = artifact_id_for(
        blob_id=blob_id,
        canonical_uri="https://mirror.authority.example.test/instrument.pdf",
        retrieved_at=first_time,
    )

    assert len({first, later, alternate_uri}) == 3


@pytest.mark.parametrize(
    "source_uri",
    (
        "https://authority.example.test/source file.pdf",
        "https://authority.example.test/source.pdf\nignored",
        "https://authority.example.test/source.pdf#page=1",
        "https://user:password@authority.example.test/source.pdf",
        "https://authority.example.test/source.pdf?access_token=do-not-store",
    ),
)
def test_source_acquisition_and_citation_share_fail_early_uri_policy(source_uri: str) -> None:
    with pytest.raises(ValueError, match="source URI"):
        validate_canonical_source_uri(source_uri)


def test_completed_review_cannot_predate_its_source_acquisition(tmp_path: Path) -> None:
    mapping = _mapping()
    mapping["versions"][0]["events"]["publication"]["validation"]["validated_at"] = "2026-07-14T23:59:59Z"

    with pytest.raises(LegalVersionBundleError, match="schema validation"):
        load_legal_version_bundle(_write_bundle(tmp_path, mapping))


def test_provision_occurrence_requires_exact_statement_hash_and_positive_section_id(tmp_path: Path) -> None:
    mapping = _mapping()
    occurrence = mapping["versions"][0]["provisions"][0]
    occurrence["provision_text_sha256"] = "0" * 64
    occurrence["document_section_id"] = 0

    with pytest.raises(LegalVersionBundleError, match="schema validation"):
        load_legal_version_bundle(_write_bundle(tmp_path, mapping))


def test_missing_supersession_evidence_breaks_the_amendment_chain(tmp_path: Path) -> None:
    mapping = _mapping()
    mapping["versions"][0]["events"]["supersession"] = None

    with pytest.raises(LegalVersionBundleError, match="schema validation"):
        load_legal_version_bundle(_write_bundle(tmp_path, mapping))


def test_bundle_rejects_noncanonical_collection_order_and_timestamp_encoding(tmp_path: Path) -> None:
    unsorted = _mapping()
    unsorted["artifacts"].reverse()
    with pytest.raises(LegalVersionBundleError, match="schema validation"):
        load_legal_version_bundle(_write_bundle(tmp_path, unsorted))

    offset_time = _mapping()
    offset_time["artifacts"][0]["retrieved_at"] = "2026-07-15T03:00:00+03:00"
    with pytest.raises(LegalVersionBundleError, match="checksum"):
        load_legal_version_bundle(_write_bundle(tmp_path, offset_time))


def test_checksum_tampering_and_duplicate_json_keys_fail_closed_without_payload_echo(tmp_path: Path) -> None:
    mapping = _mapping()
    mapping["instrument"]["canonical_title"] = "DO-NOT-ECHO-SENSITIVE-FIXTURE-TEXT"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(LegalVersionBundleError, match="checksum") as exc_info:
        load_legal_version_bundle(tampered)
    assert "DO-NOT-ECHO" not in str(exc_info.value)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(LegalVersionBundleError, match="duplicate JSON key"):
        load_legal_version_bundle(duplicate)


def test_v4_schema_normalizes_claims_and_never_conflates_extraction_versions() -> None:
    migration = V0004_CANONICAL_LEGAL_VERSIONS
    ddl = "\n".join(migration.statements).lower()

    assert migration.version == 4
    assert migration.name == "canonical_legal_version_pilot"
    for relation in (
        "public.regulatory_instruments",
        "public.regulatory_family_imports",
        "public.regulatory_source_blobs",
        "public.regulatory_source_artifacts",
        "public.regulatory_evidence",
        "public.regulatory_legal_versions",
        "public.regulatory_legal_version_artifacts",
        "public.regulatory_legal_events",
        "public.regulatory_legal_status_assertions",
        "public.regulatory_provisions",
        "public.regulatory_legal_version_provisions",
    ):
        assert f"create table {relation}" in ddl
    for field in (
        "publication",
        "effective",
        "expiry",
        "repeal",
        "supersession",
        "consolidation",
        "validation_state",
        "evidence_id",
    ):
        assert field in ddl
    assert "references public.document_versions" not in ddl
    assert "markdown_content pg_catalog" not in ddl
    assert "legal_text_sha256" in ddl
    assert "document_section_id" in ddl
    assert "references public.document_sections(id)" in ddl
    assert "unique (document_section_id)" in ddl
    assert "member_manifest pg_catalog.jsonb not null" in ddl
    assert "predecessor_bundle_sha256 pg_catalog.text" in ddl
    assert "imported_current_user pg_catalog.text not null default current_user" in ddl
    assert "imported_session_user pg_catalog.text not null default session_user" in ddl
    assert "create view public.regulatory_validated_section_citations" in ddl
    assert "security_barrier = true" in ddl
    assert "security_invoker = false" in ddl
    assert "occurrence.validation_state = 'validated'" in ddl
    assert "version.validation_state = 'validated'" in ddl
    assert "evidence.authority_level = 'authoritative'" in ddl
    assert "artifact.fixture_only = false" in ddl
    assert "pg_catalog.sha256(pg_catalog.convert_to(section.content, 'utf8'))" in ddl
    assert "pg_catalog.sha256(pg_catalog.convert_to(document.markdown_content, 'utf8'))" in ddl


@pytest.mark.asyncio
async def test_persistence_rejects_fixture_data_before_database_access() -> None:
    bundle = load_legal_version_bundle(FIXTURE)

    with pytest.raises(LegalVersionPersistenceError, match="Fixture-only"):
        await import_legal_version_bundle(
            _ForbiddenPool(),  # type: ignore[arg-type]
            bundle,
            imported_by="unit-test",
        )


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_import_rejects_completed_reviews_beyond_database_clock_allowance(pg_pool) -> None:
    mapping = _mapping()
    future = "2099-01-01T00:00:00Z"
    for version in mapping["versions"]:
        reviews = [version["validation"]]
        reviews.extend(event["validation"] for event in version["events"].values() if event is not None)
        reviews.extend(assertion["validation"] for assertion in version["status_assertions"])
        reviews.extend(occurrence["validation"] for occurrence in version["provisions"])
        for review in reviews:
            if review["state"] in {"validated", "rejected"}:
                review["validated_at"] = future
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    bundle = LegalVersionBundle.model_validate(mapping)

    with pytest.raises(LegalVersionPersistenceError, match="database clock allowance"):
        await import_legal_version_bundle(
            pg_pool,
            bundle,
            imported_by="future-review-test",
            allow_fixture=True,
        )


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_registered_v4_import_is_idempotent_and_persists_section_mapping(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            document_id = "synthetic_legal_version_fixture"
            section_content = "MADDE 1 - Yalnızca sentetik fixture hükmü."
            mapping = _trusted_citation_mapping(
                document_id=document_id,
                section_content=section_content,
            )
            mapping["fixture_only"] = True
            for artifact in mapping["artifacts"]:
                artifact["fixture_only"] = True
            section_id = await _attach_repository_section(
                connection,
                mapping,
                document_id=document_id,
                section_content=section_content,
            )
            bundle = LegalVersionBundle.model_validate(mapping)
            pool = _PinnedPool(connection)

            first = await import_legal_version_bundle(
                pool,
                bundle,
                imported_by="postgres-test",
                allow_fixture=True,
            )
            second = await import_legal_version_bundle(
                pool,
                bundle,
                imported_by="postgres-test",
                allow_fixture=True,
            )

            assert first == second
            assert first.version_count == 1
            assert (
                await connection.fetchval(
                    "SELECT count(*) FROM public.regulatory_family_imports WHERE bundle_id = $1",
                    bundle.bundle_id,
                )
                == 1
            )
            assert (
                await connection.fetchval(
                    "SELECT count(*) FROM public.regulatory_legal_versions WHERE instrument_id = $1",
                    bundle.instrument.instrument_id,
                )
                == 1
            )
            assert (
                await connection.fetchval(
                    """
                SELECT document_section_id
                FROM public.regulatory_legal_version_provisions
                WHERE legal_version_id = $1 AND provision_id = $2
                """,
                    bundle.versions[0].legal_version_id,
                    bundle.provisions[0].provision_id,
                )
                == section_id
            )
            stored_sections = await DocumentStore(pool).get_document_section(
                document_id,
                section_type="article",
                section_ref="1",
            )
            assert len(stored_sections) == 1
            assert stored_sections[0].citation_mapping is None
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_immutable_identity_conflict_rolls_back_all_new_family_rows(pg_pool) -> None:
    bundle = load_legal_version_bundle(FIXTURE)
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                """
                INSERT INTO public.regulatory_instruments (
                    instrument_id, jurisdiction, authority_code, identity_key,
                    canonical_title, instrument_type
                ) VALUES ($1, $2, $3, $4, 'Conflicting title', $5)
                """,
                bundle.instrument.instrument_id,
                bundle.instrument.jurisdiction,
                bundle.instrument.authority_code,
                bundle.instrument.identity_key,
                bundle.instrument.instrument_type,
            )

            with pytest.raises(LegalVersionPersistenceError, match="different immutable fields"):
                await import_legal_version_bundle(
                    _PinnedPool(connection),
                    bundle,
                    imported_by="postgres-test",
                    allow_fixture=True,
                )

            assert (
                await connection.fetchval(
                    """
                SELECT count(*)
                FROM public.regulatory_source_artifacts
                WHERE artifact_id = ANY($1::pg_catalog.text[])
                """,
                    [artifact.artifact_id for artifact in bundle.artifacts],
                )
                == 0
            )
            assert (
                await connection.fetchval(
                    "SELECT count(*) FROM public.regulatory_family_imports WHERE bundle_id = $1",
                    bundle.bundle_id,
                )
                == 0
            )
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_import_refuses_a_section_mapping_with_mismatched_normalized_hashes(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            document_id = "synthetic_legal_version_mismatch"
            section_content = "MADDE 1 - Bilerek eşleşmeyen sentetik hüküm."
            mapping = _trusted_citation_mapping(
                document_id=document_id,
                section_content=section_content,
            )
            await _attach_repository_section(
                connection,
                mapping,
                document_id=document_id,
                section_content=section_content,
                section_hash="f" * 64,
            )
            bundle = LegalVersionBundle.model_validate(mapping)

            with pytest.raises(LegalVersionPersistenceError, match="Provision-section mapping"):
                await import_legal_version_bundle(
                    _PinnedPool(connection),
                    bundle,
                    imported_by="postgres-test",
                )

            assert (
                await connection.fetchval(
                    "SELECT count(*) FROM public.regulatory_family_imports WHERE bundle_id = $1",
                    bundle.bundle_id,
                )
                == 0
            )
            assert (
                await connection.fetchval(
                    """
                    SELECT count(*)
                    FROM public.regulatory_legal_version_provisions
                    WHERE legal_version_id = ANY($1::pg_catalog.text[])
                    """,
                    [version.legal_version_id for version in bundle.versions],
                )
                == 0
            )
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_validation_promotion_is_monotonic_and_every_bundle_snapshot_is_audited(pg_pool) -> None:
    original_mapping = _mapping()
    original = LegalVersionBundle.model_validate(original_mapping)
    promoted_mapping = deepcopy(original_mapping)
    completed_review = deepcopy(promoted_mapping["versions"][0]["validation"])
    amendment = next(item for item in promoted_mapping["versions"] if item["version_key"] == "synthetic-v2")
    amendment["validation"] = completed_review
    for event in amendment["events"].values():
        if event is not None:
            event["validation"] = deepcopy(completed_review)
    for occurrence in amendment["provisions"]:
        occurrence["validation"] = deepcopy(completed_review)
    promoted_mapping["bundle_sha256"] = canonical_bundle_sha256(promoted_mapping)
    promoted = LegalVersionBundle.model_validate(promoted_mapping)

    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            pool = _PinnedPool(connection)
            await import_legal_version_bundle(pool, original, imported_by="review-pipeline", allow_fixture=True)
            await import_legal_version_bundle(pool, promoted, imported_by="review-pipeline", allow_fixture=True)
            await import_legal_version_bundle(pool, promoted, imported_by="review-pipeline", allow_fixture=True)

            state = await connection.fetchval(
                "SELECT validation_state FROM public.regulatory_legal_versions WHERE legal_version_id = $1",
                amendment["legal_version_id"],
            )
            event_states = await connection.fetch(
                """
                SELECT validation_state
                FROM public.regulatory_legal_events
                WHERE legal_version_id = $1
                ORDER BY event_id
                """,
                amendment["legal_version_id"],
            )
            occurrence_state = await connection.fetchval(
                """
                SELECT validation_state
                FROM public.regulatory_legal_version_provisions
                WHERE legal_version_id = $1 AND provision_id = $2
                """,
                amendment["legal_version_id"],
                amendment["provisions"][0]["provision_id"],
            )
            snapshots = await connection.fetch(
                """
                SELECT bundle_sha256, predecessor_bundle_sha256, member_manifest,
                       imported_current_user, imported_session_user
                FROM public.regulatory_family_imports
                WHERE bundle_id = $1
                ORDER BY bundle_sha256
                """,
                original.bundle_id,
            )

            assert state == "validated"
            assert {row["validation_state"] for row in event_states} == {"validated"}
            assert occurrence_state == "validated"
            assert len(snapshots) == 2
            by_checksum = {row["bundle_sha256"]: row for row in snapshots}
            assert by_checksum[original.bundle_sha256]["predecessor_bundle_sha256"] is None
            assert by_checksum[promoted.bundle_sha256]["predecessor_bundle_sha256"] == original.bundle_sha256
            manifests = {}
            for checksum, row in by_checksum.items():
                manifest = row["member_manifest"]
                manifests[checksum] = json.loads(manifest) if isinstance(manifest, str) else manifest
                assert row["imported_current_user"]
                assert row["imported_session_user"]
            assert manifests[original.bundle_sha256] != manifests[promoted.bundle_sha256]
            promoted_versions = {item["id"]: item for item in manifests[promoted.bundle_sha256]["versions"]}
            assert promoted_versions[amendment["legal_version_id"]] == {
                "id": amendment["legal_version_id"],
                "state": "validated",
                "review_record_sha256": completed_review["review_record_sha256"],
            }

            with pytest.raises(LegalVersionPersistenceError, match="terminal|backward"):
                await import_legal_version_bundle(
                    pool,
                    original,
                    imported_by="review-pipeline",
                    allow_fixture=True,
                )
            assert (
                await connection.fetchval(
                    "SELECT validation_state FROM public.regulatory_legal_versions WHERE legal_version_id = $1",
                    amendment["legal_version_id"],
                )
                == "validated"
            )
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_document_store_exposes_only_the_owner_filtered_validated_citation_view(pg_pool) -> None:
    document_id = "trusted_citation_view_test"
    section_content = "MADDE 1 - Yalnızca sentetik test hükmü."
    mapping = _trusted_citation_mapping(document_id=document_id, section_content=section_content)

    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            section_id = await _attach_repository_section(
                connection,
                mapping,
                document_id=document_id,
                section_content=section_content,
            )
            bundle = LegalVersionBundle.model_validate(mapping)
            pool = _PinnedPool(connection)
            await import_legal_version_bundle(pool, bundle, imported_by="review-pipeline")
            store = DocumentStore(pool)

            sections = await store.get_document_section(
                document_id,
                section_type="article",
                section_ref="1",
            )
            assert len(sections) == 1
            assert sections[0].citation_mapping is not None
            assert sections[0].citation_mapping.legal_version_id == bundle.versions[0].legal_version_id
            assert sections[0].citation_mapping.provision_id == bundle.provisions[0].provision_id
            assert (
                await connection.fetchval(
                    "SELECT count(*) FROM public.regulatory_validated_section_citations WHERE document_section_id = $1",
                    section_id,
                )
                == 1
            )

            await connection.execute("CREATE ROLE bddk_v4_citation_view_test_reader NOLOGIN")
            await connection.execute("GRANT USAGE ON SCHEMA public TO bddk_v4_citation_view_test_reader")
            await connection.execute(
                """
                GRANT SELECT ON public.documents, public.document_sections,
                    public.regulatory_validated_section_citations
                TO bddk_v4_citation_view_test_reader
                """
            )
            await connection.execute("SET LOCAL ROLE bddk_v4_citation_view_test_reader")
            try:
                assert await connection.fetchval(
                    "SELECT has_table_privilege(current_user, 'public.regulatory_validated_section_citations', 'SELECT')"
                )
                assert not await connection.fetchval(
                    "SELECT has_table_privilege(current_user, 'public.regulatory_legal_versions', 'SELECT')"
                )
                reader_sections = await store.get_document_section(
                    document_id,
                    section_type="article",
                    section_ref="1",
                )
                assert len(reader_sections) == 1
                assert reader_sections[0].citation_mapping is not None
            finally:
                await connection.execute("RESET ROLE")

            version = bundle.versions[0]
            occurrence = version.provisions[0]
            artifact = bundle.artifacts[0]
            blob = next(item for item in bundle.blobs if item.blob_id == artifact.blob_id)
            evidence = occurrence.evidence
            mutations = (
                (
                    "UPDATE public.regulatory_source_artifacts SET fixture_only = true WHERE artifact_id = $1",
                    (artifact.artifact_id,),
                ),
                (
                    """
                    UPDATE public.regulatory_legal_versions
                    SET validation_state = 'unvalidated', validated_by = NULL, validated_at = NULL,
                        validation_method = NULL, review_record_sha256 = NULL
                    WHERE legal_version_id = $1
                    """,
                    (version.legal_version_id,),
                ),
                (
                    """
                    UPDATE public.regulatory_legal_version_provisions
                    SET validation_state = 'unvalidated', validated_by = NULL, validated_at = NULL,
                        validation_method = NULL, review_record_sha256 = NULL
                    WHERE legal_version_id = $1 AND provision_id = $2
                    """,
                    (version.legal_version_id, occurrence.provision_id),
                ),
                (
                    "UPDATE public.regulatory_evidence SET authority_level = 'secondary' WHERE evidence_id = $1",
                    (evidence.evidence_id,),
                ),
                (
                    "UPDATE public.document_sections SET content_hash = $1 WHERE id = $2",
                    ("f" * 64, section_id),
                ),
                (
                    "UPDATE public.document_sections SET content = content || ' tampered' WHERE id = $1",
                    (section_id,),
                ),
                (
                    "UPDATE public.document_sections SET start_char = start_char + 1 WHERE id = $1",
                    (section_id,),
                ),
                (
                    """
                    UPDATE public.documents
                    SET markdown_content = markdown_content || E'\\nTampered outside the cited range.'
                    WHERE document_id = $1
                    """,
                    (document_id,),
                ),
                (
                    "UPDATE public.regulatory_legal_versions SET version_key = 'tampered-key' WHERE legal_version_id = $1",
                    (version.legal_version_id,),
                ),
                (
                    "UPDATE public.regulatory_source_blobs SET content_sha256 = $1 WHERE blob_id = $2",
                    ("f" * 64, blob.blob_id),
                ),
                (
                    "UPDATE public.regulatory_evidence SET statement_sha256 = $1 WHERE evidence_id = $2",
                    ("f" * 64, evidence.evidence_id),
                ),
                (
                    "UPDATE public.regulatory_source_artifacts SET repository_document_id = NULL WHERE artifact_id = $1",
                    (artifact.artifact_id,),
                ),
                (
                    """
                    DELETE FROM public.regulatory_legal_version_artifacts
                    WHERE legal_version_id = $1 AND artifact_id = $2
                    """,
                    (version.legal_version_id, artifact.artifact_id),
                ),
            )
            for statement, args in mutations:
                savepoint = connection.transaction()
                await savepoint.start()
                try:
                    await connection.execute(statement, *args)
                    if "SET start_char = start_char + 1" in statement:
                        assert (
                            await connection.fetchval(
                                """
                                SELECT count(*)
                                FROM public.regulatory_validated_section_citations
                                WHERE document_section_id = $1
                                """,
                                section_id,
                            )
                            == 0
                        )
                    hidden = await store.get_document_section(
                        document_id,
                        section_type="article",
                        section_ref="1",
                    )
                    assert len(hidden) == 1
                    assert hidden[0].citation_mapping is None
                finally:
                    await savepoint.rollback()
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_official_mcp_session_emits_reconstructable_citation_from_real_validated_view(pg_pool) -> None:
    """Synthetic integration evidence only; this does not assert legal authority."""

    from bddk_mcp.server import create_mcp

    document_id = "synthetic_mcp_pg_citation_integration"
    section_content = "MADDE 1 - Yalnızca sentetik MCP ve PostgreSQL bütünleşme hükmü."
    mapping = _trusted_citation_mapping(document_id=document_id, section_content=section_content)

    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            section_id = await _attach_repository_section(
                connection,
                mapping,
                document_id=document_id,
                section_content=section_content,
            )
            bundle = LegalVersionBundle.model_validate(mapping)
            pool = _PinnedPool(connection)
            await import_legal_version_bundle(pool, bundle, imported_by="synthetic-mcp-pg-test")

            direct = await connection.fetchrow(
                """
                SELECT *
                FROM public.regulatory_validated_section_citations
                WHERE document_section_id = $1
                """,
                section_id,
            )
            assert direct is not None

            await connection.execute("CREATE ROLE bddk_v4_mcp_integration_reader NOLOGIN")
            await connection.execute("GRANT USAGE ON SCHEMA public TO bddk_v4_mcp_integration_reader")
            await connection.execute(
                """
                GRANT SELECT ON public.documents, public.document_sections,
                    public.regulatory_validated_section_citations
                TO bddk_v4_mcp_integration_reader
                """
            )
            await connection.execute("SET LOCAL ROLE bddk_v4_mcp_integration_reader")
            assert not await connection.fetchval(
                "SELECT has_table_privilege(current_user, 'public.regulatory_source_blobs', 'SELECT')"
            )
            assert not await connection.fetchval(
                "SELECT has_table_privilege(current_user, 'public.regulatory_source_artifacts', 'SELECT')"
            )

            deps = Dependencies(
                pool=pool,  # type: ignore[arg-type]
                doc_store=DocumentStore(pool),  # type: ignore[arg-type]
                client=MagicMock(),
                http=None,
            )
            async with create_connected_server_and_client_session(create_mcp(deps)) as session:
                result = await session.call_tool(
                    "get_document_section",
                    {"document_id": document_id, "section_ref": "1"},
                )

            assert result.isError is False
            assert result.structuredContent is not None
            assert result.structuredContent["status"] == "ok"
            citation_payload = result.structuredContent["evidence"][0]["citation"]
            citation = CitationV1.model_validate(citation_payload)
            assert citation.source_document_id == document_id
            assert citation.instrument_id == direct["instrument_id"]
            assert citation.legal_version_id == direct["legal_version_id"]
            assert citation.artifact_id == direct["artifact_id"]
            assert citation.artifact_blob_id == direct["artifact_blob_id"]
            assert citation.artifact_sha256 == direct["artifact_sha256"]
            assert citation.evidence_id == direct["evidence_id"]
            assert citation.provision_id == direct["provision_id"]
            assert citation.artifact_fixture_only is False

            trusted = TrustedCitationContext.model_validate(
                {field: getattr(citation, field) for field in TrustedCitationContext.model_fields}
            )
            verification = verify_normalized_range_citation(
                citation,
                normalized_document=section_content,
                rendered_excerpt=result.structuredContent["results"][0]["content"],
                expected=trusted,
            )
            assert verification.valid
            assert verification.failure_codes == ()
            await connection.execute("RESET ROLE")
        finally:
            await transaction.rollback()
