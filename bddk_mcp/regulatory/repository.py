"""Transactional, immutable persistence for canonical legal-version bundles."""

from __future__ import annotations

import re
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import asyncpg

from bddk_mcp.regulatory.legal_versions import (
    LegalEvent,
    LegalVersion,
    LegalVersionBundle,
    ValidationRecord,
    canonical_bundle_sha256,
)

_IMPORTER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,199}$")
_SQL_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LOCK_TIMEOUT = "5s"


class _Pool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[Any]: ...


class LegalVersionPersistenceError(RuntimeError):
    """Sanitized fail-closed persistence error."""


@dataclass(frozen=True, slots=True)
class LegalVersionImportResult:
    """Privacy-safe identity and counts for one completed import."""

    bundle_id: str
    bundle_sha256: str
    instrument_id: str
    artifact_count: int
    version_count: int
    provision_count: int
    fixture_only: bool


def _validation_values(validation: ValidationRecord) -> dict[str, Any]:
    return {
        "validation_state": validation.state.value,
        "validated_by": validation.validated_by,
        "validated_at": validation.validated_at,
        "validation_method": validation.method,
        "review_record_sha256": validation.review_record_sha256,
    }


def _ordered_versions(bundle: LegalVersionBundle) -> tuple[LegalVersion, ...]:
    """Topologically order a closed predecessor chain without trusting input order."""

    remaining = {version.legal_version_id: version for version in bundle.versions}
    ordered: list[LegalVersion] = []
    emitted: set[str] = set()
    while remaining:
        ready = sorted(
            (
                version
                for version in remaining.values()
                if version.predecessor_version_id is None
                or version.predecessor_version_id in emitted
            ),
            key=lambda version: version.legal_version_id,
        )
        if not ready:
            raise LegalVersionPersistenceError(
                "Legal-version predecessor chain is cyclic; import refused."
            )
        for version in ready:
            ordered.append(version)
            emitted.add(version.legal_version_id)
            del remaining[version.legal_version_id]
    return tuple(ordered)


def _events(version: LegalVersion) -> tuple[LegalEvent, ...]:
    return tuple(
        event
        for event in (
            version.events.publication,
            version.events.effective,
            version.events.expiry,
            version.events.repeal,
            version.events.supersession,
            version.events.consolidation,
        )
        if event is not None
    )


def _evidence(bundle: LegalVersionBundle) -> tuple[Any, ...]:
    by_id = {}
    for version in bundle.versions:
        for event in _events(version):
            by_id[event.evidence.evidence_id] = event.evidence
        for assertion in version.status_assertions:
            by_id[assertion.evidence.evidence_id] = assertion.evidence
        for occurrence in version.provisions:
            by_id[occurrence.evidence.evidence_id] = occurrence.evidence
    return tuple(by_id[key] for key in sorted(by_id))


def _qualified_table(table: str) -> str:
    parts = table.split(".")
    if len(parts) != 2 or any(not _SQL_IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise RuntimeError("repository table identifier is invalid")
    return table


async def _immutable_insert(
    connection: Any,
    *,
    table: str,
    key_columns: tuple[str, ...],
    values: dict[str, Any],
    object_kind: str,
    compare_columns: tuple[str, ...] | None = None,
) -> None:
    """Insert once or prove an existing stable identity has identical fields."""

    table = _qualified_table(table)
    columns = tuple(values)
    if not key_columns or any(column not in values for column in key_columns):
        raise RuntimeError("repository key columns are invalid")
    if any(not _SQL_IDENTIFIER_RE.fullmatch(column) for column in columns):
        raise RuntimeError("repository column identifier is invalid")
    compared = compare_columns or tuple(
        column for column in columns if column not in key_columns
    )
    if any(column not in values for column in compared):
        raise RuntimeError("repository comparison columns are invalid")

    placeholders = ", ".join(
        "$" + str(position) for position in range(1, len(columns) + 1)
    )
    column_sql = ", ".join(columns)
    key_sql = ", ".join(key_columns)
    first_key = key_columns[0]
    if compared:
        existing = ", ".join(f"existing.{column}" for column in compared)
        excluded = ", ".join(f"EXCLUDED.{column}" for column in compared)
        condition = f"ROW({existing}) IS NOT DISTINCT FROM ROW({excluded})"
    else:
        condition = "true"
    query = (
        f"INSERT INTO {table} AS existing ({column_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT ({key_sql}) DO UPDATE SET {first_key} = EXCLUDED.{first_key} "
        f"WHERE {condition} RETURNING true"
    )
    persisted = await connection.fetchval(
        query, *(values[column] for column in columns)
    )
    if persisted is not True:
        raise LegalVersionPersistenceError(
            f"Existing {object_kind} identity has different immutable fields; import rolled back."
        )


async def import_legal_version_bundle(
    pool: _Pool,
    bundle: LegalVersionBundle,
    *,
    imported_by: str,
    allow_fixture: bool = False,
) -> LegalVersionImportResult:
    """Atomically import one immutable family bundle under an advisory lock.

    Fixture data is rejected by default. The allow_fixture switch exists only
    for disposable validation environments and does not affect resolver
    guards. Conflicting stable identities abort the transaction rather than
    overwriting a reviewed legal claim.
    """

    if not _IMPORTER_RE.fullmatch(imported_by):
        raise LegalVersionPersistenceError("imported_by is invalid; import refused.")
    has_fixture_artifact = any(artifact.fixture_only for artifact in bundle.artifacts)
    if (bundle.fixture_only or has_fixture_artifact) and not allow_fixture:
        raise LegalVersionPersistenceError(
            "Fixture-only legal-version data cannot be imported here."
        )
    if canonical_bundle_sha256(bundle) != bundle.bundle_sha256:
        raise LegalVersionPersistenceError(
            "Legal-version bundle checksum does not match; import refused."
        )

    try:
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'")
            await connection.fetchval(
                "SELECT pg_catalog.pg_advisory_xact_lock("
                "pg_catalog.hashtextextended($1::pg_catalog.text, 1280066885))",
                bundle.instrument.instrument_id,
            )
            instrument = bundle.instrument
            await _immutable_insert(
                connection,
                table="public.regulatory_instruments",
                key_columns=("instrument_id",),
                values={
                    "instrument_id": instrument.instrument_id,
                    "jurisdiction": instrument.jurisdiction,
                    "authority_code": instrument.authority_code,
                    "identity_key": instrument.identity_key,
                    "canonical_title": instrument.canonical_title,
                    "instrument_type": instrument.instrument_type,
                },
                object_kind="instrument",
            )
            for artifact in sorted(bundle.artifacts, key=lambda item: item.artifact_id):
                await _immutable_insert(
                    connection,
                    table="public.regulatory_source_artifacts",
                    key_columns=("artifact_id",),
                    values={
                        "artifact_id": artifact.artifact_id,
                        "content_sha256": artifact.content_sha256,
                        "canonical_uri": artifact.canonical_uri,
                        "source_authority": artifact.source_authority,
                        "media_type": artifact.media_type,
                        "retrieved_at": artifact.retrieved_at,
                        "repository_document_id": artifact.repository_document_id,
                        "fixture_only": artifact.fixture_only,
                    },
                    object_kind="source artifact",
                )
            for evidence in _evidence(bundle):
                await _immutable_insert(
                    connection,
                    table="public.regulatory_evidence",
                    key_columns=("evidence_id",),
                    values={
                        "evidence_id": evidence.evidence_id,
                        "artifact_id": evidence.artifact_id,
                        "locator": evidence.locator,
                        "statement_sha256": evidence.statement_sha256,
                        "authority_level": evidence.authority_level.value,
                    },
                    object_kind="evidence",
                )
            for version in _ordered_versions(bundle):
                await _immutable_insert(
                    connection,
                    table="public.regulatory_legal_versions",
                    key_columns=("legal_version_id",),
                    values={
                        "legal_version_id": version.legal_version_id,
                        "instrument_id": version.instrument_id,
                        "version_key": version.version_key,
                        "legal_text_sha256": version.legal_text_sha256,
                        "predecessor_version_id": version.predecessor_version_id,
                        "consolidation_state": version.consolidation_state.value,
                        **_validation_values(version.validation),
                    },
                    object_kind="legal version",
                )
            for version in sorted(
                bundle.versions, key=lambda item: item.legal_version_id
            ):
                for artifact_id in sorted(version.source_artifact_ids):
                    await _immutable_insert(
                        connection,
                        table="public.regulatory_legal_version_artifacts",
                        key_columns=("legal_version_id", "artifact_id", "source_role"),
                        values={
                            "legal_version_id": version.legal_version_id,
                            "artifact_id": artifact_id,
                            "source_role": "legal_text",
                        },
                        object_kind="version artifact",
                    )
            for provision in sorted(
                bundle.provisions, key=lambda item: item.provision_id
            ):
                await _immutable_insert(
                    connection,
                    table="public.regulatory_provisions",
                    key_columns=("provision_id",),
                    values={
                        "provision_id": provision.provision_id,
                        "instrument_id": provision.instrument_id,
                        "provision_kind": provision.kind,
                        "canonical_path": provision.canonical_path,
                    },
                    object_kind="provision",
                )
            for version in sorted(
                bundle.versions, key=lambda item: item.legal_version_id
            ):
                for event in sorted(_events(version), key=lambda item: item.event_id):
                    await _immutable_insert(
                        connection,
                        table="public.regulatory_legal_events",
                        key_columns=("event_id",),
                        values={
                            "event_id": event.event_id,
                            "legal_version_id": event.legal_version_id,
                            "event_type": event.event_type.value,
                            "event_date": event.event_date,
                            "evidence_id": event.evidence.evidence_id,
                            "target_legal_version_id": event.target_legal_version_id,
                            **_validation_values(event.validation),
                        },
                        object_kind="legal event",
                    )
                for assertion in sorted(
                    version.status_assertions, key=lambda item: item.assertion_id
                ):
                    await _immutable_insert(
                        connection,
                        table="public.regulatory_legal_status_assertions",
                        key_columns=("assertion_id",),
                        values={
                            "assertion_id": assertion.assertion_id,
                            "legal_version_id": assertion.legal_version_id,
                            "legal_status": assertion.status.value,
                            "valid_from": assertion.valid_from,
                            "valid_through": assertion.valid_through,
                            "evidence_id": assertion.evidence.evidence_id,
                            **_validation_values(assertion.validation),
                        },
                        object_kind="legal-status assertion",
                    )
                for occurrence in sorted(
                    version.provisions, key=lambda item: item.provision_id
                ):
                    await _immutable_insert(
                        connection,
                        table="public.regulatory_legal_version_provisions",
                        key_columns=("legal_version_id", "provision_id"),
                        values={
                            "legal_version_id": occurrence.legal_version_id,
                            "provision_id": occurrence.provision_id,
                            "normalized_text_sha256": occurrence.normalized_text_sha256,
                            "evidence_id": occurrence.evidence.evidence_id,
                        },
                        object_kind="version provision",
                    )
            await _immutable_insert(
                connection,
                table="public.regulatory_family_imports",
                key_columns=("bundle_id", "bundle_sha256"),
                values={
                    "bundle_id": bundle.bundle_id,
                    "bundle_sha256": bundle.bundle_sha256,
                    "instrument_id": bundle.instrument.instrument_id,
                    "schema_version": bundle.schema_version,
                    "fixture_only": bundle.fixture_only,
                    "imported_by": imported_by,
                },
                compare_columns=("instrument_id", "schema_version", "fixture_only"),
                object_kind="family import",
            )
    except LegalVersionPersistenceError:
        raise
    except (asyncpg.PostgresError, OSError, TypeError, ValueError):
        raise LegalVersionPersistenceError(
            "Legal-version persistence failed and was rolled back; inspect database readiness and role grants."
        ) from None

    return LegalVersionImportResult(
        bundle_id=bundle.bundle_id,
        bundle_sha256=bundle.bundle_sha256,
        instrument_id=bundle.instrument.instrument_id,
        artifact_count=len(bundle.artifacts),
        version_count=len(bundle.versions),
        provision_count=len(bundle.provisions),
        fixture_only=bundle.fixture_only,
    )
