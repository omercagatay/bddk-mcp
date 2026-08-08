"""Transactional, immutable persistence for canonical legal-version bundles."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

import asyncpg

from bddk_mcp.corpus_coordination import acquire_corpus_mutation_lock
from bddk_mcp.regulatory.legal_versions import (
    LegalEvent,
    LegalVersion,
    LegalVersionBundle,
    ValidationRecord,
    ValidationState,
    canonical_bundle_sha256,
)
from bddk_mcp.regulatory.text_profile import POSTGRES_PROVISION_BOUNDARY_WHITESPACE_V1

_IMPORTER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,199}$")
_SQL_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LOCK_TIMEOUT = "5s"
_STATEMENT_TIMEOUT = "30s"
_MAX_FUTURE_REVIEW_SKEW = timedelta(minutes=5)
_VALIDATION_COLUMNS = (
    "validation_state",
    "validated_by",
    "validated_at",
    "validation_method",
    "review_record_sha256",
)
_COLUMN_TYPES: dict[str, dict[str, str]] = {
    "public.regulatory_instruments": {
        "instrument_id": "pg_catalog.text",
        "jurisdiction": "pg_catalog.text",
        "authority_code": "pg_catalog.text",
        "identity_key": "pg_catalog.text",
        "canonical_title": "pg_catalog.text",
        "instrument_type": "pg_catalog.text",
    },
    "public.regulatory_family_imports": {
        "bundle_id": "pg_catalog.text",
        "bundle_sha256": "pg_catalog.text",
        "instrument_id": "pg_catalog.text",
        "schema_version": "pg_catalog.int4",
        "fixture_only": "pg_catalog.bool",
        "imported_by": "pg_catalog.text",
        "predecessor_bundle_sha256": "pg_catalog.text",
        "member_manifest": "pg_catalog.jsonb",
    },
    "public.regulatory_source_blobs": {
        "blob_id": "pg_catalog.text",
        "content_sha256": "pg_catalog.text",
    },
    "public.regulatory_source_artifacts": {
        "artifact_id": "pg_catalog.text",
        "blob_id": "pg_catalog.text",
        "canonical_uri": "pg_catalog.text",
        "source_authority": "pg_catalog.text",
        "media_type": "pg_catalog.text",
        "retrieved_at": "pg_catalog.timestamptz",
        "repository_document_id": "pg_catalog.text",
        "fixture_only": "pg_catalog.bool",
    },
    "public.regulatory_evidence": {
        "evidence_id": "pg_catalog.text",
        "artifact_id": "pg_catalog.text",
        "locator": "pg_catalog.text",
        "statement_sha256": "pg_catalog.text",
        "authority_level": "pg_catalog.text",
    },
    "public.regulatory_legal_versions": {
        "legal_version_id": "pg_catalog.text",
        "instrument_id": "pg_catalog.text",
        "version_key": "pg_catalog.text",
        "legal_text_sha256": "pg_catalog.text",
        "predecessor_version_id": "pg_catalog.text",
        "consolidation_state": "pg_catalog.text",
        "validation_state": "pg_catalog.text",
        "validated_by": "pg_catalog.text",
        "validated_at": "pg_catalog.timestamptz",
        "validation_method": "pg_catalog.text",
        "review_record_sha256": "pg_catalog.text",
    },
    "public.regulatory_legal_version_artifacts": {
        "legal_version_id": "pg_catalog.text",
        "artifact_id": "pg_catalog.text",
        "source_role": "pg_catalog.text",
    },
    "public.regulatory_legal_events": {
        "event_id": "pg_catalog.text",
        "legal_version_id": "pg_catalog.text",
        "event_type": "pg_catalog.text",
        "event_date": "pg_catalog.date",
        "evidence_id": "pg_catalog.text",
        "target_legal_version_id": "pg_catalog.text",
        "validation_state": "pg_catalog.text",
        "validated_by": "pg_catalog.text",
        "validated_at": "pg_catalog.timestamptz",
        "validation_method": "pg_catalog.text",
        "review_record_sha256": "pg_catalog.text",
    },
    "public.regulatory_legal_status_assertions": {
        "assertion_id": "pg_catalog.text",
        "legal_version_id": "pg_catalog.text",
        "legal_status": "pg_catalog.text",
        "valid_from": "pg_catalog.date",
        "valid_through": "pg_catalog.date",
        "evidence_id": "pg_catalog.text",
        "validation_state": "pg_catalog.text",
        "validated_by": "pg_catalog.text",
        "validated_at": "pg_catalog.timestamptz",
        "validation_method": "pg_catalog.text",
        "review_record_sha256": "pg_catalog.text",
    },
    "public.regulatory_provisions": {
        "provision_id": "pg_catalog.text",
        "instrument_id": "pg_catalog.text",
        "provision_kind": "pg_catalog.text",
        "canonical_path": "pg_catalog.text",
    },
    "public.regulatory_legal_version_provisions": {
        "legal_version_id": "pg_catalog.text",
        "provision_id": "pg_catalog.text",
        "provision_text_sha256": "pg_catalog.text",
        "document_section_id": "pg_catalog.int4",
        "evidence_id": "pg_catalog.text",
        "validation_state": "pg_catalog.text",
        "validated_by": "pg_catalog.text",
        "validated_at": "pg_catalog.timestamptz",
        "validation_method": "pg_catalog.text",
        "review_record_sha256": "pg_catalog.text",
    },
    "public.regulatory_relations": {
        "relation_id": "pg_catalog.text",
        "relation_type": "pg_catalog.text",
        "source_instrument_id": "pg_catalog.text",
        "source_provision_id": "pg_catalog.text",
        "target_instrument_id": "pg_catalog.text",
        "target_provision_id": "pg_catalog.text",
        "target_external_ref": "pg_catalog.text",
        "evidence_id": "pg_catalog.text",
        "extraction_method": "pg_catalog.text",
        "confidence": "pg_catalog.float4",
        "validation_state": "pg_catalog.text",
        "validated_by": "pg_catalog.text",
        "validated_at": "pg_catalog.timestamptz",
        "validation_method": "pg_catalog.text",
        "review_record_sha256": "pg_catalog.text",
    },
}


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
    blob_count: int
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
                if version.predecessor_version_id is None or version.predecessor_version_id in emitted
            ),
            key=lambda version: version.legal_version_id,
        )
        if not ready:
            raise LegalVersionPersistenceError("Legal-version predecessor chain is cyclic; import refused.")
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


def _review_records(bundle: LegalVersionBundle) -> tuple[ValidationRecord, ...]:
    reviews: list[ValidationRecord] = []
    for version in bundle.versions:
        reviews.append(version.validation)
        reviews.extend(event.validation for event in _events(version))
        reviews.extend(assertion.validation for assertion in version.status_assertions)
        reviews.extend(occurrence.validation for occurrence in version.provisions)
    return tuple(reviews)


def _qualified_table(table: str) -> str:
    parts = table.split(".")
    if len(parts) != 2 or any(not _SQL_IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise RuntimeError("repository table identifier is invalid")
    return table


async def _immutable_insert_many(
    connection: Any,
    *,
    table: str,
    key_columns: tuple[str, ...],
    records: Sequence[dict[str, Any]],
    object_kind: str,
    compare_columns: tuple[str, ...] | None = None,
) -> None:
    """Set-wise immutable insert with exact conflict/equality accounting."""

    if not records:
        return
    table = _qualified_table(table)
    type_map = _COLUMN_TYPES.get(table)
    if type_map is None:
        raise RuntimeError("repository table type contract is missing")
    columns = tuple(records[0])
    if (
        not columns
        or not key_columns
        or any(tuple(record) != columns for record in records)
        or any(column not in type_map for column in columns)
        or any(column not in columns for column in key_columns)
    ):
        raise RuntimeError("repository bulk record shape is invalid")
    compared = compare_columns or tuple(column for column in columns if column not in key_columns)
    if any(column not in columns for column in compared):
        raise RuntimeError("repository comparison columns are invalid")
    keys = [tuple(record[column] for column in key_columns) for record in records]
    if len(keys) != len(set(keys)):
        raise LegalVersionPersistenceError(f"Duplicate {object_kind} identity in bundle; import rolled back.")

    unnests = ", ".join(
        f"pg_catalog.unnest(${position}::{type_map[column]}[])" for position, column in enumerate(columns, start=1)
    )
    first_key = key_columns[0]
    if compared:
        existing = ", ".join(f"existing.{column}" for column in compared)
        excluded = ", ".join(f"EXCLUDED.{column}" for column in compared)
        condition = f"ROW({existing}) IS NOT DISTINCT FROM ROW({excluded})"
    else:
        condition = "true"
    query = f"""
        WITH persisted AS (
            INSERT INTO {table} AS existing ({", ".join(columns)})
            SELECT {", ".join(f"incoming.{column}" for column in columns)}
            FROM ROWS FROM ({unnests}) AS incoming ({", ".join(columns)})
            ON CONFLICT ({", ".join(key_columns)}) DO UPDATE
            SET {first_key} = EXCLUDED.{first_key}
            WHERE {condition}
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM persisted
    """
    arrays = [[record[column] for record in records] for column in columns]
    persisted = await connection.fetchval(query, *arrays)
    if persisted != len(records):
        raise LegalVersionPersistenceError(
            f"Existing {object_kind} identity has different immutable fields; import rolled back."
        )


async def _apply_validation_transitions(
    connection: Any,
    *,
    table: str,
    subjects: Sequence[tuple[dict[str, str], ValidationRecord]],
    object_kind: str,
) -> None:
    """Validate and apply one entity type's monotonic review transitions set-wise."""

    if not subjects:
        return
    table = _qualified_table(table)
    type_map = _COLUMN_TYPES.get(table)
    key_columns = tuple(subjects[0][0])
    if (
        type_map is None
        or not key_columns
        or any(tuple(keys) != key_columns for keys, _ in subjects)
        or any(column not in type_map for column in (*key_columns, *_VALIDATION_COLUMNS))
    ):
        raise RuntimeError("repository validation batch shape is invalid")
    subject_keys = [tuple(keys[column] for column in key_columns) for keys, _ in subjects]
    if len(subject_keys) != len(set(subject_keys)):
        raise LegalVersionPersistenceError(f"Duplicate {object_kind} review in bundle; import rolled back.")

    key_unnests = ", ".join(
        f"pg_catalog.unnest(${position}::{type_map[column]}[])" for position, column in enumerate(key_columns, start=1)
    )
    join = " AND ".join(f"existing.{column} IS NOT DISTINCT FROM incoming.{column}" for column in key_columns)
    rows = await connection.fetch(
        f"""
        SELECT {", ".join(f"existing.{column}" for column in (*key_columns, *_VALIDATION_COLUMNS))}
        FROM {table} AS existing
        JOIN ROWS FROM ({key_unnests}) AS incoming ({", ".join(key_columns)})
          ON {join}
        FOR UPDATE OF existing
        """,
        *[[keys[column] for keys, _ in subjects] for column in key_columns],
    )
    current_by_key = {tuple(row[column] for column in key_columns): row for row in rows}
    if set(current_by_key) != set(subject_keys):
        raise LegalVersionPersistenceError(f"Persisted {object_kind} disappeared; import rolled back.")

    updates: list[dict[str, Any]] = []
    for (key_values, validation), key in zip(subjects, subject_keys, strict=True):
        row = current_by_key[key]
        current_state = ValidationState(str(row["validation_state"]))
        current_values = tuple(row[column] for column in _VALIDATION_COLUMNS[1:])
        incoming_values = _validation_values(validation)
        incoming_review = tuple(incoming_values[column] for column in _VALIDATION_COLUMNS[1:])
        if current_state is validation.state:
            if current_values != incoming_review:
                raise LegalVersionPersistenceError(
                    f"Existing {object_kind} review has different provenance; import rolled back."
                )
            continue
        allowed = current_state is ValidationState.UNVALIDATED and validation.state in {
            ValidationState.IN_REVIEW,
            ValidationState.VALIDATED,
            ValidationState.REJECTED,
        }
        allowed = allowed or (
            current_state is ValidationState.IN_REVIEW
            and validation.state in {ValidationState.VALIDATED, ValidationState.REJECTED}
        )
        if not allowed:
            raise LegalVersionPersistenceError(
                f"Existing {object_kind} review is terminal or cannot move backward; import rolled back."
            )
        updates.append({**key_values, **incoming_values})

    if not updates:
        return
    update_columns = (*key_columns, *_VALIDATION_COLUMNS)
    unnests = ", ".join(
        f"pg_catalog.unnest(${position}::{type_map[column]}[])"
        for position, column in enumerate(update_columns, start=1)
    )
    assignments = ", ".join(f"{column} = incoming.{column}" for column in _VALIDATION_COLUMNS)
    count = await connection.fetchval(
        f"""
        WITH updated AS (
            UPDATE {table} AS existing
            SET {assignments}
            FROM ROWS FROM ({unnests}) AS incoming ({", ".join(update_columns)})
            WHERE {join}
            RETURNING 1
        )
        SELECT pg_catalog.count(*)::pg_catalog.int4 FROM updated
        """,
        *[[record[column] for record in updates] for column in update_columns],
    )
    if count != len(updates):
        raise LegalVersionPersistenceError(f"Persisted {object_kind} disappeared; import rolled back.")


def _member_manifest(bundle: LegalVersionBundle) -> str:
    """Return privacy-safe exact membership and review-state projection."""

    def reviewed(subject_id: str, validation: ValidationRecord) -> dict[str, str | None]:
        return {
            "id": subject_id,
            "state": validation.state.value,
            "review_record_sha256": validation.review_record_sha256,
        }

    payload = {
        "schema_version": 1,
        "blob_ids": [blob.blob_id for blob in bundle.blobs],
        "artifact_ids": [artifact.artifact_id for artifact in bundle.artifacts],
        "evidence_ids": [evidence.evidence_id for evidence in _evidence(bundle)],
        "provision_ids": [provision.provision_id for provision in bundle.provisions],
        "version_artifacts": [
            f"{version.legal_version_id}:{artifact_id}:legal_text"
            for version in bundle.versions
            for artifact_id in version.source_artifact_ids
        ],
        "versions": [reviewed(version.legal_version_id, version.validation) for version in bundle.versions],
        "events": [
            reviewed(event.event_id, event.validation)
            for version in bundle.versions
            for event in sorted(_events(version), key=lambda item: item.event_id)
        ],
        "status_assertions": [
            reviewed(assertion.assertion_id, assertion.validation)
            for version in bundle.versions
            for assertion in version.status_assertions
        ],
        "provision_occurrences": [
            reviewed(f"{occurrence.legal_version_id}:{occurrence.provision_id}", occurrence.validation)
            for version in bundle.versions
            for occurrence in version.provisions
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def _assert_section_mapping(connection: Any, occurrence: Any) -> None:
    if occurrence.document_section_id is None:
        return
    valid = await connection.fetchval(
        f"""
        SELECT true
        FROM public.document_sections AS section
        JOIN public.documents AS document
          ON document.document_id = section.doc_id
         AND document.content_hash = section.source_content_hash
         AND document.content_hash = pg_catalog.encode(
             pg_catalog.sha256(pg_catalog.convert_to(document.markdown_content, 'UTF8')),
             'hex'
         )
         AND section.content = pg_catalog.btrim(
             pg_catalog.substr(
                 document.markdown_content,
                 section.start_char + 1,
                 section.end_char - section.start_char
             ),
             {POSTGRES_PROVISION_BOUNDARY_WHITESPACE_V1}
         )
        JOIN public.regulatory_legal_versions AS version
          ON version.legal_version_id = $2
         AND version.legal_text_sha256 = document.content_hash
        JOIN public.regulatory_provisions AS provision
          ON provision.provision_id = $3
         AND provision.instrument_id = version.instrument_id
        JOIN public.regulatory_evidence AS evidence
          ON evidence.evidence_id = $4
         AND evidence.statement_sha256 = $5
        JOIN public.regulatory_source_artifacts AS artifact
          ON artifact.artifact_id = evidence.artifact_id
         AND artifact.repository_document_id = section.doc_id
        JOIN public.regulatory_legal_version_artifacts AS version_artifact
          ON version_artifact.legal_version_id = version.legal_version_id
         AND version_artifact.artifact_id = artifact.artifact_id
         AND version_artifact.source_role = 'legal_text'
        WHERE section.id = $1
          AND section.content_hash = $5
          AND section.content_hash = pg_catalog.encode(
              pg_catalog.sha256(pg_catalog.convert_to(section.content, 'UTF8')),
              'hex'
          )
        """,
        occurrence.document_section_id,
        occurrence.legal_version_id,
        occurrence.provision_id,
        occurrence.evidence.evidence_id,
        occurrence.provision_text_sha256,
    )
    if valid is not True:
        raise LegalVersionPersistenceError(
            "Provision-section mapping does not match its document, version, artifact, evidence, and hashes; "
            "import rolled back."
        )


# Import implementation is split into small, ordered helpers so every table is
# covered by one transaction and one instrument-scoped advisory lock.


async def _persist_identity_layers(connection: Any, bundle: LegalVersionBundle) -> None:
    instrument = bundle.instrument
    await _immutable_insert_many(
        connection,
        table="public.regulatory_instruments",
        key_columns=("instrument_id",),
        records=[
            {
                "instrument_id": instrument.instrument_id,
                "jurisdiction": instrument.jurisdiction,
                "authority_code": instrument.authority_code,
                "identity_key": instrument.identity_key,
                "canonical_title": instrument.canonical_title,
                "instrument_type": instrument.instrument_type,
            }
        ],
        object_kind="instrument",
    )
    await _immutable_insert_many(
        connection,
        table="public.regulatory_source_blobs",
        key_columns=("blob_id",),
        records=[
            {
                "blob_id": blob.blob_id,
                "content_sha256": blob.content_sha256,
            }
            for blob in sorted(bundle.blobs, key=lambda item: item.blob_id)
        ],
        object_kind="source blob",
    )
    await _immutable_insert_many(
        connection,
        table="public.regulatory_source_artifacts",
        key_columns=("artifact_id",),
        records=[
            {
                "artifact_id": artifact.artifact_id,
                "blob_id": artifact.blob_id,
                "canonical_uri": artifact.canonical_uri,
                "source_authority": artifact.source_authority,
                "media_type": artifact.media_type,
                "retrieved_at": artifact.retrieved_at,
                "repository_document_id": artifact.repository_document_id,
                "fixture_only": artifact.fixture_only,
            }
            for artifact in sorted(bundle.artifacts, key=lambda item: item.artifact_id)
        ],
        object_kind="source artifact",
    )
    await _immutable_insert_many(
        connection,
        table="public.regulatory_evidence",
        key_columns=("evidence_id",),
        records=[
            {
                "evidence_id": evidence.evidence_id,
                "artifact_id": evidence.artifact_id,
                "locator": evidence.locator,
                "statement_sha256": evidence.statement_sha256,
                "authority_level": evidence.authority_level.value,
            }
            for evidence in _evidence(bundle)
        ],
        object_kind="evidence",
    )
    versions = _ordered_versions(bundle)
    await _immutable_insert_many(
        connection,
        table="public.regulatory_legal_versions",
        key_columns=("legal_version_id",),
        records=[
            {
                "legal_version_id": version.legal_version_id,
                "instrument_id": version.instrument_id,
                "version_key": version.version_key,
                "legal_text_sha256": version.legal_text_sha256,
                "predecessor_version_id": version.predecessor_version_id,
                "consolidation_state": version.consolidation_state.value,
                **_validation_values(version.validation),
            }
            for version in versions
        ],
        compare_columns=(
            "instrument_id",
            "version_key",
            "legal_text_sha256",
            "predecessor_version_id",
            "consolidation_state",
        ),
        object_kind="legal version",
    )
    await _apply_validation_transitions(
        connection,
        table="public.regulatory_legal_versions",
        subjects=[({"legal_version_id": version.legal_version_id}, version.validation) for version in versions],
        object_kind="legal version",
    )
    await _immutable_insert_many(
        connection,
        table="public.regulatory_legal_version_artifacts",
        key_columns=("legal_version_id", "artifact_id", "source_role"),
        records=[
            {
                "legal_version_id": version.legal_version_id,
                "artifact_id": artifact_id,
                "source_role": "legal_text",
            }
            for version in sorted(bundle.versions, key=lambda item: item.legal_version_id)
            for artifact_id in sorted(version.source_artifact_ids)
        ],
        object_kind="version artifact",
    )
    await _immutable_insert_many(
        connection,
        table="public.regulatory_provisions",
        key_columns=("provision_id",),
        records=[
            {
                "provision_id": provision.provision_id,
                "instrument_id": provision.instrument_id,
                "provision_kind": provision.kind,
                "canonical_path": provision.canonical_path,
            }
            for provision in sorted(bundle.provisions, key=lambda item: item.provision_id)
        ],
        object_kind="provision",
    )


async def _persist_version_claims(connection: Any, bundle: LegalVersionBundle) -> None:
    versions = sorted(bundle.versions, key=lambda item: item.legal_version_id)
    events = sorted((event for version in versions for event in _events(version)), key=lambda item: item.event_id)
    assertions = sorted(
        (assertion for version in versions for assertion in version.status_assertions),
        key=lambda item: item.assertion_id,
    )
    occurrences = sorted(
        (occurrence for version in versions for occurrence in version.provisions),
        key=lambda item: (item.legal_version_id, item.provision_id),
    )

    await _immutable_insert_many(
        connection,
        table="public.regulatory_legal_events",
        key_columns=("event_id",),
        records=[
            {
                "event_id": event.event_id,
                "legal_version_id": event.legal_version_id,
                "event_type": event.event_type.value,
                "event_date": event.event_date,
                "evidence_id": event.evidence.evidence_id,
                "target_legal_version_id": event.target_legal_version_id,
                **_validation_values(event.validation),
            }
            for event in events
        ],
        compare_columns=(
            "legal_version_id",
            "event_type",
            "event_date",
            "evidence_id",
            "target_legal_version_id",
        ),
        object_kind="legal event",
    )
    await _apply_validation_transitions(
        connection,
        table="public.regulatory_legal_events",
        subjects=[({"event_id": event.event_id}, event.validation) for event in events],
        object_kind="legal event",
    )
    await _immutable_insert_many(
        connection,
        table="public.regulatory_legal_status_assertions",
        key_columns=("assertion_id",),
        records=[
            {
                "assertion_id": assertion.assertion_id,
                "legal_version_id": assertion.legal_version_id,
                "legal_status": assertion.status.value,
                "valid_from": assertion.valid_from,
                "valid_through": assertion.valid_through,
                "evidence_id": assertion.evidence.evidence_id,
                **_validation_values(assertion.validation),
            }
            for assertion in assertions
        ],
        compare_columns=(
            "legal_version_id",
            "legal_status",
            "valid_from",
            "valid_through",
            "evidence_id",
        ),
        object_kind="legal-status assertion",
    )
    await _apply_validation_transitions(
        connection,
        table="public.regulatory_legal_status_assertions",
        subjects=[({"assertion_id": assertion.assertion_id}, assertion.validation) for assertion in assertions],
        object_kind="legal-status assertion",
    )
    for occurrence in occurrences:
        await _assert_section_mapping(connection, occurrence)
    await _immutable_insert_many(
        connection,
        table="public.regulatory_legal_version_provisions",
        key_columns=("legal_version_id", "provision_id"),
        records=[
            {
                "legal_version_id": occurrence.legal_version_id,
                "provision_id": occurrence.provision_id,
                "provision_text_sha256": occurrence.provision_text_sha256,
                "document_section_id": occurrence.document_section_id,
                "evidence_id": occurrence.evidence.evidence_id,
                **_validation_values(occurrence.validation),
            }
            for occurrence in occurrences
        ],
        compare_columns=("provision_text_sha256", "document_section_id", "evidence_id"),
        object_kind="version provision",
    )
    await _apply_validation_transitions(
        connection,
        table="public.regulatory_legal_version_provisions",
        subjects=[
            (
                {
                    "legal_version_id": occurrence.legal_version_id,
                    "provision_id": occurrence.provision_id,
                },
                occurrence.validation,
            )
            for occurrence in occurrences
        ],
        object_kind="version provision",
    )


# public entry point follows


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
        raise LegalVersionPersistenceError("Fixture-only legal-version data cannot be imported here.")
    if canonical_bundle_sha256(bundle) != bundle.bundle_sha256:
        raise LegalVersionPersistenceError("Legal-version bundle checksum does not match; import refused.")

    try:
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'")
            await connection.execute(f"SET LOCAL statement_timeout = '{_STATEMENT_TIMEOUT}'")
            database_now = await connection.fetchval("SELECT CURRENT_TIMESTAMP")
            if database_now is None or any(
                review.validated_at is not None and review.validated_at > database_now + _MAX_FUTURE_REVIEW_SKEW
                for review in _review_records(bundle)
            ):
                raise LegalVersionPersistenceError(
                    "Legal-version review time is beyond the database clock allowance; import rolled back."
                )
            await acquire_corpus_mutation_lock(connection)
            await connection.fetchval(
                "SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended($1::pg_catalog.text, 1280066885))",
                bundle.instrument.instrument_id,
            )
            await _persist_identity_layers(connection, bundle)
            await _persist_version_claims(connection, bundle)
            predecessor_bundle_sha256 = await connection.fetchval(
                """
                SELECT bundle_sha256
                FROM public.regulatory_family_imports
                WHERE bundle_id = $1
                  AND bundle_sha256 <> $2
                ORDER BY imported_at DESC, bundle_sha256 DESC
                LIMIT 1
                """,
                bundle.bundle_id,
                bundle.bundle_sha256,
            )
            await _immutable_insert_many(
                connection,
                table="public.regulatory_family_imports",
                key_columns=("bundle_id", "bundle_sha256"),
                records=[
                    {
                        "bundle_id": bundle.bundle_id,
                        "bundle_sha256": bundle.bundle_sha256,
                        "instrument_id": bundle.instrument.instrument_id,
                        "schema_version": bundle.schema_version,
                        "fixture_only": bundle.fixture_only,
                        "imported_by": imported_by,
                        "predecessor_bundle_sha256": predecessor_bundle_sha256,
                        "member_manifest": _member_manifest(bundle),
                    }
                ],
                compare_columns=(
                    "instrument_id",
                    "schema_version",
                    "fixture_only",
                    "predecessor_bundle_sha256",
                    "member_manifest",
                ),
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
        blob_count=len(bundle.blobs),
        artifact_count=len(bundle.artifacts),
        version_count=len(bundle.versions),
        provision_count=len(bundle.provisions),
        fixture_only=bundle.fixture_only,
    )
