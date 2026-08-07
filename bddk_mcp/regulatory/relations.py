"""Typed, evidence-backed cross-reference edges between instruments/provisions."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg

from bddk_mcp.regulatory.legal_versions import Evidence, ValidationRecord
from bddk_mcp.regulatory.repository import (
    _IMPORTER_RE,
    LegalVersionPersistenceError,
    _immutable_insert,
)

RELATION_TYPES = frozenset(
    {"amends", "repeals", "replaces", "consolidates", "implements", "cites", "defines", "exception_to"}
)


@dataclass(frozen=True, slots=True)
class RegulatoryRelation:
    """One directed edge; target is either an in-corpus instrument or an external ref."""

    relation_type: str
    source_instrument_id: str
    source_provision_id: str | None
    target_instrument_id: str | None
    target_provision_id: str | None
    target_external_ref: str | None
    evidence: Evidence
    extraction_method: str
    confidence: float
    validation: ValidationRecord

    def __post_init__(self) -> None:
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(f"unknown relation_type: {self.relation_type!r}")
        if self.target_instrument_id is None and self.target_external_ref is None:
            raise ValueError("relation needs target_instrument_id or target_external_ref")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


def relation_id(relation: RegulatoryRelation) -> str:
    """Deterministic identity: same statement from the same evidence → same id."""
    key = "|".join(
        part or ""
        for part in (
            relation.relation_type,
            relation.source_instrument_id,
            relation.source_provision_id,
            relation.target_instrument_id,
            relation.target_provision_id,
            relation.target_external_ref,
            relation.evidence.evidence_id,
            relation.extraction_method,
        )
    )
    return "rel-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


async def import_relations(
    pool: Any,
    relations: Sequence[RegulatoryRelation],
    *,
    imported_by: str,
) -> int:
    """Atomically persist edges and their evidence; immutable-or-identical."""
    if not _IMPORTER_RE.fullmatch(imported_by):
        raise LegalVersionPersistenceError("imported_by is invalid; import refused.")
    if not relations:
        return 0
    try:
        async with pool.acquire() as connection, connection.transaction():
            for relation in sorted(relations, key=relation_id):
                evidence = relation.evidence
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
                await _immutable_insert(
                    connection,
                    table="public.regulatory_relations",
                    key_columns=("relation_id",),
                    values={
                        "relation_id": relation_id(relation),
                        "relation_type": relation.relation_type,
                        "source_instrument_id": relation.source_instrument_id,
                        "source_provision_id": relation.source_provision_id,
                        "target_instrument_id": relation.target_instrument_id,
                        "target_provision_id": relation.target_provision_id,
                        "target_external_ref": relation.target_external_ref,
                        "evidence_id": evidence.evidence_id,
                        "extraction_method": relation.extraction_method,
                        "confidence": relation.confidence,
                        "validation_state": relation.validation.state.value,
                        "validated_by": relation.validation.validated_by,
                        "validated_at": relation.validation.validated_at,
                        "validation_method": relation.validation.method,
                        "review_record_sha256": relation.validation.review_record_sha256,
                    },
                    object_kind="regulatory relation",
                )
    except LegalVersionPersistenceError:
        raise
    except (asyncpg.PostgresError, OSError, TypeError, ValueError):
        raise LegalVersionPersistenceError(
            "Relation persistence failed and was rolled back; inspect database readiness and role grants."
        ) from None
    return len(relations)
