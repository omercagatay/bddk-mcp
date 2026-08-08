"""Typed, evidence-backed cross-reference edges between instruments/provisions.

Edges are claims, not facts: every row carries an evidence reference and a
review record, and only reviewer-validated edges surface through the
``regulatory_validated_relations`` view that serving workloads read.  An
extracted or imported edge is never served merely because it exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bddk_mcp.corpus_coordination import acquire_corpus_mutation_lock
from bddk_mcp.regulatory.legal_versions import (
    EvidenceReference,
    ValidationRecord,
)
from bddk_mcp.regulatory.repository import (
    _IMPORTER_RE,
    _LOCK_TIMEOUT,
    _MAX_FUTURE_REVIEW_SKEW,
    _STATEMENT_TIMEOUT,
    LegalVersionPersistenceError,
    _immutable_insert_many,
    _validation_values,
)

_RELATION_ID_PATTERN = r"^rel_sha256_[0-9a-f]{64}$"
_INSTRUMENT_ID_PATTERN = r"^inst_sha256_[0-9a-f]{64}$"
_PROVISION_ID_PATTERN = r"^prov_sha256_[0-9a-f]{64}$"

RELATION_TYPES = frozenset(
    {"amends", "repeals", "replaces", "consolidates", "implements", "cites", "defines", "exception_to"}
)


def relation_id_for(
    *,
    relation_type: str,
    source_instrument_id: str,
    source_provision_id: str | None,
    target_instrument_id: str | None,
    target_provision_id: str | None,
    target_external_ref: str | None,
    evidence_id: str,
    extraction_method: str,
) -> str:
    """Build an edge identity from its immutable claim components."""

    payload = json.dumps(
        (
            relation_type,
            source_instrument_id,
            source_provision_id or "",
            target_instrument_id or "",
            target_provision_id or "",
            target_external_ref or "",
            evidence_id,
            extraction_method,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"rel_sha256_{hashlib.sha256(payload).hexdigest()}"


class RegulatoryRelation(BaseModel):
    """One directed edge; the target is an in-corpus instrument or an external ref."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str = Field(pattern=_RELATION_ID_PATTERN)
    relation_type: str
    source_instrument_id: str = Field(pattern=_INSTRUMENT_ID_PATTERN)
    source_provision_id: str | None = Field(default=None, pattern=_PROVISION_ID_PATTERN)
    target_instrument_id: str | None = Field(default=None, pattern=_INSTRUMENT_ID_PATTERN)
    target_provision_id: str | None = Field(default=None, pattern=_PROVISION_ID_PATTERN)
    target_external_ref: str | None = Field(default=None, min_length=1, max_length=500)
    evidence: EvidenceReference
    extraction_method: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)
    validation: ValidationRecord

    @model_validator(mode="after")
    def _claim_shape_is_coherent(self) -> RegulatoryRelation:
        if self.relation_type not in RELATION_TYPES:
            raise ValueError("relation_type is not a reviewed relation kind")
        if self.target_instrument_id is None and self.target_external_ref is None:
            raise ValueError("relation needs target_instrument_id or target_external_ref")
        if self.target_provision_id is not None and self.target_instrument_id is None:
            raise ValueError("a provision target requires its instrument target")
        expected = relation_id_for(
            relation_type=self.relation_type,
            source_instrument_id=self.source_instrument_id,
            source_provision_id=self.source_provision_id,
            target_instrument_id=self.target_instrument_id,
            target_provision_id=self.target_provision_id,
            target_external_ref=self.target_external_ref,
            evidence_id=self.evidence.evidence_id,
            extraction_method=self.extraction_method,
        )
        if self.relation_id != expected:
            raise ValueError("relation_id does not match its immutable claim components")
        return self


def make_relation(
    *,
    relation_type: str,
    source_instrument_id: str,
    evidence: EvidenceReference,
    extraction_method: str,
    confidence: float,
    validation: ValidationRecord,
    source_provision_id: str | None = None,
    target_instrument_id: str | None = None,
    target_provision_id: str | None = None,
    target_external_ref: str | None = None,
) -> RegulatoryRelation:
    """Construct an edge with its derived identity filled in."""

    return RegulatoryRelation(
        relation_id=relation_id_for(
            relation_type=relation_type,
            source_instrument_id=source_instrument_id,
            source_provision_id=source_provision_id,
            target_instrument_id=target_instrument_id,
            target_provision_id=target_provision_id,
            target_external_ref=target_external_ref,
            evidence_id=evidence.evidence_id,
            extraction_method=extraction_method,
        ),
        relation_type=relation_type,
        source_instrument_id=source_instrument_id,
        source_provision_id=source_provision_id,
        target_instrument_id=target_instrument_id,
        target_provision_id=target_provision_id,
        target_external_ref=target_external_ref,
        evidence=evidence,
        extraction_method=extraction_method,
        confidence=confidence,
        validation=validation,
    )


async def import_relations(
    pool: Any,
    relations: Sequence[RegulatoryRelation],
    *,
    imported_by: str,
) -> int:
    """Atomically persist edges and their evidence; immutable-or-identical.

    Re-importing an identical edge is a no-op; an existing relation_id with
    different immutable fields aborts the whole transaction.  Referenced
    artifacts must already exist (edges never create acquisition identities).
    """

    if not _IMPORTER_RE.fullmatch(imported_by):
        raise LegalVersionPersistenceError("imported_by is invalid; import refused.")
    if not relations:
        return 0
    ordered = sorted(relations, key=lambda relation: relation.relation_id)
    evidence_by_id = {relation.evidence.evidence_id: relation.evidence for relation in ordered}
    try:
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'")
            await connection.execute(f"SET LOCAL statement_timeout = '{_STATEMENT_TIMEOUT}'")
            database_now = await connection.fetchval("SELECT CURRENT_TIMESTAMP")
            if database_now is None or any(
                relation.validation.validated_at is not None
                and relation.validation.validated_at > database_now + _MAX_FUTURE_REVIEW_SKEW
                for relation in ordered
            ):
                raise LegalVersionPersistenceError(
                    "Relation review time is beyond the database clock allowance; import rolled back."
                )
            await acquire_corpus_mutation_lock(connection)
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
                    for evidence in sorted(evidence_by_id.values(), key=lambda item: item.evidence_id)
                ],
                object_kind="evidence",
            )
            await _immutable_insert_many(
                connection,
                table="public.regulatory_relations",
                key_columns=("relation_id",),
                records=[
                    {
                        "relation_id": relation.relation_id,
                        "relation_type": relation.relation_type,
                        "source_instrument_id": relation.source_instrument_id,
                        "source_provision_id": relation.source_provision_id,
                        "target_instrument_id": relation.target_instrument_id,
                        "target_provision_id": relation.target_provision_id,
                        "target_external_ref": relation.target_external_ref,
                        "evidence_id": relation.evidence.evidence_id,
                        "extraction_method": relation.extraction_method,
                        "confidence": relation.confidence,
                        **_validation_values(relation.validation),
                    }
                    for relation in ordered
                ],
                object_kind="regulatory relation",
            )
    except LegalVersionPersistenceError:
        raise
    except (asyncpg.PostgresError, OSError, TypeError, ValueError):
        raise LegalVersionPersistenceError(
            "Relation persistence failed and was rolled back; inspect database readiness and role grants."
        ) from None
    return len(ordered)
