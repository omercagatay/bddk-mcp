"""Immutable edge writes into regulatory_relations and validated-view gating."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    EvidenceReference,
    ValidationRecord,
    ValidationState,
    evidence_id_for,
)
from bddk_mcp.regulatory.relations import (
    RELATION_TYPES,
    import_relations,
    make_relation,
    relation_id_for,
)
from bddk_mcp.regulatory.repository import (
    LegalVersionPersistenceError,
    import_legal_version_bundle,
)
from tests.test_legal_versions import _trusted_test_bundle

pytestmark = pytest.mark.asyncio

VALIDATED = ValidationRecord(
    state=ValidationState.VALIDATED,
    validated_by="reviewer@example.test",
    validated_at=datetime(2026, 1, 1, tzinfo=UTC),
    method="manual-review",
    review_record_sha256="a" * 64,
)
UNVALIDATED = ValidationRecord(state=ValidationState.UNVALIDATED)


def make_evidence(artifact_id: str, *, statement: str = "synthetic edge claim") -> EvidenceReference:
    statement_sha256 = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    locator = "page=1"
    return EvidenceReference(
        evidence_id=evidence_id_for(
            artifact_id=artifact_id,
            locator=locator,
            statement_sha256=statement_sha256,
            authority_level=AuthorityLevel.SECONDARY,
        ),
        artifact_id=artifact_id,
        locator=locator,
        statement_sha256=statement_sha256,
        authority_level=AuthorityLevel.SECONDARY,
    )


async def seed_trusted_family(pool):
    """Import the non-fixture synthetic family and return the bundle."""
    bundle = _trusted_test_bundle()
    await import_legal_version_bundle(pool, bundle, imported_by="test-suite")
    return bundle


def external_relation(bundle, *, statement: str = "synthetic edge claim", **overrides):
    keywords = {
        "relation_type": "cites",
        "source_instrument_id": bundle.instrument.instrument_id,
        "target_external_ref": "5411 sayılı Bankacılık Kanunu madde 93",
        "evidence": make_evidence(bundle.artifacts[0].artifact_id, statement=statement),
        "extraction_method": "regex:v1",
        "confidence": 0.9,
        "validation": UNVALIDATED,
    }
    keywords.update(overrides)
    return make_relation(**keywords)


def test_relation_types_match_spec():
    assert RELATION_TYPES == frozenset(
        {"amends", "repeals", "replaces", "consolidates", "implements", "cites", "defines", "exception_to"}
    )


def test_relation_id_is_deterministic():
    bundle = _trusted_test_bundle()
    first = external_relation(bundle)
    second = external_relation(bundle)
    assert first.relation_id == second.relation_id
    assert first.relation_id != external_relation(bundle, relation_type="implements").relation_id
    assert first.relation_id == relation_id_for(
        relation_type=first.relation_type,
        source_instrument_id=first.source_instrument_id,
        source_provision_id=None,
        target_instrument_id=None,
        target_provision_id=None,
        target_external_ref=first.target_external_ref,
        evidence_id=first.evidence.evidence_id,
        extraction_method=first.extraction_method,
    )


def test_relation_requires_some_target():
    bundle = _trusted_test_bundle()
    with pytest.raises(ValueError):
        external_relation(bundle, target_external_ref=None)


def test_relation_rejects_unknown_type():
    bundle = _trusted_test_bundle()
    with pytest.raises(ValueError):
        external_relation(bundle, relation_type="mentions")


def test_provision_target_requires_instrument_target():
    bundle = _trusted_test_bundle()
    with pytest.raises(ValueError):
        external_relation(
            bundle,
            target_provision_id=bundle.provisions[0].provision_id,
        )


async def test_import_and_idempotent_reimport(regulatory_pool):
    bundle = await seed_trusted_family(regulatory_pool)
    edge = external_relation(bundle)
    assert await import_relations(regulatory_pool, [edge], imported_by="test-suite") == 1
    assert await import_relations(regulatory_pool, [edge], imported_by="test-suite") == 1
    count = await regulatory_pool.fetchval("SELECT count(*) FROM public.regulatory_relations")
    assert count == 1


async def test_conflicting_reimport_aborts(regulatory_pool):
    bundle = await seed_trusted_family(regulatory_pool)
    edge = external_relation(bundle)
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    tampered = await regulatory_pool.fetchval(
        "UPDATE public.regulatory_relations SET confidence = 0.1 WHERE relation_id = $1 RETURNING relation_id",
        edge.relation_id,
    )
    assert tampered is not None
    with pytest.raises(LegalVersionPersistenceError):
        await import_relations(regulatory_pool, [edge], imported_by="test-suite")


async def test_invalid_importer_is_refused(regulatory_pool):
    bundle = await seed_trusted_family(regulatory_pool)
    with pytest.raises(LegalVersionPersistenceError):
        await import_relations(regulatory_pool, [external_relation(bundle)], imported_by="bad importer!")


async def test_only_validated_edges_are_visible_in_the_view(regulatory_pool):
    bundle = await seed_trusted_family(regulatory_pool)
    unvalidated_edge = external_relation(bundle, statement="unreviewed claim")
    validated_edge = external_relation(bundle, statement="reviewed claim", validation=VALIDATED)
    await import_relations(
        regulatory_pool, [unvalidated_edge, validated_edge], imported_by="test-suite"
    )
    rows = await regulatory_pool.fetch(
        "SELECT relation_id FROM public.regulatory_validated_relations ORDER BY relation_id"
    )
    assert [row["relation_id"] for row in rows] == [validated_edge.relation_id]


async def test_fixture_backed_edges_stay_out_of_the_view(regulatory_pool):
    bundle = await seed_trusted_family(regulatory_pool)
    edge = external_relation(bundle, validation=VALIDATED)
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    await regulatory_pool.execute(
        "UPDATE public.regulatory_source_artifacts SET fixture_only = true WHERE artifact_id = $1",
        bundle.artifacts[0].artifact_id,
    )
    count = await regulatory_pool.fetchval("SELECT count(*) FROM public.regulatory_validated_relations")
    assert count == 0
