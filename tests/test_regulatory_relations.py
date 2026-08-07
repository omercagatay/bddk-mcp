"""Immutable edge writes into regulatory_relations."""

from __future__ import annotations

import dataclasses

import pytest

from bddk_mcp.regulatory.legal_versions import ValidationRecord, ValidationState
from bddk_mcp.regulatory.relations import (
    RELATION_TYPES,
    RegulatoryRelation,
    import_relations,
    relation_id,
)
from bddk_mcp.regulatory.repository import (
    LegalVersionPersistenceError,
    import_legal_version_bundle,
)
from tests.test_regulatory_legal_versions import _evidence, make_fixture_bundle
from tests.test_regulatory_repository import regulatory_pool  # noqa: F401

pytestmark = pytest.mark.asyncio

_UNVALIDATED = ValidationRecord(
    state=ValidationState.MACHINE_VALIDATED,
    validated_by=None,
    validated_at=None,
    method="regex:v1",
    review_record_sha256=None,
)


def _relation(**overrides) -> RegulatoryRelation:
    base = RegulatoryRelation(
        relation_type="cites",
        source_instrument_id="inst-tfrs9",
        source_provision_id="prov-943-ilke-5",
        target_instrument_id=None,
        target_provision_id=None,
        target_external_ref="5411 sayılı Bankacılık Kanunu madde 93",
        evidence=_evidence("ev-rel-1"),
        extraction_method="regex:v1",
        confidence=0.9,
        validation=_UNVALIDATED,
    )
    return dataclasses.replace(base, **overrides)


def test_relation_types_match_spec():
    assert RELATION_TYPES == frozenset(
        {"amends", "repeals", "replaces", "consolidates", "implements", "cites", "defines", "exception_to"}
    )


def test_relation_id_is_deterministic():
    assert relation_id(_relation()) == relation_id(_relation())
    assert relation_id(_relation()) != relation_id(_relation(relation_type="implements"))


def test_relation_requires_some_target():
    with pytest.raises(ValueError):
        _relation(target_external_ref=None)


def test_relation_rejects_unknown_type():
    with pytest.raises(ValueError):
        _relation(relation_type="mentions")


async def test_import_and_idempotent_reimport(regulatory_pool):  # noqa: F811
    bundle = make_fixture_bundle()
    await import_legal_version_bundle(regulatory_pool, bundle, imported_by="test-suite", allow_fixture=True)
    edge = _relation()
    assert await import_relations(regulatory_pool, [edge], imported_by="test-suite") == 1
    assert await import_relations(regulatory_pool, [edge], imported_by="test-suite") == 1
    count = await regulatory_pool.fetchval("SELECT count(*) FROM regulatory_relations")
    assert count == 1


async def test_conflicting_reimport_aborts(regulatory_pool):  # noqa: F811
    bundle = make_fixture_bundle()
    await import_legal_version_bundle(regulatory_pool, bundle, imported_by="test-suite", allow_fixture=True)
    edge = _relation()
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    tampered_row = await regulatory_pool.fetchval(
        "UPDATE regulatory_relations SET confidence = 0.1 WHERE relation_id = $1 RETURNING relation_id",
        relation_id(edge),
    )
    assert tampered_row is not None
    with pytest.raises(LegalVersionPersistenceError):
        await import_relations(regulatory_pool, [edge], imported_by="test-suite")
