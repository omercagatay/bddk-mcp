"""End-to-end bundle import against the regulatory schema."""

from __future__ import annotations

import dataclasses

import pytest

from bddk_mcp.regulatory.repository import (
    LegalVersionPersistenceError,
    import_legal_version_bundle,
)
from bddk_mcp.regulatory.schema import apply_regulatory_schema
from tests.test_regulatory_legal_versions import make_fixture_bundle

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def regulatory_pool(pg_pool):
    await apply_regulatory_schema(pg_pool)
    tables = (
        "regulatory_family_imports",
        "regulatory_relations",
        "regulatory_legal_version_provisions",
        "regulatory_legal_status_assertions",
        "regulatory_legal_events",
        "regulatory_legal_version_artifacts",
        "regulatory_provisions",
        "regulatory_legal_versions",
        "regulatory_evidence",
        "regulatory_source_artifacts",
        "regulatory_instruments",
    )
    for table in tables:
        await pg_pool.execute(f"TRUNCATE {table} CASCADE")
    yield pg_pool


async def test_import_bundle_roundtrip(regulatory_pool):
    bundle = make_fixture_bundle()
    result = await import_legal_version_bundle(
        regulatory_pool, bundle, imported_by="test-suite", allow_fixture=True
    )
    assert result.instrument_id == "inst-tfrs9"
    assert result.version_count == 2
    versions = await regulatory_pool.fetch(
        "SELECT legal_version_id, predecessor_version_id FROM regulatory_legal_versions ORDER BY version_key"
    )
    assert [dict(row) for row in versions] == [
        {"legal_version_id": "ver-1", "predecessor_version_id": None},
        {"legal_version_id": "ver-2", "predecessor_version_id": "ver-1"},
    ]


async def test_reimport_identical_bundle_is_noop(regulatory_pool):
    bundle = make_fixture_bundle()
    await import_legal_version_bundle(regulatory_pool, bundle, imported_by="test-suite", allow_fixture=True)
    await import_legal_version_bundle(regulatory_pool, bundle, imported_by="test-suite", allow_fixture=True)
    count = await regulatory_pool.fetchval("SELECT count(*) FROM regulatory_legal_versions")
    assert count == 2


async def test_conflicting_identity_aborts(regulatory_pool):
    bundle = make_fixture_bundle()
    await import_legal_version_bundle(regulatory_pool, bundle, imported_by="test-suite", allow_fixture=True)
    tampered_instrument = dataclasses.replace(bundle.instrument, canonical_title="Different Title")
    tampered = dataclasses.replace(bundle, instrument=tampered_instrument)
    from bddk_mcp.regulatory.legal_versions import canonical_bundle_sha256

    tampered = dataclasses.replace(tampered, bundle_sha256=canonical_bundle_sha256(tampered))
    with pytest.raises(LegalVersionPersistenceError):
        await import_legal_version_bundle(
            regulatory_pool, tampered, imported_by="test-suite", allow_fixture=True
        )


async def test_fixture_bundle_rejected_without_flag(regulatory_pool):
    bundle = make_fixture_bundle()
    with pytest.raises(LegalVersionPersistenceError):
        await import_legal_version_bundle(regulatory_pool, bundle, imported_by="test-suite")
