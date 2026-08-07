"""Recursive traversal over versions and relation edges."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from bddk_mcp.regulatory.bridge import ensure_section_provision_map, refresh_section_provision_map
from bddk_mcp.regulatory.graph_queries import amendment_chain, cross_references
from bddk_mcp.regulatory.legal_versions import ValidationRecord, ValidationState
from bddk_mcp.regulatory.relations import RegulatoryRelation, import_relations
from bddk_mcp.regulatory.repository import import_legal_version_bundle
from tests.test_regulatory_legal_versions import _evidence, make_fixture_bundle
from tests.test_regulatory_repository import regulatory_pool  # noqa: F401

pytestmark = pytest.mark.asyncio

_HUMAN_OK = ValidationRecord(
    state=ValidationState.HUMAN_VALIDATED,
    validated_by="reviewer@example.test",
    validated_at=datetime(2026, 8, 1, tzinfo=UTC),
    method="manual-review",
    review_record_sha256="a" * 64,
)
_MACHINE_ONLY = dataclasses.replace(_HUMAN_OK, state=ValidationState.MACHINE_VALIDATED)


async def _seed(pool):
    bundle = make_fixture_bundle()
    await import_legal_version_bundle(pool, bundle, imported_by="test-suite", allow_fixture=True)
    validated_edge = RegulatoryRelation(
        relation_type="amends",
        source_instrument_id="inst-tfrs9",
        source_provision_id=None,
        target_instrument_id="inst-tfrs9",
        target_provision_id="prov-943-ilke-5",
        target_external_ref=None,
        evidence=_evidence("ev-edge-1"),
        extraction_method="manual",
        confidence=1.0,
        validation=_HUMAN_OK,
    )
    unvalidated_edge = dataclasses.replace(
        validated_edge,
        relation_type="cites",
        target_instrument_id=None,
        target_provision_id=None,
        target_external_ref="5411 sayılı Bankacılık Kanunu",
        evidence=_evidence("ev-edge-2"),
        validation=_MACHINE_ONLY,
    )
    await import_relations(pool, [validated_edge, unvalidated_edge], imported_by="test-suite")


async def test_amendment_chain_orders_versions(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    chain = await amendment_chain(regulatory_pool, instrument_id="inst-tfrs9")
    assert [entry["legal_version_id"] for entry in chain] == ["ver-1", "ver-2"]
    assert chain[1]["predecessor_version_id"] == "ver-1"
    supersessions = [e for e in chain[1]["events"] if e["event_type"] == "supersession"]
    assert supersessions and supersessions[0]["evidence_id"] == "ev-4"


async def test_amendment_chain_by_doc_id(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    chain = await amendment_chain(regulatory_pool, doc_id="943")
    assert len(chain) == 2


async def test_amendment_chain_unknown_instrument_is_empty(regulatory_pool):  # noqa: F811
    assert await amendment_chain(regulatory_pool, instrument_id="inst-nope") == []


async def test_cross_references_validated_only_by_default(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    edges = await cross_references(regulatory_pool, doc_id="943", section_type=None, section_ref=None)
    assert {e["relation_type"] for e in edges} == {"amends"}
    all_edges = await cross_references(
        regulatory_pool, doc_id="943", section_type=None, section_ref=None, include_unvalidated=True
    )
    assert {e["relation_type"] for e in all_edges} == {"amends", "cites"}
    cites = next(e for e in all_edges if e["relation_type"] == "cites")
    assert cites["validation_state"] == "machine_validated"
    assert cites["target_external_ref"] == "5411 sayılı Bankacılık Kanunu"


async def test_cross_references_direction_filter(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    incoming = await cross_references(
        regulatory_pool, doc_id="943", section_type=None, section_ref=None, direction="incoming"
    )
    assert all(e["direction"] == "incoming" for e in incoming)
    with pytest.raises(ValueError):
        await cross_references(
            regulatory_pool, doc_id="943", section_type=None, section_ref=None, direction="sideways"
        )


async def test_cross_references_no_coverage_is_empty(regulatory_pool):  # noqa: F811
    await ensure_section_provision_map(regulatory_pool)
    await refresh_section_provision_map(regulatory_pool)
    assert await cross_references(
        regulatory_pool, doc_id="unmapped-doc", section_type=None, section_ref=None
    ) == []
