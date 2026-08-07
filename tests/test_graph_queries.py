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
_REJECTED = dataclasses.replace(_HUMAN_OK, state=ValidationState.REJECTED)


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


async def _seed_contested_amends_edges(pool):
    """Extra `amends` edges on top of _seed(): one machine-only, one human-rejected.

    Kept additive and separate so the shared _seed() fixture (and the counts
    other test modules pin) stays untouched.
    """
    machine_edge = RegulatoryRelation(
        relation_type="amends",
        source_instrument_id="inst-tfrs9",
        source_provision_id=None,
        target_instrument_id="inst-tfrs9",
        target_provision_id=None,
        target_external_ref=None,
        evidence=_evidence("ev-edge-machine"),
        extraction_method="regex:v1",
        confidence=0.8,
        validation=_MACHINE_ONLY,
    )
    rejected_edge = dataclasses.replace(
        machine_edge,
        evidence=_evidence("ev-edge-rejected"),
        validation=_REJECTED,
    )
    await import_relations(pool, [machine_edge, rejected_edge], imported_by="test-suite")


async def _seed_incoming_edge(pool):
    """Second instrument amending inst-tfrs9: a real (non-self-loop) incoming edge."""
    await pool.execute(
        """
        INSERT INTO regulatory_instruments
            (instrument_id, jurisdiction, authority_code, identity_key,
             canonical_title, instrument_type)
        VALUES ('inst-other', 'TR', 'BDDK', 'rehber:999',
                'TFRS 9 Değişiklik Rehberi', 'Rehber')
        ON CONFLICT (instrument_id) DO NOTHING
        """
    )
    edge = RegulatoryRelation(
        relation_type="amends",
        source_instrument_id="inst-other",
        source_provision_id=None,
        target_instrument_id="inst-tfrs9",
        target_provision_id=None,
        target_external_ref=None,
        evidence=_evidence("ev-edge-incoming"),
        extraction_method="manual",
        confidence=1.0,
        validation=_HUMAN_OK,
    )
    await import_relations(pool, [edge], imported_by="test-suite")


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


async def test_amendment_chain_edges_validated_only_by_default(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    await _seed_contested_amends_edges(regulatory_pool)
    chain = await amendment_chain(regulatory_pool, instrument_id="inst-tfrs9")
    edges = chain[0]["edges"]
    assert edges, "default chain must keep the human-validated edge"
    assert {e["validation_state"] for e in edges} == {"human_validated"}
    assert {e["evidence_id"] for e in edges} == {"ev-edge-1"}


async def test_amendment_chain_include_unvalidated_never_rejected(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    await _seed_contested_amends_edges(regulatory_pool)
    chain = await amendment_chain(regulatory_pool, instrument_id="inst-tfrs9", include_unvalidated=True)
    edges = chain[0]["edges"]
    evidence_ids = {e["evidence_id"] for e in edges}
    assert "ev-edge-machine" in evidence_ids  # machine_validated included on opt-in
    assert "ev-edge-rejected" not in evidence_ids  # rejected never rides along
    assert "rejected" not in {e["validation_state"] for e in edges}


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


async def test_cross_references_include_unvalidated_never_rejected(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    await _seed_contested_amends_edges(regulatory_pool)
    edges = await cross_references(
        regulatory_pool, doc_id="943", section_type=None, section_ref=None, include_unvalidated=True
    )
    evidence_ids = {e["evidence_id"] for e in edges}
    assert "ev-edge-machine" in evidence_ids
    assert "ev-edge-rejected" not in evidence_ids
    assert "rejected" not in {e["validation_state"] for e in edges}


async def test_cross_references_direction_filter(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    await _seed_incoming_edge(regulatory_pool)
    incoming = await cross_references(
        regulatory_pool, doc_id="943", section_type=None, section_ref=None, direction="incoming"
    )
    assert incoming, "direction filter must be asserted over non-empty results"
    assert all(e["direction"] == "incoming" for e in incoming)
    assert {e["source_instrument_id"] for e in incoming} == {"inst-other"}
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
