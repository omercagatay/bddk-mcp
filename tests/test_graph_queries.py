"""Validated-view traversal: amendment chains and cross-reference neighborhoods."""

from __future__ import annotations

import json

import pytest

from bddk_mcp.regulatory.graph_queries import (
    _instrument_for_doc,
    amendment_chain,
    cross_references,
    one_hop_section_refs,
)
from bddk_mcp.regulatory.legal_versions import (
    LegalVersionBundle,
    canonical_bundle_sha256,
)
from bddk_mcp.regulatory.relations import import_relations
from bddk_mcp.regulatory.repository import import_legal_version_bundle
from tests.test_legal_versions import FIXTURE
from tests.test_regulatory_relations import VALIDATED, external_relation

pytestmark = pytest.mark.asyncio


async def seed_family_for_doc(pool, *, document_id: str = "943") -> LegalVersionBundle:
    """Import the trusted synthetic family bound to a stored document ID.

    repository_document_id and fixture_only are not identity components, so
    the derived artifact IDs stay valid after the rebind.
    """
    mapping = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mapping["fixture_only"] = False
    for artifact in mapping["artifacts"]:
        artifact["fixture_only"] = False
        artifact["repository_document_id"] = document_id
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    bundle = LegalVersionBundle.model_validate(mapping)
    await pool.execute(
        "INSERT INTO public.documents (document_id, title) VALUES ($1, 'Synthetic fixture')"
        " ON CONFLICT (document_id) DO NOTHING",
        document_id,
    )
    await import_legal_version_bundle(pool, bundle, imported_by="test-suite")
    return bundle


async def test_amendment_chain_returns_only_validated_versions(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool)
    chain = await amendment_chain(regulatory_pool, instrument_id=bundle.instrument.instrument_id)
    # The synthetic family has a validated v1 and an unvalidated v2 successor:
    # only v1 may surface, and events on it must all be validated claims.
    assert [entry["version_key"] for entry in chain] == ["synthetic-v1"]
    assert chain[0]["predecessor_version_id"] is None
    event_types = {event["event_type"] for event in chain[0]["events"]}
    assert "publication" in event_types
    assert all(event["evidence_id"].startswith("evid_sha256_") for event in chain[0]["events"])


async def test_amendment_chain_resolves_doc_id_through_validated_artifacts(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool, document_id="943")
    assert await _instrument_for_doc(regulatory_pool, "943") == bundle.instrument.instrument_id
    chain = await amendment_chain(regulatory_pool, doc_id="943")
    assert chain and chain[0]["version_key"] == "synthetic-v1"


async def test_amendment_chain_lists_validated_incoming_edges_only(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool)
    instrument_id = bundle.instrument.instrument_id
    validated_edge = external_relation(
        bundle,
        statement="validated amends claim",
        relation_type="amends",
        target_instrument_id=instrument_id,
        target_external_ref=None,
        validation=VALIDATED,
    )
    unvalidated_edge = external_relation(
        bundle,
        statement="machine amends claim",
        relation_type="repeals",
        target_instrument_id=instrument_id,
        target_external_ref=None,
    )
    await import_relations(regulatory_pool, [validated_edge, unvalidated_edge], imported_by="test-suite")
    chain = await amendment_chain(regulatory_pool, instrument_id=instrument_id)
    edges = chain[0]["edges"]
    assert [edge["relation_type"] for edge in edges] == ["amends"]
    assert edges[0]["evidence_id"] == validated_edge.evidence.evidence_id


async def test_amendment_chain_for_unknown_doc_is_empty(regulatory_pool):
    assert await amendment_chain(regulatory_pool, doc_id="no-such-doc") == []


async def test_amendment_chain_requires_a_subject(regulatory_pool):
    with pytest.raises(ValueError):
        await amendment_chain(regulatory_pool)


async def test_cross_references_serves_validated_edges_only(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool)
    validated_edge = external_relation(bundle, statement="validated cite", validation=VALIDATED)
    unvalidated_edge = external_relation(bundle, statement="machine cite")
    await import_relations(regulatory_pool, [validated_edge, unvalidated_edge], imported_by="test-suite")
    edges = await cross_references(regulatory_pool, doc_id="943", section_type=None, section_ref=None)
    assert len(edges) == 1
    assert edges[0]["evidence_id"] == validated_edge.evidence.evidence_id
    assert edges[0]["target_external_ref"] == "5411 sayılı Bankacılık Kanunu madde 93"
    assert edges[0]["direction"] == "outgoing"
    assert edges[0]["depth"] == 1


async def test_cross_references_direction_filter(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool)
    edge = external_relation(bundle, statement="validated cite", validation=VALIDATED)
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    incoming_only = await cross_references(
        regulatory_pool, doc_id="943", section_type=None, section_ref=None, direction="incoming"
    )
    assert incoming_only == []


async def test_cross_references_rejects_unknown_direction(regulatory_pool):
    with pytest.raises(ValueError):
        await cross_references(regulatory_pool, doc_id="943", section_type=None, section_ref=None, direction="sideways")


async def test_cross_references_unmapped_doc_is_empty(regulatory_pool):
    assert await cross_references(regulatory_pool, doc_id="no-such-doc", section_type=None, section_ref=None) == []


async def test_cross_references_unmapped_section_is_empty(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool)
    edge = external_relation(bundle, statement="validated cite", validation=VALIDATED)
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    # The section never resolves to a validated provision citation, so section
    # narrowing fails closed instead of falling back to document scope.
    edges = await cross_references(regulatory_pool, doc_id="943", section_type="madde", section_ref="9")
    assert edges == []


async def test_one_hop_section_refs_without_citations_is_empty(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool)
    edge = external_relation(bundle, statement="validated cite", validation=VALIDATED)
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    neighbors = await one_hop_section_refs(regulatory_pool, doc_id="943", section_type="ilke", section_ref="5")
    assert neighbors == []
