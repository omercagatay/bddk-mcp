"""Drift contract: ``ontology/ontology.yml`` and the relation code must agree.

Like the tool-registry contract, this fails when a reviewed vocabulary changes
in one place only: a relation kind added to code without the spec, a spec link
type stored nowhere, or an amendment-chain type the graph tool does not follow.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

from bddk_mcp.migrations.v0009_regulatory_relations import V0009_REGULATORY_RELATIONS
from bddk_mcp.ontology.spec import DEFAULT_SPEC_PATH, OntologySpecError, load_ontology_spec
from bddk_mcp.regulatory import graph_queries, relation_extract
from bddk_mcp.regulatory.legal_versions import AuthorityLevel, LegalEventType, ValidationState
from bddk_mcp.regulatory.relations import RELATION_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_LINK_TYPES = frozenset({"contains", "creates"})


@pytest.fixture(scope="module")
def spec():
    return load_ontology_spec()


def _migration_relation_types() -> frozenset[str]:
    ddl = next(s for s in V0009_REGULATORY_RELATIONS.statements if "regulatory_relations_type_check" in s)
    clause = re.search(r"relation_type IN \(([^)]*)\)", ddl)
    assert clause is not None, "v0009 relation_type CHECK constraint not found"
    return frozenset(re.findall(r"'([a-z_]+)'", clause.group(1)))


def _mapping() -> dict:
    return yaml.safe_load(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, mapping: dict) -> Path:
    path = tmp_path / "ontology.yml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_spec_lives_at_the_repository_root_and_loads(spec):
    assert DEFAULT_SPEC_PATH == REPO_ROOT / "ontology" / "ontology.yml"
    assert spec.schema_version == 1
    assert spec.pilot.instrument_document_id == "mevzuat_22599"
    assert (REPO_ROOT / spec.pilot.plan).is_file()


def test_stored_link_types_match_relations_module(spec):
    assert spec.stored_relation_types() == RELATION_TYPES


def test_stored_link_types_match_v0009_check_constraint(spec):
    assert spec.stored_relation_types() == _migration_relation_types()


def test_new_link_types_are_declared_but_not_stored(spec):
    assert spec.unstored_link_types() == NEW_LINK_TYPES
    assert not NEW_LINK_TYPES & RELATION_TYPES
    assert spec.link_types["contains"].storage == "derived"
    assert spec.link_types["creates"].storage == "planned"


def test_amendment_chain_link_types_match_graph_queries(spec):
    assert spec.amendment_chain_types() == frozenset(graph_queries._CHAIN_EDGE_TYPES)


def test_regex_extractor_emits_only_stored_link_types(spec):
    emitted = {relation_type for relation_type, _, _, _ in relation_extract._PATTERNS}
    assert emitted <= spec.stored_relation_types()


def test_reviewed_vocabularies_are_exact(spec):
    assert set(spec.object_types) == {"instrument", "legal_version", "provision", "obligation"}
    assert set(spec.link_sources) == {"rule", "ai_approved", "action"}
    assert set(spec.proposal_statuses) == {"proposed", "approved", "rejected"}
    assert set(spec.action_types) == {"review_link_proposal", "confirm_obligation"}


def test_only_approved_proposals_reach_the_corpus_as_validated_rows(spec):
    assert spec.corpus_writing_status() == "approved"
    assert spec.proposal_statuses["approved"].validation_state == ValidationState.VALIDATED.value
    for name in ("proposed", "rejected"):
        assert spec.proposal_statuses[name].writes_to_corpus is False
        assert spec.proposal_statuses[name].validation_state is None
    for action in spec.action_types.values():
        assert action.writes == "review_store"


def test_review_states_lifecycle_events_and_authority_levels_match_enums(spec):
    assert spec.storage_review_states == tuple(state.value for state in ValidationState)
    assert spec.object_types["legal_version"].lifecycle_events == tuple(event.value for event in LegalEventType)
    authority_levels = {level.value for level in AuthorityLevel}
    assert {source.evidence_authority_level for source in spec.link_sources.values()} <= authority_levels
    assert spec.link_sources["rule"].evidence_authority_level == AuthorityLevel.AUTHORITATIVE.value


def test_link_sources_map_to_distinct_extraction_method_prefixes(spec):
    prefixes = [source.extraction_method_prefix for source in spec.link_sources.values()]
    assert len(set(prefixes)) == len(prefixes)
    assert all(prefix.endswith(":") for prefix in prefixes)
    assert set(prefixes) == {"rule:", "ai_approved:", "action:"}


def test_obligation_is_the_only_unpersisted_object_type(spec):
    unpersisted = {name for name, item in spec.object_types.items() if not item.is_persisted}
    assert unpersisted == {"obligation"}
    for name, item in spec.object_types.items():
        if item.is_persisted:
            assert item.storage.table.startswith("public.regulatory_"), name
            assert item.storage.model is not None, name


def test_every_object_and_link_is_explainable_from_the_file(spec):
    for name, item in {**spec.object_types, **spec.link_types}.items():
        assert item.label.strip(), name
        assert item.description.strip(), name
        assert item.example.strip(), name


def test_loader_rejects_a_relation_kind_the_code_does_not_know(tmp_path):
    """Drift in the other direction: a new stored link type in the spec must fail the contract."""

    mapping = _mapping()
    # Deep copy: a shared nested list would make yaml.safe_dump emit an alias the loader rejects.
    mapping["link_types"]["supplements"] = copy.deepcopy(mapping["link_types"]["cites"])
    drifted = load_ontology_spec(_write(tmp_path, mapping))
    assert drifted.stored_relation_types() != RELATION_TYPES
    assert drifted.stored_relation_types() != _migration_relation_types()


def test_loader_rejects_unknown_keys_and_dangling_references(tmp_path):
    with_extra = _mapping()
    with_extra["link_types"]["cites"]["severity"] = "high"
    with pytest.raises(OntologySpecError):
        load_ontology_spec(_write(tmp_path, with_extra))

    dangling = _mapping()
    dangling["link_types"]["cites"]["to_types"] = ["control"]
    with pytest.raises(OntologySpecError):
        load_ontology_spec(_write(tmp_path, dangling))

    stored_to_unpersisted = _mapping()
    stored_to_unpersisted["link_types"]["creates"]["storage"] = "regulatory_relations"
    with pytest.raises(OntologySpecError):
        load_ontology_spec(_write(tmp_path, stored_to_unpersisted))

    two_writers = _mapping()
    two_writers["proposal_statuses"]["rejected"]["writes_to_corpus"] = True
    two_writers["proposal_statuses"]["rejected"]["validation_state"] = "rejected"
    with pytest.raises(OntologySpecError):
        load_ontology_spec(_write(tmp_path, two_writers))


def test_loader_rejects_missing_or_ambiguous_files(tmp_path):
    with pytest.raises(OntologySpecError):
        load_ontology_spec(tmp_path / "missing.yml")
    aliased = tmp_path / "aliased.yml"
    aliased.write_text("schema_version: &v 1\npilot: *v\n", encoding="utf-8")
    with pytest.raises(OntologySpecError):
        load_ontology_spec(aliased)
