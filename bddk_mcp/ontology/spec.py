"""Closed-schema loader for ``ontology/ontology.yml``.

The specification is reviewed documentation first and a code contract
second: every object type and link type must carry a Turkish label, a
description and a pilot example, so the owner can explain the model from the
file alone.  The loader rejects unknown keys, dangling references and
ambiguous YAML; the drift tests then compare the loaded vocabularies with the
relation code and the schema migration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bddk_mcp.release_yaml import ReleaseYamlError, load_bounded_release_yaml

DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "ontology" / "ontology.yml"
_MAX_SPEC_BYTES = 256 * 1024
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

LinkStorage = Literal["regulatory_relations", "derived", "planned"]
REGULATORY_RELATIONS: LinkStorage = "regulatory_relations"


class OntologySpecError(ValueError):
    """The specification is missing, unreadable, ambiguous, or violates its closed schema."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Principle(_Frozen):
    id: str = Field(pattern=_IDENTIFIER_RE.pattern)
    text: str = Field(min_length=1, max_length=500)


class Pilot(_Frozen):
    instrument_document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    plan: str = Field(min_length=1, max_length=200)


class Storage(_Frozen):
    """Where an object type lives; ``None`` fields mean "not persisted yet"."""

    table: str | None = Field(default=None, pattern=r"^[a-z_]+\.[a-z_]+$")
    id_prefix: str | None = Field(default=None, pattern=r"^[a-z]+_sha256_$")
    model: str | None = Field(default=None, pattern=r"^bddk_mcp(\.[a-z_]+)+\.[A-Za-z]+$")


class ObjectType(_Frozen):
    label: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    identity: tuple[str, ...] = Field(min_length=1)
    storage: Storage
    lifecycle_events: tuple[str, ...] = ()
    example: str = Field(min_length=1, max_length=500)

    @property
    def is_persisted(self) -> bool:
        return self.storage.table is not None


class LinkType(_Frozen):
    label: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    from_types: tuple[str, ...] = Field(min_length=1)
    to_types: tuple[str, ...] = Field(min_length=1)
    storage: LinkStorage
    amendment_chain: bool
    example: str = Field(min_length=1, max_length=500)


class LinkSource(_Frozen):
    label: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    extraction_method_prefix: str = Field(pattern=r"^[a-z_]+:$")
    approval: str = Field(min_length=1, max_length=1000)
    evidence_authority_level: Literal["authoritative", "secondary", "repository_fixture"]
    example: str = Field(min_length=1, max_length=500)


class ProposalStatus(_Frozen):
    label: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    writes_to_corpus: bool
    validation_state: str | None = None
    transitions: tuple[str, ...] = ()


class ActionType(_Frozen):
    label: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    inputs: tuple[str, ...] = Field(min_length=1)
    decision_values: tuple[str, ...] = Field(min_length=1)
    writes: Literal["review_store"]
    audit_fields: tuple[str, ...] = Field(min_length=1)
    rules: tuple[str, ...] = Field(min_length=1)


class OntologySpec(_Frozen):
    """The whole reviewed specification; references between sections must close."""

    schema_version: Literal[1]
    pilot: Pilot
    principles: tuple[Principle, ...] = Field(min_length=1)
    object_types: dict[str, ObjectType] = Field(min_length=1)
    link_types: dict[str, LinkType] = Field(min_length=1)
    link_sources: dict[str, LinkSource] = Field(min_length=1)
    proposal_statuses: dict[str, ProposalStatus] = Field(min_length=1)
    storage_review_states: tuple[str, ...] = Field(min_length=1)
    action_types: dict[str, ActionType] = Field(min_length=1)

    @model_validator(mode="after")
    def _references_are_closed(self) -> OntologySpec:
        for section in (
            self.object_types,
            self.link_types,
            self.link_sources,
            self.proposal_statuses,
            self.action_types,
        ):
            for key in section:
                if not _IDENTIFIER_RE.fullmatch(key):
                    raise ValueError("ontology keys must be snake_case identifiers")
        for name, link in self.link_types.items():
            for endpoint in (*link.from_types, *link.to_types):
                if endpoint not in self.object_types:
                    raise ValueError(f"link type {name} references an unknown object type")
            if link.storage == REGULATORY_RELATIONS:
                unstored = [
                    endpoint
                    for endpoint in (*link.from_types, *link.to_types)
                    if not self.object_types[endpoint].is_persisted
                ]
                if unstored:
                    raise ValueError(f"stored link type {name} connects an object type that is not persisted")
        writers = [name for name, status in self.proposal_statuses.items() if status.writes_to_corpus]
        for name, status in self.proposal_statuses.items():
            for target in status.transitions:
                if target not in self.proposal_statuses:
                    raise ValueError(f"proposal status {name} transitions to an unknown status")
            if status.writes_to_corpus != (status.validation_state is not None):
                raise ValueError(f"proposal status {name} must map to a review state exactly when it writes")
            if status.validation_state is not None and status.validation_state not in self.storage_review_states:
                raise ValueError(f"proposal status {name} maps to an unknown review state")
        if len(writers) != 1:
            raise ValueError("exactly one proposal status may write to the corpus")
        for name, action in self.action_types.items():
            for decision in action.decision_values:
                if decision not in self.proposal_statuses:
                    raise ValueError(f"action type {name} decides an unknown proposal status")
        if len(set(self.storage_review_states)) != len(self.storage_review_states):
            raise ValueError("storage_review_states are duplicated")
        return self

    def stored_relation_types(self) -> frozenset[str]:
        """Link kinds persisted as edges in ``public.regulatory_relations``."""

        return frozenset(name for name, link in self.link_types.items() if link.storage == REGULATORY_RELATIONS)

    def unstored_link_types(self) -> frozenset[str]:
        """Link kinds the spec declares but no table stores yet (derived or planned)."""

        return frozenset(name for name, link in self.link_types.items() if link.storage != REGULATORY_RELATIONS)

    def amendment_chain_types(self) -> frozenset[str]:
        """Link kinds the amendment-chain tool follows."""

        return frozenset(name for name, link in self.link_types.items() if link.amendment_chain)

    def corpus_writing_status(self) -> str:
        """The single proposal status whose decisions may reach the corpus."""

        return next(name for name, status in self.proposal_statuses.items() if status.writes_to_corpus)


def load_ontology_spec(path: str | Path = DEFAULT_SPEC_PATH) -> OntologySpec:
    """Load and validate the specification from a bounded regular file.

    No database or network access.  Errors are summarized without copying the
    file so a log line cannot leak its content.
    """

    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise OntologySpecError("Ontology specification is missing or is not a regular file.")
        size = source.stat().st_size
        if size <= 0 or size > _MAX_SPEC_BYTES:
            raise OntologySpecError("Ontology specification is empty or exceeds its size limit.")
        payload = load_bounded_release_yaml(source.read_bytes(), maximum_bytes=_MAX_SPEC_BYTES)
        if not isinstance(payload, dict):
            raise OntologySpecError("Ontology specification root must be a mapping.")
        return OntologySpec.model_validate(payload)
    except OntologySpecError:
        raise
    except (OSError, ReleaseYamlError, ValidationError, TypeError, ValueError) as exc:
        raise OntologySpecError(f"Ontology specification failed validation: {_summarize(exc)}") from None


def _summarize(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        locations = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        return "invalid fields " + ", ".join(locations[:8])
    return type(exc).__name__
