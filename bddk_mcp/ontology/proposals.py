"""Proposal file model for the ontology pilot: candidates that count for nothing until approved.

A proposal is a machine-made claim about a link (``amends``, ``repeals`` …) or a
dated event (``publication``, ``effective``), bound to the exact stored text it
was read from: the document's content hash, character offsets and the verbatim
quote.  Proposals live in a JSON file under ``docs/evidence/legal-versions/``.
The review store decides on them (Step 3) and only approved decisions reach a
legal-version bundle (Step 4).  Nothing in this module touches a database.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROPOSAL_SCHEMA_VERSION = 1
_PROPOSAL_ID_PATTERN = r"^prop_sha256_[0-9a-f]{64}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_RULE_ID_PATTERN = r"^[a-z][a-z0-9_]*:v[0-9]+$"
_PROVISION_PATH_PATTERN = r"^[a-z_]+/[0-9a-z/]+$"
_MAX_FILE_BYTES = 4 * 1024 * 1024


class ProposalFileError(ValueError):
    """The proposals file is missing, unreadable, ambiguous, or violates its closed schema."""


class ProposalSource(StrEnum):
    """How a proposal was made; mirrors ``link_sources`` in ``ontology/ontology.yml``."""

    RULE = "rule"
    AI_APPROVED = "ai_approved"
    ACTION = "action"


class ProposalKind(StrEnum):
    LINK = "link"
    EVENT = "event"


class ProposalFlag(StrEnum):
    CROSS_CONFIRMED = "cross_confirmed"
    NEEDS_REVIEW = "needs_review"
    RG_NUMBER_MALFORMED = "rg_number_malformed"
    NOT_IN_RG_TABLE = "not_in_rg_table"


class TargetKind(StrEnum):
    RG_INSTRUMENT = "rg_instrument"
    EXTERNAL_REF = "external_ref"
    DATE = "date"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _turkish_date(value: date) -> str:
    return f"{value.day}/{value.month}/{value.year}"


def dotted_date(value: date) -> str:
    """Turkish written form, zero-padded: ``01.01.2018``."""

    return value.strftime("%d.%m.%Y")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextEvidence(_Frozen):
    """The verbatim quote and its exact position in the stored Markdown."""

    quote: str = Field(min_length=1, max_length=4000)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    statement_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _quote_matches_its_span_and_hash(self) -> TextEvidence:
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError("evidence span length does not match the quote")
        if self.statement_sha256 != sha256_text(self.quote):
            raise ValueError("evidence hash does not match the quote")
        return self

    @classmethod
    def from_span(cls, text: str, start: int, end: int) -> TextEvidence:
        quote = text[start:end]
        return cls(quote=quote, start_char=start, end_char=end, statement_sha256=sha256_text(quote))


class RgReference(_Frozen):
    """One Resmî Gazete issue: date and number, plus the text as written."""

    date: date
    number: str = Field(pattern=r"^[0-9]{1,8}$")
    raw: str = Field(min_length=1, max_length=100)

    @property
    def is_malformed(self) -> bool:
        """Resmî Gazete numbers are five digits; anything else is an extraction artifact."""

        return len(self.number) != 5

    def key(self) -> str:
        return f"{self.date.isoformat()}-{self.number}"

    def label_tr(self) -> str:
        """Compact form mirroring the source tag, e.g. ``RG 14/12/2016-29918``."""

        return f"RG {_turkish_date(self.date)}-{self.number}"

    def phrase_tr(self) -> str:
        """The official citation form: ``14.12.2016 tarihli ve 29918 sayılı Resmî Gazete``."""

        return f"{dotted_date(self.date)} tarihli ve {self.number} sayılı Resmî Gazete"


class ProposalTarget(_Frozen):
    """What the link points at, or the date an event claims; exactly one field is set."""

    kind: TargetKind
    rg: RgReference | None = None
    external_ref: str | None = Field(default=None, min_length=1, max_length=500)
    value_date: date | None = None

    @model_validator(mode="after")
    def _exactly_the_field_for_the_kind(self) -> ProposalTarget:
        expected = {
            TargetKind.RG_INSTRUMENT: self.rg is not None and self.external_ref is None and self.value_date is None,
            TargetKind.EXTERNAL_REF: self.rg is None and self.external_ref is not None and self.value_date is None,
            TargetKind.DATE: self.rg is None and self.external_ref is None and self.value_date is not None,
        }
        if not expected[self.kind]:
            raise ValueError("proposal target must set exactly the field that matches its kind")
        return self

    def key(self) -> str:
        if self.rg is not None:
            return f"rg:{self.rg.key()}"
        if self.external_ref is not None:
            return f"external:{sha256_text(self.external_ref)}"
        assert self.value_date is not None
        return f"date:{self.value_date.isoformat()}"


def proposal_id_for(
    *,
    document_id: str,
    content_hash: str,
    rule_id: str,
    kind: ProposalKind | str,
    link_type: str | None,
    event_type: str | None,
    subject_provision: str | None,
    target_key: str,
    start_char: int,
    end_char: int,
) -> str:
    """Deterministic identity: the same text, rule and span always yield the same proposal."""

    payload = json.dumps(
        [
            document_id,
            content_hash,
            rule_id,
            str(kind),
            link_type or "",
            event_type or "",
            subject_provision or "",
            target_key,
            start_char,
            end_char,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"prop_sha256_{hashlib.sha256(payload).hexdigest()}"


class Proposal(_Frozen):
    """One reviewable claim.  ``status`` is deliberately absent: decisions live in the review store."""

    proposal_id: str = Field(pattern=_PROPOSAL_ID_PATTERN)
    kind: ProposalKind
    source: ProposalSource
    rule_id: str = Field(pattern=_RULE_ID_PATTERN)
    link_type: str | None = Field(default=None, pattern=r"^[a-z_]+$")
    event_type: str | None = Field(default=None, pattern=r"^[a-z_]+$")
    change_kind: str | None = Field(default=None, min_length=1, max_length=100)
    subject_document_id: str = Field(min_length=1, max_length=200)
    subject_provision: str | None = Field(default=None, pattern=_PROVISION_PATH_PATTERN)
    target: ProposalTarget
    evidence: TextEvidence
    confidence: float = Field(ge=0.0, le=1.0)
    flags: tuple[ProposalFlag, ...] = ()
    preview_tr: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _shape_is_coherent(self) -> Proposal:
        if self.kind is ProposalKind.LINK and (self.link_type is None or self.event_type is not None):
            raise ValueError("a link proposal needs link_type and no event_type")
        if self.kind is ProposalKind.EVENT and (self.event_type is None or self.link_type is not None):
            raise ValueError("an event proposal needs event_type and no link_type")
        if self.kind is ProposalKind.EVENT and self.target.kind is not TargetKind.DATE:
            raise ValueError("an event proposal targets a date")
        if self.kind is ProposalKind.LINK and self.target.kind is TargetKind.DATE:
            raise ValueError("a link proposal cannot target a date")
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("proposal flags are duplicated")
        return self

    def verify_identity(self, *, document_id: str, content_hash: str) -> bool:
        return self.subject_document_id == document_id and self.proposal_id == proposal_id_for(
            document_id=document_id,
            content_hash=content_hash,
            rule_id=self.rule_id,
            kind=self.kind,
            link_type=self.link_type,
            event_type=self.event_type,
            subject_provision=self.subject_provision,
            target_key=self.target.key(),
            start_char=self.evidence.start_char,
            end_char=self.evidence.end_char,
        )


class RgTableRow(_Frozen):
    """One row of the Resmî Gazete table at the end of a mevzuat.gov.tr text."""

    ordinal: int | None = Field(default=None, ge=1)
    rg: RgReference
    evidence: TextEvidence


class RgTable(_Frozen):
    original: RgTableRow
    amendments: tuple[RgTableRow, ...] = ()

    @model_validator(mode="after")
    def _rows_are_well_formed(self) -> RgTable:
        if self.original.ordinal is not None:
            raise ValueError("the original publication row carries no ordinal")
        ordinals = [row.ordinal for row in self.amendments]
        if ordinals != list(range(1, len(ordinals) + 1)):
            raise ValueError("amendment rows must be numbered 1..n in order")
        keys = [row.rg.key() for row in self.amendments]
        if len(set(keys)) != len(keys):
            raise ValueError("amendment rows repeat a Resmî Gazete issue")
        return self

    def amendment_keys(self) -> frozenset[str]:
        return frozenset(row.rg.key() for row in self.amendments)


class ProposalCounts(_Frozen):
    total: int = Field(ge=0)
    links: int = Field(ge=0)
    events: int = Field(ge=0)
    by_rule: dict[str, int]
    by_flag: dict[str, int]


def counts_for(proposals: tuple[Proposal, ...] | list[Proposal]) -> ProposalCounts:
    by_rule: dict[str, int] = {}
    by_flag: dict[str, int] = {}
    for proposal in proposals:
        by_rule[proposal.rule_id] = by_rule.get(proposal.rule_id, 0) + 1
        for flag in proposal.flags:
            by_flag[flag.value] = by_flag.get(flag.value, 0) + 1
    return ProposalCounts(
        total=len(proposals),
        links=sum(1 for item in proposals if item.kind is ProposalKind.LINK),
        events=sum(1 for item in proposals if item.kind is ProposalKind.EVENT),
        by_rule=dict(sorted(by_rule.items())),
        by_flag=dict(sorted(by_flag.items())),
    )


class ProposalFile(_Frozen):
    """Every proposal one generator produced for one exact stored text."""

    schema_version: Literal[1]
    generator: str = Field(pattern=_RULE_ID_PATTERN)
    ontology_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    document_id: str = Field(min_length=1, max_length=200)
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    title: str = Field(min_length=1, max_length=1000)
    rg_table: RgTable
    proposals: tuple[Proposal, ...]
    counts: ProposalCounts

    @model_validator(mode="after")
    def _file_is_internally_consistent(self) -> ProposalFile:
        ids = [item.proposal_id for item in self.proposals]
        if len(set(ids)) != len(ids):
            raise ValueError("proposal identities are duplicated")
        order = [(item.evidence.start_char, item.proposal_id) for item in self.proposals]
        if order != sorted(order):
            raise ValueError("proposals must be ordered by text position, then identity")
        for item in self.proposals:
            if not item.verify_identity(document_id=self.document_id, content_hash=self.content_hash):
                raise ValueError("a proposal identity does not match its components")
        if counts_for(self.proposals) != self.counts:
            raise ValueError("proposal counts do not match the proposals")
        return self


def dump_proposal_file(file: ProposalFile) -> str:
    """Stable JSON: the same proposals always serialize to the same bytes."""

    return json.dumps(file.model_dump(mode="json"), ensure_ascii=False, indent=1) + "\n"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProposalFileError("Proposals file contains a duplicate JSON key.")
        result[key] = value
    return result


def load_proposal_file(path: str | Path) -> ProposalFile:
    """Load one bounded, self-consistent proposals file; no database or network access."""

    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise ProposalFileError("Proposals file is missing or is not a regular file.")
        size = source.stat().st_size
        if size <= 0 or size > _MAX_FILE_BYTES:
            raise ProposalFileError("Proposals file is empty or exceeds its size limit.")
        mapping = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(mapping, dict):
            raise ProposalFileError("Proposals file root must be a JSON object.")
        return ProposalFile.model_validate(mapping)
    except ProposalFileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise ProposalFileError(f"Proposals file failed validation: {type(exc).__name__}") from None


def provision_label_tr(path: str | None) -> str:
    """Human label for a canonical provision path, e.g. ``madde/23`` → ``Madde 23``."""

    if path is None:
        return "yönetmeliğin bütünü"
    kind, _, ref = path.partition("/")
    names = {"madde": "Madde", "gecici_madde": "Geçici Madde", "ek_madde": "Ek Madde", "fikra": "Fıkra", "bent": "Bent"}
    return f"{names.get(kind, kind)} {ref}"
