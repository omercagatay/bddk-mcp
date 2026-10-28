"""Rule-based proposal extraction from the stored text of one regulation.

Three deterministic rules read the consolidated Markdown of a mevzuat.gov.tr
regulation and write *proposals*, never edges:

* ``rg_table:v1`` — the Resmî Gazete table at the end of the text: the original
  publication row and one row per amending regulation.
* ``inline_tag:v1`` — every ``(Değişik|Ek|Mülga:RG-DD/MM/YYYY-NNNNN)`` tag and the
  article it sits in, located with the same section parser that later maps
  provisions, so paths such as ``madde/23`` line up.
* ``article_clause:v1`` — the repeal clause ("… sayılı Resmî Gazete'de
  yayımlanan … yürürlükten kaldırılmıştır") and the entry-into-force clause
  ("Bu Yönetmelik … tarihinde yürürlüğe girer").

Each proposal quotes the exact text with character offsets so a reviewer can
check it.  Flags: ``cross_confirmed`` when a tag's Resmî Gazete date and number
appear in the table (and when a table row is echoed by at least one tag),
``needs_review`` for legal claims (publication, effective, repeal),
``rg_number_malformed`` when a Resmî Gazete number is not five digits (a glued
footnote such as ``294599``), ``not_in_rg_table`` for a tag the table does not
know.  The module imports no database driver.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from datetime import date

from bddk_mcp.ontology.proposals import (
    PROPOSAL_SCHEMA_VERSION,
    Proposal,
    ProposalCounts,
    ProposalFile,
    ProposalFlag,
    ProposalKind,
    ProposalSource,
    ProposalTarget,
    RgReference,
    RgTable,
    RgTableRow,
    TargetKind,
    TextEvidence,
    counts_for,
    proposal_id_for,
    provision_label_tr,
    sha256_text,
)
from bddk_mcp.store.section_index import extract_document_sections

GENERATOR = "rule_extract:v1"
RULE_RG_TABLE = "rg_table:v1"
RULE_INLINE_TAG = "inline_tag:v1"
RULE_ARTICLE_CLAUSE = "article_clause:v1"

CONFIDENCE_TABLE_ROW = 0.98
CONFIDENCE_TAG_CONFIRMED = 0.95
CONFIDENCE_TAG_UNCONFIRMED = 0.6
CONFIDENCE_MALFORMED = 0.3
CONFIDENCE_PUBLICATION = 0.95
CONFIDENCE_EFFECTIVE = 0.9
CONFIDENCE_REPEAL = 0.9

_ARTICLE_SECTION_TYPES = frozenset({"madde", "gecici_madde", "ek_madde"})
_APOSTROPHE = "[’']"
_DATE = r"(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})"
_RG_REF_RE = re.compile(r"RG-?\s*" + _DATE + r"\s*-?\s*(?P<number>\d+)")
_TAG_RE = re.compile(
    r"\((?P<kind>[^():\n]{0,60}?(?:Değişik|Ek|Mülga)[^():\n]{0,40}?):\s*"
    r"(?P<rg>RG-?\s*\d{1,2}/\d{1,2}/\d{4}\s*-?\s*\d+)\)"
)
_TABLE_HEAD_RE = re.compile(r"Yönetmeliğin Yayımlandığı Resmî Gazete" + _APOSTROPHE + "nin")
# Horizontal whitespace only after the last cell: a quote must never swallow the line break.
_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<ordinal>\d*)\s*\|\s*" + _DATE + r"\s*\|\s*(?P<number>\d+)\s*\|[ \t]*$", re.MULTILINE
)
_REPEAL_RE = re.compile(
    _DATE + r" tarihli ve (?P<number>\d+) sayılı Resmî Gazete" + _APOSTROPHE + r"de yayımlanan "
    r"(?P<title>[^.\n]{5,300}?) yürürlükten kaldırılmıştır\."
)
_EFFECTIVE_RE = re.compile(r"Bu Yönetmelik " + _DATE + r" tarihinde yürürlüğe girer\.")


class RuleExtractionError(ValueError):
    """The text does not carry the structure a rule needs; nothing is guessed."""


@dataclass(frozen=True, slots=True)
class ArticleSpan:
    path: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class InlineTag:
    kind: str
    rg: RgReference
    start: int
    end: int


def _rg_from_match(match: re.Match[str], raw: str) -> RgReference:
    try:
        issue = date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
    except ValueError:
        raise RuleExtractionError("Resmî Gazete reference carries an impossible date") from None
    return RgReference(date=issue, number=match.group("number"), raw=raw.strip())


def article_spans(document_id: str, text: str) -> list[ArticleSpan]:
    """Top-level articles as the section parser sees them, ordered by position."""

    spans = [
        ArticleSpan(
            path=f"{section.section_type}/{section.section_ref}", start=section.start_char, end=section.end_char
        )
        for section in extract_document_sections(document_id, text)
        if section.section_type in _ARTICLE_SECTION_TYPES
    ]
    return sorted(spans, key=lambda span: span.start)


def article_for(offset: int, spans: list[ArticleSpan]) -> str | None:
    starts = [span.start for span in spans]
    index = bisect.bisect_right(starts, offset) - 1
    if index < 0:
        return None
    span = spans[index]
    return span.path if offset < span.end else None


def extract_rg_table(text: str) -> RgTable:
    """Parse the Resmî Gazete table: the original row, then numbered amendment rows."""

    head = _TABLE_HEAD_RE.search(text)
    if head is None:
        raise RuleExtractionError("Resmî Gazete table heading not found")
    original: RgTableRow | None = None
    amendments: list[RgTableRow] = []
    for match in _TABLE_ROW_RE.finditer(text, head.start()):
        rg = _rg_from_match(
            match, f"{match.group('day')}/{match.group('month')}/{match.group('year')}-{match.group('number')}"
        )
        evidence = TextEvidence.from_span(text, match.start(), match.end())
        ordinal = match.group("ordinal")
        if not ordinal:
            if original is not None:
                raise RuleExtractionError("Resmî Gazete table has more than one original publication row")
            original = RgTableRow(ordinal=None, rg=rg, evidence=evidence)
        else:
            amendments.append(RgTableRow(ordinal=int(ordinal), rg=rg, evidence=evidence))
    if original is None:
        raise RuleExtractionError("Resmî Gazete table has no original publication row")
    return RgTable(original=original, amendments=tuple(amendments))


def extract_inline_tags(text: str) -> list[InlineTag]:
    """Every amendment tag with its exact span; a glued footnote marker stays outside the quote."""

    tags: list[InlineTag] = []
    for match in _TAG_RE.finditer(text):
        rg_match = _RG_REF_RE.search(match.group("rg"))
        if rg_match is None:
            continue
        tags.append(
            InlineTag(
                kind=" ".join(match.group("kind").split()),
                rg=_rg_from_match(rg_match, match.group("rg")),
                start=match.start(),
                end=match.end(),
            )
        )
    return tags


def _confidence_and_flags(rg: RgReference, table_keys: frozenset[str]) -> tuple[float, tuple[ProposalFlag, ...]]:
    flags: list[ProposalFlag] = []
    if rg.key() in table_keys:
        flags.append(ProposalFlag.CROSS_CONFIRMED)
        confidence = CONFIDENCE_TAG_CONFIRMED
    else:
        flags.append(ProposalFlag.NOT_IN_RG_TABLE)
        confidence = CONFIDENCE_TAG_UNCONFIRMED
    if rg.is_malformed:
        flags.append(ProposalFlag.RG_NUMBER_MALFORMED)
        confidence = min(confidence, CONFIDENCE_MALFORMED)
    return confidence, tuple(flags)


def _proposal(
    *,
    document_id: str,
    content_hash: str,
    rule_id: str,
    kind: ProposalKind,
    link_type: str | None,
    event_type: str | None,
    subject_provision: str | None,
    target: ProposalTarget,
    evidence: TextEvidence,
    confidence: float,
    flags: tuple[ProposalFlag, ...],
    preview_tr: str,
    change_kind: str | None = None,
) -> Proposal:
    return Proposal(
        proposal_id=proposal_id_for(
            document_id=document_id,
            content_hash=content_hash,
            rule_id=rule_id,
            kind=kind,
            link_type=link_type,
            event_type=event_type,
            subject_provision=subject_provision,
            target_key=target.key(),
            start_char=evidence.start_char,
            end_char=evidence.end_char,
        ),
        kind=kind,
        source=ProposalSource.RULE,
        rule_id=rule_id,
        link_type=link_type,
        event_type=event_type,
        change_kind=change_kind,
        subject_document_id=document_id,
        subject_provision=subject_provision,
        target=target,
        evidence=evidence,
        confidence=confidence,
        flags=flags,
        preview_tr=preview_tr,
    )


def build_proposal_file(
    *,
    document_id: str,
    content_hash: str,
    title: str,
    text: str,
    ontology_spec_sha256: str,
) -> ProposalFile:
    """Run all three rules over one stored text and return the reviewable proposal set."""

    if sha256_text(text) != content_hash:
        raise RuleExtractionError("stored text does not match its content hash; refusing to extract")
    table = extract_rg_table(text)
    table_keys = table.amendment_keys()
    tags = extract_inline_tags(text)
    tag_keys = frozenset(tag.rg.key() for tag in tags)
    spans = article_spans(document_id, text)
    proposals: list[Proposal] = []

    # (a) Resmî Gazete table: one amends edge per amending regulation, plus the publication event.
    for row in table.amendments:
        echoed = row.rg.key() in tag_keys
        flags: tuple[ProposalFlag, ...] = (ProposalFlag.CROSS_CONFIRMED,) if echoed else ()
        if row.rg.is_malformed:
            flags = (*flags, ProposalFlag.RG_NUMBER_MALFORMED)
        proposals.append(
            _proposal(
                document_id=document_id,
                content_hash=content_hash,
                rule_id=RULE_RG_TABLE,
                kind=ProposalKind.LINK,
                link_type="amends",
                event_type=None,
                subject_provision=None,
                target=ProposalTarget(kind=TargetKind.RG_INSTRUMENT, rg=row.rg),
                evidence=row.evidence,
                confidence=CONFIDENCE_MALFORMED if row.rg.is_malformed else CONFIDENCE_TABLE_ROW,
                flags=flags,
                preview_tr=(
                    f"{row.rg.label_tr()} değişiklik yönetmeliğini {document_id} değişiklik zincirine ekler"
                    f" (amends, tablo satırı {row.ordinal})."
                ),
            )
        )
    publication_flags = (ProposalFlag.NEEDS_REVIEW,)
    if table.original.rg.is_malformed:
        publication_flags = (*publication_flags, ProposalFlag.RG_NUMBER_MALFORMED)
    proposals.append(
        _proposal(
            document_id=document_id,
            content_hash=content_hash,
            rule_id=RULE_RG_TABLE,
            kind=ProposalKind.EVENT,
            link_type=None,
            event_type="publication",
            subject_provision=None,
            target=ProposalTarget(kind=TargetKind.DATE, value_date=table.original.rg.date),
            evidence=table.original.evidence,
            confidence=CONFIDENCE_MALFORMED if table.original.rg.is_malformed else CONFIDENCE_PUBLICATION,
            flags=publication_flags,
            preview_tr=f"Yayım olayı: {table.original.rg.label_tr()}; sürümün publication kanıtı olur.",
        )
    )

    # (b) Inline tags: amends edges that point at the article carrying the tag.
    for tag in tags:
        provision = article_for(tag.start, spans)
        confidence, flags = _confidence_and_flags(tag.rg, table_keys)
        proposals.append(
            _proposal(
                document_id=document_id,
                content_hash=content_hash,
                rule_id=RULE_INLINE_TAG,
                kind=ProposalKind.LINK,
                link_type="amends",
                event_type=None,
                subject_provision=provision,
                target=ProposalTarget(kind=TargetKind.RG_INSTRUMENT, rg=tag.rg),
                evidence=TextEvidence.from_span(text, tag.start, tag.end),
                confidence=confidence,
                flags=flags,
                change_kind=tag.kind,
                preview_tr=(
                    f"{tag.rg.label_tr()} yönetmeliği → {provision_label_tr(provision)} ({tag.kind});"
                    " değişiklik zincirine madde düzeyinde ilişki ekler."
                ),
            )
        )

    # (c) Article clauses: the repeal of another regulation and the entry-into-force date.
    for match in _REPEAL_RE.finditer(text):
        rg = _rg_from_match(
            match, f"{match.group('day')}/{match.group('month')}/{match.group('year')}-{match.group('number')}"
        )
        provision = article_for(match.start(), spans)
        external_ref = (
            f"{match.group('day')}/{match.group('month')}/{match.group('year')} tarihli ve {match.group('number')} sayılı"
            f" Resmî Gazete'de yayımlanan {' '.join(match.group('title').split())}"
        )
        flags = (ProposalFlag.NEEDS_REVIEW,)
        if rg.is_malformed:
            flags = (*flags, ProposalFlag.RG_NUMBER_MALFORMED)
        proposals.append(
            _proposal(
                document_id=document_id,
                content_hash=content_hash,
                rule_id=RULE_ARTICLE_CLAUSE,
                kind=ProposalKind.LINK,
                link_type="repeals",
                event_type=None,
                subject_provision=provision,
                target=ProposalTarget(kind=TargetKind.EXTERNAL_REF, external_ref=external_ref),
                evidence=TextEvidence.from_span(text, match.start(), match.end()),
                confidence=CONFIDENCE_MALFORMED if rg.is_malformed else CONFIDENCE_REPEAL,
                flags=flags,
                preview_tr=(
                    f"{provision_label_tr(provision)} → {rg.label_tr()} ile yayımlanan yönetmeliği yürürlükten kaldırır"
                    " (repeals, dış referans: hedef korpusta yok)."
                ),
            )
        )
    for match in _EFFECTIVE_RE.finditer(text):
        try:
            effective_on = date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
        except ValueError:
            raise RuleExtractionError("entry-into-force clause carries an impossible date") from None
        provision = article_for(match.start(), spans)
        proposals.append(
            _proposal(
                document_id=document_id,
                content_hash=content_hash,
                rule_id=RULE_ARTICLE_CLAUSE,
                kind=ProposalKind.EVENT,
                link_type=None,
                event_type="effective",
                subject_provision=provision,
                target=ProposalTarget(kind=TargetKind.DATE, value_date=effective_on),
                evidence=TextEvidence.from_span(text, match.start(), match.end()),
                confidence=CONFIDENCE_EFFECTIVE,
                flags=(ProposalFlag.NEEDS_REVIEW,),
                preview_tr=(
                    f"Yürürlük olayı: {effective_on.day}/{effective_on.month}/{effective_on.year}"
                    f" ({provision_label_tr(provision)}); sürümün effective kanıtı olur."
                ),
            )
        )

    ordered = tuple(sorted(proposals, key=lambda item: (item.evidence.start_char, item.proposal_id)))
    counts: ProposalCounts = counts_for(ordered)
    return ProposalFile(
        schema_version=PROPOSAL_SCHEMA_VERSION,
        generator=GENERATOR,
        ontology_spec_sha256=ontology_spec_sha256,
        document_id=document_id,
        content_hash=content_hash,
        title=title,
        rg_table=table,
        proposals=ordered,
        counts=counts,
    )
