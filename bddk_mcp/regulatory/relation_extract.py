"""Regex extraction of cross-reference candidates from Turkish legal text.

Query parsing lives in ``bddk_mcp/store/legal_ref.py``; this module is its
document-body counterpart. Patterns favor precision: anything ambiguous
becomes an external-ref candidate for later resolution, never a guess.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    EvidenceReference,
    ValidationRecord,
    ValidationState,
    evidence_id_for,
)
from bddk_mcp.regulatory.relations import RegulatoryRelation, import_relations, make_relation

EXTRACTION_METHOD = "regex:v1"

# Ordinal suffixes: "93 üncü", "12 nci", "9 uncu"; article ids like 26/A.
_ARTICLE = r"(?P<article>\d+(?:/[A-ZÇĞİÖŞÜ])?)\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü|üncu)?"
# Instrument mention: "5411 sayılı Bankacılık Kanunu(nun)" or "Aynı Yönetmeliğin".
# Instrument nouns appear with Turkish consonant mutation (Yönetmelik → Yönetmeliğin).
_MENTION = r"(?P<mention>(?:\d{3,5}\s+sayılı\s+)?[^.]{0,80}?(?:Kanun|Yönetmeli[kğ]|Tebli[ğg]|Rehber|Genelge)\w*)"

# Ordered highest-precision first: overlap dedup lets earlier patterns claim spans.
_PATTERNS: tuple[tuple[str, str, float, re.Pattern[str]], ...] = (
    (
        "amends",
        "amend-degistirilmistir",
        0.9,
        re.compile(
            _MENTION + r"\s+" + _ARTICLE + r"\s+maddesi[^.]{0,120}?değiştirilmiştir",
            re.IGNORECASE,
        ),
    ),
    (
        "repeals",
        "repeal-kaldirilmistir",
        0.9,
        re.compile(
            _MENTION + r"\s+" + _ARTICLE + r"\s+maddesi\s+yürürlükten\s+kaldırılmıştır",
            re.IGNORECASE,
        ),
    ),
    (
        "implements",
        "implements-dayanilarak",
        0.85,
        re.compile(
            _MENTION + r"\s+" + _ARTICLE + r"\s+maddesine\s+dayanılarak",
            re.IGNORECASE,
        ),
    ),
    (
        "exception_to",
        "exception-saklidir",
        0.75,
        re.compile(
            _ARTICLE + r"\s+(?:uncu\s+|üncü\s+|inci\s+|ıncı\s+)?madde\s+hükümleri\s+saklıdır",
            re.IGNORECASE,
        ),
    ),
    (
        "cites",
        "cites-sayili",
        0.7,
        re.compile(
            r"(?P<mention>\d{3,5}\s+sayılı\s+[^.]{3,80}?Kanun\w*)(?!\w)"
            r"(?![^.]{0,120}?(?:değiştirilmiştir|dayanılarak|yürürlükten))",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class CandidateRelation:
    relation_type: str
    target_mention: str
    target_article: str | None
    span: tuple[int, int]
    pattern_id: str
    confidence: float


def extract_candidate_relations(text: str) -> list[CandidateRelation]:
    """Scan text; return candidates ordered by position, deduplicated by span+type."""
    candidates: list[CandidateRelation] = []
    claimed: set[tuple[int, int]] = set()
    for relation_type, pattern_id, confidence, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(_overlaps(span, other) for other in claimed):
                continue
            groups = match.groupdict()
            candidates.append(
                CandidateRelation(
                    relation_type=relation_type,
                    target_mention=(groups.get("mention") or match.group(0)).strip(),
                    target_article=groups.get("article"),
                    span=span,
                    pattern_id=pattern_id,
                    confidence=confidence,
                )
            )
            claimed.add(span)
    return sorted(candidates, key=lambda c: c.span)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _resolve_target_instrument(number: str, rows: Sequence[Any]) -> str | None:
    """Digit-boundary match of a cited instrument number against identity keys.

    Spec §4: never fuzzy — "541" must not resolve to ``kanun:5411`` (the number
    must not be a substring of a longer digit run), and an ambiguous number
    (matching more than one instrument) fails closed so the caller falls back
    to ``target_external_ref``. Rows carry ``instrument_id`` and ``identity_key``.
    """
    boundary = re.compile(rf"(?<!\d){re.escape(number)}(?!\d)")
    matched = {row["instrument_id"] for row in rows if boundary.search(row["identity_key"])}
    if len(matched) == 1:
        return next(iter(matched))
    return None


async def resolve_and_import(
    pool: Any,
    *,
    doc_id: str,
    artifact_id: str,
    candidates: list[CandidateRelation],
    imported_by: str,
) -> tuple[int, int]:
    """Resolve candidate targets against known instruments; import as edges.

    Source instrument is resolved from doc_id via the artifact chain. Targets
    resolve by cited-number match against identity_key on digit boundaries
    only, and only when exactly one instrument matches; everything else stays
    external. Returns (resolved_count, external_count).
    """
    source = await pool.fetchrow(
        """
        SELECT lv.instrument_id
        FROM regulatory_source_artifacts a
        JOIN regulatory_legal_version_artifacts lva ON lva.artifact_id = a.artifact_id
        JOIN regulatory_legal_versions lv ON lv.legal_version_id = lva.legal_version_id
        WHERE a.repository_document_id = $1
        LIMIT 1
        """,
        doc_id,
    )
    if source is None:
        return (0, 0)
    source_instrument_id = source["instrument_id"]

    # Machine extraction never counts as review: candidates land unvalidated
    # and stay invisible to the serving views until a human verdict.
    unvalidated = ValidationRecord(state=ValidationState.UNVALIDATED)
    relations: list[RegulatoryRelation] = []
    resolved = external = 0
    for candidate in candidates:
        number_match = re.search(r"\b(\d{3,5})\b", candidate.target_mention)
        target_instrument_id = None
        if number_match:
            number = number_match.group(1)
            rows = await pool.fetch(
                "SELECT instrument_id, identity_key FROM regulatory_instruments"
                " WHERE identity_key LIKE '%' || $1 || '%'",
                number,
            )
            target_instrument_id = _resolve_target_instrument(number, rows)
        external_ref = None if target_instrument_id else candidate.target_mention
        if target_instrument_id:
            resolved += 1
        else:
            external += 1
        statement = f"{doc_id}:{candidate.span[0]}-{candidate.span[1]}:{candidate.target_mention}"
        statement_sha256 = hashlib.sha256(statement.encode("utf-8")).hexdigest()
        locator = f"chars={candidate.span[0]}-{candidate.span[1]}"
        evidence = EvidenceReference(
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
        relations.append(
            make_relation(
                relation_type=candidate.relation_type,
                source_instrument_id=source_instrument_id,
                target_instrument_id=target_instrument_id,
                target_external_ref=external_ref,
                evidence=evidence,
                extraction_method=EXTRACTION_METHOD,
                confidence=candidate.confidence,
                validation=unvalidated,
            )
        )
    await import_relations(pool, relations, imported_by=imported_by)
    return (resolved, external)


async def extract_relations_batch(
    pool: Any,
    items: Sequence[tuple[str, str, str]],
    *,
    imported_by: str,
) -> dict:
    """Operator batch flow over (doc_id, artifact_id, text) items.

    Per-document failures are logged and skipped; the batch never aborts
    (spec §6). Content loading is the caller's job (operator script /
    doc_store), keeping this function free of documents-table coupling.
    """
    logger = logging.getLogger(__name__)
    summary = {"processed": 0, "failed": 0, "resolved": 0, "external": 0}
    for doc_id, artifact_id, text in items:
        try:
            candidates = extract_candidate_relations(text)
            resolved, external = await resolve_and_import(
                pool,
                doc_id=doc_id,
                artifact_id=artifact_id,
                candidates=candidates,
                imported_by=imported_by,
            )
            summary["processed"] += 1
            summary["resolved"] += resolved
            summary["external"] += external
        except Exception:  # noqa: BLE001 — spec §6: one bad document must not stop the batch
            summary["failed"] += 1
            logger.exception("relation extraction failed for %s; continuing", doc_id)
    return summary
