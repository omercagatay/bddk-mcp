"""Exact quotation checks against a hash-bound reference, without text normalization.

A reference match does not establish fidelity of an extraction to its original
HTML/PDF/image, legal status, completeness of that original, or semantic entailment.
Public callers must retrieve the reference and its digest from trusted storage;
they must not accept a client-supplied reference as its own verification evidence.

The status is always the character-exact result.  Two optional diagnostics sit
next to it and never change it: the offsets of every occurrence of an ambiguous
quotation (capped), and, for a quotation that is not found, the one stored place
it matches after only these differences are ignored: runs of whitespace and line
breaks, typographic vs ASCII apostrophes and quotation marks, asterisk or
underscore markers (``**`` and ``__``: Markdown emphasis, or footnote markers and
form blanks in the source) and soft hyphens.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_LISTED_OCCURRENCES = 5
NearMatchDifference = Literal[
    "whitespace", "apostrophes_or_quotation_marks", "asterisk_or_underscore_markers", "soft_hyphens"
]
_NEAR_MATCH_DIFFERENCES: tuple[NearMatchDifference, ...] = (
    "whitespace",
    "apostrophes_or_quotation_marks",
    "asterisk_or_underscore_markers",
    "soft_hyphens",
)
_TYPOGRAPHIC_MARKS = {"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"', "„": '"'}
_EMPHASIS_MARKERS = ("**", "__")
_SOFT_HYPHEN = "\u00ad"


class QuotationSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_char: int = Field(ge=0, description="Unicode offset in the supplied reference text.")
    end_char: int = Field(ge=0, description="Exclusive Unicode offset in that same reference.")


class NearQuotationMatch(BaseModel):
    """The one stored place a not-found quotation matches once the listed differences are ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    differences: list[NearMatchDifference] = Field(
        description="The only kinds of difference between the quotation and the stored characters."
    )
    start_char: int = Field(ge=0, description="Unicode offset of the stored characters in the reference text.")
    end_char: int = Field(ge=0, description="Exclusive Unicode offset of the stored characters.")
    stored_text: str = Field(description="The stored characters at that place, exactly; quote these.")


class ExactQuotationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["exact_reference_match", "mismatch", "incomplete", "ambiguous", "unavailable"]
    reason: str
    reference_sha256: str | None = None
    quotation_sha256: str | None = None
    start_char: int | None = Field(default=None, ge=0, description="Unicode offset in the supplied reference text.")
    end_char: int | None = Field(default=None, ge=0, description="Exclusive Unicode offset in that same reference.")
    extent: Literal["whole_reference", "excerpt"] | None = None
    normalization: Literal["none"] = "none"
    original_source_fidelity: Literal["not_established"] = "not_established"
    legal_applicability: Literal["not_assessed"] = "not_assessed"
    occurrence_count: int | None = Field(
        default=None, ge=2, description="For an ambiguous quotation: how many times it occurs exactly."
    )
    occurrences: list[QuotationSpan] | None = Field(
        default=None,
        description=f"For an ambiguous quotation: the first {MAX_LISTED_OCCURRENCES} exact occurrences.",
    )
    near_match_places: Literal["one", "several"] | None = Field(
        default=None,
        description=(
            "For a quotation not found exactly: whether it matches at one or at several stored places once "
            "only whitespace, apostrophe/quotation-mark, asterisk/underscore-marker (Markdown emphasis or "
            "footnote/blank markers) and soft-hyphen differences are "
            "ignored. None when it matches nowhere even then. The status stays not found either way."
        ),
    )
    near_match: NearQuotationMatch | None = Field(
        default=None, description="The stored characters at that place, given only when it is exactly one place."
    )


def _normalised_with_map(text: str, *, skip: str | None = None) -> tuple[str, list[int], list[int]]:
    """Text with the near-match differences removed, and for each kept character its raw [start, end).

    ``skip`` names one difference that is left as it is (used to tell which differences matter).
    """
    kept: list[tuple[str, int]] = []
    index = 0
    while index < len(text):
        if skip != "asterisk_or_underscore_markers" and text.startswith(_EMPHASIS_MARKERS, index):
            index += 2
            continue
        char = text[index]
        if not (skip != "soft_hyphens" and char == _SOFT_HYPHEN):
            if skip != "apostrophes_or_quotation_marks":
                char = _TYPOGRAPHIC_MARKS.get(char, char)
            kept.append((char, index))
        index += 1
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    collapse = skip != "whitespace"
    for char, raw in kept:
        if collapse and char.isspace():
            if chars and chars[-1] == " " and text[starts[-1]].isspace():
                ends[-1] = raw + 1
                continue
            char = " "
        chars.append(char)
        starts.append(raw)
        ends.append(raw + 1)
    return "".join(chars), starts, ends


def _normalised(text: str, *, skip: str | None = None) -> str:
    return _normalised_with_map(text, skip=skip)[0]


def _balanced_span(reference: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) over adjacent emphasis markers so that the stored span opens and closes each one."""
    for marker in _EMPHASIS_MARKERS:
        if reference.count(marker, start, end) % 2 == 0:
            continue
        if start >= len(marker) and reference.startswith(marker, start - len(marker)):
            start -= len(marker)
        elif reference.startswith(marker, end):
            end += len(marker)
    return start, end


def _near_match(reference: str, quotation: str) -> tuple[Literal["one", "several"] | None, NearQuotationMatch | None]:
    normal_reference, starts, ends = _normalised_with_map(reference)
    normal_quotation = _normalised(quotation)
    if not normal_quotation.strip():
        return None, None
    found = normal_reference.find(normal_quotation)
    if found < 0:
        return None, None
    if normal_reference.find(normal_quotation, found + 1) >= 0:
        return "several", None
    start, end = _balanced_span(reference, starts[found], ends[found + len(normal_quotation) - 1])
    stored = reference[start:end]
    differences = [
        difference
        for difference in _NEAR_MATCH_DIFFERENCES
        if _normalised(stored, skip=difference) != _normalised(quotation, skip=difference)
    ]
    if not differences:
        # Differences that cancel out only in combination; name them all rather than none.
        differences = list(_NEAR_MATCH_DIFFERENCES)
    return "one", NearQuotationMatch(differences=differences, start_char=start, end_char=end, stored_text=stored)


def _occurrences(reference: str, quotation: str, first: int) -> tuple[int, list[QuotationSpan]]:
    count = 0
    listed: list[QuotationSpan] = []
    start = first
    while start >= 0:
        count += 1
        if len(listed) < MAX_LISTED_OCCURRENCES:
            listed.append(QuotationSpan(start_char=start, end_char=start + len(quotation)))
        start = reference.find(quotation, start + 1)
    return count, listed


def check_exact_quotation(
    reference: str | None,
    quotation: str,
    *,
    expected_reference_sha256: str | None,
    require_complete: bool = False,
) -> ExactQuotationCheck:
    """Require exact, uniquely locatable characters, preserving all whitespace.

    `require_complete` additionally requires the entire supplied reference, not
    merely a substring. Offsets are relative to that reference, never PDF pages.
    """
    if reference is None or expected_reference_sha256 is None:
        return ExactQuotationCheck(status="unavailable", reason="reference_unavailable")
    try:
        reference_sha256 = hashlib.sha256(reference.encode("utf-8")).hexdigest()
        quotation_sha256 = hashlib.sha256(quotation.encode("utf-8")).hexdigest()
    except UnicodeError:
        return ExactQuotationCheck(status="unavailable", reason="invalid_unicode")
    hashes = {"reference_sha256": reference_sha256, "quotation_sha256": quotation_sha256}
    if reference_sha256 != expected_reference_sha256:
        return ExactQuotationCheck(status="unavailable", reason="reference_hash_mismatch", **hashes)
    # This rejects empty input only; it does NOT alter the string being compared.
    if not quotation.strip():
        return ExactQuotationCheck(status="unavailable", reason="invalid_quotation", **hashes)
    start = reference.find(quotation)
    if start < 0:
        places, near = _near_match(reference, quotation)
        return ExactQuotationCheck(
            status="mismatch", reason="not_found", near_match_places=places, near_match=near, **hashes
        )
    if reference.find(quotation, start + 1) >= 0:
        count, listed = _occurrences(reference, quotation, start)
        return ExactQuotationCheck(
            status="ambiguous",
            reason="ambiguous_location",
            occurrence_count=count,
            occurrences=listed,
            **hashes,
        )
    complete = quotation == reference
    return ExactQuotationCheck(
        status="incomplete" if require_complete and not complete else "exact_reference_match",
        reason="incomplete_reference" if require_complete and not complete else "matched",
        start_char=start,
        end_char=start + len(quotation),
        extent="whole_reference" if complete else "excerpt",
        **hashes,
    )
