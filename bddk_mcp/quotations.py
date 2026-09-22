"""Exact quotation checks against a hash-bound reference, without text normalization.

A reference match does not establish fidelity of an extraction to its original
HTML/PDF/image, legal status, completeness of that original, or semantic entailment.
Public callers must retrieve the reference and its digest from trusted storage;
they must not accept a client-supplied reference as its own verification evidence.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
        return ExactQuotationCheck(status="mismatch", reason="not_found", **hashes)
    if reference.find(quotation, start + 1) >= 0:
        return ExactQuotationCheck(status="ambiguous", reason="ambiguous_location", **hashes)
    complete = quotation == reference
    return ExactQuotationCheck(
        status="incomplete" if require_complete and not complete else "exact_reference_match",
        reason="incomplete_reference" if require_complete and not complete else "matched",
        start_char=start,
        end_char=start + len(quotation),
        extent="whole_reference" if complete else "excerpt",
        **hashes,
    )
