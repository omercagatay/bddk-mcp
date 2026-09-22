"""Verbatim matching is an exact string check, never a semantic/summarization check."""

import hashlib

import pytest


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "changed",
    [
        "Banka  yükümlülüğü yerine getirir.",
        "Banka\nyükümlülüğü yerine getirir.",
        "banka yükümlülüğü yerine getirir.",
        "Banka yükümlülüğü yerine getirir!",
        "Banka yükümlülüklerini yerine getirir.",
        "Banka sorumluluğunu yerine getirir.",
    ],
)
def test_no_whitespace_case_punctuation_word_or_semantic_normalization(changed):
    from bddk_mcp.quotations import check_exact_quotation

    reference = "Banka yükümlülüğü yerine getirir."
    result = check_exact_quotation(reference, changed, expected_reference_sha256=sha(reference))
    assert result.status == "mismatch"
    assert result.start_char is None and result.end_char is None
    assert result.normalization == "none"


def test_exact_excerpt_has_reconstructable_positions_and_hashes():
    from bddk_mcp.quotations import check_exact_quotation

    reference = "Önce. İkinci hüküm. Sonra."
    quote = "İkinci hüküm."
    result = check_exact_quotation(reference, quote, expected_reference_sha256=sha(reference))
    assert result.status == "exact_reference_match"
    assert reference[result.start_char : result.end_char] == quote
    assert result.reference_sha256 == sha(reference)
    assert result.quotation_sha256 == sha(quote)
    assert result.extent == "excerpt"
    assert result.original_source_fidelity == "not_established"


def test_full_reference_cannot_be_substituted_with_partial_text():
    from bddk_mcp.quotations import check_exact_quotation

    reference = "İstisna hariç uygulanır."
    full = check_exact_quotation(reference, reference, expected_reference_sha256=sha(reference), require_complete=True)
    assert full.status == "exact_reference_match" and full.extent == "whole_reference"
    partial = check_exact_quotation(
        reference, "uygulanır.", expected_reference_sha256=sha(reference), require_complete=True
    )
    assert partial.status == "incomplete"


def test_missing_or_tampered_reference_is_unavailable_not_verified():
    from bddk_mcp.quotations import check_exact_quotation

    assert check_exact_quotation(None, "metin", expected_reference_sha256=None).status == "unavailable"
    assert check_exact_quotation("metin", "metin", expected_reference_sha256=None).status == "unavailable"
    assert (
        check_exact_quotation("değişmiş metin", "metin", expected_reference_sha256=sha("metin")).status == "unavailable"
    )


@pytest.mark.parametrize("quote", ["", " ", "\n\t"])
def test_empty_or_whitespace_only_input_is_not_a_quote(quote):
    from bddk_mcp.quotations import check_exact_quotation

    reference = "  Metin.\n\t"
    assert check_exact_quotation(reference, quote, expected_reference_sha256=sha(reference)).status == "unavailable"


def test_repeated_match_does_not_invent_a_unique_location():
    from bddk_mcp.quotations import check_exact_quotation

    reference = "İstisna. Başka istisna. İstisna."
    result = check_exact_quotation(reference, "İstisna.", expected_reference_sha256=sha(reference))
    assert result.status == "ambiguous"
    assert result.start_char is None and result.end_char is None


def test_unicode_equivalence_is_not_character_equality():
    from bddk_mcp.quotations import check_exact_quotation

    reference = "İlgili hüküm."
    decomposed = "I\u0307lgili hüküm."
    assert check_exact_quotation(reference, decomposed, expected_reference_sha256=sha(reference)).status == "mismatch"


def test_public_quotation_argument_preserves_boundary_whitespace():
    from pydantic import TypeAdapter

    from bddk_mcp.tools.contract_types import OptionalQuotation

    quote = "  Birebir metin.\n"
    assert TypeAdapter(OptionalQuotation).validate_python(quote) == quote


def test_stricter_quotation_wire_semantics_have_an_explicit_output_version():
    from bddk_mcp.tools.structured_outputs import SCHEMA_VERSION, RegulationStatusResponse, RetrievalResponse

    assert SCHEMA_VERSION == "2.0"
    for model in (RetrievalResponse, RegulationStatusResponse):
        assert model.model_json_schema()["properties"]["schema_version"]["const"] == "2.0"
