"""Citation v1 reconstruction and fail-closed mismatch tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from bddk_mcp.citations import (
    CitationQuality,
    CitationV1,
    NormalizedTextRange,
    TrustedCitationContext,
    build_normalized_range_citation,
    citation_id_for,
    section_retrieval_profile_sha256,
    verify_normalized_range_citation,
)
from bddk_mcp.quality.markdown_quality import sanitize_markdown_for_context
from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    artifact_id_for,
    blob_id_for,
    evidence_id_for,
    instrument_id_for,
    legal_version_id_for,
    provision_id_for,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fixture(
    *,
    suffix: str = "Sonraki sentetik bölüm.\n",
    source_range: str = "  **MADDE 5** - Sentetik başlık\n(1) Yalnızca test hükmü.\n\n",
):
    prefix = "Sentetik test belgesi\n\n"
    normalized_document = prefix + source_range + suffix
    provision_text = source_range.strip()
    rendered_excerpt = sanitize_markdown_for_context(provision_text)
    start = len(prefix)
    end = start + len(source_range)

    jurisdiction = "ZZ"
    authority_code = "SYNTHETIC_TEST_AUTHORITY"
    identity_key = "citation-contract-only"
    instrument_id = instrument_id_for(
        jurisdiction=jurisdiction,
        authority_code=authority_code,
        identity_key=identity_key,
    )
    normalized_document_sha256 = _sha(normalized_document)
    version_key = "synthetic-citation-v1"
    legal_version_id = legal_version_id_for(
        instrument_id=instrument_id,
        version_key=version_key,
        legal_text_sha256=normalized_document_sha256,
    )
    artifact_sha256 = _sha("synthetic artifact bytes identity")
    artifact_blob_id = blob_id_for(content_sha256=artifact_sha256)
    source_url = "https://regulator.example.test/synthetic/one.pdf"
    retrieved_at = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    artifact_id = artifact_id_for(
        blob_id=artifact_blob_id,
        canonical_uri=source_url,
        retrieved_at=retrieved_at,
    )
    provision_kind = "madde"
    provision_path = "madde/5"
    provision_id = provision_id_for(
        instrument_id=instrument_id,
        kind=provision_kind,
        canonical_path=provision_path,
    )
    statement_sha256 = _sha(provision_text)
    evidence_locator = "normalized/madde/5"
    evidence_id = evidence_id_for(
        artifact_id=artifact_id,
        locator=evidence_locator,
        statement_sha256=statement_sha256,
        authority_level=AuthorityLevel.AUTHORITATIVE,
    )
    quality = CitationQuality(label="clean")
    locator = NormalizedTextRange(
        start_char=start,
        end_char=end,
        normalized_range_sha256=_sha(source_range),
    )
    trusted = TrustedCitationContext(
        instrument_id=instrument_id,
        instrument_jurisdiction=jurisdiction,
        instrument_authority_code=authority_code,
        instrument_identity_key=identity_key,
        legal_version_id=legal_version_id,
        legal_version_key=version_key,
        legal_validation_record_sha256="6" * 64,
        provision_validation_record_sha256="7" * 64,
        artifact_id=artifact_id,
        artifact_blob_id=artifact_blob_id,
        artifact_sha256=artifact_sha256,
        source_url=source_url,
        artifact_retrieved_at=retrieved_at,
        source_document_id="synthetic-citation-one",
        normalized_document_sha256=normalized_document_sha256,
        evidence_id=evidence_id,
        evidence_locator=evidence_locator,
        evidence_statement_sha256=statement_sha256,
        provision_id=provision_id,
        provision_kind=provision_kind,
        provision_path=provision_path,
        provision_text_sha256=statement_sha256,
        locator=locator,
        excerpt_sha256=_sha(rendered_excerpt),
        excerpt_length=len(rendered_excerpt),
        retrieval_profile_sha256=section_retrieval_profile_sha256(),
        quality=quality,
    )
    citation = build_normalized_range_citation(
        trusted=trusted,
        provision_text=provision_text,
        normalized_source_range=source_range,
        rendered_excerpt=rendered_excerpt,
        generated_at=datetime(2026, 7, 15, 9, 0, tzinfo=UTC),
    )
    return citation, trusted, normalized_document, rendered_excerpt


def test_normalized_range_citation_reconstructs_exact_excerpt():
    citation, trusted, normalized_document, rendered_excerpt = _fixture()

    result = verify_normalized_range_citation(
        citation,
        normalized_document=normalized_document,
        rendered_excerpt=rendered_excerpt,
        expected=trusted,
    )

    assert result.valid is True
    assert result.failure_codes == ()
    assert citation.locator.kind == "normalized_text_range"
    assert "page" not in citation.locator.model_json_schema()["properties"]


def test_forged_self_consistent_document_fails_against_independent_trusted_context():
    _, original_trust, _, _ = _fixture()
    forged, _, forged_document, forged_excerpt = _fixture(suffix="Başka, kendi içinde tutarlı sentetik belge.\n")

    result = verify_normalized_range_citation(
        forged,
        normalized_document=forged_document,
        rendered_excerpt=forged_excerpt,
        expected=original_trust,
    )

    assert result.valid is False
    assert "legal_version_id_mismatch" in result.failure_codes
    assert "normalized_document_sha256_mismatch" in result.failure_codes


def test_wrong_legal_version_fails_against_independent_expected_identity():
    citation, trusted, normalized_document, rendered_excerpt = _fixture()
    wrong_expected = trusted.model_copy(update={"legal_version_id": "ver_sha256_" + "9" * 64})

    result = verify_normalized_range_citation(
        citation,
        normalized_document=normalized_document,
        rendered_excerpt=rendered_excerpt,
        expected=wrong_expected,
    )

    assert result.valid is False
    assert "legal_version_id_mismatch" in result.failure_codes


def test_wrong_occurrence_review_record_fails_against_independent_trust():
    citation, trusted, normalized_document, rendered_excerpt = _fixture()
    wrong_expected = trusted.model_copy(update={"provision_validation_record_sha256": "8" * 64})

    result = verify_normalized_range_citation(
        citation,
        normalized_document=normalized_document,
        rendered_excerpt=rendered_excerpt,
        expected=wrong_expected,
    )

    assert result.valid is False
    assert "provision_validation_record_sha256_mismatch" in result.failure_codes


def test_wrong_hash_and_range_fail_reconstruction_without_returning_source_text():
    citation, trusted, normalized_document, rendered_excerpt = _fixture()
    moved_locator = citation.locator.model_copy(
        update={"start_char": citation.locator.start_char + 1, "end_char": citation.locator.end_char + 1}
    )
    tampered = citation.model_copy(update={"locator": moved_locator})

    result = verify_normalized_range_citation(
        tampered,
        normalized_document=normalized_document + "tampered",
        rendered_excerpt=rendered_excerpt + "x",
        expected=trusted,
    )

    assert result.valid is False
    assert {
        "citation_id_mismatch",
        "normalized_document_sha256_mismatch",
        "normalized_range_sha256_mismatch",
        "excerpt_sha256_mismatch",
    }.issubset(result.failure_codes)
    assert "Sentetik" not in result.model_dump_json()


def test_builder_refuses_non_reconstructable_or_truncated_section_text():
    _, trusted, _, rendered_excerpt = _fixture()

    with pytest.raises(ValueError, match="stored provision cannot be reconstructed"):
        build_normalized_range_citation(
            trusted=trusted,
            provision_text="different or parser-truncated content",
            normalized_source_range="source text",
            rendered_excerpt=rendered_excerpt,
        )


def test_citation_identity_is_stable_across_request_time_but_profile_is_pinned():
    citation, _, _, _ = _fixture()
    later = citation.model_copy(update={"generated_at": citation.generated_at + timedelta(minutes=10)})
    assert citation_id_for(later) == citation.citation_id
    assert section_retrieval_profile_sha256() == "58b3dafba8b690e12386ef39a7f7f79b11002a1e39fb677097f828a1d019cbf6"

    payload = citation.model_dump(mode="json")
    payload["retrieval_profile_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="retrieval profile"):
        CitationV1.model_validate(payload)


def test_citation_schema_rejects_page_claims_unsafe_url_and_unstable_flags():
    citation, _, _, _ = _fixture()
    payload = citation.model_dump(mode="json")
    payload["locator"]["source_page"] = 5
    with pytest.raises(ValidationError):
        CitationV1.model_validate(payload)

    payload = citation.model_dump(mode="json")
    payload["source_url"] = "https://regulator.example.test/source.pdf?access_token=sensitive"
    with pytest.raises(ValidationError, match="sensitive query"):
        CitationV1.model_validate(payload)

    with pytest.raises(ValidationError, match="canonically ordered"):
        CitationQuality(label="warning", flags=("z_flag", "a_flag"))


@pytest.mark.parametrize(
    "source_range",
    [
        "  MADDE 5 - İ\u0307çerik\r\nAstral: \U0001f4da\r\n",
        "  MADDE 5 - İçerik\nAstral: \U0001f4da\n",
    ],
)
def test_code_point_ranges_preserve_stored_unicode_and_line_endings(source_range: str):
    citation, trusted, normalized_document, rendered_excerpt = _fixture(source_range=source_range)

    result = verify_normalized_range_citation(
        citation,
        normalized_document=normalized_document,
        rendered_excerpt=rendered_excerpt,
        expected=trusted,
    )

    assert result.valid is True
    assert citation.locator.end_char - citation.locator.start_char == len(source_range)
    assert normalized_document[citation.locator.start_char : citation.locator.end_char] == source_range
