from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bddk_mcp.citations import (
    CitationQuality,
    NormalizedTextRange,
    TrustedCitationContext,
    build_normalized_range_citation,
    section_retrieval_profile_sha256,
)
from bddk_mcp.corpus_manifest import CorpusArtifact
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
from benchmark.expert_evaluation import (
    EXPERT_EVALUATION_DRAFT_PATH,
    ExpertEvaluationError,
    ExpertEvaluationReleaseError,
    _load_bound_corpus_json,
    canonical_dataset_payload,
    canonical_dataset_sha256,
    canonical_legal_attestation_payload,
    canonical_legal_attestation_sha256,
    load_expert_evaluation_dataset,
    profile_expert_evaluation_dataset,
)


def _raw_dataset() -> dict[str, Any]:
    value = yaml.safe_load(EXPERT_EVALUATION_DRAFT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_sealed_dataset(tmp_path: Path, raw: dict[str, Any]) -> Path:
    value = copy.deepcopy(raw)
    value["integrity"]["dataset_sha256"] = canonical_dataset_sha256(value)
    path = tmp_path / "expert-evaluation.yml"
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_release_index_reader_rehashes_the_exact_bytes_it_uses(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    original = b'[{"document_id":"one"}]'
    path.write_bytes(original)
    artifact = CorpusArtifact(
        role="documents",
        path=path.name,
        sha256=hashlib.sha256(original).hexdigest(),
        bytes=len(original),
        records=1,
    )

    assert _load_bound_corpus_json(tmp_path, artifact) == [{"document_id": "one"}]

    path.write_bytes(b'[{"document_id":"two"}]')
    with pytest.raises(ExpertEvaluationError, match="changed after manifest validation"):
        _load_bound_corpus_json(tmp_path, artifact)


def _verified_tracked_citation(raw: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).parents[1] / "seed_data"
    evidence = raw["evidence_catalog"][0]
    documents = json.loads((root / "documents.json").read_text(encoding="utf-8"))
    chunks = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
    document = next(item for item in documents if item["document_id"] == evidence["document_id"])
    section = next(
        item
        for item in chunks
        if item.get("doc_id") == evidence["document_id"]
        and item.get("section_type") == evidence["section_type"]
        and item.get("section_ref") == evidence["section_ref"]
        and item.get("section_content_hash") == evidence["section_content_sha256"]
    )
    normalized_document = document["markdown_content"]
    start = section["section_start_char"]
    end = section["section_end_char"]
    normalized_range = normalized_document[start:end]
    provision_text = normalized_range.strip()
    assert hashlib.sha256(provision_text.encode()).hexdigest() == evidence["section_content_sha256"]
    rendered_excerpt = sanitize_markdown_for_context(provision_text)
    instrument_identity = "expert-evaluation-test-943"
    version_identity = "expert-evaluation-test-v1"

    instrument_id = instrument_id_for(
        jurisdiction="TR",
        authority_code="BDDK",
        identity_key=instrument_identity,
    )
    legal_version_id = legal_version_id_for(
        instrument_id=instrument_id,
        version_key=version_identity,
        legal_text_sha256=evidence["document_content_sha256"],
    )
    artifact_sha256 = evidence["document_content_sha256"]
    artifact_blob_id = blob_id_for(content_sha256=artifact_sha256)
    retrieved_at = datetime.fromtimestamp(document["downloaded_at"], UTC)
    artifact_id = artifact_id_for(
        blob_id=artifact_blob_id,
        canonical_uri=evidence["source_url"],
        retrieved_at=retrieved_at,
    )
    provision_id = provision_id_for(
        instrument_id=instrument_id,
        kind=evidence["section_type"],
        canonical_path=f"{evidence['section_type']}/{evidence['section_ref']}",
    )
    evidence_locator = f"normalized:{start}:{end}"
    evidence_id = evidence_id_for(
        artifact_id=artifact_id,
        locator=evidence_locator,
        statement_sha256=evidence["section_content_sha256"],
        authority_level=AuthorityLevel.AUTHORITATIVE,
    )
    trusted = TrustedCitationContext(
        instrument_id=instrument_id,
        instrument_jurisdiction="TR",
        instrument_authority_code="BDDK",
        instrument_identity_key=instrument_identity,
        legal_version_id=legal_version_id,
        legal_version_key=version_identity,
        legal_validation_record_sha256="1" * 64,
        provision_validation_record_sha256="2" * 64,
        artifact_id=artifact_id,
        artifact_blob_id=artifact_blob_id,
        artifact_sha256=artifact_sha256,
        evidence_authority="authoritative",
        source_url=evidence["source_url"],
        artifact_retrieved_at=retrieved_at,
        source_document_id=evidence["document_id"],
        normalized_document_sha256=evidence["document_content_sha256"],
        evidence_id=evidence_id,
        evidence_locator=evidence_locator,
        evidence_statement_sha256=evidence["section_content_sha256"],
        provision_id=provision_id,
        provision_kind=evidence["section_type"],
        provision_path=f"{evidence['section_type']}/{evidence['section_ref']}",
        provision_text_sha256=evidence["section_content_sha256"],
        locator=NormalizedTextRange(
            start_char=start,
            end_char=end,
            normalized_range_sha256=hashlib.sha256(normalized_range.encode()).hexdigest(),
        ),
        excerpt_sha256=hashlib.sha256(rendered_excerpt.encode()).hexdigest(),
        excerpt_length=len(rendered_excerpt),
        retrieval_profile_sha256=section_retrieval_profile_sha256(),
        quality=CitationQuality(label="clean"),
    )
    return build_normalized_range_citation(
        trusted=trusted,
        provision_text=provision_text,
        normalized_source_range=normalized_range,
        rendered_excerpt=rendered_excerpt,
        generated_at=datetime(2026, 7, 15, 12, tzinfo=UTC),
    ).model_dump(mode="json")


def test_tracked_pilot_is_complete_but_explicitly_draft() -> None:
    validation = load_expert_evaluation_dataset()
    dataset = validation.dataset

    assert len(dataset.cases) == 20
    assert len({case.case_id for case in dataset.cases}) == 20
    assert len({case.domain for case in dataset.cases}) >= 5
    assert dataset.dataset_version.endswith("-draft.1")
    assert dataset.approval.state == "draft"
    assert all(case.approval.state == "draft" for case in dataset.cases)
    assert all(len(case.annotations) >= 2 for case in dataset.cases)
    assert all(annotation.status == "pending" for case in dataset.cases for annotation in case.annotations)
    assert all(case.adjudication.status == "pending" for case in dataset.cases)

    supported = [case for case in dataset.cases if case.answerability == "supported"]
    abstentions = [case for case in dataset.cases if case.answerability == "abstain"]
    assert len(supported) == 15
    assert len(abstentions) == 5
    assert all(case.positive_evidence_ids and case.hard_negative_evidence_ids for case in supported)
    assert all(case.no_answer is not None for case in abstentions)
    assert all(not case.positive_evidence_ids and not case.hard_negative_evidence_ids for case in abstentions)
    assert all(item.legal_currentness == "not_verified" for item in dataset.evidence_catalog)
    assert all(item.citation_v1_status == "pending_legal_mapping" for item in dataset.evidence_catalog)
    assert all(item.citation_v1_id is None for item in dataset.evidence_catalog)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda raw: raw["cases"][0].pop("query"), "schema validation failed"),
        (
            lambda raw: raw["cases"][1].update(case_id=raw["cases"][0]["case_id"]),
            "schema validation failed",
        ),
        (
            lambda raw: raw["cases"][0].update(domain="not_a_banking_domain"),
            "schema validation failed",
        ),
        (
            lambda raw: raw["cases"][0].update(query_class="free_form_guess"),
            "schema validation failed",
        ),
        (
            lambda raw: raw["cases"][0].update(positive_evidence_ids=["ev-missing-reference"]),
            "schema validation failed",
        ),
    ],
)
def test_schema_completeness_uniqueness_and_enums_fail_closed(
    tmp_path: Path,
    mutation: Any,
    expected: str,
) -> None:
    raw = _raw_dataset()
    mutation(raw)
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match=expected):
        load_expert_evaluation_dataset(path)


def test_verified_citation_status_requires_the_complete_citation_bundle(tmp_path: Path) -> None:
    raw = _raw_dataset()
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id="cite_sha256_" + "a" * 64,
    )
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="schema validation failed"):
        load_expert_evaluation_dataset(path)


def test_annotation_cannot_relabel_a_hard_negative_as_positive(tmp_path: Path) -> None:
    raw = _raw_dataset()
    case = raw["cases"][0]
    annotation = case["annotations"][0]
    annotation.update(
        status="completed",
        annotator_id="domain-reviewer-1",
        verdict="supported",
        selected_positive_evidence_ids=[case["hard_negative_evidence_ids"][0]],
        completed_at="2026-07-15T10:00:00Z",
    )
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="schema validation failed"):
        load_expert_evaluation_dataset(path)


def test_unmodeled_currentness_cannot_be_promoted_to_a_supported_case(tmp_path: Path) -> None:
    raw = _raw_dataset()
    case = next(item for item in raw["cases"] if item["query_class"] == "currentness")
    case.update(
        answerability="supported",
        positive_evidence_ids=[raw["evidence_catalog"][0]["evidence_id"]],
        hard_negative_evidence_ids=[raw["evidence_catalog"][1]["evidence_id"]],
        no_answer=None,
    )
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="schema validation failed"):
        load_expert_evaluation_dataset(path)


def test_complete_citation_bundle_is_reconstructed_against_the_bound_corpus(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    path = _write_sealed_dataset(tmp_path, raw)

    validation = load_expert_evaluation_dataset(path)

    assert validation.dataset.evidence_catalog[0].citation_v1 is not None


def test_separately_signed_legal_pack_attests_the_exact_citation_inventory(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)

    pack = {
        "schema_version": 1,
        "export_id": "legal-pack-test-v1",
        "source_relation": "public.regulatory_validated_section_citations",
        "exported_at": "2026-07-15T12:30:00Z",
        "citations": [citation],
    }
    pack_path = tmp_path / "validated-legal-pack.yml"
    pack_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    trusted_key = tmp_path / "trusted-legal-curator.pem"
    trusted_key.write_bytes(public_key)
    attestation = {
        "schema_version": 1,
        "attestation_id": "legal-attestation-test-v1",
        "curator_role": "legal_curator",
        "pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "citation_ids": [citation["citation_id"]],
        "attested_at": "2026-07-15T13:00:00Z",
        "integrity": {
            "attestation_sha256": "0" * 64,
            "signature_algorithm": "ed25519",
            "signature_reference": "legal-attestation.sig",
            "signature_public_key_sha256": hashlib.sha256(public_key).hexdigest(),
        },
    }
    (tmp_path / "legal-attestation.sig").write_bytes(private_key.sign(canonical_legal_attestation_payload(attestation)))
    attestation["integrity"]["attestation_sha256"] = canonical_legal_attestation_sha256(attestation)
    attestation_path = tmp_path / "legal-attestation.yml"
    attestation_path.write_text(yaml.safe_dump(attestation, sort_keys=False), encoding="utf-8")

    validation = load_expert_evaluation_dataset(
        dataset_path,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=trusted_key,
        now=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert validation.legal_attestation_verified is True
    assert validation.legal_attestation_key_sha256 == hashlib.sha256(public_key).hexdigest()


def test_partial_legal_attestation_inputs_fail_closed(tmp_path: Path) -> None:
    path = _write_sealed_dataset(tmp_path, _raw_dataset())

    with pytest.raises(ExpertEvaluationError, match="all required"):
        load_expert_evaluation_dataset(path, validated_legal_pack_path=tmp_path / "pack.yml")


def test_document_hash_drift_breaks_evidence_integrity(tmp_path: Path) -> None:
    raw = _raw_dataset()
    raw["evidence_catalog"][0]["document_content_sha256"] = "a" * 64
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="document metadata or hash has drifted"):
        load_expert_evaluation_dataset(path)


def test_section_hash_drift_breaks_referential_integrity(tmp_path: Path) -> None:
    raw = _raw_dataset()
    section_evidence = next(item for item in raw["evidence_catalog"] if item["granularity"] == "section")
    section_evidence["section_content_sha256"] = "b" * 64
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="section locator or hash has drifted"):
        load_expert_evaluation_dataset(path)


def test_corpus_manifest_version_drift_refuses_dataset(tmp_path: Path) -> None:
    raw = _raw_dataset()
    drifted_hash = "c" * 64
    raw["corpus"]["manifest_sha256"] = drifted_hash
    for evidence in raw["evidence_catalog"]:
        evidence["corpus_manifest_sha256"] = drifted_hash
    for case in raw["cases"]:
        if case.get("no_answer"):
            case["no_answer"]["basis_corpus_manifest_sha256"] = drifted_hash
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="manifest checksum has drifted"):
        load_expert_evaluation_dataset(path)


def test_validation_refuses_a_time_before_the_reviewed_snapshot() -> None:
    with pytest.raises(ExpertEvaluationError, match="corpus validation failed"):
        load_expert_evaluation_dataset(now=datetime.fromisoformat("2026-07-14T00:00:00+03:00"))


def test_future_annotation_timestamp_fails_dataset_integrity(tmp_path: Path) -> None:
    raw = _raw_dataset()
    annotation = raw["cases"][0]["annotations"][0]
    annotation.update(
        status="completed",
        annotator_id="domain-reviewer-1",
        verdict="supported",
        selected_positive_evidence_ids=raw["cases"][0]["positive_evidence_ids"],
        completed_at="2027-01-01T00:00:00Z",
    )
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="annotation timestamp is outside"):
        load_expert_evaluation_dataset(path, now=datetime.fromisoformat("2026-07-16T00:00:00+03:00"))


def test_verified_dataset_signature_requires_a_separate_trust_anchor(tmp_path: Path) -> None:
    raw = _raw_dataset()
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    trusted_key = tmp_path / "trusted-evaluation-key.pem"
    trusted_key.write_bytes(public_key)
    raw["integrity"].update(
        signature_status="verified",
        signature_algorithm="ed25519",
        signature_reference="expert-evaluation.sig",
        signature_public_key_sha256=hashlib.sha256(public_key).hexdigest(),
    )
    (tmp_path / "expert-evaluation.sig").write_bytes(private_key.sign(canonical_dataset_payload(raw)))
    path = _write_sealed_dataset(tmp_path, raw)

    validation = load_expert_evaluation_dataset(path, trusted_dataset_signing_key=trusted_key)
    assert validation.dataset_signature_verified is True

    with pytest.raises(ExpertEvaluationError, match="separately supplied trusted public key"):
        load_expert_evaluation_dataset(path)

    (tmp_path / "expert-evaluation.sig").write_bytes(b"0" * 64)
    with pytest.raises(ExpertEvaluationError, match="detached signature verification failed"):
        load_expert_evaluation_dataset(path, trusted_dataset_signing_key=trusted_key)


def test_release_use_refuses_pending_expert_and_owner_work() -> None:
    with pytest.raises(ExpertEvaluationReleaseError) as exc_info:
        load_expert_evaluation_dataset(require_release_ready=True)

    message = str(exc_info.value)
    assert "dataset_not_owner_approved=1" in message
    assert "case_not_owner_approved=20" in message
    assert "pending_annotations=40" in message
    assert "pending_adjudications=20" in message
    assert "pending_citation_v1_evidence=21" in message
    assert "dataset_signature_not_verified=1" in message
    assert "corpus_signature_not_verified=1" in message
    assert "legal_citation_attestation_not_verified=1" in message
    assert "unquantified_corpus_freshness_objectives=3" in message
    assert "corpus_freshness_slo_not_measured=1" in message
    assert "missing_required_query_classes=2" in message
    assert "tr-" not in message


def test_quality_profile_is_aggregate_only_and_release_aware() -> None:
    validation = load_expert_evaluation_dataset()
    profile = profile_expert_evaluation_dataset(validation)
    serialized = json.dumps(profile.as_dict(), ensure_ascii=False, sort_keys=True)

    assert profile.case_count == 20
    assert profile.evidence_count == 21
    assert profile.answerability_counts == {"abstain": 5, "supported": 15}
    assert profile.annotation_status_counts == {"pending": 40}
    assert profile.currentness_unverified_evidence_count == 21
    assert profile.citation_v1_status_counts == {"pending_legal_mapping": 21}
    assert profile.dataset_signature_verified is False
    assert profile.legal_attestation_verified is False
    assert profile.corpus_signature_verified is False
    assert profile.corpus_freshness_quantified is False
    assert profile.corpus_freshness_measured is False
    assert profile.release_ready is False
    assert "Bugün itibarıyla" not in serialized
    assert "DokumanGetir" not in serialized
    assert "markdown_content" not in serialized
    assert "annotator_id" not in serialized
