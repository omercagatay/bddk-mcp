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

import benchmark.legal_release_evidence as legal_release_evidence
from bddk_mcp.citations import (
    CitationQuality,
    CitationV1,
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
from benchmark.legal_release_evidence import (
    canonical_checkpoint_payload,
    canonical_checkpoint_sha256,
    validate_legal_release_evidence,
)

# The tracked corpus manifest is Ed25519-signed, so every load of the tracked
# seed corpus must supply the repository trust anchor explicitly.
_TRACKED_CORPUS_TRUST_KEY = Path(__file__).parents[1] / "deploy" / "trust" / "corpus-signing-public-key.pem"
_load_expert_evaluation_dataset_upstream = load_expert_evaluation_dataset


def load_expert_evaluation_dataset(*args: Any, **kwargs: Any):
    kwargs.setdefault("trusted_corpus_signing_key", _TRACKED_CORPUS_TRUST_KEY)
    return _load_expert_evaluation_dataset_upstream(*args, **kwargs)


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
    assert dataset.dataset_version.endswith("-draft.2")
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
        completed_at="2026-08-14T10:00:00Z",
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


def _write_signed_legal_pack(
    tmp_path: Path,
    citation: dict[str, Any],
    *,
    private_key: Ed25519PrivateKey | None = None,
    trusted_key_bytes: bytes | None = None,
    attested_at: str = "2026-08-27T13:00:00Z",
) -> tuple[Path, Path, Path, str]:
    pack = {
        "schema_version": 1,
        "export_id": "legal-pack-test-v1",
        "source_relation": "public.regulatory_validated_section_citations",
        "exported_at": "2026-08-27T12:30:00Z",
        "citations": [citation],
    }
    pack_path = tmp_path / "validated-legal-pack.yml"
    pack_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")

    private_key = private_key or Ed25519PrivateKey.generate()
    canonical_public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_key = trusted_key_bytes or canonical_public_key
    trusted_key = tmp_path / "trusted-legal-curator.pem"
    trusted_key.write_bytes(public_key)
    attestation = {
        "schema_version": 1,
        "attestation_id": "legal-attestation-test-v1",
        "curator_role": "legal_curator",
        "pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "citation_ids": [citation["citation_id"]],
        "attested_at": attested_at,
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
    return pack_path, attestation_path, trusted_key, hashlib.sha256(public_key).hexdigest()


def test_separately_signed_legal_pack_attests_the_exact_citation_inventory(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, trusted_key, key_sha256 = _write_signed_legal_pack(tmp_path, citation)

    validation = load_expert_evaluation_dataset(
        dataset_path,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=trusted_key,
        now=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert validation.legal_attestation_verified is True
    assert validation.legal_attestation_key_sha256 == key_sha256


def test_same_ed25519_key_with_different_pem_bytes_is_not_a_separate_signer(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    private_key = Ed25519PrivateKey.generate()
    canonical_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    dataset_pem = canonical_pem
    curator_pem = canonical_pem + b"\n"
    assert hashlib.sha256(dataset_pem).digest() != hashlib.sha256(curator_pem).digest()

    dataset_key = tmp_path / "trusted-dataset-key.pem"
    dataset_key.write_bytes(dataset_pem)
    raw["integrity"].update(
        signature_status="verified",
        signature_algorithm="ed25519",
        signature_reference="expert-evaluation-shared-key.sig",
        signature_public_key_sha256=hashlib.sha256(dataset_pem).hexdigest(),
    )
    (tmp_path / "expert-evaluation-shared-key.sig").write_bytes(private_key.sign(canonical_dataset_payload(raw)))
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(
        tmp_path,
        citation,
        private_key=private_key,
        trusted_key_bytes=curator_pem,
    )

    validation = load_expert_evaluation_dataset(
        dataset_path,
        trusted_dataset_signing_key=dataset_key,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=curator_key,
        now=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert validation.dataset_signing_key_fingerprint_sha256
    assert validation.dataset_signing_key_fingerprint_sha256 == validation.legal_attestation_key_fingerprint_sha256
    assert (
        profile_expert_evaluation_dataset(validation).release_blocker_counts["dataset_and_legal_signers_not_separated"]
        == 1
    )


def _sealed_file(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "reference": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _write_legal_release_checkpoint(
    tmp_path: Path,
    *,
    raw_dataset: dict[str, Any],
    citation: dict[str, Any],
    pack_path: Path,
    page_text_content: str | None = None,
    rotate_predecessor_key: bool = False,
    predecessor_key_outputs: list[Path] | None = None,
    page_mapping_schema_version: int = 1,
    reviewer_owner_id: str | None = None,
) -> tuple[Path, Path, str, Path, Path, Path, Path]:
    source_root = tmp_path / "retained-legal-source"
    source_root.mkdir()
    documents = json.loads((Path(__file__).parents[1] / "seed_data" / "documents.json").read_text())
    document = next(item for item in documents if item["document_id"] == citation["source_document_id"])
    source_bytes = document["markdown_content"].encode()
    source_path = source_root / "authoritative-source.bin"
    source_path.write_bytes(source_bytes)
    assert hashlib.sha256(source_bytes).hexdigest() == citation["artifact_sha256"]

    acquisition_path = source_root / "acquisition.json"
    _write_json(
        acquisition_path,
        {
            "schema_version": 1,
            "artifact_id": citation["artifact_id"],
            "blob_id": citation["artifact_blob_id"],
            "canonical_uri": citation["source_url"],
            "retrieved_at": citation["artifact_retrieved_at"],
            "captured_at": "2026-08-27T12:15:00Z",
            "source_authority": "BDDK",
            "media_type": "text/markdown",
            "response_status": 200,
            "response_body_sha256": citation["artifact_sha256"],
            "response_body_bytes": len(source_bytes),
        },
    )

    start = citation["locator"]["start_char"]
    end = citation["locator"]["end_char"]
    excerpt = sanitize_markdown_for_context(document["markdown_content"][start:end].strip())
    assert hashlib.sha256(excerpt.encode()).hexdigest() == citation["excerpt_sha256"]
    page_text_path = source_root / "page-1.txt"
    page_text_path.write_text(page_text_content or document["markdown_content"], encoding="utf-8")
    excerpt_path = source_root / "citation-excerpt.txt"
    excerpt_path.write_text(excerpt, encoding="utf-8")
    page_proof_path = source_root / "page-proof.json"
    page_proof = {
        "schema_version": page_mapping_schema_version,
        "proof_method": f"reviewed_source_page_mapping_v{page_mapping_schema_version}",
        "mapping_profile": "exact_utf8_excerpt_in_concatenated_page_text_v1",
        "artifact_id": citation["artifact_id"],
        "source_bytes_sha256": citation["artifact_sha256"],
        "source_bytes": len(source_bytes),
        "pages": [{"page_number": 1, "rendered_text": _sealed_file(page_text_path, source_root)}],
        "citation_mappings": [
            {
                "citation_id": citation["citation_id"],
                "page_numbers": [1],
                "rendered_excerpt": _sealed_file(excerpt_path, source_root),
            }
        ],
        "reviewed_by_role": "legal_source_reviewer",
        "reviewed_at": "2026-08-27T13:30:00Z",
    }
    if reviewer_owner_id is not None:
        page_proof["reviewed_by_owner_id"] = reviewer_owner_id
    _write_json(page_proof_path, page_proof)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    trusted_key = tmp_path / "trusted-legal-release.pem"
    trusted_key.write_bytes(public_key)
    predecessor_private_key = Ed25519PrivateKey.generate() if rotate_predecessor_key else private_key
    predecessor_public_key = predecessor_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if rotate_predecessor_key:
        predecessor_trusted_key = tmp_path / "trusted-legal-release-predecessor.pem"
        predecessor_trusted_key.write_bytes(predecessor_public_key)
        if predecessor_key_outputs is not None:
            predecessor_key_outputs.append(predecessor_trusted_key)
    checkpoint_common = {
        "schema_version": 2,
        "signer_role": "legal_release_certifier",
        "legal_pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": raw_dataset["corpus"]["manifest_sha256"],
        "artifacts": [
            {
                "artifact_id": citation["artifact_id"],
                "blob_id": citation["artifact_blob_id"],
                "citation_ids": [citation["citation_id"]],
                "source_bytes": _sealed_file(source_path, source_root),
                "acquisition_record": _sealed_file(acquisition_path, source_root),
                "page_mapping_proof": _sealed_file(page_proof_path, source_root),
            }
        ],
    }
    historical_source_path = source_root / "historical-authoritative-source.bin"
    historical_source_path.write_bytes(source_bytes)
    predecessor_artifacts = copy.deepcopy(checkpoint_common["artifacts"])
    predecessor_artifacts[0]["source_bytes"] = _sealed_file(historical_source_path, source_root)
    oldest_source_path = source_root / "oldest-authoritative-source.bin"
    oldest_source_path.write_bytes(source_bytes)
    oldest_artifacts = copy.deepcopy(checkpoint_common["artifacts"])
    oldest_artifacts[0]["source_bytes"] = _sealed_file(oldest_source_path, source_root)

    def seal_checkpoint(
        checkpoint: dict[str, Any],
        *,
        name: str,
        signing_private_key: Ed25519PrivateKey = private_key,
        signing_public_key: bytes = public_key,
    ) -> tuple[Path, str]:
        signature_name = f"{name}.sig"
        checkpoint["integrity"] = {
            "checkpoint_sha256": "0" * 64,
            "signature_algorithm": "ed25519",
            "signature_reference": signature_name,
            "signature_public_key_sha256": hashlib.sha256(signing_public_key).hexdigest(),
        }
        (tmp_path / signature_name).write_bytes(signing_private_key.sign(canonical_checkpoint_payload(checkpoint)))
        checkpoint["integrity"]["checkpoint_sha256"] = canonical_checkpoint_sha256(checkpoint)
        checkpoint_path = tmp_path / f"{name}.yml"
        checkpoint_path.write_text(yaml.safe_dump(checkpoint, sort_keys=False), encoding="utf-8")
        return checkpoint_path, checkpoint["integrity"]["checkpoint_sha256"]

    oldest_path, oldest_sha256 = seal_checkpoint(
        {
            **checkpoint_common,
            "artifacts": oldest_artifacts,
            "checkpoint_id": "legal-release-test-oldest",
            "created_at": "2026-08-27T13:35:00Z",
            "predecessor_checkpoint_sha256": None,
            "predecessor_checkpoint_reference": None,
        },
        name="legal-release-oldest",
        signing_private_key=predecessor_private_key,
        signing_public_key=predecessor_public_key,
    )
    predecessor_path, predecessor_sha256 = seal_checkpoint(
        {
            **checkpoint_common,
            "artifacts": predecessor_artifacts,
            "checkpoint_id": "legal-release-test-predecessor",
            "created_at": "2026-08-27T13:45:00Z",
            "predecessor_checkpoint_sha256": oldest_sha256,
            "predecessor_checkpoint_reference": "legal-release-oldest.yml",
        },
        name="legal-release-predecessor",
        signing_private_key=predecessor_private_key,
        signing_public_key=predecessor_public_key,
    )
    checkpoint_path, checkpoint_sha256 = seal_checkpoint(
        {
            **checkpoint_common,
            "checkpoint_id": "legal-release-test-v1",
            "created_at": "2026-08-27T14:00:00Z",
            "predecessor_checkpoint_sha256": predecessor_sha256,
            "predecessor_checkpoint_reference": "legal-release-predecessor.yml",
        },
        name="legal-release",
    )
    return (
        checkpoint_path,
        trusted_key,
        checkpoint_sha256,
        source_root,
        predecessor_path,
        oldest_path,
        oldest_source_path,
    )


def test_legal_release_checkpoint_binds_source_acquisition_pages_and_external_latest_hash(
    tmp_path: Path,
) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(tmp_path, citation)
    (
        checkpoint_path,
        release_key,
        latest_hash,
        source_root,
        predecessor_path,
        oldest_path,
        oldest_source,
    ) = _write_legal_release_checkpoint(tmp_path, raw_dataset=raw, citation=citation, pack_path=pack_path)

    validation = load_expert_evaluation_dataset(
        dataset_path,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=curator_key,
        legal_release_checkpoint_path=checkpoint_path,
        legal_release_source_root=source_root,
        trusted_legal_release_signing_key=release_key,
        predecessor_legal_release_checkpoint_path=predecessor_path,
        trusted_latest_legal_checkpoint_sha256=latest_hash,
        now=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert validation.legal_release_evidence_verified is True
    assert validation.legal_release_latest_checkpoint_verified is True
    assert validation.legal_release_checkpoint_sha256 == latest_hash
    blockers = profile_expert_evaluation_dataset(validation).release_blocker_counts
    assert "legal_release_evidence_not_verified" not in blockers
    assert "latest_legal_release_checkpoint_not_verified" not in blockers

    with pytest.raises(ExpertEvaluationError, match="not the bank-approved latest checkpoint"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            predecessor_legal_release_checkpoint_path=predecessor_path,
            trusted_latest_legal_checkpoint_sha256="f" * 64,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )

    without_external_latest = load_expert_evaluation_dataset(
        dataset_path,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=curator_key,
        legal_release_checkpoint_path=checkpoint_path,
        legal_release_source_root=source_root,
        trusted_legal_release_signing_key=release_key,
        predecessor_legal_release_checkpoint_path=predecessor_path,
        now=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert without_external_latest.legal_release_evidence_verified is True
    assert without_external_latest.legal_release_latest_checkpoint_verified is False

    unrelated_predecessor = tmp_path / "unrelated-predecessor.yml"
    unrelated_predecessor.write_text("not used", encoding="utf-8")
    with pytest.raises(ExpertEvaluationError, match="supplied legal release predecessor differs"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            predecessor_legal_release_checkpoint_path=unrelated_predecessor,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )

    predecessor_bytes = predecessor_path.read_bytes()
    predecessor_path.unlink()
    with pytest.raises(ExpertEvaluationError, match="predecessor checkpoint is unavailable"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )
    predecessor_path.write_bytes(predecessor_bytes)

    oldest_bytes = oldest_path.read_bytes()
    oldest_path.unlink()
    with pytest.raises(ExpertEvaluationError, match="predecessor checkpoint is unavailable"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )
    oldest_path.write_bytes(oldest_bytes)

    oldest_source_bytes = oldest_source.read_bytes()
    oldest_source.unlink()
    with pytest.raises(ExpertEvaluationError, match="retained authoritative source bytes is unavailable"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )
    oldest_source.write_bytes(oldest_source_bytes)

    historical_source = source_root / "historical-authoritative-source.bin"
    historical_bytes = historical_source.read_bytes()
    historical_source.unlink()
    with pytest.raises(ExpertEvaluationError, match="retained authoritative source bytes is unavailable"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )
    historical_source.write_bytes(historical_bytes)

    (source_root / "authoritative-source.bin").write_bytes(b"tampered")
    with pytest.raises(ExpertEvaluationError, match="retained authoritative source bytes differs"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            predecessor_legal_release_checkpoint_path=predecessor_path,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )


def test_legal_release_chain_accepts_an_explicit_rotated_predecessor_key(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(tmp_path, citation)
    predecessor_keys: list[Path] = []
    checkpoint_path, release_key, latest_hash, source_root, predecessor_path, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
        rotate_predecessor_key=True,
        predecessor_key_outputs=predecessor_keys,
    )

    with pytest.raises(ExpertEvaluationError, match="untrusted signing key"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            predecessor_legal_release_checkpoint_path=predecessor_path,
            trusted_latest_legal_checkpoint_sha256=latest_hash,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )

    validation = load_expert_evaluation_dataset(
        dataset_path,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=curator_key,
        legal_release_checkpoint_path=checkpoint_path,
        legal_release_source_root=source_root,
        trusted_legal_release_signing_key=release_key,
        trusted_legal_release_predecessor_signing_keys=predecessor_keys,
        predecessor_legal_release_checkpoint_path=predecessor_path,
        trusted_latest_legal_checkpoint_sha256=latest_hash,
        now=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert validation.legal_release_chain_checkpoint_count == 3
    assert len(validation.legal_release_chain_signers) == 3
    assert len({item.signing_key_fingerprint_sha256 for item in validation.legal_release_chain_signers}) == 2
    assert validation.legal_release_chain_signers[-1].signing_key_fingerprint_sha256 == (
        validation.legal_release_signing_key_fingerprint_sha256
    )
    assert validation.legal_release_configured_key_fingerprints_sha256[0] == (
        validation.legal_release_signing_key_fingerprint_sha256
    )
    assert set(validation.legal_release_configured_key_fingerprints_sha256) == {
        item.signing_key_fingerprint_sha256 for item in validation.legal_release_chain_signers
    }

    with pytest.raises(ExpertEvaluationError, match="does not use the primary signing key"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=predecessor_keys[0],
            trusted_legal_release_predecessor_signing_keys=[release_key],
            predecessor_legal_release_checkpoint_path=predecessor_path,
            trusted_latest_legal_checkpoint_sha256=latest_hash,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )

    with pytest.raises(ExpertEvaluationError, match="duplicate signer"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            trusted_legal_release_predecessor_signing_keys=[release_key],
            predecessor_legal_release_checkpoint_path=predecessor_path,
            trusted_latest_legal_checkpoint_sha256=latest_hash,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )


def test_page_mapping_v2_retains_policy_authorizable_reviewer_history(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(tmp_path, citation)
    checkpoint_path, release_key, latest_hash, source_root, predecessor_path, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
        page_mapping_schema_version=2,
        reviewer_owner_id="page-reviewer",
    )

    validation = load_expert_evaluation_dataset(
        dataset_path,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=curator_key,
        legal_release_checkpoint_path=checkpoint_path,
        legal_release_source_root=source_root,
        trusted_legal_release_signing_key=release_key,
        predecessor_legal_release_checkpoint_path=predecessor_path,
        trusted_latest_legal_checkpoint_sha256=latest_hash,
        now=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert len(validation.legal_source_reviews) == 3
    assert {review.proof_schema_version for review in validation.legal_source_reviews} == {2}
    assert {review.reviewer_owner_id for review in validation.legal_source_reviews} == {"page-reviewer"}
    assert tuple(review.checkpoint_sha256 for review in validation.legal_source_reviews) == tuple(
        signer.checkpoint_sha256 for signer in validation.legal_release_chain_signers
    )
    assert {review.artifact_id for review in validation.legal_source_reviews} == {citation["artifact_id"]}


def test_page_mapping_v2_requires_a_reviewer_owner_identity(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(tmp_path, citation)
    checkpoint_path, release_key, latest_hash, source_root, _, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
        page_mapping_schema_version=2,
    )

    with pytest.raises(ExpertEvaluationError, match="retained legal source evidence schema validation failed"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            trusted_latest_legal_checkpoint_sha256=latest_hash,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )


def test_page_mapping_proof_versions_do_not_infer_reviewer_identity(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    pack_path, _, _, _ = _write_signed_legal_pack(tmp_path, citation)
    _, _, _, source_root, _, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
    )
    base = json.loads((source_root / "page-proof.json").read_text())

    invalid_proofs = []
    v1_with_owner = copy.deepcopy(base)
    v1_with_owner["reviewed_by_owner_id"] = "page-reviewer"
    invalid_proofs.append(v1_with_owner)
    v1_with_v2_method = copy.deepcopy(base)
    v1_with_v2_method["proof_method"] = "reviewed_source_page_mapping_v2"
    invalid_proofs.append(v1_with_v2_method)
    v2_without_owner = copy.deepcopy(base)
    v2_without_owner.update(schema_version=2, proof_method="reviewed_source_page_mapping_v2")
    invalid_proofs.append(v2_without_owner)
    v2_with_v1_method = copy.deepcopy(base)
    v2_with_v1_method.update(schema_version=2, reviewed_by_owner_id="page-reviewer")
    invalid_proofs.append(v2_with_v1_method)
    v2_with_invalid_owner = copy.deepcopy(base)
    v2_with_invalid_owner.update(
        schema_version=2,
        proof_method="reviewed_source_page_mapping_v2",
        reviewed_by_owner_id="../reviewer",
    )
    invalid_proofs.append(v2_with_invalid_owner)
    coercible_v1_schema = copy.deepcopy(base)
    coercible_v1_schema["schema_version"] = True
    invalid_proofs.append(coercible_v1_schema)
    coercible_v2_schema = copy.deepcopy(base)
    coercible_v2_schema.update(
        schema_version=2.0,
        proof_method="reviewed_source_page_mapping_v2",
        reviewed_by_owner_id="page-reviewer",
    )
    invalid_proofs.append(coercible_v2_schema)
    numeric_review_timestamp = copy.deepcopy(base)
    numeric_review_timestamp["reviewed_at"] = 1_752_000_000
    invalid_proofs.append(numeric_review_timestamp)
    numeric_string_review_timestamp = copy.deepcopy(base)
    numeric_string_review_timestamp["reviewed_at"] = "1752000000"
    invalid_proofs.append(numeric_string_review_timestamp)

    for proof in invalid_proofs:
        with pytest.raises(ValueError):
            legal_release_evidence.PageMappingProof.model_validate(proof)


def test_legal_release_parses_the_exact_acquisition_bytes_that_were_hash_checked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(tmp_path, citation)
    checkpoint_path, release_key, latest_hash, source_root, _, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
    )

    original_verify = legal_release_evidence._verify_sealed_file
    replaced_acquisition: dict[str, tuple[Path, bytes]] = {}

    def verify_then_replace(root, sealed, *, label, maximum_bytes):
        if label == "retained source acquisition record" and replaced_acquisition:
            path, original_bytes = replaced_acquisition.popitem()[1]
            path.write_bytes(original_bytes)
        result = original_verify(root, sealed, label=label, maximum_bytes=maximum_bytes)
        if label == "retained source acquisition record" and not replaced_acquisition:
            path, original_bytes = result
            replaced_acquisition["pending"] = (path, original_bytes)
            path.write_bytes(b"{}")
        return result

    monkeypatch.setattr(legal_release_evidence, "_verify_sealed_file", verify_then_replace)

    validation = load_expert_evaluation_dataset(
        dataset_path,
        validated_legal_pack_path=pack_path,
        legal_attestation_path=attestation_path,
        trusted_legal_attestation_key=curator_key,
        legal_release_checkpoint_path=checkpoint_path,
        legal_release_source_root=source_root,
        trusted_legal_release_signing_key=release_key,
        trusted_latest_legal_checkpoint_sha256=latest_hash,
        now=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert validation.legal_release_evidence_verified is True


def test_release_significant_json_rejects_nested_duplicate_keys() -> None:
    with pytest.raises(legal_release_evidence.LegalReleaseEvidenceError, match="is invalid"):
        legal_release_evidence._parse_mapping_bytes(
            b'{"outer":{"artifact_id":"first","artifact_id":"second"}}',
            label="retained source acquisition record",
            json_only=True,
        )


def test_legal_release_rejects_an_excerpt_absent_from_its_signed_page_mapping(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(tmp_path, citation)
    checkpoint_path, release_key, _, source_root, _, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
        page_text_content="signed page that does not contain the citation excerpt",
    )

    with pytest.raises(ExpertEvaluationError, match="excerpt is absent from its mapped pages"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )


def test_legal_release_path_and_aggregate_budget_guards(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (root / "linked.bin").symlink_to(outside)
    sealed_link = legal_release_evidence.SealedFile(
        reference="linked.bin",
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        bytes=outside.stat().st_size,
    )

    with pytest.raises(legal_release_evidence.LegalReleaseEvidenceError, match="symbolic-link path"):
        legal_release_evidence._verify_sealed_file(
            root,
            sealed_link,
            label="retained test artifact",
            maximum_bytes=1024,
        )
    inside = root / "inside.bin"
    inside.write_bytes(b"inside")
    (root / "inside-link.bin").symlink_to(inside)
    sealed_inside_link = legal_release_evidence.SealedFile(
        reference="inside-link.bin",
        sha256=hashlib.sha256(inside.read_bytes()).hexdigest(),
        bytes=inside.stat().st_size,
    )
    with pytest.raises(legal_release_evidence.LegalReleaseEvidenceError, match="symbolic-link path"):
        legal_release_evidence._verify_sealed_file(
            root,
            sealed_inside_link,
            label="retained test artifact",
            maximum_bytes=1024,
        )
    with pytest.raises(ValueError, match="normalized relative path"):
        legal_release_evidence.SealedFile(reference="../outside.bin", sha256="0" * 64, bytes=1)

    budget = legal_release_evidence._RetentionBudget()
    budget.consume(
        legal_release_evidence.SealedFile(
            reference="first.bin",
            sha256="0" * 64,
            bytes=legal_release_evidence._MAX_RETAINED_BYTES_PER_CHAIN,
        )
    )
    with pytest.raises(legal_release_evidence.LegalReleaseEvidenceError, match="retained bytes exceed"):
        budget.consume(legal_release_evidence.SealedFile(reference="second.bin", sha256="0" * 64, bytes=1))


def test_legal_release_checkpoint_must_follow_curator_attestation(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    raw["evidence_catalog"][0].update(
        citation_v1_status="verified",
        citation_v1_id=citation["citation_id"],
        citation_v1=citation,
    )
    dataset_path = _write_sealed_dataset(tmp_path, raw)
    pack_path, attestation_path, curator_key, _ = _write_signed_legal_pack(
        tmp_path,
        citation,
        attested_at="2026-08-27T14:30:00Z",
    )
    checkpoint_path, release_key, _, source_root, _, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
    )

    with pytest.raises(ExpertEvaluationError, match="predates legal-curator approval"):
        load_expert_evaluation_dataset(
            dataset_path,
            validated_legal_pack_path=pack_path,
            legal_attestation_path=attestation_path,
            trusted_legal_attestation_key=curator_key,
            legal_release_checkpoint_path=checkpoint_path,
            legal_release_source_root=source_root,
            trusted_legal_release_signing_key=release_key,
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )


def test_legal_release_checkpoint_must_follow_corpus_scope_approval(tmp_path: Path) -> None:
    raw = _raw_dataset()
    citation = _verified_tracked_citation(raw)
    pack_path, _, _, _ = _write_signed_legal_pack(tmp_path, citation)
    checkpoint_path, release_key, _, source_root, _, _, _ = _write_legal_release_checkpoint(
        tmp_path,
        raw_dataset=raw,
        citation=citation,
        pack_path=pack_path,
    )

    with pytest.raises(
        legal_release_evidence.LegalReleaseEvidenceError,
        match="predates corpus build or scope approval",
    ):
        validate_legal_release_evidence(
            checkpoint_path=checkpoint_path,
            trusted_signing_key=release_key,
            source_root=source_root,
            legal_pack_sha256=hashlib.sha256(pack_path.read_bytes()).hexdigest(),
            legal_pack_exported_at=datetime(2026, 8, 27, 12, 30, tzinfo=UTC),
            legal_pack_attested_at=datetime(2026, 8, 27, 13, 0, tzinfo=UTC),
            corpus_manifest_sha256=raw["corpus"]["manifest_sha256"],
            corpus_approved_at=datetime(2026, 8, 27, 14, 30, tzinfo=UTC),
            citations=[CitationV1.model_validate(citation)],
            now=datetime(2026, 8, 28, tzinfo=UTC),
        )


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
        selected_positive_evidence_ids=list(raw["cases"][0]["positive_evidence_ids"]),
        completed_at="2027-01-01T00:00:00Z",
    )
    path = _write_sealed_dataset(tmp_path, raw)

    with pytest.raises(ExpertEvaluationError, match="annotation timestamp is outside"):
        load_expert_evaluation_dataset(path, now=datetime.fromisoformat("2026-08-28T00:00:00+03:00"))


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
    assert "corpus_signature_not_verified" not in message
    assert "legal_citation_attestation_not_verified=1" in message
    assert "legal_release_evidence_not_verified=1" in message
    assert "latest_legal_release_checkpoint_not_verified=1" in message
    assert "unquantified_corpus_freshness_objectives" not in message
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
    assert profile.legal_release_evidence_verified is False
    assert profile.legal_release_latest_checkpoint_verified is False
    assert profile.corpus_signature_verified is True
    assert profile.corpus_freshness_quantified is True
    assert profile.corpus_freshness_measured is False
    assert profile.release_ready is False
    assert "Bugün itibarıyla" not in serialized
    assert "DokumanGetir" not in serialized
    assert "markdown_content" not in serialized
    assert "annotator_id" not in serialized
