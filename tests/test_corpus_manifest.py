from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bddk_mcp.corpus_manifest import (
    CORPUS_SCOPE_WARNING,
    CorpusManifestError,
    assert_corpus_manifest_freshness_current,
    canonical_manifest_payload,
    canonical_manifest_sha256,
    load_and_validate_corpus_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trusted_key_path(corpus_root: Path) -> Path:
    return corpus_root.parent / f"{corpus_root.name}-trusted-public-key.pem"


def _write_manifest(
    root: Path,
    *,
    max_age_seconds: int | None = None,
    signature_status: str = "not_configured",
    measured: bool = False,
) -> Path:
    observed_start = datetime(2026, 1, 1, tzinfo=UTC)
    observed_end = datetime(2026, 1, 2, tzinfo=UTC)
    built_at = datetime(2026, 1, 3, tzinfo=UTC)
    documents = [
        {
            "document_id": "doc-1",
            "downloaded_at": observed_start.timestamp(),
            "extracted_at": built_at.timestamp(),
        },
        {
            "document_id": "doc-2",
            "downloaded_at": observed_end.timestamp(),
            "extracted_at": built_at.timestamp(),
        },
    ]
    if measured:
        for row in documents:
            row["authoritative_published_at"] = row["downloaded_at"] - 30
            row["source_detected_at"] = row["downloaded_at"] - 10
            row["retrieval_published_at"] = built_at.timestamp() + 10
    artifacts = {
        "documents.json": documents,
        "chunks.json": [{"doc_id": "doc-1"}],
        "decision_cache.json": [{"document_id": "doc-1"}],
    }
    for name, value in artifacts.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")

    raw = {
        "schema_version": 1,
        "manifest_id": "test-corpus-v1",
        "selection_owner": "test-owner",
        "purpose": "Test-only selected public regulatory corpus.",
        "exhaustive": False,
        "included_source_classes": ["selected-public-regulations"],
        "excluded_source_classes": ["private-bank-documents"],
        "known_gaps": ["legal-currentness-not-validated"],
        "freshness": {
            "source_observed_start": observed_start.isoformat(),
            "source_observed_end": observed_end.isoformat(),
            "corpus_built_at": built_at.isoformat(),
            "scope_reviewed_at": built_at.isoformat(),
            "business_expectation": "immediate",
            "source_detection_slo_seconds": 60 if max_age_seconds is not None or measured else None,
            "publication_slo_seconds": 200_000 if measured else (60 if max_age_seconds is not None else None),
            "max_manifest_age_seconds": (max_age_seconds or 1_000_000) if measured else max_age_seconds,
            "slo_evidence_status": "measured" if measured else "not_measured",
        },
        "artifacts": [
            {
                "role": role,
                "path": name,
                "sha256": _sha256(root / name),
                "bytes": (root / name).stat().st_size,
                "records": len(artifacts[name]),
            }
            for role, name in (
                ("documents", "documents.json"),
                ("chunks", "chunks.json"),
                ("decision_cache", "decision_cache.json"),
            )
        ],
        "integrity": {
            "manifest_sha256": "0" * 64,
            "signature_status": signature_status,
            "signature_algorithm": None,
            "signature_reference": None,
            "signature_public_key_sha256": None,
        },
    }
    if signature_status == "verified":
        private_key = Ed25519PrivateKey.generate()
        public_key_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        _trusted_key_path(root).write_bytes(public_key_bytes)
        raw["integrity"].update(
            signature_algorithm="ed25519",
            signature_reference="corpus_scope.sig",
            signature_public_key_sha256=hashlib.sha256(public_key_bytes).hexdigest(),
        )
        (root / "corpus_scope.sig").write_bytes(private_key.sign(canonical_manifest_payload(raw)))
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    path = root / "corpus_scope.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_tracked_corpus_manifest_matches_all_reviewed_seed_artifacts():
    root = Path(__file__).parents[1] / "seed_data"
    validation = load_and_validate_corpus_manifest(root / "corpus_scope.yml", corpus_root=root)

    assert validation.manifest.schema_version == 1
    assert validation.manifest.exhaustive is False
    assert validation.manifest.selection_owner == "project_owner"
    assert {artifact.role for artifact in validation.manifest.artifacts} == {
        "documents",
        "chunks",
        "decision_cache",
    }
    assert CORPUS_SCOPE_WARNING in validation.warnings
    assert any("not yet quantified" in warning for warning in validation.warnings)
    assert any("no digital signature" in warning for warning in validation.warnings)


def test_manifest_rejects_tampered_artifact_without_leaking_content(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    (tmp_path / "documents.json").write_text("secret-corpus-payload", encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="size differs") as error:
        load_and_validate_corpus_manifest(manifest)

    assert "secret-corpus-payload" not in str(error.value)


def test_manifest_rejects_checksum_tampering_before_artifact_use(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["purpose"] = "Silently changed scope"
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="checksum mismatch"):
        load_and_validate_corpus_manifest(manifest)


@pytest.mark.parametrize(
    "payload",
    (
        "schema_version: 1\nschema_version: 2\n",
        "schema_version: &version 1\ncopied_version: *version\n",
    ),
)
def test_manifest_rejects_ambiguous_yaml_before_schema_validation(tmp_path: Path, payload: str):
    manifest = tmp_path / "corpus_scope.yml"
    manifest.write_text(payload, encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="corpus manifest YAML is invalid"):
        load_and_validate_corpus_manifest(manifest)


def test_manifest_rejects_scope_review_that_predates_the_corpus_build(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["freshness"]["scope_reviewed_at"] = datetime(2026, 1, 2, tzinfo=UTC).isoformat()
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="schema validation failed"):
        load_and_validate_corpus_manifest(manifest)


def test_manifest_rejects_a_document_extracted_before_its_download(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    documents_path = tmp_path / "documents.json"
    documents = json.loads(documents_path.read_text(encoding="utf-8"))
    documents[0]["extracted_at"] = documents[0]["downloaded_at"] - 1
    documents_path.write_text(json.dumps(documents), encoding="utf-8")
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    artifact = next(item for item in raw["artifacts"] if item["role"] == "documents")
    artifact["sha256"] = _sha256(documents_path)
    artifact["bytes"] = documents_path.stat().st_size
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="extraction timestamps before download"):
        load_and_validate_corpus_manifest(manifest)


def test_manifest_rejects_missing_file_and_path_traversal(tmp_path: Path):
    with pytest.raises(CorpusManifestError, match="required corpus manifest is missing"):
        load_and_validate_corpus_manifest(tmp_path / "corpus_scope.yml")

    manifest = _write_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["artifacts"][0]["path"] = "../documents.json"
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(CorpusManifestError, match="schema validation failed"):
        load_and_validate_corpus_manifest(manifest)


def test_manifest_rejects_duplicate_well_known_artifact_roles(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    duplicate_path = tmp_path / "duplicate-documents.json"
    duplicate_path.write_bytes((tmp_path / "documents.json").read_bytes())
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    duplicate = dict(next(item for item in raw["artifacts"] if item["role"] == "documents"))
    duplicate["path"] = duplicate_path.name
    raw["artifacts"].append(duplicate)
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="schema validation failed"):
        load_and_validate_corpus_manifest(manifest)


def test_unquantified_freshness_and_unsigned_manifest_fail_when_required(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    now = datetime(2026, 1, 4, tzinfo=UTC)

    with pytest.raises(CorpusManifestError, match="freshness objectives are not quantified"):
        load_and_validate_corpus_manifest(manifest, now=now, require_quantified_freshness=True)
    with pytest.raises(CorpusManifestError, match="signature is not verified"):
        load_and_validate_corpus_manifest(manifest, now=now, require_verified_signature=True)


def test_verified_signature_requires_the_separate_trust_anchor_and_rejects_tampering(tmp_path: Path):
    manifest = _write_manifest(tmp_path, signature_status="verified")
    trusted_key = _trusted_key_path(tmp_path)

    validation = load_and_validate_corpus_manifest(
        manifest,
        trusted_signing_key=trusted_key,
        require_verified_signature=True,
    )
    assert validation.manifest.integrity.signature_algorithm == "ed25519"
    assert validation.signature_sha256 == hashlib.sha256((tmp_path / "corpus_scope.sig").read_bytes()).hexdigest()

    with pytest.raises(CorpusManifestError, match="separately supplied trusted public key"):
        load_and_validate_corpus_manifest(manifest, require_verified_signature=True)

    (tmp_path / "corpus_scope.sig").write_bytes(b"0" * 64)
    with pytest.raises(CorpusManifestError, match="detached signature verification failed"):
        load_and_validate_corpus_manifest(
            manifest,
            trusted_signing_key=trusted_key,
            require_verified_signature=True,
        )


def test_verified_signature_rejects_a_trust_anchor_controlled_by_the_corpus_root(tmp_path: Path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest = _write_manifest(corpus_root, signature_status="verified")
    external_key = _trusted_key_path(corpus_root)
    corpus_controlled_key = corpus_root / "trusted-public-key.pem"
    corpus_controlled_key.write_bytes(external_key.read_bytes())

    with pytest.raises(CorpusManifestError, match="outside the corpus root"):
        load_and_validate_corpus_manifest(
            manifest,
            corpus_root=corpus_root,
            trusted_signing_key=corpus_controlled_key,
            require_verified_signature=True,
        )

    corpus_controlled_link = corpus_root / "trusted-public-key-link.pem"
    corpus_controlled_link.symlink_to(external_key)
    with pytest.raises(CorpusManifestError, match="outside the corpus root"):
        load_and_validate_corpus_manifest(
            manifest,
            corpus_root=corpus_root,
            trusted_signing_key=corpus_controlled_link,
            require_verified_signature=True,
        )


def test_quantified_stale_manifest_fails_closed(tmp_path: Path):
    manifest = _write_manifest(tmp_path, max_age_seconds=60, signature_status="verified")
    now = datetime(2026, 1, 3, tzinfo=UTC) + timedelta(seconds=61)

    with pytest.raises(CorpusManifestError, match="stale"):
        load_and_validate_corpus_manifest(
            manifest,
            now=now,
            require_quantified_freshness=True,
            require_verified_signature=True,
            trusted_signing_key=_trusted_key_path(tmp_path),
        )


def test_time_dependent_freshness_can_be_rechecked_without_reopening_artifacts(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, max_age_seconds=60, signature_status="verified")
    validation = load_and_validate_corpus_manifest(
        manifest_path,
        now=datetime(2026, 1, 3, tzinfo=UTC) + timedelta(seconds=59),
        require_quantified_freshness=True,
        require_verified_signature=True,
        trusted_signing_key=_trusted_key_path(tmp_path),
    )

    with pytest.raises(CorpusManifestError, match="stale"):
        assert_corpus_manifest_freshness_current(
            validation.manifest,
            now=datetime(2026, 1, 3, tzinfo=UTC) + timedelta(seconds=61),
        )


def test_measured_freshness_requires_per_document_events_within_declared_slos(tmp_path: Path):
    manifest = _write_manifest(tmp_path, measured=True)
    validation = load_and_validate_corpus_manifest(
        manifest,
        now=datetime(2026, 1, 3, tzinfo=UTC) + timedelta(seconds=20),
        require_quantified_freshness=True,
        require_measured_freshness=True,
    )
    assert validation.manifest.freshness.slo_evidence_status == "measured"

    documents_path = tmp_path / "documents.json"
    documents = json.loads(documents_path.read_text(encoding="utf-8"))
    documents[0]["authoritative_published_at"] = documents[0]["source_detected_at"] - 61
    documents_path.write_text(json.dumps(documents), encoding="utf-8")
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    artifact = next(item for item in raw["artifacts"] if item["role"] == "documents")
    artifact["sha256"] = _sha256(documents_path)
    artifact["bytes"] = documents_path.stat().st_size
    raw["integrity"]["manifest_sha256"] = canonical_manifest_sha256(raw)
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="exceeds the source-detection SLO"):
        load_and_validate_corpus_manifest(
            manifest,
            now=datetime(2026, 1, 3, tzinfo=UTC) + timedelta(seconds=20),
            require_measured_freshness=True,
        )
