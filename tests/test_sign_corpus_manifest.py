"""Tests for scripts/sign_corpus_manifest.py — owner signing helper."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bddk_mcp.corpus_manifest import CorpusManifestError
from scripts.sign_corpus_manifest import sign_manifest

ROOT = Path(__file__).parents[1]


def _write_key_pair(directory: Path) -> tuple[Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_path = directory / "ed25519-private.pem"
    public_path = directory / "ed25519-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def _stage_corpus(tmp_path: Path) -> Path:
    """Copy the tracked manifest with tiny artifact stand-ins that match it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = (ROOT / "seed_data" / "corpus_scope.yml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    freshness = raw["freshness"]
    observed_start = datetime.fromisoformat(freshness["source_observed_start"]).timestamp()
    observed_end = datetime.fromisoformat(freshness["source_observed_end"]).timestamp()
    built_at = datetime.fromisoformat(freshness["corpus_built_at"]).timestamp()
    fixtures: dict[str, list[dict]] = {
        # The loader cross-checks document timestamps against the manifest
        # freshness window, so the stand-ins must span it exactly.
        "documents": [
            {"fixture": "documents", "downloaded_at": observed_start, "extracted_at": observed_start},
            {"fixture": "documents", "downloaded_at": observed_end, "extracted_at": built_at},
        ],
        "chunks": [{"fixture": "chunks"}],
        "decision_cache": [{"fixture": "decision_cache"}],
    }
    for artifact in raw["artifacts"]:
        payload = json.dumps(fixtures[artifact["role"]]).encode("utf-8")
        (corpus / artifact["path"]).write_bytes(payload)
        text = text.replace(artifact["sha256"], hashlib.sha256(payload).hexdigest())
        text = text.replace(f"bytes: {artifact['bytes']}", f"bytes: {len(payload)}")
        text = re.sub(
            rf"(records: ){artifact['records']}\b",
            rf"\g<1>{len(fixtures[artifact['role']])}",
            text,
            count=1,
        )
    (corpus / "corpus_scope.yml").write_text(text, encoding="utf-8")
    return corpus


def test_sign_manifest_produces_loader_verifiable_signature(tmp_path):
    corpus = _stage_corpus(tmp_path)
    keys = tmp_path / "keys"
    keys.mkdir()
    private_path, public_path = _write_key_pair(keys)

    manifest_path = corpus / "corpus_scope.yml"
    sign_manifest(
        manifest_path=manifest_path,
        private_key_path=private_path,
        trusted_public_key_path=public_path,
        reviewed_at="2026-08-26T00:00:00+00:00",
    )

    from bddk_mcp.corpus_manifest import load_and_validate_corpus_manifest

    validation = load_and_validate_corpus_manifest(
        manifest_path,
        require_verified_signature=True,
        trusted_signing_key=public_path,
    )
    assert validation.manifest.integrity.signature_status == "verified"
    assert validation.manifest.freshness.scope_reviewed_at == datetime.fromisoformat("2026-08-26T00:00:00+00:00")


def test_sign_manifest_refuses_wrong_private_key(tmp_path):
    corpus = _stage_corpus(tmp_path)
    keys = tmp_path / "keys"
    keys.mkdir()
    private_path, _ = _write_key_pair(keys)
    other = tmp_path / "other"
    other.mkdir()
    _, unrelated_public = _write_key_pair(other)

    with pytest.raises(CorpusManifestError, match="does not correspond"):
        sign_manifest(
            manifest_path=corpus / "corpus_scope.yml",
            private_key_path=private_path,
            trusted_public_key_path=unrelated_public,
        )


def test_sign_manifest_refuses_artifact_drift(tmp_path):
    corpus = _stage_corpus(tmp_path)
    keys = tmp_path / "keys"
    keys.mkdir()
    private_path, public_path = _write_key_pair(keys)

    # Corrupt one artifact after the manifest declared it.
    (corpus / "chunks.json").write_bytes(b"[]")

    with pytest.raises(CorpusManifestError, match="does not match the manifest declaration"):
        sign_manifest(
            manifest_path=corpus / "corpus_scope.yml",
            private_key_path=private_path,
            trusted_public_key_path=public_path,
        )


def test_sign_manifest_is_idempotent_over_verified_manifests(tmp_path):
    """Re-signing an already verified manifest must succeed cleanly."""
    corpus = _stage_corpus(tmp_path)
    keys = tmp_path / "keys"
    keys.mkdir()
    private_path, public_path = _write_key_pair(keys)
    manifest_path = corpus / "corpus_scope.yml"

    first = sign_manifest(
        manifest_path=manifest_path,
        private_key_path=private_path,
        trusted_public_key_path=public_path,
        reviewed_at="2026-08-26T00:00:00+00:00",
    )
    second = sign_manifest(
        manifest_path=manifest_path,
        private_key_path=private_path,
        trusted_public_key_path=public_path,
        reviewed_at="2026-08-26T00:00:00+00:00",
    )
    assert first == second
