"""Tests for the admin console's read-only governance (signature) service."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bddk_mcp.admin.services.governance import GovernanceService, resolve_governance_paths
from bddk_mcp.corpus_manifest import canonical_manifest_sha256
from bddk_mcp.corpus_publication import CorpusPublicationError, CorpusReleaseIdentity
from bddk_mcp.ingest.seed import SEED_DIR
from scripts.sign_corpus_manifest import sign_manifest

ROOT = Path(__file__).resolve().parent.parent

_PLACEHOLDER_SHA = "f" * 64
_INTEGRITY_BLOCK_RE = re.compile(r"^integrity:\n(?:  .+\n?)+", re.MULTILINE)


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


def _stage_unsigned_corpus(tmp_path: Path) -> Path:
    """Stage the tracked manifest with stand-in artifacts, forced unsigned.

    The integrity block is rewritten to ``not_configured`` regardless of the
    working tree's current signing state, so these tests keep meaning the same
    thing after the release manifest is signed.
    """
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

    unsigned_block = f'integrity:\n  manifest_sha256: "{_PLACEHOLDER_SHA}"\n  signature_status: not_configured\n'
    assert _INTEGRITY_BLOCK_RE.search(text), "integrity block not found in tracked manifest"
    text = _INTEGRITY_BLOCK_RE.sub(unsigned_block, text)
    checksum = canonical_manifest_sha256(yaml.safe_load(text))
    text = text.replace(_PLACEHOLDER_SHA, checksum)
    (corpus / "corpus_scope.yml").write_text(text, encoding="utf-8")
    return corpus


def _service(corpus: Path, trusted_key: Path | None) -> GovernanceService:
    async def no_release(_pool):
        return None

    return GovernanceService(None, seed_dir=corpus, trusted_signing_key=trusted_key, release_inspector=no_release)


def _release_identity() -> CorpusReleaseIdentity:
    return CorpusReleaseIdentity(
        release_id="corpus_release_sha256_" + "a" * 64,
        manifest_id="bddk-job-corpus-2026-08-14",
        manifest_sha256="b" * 64,
        signer_key_sha256="c" * 64,
        freshness_policy_result="quantified_unmeasured_signature_verified_pass",
        source_detection_slo_seconds=604800,
        publication_slo_seconds=1209600,
        max_manifest_age_seconds=15552000,
        retrieval_profile_sha256="d" * 64,
        corpus_state_sha256="e" * 64,
        completed_at=datetime(2026, 8, 14, 12, 7, 45, tzinfo=UTC),
    )


def test_unsigned_manifest_reports_unsigned_not_failed(tmp_path: Path) -> None:
    corpus = _stage_unsigned_corpus(tmp_path)
    status = asyncio.run(_service(corpus, trusted_key=None).status())

    assert status.staged.verdict == "unsigned"
    assert status.staged.error is None
    assert status.staged.declared_signature_status == "not_configured"
    assert status.staged.manifest_id
    assert status.staged.signing_key_fingerprint_sha256 is None
    assert any("signature" in warning.lower() for warning in status.staged.warnings)


def test_signed_manifest_verifies_against_trusted_key(tmp_path: Path) -> None:
    corpus = _stage_unsigned_corpus(tmp_path)
    private_path, public_path = _write_key_pair(tmp_path)
    sign_manifest(
        manifest_path=corpus / "corpus_scope.yml",
        private_key_path=private_path,
        trusted_public_key_path=public_path,
    )

    status = asyncio.run(_service(corpus, trusted_key=public_path).status())

    assert status.staged.verdict == "verified"
    assert status.staged.error is None
    assert status.staged.declared_signature_status == "verified"
    assert status.staged.signing_key_fingerprint_sha256


def test_signed_manifest_without_configured_key_reads_as_key_missing(tmp_path: Path) -> None:
    corpus = _stage_unsigned_corpus(tmp_path)
    private_path, public_path = _write_key_pair(tmp_path)
    sign_manifest(
        manifest_path=corpus / "corpus_scope.yml",
        private_key_path=private_path,
        trusted_public_key_path=public_path,
    )

    status = asyncio.run(_service(corpus, trusted_key=None).status())

    assert status.staged.verdict == "key_missing"
    assert status.staged.error is not None


def test_artifact_drift_reports_failed(tmp_path: Path) -> None:
    corpus = _stage_unsigned_corpus(tmp_path)
    chunks_path = corpus / "chunks.json"
    payload = bytearray(chunks_path.read_bytes())
    payload[0] ^= 0x01  # same length, different checksum
    chunks_path.write_bytes(bytes(payload))

    status = asyncio.run(_service(corpus, trusted_key=None).status())

    assert status.staged.verdict == "failed"
    assert "checksum" in (status.staged.error or "")


def test_active_release_row_none_and_error_are_distinct(tmp_path: Path) -> None:
    corpus = _stage_unsigned_corpus(tmp_path)
    identity = _release_identity()

    async def found(_pool):
        return identity

    async def unavailable(_pool):
        raise CorpusPublicationError("Active corpus release evidence could not be verified.")

    with_release = GovernanceService(None, seed_dir=corpus, trusted_signing_key=None, release_inspector=found)
    status = asyncio.run(with_release.status())
    assert status.active.release is identity
    assert status.active.error is None

    without_release = _service(corpus, trusted_key=None)
    status = asyncio.run(without_release.status())
    assert status.active.release is None
    assert status.active.error is None

    failing = GovernanceService(None, seed_dir=corpus, trusted_signing_key=None, release_inspector=unavailable)
    status = asyncio.run(failing.status())
    assert status.active.release is None
    assert status.active.error is not None


def test_resolve_governance_paths_defaults_to_checkout(tmp_path: Path) -> None:
    seed_dir, trusted_key = resolve_governance_paths({})
    assert seed_dir == SEED_DIR
    assert trusted_key == SEED_DIR.parent / "deploy" / "trust" / "corpus-signing-public-key.pem"

    seed_dir, trusted_key = resolve_governance_paths({"BDDK_SEED_DIR": str(tmp_path)})
    assert seed_dir == tmp_path.resolve()
    assert trusted_key is None  # no deploy/trust anchor next to an arbitrary seed dir

    override = tmp_path / "anchor.pem"
    seed_dir, trusted_key = resolve_governance_paths(
        {"BDDK_SEED_DIR": str(tmp_path), "BDDK_TRUSTED_SIGNING_KEY": str(override)}
    )
    assert trusted_key == override.resolve()


def test_resolve_governance_paths_ignores_key_planted_beside_overridden_seed(tmp_path: Path) -> None:
    seed = tmp_path / "corpus"
    seed.mkdir()
    planted = tmp_path / "deploy" / "trust"
    planted.mkdir(parents=True)
    (planted / "corpus-signing-public-key.pem").write_text("not-a-trust-anchor", encoding="utf-8")

    seed_dir, trusted_key = resolve_governance_paths({"BDDK_SEED_DIR": str(seed)})
    assert seed_dir == seed.resolve()
    assert trusted_key is None
