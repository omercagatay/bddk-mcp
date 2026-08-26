"""Finalize and Ed25519-sign the corpus scope manifest.

Implements steps 5-7 of the release procedure in docs/CORPUS_GOVERNANCE.md for
the selection owner: refresh `scope_reviewed_at`, write the `verified`
integrity block, recompute the canonical manifest checksum, produce the
detached raw Ed25519 signature, and immediately re-verify it against the
project trust anchor. The private key is supplied by path at invocation time
and is never read from, or written into, the repository.

Run this only after independently reviewing the regenerated artifacts the
manifest declares. Signing is an owner attestation, not a formality.

Usage:
    uv run python scripts/sign_corpus_manifest.py --private-key /secure/path/ed25519-private.pem
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bddk_mcp.corpus_manifest import (  # noqa: E402
    CorpusManifestError,
    CorpusScopeManifest,
    canonical_manifest_payload,
    canonical_manifest_sha256,
)

_PLACEHOLDER_SHA = "0" * 64


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_artifacts_match_disk(manifest: CorpusScopeManifest, corpus_root: Path) -> None:
    """Never sign a declaration whose artifact facts differ from disk."""
    for artifact in manifest.artifacts:
        artifact_path = corpus_root / artifact.path
        data = artifact_path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != artifact.sha256 or len(data) != artifact.bytes:
            raise CorpusManifestError(
                f"artifact {artifact.path} on disk does not match the manifest declaration; "
                "regenerate or correct the declaration before signing"
            )


def _replace_scope_reviewed_at(text: str, reviewed_at: str) -> str:
    pattern = re.compile(r'(^  scope_reviewed_at: )"[^"]+"$', re.MULTILINE)
    if not pattern.search(text):
        raise CorpusManifestError("scope_reviewed_at line not found in manifest")
    return pattern.sub(rf'\g<1>"{reviewed_at}"', text)


def _replace_integrity_block(text: str, manifest_sha256: str, public_key_sha256: str) -> str:
    pattern = re.compile(r"^integrity:\n(?:  .+\n?)+", re.MULTILINE)
    if not pattern.search(text):
        raise CorpusManifestError("integrity block not found in manifest")
    block = (
        "integrity:\n"
        f'  manifest_sha256: "{manifest_sha256}"\n'
        "  signature_status: verified\n"
        "  signature_algorithm: ed25519\n"
        "  signature_reference: corpus_scope.sig\n"
        f'  signature_public_key_sha256: "{public_key_sha256}"\n'
    )
    return pattern.sub(block, text)


def sign_manifest(
    *,
    manifest_path: Path,
    private_key_path: Path,
    trusted_public_key_path: Path,
    reviewed_at: str | None = None,
) -> str:
    """Finalize, sign, and re-verify the manifest; returns the new manifest SHA-256."""
    key_bytes = private_key_path.read_bytes()
    private_key = serialization.load_pem_private_key(key_bytes, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise CorpusManifestError("the supplied private key is not an Ed25519 key")

    public_key_bytes = trusted_public_key_path.read_bytes()
    trusted_public = serialization.load_pem_public_key(public_key_bytes)
    if not isinstance(trusted_public, Ed25519PublicKey):
        raise CorpusManifestError("trusted public key is not an Ed25519 key")
    raw_public = trusted_public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    derived_raw_public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if derived_raw_public != raw_public:
        raise CorpusManifestError(
            "the supplied private key does not correspond to the trusted public key; "
            "refusing to sign against the wrong trust anchor"
        )
    # The manifest binds the sha256 of the trust-anchor PEM bytes, matching
    # the loader's _verify_manifest_signature check.
    public_key_sha256 = hashlib.sha256(public_key_bytes).hexdigest()

    text = manifest_path.read_text(encoding="utf-8")
    stamp = reviewed_at or datetime.now(UTC).strftime("%Y-%m-%dT00:00:00+00:00")
    text = _replace_scope_reviewed_at(text, stamp)
    # Write the verified-shape integrity block with a placeholder checksum,
    # compute the canonical checksum over that exact declaration, then fill it.
    text = _replace_integrity_block(text, _PLACEHOLDER_SHA, public_key_sha256)
    manifest_path.write_text(text, encoding="utf-8")

    raw = _load_yaml(manifest_path)
    manifest = CorpusScopeManifest.model_validate(raw)
    _assert_artifacts_match_disk(manifest, manifest_path.parent)

    manifest_sha = canonical_manifest_sha256(raw)
    text = text.replace(_PLACEHOLDER_SHA, manifest_sha)
    manifest_path.write_text(text, encoding="utf-8")

    raw = _load_yaml(manifest_path)
    signature = private_key.sign(canonical_manifest_payload(raw))
    signature_path = manifest_path.parent / "corpus_scope.sig"
    signature_path.write_bytes(signature)

    trusted_public.verify(signature_path.read_bytes(), canonical_manifest_payload(raw))
    return manifest_sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-key", type=Path, required=True, help="Ed25519 private key PEM (kept outside Git)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "seed_data" / "corpus_scope.yml")
    parser.add_argument(
        "--trusted-public-key",
        type=Path,
        default=ROOT / "deploy" / "trust" / "corpus-signing-public-key.pem",
    )
    parser.add_argument(
        "--reviewed-at",
        default=None,
        help='ISO-8601 review stamp for freshness.scope_reviewed_at (default: today, UTC, "T00:00:00+00:00")',
    )
    args = parser.parse_args()

    try:
        manifest_sha = sign_manifest(
            manifest_path=args.manifest,
            private_key_path=args.private_key,
            trusted_public_key_path=args.trusted_public_key,
            reviewed_at=args.reviewed_at,
        )
    except CorpusManifestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"signed: manifest_sha256={manifest_sha}")
    print("verify with: uv run bddk-mcp verify-corpus --trusted-signing-key deploy/trust/corpus-signing-public-key.pem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
