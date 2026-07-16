"""Canonical signer identities used by evaluation trust-separation gates."""

from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def ed25519_public_key_fingerprint_sha256(public_key: Ed25519PublicKey) -> str:
    """Hash canonical raw key material, independent of PEM formatting."""

    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


__all__ = ("ed25519_public_key_fingerprint_sha256",)
