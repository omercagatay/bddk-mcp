"""Read-only corpus signature and release status for the admin console.

The service only observes: it validates the staged seed manifest with the
same engine function the ``verify-corpus`` CLI uses and reads the active
release through the path-free ``inspect_active_corpus_release`` accessor.
It never signs, repairs, imports, or activates anything — signing stays an
owner ceremony on the owner's machine (``scripts/sign_corpus_manifest.py``).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bddk_mcp.corpus_manifest import (
    CORPUS_MANIFEST_FILENAME,
    CorpusManifestError,
    load_and_validate_corpus_manifest,
)
from bddk_mcp.corpus_publication import (
    CorpusPublicationError,
    CorpusReleaseIdentity,
    inspect_active_corpus_release,
)
from bddk_mcp.ingest.seed import SEED_DIR


def resolve_governance_paths(env: Mapping[str, str] | None = None) -> tuple[Path, Path | None]:
    """Resolve the corpus directory and trusted signing key.

    ``BDDK_SEED_DIR`` overrides the checkout seed directory and
    ``BDDK_TRUSTED_SIGNING_KEY`` overrides the checkout trust anchor. The
    checkout key is used only for the checkout seed directory, never for an
    overridden corpus path (that path's parent is not a trust anchor).
    """
    source = os.environ if env is None else env
    raw_seed = (source.get("BDDK_SEED_DIR") or "").strip()
    seed_dir = Path(raw_seed).resolve() if raw_seed else SEED_DIR
    raw_key = (source.get("BDDK_TRUSTED_SIGNING_KEY") or "").strip()
    if raw_key:
        return seed_dir, Path(raw_key).resolve()
    if raw_seed:
        return seed_dir, None
    default_key = SEED_DIR.parent / "deploy" / "trust" / "corpus-signing-public-key.pem"
    return SEED_DIR, default_key if default_key.is_file() else None


@dataclass(frozen=True, slots=True)
class StagedManifestStatus:
    """Signature evidence for the staged (on-disk) corpus manifest."""

    seed_dir: str
    trusted_key: str | None
    verdict: str  # "verified" | "unsigned" | "key_missing" | "failed"
    declared_signature_status: str | None = None
    manifest_id: str | None = None
    manifest_sha256: str | None = None
    signing_key_fingerprint_sha256: str | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveReleaseStatus:
    """The active DB release identity, or the reason it could not be read."""

    release: CorpusReleaseIdentity | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GovernanceStatus:
    """Everything the read-only signature panel renders."""

    staged: StagedManifestStatus
    active: ActiveReleaseStatus


class GovernanceService:
    """Observe signature and release state; never mutate or sign."""

    def __init__(
        self,
        pool: Any,
        *,
        seed_dir: Path,
        trusted_signing_key: Path | None,
        release_inspector: Callable[[Any], Awaitable[CorpusReleaseIdentity | None]] = inspect_active_corpus_release,
    ) -> None:
        self._pool = pool
        self._seed_dir = seed_dir
        self._trusted_key = trusted_signing_key
        self._release_inspector = release_inspector

    async def status(self) -> GovernanceStatus:
        # Manifest validation hashes every artifact on disk; keep the event
        # loop responsive while it runs.
        staged = await asyncio.to_thread(self._staged_status)
        active = await self._active_status()
        return GovernanceStatus(staged=staged, active=active)

    def _declared_signature_status(self) -> str | None:
        """Read only integrity.signature_status, tolerating a broken manifest."""
        try:
            raw = yaml.safe_load((self._seed_dir / CORPUS_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            return None
        if isinstance(raw, dict) and isinstance(raw.get("integrity"), dict):
            value = raw["integrity"].get("signature_status")
            return str(value) if value is not None else None
        return None

    def _staged_status(self) -> StagedManifestStatus:
        declared = self._declared_signature_status()
        base: dict[str, Any] = {
            "seed_dir": str(self._seed_dir),
            "trusted_key": str(self._trusted_key) if self._trusted_key is not None else None,
            "declared_signature_status": declared,
        }
        try:
            validation = load_and_validate_corpus_manifest(
                self._seed_dir / CORPUS_MANIFEST_FILENAME,
                corpus_root=self._seed_dir,
                trusted_signing_key=self._trusted_key,
            )
        except CorpusManifestError as exc:
            key_missing = declared == "verified" and self._trusted_key is None
            return StagedManifestStatus(
                verdict="key_missing" if key_missing else "failed",
                error=str(exc),
                **base,
            )
        except OSError as exc:  # unreadable seed dir; surfaced verbatim, never rendered as "unsigned"
            return StagedManifestStatus(verdict="failed", error=f"{type(exc).__name__}: {exc}", **base)
        verdict = "verified" if validation.signing_key_fingerprint_sha256 else "unsigned"
        return StagedManifestStatus(
            verdict=verdict,
            manifest_id=validation.manifest.manifest_id,
            manifest_sha256=validation.manifest_sha256,
            signing_key_fingerprint_sha256=validation.signing_key_fingerprint_sha256,
            warnings=tuple(validation.warnings),
            **base,
        )

    async def _active_status(self) -> ActiveReleaseStatus:
        try:
            release = await self._release_inspector(self._pool)
        except CorpusPublicationError as exc:
            return ActiveReleaseStatus(error=str(exc))
        except Exception as exc:  # pool unavailable etc.; surfaced verbatim like DocumentService
            return ActiveReleaseStatus(error=f"{type(exc).__name__}: {exc}")
        return ActiveReleaseStatus(release=release)
