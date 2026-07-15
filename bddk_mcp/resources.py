"""Reviewed MCP resources that expose path-free operational identities."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from bddk_mcp.corpus_publication import CorpusPublicationError, inspect_active_corpus_release

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)

ACTIVE_CORPUS_RELEASE_RESOURCE_URI = "bddk://corpus/active-release"
ACTIVE_CORPUS_RELEASE_RESOURCE_SCHEMA_VERSION = "1.0"
_RESOURCE_READ_TIMEOUT_SECONDS = 5.0


def _unavailable_payload() -> dict[str, str]:
    return {
        "schema_version": ACTIVE_CORPUS_RELEASE_RESOURCE_SCHEMA_VERSION,
        "status": "unavailable",
    }


def _active_release_payload(release: Any) -> dict[str, str]:
    """Reduce the database attestation to the identity needed by evaluators.

    The signer fingerprint and corpus-state fingerprint deliberately stay on
    the operator-only evidence surface.  They are not required to prove that
    two benchmark runs addressed the same activated manifest and retrieval
    profile.
    """

    return {
        "schema_version": ACTIVE_CORPUS_RELEASE_RESOURCE_SCHEMA_VERSION,
        "status": "active",
        "release_id": release.release_id,
        "manifest_id": release.manifest_id,
        "manifest_sha256": release.manifest_sha256,
        "retrieval_profile_sha256": release.retrieval_profile_sha256,
    }


def register_resources(server: Any, deps: Dependencies) -> None:
    """Register the fixed public resource set on one MCP server instance."""

    @server.resource(
        ACTIVE_CORPUS_RELEASE_RESOURCE_URI,
        name="active_corpus_release",
        title="Active regulatory corpus release",
        description=(
            "Path-free identity of the strictly activated corpus manifest and retrieval profile. "
            "Returns unavailable when no verified release is active."
        ),
        mime_type="application/json",
    )
    async def active_corpus_release() -> str:
        pool = deps.pool
        if pool is None:
            return json.dumps(_unavailable_payload(), sort_keys=True, separators=(",", ":"))
        try:
            async with asyncio.timeout(_RESOURCE_READ_TIMEOUT_SECONDS):
                release = await inspect_active_corpus_release(pool)
        except (TimeoutError, CorpusPublicationError, OSError) as error:
            logger.warning(
                "Active corpus release MCP resource is unavailable",
                extra={"error_type": type(error).__name__},
            )
            return json.dumps(_unavailable_payload(), sort_keys=True, separators=(",", ":"))
        payload = _unavailable_payload() if release is None else _active_release_payload(release)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = (
    "ACTIVE_CORPUS_RELEASE_RESOURCE_SCHEMA_VERSION",
    "ACTIVE_CORPUS_RELEASE_RESOURCE_URI",
    "register_resources",
)
