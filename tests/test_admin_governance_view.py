"""Tests for the read-only /governance signature panel."""

from __future__ import annotations

from datetime import UTC, datetime

from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.admin.services.governance import (
    ActiveReleaseStatus,
    GovernanceStatus,
    StagedManifestStatus,
)
from bddk_mcp.corpus_publication import CorpusReleaseIdentity
from bddk_mcp.store.doc_store import StoreStats

CONFIG = AdminConfig(bind_host="127.0.0.1", port=8100, database_url="postgresql://x", loopback_only=True)


class EmptyStore:
    async def list_documents(self, category=None, limit=100, offset=0):
        return []

    async def stats(self):
        return StoreStats(categories={}, total_documents=0)


class FakeGovernanceService:
    def __init__(self, status: GovernanceStatus):
        self._status = status

    async def status(self) -> GovernanceStatus:
        return self._status


def _client(status: GovernanceStatus) -> TestClient:
    return TestClient(
        create_app(CONFIG, DocumentService(EmptyStore()), FakeGovernanceService(status)),
        base_url="http://127.0.0.1",
    )


def _release() -> CorpusReleaseIdentity:
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


def test_unsigned_staged_manifest_renders_warning_pill_and_active_release() -> None:
    status = GovernanceStatus(
        staged=StagedManifestStatus(
            seed_dir="/srv/corpus",
            trusted_key=None,
            verdict="unsigned",
            declared_signature_status="not_configured",
            manifest_id="bddk-job-corpus-2026-08-26",
            manifest_sha256="49" * 32,
            warnings=("Corpus manifest checksum is verified, but no digital signature is configured.",),
        ),
        active=ActiveReleaseStatus(release=_release()),
    )
    response = _client(status).get("/governance")

    assert response.status_code == 200
    assert "İmzasız" in response.text
    assert "bddk-job-corpus-2026-08-26" in response.text
    assert "bddk-job-corpus-2026-08-14" in response.text
    assert "no digital signature" in response.text


def test_verified_manifest_and_missing_release_render_distinctly() -> None:
    status = GovernanceStatus(
        staged=StagedManifestStatus(
            seed_dir="/srv/corpus",
            trusted_key="/srv/trust/key.pem",
            verdict="verified",
            declared_signature_status="verified",
            manifest_id="bddk-job-corpus-2026-08-26",
            manifest_sha256="49" * 32,
            signing_key_fingerprint_sha256="ab" * 32,
        ),
        active=ActiveReleaseStatus(release=None),
    )
    response = _client(status).get("/governance")

    assert response.status_code == 200
    assert "İmza doğrulandı" in response.text
    assert "Aktif corpus release yok." in response.text


def test_failures_are_surfaced_and_page_stays_read_only() -> None:
    status = GovernanceStatus(
        staged=StagedManifestStatus(
            seed_dir="/srv/corpus",
            trusted_key=None,
            verdict="failed",
            declared_signature_status="verified",
            error="corpus manifest checksum mismatch",
        ),
        active=ActiveReleaseStatus(error="Active corpus release evidence could not be verified."),
    )
    response = _client(status).get("/governance")

    assert response.status_code == 200
    assert "Doğrulama başarısız" in response.text
    assert "corpus manifest checksum mismatch" in response.text
    assert "Active corpus release evidence could not be verified." in response.text
    # Read-only by design: the panel must never grow signing controls.
    assert "<form" not in response.text.lower()
    assert "<button" not in response.text.lower()


def test_navigation_links_to_the_panel() -> None:
    status = GovernanceStatus(
        staged=StagedManifestStatus(seed_dir="/srv/corpus", trusted_key=None, verdict="unsigned"),
        active=ActiveReleaseStatus(release=None),
    )
    response = _client(status).get("/governance")
    assert 'href="/governance"' in response.text
    assert "İmza Durumu" in response.text
