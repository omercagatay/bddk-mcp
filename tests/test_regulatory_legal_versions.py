"""Tests for reconstructed legal-version models and canonical checksum."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    ConsolidationState,
    Evidence,
    Instrument,
    LegalEvent,
    LegalEventSet,
    LegalEventType,
    LegalStatus,
    LegalStatusAssertion,
    LegalVersion,
    LegalVersionBundle,
    Provision,
    ProvisionOccurrence,
    SourceArtifact,
    ValidationRecord,
    ValidationState,
    canonical_bundle_sha256,
)

_VALIDATION = ValidationRecord(
    state=ValidationState.HUMAN_VALIDATED,
    validated_by="reviewer@example.test",
    validated_at=datetime(2026, 8, 1, tzinfo=UTC),
    method="manual-review",
    review_record_sha256="a" * 64,
)


def _evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        artifact_id="art-1",
        locator="page=3;chars=120-240",
        statement_sha256="b" * 64,
        authority_level=AuthorityLevel.OFFICIAL_GAZETTE,
    )


def make_fixture_bundle() -> LegalVersionBundle:
    """Two-version fixture family: v2 supersedes v1, one shared provision."""
    instrument = Instrument(
        instrument_id="inst-tfrs9",
        jurisdiction="TR",
        authority_code="BDDK",
        identity_key="rehber:943",
        canonical_title="TFRS 9 Uygulama Rehberi",
        instrument_type="Rehber",
    )
    artifact = SourceArtifact(
        artifact_id="art-1",
        content_sha256="c" * 64,
        canonical_uri="https://www.bddk.org.tr/example/943.pdf",
        source_authority="bddk.org.tr",
        media_type="application/pdf",
        retrieved_at=datetime(2026, 7, 1, tzinfo=UTC),
        repository_document_id="943",
        fixture_only=True,
    )
    provision = Provision(
        provision_id="prov-943-ilke-5",
        instrument_id="inst-tfrs9",
        kind="ilke",
        canonical_path="ilke/5",
    )
    v1 = LegalVersion(
        legal_version_id="ver-1",
        instrument_id="inst-tfrs9",
        version_key="2024-01",
        legal_text_sha256="d" * 64,
        predecessor_version_id=None,
        consolidation_state=ConsolidationState.AS_ENACTED,
        validation=_VALIDATION,
        events=LegalEventSet(
            publication=LegalEvent(
                event_id="evt-pub-1",
                legal_version_id="ver-1",
                event_type=LegalEventType.PUBLICATION,
                event_date=date(2024, 1, 15),
                evidence=_evidence("ev-1"),
                validation=_VALIDATION,
                target_legal_version_id=None,
            )
        ),
        status_assertions=(
            LegalStatusAssertion(
                assertion_id="as-1",
                legal_version_id="ver-1",
                status=LegalStatus.SUPERSEDED,
                valid_from=date(2024, 1, 15),
                valid_through=date(2026, 3, 1),
                evidence=_evidence("ev-2"),
                validation=_VALIDATION,
            ),
        ),
        provisions=(
            ProvisionOccurrence(
                legal_version_id="ver-1",
                provision_id="prov-943-ilke-5",
                normalized_text_sha256="e" * 64,
                evidence=_evidence("ev-3"),
            ),
        ),
        source_artifact_ids=("art-1",),
    )
    v2 = dataclasses.replace(
        v1,
        legal_version_id="ver-2",
        version_key="2026-03",
        legal_text_sha256="f" * 64,
        predecessor_version_id="ver-1",
        status_assertions=(),
        provisions=(),
        events=LegalEventSet(
            supersession=LegalEvent(
                event_id="evt-sup-2",
                legal_version_id="ver-2",
                event_type=LegalEventType.SUPERSESSION,
                event_date=date(2026, 3, 1),
                evidence=_evidence("ev-4"),
                validation=_VALIDATION,
                target_legal_version_id="ver-1",
            )
        ),
    )
    draft = LegalVersionBundle(
        bundle_id="bundle-tfrs9-1",
        bundle_sha256="0" * 64,
        schema_version="1",
        fixture_only=True,
        instrument=instrument,
        artifacts=(artifact,),
        versions=(v1, v2),
        provisions=(provision,),
    )
    return dataclasses.replace(draft, bundle_sha256=canonical_bundle_sha256(draft))


def test_checksum_is_stable_and_excludes_itself():
    bundle = make_fixture_bundle()
    assert bundle.bundle_sha256 == canonical_bundle_sha256(bundle)
    relabeled = dataclasses.replace(bundle, bundle_sha256="1" * 64)
    assert canonical_bundle_sha256(relabeled) == bundle.bundle_sha256


def test_checksum_changes_when_content_changes():
    bundle = make_fixture_bundle()
    changed = dataclasses.replace(bundle, schema_version="2")
    assert canonical_bundle_sha256(changed) != bundle.bundle_sha256


def test_models_are_immutable():
    bundle = make_fixture_bundle()
    try:
        bundle.bundle_id = "other"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("LegalVersionBundle must be frozen")


def test_repository_module_imports():
    import bddk_mcp.regulatory.repository  # noqa: F401
