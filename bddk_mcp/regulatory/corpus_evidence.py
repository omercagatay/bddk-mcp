"""Portable, manifest-bound legal evidence for the existing dated-status resolver.

This module admits recorded evidence, not legal approval. A signature does not
promote a review state or widen the date interval the reviewer actually supplied.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bddk_mcp.corpus_coordination import acquire_corpus_mutation_lock
from bddk_mcp.corpus_manifest import (
    CorpusArtifact,
    CorpusManifestError,
    CorpusManifestValidation,
    _artifact_path,
    _bounded_regular_file,
)
from bddk_mcp.regulatory.legal_versions import (
    _MAX_BUNDLE_BYTES,
    LegalVersionBundle,
    _reject_duplicate_keys,
    canonical_bundle_sha256,
)
from bddk_mcp.regulatory.repository import (
    _COLUMN_TYPES,
    _review_records,
    import_legal_version_bundle_on_connection,
    project_bundle_rows,
)
from bddk_mcp.regulatory.text_profile import POSTGRES_PROVISION_BOUNDARY_WHITESPACE_V1

_SHA256 = r"^[0-9a-f]{64}$"


class LegalEvidenceMembershipError(RuntimeError):
    """The signed legal row set and database differ (not an I/O or binding failure)."""


class SectionBinding(BaseModel):
    """Exact normalized source identity, independent of a database's sequence IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    legal_version_id: str = Field(pattern=r"^ver_sha256_[0-9a-f]{64}$")
    provision_id: str = Field(pattern=r"^prov_sha256_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1, max_length=500)
    document_sha256: str = Field(pattern=_SHA256)
    section_type: str = Field(min_length=1, max_length=100)
    section_ref: str = Field(min_length=1, max_length=500)
    start_char: int = Field(strict=True, ge=0)
    end_char: int = Field(strict=True, gt=0)
    content_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _ordered_span(self):
        if self.end_char <= self.start_char:
            raise ValueError("section binding requires a nonempty forward span")
        return self


class LegalEvidencePackage(BaseModel):
    """One optional signed artifact; legacy manifests continue to require empty legal state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    bundles: tuple[LegalVersionBundle, ...] = Field(min_length=1, max_length=100)
    bindings: tuple[SectionBinding, ...] = Field(max_length=10000)

    @model_validator(mode="after")
    def _portable_production_contract(self):
        instruments = [bundle.instrument.instrument_id for bundle in self.bundles]
        if len(instruments) != len(set(instruments)):
            raise ValueError("duplicate instrument family in legal evidence package")
        bindings = {(item.legal_version_id, item.provision_id): item for item in self.bindings}
        if len(bindings) != len(self.bindings):
            raise ValueError("duplicate section binding")
        required = set()
        for bundle in self.bundles:
            if bundle.fixture_only or any(artifact.fixture_only for artifact in bundle.artifacts):
                raise ValueError("fixture evidence cannot be published")
            if bundle.bundle_sha256 != canonical_bundle_sha256(bundle):
                raise ValueError("legal bundle checksum does not match")
            artifacts = {artifact.artifact_id: artifact for artifact in bundle.artifacts}
            for version in bundle.versions:
                for occurrence in version.provisions:
                    if occurrence.document_section_id is not None:
                        raise ValueError("database-local section IDs cannot be signed as portable evidence")
                    artifact = artifacts[occurrence.evidence.artifact_id]
                    if artifact.repository_document_id is None:
                        continue
                    key = (version.legal_version_id, occurrence.provision_id)
                    required.add(key)
                    binding = bindings.get(key)
                    if (
                        binding is None
                        or binding.document_id != artifact.repository_document_id
                        or binding.document_sha256 != version.legal_text_sha256
                        or binding.content_sha256 != occurrence.provision_text_sha256
                    ):
                        raise ValueError("section binding does not match its signed document, version and provision")
        if set(bindings) != required:
            raise ValueError("section bindings must cover exactly the mapped provisions")
        return self


def assert_material_coverage(package: LegalEvidencePackage, artifacts: list[CorpusArtifact]) -> set[str]:
    """Require raw source and completed-review bytes, not unbacked hash strings."""
    required = {blob.content_sha256 for bundle in package.bundles for blob in bundle.blobs}
    required.update(
        review.review_record_sha256
        for bundle in package.bundles
        for review in _review_records(bundle)
        if review.review_record_sha256 is not None
    )
    available = {artifact.sha256 for artifact in artifacts if artifact.role == "other"}
    if not required <= available:
        raise RuntimeError("Legal source or review material is not covered by the signed manifest.")
    return required


def _read_bound_artifact(root: Path, artifact: CorpusArtifact) -> bytes:
    try:
        path = _artifact_path(root.resolve(), artifact.path)
        payload = _bounded_regular_file(path, label="legal material", maximum_bytes=artifact.bytes)
    except CorpusManifestError:
        raise RuntimeError("Signed legal evidence material could not be read safely.") from None
    if len(payload) != artifact.bytes or hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise RuntimeError("Legal evidence artifact changed after manifest validation.")
    return payload


def load_legal_evidence(root: Path, validation: CorpusManifestValidation) -> LegalEvidencePackage | None:
    """Load an optional package only after signature verification, rechecking its bytes.

    The caller obtains validation through load_and_validate_corpus_manifest with
    the external trust anchor; a prior file check is never a trusted byte handoff.
    """
    artifacts = validation.manifest.artifacts
    selected = [artifact for artifact in artifacts if artifact.role == "legal_evidence"]
    if not selected:
        return None
    if not validation.signature_sha256:
        raise RuntimeError("Legal evidence admission requires a verified signature.")
    if len(selected) != 1:
        raise RuntimeError("Exactly one legal evidence artifact is required.")
    payload = _read_bound_artifact(root, selected[0])
    try:
        mapping = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        if isinstance(mapping, dict) and isinstance(mapping.get("bundles"), list):
            if any(
                len(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode()) > _MAX_BUNDLE_BYTES
                for bundle in mapping["bundles"]
            ):
                raise RuntimeError("Legal evidence bundle exceeds the existing family size limit.")
        package = LegalEvidencePackage.model_validate(mapping)
    except (ValueError, ValidationError, TypeError):
        raise RuntimeError("Legal evidence artifact failed schema validation.") from None
    reviewed_at = validation.manifest.freshness.scope_reviewed_at
    for bundle in package.bundles:
        timestamps = [artifact.retrieved_at for artifact in bundle.artifacts]
        timestamps.extend(review.validated_at for review in _review_records(bundle) if review.validated_at is not None)
        if any(timestamp > reviewed_at for timestamp in timestamps):
            raise RuntimeError("Legal material or validation postdates the declared corpus scope review.")
    required = assert_material_coverage(package, artifacts)
    for artifact in artifacts:
        if artifact.role == "other" and artifact.sha256 in required:
            _read_bound_artifact(root, artifact)
    return package


async def resolve_legal_bundles(connection, package: LegalEvidencePackage) -> tuple[LegalVersionBundle, ...]:
    """Resolve exact source identities under the caller's transaction/lock.

    A database-local checksum is derived, not mistaken for the portable signed
    checksum. Verifier and importer independently repeat this deterministic step.
    No review state, claim date or provenance is changed.
    """
    section_ids = {}
    for binding in package.bindings:
        rows = await connection.fetch(
            f"""
            SELECT section.id
            FROM public.document_sections AS section
            JOIN public.documents AS document ON document.document_id = section.doc_id
            WHERE section.doc_id = $1 AND document.content_hash = $2
              AND section.section_type = $3 AND section.section_ref = $4
              AND section.start_char = $5 AND section.end_char = $6
              AND section.content_hash = $7 AND section.source_content_hash = $2
              AND document.content_hash = pg_catalog.encode(
                  pg_catalog.sha256(pg_catalog.convert_to(document.markdown_content, 'UTF8')), 'hex')
              AND section.content_hash = pg_catalog.encode(
                  pg_catalog.sha256(pg_catalog.convert_to(section.content, 'UTF8')), 'hex')
              AND section.content = pg_catalog.btrim(
                  pg_catalog.substr(document.markdown_content, section.start_char + 1,
                                    section.end_char - section.start_char),
                  {POSTGRES_PROVISION_BOUNDARY_WHITESPACE_V1})
            LIMIT 2
            """,
            binding.document_id,
            binding.document_sha256,
            binding.section_type,
            binding.section_ref,
            binding.start_char,
            binding.end_char,
            binding.content_sha256,
        )
        if len(rows) != 1:
            raise RuntimeError("Signed legal evidence does not identify one exact section; import/publication refused.")
        section_ids[(binding.legal_version_id, binding.provision_id)] = rows[0]["id"]
    resolved = []
    for bundle in package.bundles:
        mapping = bundle.model_dump(mode="json")
        for version in mapping["versions"]:
            for occurrence in version["provisions"]:
                occurrence["document_section_id"] = section_ids.get(
                    (version["legal_version_id"], occurrence["provision_id"])
                )
        mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
        resolved.append(LegalVersionBundle.model_validate(mapping))
    return tuple(resolved)


async def import_legal_evidence(connection, package: LegalEvidencePackage) -> list[str]:
    """Import all families atomically; the CLI supplies the verified owner identity.

    This does not sign, promote a review, stage or activate a release. Extra legal
    history is refused rather than deleted to manufacture exact membership.
    """
    async with connection.transaction():
        await acquire_corpus_mutation_lock(connection)
        bundles = await resolve_legal_bundles(connection, package)
        try:
            await assert_legal_evidence_membership(connection, package)
        except LegalEvidenceMembershipError:
            for bundle in bundles:
                await import_legal_version_bundle_on_connection(connection, bundle, imported_by="signed-corpus")
            await assert_legal_evidence_membership(connection, package)
        # Exact no-op imports must not bump the corpus epoch and revoke a release.
    return [bundle.bundle_sha256 for bundle in bundles]


def _canonical_row(record: dict) -> str:
    record = dict(record)
    if isinstance(record.get("member_manifest"), str):
        record["member_manifest"] = json.loads(record["member_manifest"])
    return json.dumps(record, sort_keys=True, ensure_ascii=False, default=lambda value: value.isoformat())


async def assert_legal_evidence_membership(connection, package: LegalEvidencePackage) -> None:
    """Compare every signed legal fact, including review state, not just row counts.

    The caller must hold the corpus publication lock. Unrepresented relations or
    prior import history are refused; admission never deletes evidence to fit.
    """
    expected = {table: set() for table in _COLUMN_TYPES}
    for bundle in await resolve_legal_bundles(connection, package):
        for table, records in project_bundle_rows(bundle, imported_by="signed-corpus").items():
            expected[table].update(_canonical_row(record) for record in records)
    for table, rows in expected.items():
        # Both identifiers are fixed repository constants; signed input is never SQL.
        columns = ", ".join(_COLUMN_TYPES[table])
        actual = await connection.fetch(f"SELECT {columns} FROM {table} LIMIT {len(rows) + 1}")
        if len(actual) != len(rows) or {_canonical_row(dict(record)) for record in actual} != rows:
            raise LegalEvidenceMembershipError(
                "Legal database membership is not exactly represented by the signed evidence artifact."
            )
