"""Record the owner-accepted text identity for BDDK guide 943.

No applicability date is claimed. Section database ids are left empty; they
are assigned only when this bundle is imported into a database that already
holds this exact repaired text. This script does not open a database.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    CanonicalInstrument,
    ConsolidationState,
    LegalVersion,
    LegalVersionBundle,
    ProvisionIdentity,
    ProvisionOccurrence,
    SourceArtifact,
    SourceBlob,
    ValidationRecord,
    ValidationState,
    VersionEvents,
    artifact_id_for,
    blob_id_for,
    bundle_id_for,
    canonical_bundle_sha256,
    evidence_id_for,
    instrument_id_for,
    legal_version_id_for,
    load_legal_version_bundle,
    provision_id_for,
)
from bddk_mcp.store.section_index import extract_document_sections

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "seed_data" / "documents.json"
PDF = Path("/tmp/bddk-943/download.bin")
OUT = ROOT / "docs" / "evidence" / "legal-versions" / "943.json"
PDF_SHA = "28ca83020bffb0d533ec4cd51c547fc9afeec1cda34238732dad15398fff67fc"
TEXT_SHA = "6936b7a5971f5210b7eb10ff628d8f5505219a9476ef41f1c735bc119cc4f0ec"
SOURCE = "https://www.bddk.org.tr/Mevzuat/DokumanGetir/943"
ACQUIRED = datetime(2026, 9, 23, 18, 40, tzinfo=UTC)
REVIEWED = datetime(2026, 9, 23, 18, 59, 49, tzinfo=UTC)
METHOD = (
    "Owner accepted the repaired Markdown as the normalized text of official PDF 943. "
    "Only running headers and printed page numbers were removed. "
    "No publication, effective, or applicability date is claimed."
)


def _review(statement: str) -> ValidationRecord:
    return ValidationRecord(
        state=ValidationState.VALIDATED,
        validated_by="project_owner",
        validated_at=REVIEWED,
        method=METHOD,
        review_record_sha256=hashlib.sha256(statement.encode()).hexdigest(),
    )


def build() -> LegalVersionBundle:
    pdf = PDF.read_bytes()
    if hashlib.sha256(pdf).hexdigest() != PDF_SHA:
        raise SystemExit("PDF bytes do not match the accepted hash")
    document = next(item for item in json.loads(DOCS.read_text()) if item["document_id"] == "943")
    if document["content_hash"] != TEXT_SHA:
        raise SystemExit("seed 943 is not the accepted repaired text")
    instrument_id = instrument_id_for(jurisdiction="TR", authority_code="BDDK", identity_key="943")
    blob_id = blob_id_for(content_sha256=PDF_SHA)
    artifact_id = artifact_id_for(blob_id=blob_id, canonical_uri=SOURCE, retrieved_at=ACQUIRED)
    version_id = legal_version_id_for(
        instrument_id=instrument_id,
        version_key=PDF_SHA,
        legal_text_sha256=TEXT_SHA,
    )
    review = _review(f"943 text identity\n{PDF_SHA}\n{TEXT_SHA}\nno applicability date\n")
    provisions: list[ProvisionIdentity] = []
    occurrences: list[ProvisionOccurrence] = []
    for section in extract_document_sections("943", document["markdown_content"]):
        if section.section_type not in {"ilke", "paragraf"}:
            continue
        path = f"{section.section_type}/{section.section_ref}"
        provision_id = provision_id_for(instrument_id=instrument_id, kind=section.section_type, canonical_path=path)
        evidence_id = evidence_id_for(
            artifact_id=artifact_id,
            locator=f"normalized/{path}",
            statement_sha256=section.content_hash,
            authority_level=AuthorityLevel.AUTHORITATIVE,
        )
        provisions.append(
            ProvisionIdentity(
                provision_id=provision_id,
                instrument_id=instrument_id,
                kind=section.section_type,
                canonical_path=path,
            )
        )
        occurrences.append(
            ProvisionOccurrence(
                provision_id=provision_id,
                legal_version_id=version_id,
                provision_text_sha256=section.content_hash,
                evidence={
                    "evidence_id": evidence_id,
                    "artifact_id": artifact_id,
                    "locator": f"normalized/{path}",
                    "statement_sha256": section.content_hash,
                    "authority_level": AuthorityLevel.AUTHORITATIVE,
                },
                validation=review,
            )
        )
    bundle = LegalVersionBundle(
        schema_version=1,
        bundle_id=bundle_id_for(instrument_id=instrument_id),
        bundle_sha256="0" * 64,
        fixture_only=False,
        instrument=CanonicalInstrument(
            instrument_id=instrument_id,
            jurisdiction="TR",
            authority_code="BDDK",
            identity_key="943",
            canonical_title=document["title"],
            instrument_type="rehber",
        ),
        blobs=(SourceBlob(blob_id=blob_id, content_sha256=PDF_SHA),),
        artifacts=(
            SourceArtifact(
                artifact_id=artifact_id,
                blob_id=blob_id,
                canonical_uri=SOURCE,
                source_authority="BDDK",
                media_type="application/pdf",
                retrieved_at=ACQUIRED,
                repository_document_id="943",
            ),
        ),
        provisions=tuple(sorted(provisions, key=lambda item: item.provision_id)),
        versions=(
            LegalVersion(
                legal_version_id=version_id,
                instrument_id=instrument_id,
                version_key=PDF_SHA,
                legal_text_sha256=TEXT_SHA,
                source_artifact_ids=(artifact_id,),
                consolidation_state=ConsolidationState.UNKNOWN,
                validation=review,
                events=VersionEvents(),
                provisions=tuple(sorted(occurrences, key=lambda item: item.provision_id)),
            ),
        ),
    )
    return bundle.model_copy(update={"bundle_sha256": canonical_bundle_sha256(bundle)})


def main() -> int:
    bundle = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    loaded = load_legal_version_bundle(OUT)
    if loaded.bundle_sha256 != bundle.bundle_sha256 or loaded.versions[0].status_assertions:
        raise SystemExit("written bundle failed its own check")
    print(f"wrote {OUT}")
    print(f"provisions={len(loaded.provisions)} status_assertions=0 sha256={loaded.bundle_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
