"""Rebind the in-review candidate to the stored corpus text; this NEVER reviews or approves it.

A corpus text repair changes the stored document and provision hashes and therefore the
legal-version identity derived from them. Only those hash-bound fields, the identities
derived from them, the review request's package reference and the whole-document
comparison are rewritten. Legal dates, sources, locators and review states are kept.
A package carrying any completed review is refused: a review covers exact identities
and must be redone for new ones, never carried over.

Run with the project's existing environment: .venv/bin/python <this-file>
"""

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage
from bddk_mcp.regulatory.legal_versions import (
    ValidationState,
    canonical_bundle_sha256,
    event_id_for,
    evidence_id_for,
    legal_version_id_for,
    status_assertion_id_for,
)
from bddk_mcp.regulatory.repository import _review_records
from bddk_mcp.store.section_index import extract_document_sections

root = Path(__file__).resolve().parent
candidate_path = root / "legal_evidence.candidate.json"
request_path = root / "review-request.json"
comparison_path = root / "whole-document-comparison.json"
seed_documents = json.loads((root.parents[3] / "seed_data" / "documents.json").read_text(encoding="utf-8"))
_SEPARATOR = "\n" + "_" * 13 + "\n"
_TRANSFORM = (
    "Whitespace removal; Markdown table syntax of the closing amendment schedule removed (separator row, "
    "cell delimiters and empty cells dropped; a merged header repeated across spanned columns counted once); "
    "removal of the single standalone 13-underscore editorial separator from the official extraction. "
    "Comparison ONLY, not a change to stored text or Citation v1 quotation matching."
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def comparable_stored_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            lines.append(line)
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        cells = [cell for cell in cells if cell]
        lines.extend(cell for index, cell in enumerate(cells) if index == 0 or cell != cells[index - 1])
    return re.sub(r"\s+", "", "\n".join(lines))


previous = candidate_path.read_text(encoding="utf-8")
for bundle in LegalEvidencePackage.model_validate_json(previous).bundles:
    states = {review.state for review in _review_records(bundle)}
    assert states <= {ValidationState.UNVALIDATED, ValidationState.IN_REVIEW}, "reviewed package: prepare a new one"

package = json.loads(previous)
(binding,) = package["bindings"]
(bundle,) = package["bundles"]
(version,) = bundle["versions"]
(occurrence,) = version["provisions"]
assert version["predecessor_version_id"] is None and version["events"]["supersession"] is None
assert occurrence["provision_id"] == binding["provision_id"]

(document,) = [item for item in seed_documents if item["document_id"] == binding["document_id"]]
text = document["markdown_content"]
document_sha256 = sha256(text)
assert document["content_hash"] == document_sha256, "seed document hash does not match its text"
(section,) = [
    item
    for item in extract_document_sections(binding["document_id"], text)
    if item.section_type == binding["section_type"] and item.section_ref == binding["section_ref"]
]
content_sha256 = sha256(section.content)
legal_version_id = legal_version_id_for(
    instrument_id=version["instrument_id"],
    version_key=version["version_key"],
    legal_text_sha256=document_sha256,
)

binding.update(
    legal_version_id=legal_version_id,
    document_sha256=document_sha256,
    content_sha256=content_sha256,
    start_char=section.start_char,
    end_char=section.end_char,
)
version.update(legal_version_id=legal_version_id, legal_text_sha256=document_sha256)
evidence = occurrence["evidence"]
evidence.update(
    statement_sha256=content_sha256,
    evidence_id=evidence_id_for(
        artifact_id=evidence["artifact_id"],
        locator=evidence["locator"],
        statement_sha256=content_sha256,
        authority_level=evidence["authority_level"],
    ),
)
occurrence.update(legal_version_id=legal_version_id, provision_text_sha256=content_sha256)
for event in filter(None, version["events"].values()):
    event.update(
        legal_version_id=legal_version_id,
        event_id=event_id_for(
            legal_version_id=legal_version_id,
            event_type=event["event_type"],
            event_date=date.fromisoformat(event["event_date"]),
            evidence_id=event["evidence"]["evidence_id"],
        ),
    )
for assertion in version["status_assertions"]:
    assertion.update(
        legal_version_id=legal_version_id,
        assertion_id=status_assertion_id_for(
            legal_version_id=legal_version_id,
            status=assertion["status"],
            valid_from=date.fromisoformat(assertion["valid_from"]),
            valid_through=date.fromisoformat(assertion["valid_through"]),
            evidence_id=assertion["evidence"]["evidence_id"],
        ),
    )
version["status_assertions"].sort(key=lambda item: item["assertion_id"])
bundle["bundle_sha256"] = canonical_bundle_sha256(bundle)
LegalEvidencePackage.model_validate(package)

official = (root / "text" / "consolidated-html.txt").read_text(encoding="utf-8")
assert official.count(_SEPARATOR) == 1
comparison = {
    "document_sha256": document_sha256,
    "official_extracted_text_sha256": sha256(official),
    "equal_after_declared_comparison_transform": comparable_stored_text(text)
    == re.sub(r"\s+", "", official.replace(_SEPARATOR, "\n")),
    "transform": _TRANSFORM,
    "punctuation_counts": {"U+0092": text.count("\u0092"), "U+0096": text.count("\u0096")},
    "independent_legal_review": False,
}
comparison_path.write_text(dump(comparison), encoding="utf-8")

rebound = dump(package)
if rebound == previous:
    print("Candidate already bound to the stored text; nothing changed.")
else:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request.update(
        candidate_package_sha256=sha256(rebound),
        supersedes_candidate_package_sha256=sha256(previous),
        legal_version_id=legal_version_id,
    )
    candidate_path.write_text(rebound, encoding="utf-8")
    request_path.write_text(dump(request), encoding="utf-8")
    print(f"Rebound candidate {sha256(previous)} -> {sha256(rebound)}.")
print(
    f"document_sha256={document_sha256} section={section.start_char}-{section.end_char} content_sha256={content_sha256}"
)
print(f"legal_version_id={legal_version_id}")
print(f"whole-document comparison equal: {comparison['equal_after_declared_comparison_transform']}")
print("Every review state is unchanged; this is NOT a legal review or approval.")
