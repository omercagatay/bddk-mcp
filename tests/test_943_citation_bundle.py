"""Owner-accepted text identity for guide 943. Not an applicability claim."""

from pathlib import Path

from bddk_mcp.regulatory.legal_versions import load_legal_version_bundle

BUNDLE = Path(__file__).parents[1] / "docs" / "evidence" / "legal-versions" / "943.json"


def test_943_text_identity_has_no_applicability_date() -> None:
    bundle = load_legal_version_bundle(BUNDLE)
    version = bundle.versions[0]
    assert bundle.fixture_only is False
    assert bundle.instrument.identity_key == "943"
    assert bundle.instrument.authority_code == "BDDK"
    assert version.legal_text_sha256 == "6936b7a5971f5210b7eb10ff628d8f5505219a9476ef41f1c735bc119cc4f0ec"
    assert version.version_key == "28ca83020bffb0d533ec4cd51c547fc9afeec1cda34238732dad15398fff67fc"
    assert version.status_assertions == ()
    assert version.events.publication is None
    assert version.events.effective is None
    assert {item.canonical_path for item in bundle.provisions} >= {"ilke/6", "paragraf/44"}
    assert all(item.document_section_id is None for item in version.provisions)
