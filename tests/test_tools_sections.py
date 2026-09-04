"""Tests for tools/sections.py."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    artifact_id_for,
    blob_id_for,
    evidence_id_for,
    instrument_id_for,
    legal_version_id_for,
    provision_id_for,
)
from bddk_mcp.store.doc_store import StoredDocumentSection, StoredSectionCitationMapping
from bddk_mcp.tools.sections import register


def _capture_tool(deps: Dependencies, name: str):
    mcp = MagicMock()
    register(mcp, deps)
    for call in mcp.tool.return_value.call_args_list:
        fn = call.args[0]
        if fn.__name__ == name:
            return fn
    raise AssertionError(f"{name} not registered")


def _section(
    doc_id: str = "943",
    section_type: str = "ilke",
    section_ref: str = "5",
    content: str = "İlke 5\nBankalar model validasyonunu yapar.",
) -> StoredDocumentSection:
    return StoredDocumentSection(
        doc_id=doc_id,
        section_type=section_type,
        section_ref=section_ref,
        heading="Model validasyonu",
        start_char=10,
        end_char=80,
        content=content,
        content_hash="abc123",
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _citable_section() -> StoredDocumentSection:
    prefix = "Önsöz\n\n"
    source_range = "  **MADDE 5** - Model validasyonu\n(1) Banka bağımsız validasyon yapar.\n\n"
    suffix = "Sonraki madde\n"
    normalized_document = prefix + source_range + suffix
    provision_text = source_range.strip()
    instrument_id = instrument_id_for(
        jurisdiction="TR",
        authority_code="BDDK",
        identity_key="synthetic-citation-tool-contract",
    )
    normalized_document_sha256 = _sha(normalized_document)
    version_key = "synthetic-v1"
    artifact_sha256 = _sha("synthetic source artifact bytes")
    artifact_blob_id = blob_id_for(content_sha256=artifact_sha256)
    source_url = "https://regulator.example.test/synthetic/source.pdf"
    retrieved_at = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    artifact_id = artifact_id_for(
        blob_id=artifact_blob_id,
        canonical_uri=source_url,
        retrieved_at=retrieved_at,
    )
    provision_id = provision_id_for(
        instrument_id=instrument_id,
        kind="madde",
        canonical_path="madde/5",
    )
    evidence_locator = "normalized/madde/5"
    statement_sha256 = _sha(provision_text)
    start = len(prefix)
    return StoredDocumentSection(
        doc_id="citation-contract",
        section_type="madde",
        section_ref="5",
        heading="Model validasyonu",
        start_char=start,
        end_char=start + len(source_range),
        content=provision_text,
        content_hash=statement_sha256,
        normalized_source_range=source_range,
        source_content_hash=normalized_document_sha256,
        citation_mapping=StoredSectionCitationMapping(
            instrument_id=instrument_id,
            instrument_jurisdiction="TR",
            instrument_authority_code="BDDK",
            instrument_identity_key="synthetic-citation-tool-contract",
            legal_version_id=legal_version_id_for(
                instrument_id=instrument_id,
                version_key=version_key,
                legal_text_sha256=normalized_document_sha256,
            ),
            legal_version_key=version_key,
            legal_validation_record_sha256="6" * 64,
            provision_validation_record_sha256="7" * 64,
            artifact_id=artifact_id,
            artifact_blob_id=artifact_blob_id,
            artifact_sha256=artifact_sha256,
            source_url=source_url,
            artifact_retrieved_at=retrieved_at,
            evidence_id=evidence_id_for(
                artifact_id=artifact_id,
                locator=evidence_locator,
                statement_sha256=statement_sha256,
                authority_level=AuthorityLevel.AUTHORITATIVE,
            ),
            evidence_locator=evidence_locator,
            evidence_statement_sha256=statement_sha256,
            provision_id=provision_id,
            provision_kind="madde",
            provision_path="madde/5",
        ),
    )


def test_register_exposes_section_tools():
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=MagicMock(), client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {"get_document_section", "search_document_sections"}


@pytest.mark.asyncio
async def test_get_document_section_returns_exact_match():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section()])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref="5")

    assert "Document ID: 943" in out
    assert "Section: ilke 5" in out
    assert "Model validasyonu" in out
    assert "Bankalar model validasyonunu yapar." in out
    doc_store.get_document_section.assert_awaited_once_with(
        "943", section_type="ilke", section_ref="5", heading=None, limit=11
    )


@pytest.mark.asyncio
async def test_exact_validated_section_emits_same_citation_in_structured_and_text_channels():
    section = _citable_section()
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")(
        section.doc_id,
        section_type="madde",
        section_ref="5",
    )

    citation = result.structuredContent["evidence"][0]["citation"]
    assert citation["schema_version"] == "1.0"
    assert citation["citation_id"] in result.text
    assert citation["normalized_document_sha256"] in result.text
    assert citation["provision_text_sha256"] in result.text
    assert citation["locator"]["normalized_range_sha256"] in result.text
    assert citation["excerpt_sha256"] in result.text
    assert "not source PDF pages" in result.text
    assert citation["source_url"] == result.structuredContent["evidence"][0]["source_url"]
    assert not any("citation_v1_unavailable" in warning for warning in result.structuredContent["warnings"])


@pytest.mark.asyncio
async def test_exact_section_without_validated_mapping_reports_citation_unavailable():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section()])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")("943", section_type="ilke", section_ref="5")

    assert "citation" not in result.structuredContent["evidence"][0]
    assert any(
        "citation_v1_unavailable_no_validated_mapping" in warning for warning in result.structuredContent["warnings"]
    )


@pytest.mark.asyncio
async def test_inconsistent_validated_mapping_fails_closed_without_a_citation():
    section = _citable_section().model_copy(update={"normalized_source_range": "different normalized source"})
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")(
        section.doc_id,
        section_type="madde",
        section_ref="5",
    )

    assert "citation" not in result.structuredContent["evidence"][0]
    assert any(
        "citation_v1_unavailable_reconstruction_mismatch" in warning for warning in result.structuredContent["warnings"]
    )


@pytest.mark.asyncio
async def test_get_document_section_surfaces_configured_failure_before_content():
    section = _section("903", "madde", "1", "MADDE 1 - Temiz görünen mevzuat metni.")
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("903", section_type="madde", section_ref="1")

    assert "Quality: fail" in out
    assert "configured_quality_failure" in out
    assert "listed in the configured quality-failure registry" in out
    assert out.index("Quality warning") < out.index("MADDE 1")


@pytest.mark.asyncio
async def test_get_document_section_accepts_integer_section_ref():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section()])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref=5)

    assert "Section: ilke 5" in out
    doc_store.get_document_section.assert_awaited_once_with(
        "943", section_type="ilke", section_ref="5", heading=None, limit=11
    )


@pytest.mark.asyncio
async def test_get_document_section_accepts_dotted_outline_section_ref():
    """The numbered-paragraph fallback emits dotted refs ("2.1") and the tool
    prints them in its own disambiguation listing, so they must be accepted
    back as input to the same tool."""
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section("1135", "paragraf", "2.1", "Açıklama metni.")])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("1135", section_type="paragraf", section_ref="2.1")

    assert "Section: paragraf 2.1" in out
    doc_store.get_document_section.assert_awaited_once_with(
        "1135", section_type="paragraf", section_ref="2.1", heading=None, limit=11
    )


@pytest.mark.asyncio
async def test_get_document_section_resolves_bare_mevzuat_alias():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(
        side_effect=[[], [_section("mevzuat_22599", "madde", "9", "MADDE 9 - Karşılıklar")]]
    )
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("22599", section_type="madde", section_ref="9")

    assert "Document ID: mevzuat_22599" in out
    assert [call.args[0] for call in doc_store.get_document_section.await_args_list] == [
        "22599",
        "mevzuat_22599",
    ]


@pytest.mark.asyncio
async def test_get_document_section_no_match_suggests_search():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="madde", section_ref="99")

    assert "No section found" in out
    assert "search_document_sections" in out
    assert "943 madde 99" in out


@pytest.mark.asyncio
async def test_get_document_section_integer_ref_no_match_suggests_search():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="madde", section_ref=99)

    assert "No section found" in out
    assert "943 madde 99" in out


@pytest.mark.asyncio
async def test_get_document_section_disambiguates_duplicate_matches():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(
        return_value=[
            _section(content="İlke 5\nBirinci eşleşme."),
            _section(content="İlke 5\nİkinci eşleşme."),
        ]
    )
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref="5")

    assert "Multiple sections matched" in out
    assert "start_char=10" in out
    assert "İkinci eşleşme" in out


@pytest.mark.asyncio
async def test_search_document_sections_outputs_ranked_sections():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    doc_store.search_document_sections = AsyncMock(
        return_value=[
            _section("943", "ilke", "5", "İlke 5\nModel validasyonu yapılır."),
            _section("mevzuat_22599", "madde", "9", "MADDE 9\nTFRS 9 karşılık ayrılır."),
        ]
    )
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool("943 İlke 5 model validasyonu", limit=5)

    assert "Found 2 section result(s)" in out
    assert "943 — ilke 5" in out
    assert "mevzuat_22599 — madde 9" in out
    assert "Model validasyonu yapılır" in out
    doc_store.search_document_sections.assert_awaited_once_with(
        "943 İlke 5 model validasyonu", document_id="943", section_type="ilke", limit=5
    )


@pytest.mark.asyncio
async def test_search_document_sections_surfaces_configured_failure():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    doc_store.search_document_sections = AsyncMock(
        return_value=[_section("903", "madde", "1", "MADDE 1 - Temiz görünen mevzuat metni.")]
    )
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool("örnek", limit=1)

    assert "Quality: fail" in out
    assert "configured_quality_failure" in out
    assert "listed in the configured quality-failure registry" in out


@pytest.mark.asyncio
async def test_search_document_sections_boosts_exact_legal_reference():
    exact = _section("943", "ilke", "5", "İlke 5\nExact legal reference match.")
    fts = _section("943", "ilke", "6", "İlke 6\nFTS result.")
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[exact])
    doc_store.search_document_sections = AsyncMock(return_value=[fts, exact])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool("943 İlke 5 model validasyonu", limit=5)

    assert out.index("ilke 5") < out.index("ilke 6")
    assert out.count("943 — ilke 5") == 1
    doc_store.get_document_section.assert_awaited_once_with("943", section_type="ilke", section_ref="5", limit=5)


@pytest.mark.asyncio
async def test_exact_section_returns_a_bounded_explicitly_partial_body():
    content = "A" * 40_000
    section = _section(content=content).model_copy(update={"end_char": 40_010})
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")("943", section_type="ilke", section_ref="5")

    structured = result.structuredContent
    assert structured["status"] == "partial"
    assert len(structured["results"][0]["content"]) == 30_000
    assert structured["results"][0]["content_truncated"] is True
    assert structured["results"][0]["excerpt_start_char"] == 10
    assert 29_000 < structured["results"][0]["excerpt_end_char"] <= 30_010
    assert any("bounded excerpts" in warning for warning in structured["warnings"])


@pytest.mark.asyncio
async def test_section_search_centres_a_bounded_excerpt_on_the_query():
    content = "A" * 35_000 + "eşsizhedef" + "B" * 5_000
    section = _section(content=content).model_copy(update={"end_char": len(content) + 10})
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    doc_store.search_document_sections = AsyncMock(return_value=[section])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "search_document_sections")("eşsizhedef", limit=1)

    item = result.structuredContent["results"][0]
    assert result.structuredContent["status"] == "partial"
    assert len(item["content"]) <= 2_000
    assert "eşsizhedef" in item["content"]
    assert "eşsizhedef" in result.text
    assert item["content_truncated"] is True
    assert item["excerpt_start_char"] > 10


@pytest.mark.asyncio
async def test_search_visible_text_includes_a_limit_past_the_old_220_char_preview():
    content = (
        "MADDE 4- (1) Bankacılık hesaplarından kaynaklanan faiz oranı riski standart rasyosu, "
        "ekonomik değer değişimi risk tutarının ana sermayeye bölünmesi suretiyle hesaplanır. "
        "(2) Konsolide ve konsolide olmayan bankacılık hesaplarından kaynaklanan faiz oranı riski "
        "standart rasyosu %15’i aşamaz. "
        "(3) Katılma hesabı kaynaklı olanlar Kurulca belirlenecek oranda dikkate alınır."
    )
    assert len(" ".join(content.split())) > 220
    section = _section("mevzuat_42628", "madde", "4", content)
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    doc_store.search_document_sections = AsyncMock(return_value=[section])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    out = await _capture_tool(deps, "search_document_sections")("faiz oranı riski aşamaz", limit=1)

    assert "%15" in out


@pytest.mark.asyncio
async def test_bare_section_lookup_caps_disambiguation_results():
    sections = [
        _section(section_ref=str(index), content=f"Madde {index}").model_copy(
            update={"start_char": index * 100, "end_char": index * 100 + 20}
        )
        for index in range(11)
    ]
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=sections)
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")("943")

    assert result.structuredContent["status"] == "partial"
    assert len(result.structuredContent["results"]) == 10
    assert len(result.structuredContent["evidence"]) == 10
    assert "additional matches omitted" in result.text


@pytest.mark.asyncio
async def test_search_document_sections_uses_loose_fallback_when_strict_misses():
    query = "Bilgi Sistemleri ve İş Süreçleri Bağımsız Denetimi denetim teknikleri dış teyit tetkik gözlem"
    target = _section(
        "mevzuat_39257",
        "madde",
        "31",
        "MADDE 31 - Denetim teknikleri\nDenetçi; tetkik, gözlem ve yeniden uygulama tekniklerini kullanır.",
    )
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])

    async def search_side_effect(search_query, *, document_id=None, section_type=None, limit=10):
        if search_query == query:
            return []
        if search_query in {"denetim", "teknikleri", "tetkik", "gözlem"}:
            return [target]
        return []

    doc_store.search_document_sections = AsyncMock(side_effect=search_side_effect)
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool(query, limit=5)

    assert "Found 1 section result(s)" in out
    assert "mevzuat_39257 — madde 31" in out
    assert "Denetim teknikleri" in out
    assert doc_store.search_document_sections.await_args_list[0].kwargs == {
        "document_id": None,
        "section_type": None,
        "limit": 5,
    }


@pytest.mark.asyncio
async def test_loose_fallback_ranks_operative_madde_ahead_of_govde_gecici_and_ipc():
    query = "likidite yeterlilik oranı asgari oran yaptırım idari para cezası"
    lyo = _section(
        "mevzuat_10749",
        "madde",
        "13",
        "MADDE 13 - Asgari likidite yeterlilik oranı yüzde yüzden az olamaz.",
    ).model_copy(update={"content_hash": "lyo13", "start_char": 9000, "heading": "Asgari likidite yeterliliği oranı"})
    gecici = _section(
        "mevzuat_10749",
        "gecici_madde",
        "1",
        "GEÇİCİ MADDE 1 - likidite yeterlilik oranı yüzde beşten az olamaz.",
    ).model_copy(update={"content_hash": "g1", "start_char": 100})
    ipc = _section(
        "mevzuat_5464",
        "madde",
        "35",
        "MADDE 35 - idari para cezası. aykırılık tutarının yüzde biri oranına kadar.",
    ).model_copy(update={"content_hash": "ipc35", "start_char": 10})
    glue = {"idari", "para", "oran", "oranı", "asgari", "yaptırım", "cezası", "yüzde"}
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])

    async def search_side_effect(search_query, *, document_id=None, section_type=None, limit=10):
        if search_query == query:
            return []
        if search_query in glue:
            raise AssertionError(f"glue term {search_query!r} must not be issued as standalone FTS")
        if search_query in {"likidite", "yeterlilik"}:
            return [gecici, ipc, lyo]
        return []

    doc_store.search_document_sections = AsyncMock(side_effect=search_side_effect)
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    out = await _capture_tool(deps, "search_document_sections")(query, limit=4)

    assert out.index("mevzuat_10749 — madde 13") < out.index("gecici_madde 1")
    assert out.index("mevzuat_10749 — madde 13") < out.index("mevzuat_5464 — madde 35")
