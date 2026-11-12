"""Tests for tools/sections.py."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import call as mock_call

import pytest

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.store.doc_store import StoredDocumentSection
from bddk_mcp.store.section_index import GOVDE_HEADING
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


def test_register_exposes_section_tools():
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=MagicMock(), client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {"get_document_section", "search_document_sections"}


@pytest.mark.asyncio
async def test_get_document_section_returns_exact_match():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section()])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref="5")

    assert "Document ID: 943" in out
    assert "Section: ilke 5" in out
    assert "Model validasyonu" in out
    assert "Bankalar model validasyonunu yapar." in out
    doc_store.get_document_section.assert_awaited_once_with(
        "943", section_type="ilke", section_ref="5", heading=None, limit=11, start_char=None
    )


@pytest.mark.asyncio
async def test_get_document_section_surfaces_configured_failure_before_content(historical_quality_registry):
    section = _section("903", "madde", "1", "MADDE 1 - Temiz görünen mevzuat metni.")
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref=5)

    assert "Section: ilke 5" in out
    doc_store.get_document_section.assert_awaited_once_with(
        "943", section_type="ilke", section_ref="5", heading=None, limit=11, start_char=None
    )


@pytest.mark.asyncio
async def test_get_document_section_accepts_dotted_outline_section_ref():
    """The numbered-paragraph fallback emits dotted refs ("2.1") and the tool
    prints them in its own disambiguation listing, so they must be accepted
    back as input to the same tool."""
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section("1135", "paragraf", "2.1", "Açıklama metni.")])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("1135", section_type="paragraf", section_ref="2.1")

    assert "Section: paragraf 2.1" in out
    doc_store.get_document_section.assert_awaited_once_with(
        "1135", section_type="paragraf", section_ref="2.1", heading=None, limit=11, start_char=None
    )


@pytest.mark.asyncio
async def test_get_document_section_resolves_bare_mevzuat_alias():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(
        side_effect=[[], [_section("mevzuat_22599", "madde", "9", "MADDE 9 - Karşılıklar")]]
    )
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="madde", section_ref="99")

    assert "No section found" in out
    assert "search_document_sections" in out
    assert "943 madde 99" in out


@pytest.mark.asyncio
async def test_get_document_section_integer_ref_no_match_suggests_search():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    doc_store.has_document = AsyncMock(return_value=True)
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool("943 İlke 5 model validasyonu", limit=5)

    assert "Found 2 section result(s)" in out
    assert "943 — ilke 5" in out
    assert "mevzuat_22599 — madde 9" in out
    assert "Model validasyonu yapılır" in out
    # Strict pass first; with fewer than `limit` hits it is topped up by the prefix-stem
    # pass in the same document and section scope (the mock's duplicates are dropped).
    assert doc_store.search_document_sections.await_args_list[0] == mock_call(
        "943 İlke 5 model validasyonu", document_id="943", section_type="ilke", limit=5
    )
    assert doc_store.search_document_sections.await_args_list[1] == mock_call(
        "943 İlke 5 model validasyonu", document_id="943", section_type="ilke", limit=7, prefix_stem=True
    )
    assert doc_store.search_document_sections.await_count == 2


@pytest.mark.asyncio
async def test_search_document_sections_surfaces_configured_failure(historical_quality_registry):
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    doc_store.search_document_sections = AsyncMock(
        return_value=[_section("903", "madde", "1", "MADDE 1 - Temiz görünen mevzuat metni.")]
    )
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    doc_store.has_document = AsyncMock(return_value=True)
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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

    async def search_side_effect(search_query, *, document_id=None, section_type=None, limit=10, **_kwargs):
        if search_query == query:
            return []
        if search_query in {"denetim", "teknikleri", "tetkik", "gözlem"}:
            return [target]
        return []

    doc_store.search_document_sections = AsyncMock(side_effect=search_side_effect)
    doc_store.hydrate_source_ranges = AsyncMock(side_effect=lambda sections: sections)
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

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

    async def search_side_effect(search_query, *, document_id=None, section_type=None, limit=10, **_kwargs):
        if search_query == query:
            return []
        if search_query in glue:
            raise AssertionError(f"glue term {search_query!r} must not be issued as standalone FTS")
        if search_query in {"likidite", "yeterlilik"}:
            return [gecici, ipc, lyo]
        return []

    doc_store.search_document_sections = AsyncMock(side_effect=search_side_effect)
    doc_store.hydrate_source_ranges = AsyncMock(side_effect=lambda sections: sections)
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    out = await _capture_tool(deps, "search_document_sections")(query, limit=4)

    assert out.index("mevzuat_10749 — madde 13") < out.index("gecici_madde 1")
    assert out.index("mevzuat_10749 — madde 13") < out.index("mevzuat_5464 — madde 35")


def test_excerpt_offsets_survive_turkish_capitals_and_leading_whitespace():
    from bddk_mcp.tools.sections import _section_excerpt

    content = "İ" * 4000 + "BKZ genel açıklama. " * 100 + "BKZ model\nvalidasyonu" + " son." * 800
    source_range = " \n  " + content + "\n"
    section = _section(content=content).model_copy(
        update={
            "end_char": 10 + len(source_range),
            "normalized_source_range": source_range,
        }
    )
    excerpt, truncated, start, end = _section_excerpt(section, max_chars=2000, query="BKZ model validasyonu")
    assert truncated
    assert "BKZ model\nvalidasyonu" in excerpt
    # Verbatim contract: the excerpt must be the exact stored characters, not a sanitized rendering.
    assert source_range[start - 10 : end - 10] == excerpt


@pytest.mark.asyncio
async def test_parser_capped_section_is_partial_even_below_tool_excerpt_limit():
    content = "İlke 5 - Metin.\n[BÖLÜM KESİLDİ: içerik kısaltıldı — tam metin için get_bddk_document kullanın]"
    store = MagicMock()
    store.get_document_section = AsyncMock(return_value=[_section(content=content)])
    deps = Dependencies(db=None, doc_store=store, client=None, http=None)
    result = await _capture_tool(deps, "get_document_section")("943", section_type="ilke", section_ref="5")
    assert result.structuredContent["status"] == "partial"
    assert result.structuredContent["results"][0]["content_truncated"]
    assert "bounded excerpts" in result.text


def test_loose_terms_preserve_word_boundaries_and_deduplicate_acronyms():
    from bddk_mcp.tools.sections import _loose_search_terms, _loose_search_text

    assert _loose_search_terms("LGD/THK geriye-dönük test") == ["thk", "geriye", "dönük", "test"]
    assert _loose_search_text("LGD THK ILGDX") == "thk thk ılgdx"


@pytest.mark.asyncio
async def test_loose_acronym_expansion_searches_both_forms_without_rewriting_evidence():
    query = "LGD geriye-dönük test"
    sections = [
        _section("english", "paragraf", "1", "LGD geriye dönük test yapılır."),
        _section("turkish", "paragraf", "1", "THK geriye dönük test yapılır."),
    ]
    store = MagicMock()
    store.get_document_section = AsyncMock(return_value=[])

    async def search(value, **kwargs):
        return [sections[0]] if value == "lgd" else [sections[1]] if value == "thk" else []

    store.search_document_sections = AsyncMock(side_effect=search)
    store.hydrate_source_ranges = AsyncMock(side_effect=lambda sections: sections)
    deps = Dependencies(db=None, doc_store=store, client=None, http=None)
    result = await _capture_tool(deps, "search_document_sections")(query)
    contents = {item["document_id"]: item["content"] for item in result.structuredContent["results"]}
    assert contents == {section.doc_id: section.content for section in sections}
    assert list(contents) == ["english", "turkish"]  # Stable ties, not per-term retrieval order.
    assert all("rank" not in item for item in result.structuredContent["results"])
    assert {"lgd", "thk"} <= {call.args[0] for call in store.search_document_sections.await_args_list}
    assert result.structuredContent["loose_fallback_used"]
    assert "not a legal-confidence score" in result.text


@pytest.mark.asyncio
async def test_inserted_article_request_never_returns_the_base_article_as_exact():
    store = MagicMock()
    store.get_document_section = AsyncMock(return_value=[])
    store.search_document_sections = AsyncMock(return_value=[_section("mevzuat_5411", "madde", "66")])
    store.has_document = AsyncMock(return_value=True)
    deps = Dependencies(db=None, doc_store=store, client=None, http=None)

    result = await _capture_tool(deps, "search_document_sections")("mevzuat_5411 Madde 66/A sistemik")

    store.get_document_section.assert_awaited_once_with(
        "mevzuat_5411", section_type="madde", section_ref="66/a", limit=10
    )
    assert result.structuredContent["status"] == "partial"
    assert "requested exact provision was not found" in result.text


@pytest.mark.asyncio
async def test_missing_exact_reference_does_not_present_lexical_matches_as_complete():
    store = MagicMock()
    store.get_document_section = AsyncMock(return_value=[])
    store.search_document_sections = AsyncMock(return_value=[_section("943", "paragraf", "43")])
    store.has_document = AsyncMock(return_value=True)
    deps = Dependencies(db=None, doc_store=store, client=None, http=None)
    result = await _capture_tool(deps, "search_document_sections")("943 paragraf 999 validasyon")
    assert result.structuredContent["status"] == "partial"
    assert result.structuredContent["section_ref"] == "999"
    assert result.structuredContent["results"][0]["section_ref"] == "43"
    assert "requested exact provision was not found" in result.text
    assert any("requested exact provision was not found" in w for w in result.structuredContent["warnings"])


@pytest.mark.asyncio
async def test_quotation_is_checked_character_for_character_against_the_resolved_section():
    content = "İlke 5\nBankalar model validasyonunu yapar."
    section = _section(content=content).model_copy(
        update={"content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    )
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)
    tool = _capture_tool(deps, "get_document_section")

    exact = await tool("943", section_type="ilke", section_ref="5", quotation="model validasyonunu yapar")
    paraphrased = await tool("943", section_type="ilke", section_ref="5", quotation="model validasyonu yapılır")

    assert exact.structuredContent["answer_assessment"]["quotation_status"] == "exact_reference_match"
    assert exact.structuredContent["answer_assessment"]["basis"] == "local_text"
    assert exact.structuredContent["answer_assessment"]["scope_and_entailment"] == "not_assessed"
    assert exact.structuredContent["status"] == "ok"
    assert paraphrased.structuredContent["answer_assessment"]["quotation_status"] == "not_found"
    assert paraphrased.structuredContent["status"] == "partial"
    assert "Quotation in stored reference (no normalization): not_found" in paraphrased.text


@pytest.mark.asyncio
async def test_quotation_is_unavailable_without_exactly_one_resolved_section():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")("943", section_ref="99", quotation="metin")

    assessment = result.structuredContent["answer_assessment"]
    assert assessment == {
        "basis": "insufficient",
        "quotation_status": "unavailable",
        "gaps": ["exact_section_not_resolved"],
        "scope_and_entailment": "not_assessed",
    }


def _capped_section(legal_text: str) -> StoredDocumentSection:
    from bddk_mcp.store.section_index import TRUNCATION_NOTICE_PREFIX

    content = (
        f"{legal_text}\n\n{TRUNCATION_NOTICE_PREFIX} içerik 25000 karakterden 20000 karaktere kısaltıldı"
        " — tam metin için get_bddk_document kullanın]"
    )
    return _section("mevzuat_1", "ek", "1", content).model_copy(
        update={"content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    )


@pytest.mark.asyncio
async def test_quotation_check_excludes_the_storage_truncation_notice_of_a_capped_section():
    legal_text = "EK 1\nBankalar aşağıdaki tabloyu doldurur."
    section = _capped_section(legal_text)
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    tool = _capture_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None), "get_document_section")

    notice = await tool("mevzuat_1", section_type="ek", section_ref="1", quotation="[BÖLÜM KESİLDİ: içerik 25000")
    spanning = await tool("mevzuat_1", section_type="ek", section_ref="1", quotation="doldurur.\n\n[BÖLÜM")
    legal = await tool("mevzuat_1", section_type="ek", section_ref="1", quotation="tabloyu doldurur.")

    # The notice is never legal text; on a capped reference a miss is inconclusive, not not_found.
    for result in (notice, spanning):
        assert result.structuredContent["answer_assessment"]["quotation_status"] == "unavailable"
        assert result.structuredContent["answer_assessment"]["gaps"] == ["quotation_reference_truncated"]
    assessment = legal.structuredContent["answer_assessment"]
    check = assessment["quotation_check"]
    assert assessment["quotation_status"] == "exact_reference_match"
    assert check["start_char"] == legal_text.index("tabloyu doldurur.")
    assert check["end_char"] == len(legal_text)
    # Bound to exactly the characters compared: the legal text without the notice.
    assert check["reference_sha256"] == hashlib.sha256(legal_text.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_quotation_past_the_cap_of_a_capped_section_is_unavailable_not_not_found():
    legal_text = "EK 1\nBankalar aşağıdaki tabloyu doldurur."
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_capped_section(legal_text)])
    tool = _capture_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None), "get_document_section")

    result = await tool("mevzuat_1", section_type="ek", section_ref="1", quotation="sonraki satırdaki metin")

    assessment = result.structuredContent["answer_assessment"]
    assert assessment["quotation_status"] == "unavailable"
    assert assessment["gaps"] == ["quotation_reference_truncated"]
    check = assessment["quotation_check"]
    assert (check["status"], check["reason"]) == ("unavailable", "reference_truncated")
    assert check["reference_sha256"] == hashlib.sha256(legal_text.encode("utf-8")).hexdigest()
    assert "Quotation in stored reference (no normalization): unavailable" in result.text
    assert "text beyond it was NOT compared" in result.text
    assert "The rest of the provision is in full in get_bddk_document" in result.text
    assert "nested fikra/bent/paragraf rows and/or govde windows" in result.text
    assert "continuation govde rows" not in result.text


@pytest.mark.asyncio
async def test_quotation_of_a_whole_capped_window_is_an_excerpt_not_the_whole_reference():
    legal_text = "EK 1\nBankalar aşağıdaki tabloyu doldurur."
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_capped_section(legal_text)])
    tool = _capture_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None), "get_document_section")

    result = await tool("mevzuat_1", section_type="ek", section_ref="1", quotation=legal_text)

    assessment = result.structuredContent["answer_assessment"]
    assert assessment["quotation_status"] == "exact_reference_match"
    assert assessment["quotation_check"]["extent"] == "excerpt"
    assert "text beyond it was NOT compared" in result.text


@pytest.mark.asyncio
async def test_quotation_check_of_an_uncapped_section_is_unchanged():
    content = "İlke 5\nBankalar model validasyonunu yapar."
    section = _section(content=content).model_copy(
        update={"content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    )
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    tool = _capture_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None), "get_document_section")

    whole = await tool("943", section_type="ilke", section_ref="5", quotation=content)
    missing = await tool("943", section_type="ilke", section_ref="5", quotation="başka metin")

    assert whole.structuredContent["answer_assessment"]["quotation_check"]["extent"] == "whole_reference"
    assert missing.structuredContent["answer_assessment"]["gaps"] == ["quotation_not_found"]
    assert "NOT compared" not in whole.text + missing.text


@pytest.mark.asyncio
async def test_quotation_check_is_unavailable_when_stored_content_no_longer_matches_its_hash():
    section = _capped_section("EK 1\nBankalar aşağıdaki tabloyu doldurur.").model_copy(
        update={"content_hash": "0" * 64}
    )
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[section])
    tool = _capture_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None), "get_document_section")

    result = await tool("mevzuat_1", section_type="ek", section_ref="1", quotation="tabloyu doldurur.")

    assessment = result.structuredContent["answer_assessment"]
    assert assessment["quotation_status"] == "unavailable"
    assert assessment["gaps"] == ["quotation_reference_hash_mismatch"]


def test_loose_section_score_has_no_single_word_bonus():
    from bddk_mcp.tools.sections import _loose_section_score

    terms = ["uyumsuzluk", "model", "validasyonu"]
    weights = dict.fromkeys(terms, 1.0)
    one_word = _section(content="Uyumsuzluk halinde bildirim yapılır.").model_copy(update={"heading": ""})
    all_others = _section(content="Model validasyonu yıllık yapılır.").model_copy(update={"heading": ""})

    assert _loose_section_score(all_others, terms, weights) > _loose_section_score(one_word, terms, weights)


@pytest.mark.asyncio
async def test_ambiguous_fikra_rows_point_to_the_parent_madde():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(
        return_value=[_section("mevzuat_1", "fikra", "2", f"(2) Fıkra metni {n}.") for n in range(3)]
    )
    doc_store.find_parent_articles = AsyncMock(return_value={})
    tool = _capture_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None), "get_document_section")

    result = await tool("mevzuat_1", section_type="fikra", section_ref="2")

    assert "retrieve the parent madde" in result.text


async def _store_inference_fixture(doc_store) -> Dependencies:
    from bddk_mcp.store.doc_store import StoredDocument

    for doc_id, content in (
        ("mevzuat_7001", "MADDE 1 - Kiralama işlemleri TFRS 16 kapsamında muhasebeleştirilir."),
        ("mevzuat_7002", "MADDE 1 - Finansal araçlar TMS 39 hükümlerine göre ölçülür."),
        ("mevzuat_7003", "MADDE 1 - 2024 yılı için kredi kartı asgari ödeme tutarı yeniden belirlenir."),
        ("39", "MADDE 1 - Bağımsız denetim kuruluşlarının yetkilendirilmesi."),
        ("16", "MADDE 1 - Faktoring şirketlerinin kuruluşu."),
        ("2024", "MADDE 1 - Kurul toplantı usulleri."),
        ("943", "İlke 5\nBankalar model validasyonunu bağımsız olarak yapar.\n\nİlke 6\nUzman görüşü belgelenir."),
        ("mevzuat_7004", "MADDE 1 - Bankalar model validasyonunu yıllık yapar."),
    ):
        await doc_store.store_document(
            StoredDocument(document_id=doc_id, title=f"Belge {doc_id}", markdown_content=content)
        )
    return Dependencies(db=None, doc_store=doc_store, client=None, http=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("TFRS 16 kiralama işlemleri", "mevzuat_7001"),
        ("TMS 39 finansal araçlar", "mevzuat_7002"),
        ("2024 yılı kredi kartı asgari ödeme", "mevzuat_7003"),
    ],
)
async def test_standard_numbers_and_years_never_scope_the_section_search(doc_store, query, expected_doc):
    tool = _capture_tool(await _store_inference_fixture(doc_store), "search_document_sections")

    result = await tool(query)

    structured = result.structuredContent
    assert structured["status"] != "no_results", result.text
    assert structured["results"][0]["document_id"] == expected_doc
    assert structured.get("document_id") is None
    assert structured["inferred_document_filter_dropped"] is False
    assert "not applied as a document filter" not in result.text


@pytest.mark.asyncio
async def test_inferred_id_without_a_matching_section_is_dropped_and_retried_unscoped(doc_store):
    tool = _capture_tool(await _store_inference_fixture(doc_store), "search_document_sections")

    result = await tool("39 finansal araçlar")

    structured = result.structuredContent
    assert structured["results"][0]["document_id"] == "mevzuat_7002"
    assert structured.get("document_id") is None
    assert structured["inferred_document_filter_dropped"] is True
    assert (
        "the number 39 in the query was not applied as a document filter "
        "(no section in document 39 matched the query)" in result.text
    )


@pytest.mark.asyncio
async def test_inferred_id_of_an_unstored_document_is_ignored(doc_store):
    tool = _capture_tool(await _store_inference_fixture(doc_store), "search_document_sections")

    result = await tool("5555 madde 1 finansal araçlar")

    structured = result.structuredContent
    assert structured["results"][0]["document_id"] == "mevzuat_7002"
    assert structured.get("document_id") is None
    assert structured["exact_reference_detected"] is False
    assert structured["inferred_document_filter_dropped"] is True
    assert "the number 5555 in the query was not applied as a document filter (no stored document" in result.text


@pytest.mark.asyncio
async def test_bare_document_id_with_a_principle_still_resolves_inside_that_document(doc_store):
    tool = _capture_tool(await _store_inference_fixture(doc_store), "search_document_sections")

    result = await tool("943 İlke 5 model validasyonu")

    structured = result.structuredContent
    assert (structured["results"][0]["document_id"], structured["results"][0]["section_ref"]) == ("943", "5")
    assert {item["document_id"] for item in structured["results"]} == {"943"}
    assert structured["document_id"] == "943"
    assert structured["exact_reference_detected"] is True
    assert structured["inferred_document_filter_dropped"] is False


@pytest.mark.asyncio
async def test_explicit_document_id_stays_a_hard_filter(doc_store):
    tool = _capture_tool(await _store_inference_fixture(doc_store), "search_document_sections")

    result = await tool("finansal araçlar", document_id="39")

    structured = result.structuredContent
    assert structured["status"] == "no_results"
    assert structured["document_id"] == "39"
    assert structured["inferred_document_filter_dropped"] is False
    assert "remove the document_id filter" in result.text


@pytest.mark.asyncio
async def test_no_results_suggests_removing_only_filters_the_caller_passed(doc_store):
    tool = _capture_tool(await _store_inference_fixture(doc_store), "search_document_sections")

    result = await tool("39 eşsizbulunmayanterim")

    assert result.structuredContent["status"] == "no_results"
    assert result.structuredContent["inferred_document_filter_dropped"] is True
    assert "remove the" not in result.text
    assert "the number 39 in the query was not applied as a document filter" in result.text


_CAPPED_BODY = "Söz konusu yükümlülük yerine getirilir ve kayıt altına alınır. " * 500
_CAPPED_DOCUMENT = (
    "MADDE 1 - Amaç\nBu yönetmeliğin amacı.\n\nMADDE 2 - Yükümlülük\n"
    + _CAPPED_BODY
    + "\n\nMADDE 3 - Yürürlük\nBu yönetmelik yayımı tarihinde yürürlüğe girer.\n"
)


async def _store_capped_document(doc_store) -> Dependencies:
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(document_id="mevzuat_9", title="Uzun Yönetmelik", markdown_content=_CAPPED_DOCUMENT)
    )
    return Dependencies(db=None, doc_store=doc_store, client=None, http=None)


def test_truncation_notice_original_length_reads_the_recorded_length():
    from bddk_mcp.store.section_index import truncation_notice_original_length

    notice = "[BÖLÜM KESİLDİ: içerik 40048 karakterden 20000 karaktere kısaltıldı — tam metin için get_bddk_document]"
    assert truncation_notice_original_length(notice) == 40048
    assert truncation_notice_original_length("[BÖLÜM KESİLDİ: içerik kısaltıldı]") is None
    assert truncation_notice_original_length(None) is None


@pytest.mark.asyncio
async def test_capped_section_reports_its_true_extent(doc_store):
    tool = _capture_tool(await _store_capped_document(doc_store), "get_document_section")
    madde_2_start = _CAPPED_DOCUMENT.index("MADDE 2")
    true_end = _CAPPED_DOCUMENT.index("MADDE 3") - 1

    result = await tool("mevzuat_9", section_type="madde", section_ref="2")

    item = result.structuredContent["results"][0]
    assert (item["start_char"], item["end_char"]) == (madde_2_start, madde_2_start + 20_000)
    assert item["stored_span_capped"] is True
    assert item["original_end_char"] == true_end
    assert (
        f"Stored window capped at 20,000 of {true_end - madde_2_start:,} characters; "
        f"the provision continues to code point {true_end}" in result.text
    )
    assert "The rest of the provision is in full in get_bddk_document" in result.text
    assert "following govde rows" not in result.text


@pytest.mark.asyncio
async def test_uncapped_section_reports_no_capped_extent(doc_store):
    tool = _capture_tool(await _store_capped_document(doc_store), "get_document_section")

    result = await tool("mevzuat_9", section_type="madde", section_ref="3")

    item = result.structuredContent["results"][0]
    assert item["stored_span_capped"] is False
    assert item.get("original_end_char") is None
    assert "Stored window capped" not in result.text


@pytest.mark.asyncio
async def test_doc_store_links_govde_rows_to_the_capped_section_they_continue(doc_store):
    await _store_capped_document(doc_store)
    sections = await doc_store.get_document_section("mevzuat_9")
    govde = [section for section in sections if section.section_type == "govde"]
    assert govde, "the overflow of the capped madde must be stored as govde windows"

    continuations = await doc_store.find_capped_continuations(sections)

    true_end = _CAPPED_DOCUMENT.index("MADDE 3") - 1
    assert continuations == {
        (row.doc_id, row.source_content_hash, row.start_char): ("madde", "2", true_end, False) for row in govde
    }


@pytest.mark.asyncio
async def test_doc_store_maps_a_shared_govde_row_to_the_innermost_capped_section(doc_store):
    """Nested capped spans (e.g. paragraf 2 containing capped paragraf 2.2.4) share overflow govde rows."""
    from bddk_mcp.store.section_index import TRUNCATION_NOTICE_PREFIX

    await _store_capped_document(doc_store)
    sections = await doc_store.get_document_section("mevzuat_9")
    outer = next(section for section in sections if (section.section_type, section.section_ref) == ("madde", "2"))
    govde = [section for section in sections if section.section_type == "govde"]
    true_end = _CAPPED_DOCUMENT.index("MADDE 3") - 1
    inner_start = outer.start_char + 100
    inner_length = true_end - inner_start
    inner_content = (
        _CAPPED_DOCUMENT[inner_start : inner_start + 20_000]
        + f"\n\n{TRUNCATION_NOTICE_PREFIX} içerik {inner_length} karakterden 20000 karaktere kısaltıldı"
        " — tam metin için get_bddk_document kullanın]"
    )

    def insert_inner(conn):
        conn.execute(
            "INSERT INTO document_sections (doc_id, section_type, section_ref, heading, start_char, end_char, "
            "content, content_hash, source_content_hash) VALUES (?, ?, ?, '', ?, ?, ?, ?, ?)",
            [
                "mevzuat_9",
                "paragraf",
                "2.2.4",
                inner_start,
                inner_start + 20_000,
                inner_content,
                hashlib.sha256(inner_content.encode("utf-8")).hexdigest(),
                outer.source_content_hash,
            ],
        )

    await doc_store._db.transaction(insert_inner)

    continuations = await doc_store.find_capped_continuations(sections)

    assert govde
    assert continuations == {
        (row.doc_id, row.source_content_hash, row.start_char): ("paragraf", "2.2.4", true_end, False) for row in govde
    }


@pytest.mark.asyncio
async def test_continuation_govde_row_names_its_provision_in_both_section_tools(doc_store):
    deps = await _store_capped_document(doc_store)
    exact = await _capture_tool(deps, "get_document_section")("mevzuat_9", section_type="govde", section_ref="1")
    search = await _capture_tool(deps, "search_document_sections")(
        "yükümlülük kayıt altına alınır", document_id="mevzuat_9", section_type="govde"
    )

    for result in (exact, search):
        item = result.structuredContent["results"][0]
        assert (item["section_type"], item["section_ref"]) == ("govde", "1")
        assert (item["continuation_of_section_type"], item["continuation_of_section_ref"]) == ("madde", "2")
        assert "continuation of madde 2 beyond its 20,000-character stored window" in result.text
        warnings = result.structuredContent["warnings"]
        assert any(
            "this text belongs to madde 2. Cite that provision and verify the full text with get_bddk_document"
            in warning
            for warning in warnings
        )
        assert not any("Do not cite them as a legal provision" in warning for warning in warnings)
    assert "govde remainder — not a legal provision identity" not in search.text


@pytest.mark.asyncio
async def test_genuine_govde_remainder_keeps_the_remainder_warning(doc_store):
    from bddk_mcp.store.doc_store import StoredDocument

    footnote = "Dipnot: Bu yönetmelik değişiklikleri ile birlikte okunmalıdır ve ekleri kurul kararı ile güncellenir."
    await doc_store.store_document(
        StoredDocument(
            document_id="mevzuat_8",
            title="Kısa Yönetmelik",
            markdown_content=f"{footnote}\n\nMADDE 1 - Amaç\nBu yönetmeliğin amacı bankaların denetimidir.\n",
        )
    )
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")("mevzuat_8", section_type="govde", section_ref="1")

    item = result.structuredContent["results"][0]
    assert item.get("continuation_of_section_type") is None
    assert any("Do not cite them as a legal provision" in warning for warning in result.structuredContent["warnings"])
    assert "continuation of" not in result.text


async def _store_repeated_reference_document(doc_store) -> tuple[Dependencies, list[int]]:
    from bddk_mcp.store.doc_store import StoredDocument

    text = (
        "MADDE 1 - Değişiklik\nBirinci değişiklik hükmü uygulanır.\n\n"
        "MADDE 1 - Değişiklik\nİkinci değişiklik hükmü uygulanır.\n"
    )
    await doc_store.store_document(StoredDocument(document_id="mevzuat_802", title="Değişiklik", markdown_content=text))
    starts = [section.start_char for section in await doc_store.get_document_section("mevzuat_802")]
    return Dependencies(db=None, doc_store=doc_store, client=None, http=None), starts


@pytest.mark.asyncio
async def test_start_char_selects_one_of_several_sections_sharing_a_reference(doc_store):
    deps, starts = await _store_repeated_reference_document(doc_store)
    assert len(starts) == 2
    tool = _capture_tool(deps, "get_document_section")

    ambiguous = await tool("mevzuat_802", section_type="madde", section_ref="1", quotation="İkinci değişiklik")
    selected = await tool(
        "mevzuat_802", section_type="madde", section_ref="1", quotation="İkinci değişiklik", start_char=starts[1]
    )

    assert len(ambiguous.structuredContent["results"]) == 2
    assert ambiguous.structuredContent["answer_assessment"]["gaps"] == ["exact_section_not_resolved"]
    assert "repeat the call with start_char=<value>" in ambiguous.text
    assert f"start_char={starts[1]}" in ambiguous.text
    [item] = selected.structuredContent["results"]
    assert item["start_char"] == starts[1]
    assert selected.structuredContent["answer_assessment"]["quotation_status"] == "exact_reference_match"


@pytest.mark.asyncio
async def test_start_char_that_matches_no_section_returns_no_results(doc_store):
    deps, starts = await _store_repeated_reference_document(doc_store)

    result = await _capture_tool(deps, "get_document_section")("mevzuat_802", section_ref="1", start_char=starts[1] + 1)

    assert result.structuredContent["status"] == "no_results"


def test_start_char_docstring_says_it_is_not_a_citation():
    tool = _capture_tool(Dependencies(db=None, doc_store=MagicMock(), client=None, http=None), "get_document_section")

    assert "valid only for the corpus file being served and is not a citation" in " ".join(tool.__doc__.split())


_DECISION_TEXT = (
    "Bankacılık Düzenleme ve Denetleme Kurulunun kararı ile ikinci grupta sınıflandırılması gereken krediler "
    "için gecikme süresi doksan gün olarak uygulanmaya devam edilmesine karar verilmiştir."
)


async def _store_structureless_and_structured(doc_store) -> Dependencies:
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(document_id="779", title="Kurul Kararı 779", markdown_content=_DECISION_TEXT)
    )
    await doc_store.store_document(
        StoredDocument(
            document_id="mevzuat_8",
            title="Kısa Yönetmelik",
            markdown_content=(
                "Dipnot: Bu yönetmelik değişiklikleri ile birlikte okunmalıdır; ikinci grupta sınıflandırılması "
                "gecikme süresi kurul kararı ile güncellenir.\n\nMADDE 1 - Amaç\nBu yönetmeliğin amacı denetimdir.\n"
            ),
        )
    )
    return Dependencies(db=None, doc_store=doc_store, client=None, http=None)


@pytest.mark.asyncio
async def test_section_search_finds_govde_of_a_structureless_document_by_default(doc_store):
    await _store_structureless_and_structured(doc_store)

    hits = await doc_store.search_document_sections("ikinci grupta sınıflandırılması gecikme süresi")

    # 779 has no article structure, so its govde window is its body; mevzuat_8's govde is a remainder.
    assert [(hit.doc_id, hit.section_type, hit.document_structureless) for hit in hits] == [("779", "govde", True)]
    remainder = await doc_store.search_document_sections(
        "ikinci grupta sınıflandırılması gecikme süresi", section_type="govde"
    )
    assert {(hit.doc_id, hit.document_structureless) for hit in remainder} == {("779", True), ("mevzuat_8", False)}


@pytest.mark.asyncio
async def test_structureless_document_rows_are_labelled_full_text_not_remainder(doc_store):
    deps = await _store_structureless_and_structured(doc_store)
    search = await _capture_tool(deps, "search_document_sections")("ikinci grupta sınıflandırılması gecikme süresi")
    exact = await _capture_tool(deps, "get_document_section")(
        "779", quotation="gecikme süresi doksan gün olarak uygulanmaya"
    )

    assert search.structuredContent["results"][0]["document_id"] == "779"
    for result in (search, exact):
        item = result.structuredContent["results"][0]
        assert item["document_structureless"] is True
        assert "full-text window (this document has no article structure)" in result.text
        warnings = result.structuredContent["warnings"]
        assert any("Cite such a document by its title and decision date/number" in w for w in warnings)
        assert not any("Do not cite them as a legal provision" in w for w in warnings)
        # The stored generic govde heading ("gövde/dipnot kalanı") would contradict the kind label.
        assert "gövde/dipnot kalanı" not in result.text
        assert item["heading"] == GOVDE_HEADING  # the structured field stays as stored
    assert "govde remainder — not a legal provision identity" not in search.text
    assert exact.structuredContent["answer_assessment"]["quotation_status"] == "exact_reference_match"


@pytest.mark.asyncio
async def test_remainder_of_a_structured_document_keeps_the_do_not_cite_warning(doc_store):
    deps = await _store_structureless_and_structured(doc_store)

    result = await _capture_tool(deps, "search_document_sections")(
        "yönetmelik değişiklikleri ile birlikte okunmalıdır", section_type="govde"
    )

    [item] = result.structuredContent["results"]
    assert (item["document_id"], item["document_structureless"]) == ("mevzuat_8", False)
    assert "govde remainder — not a legal provision identity" in result.text
    assert any("Do not cite them as a legal provision" in w for w in result.structuredContent["warnings"])


def _categorised(category: str, **kwargs) -> StoredDocumentSection:
    return _section(**kwargs).model_copy(update={"document_category": category, "document_title": "4389 sayılı Kanun"})


@pytest.mark.asyncio
async def test_repealed_and_draft_documents_are_marked_in_every_section_output():
    repealed = _categorised("Mülga Düzenleme", doc_id="1230", section_type="madde", section_ref="1")
    draft = _categorised("Düzenleme Taslağı", doc_id="1290", section_type="madde", section_ref="2")
    in_force = _categorised("Yönetmelik", doc_id="22599", section_type="madde", section_ref="3")
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[repealed])
    doc_store.has_document = AsyncMock(return_value=False)
    doc_store.search_document_sections = AsyncMock(return_value=[repealed, draft, in_force])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    single = await _capture_tool(deps, "get_document_section")("1230", section_type="madde", section_ref="1")
    assert "- Document status: BDDK catalog class: MÜLGA (repealed)" in single.text
    assert any(w.startswith("BDDK catalog class: MÜLGA") for w in single.structuredContent["warnings"])

    doc_store.get_document_section = AsyncMock(return_value=[repealed, repealed.model_copy(update={"start_char": 90})])
    listed = await _capture_tool(deps, "get_document_section")("1230", section_type="madde", section_ref="1")
    assert "- Document status: BDDK catalog class: MÜLGA (repealed)" in listed.text

    hits = await _capture_tool(deps, "search_document_sections")("kanun kapsamı")
    assert hits.text.count("Document status: BDDK catalog class: MÜLGA (repealed)") == 1
    assert hits.text.count("Document status: BDDK catalog class: TASLAK (draft)") == 1
    assert hits.text.count("Document status:") == 2
    warnings = hits.structuredContent["warnings"]
    assert any("MÜLGA" in w for w in warnings) and any("TASLAK" in w for w in warnings)


@pytest.mark.asyncio
async def test_in_force_category_gets_no_status_line():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_categorised("Yönetmelik")])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    out = await _capture_tool(deps, "get_document_section")("943", section_type="ilke", section_ref="5")

    assert "Document status:" not in out.text
    assert not any("catalog class" in w for w in out.structuredContent["warnings"])


# 2026-04-08 23:30 UTC is already 2026-04-09 in UTC+14; the date line must stay in UTC.
_LATE_UTC_DOWNLOAD = 1775691000.0


@pytest.fixture
def far_east_timezone(monkeypatch):
    import time as _time

    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    _time.tzset()
    yield
    monkeypatch.undo()
    _time.tzset()


async def _store_provenance_document(doc_store) -> Dependencies:
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(
            document_id="1291",
            title="Sermaye Yeterliliği Rehberi",
            category="Rehber",
            decision_date="15.03.2024",
            decision_number="11200",
            source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1291",
            markdown_content="MADDE 1 - Kredi riski için asgari sermaye oranı belirlenir.\n",
            downloaded_at=_LATE_UTC_DOWNLOAD,
        )
    )
    return Dependencies(db=None, doc_store=doc_store, client=None, http=None)


@pytest.mark.asyncio
async def test_section_answers_carry_decision_identifiers_and_the_utc_source_date(doc_store, far_east_timezone):
    from datetime import datetime

    assert datetime.fromtimestamp(_LATE_UTC_DOWNLOAD).date().isoformat() == "2026-04-09"
    deps = await _store_provenance_document(doc_store)

    single = await _capture_tool(deps, "get_document_section")("1291", section_type="madde", section_ref="1")
    hits = await _capture_tool(deps, "search_document_sections")("asgari sermaye oranı", document_id="1291")

    for text, prefix in ((single.text, "- "), (hits.text, "  ")):
        assert f"{prefix}Decision date / number: 15.03.2024 / 11200" in text
        assert (
            f"{prefix}Text obtained from the source on: 2026-04-08 (UTC download date; not a statement that the "
            "text is current)"
        ) in text
        assert "days ago" not in text
    for result in (single, hits):
        evidence = result.structuredContent["evidence"][0]
        assert (evidence["decision_date"], evidence["decision_number"]) == ("15.03.2024", "11200")
        assert evidence["source_downloaded_on"] == "2026-04-08"


@pytest.mark.asyncio
async def test_section_without_decision_identifiers_prints_no_decision_line():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section()])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    out = await _capture_tool(deps, "get_document_section")("943", section_type="ilke", section_ref="5")

    assert "Decision date / number:" not in out.text
    assert "- Text obtained from the source on: unknown" in out.text
    assert "source_downloaded_on" not in out.structuredContent["evidence"][0]


async def _store_inflection_fixture(doc_store) -> Dependencies:
    from bddk_mcp.store.doc_store import StoredDocument

    texts = {
        "mevzuat_1": "MADDE 1 - Süre\nKredilerin gecikme süresi doksan gündür.\n",
        "mevzuat_2": "MADDE 2 - Değişiklik\nKurul gecikme süresinin uzatılmasına karar verebilir.\n",
        "mevzuat_3": "GEÇİCİ MADDE 1 - Geçiş\nGeçici süreler için kurul kararı uygulanır.\n",
    }
    for doc_id, text in texts.items():
        await doc_store.store_document(
            StoredDocument(document_id=doc_id, title=f"Yönetmelik {doc_id}", markdown_content=text)
        )
    return Dependencies(db=None, doc_store=doc_store, client=None, http=None)


@pytest.mark.asyncio
async def test_section_search_tops_up_with_inflection_tolerant_matches_after_strict_hits(doc_store):
    deps = await _store_inflection_fixture(doc_store)

    result = await _capture_tool(deps, "search_document_sections")("gecikme süresi", limit=10)

    items = result.structuredContent["results"]
    assert [(i["document_id"], i["stem_match"]) for i in items] == [("mevzuat_1", False), ("mevzuat_2", True)]
    assert items[0].get("rank") is not None
    assert items[1].get("rank") is None
    assert result.text.count("inflection-tolerant match") == 1
    assert result.structuredContent["loose_fallback_used"] is False


@pytest.mark.asyncio
async def test_section_search_skips_the_top_up_when_strict_hits_fill_the_limit(doc_store):
    deps = await _store_inflection_fixture(doc_store)

    result = await _capture_tool(deps, "search_document_sections")("gecikme süresi", limit=1)

    assert [i["document_id"] for i in result.structuredContent["results"]] == ["mevzuat_1"]
    assert "inflection-tolerant" not in result.text


@pytest.mark.asyncio
async def test_inflection_tolerant_top_up_alone_is_not_the_loose_fallback(doc_store):
    deps = await _store_inflection_fixture(doc_store)

    result = await _capture_tool(deps, "search_document_sections")("süresinin uzatılması")

    items = result.structuredContent["results"]
    assert [(i["document_id"], i["stem_match"]) for i in items] == [("mevzuat_2", True)]
    assert result.structuredContent["loose_fallback_used"] is False


@pytest.mark.asyncio
async def test_prefix_floor_keeps_gecici_apart_from_gecikme(doc_store):
    deps = await _store_inflection_fixture(doc_store)

    result = await _capture_tool(deps, "search_document_sections")("gecikme", limit=10, section_type="gecici_madde")

    assert result.structuredContent["status"] == "no_results"


_NESTED_TEXT = (
    "MADDE 1 - Amaç\n(1) Bu yönetmeliğin amacı denetimdir.\n(2) Bankalar şunları yapar:\na) Kredi faiz riskini "
    "ölçmek,\nb) Raporlamak.\n\nMADDE 2 - Kapsam\n(1) Kapsam bankalardır.\n\nGEÇİCİ MADDE 1 - Geçiş\n"
    "(1) Geçiş hükmü uygulanır.\n"
)


async def _store_nested_fixture(doc_store) -> Dependencies:
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(document_id="mevzuat_1", title="Yönetmelik", markdown_content=_NESTED_TEXT)
    )
    return Dependencies(db=None, doc_store=doc_store, client=None, http=None)


@pytest.mark.asyncio
async def test_fikra_search_hit_names_its_parent_article(doc_store):
    deps = await _store_nested_fixture(doc_store)

    result = await _capture_tool(deps, "search_document_sections")("kredi faiz riskini", section_type="fikra")

    [item] = result.structuredContent["results"]
    assert (item["section_type"], item["section_ref"]) == ("fikra", "2")
    assert (item["parent_section_type"], item["parent_section_ref"]) == ("madde", "1")
    assert "  Parent: madde 1" in result.text


@pytest.mark.asyncio
async def test_get_document_section_names_the_parent_of_each_fikra(doc_store):
    deps = await _store_nested_fixture(doc_store)
    tool = _capture_tool(deps, "get_document_section")

    single = await tool("mevzuat_1", section_type="fikra", section_ref="2")
    several = await tool("mevzuat_1", section_type="fikra", section_ref="1")

    assert "- Parent: madde 1" in single.text
    assert single.structuredContent["results"][0]["parent_section_ref"] == "1"
    parents = [(i["parent_section_type"], i["parent_section_ref"]) for i in several.structuredContent["results"]]
    assert parents == [("madde", "1"), ("madde", "2"), ("gecici_madde", "1")]
    assert "  Parent: gecici_madde 1" in several.text


@pytest.mark.asyncio
async def test_article_rows_and_rows_outside_any_article_have_no_parent(doc_store):
    deps = await _store_nested_fixture(doc_store)
    [madde] = await doc_store.get_document_section("mevzuat_1", section_type="madde", section_ref="1")
    [fikra] = await doc_store.get_document_section("mevzuat_1", section_type="fikra", section_ref="2")
    orphan = fikra.model_copy(update={"start_char": 10_000})

    assert await doc_store.find_parent_articles([madde, orphan]) == {}
    result = await _capture_tool(deps, "get_document_section")("mevzuat_1", section_type="madde", section_ref="1")
    assert "Parent:" not in result.text
    assert result.structuredContent["results"][0].get("parent_section_type") is None


@pytest.mark.asyncio
async def test_doc_store_flags_a_continuation_row_with_text_past_the_provision_end(doc_store):
    """A govde window can run past the capped provision's true end (mevzuat_21193 govde 6, 1305 govde 8)."""
    from bddk_mcp.store.section_index import TRUNCATION_NOTICE_PREFIX

    await _store_capped_document(doc_store)
    sections = await doc_store.get_document_section("mevzuat_9")
    outer = next(section for section in sections if (section.section_type, section.section_ref) == ("madde", "2"))
    govde = sorted((section for section in sections if section.section_type == "govde"), key=lambda s: s.start_char)
    last = govde[-1]
    short_start = outer.start_char + 50
    short_end = last.start_char + 100
    short_content = (
        _CAPPED_DOCUMENT[short_start : short_start + 20_000]
        + f"\n\n{TRUNCATION_NOTICE_PREFIX} içerik {short_end - short_start} karakterden 20000 karaktere kısaltıldı"
        " — tam metin için get_bddk_document kullanın]"
    )

    def insert_short(conn):
        conn.execute(
            "INSERT INTO document_sections (doc_id, section_type, section_ref, heading, start_char, end_char, "
            "content, content_hash, source_content_hash) VALUES (?, 'ek', '1', '', ?, ?, ?, ?, ?)",
            [
                "mevzuat_9",
                short_start,
                short_start + 20_000,
                short_content,
                hashlib.sha256(short_content.encode("utf-8")).hexdigest(),
                outer.source_content_hash,
            ],
        )

    await doc_store._db.transaction(insert_short)

    continuations = await doc_store.find_capped_continuations(sections)

    assert continuations[(last.doc_id, last.source_content_hash, last.start_char)] == ("ek", "1", short_end, True)
    for row in govde[:-1]:
        assert continuations[(row.doc_id, row.source_content_hash, row.start_char)][3] is False


@pytest.mark.asyncio
async def test_capped_section_whose_overflow_is_in_nested_rows_does_not_claim_govde_rows():
    """Real case: 1135 paragraf 2 is capped and its overflow is in nested paragraf rows, not govde."""
    from bddk_mcp.store.section_index import TRUNCATION_NOTICE_PREFIX

    content = (
        "2. Bankalar risklerini ölçer.\n\n"
        f"{TRUNCATION_NOTICE_PREFIX} içerik 26603 karakterden 20000 karaktere kısaltıldı"
        " — tam metin için get_bddk_document kullanın]"
    )
    capped = _section("1135", "paragraf", "2", content).model_copy(update={"start_char": 3590, "end_char": 23590})
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[capped])
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")("1135", section_type="paragraf", section_ref="2")
    quoted = await _capture_tool(deps, "get_document_section")(
        "1135", section_type="paragraf", section_ref="2", quotation="metinde olmayan cümle"
    )

    assert "the provision continues to code point 30193" in result.text
    for text in (result.text, quoted.text):
        assert "nested fikra/bent/paragraf rows and/or govde windows" in text
        assert "in full in get_bddk_document" in text
        assert "following govde rows" not in text
        assert "continuation govde rows" not in text


@pytest.mark.asyncio
async def test_continuation_row_running_past_the_provision_end_states_the_outside_tail():
    """Real case: mevzuat_21193 govde 6 is [70856, 71187) while madde 10 ends at 71109."""
    row = _section("mevzuat_21193", "govde", "6", "Devam eden metin. MADDE 11 - Sonraki").model_copy(
        update={"start_char": 70856, "end_char": 71187}
    )
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[row])
    doc_store.find_capped_continuations = AsyncMock(
        return_value={(row.doc_id, row.source_content_hash, row.start_char): ("madde", "10", 71109, True)}
    )
    deps = Dependencies(db=None, doc_store=doc_store, client=None, http=None)

    result = await _capture_tool(deps, "get_document_section")("mevzuat_21193", section_type="govde", section_ref="6")

    item = result.structuredContent["results"][0]
    assert (item["continuation_of_section_type"], item["continuation_of_section_ref"]) == ("madde", "10")
    assert item["continuation_of_section_end_char"] == 71109
    tail = "the characters from code point 71109 on are outside madde 10"
    assert f"up to code point 71109; {tail}" in result.text
    assert any(
        f"this text belongs to madde 10 up to code point 71109; {tail}" in warning
        for warning in result.structuredContent["warnings"]
    )


@pytest.mark.asyncio
async def test_continuation_row_inside_the_provision_has_no_outside_tail(doc_store):
    deps = await _store_capped_document(doc_store)

    result = await _capture_tool(deps, "get_document_section")("mevzuat_9", section_type="govde", section_ref="1")

    item = result.structuredContent["results"][0]
    assert item["continuation_of_section_end_char"] == _CAPPED_DOCUMENT.index("MADDE 3") - 1
    assert "are outside madde 2" not in result.text


@pytest.mark.asyncio
async def test_exact_reference_hit_found_again_by_the_top_up_is_not_labelled_a_stem_match(doc_store):
    deps = await _store_inflection_fixture(doc_store)

    result = await _capture_tool(deps, "search_document_sections")("madde 2 gecikme süresi", document_id="mevzuat_2")

    assert result.structuredContent["exact_reference_detected"] is True
    items = result.structuredContent["results"]
    assert [(i["section_type"], i["section_ref"], i["stem_match"]) for i in items] == [("madde", "2", False)]
    assert "inflection-tolerant match" not in result.text
