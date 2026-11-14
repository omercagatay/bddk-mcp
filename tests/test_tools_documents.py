"""Tests for tools/documents.py — tool registration and bare-ID resolver."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.models import BddkDecisionSummary
from bddk_mcp.quality.markdown_quality import assess_markdown_quality
from bddk_mcp.store.doc_store import DocumentPage
from bddk_mcp.tools.documents import _is_formula_aware, register


def test_register_exposes_end_user_tools_only_by_default():
    """Only the two document tools are registered."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_document",
        "get_document_history",
    }


def _capture_get_bddk_document(deps: Dependencies):
    """Register tools on a stub MCP and return the get_bddk_document callable."""
    mcp = MagicMock()
    register(mcp, deps)
    for call in mcp.tool.return_value.call_args_list:
        fn = call.args[0]
        if fn.__name__ == "get_bddk_document":
            return fn
    raise AssertionError("get_bddk_document not registered")


def _make_deps(*, doc_store, client=None, vector_store=None) -> Dependencies:
    doc_store.get_document_quality = AsyncMock(side_effect=lambda doc_id: assess_markdown_quality("", doc_id))
    deps = Dependencies(db=None, doc_store=doc_store, client=client or MagicMock(), http=None)
    deps.vector_store = vector_store
    if client is None:
        deps.client.find_by_id = MagicMock(return_value=None)
    return deps


@pytest.mark.asyncio
async def test_bare_numeric_id_resolves_to_mevzuat_prefix():
    """get_bddk_document('21192') should auto-resolve to 'mevzuat_21192' on bare-ID miss."""
    page = DocumentPage(
        document_id="mevzuat_21192",
        title="Sermaye Yeterliliği Yönetmeliği",
        markdown_content="BANKALARIN SERMAYE YETERLİLİĞİ...",
        page_number=1,
        total_pages=28,
    )

    async def fake_get_page(doc_id, page_number):
        return page if doc_id == "mevzuat_21192" else None

    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(side_effect=fake_get_page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("21192", 1)

    assert "BANKALARIN SERMAYE YETERLİLİĞİ" in out
    assert "Document ID: mevzuat_21192" in out
    assert "Resolved from: `21192` -> `mevzuat_21192`" in out
    # Must have probed the bare ID before falling back to the prefixed form
    assert doc_store.get_document_page.await_args_list[0].args == ("21192", 1)
    assert doc_store.get_document_page.await_args_list[1].args == ("mevzuat_21192", 1)


@pytest.mark.asyncio
async def test_bare_id_exact_match_skips_prefix_lookup():
    """If the bare ID exists as-is, the prefix variants must not be probed."""
    page = DocumentPage(
        document_id="1291",
        title="Karar 1291",
        markdown_content="Kurul kararı...",
        page_number=1,
        total_pages=1,
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("1291", 1)

    assert "Resolved from" not in out
    assert "Document ID: 1291" in out
    assert doc_store.get_document_page.await_count == 1


@pytest.mark.asyncio
async def test_prefixed_id_does_not_get_expanded():
    """An already-prefixed ID must be tried only as-is — no '21192' or 'bddk_21192' probes."""
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    with pytest.raises(ToolError) as exc_info:
        await tool("mevzuat_21192", 1)
    out = str(exc_info.value)

    assert "not available in the local store" in out
    assert "airlocked" in out
    assert doc_store.get_document_page.await_count == 1
    assert doc_store.get_document_page.await_args_list[0].args == ("mevzuat_21192", 1)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("lightocr", True),
        ("mevzuat_pdf+lightocr", True),
        ("cached_pdf+pp_structure", True),
        ("manual_latex", True),
        ("html_parser+manual_latex", True),
        ("markitdown", False),
        ("markitdown_degraded", False),
        ("manual_pdf+markitdown", False),
        ("html_parser", False),
        ("", False),
        ("glm_ocr", False),
    ],
)
def test_is_formula_aware_classification(method, expected):
    assert _is_formula_aware(method) is expected


@pytest.mark.asyncio
async def test_degraded_extraction_emits_warning_and_method_line():
    """A document extracted by markitdown must surface a formula-loss warning."""
    page = DocumentPage(
        document_id="mevzuat_20029",
        title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
        markdown_content="MADDE 46 ... aşağıda yer alan formül kullanılarak artırılır.",
        page_number=18,
        total_pages=24,
        extraction_method="manual_pdf+markitdown",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_20029", 18)

    assert "- Extraction: manual_pdf+markitdown (formula-unaware" in out
    assert "⚠" in out
    assert "formülü hafızadan veya standart literatürden yeniden kurma" in out
    # The extraction line must come before the content boundary.
    assert out.index("- Extraction:") < out.index("---")


@pytest.mark.asyncio
async def test_formula_aware_extraction_no_warning():
    """A lightocr-extracted document must NOT emit the formula-loss warning."""
    page = DocumentPage(
        document_id="mevzuat_20029",
        title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
        markdown_content="formül içeren metin",
        page_number=1,
        total_pages=1,
        extraction_method="mevzuat_pdf+lightocr",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_20029", 1)

    assert "- Extraction: mevzuat_pdf+lightocr" in out
    assert "formula-unaware" not in out
    assert "⚠" not in out


@pytest.mark.asyncio
async def test_large_document_emits_targeted_retrieval_warning():
    """Long documents should warn callers away from page-by-page full retrieval."""
    page = DocumentPage(
        document_id="mevzuat_5411",
        title="5411 sayılı Bankacılık Kanunu",
        markdown_content="MADDE 1 — Amaç",
        page_number=1,
        total_pages=67,
        extraction_method="mevzuat_pdf+lightocr",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_5411", 1)

    assert "- Page: 1/67" in out
    assert "Bu belge 67 sayfa" in out
    assert "search_document_sections" in out
    assert "get_document_section" in out


@pytest.mark.asyncio
async def test_get_bddk_document_can_return_limited_consecutive_pages():
    """max_pages returns a bounded consecutive page window from the resolved document."""
    pages = {
        2: DocumentPage(
            document_id="mevzuat_5411",
            title="5411 sayılı Bankacılık Kanunu",
            markdown_content="MADDE 2 — Tanımlar",
            page_number=2,
            total_pages=67,
            extraction_method="mevzuat_pdf+lightocr",
        ),
        3: DocumentPage(
            document_id="mevzuat_5411",
            title="5411 sayılı Bankacılık Kanunu",
            markdown_content="MADDE 3 — Kapsam",
            page_number=3,
            total_pages=67,
            extraction_method="mevzuat_pdf+lightocr",
        ),
    }

    async def fake_get_page(doc_id, page_number):
        assert doc_id == "mevzuat_5411"
        return pages.get(page_number)

    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(side_effect=fake_get_page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_5411", page_number=2, max_pages=2)

    assert "- Page: 2-3/67" in out
    assert "### Page 2/67" in out
    assert "MADDE 2 — Tanımlar" in out
    assert "### Page 3/67" in out
    assert "MADDE 3 — Kapsam" in out
    assert doc_store.get_document_page.await_count == 2


@pytest.mark.asyncio
async def test_get_bddk_document_rejects_oversized_page_window():
    """Large max_pages requests fail instead of silently changing arguments."""
    pages = {
        page_number: DocumentPage(
            document_id="mevzuat_5411",
            title="5411 sayılı Bankacılık Kanunu",
            markdown_content=f"Sayfa {page_number}",
            page_number=page_number,
            total_pages=67,
            extraction_method="mevzuat_pdf+lightocr",
        )
        for page_number in range(1, 7)
    }
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(side_effect=lambda _doc_id, page_number: pages.get(page_number))
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await tool("mevzuat_5411", page_number=1, max_pages=99)

    doc_store.get_document_page.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_bddk_document_refuses_rather_than_rewriting_unsafe_verbatim_text():
    """Unsafe extraction blobs must never leak, and document text is never rewritten/sanitized."""
    page = DocumentPage(
        document_id="mevzuat_21192",
        title="Risk Yönetimi",
        markdown_content="<div><img src='data:image/x-wmf;base64,AAA'>cid:12</div>Madde 1",
        page_number=1,
        total_pages=1,
        extraction_method="markitdown",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)
    doc_store.get_document_quality = AsyncMock(
        return_value=assess_markdown_quality(page.markdown_content, page.document_id)
    )

    tool = _capture_get_bddk_document(deps)
    with pytest.raises(ToolError) as exc_info:
        await tool("mevzuat_21192", 1)
    out = str(exc_info.value)

    assert out.splitlines()[0] == "[ERROR:VERBATIM_UNAVAILABLE] retryable=false"
    assert "cannot be returned verbatim" in out
    assert "does not rewrite" in out
    # The unsafe artifact and the rewritable document text must not leak at all.
    for leaked in ("data:image/", "base64", "<img", "<div", "cid:12", "Madde 1"):
        assert leaked not in out


@pytest.mark.asyncio
async def test_get_bddk_document_surfaces_configured_failure_for_clean_page_content(historical_quality_registry):
    """Registry membership must remain visible even when a retrieved page looks clean."""
    page = DocumentPage(
        document_id="903",
        title="Configured quality failure",
        markdown_content="MADDE 1 - Temiz görünen mevzuat metni.",
        page_number=1,
        total_pages=1,
        extraction_method="manual_latex",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("903", 1)

    assert "Quality: fail" in out
    assert "configured_quality_failure" in out
    assert "concatenation_formula_table_artifacts" in out
    assert "listed in the configured quality-failure registry" in out
    assert out.index("Quality warning") < out.index("MADDE 1")


@pytest.mark.asyncio
async def test_missing_doc_returns_airlocked_error_for_all_candidates():
    """When no candidate hits, return the airlock error referencing the original ID."""
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    with pytest.raises(ToolError) as exc_info:
        await tool("99999999", 1)
    out = str(exc_info.value)

    assert "Document 99999999 is not available" in out
    assert "airlocked" in out
    # Structured error: machine-parseable first line so callers can tell
    # NOT_FOUND apart from transient failures.
    assert out.splitlines()[0] == "[ERROR:NOT_FOUND] retryable=false"
    # All three candidates probed: bare, mevzuat_, bddk_
    assert doc_store.get_document_page.await_count == 3


@pytest.mark.asyncio
async def test_multi_page_gap_is_surfaced_as_warning():
    """A missing middle page must produce an explicit gap warning, not silent truncation."""

    def make_page(n: int) -> DocumentPage:
        return DocumentPage(
            document_id="mevzuat_1",
            title="Test Yönetmeliği",
            markdown_content=f"içerik sayfa {n}",
            page_number=n,
            total_pages=4,
        )

    async def fake_get_page(doc_id, page_number):
        if page_number == 2:
            return None  # missing/corrupt middle page
        return make_page(page_number)

    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(side_effect=fake_get_page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_1", 1, 3)

    assert "içerik sayfa 1" in out
    assert "içerik sayfa 2" not in out
    assert "Sayfa 2 local store'dan alınamadı" in out
    # Still a successful (partial) response, not an error
    assert "[ERROR" not in out


@pytest.mark.asyncio
async def test_document_quality_lookup_failure_cannot_claim_clean():
    from bddk_mcp.core.exceptions import BddkStorageError

    store = MagicMock()
    store.get_document_page = AsyncMock(
        return_value=DocumentPage(document_id="x", title="x", markdown_content="Temiz sayfa.")
    )
    deps = _make_deps(doc_store=store)
    store.get_document_quality = AsyncMock(side_effect=BddkStorageError("unavailable"))
    result = await _capture_get_bddk_document(deps)("x")
    assert result.structuredContent["quality"]["label"] == "unknown"
    assert "document_quality_unavailable" in result.structuredContent["quality"]["flags"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,expected_label,expected_flags",
    [
        ("| A | B |\n| --- | --- |\n| x | y |", "clean", []),
        ("Temiz sayfa.\n\nSonraki sayfa: \ufffd", "fail", ["replacement_char"]),
        ("Temiz sayfa.\n\naşağıdaki formül kullanılır.", "warning", ["formula_ref_without_latex_or_image"]),
    ],
)
async def test_document_quality_is_assessed_on_the_canonical_body(doc_store, body, expected_label, expected_flags):
    """The quality label comes from the whole stored body, not from the page that was returned."""
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(StoredDocument(document_id="x", title="x", markdown_content=body))
    client = MagicMock()
    client.find_by_id = MagicMock(return_value=None)
    deps = Dependencies(db=None, doc_store=doc_store, client=client, http=None)

    result = await _capture_get_bddk_document(deps)("x")

    quality = result.structuredContent["quality"]
    assert quality["label"] == expected_label
    assert quality["flags"] == expected_flags
    assert result.structuredContent["evidence"][0]["quality"] == quality
    assert result.structuredContent["pages"][0]["content"] == body


@pytest.mark.asyncio
async def test_page_beyond_the_end_of_an_existing_document_is_invalid_input(doc_store):
    """An out-of-range page names the page count; it must not claim the document is absent."""
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(StoredDocument(document_id="x", title="x", markdown_content="Kısa belge."))
    client = MagicMock()
    client.find_by_id = MagicMock(return_value=None)
    deps = Dependencies(db=None, doc_store=doc_store, client=client, http=None)

    with pytest.raises(ToolError) as exc_info:
        await _capture_get_bddk_document(deps)("x", page_number=50)
    out = str(exc_info.value)

    assert out.splitlines()[0] == "[ERROR:INVALID_INPUT] retryable=false"
    assert "has 1 page(s)" in out
    assert "not available in the local store" not in out


@pytest.mark.asyncio
async def test_a_real_page_mentioning_invalid_page_is_returned():
    """Stored legal text is never treated as the out-of-range sentinel because of its wording."""
    page = DocumentPage(
        document_id="x",
        title="x",
        markdown_content="Invalid page ifadesi geçen gerçek metin.",
        page_number=1,
        total_pages=1,
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    out = await _capture_get_bddk_document(_make_deps(doc_store=doc_store))("x", 1)
    assert "Invalid page ifadesi geçen gerçek metin." in out


@pytest.mark.asyncio
async def test_empty_document_history_explains_that_only_replaced_bodies_are_listed():
    mcp = MagicMock()
    doc_store = MagicMock()
    doc_store.get_current_body = AsyncMock(
        return_value={"content_hash": "d" * 64, "content_length": 5, "downloaded_at": None, "extraction_method": ""}
    )
    doc_store.get_document_history = AsyncMock(return_value=[])
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None))
    tool = next(
        call.args[0] for call in mcp.tool.return_value.call_args_list if call.args[0].__name__ == "get_document_history"
    )
    result = await tool("943")
    assert result.structuredContent["status"] == "no_results"
    assert "No version history found for document 943." in result.text
    assert "Current stored body (not a version)" in result.text
    assert result.structuredContent["versions"] == []
    assert result.structuredContent["current_body"]["content_hash"] == "d" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "marker"),
    [("Mülga Düzenleme", "MÜLGA (repealed)"), ("Düzenleme Taslağı", "TASLAK (draft)"), ("Yönetmelik", None)],
)
async def test_get_bddk_document_marks_repealed_and_draft_documents(category, marker):
    from bddk_mcp.core.models import BddkDecisionSummary

    page = DocumentPage(document_id="1230", title="4389 sayılı Bankalar Kanunu", markdown_content="MADDE 1 - Metin.")
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    client = MagicMock()
    client.find_by_id = MagicMock(
        return_value=BddkDecisionSummary(
            title="4389 sayılı Bankalar Kanunu", document_id="1230", content="", category=category
        )
    )
    deps = _make_deps(doc_store=doc_store, client=client)

    out = await _capture_get_bddk_document(deps)("1230")

    if marker is None:
        assert "Document status:" not in out.text
    else:
        assert f"- Document status: BDDK catalog class: {marker}" in out.text
        assert any(marker in warning for warning in out.structuredContent["warnings"])


@pytest.mark.asyncio
async def test_get_bddk_document_states_the_utc_source_date_and_the_body_hash(doc_store, monkeypatch):
    import hashlib
    import time as _time

    from bddk_mcp.store.doc_store import StoredDocument

    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    _time.tzset()
    try:
        body = "MADDE 1 - Kredi riski için asgari sermaye oranı belirlenir.\n"
        await doc_store.store_document(
            StoredDocument(document_id="1291", title="Rehber", markdown_content=body, downloaded_at=1775691000.0)
        )
        deps = Dependencies(db=None, doc_store=doc_store, client=MagicMock(), http=None)
        deps.client.find_by_id = MagicMock(return_value=None)

        out = await _capture_get_bddk_document(deps)("1291")
    finally:
        monkeypatch.undo()
        _time.tzset()

    assert (
        "- Text obtained from the source on: 2026-04-08 (UTC download date; not a statement that the text is current)"
    ) in out.text
    evidence = out.structuredContent["evidence"][0]
    assert evidence["source_downloaded_on"] == "2026-04-08"
    assert evidence["content_hash"] == hashlib.sha256(body.encode("utf-8")).hexdigest()


def _history_tool(deps: Dependencies):
    mcp = MagicMock()
    register(mcp, deps)
    return next(
        call.args[0] for call in mcp.tool.return_value.call_args_list if call.args[0].__name__ == "get_document_history"
    )


@pytest.mark.asyncio
async def test_document_history_of_an_unknown_id_is_not_found(doc_store):
    with pytest.raises(ToolError, match="NOT_FOUND"):
        await _history_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None))("no-such-doc")


@pytest.mark.asyncio
async def test_document_history_is_framed_as_a_local_text_change_log_not_amendment_history(doc_store, monkeypatch):
    import hashlib
    import time as _time

    from bddk_mcp.store.doc_store import StoredDocument
    from bddk_mcp.tools.structured_outputs import SOURCE_DATA_BEGIN

    first = StoredDocument(document_id="mevzuat_21192", title="Y", markdown_content="MADDE 1 - Eski metin.\n")
    first.downloaded_at = 1775667192.0
    await doc_store.store_document(first)
    replaced_at = 1775691000.0  # 2026-04-08 23:30 UTC, already 2026-04-09 in UTC+14
    monkeypatch.setattr("bddk_mcp.store.doc_store.time.time", lambda: replaced_at)
    second = StoredDocument(
        document_id="mevzuat_21192",
        title="Y",
        markdown_content="MADDE 1 - Yeni metin.\n",
        downloaded_at=replaced_at,
        extraction_method="manual",
    )
    await doc_store.store_document(second)
    monkeypatch.undo()
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    _time.tzset()
    try:
        result = await _history_tool(Dependencies(db=None, doc_store=doc_store, client=None, http=None))("21192")
    finally:
        monkeypatch.undo()
        _time.tzset()

    text = result.text
    assert text.startswith("This is the local text-change log of this corpus file")
    assert "not the legal amendment history" in text.split(SOURCE_DATA_BEGIN)[0]
    assert "mevzuat.gov.tr / Resmî Gazete" in text.split(SOURCE_DATA_BEGIN)[0]
    assert "Resolved from: `21192` -> `mevzuat_21192`" in text
    assert "v1 — replaced on 2026-04-08 23:30 UTC" in text
    current_hash = hashlib.sha256(b"MADDE 1 - Yeni metin.\n").hexdigest()
    assert f"hash {current_hash[:12]}..." in text
    assert "text obtained from the source on: 2026-04-08 (UTC download date" in text
    structured = result.structuredContent
    assert structured["resolved_document_id"] == "mevzuat_21192"
    assert structured["current_body"] == {
        "content_hash": current_hash,
        "content_length": len("MADDE 1 - Yeni metin.\n"),
        "source_downloaded_on": "2026-04-08",
        "extraction_method": "manual",
    }
    assert [v["synced_at"] for v in structured["versions"]] == ["2026-04-08 23:30 UTC"]
    assert current_hash not in {v["content_hash"] for v in structured["versions"]}


def _paged_store(pages: list[str]) -> MagicMock:
    async def fake_get_page(doc_id, page_number):
        # The same four structural flags DocumentStore.get_document_page derives from the whole body.
        body = "".join(pages)
        start = sum(len(page) for page in pages[: page_number - 1])
        content = pages[page_number - 1]
        end = start + len(content)
        previous = body[:start]
        last_line = content.rsplit("\n", 1)[-1]
        previous_last_line = previous.rsplit("\n", 1)[-1]
        return DocumentPage(
            document_id=doc_id,
            title="Formül Rehberi",
            markdown_content=content,
            page_number=page_number,
            total_pages=len(pages),
            extraction_method="manual_latex",
            starts_inside_table_row=start > 0
            and not previous.endswith("\n")
            and previous_last_line.startswith("|")
            and body[start] not in "\r\n",
            starts_inside_formula=previous.count("$$") % 2 == 1,
            ends_inside_table_row=end < len(body)
            and not content.endswith("\n")
            and last_line.startswith("|")
            and body[end] not in "\r\n",
            ends_inside_formula=body[:end].count("$$") % 2 == 1,
        )

    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(side_effect=fake_get_page)
    return doc_store


@pytest.mark.asyncio
async def test_header_states_fixed_page_windows_and_separator_lines():
    from bddk_mcp.core.config import PAGE_SIZE

    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(["a\n", "b\n"])))

    single = await tool("1240", 1, 1)
    multi = await tool("1240", 1, 2)

    for out in (single, multi):
        assert f"Pages are fixed windows of {PAGE_SIZE:,} characters" in out.text
        assert "inside a word, a table row or a formula" in out.text
        assert "wrapping" not in out.text
    assert "tool separators" not in single.text
    assert "'---' and '### Page N/T' lines are tool separators between pages, not source text" in multi.text


@pytest.mark.asyncio
async def test_page_ending_inside_a_formula_or_table_row_is_warned():
    pages = ["Giriş\n$$\n\\text{İKÇ} = \\ln\\left(\\ex", "p(1)\n$$\n| a | b |\n| 1 | 2", "| 3 | 4 |\nson\n"]
    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(pages)))

    first = await tool("1240", 1, 1)
    second = await tool("1240", 2, 1)

    assert "Page 1 ends inside a formula that continues on page 2" in first.text
    assert any("Page 1 ends inside a formula" in w for w in first.structuredContent["warnings"])
    # Page 1 has no previous page; its only neighbour is page 2.
    assert "Retrieve page 2 as well before quoting that formula" in first.text
    # Page 2 only closes the formula begun on page 1: the start flag is exact, so it names page 1 only.
    assert "Page 2 starts inside a formula opened on page 1" in second.text
    assert "Retrieve page 1 as well before quoting that formula" in second.text
    assert "ends inside a formula" not in second.text and "cannot tell which" not in second.text
    # The table row really is an end-of-page cut, so it names the next page.
    assert "Page 2 ends inside a table row that continues on the next page" in second.text
    assert "Retrieve page 3 as well before quoting that row" in second.text


@pytest.mark.asyncio
async def test_cut_advice_does_not_ask_for_a_page_already_in_the_response():
    pages = ["Giriş\n$$\n\\text{İKÇ} = \\ln\\left(\\ex", "p(1)\n$$\n| a | b |\n| 1 | 2", "| 3 | 4 |\nson\n"]
    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(pages)))

    result = await tool("1240", 1, 3)

    assert "Retrieve page" not in result.text
    assert "Its continuation is at the start of page 2 in this response" in result.text
    assert "Its continuation is at the start of page 3 in this response" in result.text


@pytest.mark.asyncio
async def test_last_page_and_clean_boundaries_get_no_cut_warning():
    pages = ["MADDE 1 - Metin.\n| a | b |\n", "Son $$ formül"]
    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(pages)))

    first = await tool("1240", 1, 1)
    last = await tool("1240", 2, 1)

    assert "formula delimiters" not in first.text and "table row that continues" not in first.text
    assert "formula delimiters" not in last.text


def _documents_server(doc_store) -> object:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("documents-contract-tests")
    deps = _make_deps(doc_store=doc_store)
    register(server, deps)
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"document_id": "1291", "page_number": True},
        {"document_id": "1291", "page_number": "2"},
        {"document_id": "1291", "max_pages": True},
        {"document_id": "1291", "max_pages": 6},
        {"document_id": "a.b"},
        {"document_id": "x" * 65},
    ],
)
async def test_document_arguments_use_the_strict_shared_contract_types(arguments):
    from mcp.shared.memory import create_connected_server_and_client_session

    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    async with create_connected_server_and_client_session(_documents_server(doc_store)) as session:
        result = await session.call_tool("get_bddk_document", arguments)

    assert result.isError is True
    assert "[ERROR:INVALID_INPUT]" in result.content[0].text
    doc_store.get_document_page.assert_not_awaited()


def test_every_seeded_document_id_satisfies_the_document_id_contract():
    import json
    from pathlib import Path

    from pydantic import TypeAdapter

    from bddk_mcp.tools.contract_types import DocumentId

    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    if not seed.is_file():
        pytest.skip("seed_data/documents.json is not present")
    documents = json.loads(seed.read_text(encoding="utf-8"))
    adapter = TypeAdapter(DocumentId)
    for document in documents:
        assert adapter.validate_python(document["document_id"]) == document["document_id"]


@pytest.mark.asyncio
async def test_status_mark_falls_back_to_the_stored_page_category_without_a_catalog_row():
    page = DocumentPage(
        document_id="1100",
        title="Eski Yönetmelik",
        markdown_content="MADDE 1 - Metin.\n",
        category="Mülga Düzenleme",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)
    assert deps.client.find_by_id("1100") is None

    result = await _capture_get_bddk_document(deps)("1100", 1, 1)

    assert "## Eski Yönetmelik\n" in result.text
    assert result.structuredContent["title"] == "Eski Yönetmelik"
    assert "- Category: Mülga Düzenleme" in result.text
    assert "- Document status: BDDK catalog class: MÜLGA" in result.text
    assert any("MÜLGA" in warning for warning in result.structuredContent["warnings"])
    assert result.structuredContent["category"] == "Mülga Düzenleme"


@pytest.mark.asyncio
async def test_header_prints_the_combined_decision_line_only_when_stored():
    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(["a\n"])))

    result = await tool("1240", 1, 1)

    assert "Decision date / number:" not in result.text
    assert "- Decision Date:" not in result.text and "- Decision Number:" not in result.text


def _catalog_client(entries):
    from bddk_mcp.ingest.client import BddkApiClient
    from tests.conftest import MockStore

    client = BddkApiClient(store=MockStore())
    client._cache = list(entries)
    client._cache_timestamp = 9999999999.0
    return client


_DECISION_8970 = BddkDecisionSummary(
    title="Kredilerin İkinci Grupta Sınıflandırılması için Öngörülen Gecikme Süresinin Değiştirilmesi hk. Kurul Kararı",
    document_id="779",
    decision_date="27.03.2020",
    decision_number="8970",
    category="Kurul Kararı",
)
_DECISION_10747 = BddkDecisionSummary(
    title="SYR Hesaplamasına İlişkin Değişiklikler Hk. Kurul Kararı",
    document_id="1207",
    decision_date="12.12.2023",
    decision_number="10747",
    category="Kurul Kararı",
)
_REGULATION_10747 = BddkDecisionSummary(
    title="Bankaların Muhasebe Uygulamalarına ve Belgelerin Saklanmasına İlişkin Usul ve Esaslar Hakkında Yönetmelik",
    document_id="mevzuat_10747",
    category="Yönetmelik",
)


@pytest.mark.asyncio
async def test_a_decision_number_given_as_document_id_names_the_document_carrying_it():
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    deps = _make_deps(doc_store=doc_store, client=_catalog_client([_DECISION_8970, _REGULATION_10747]))

    with pytest.raises(ToolError) as exc_info:
        await _capture_get_bddk_document(deps)("8970", 1)
    out = str(exc_info.value)

    assert out.splitlines()[0] == "[ERROR:NOT_FOUND] retryable=false"
    assert "decision number" in out
    assert "779" in out
    assert 'get_bddk_document(document_id="779")' in out
    assert "search_bddk_regulations" in out
    assert "bddk-mcp build" not in out and "bddk-mcp sync" not in out


@pytest.mark.asyncio
async def test_a_number_that_is_no_decision_number_gets_no_decision_note():
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    deps = _make_deps(doc_store=doc_store, client=_catalog_client([_DECISION_8970]))

    with pytest.raises(ToolError) as exc_info:
        await _capture_get_bddk_document(deps)("99999999", 1)
    out = str(exc_info.value)

    assert "Document 99999999 is not available" in out
    assert "decision number of" not in out


@pytest.mark.asyncio
async def test_a_stored_id_that_is_also_another_documents_decision_number_names_that_document():
    page = DocumentPage(
        document_id="mevzuat_10747",
        title=_REGULATION_10747.title,
        markdown_content="BANKALARIN MUHASEBE UYGULAMALARI...",
        page_number=1,
        total_pages=1,
    )

    async def fake_get_page(doc_id, page_number):
        return page if doc_id == "mevzuat_10747" else None

    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(side_effect=fake_get_page)
    deps = _make_deps(doc_store=doc_store, client=_catalog_client([_DECISION_10747, _REGULATION_10747]))

    result = await _capture_get_bddk_document(deps)("10747", 1)

    assert "Document ID: mevzuat_10747" in result.text
    assert "BANKALARIN MUHASEBE UYGULAMALARI" in result.text
    note = [line for line in result.text.splitlines() if "1207" in line]
    assert len(note) == 1
    assert "decision number" in note[0]


@pytest.mark.asyncio
async def test_a_resmi_gazete_identifier_is_not_labelled_as_a_kurul_decision():
    entry = BddkDecisionSummary(
        title="Bankalarca Kamuya Açıklanacak Finansal Tablolar Tebliğinde Değişiklik - Net İstikrarlı Fonlama Oranı",
        document_id="rg_32202_20230526_6",
        decision_date="26.05.2023",
        decision_number="32202",
        category="Tebliğ",
    )
    doc_store = _paged_store(["metin\n"])
    deps = _make_deps(doc_store=doc_store, client=_catalog_client([entry]))

    result = await _capture_get_bddk_document(deps)("rg_32202_20230526_6", 1, 1)

    assert "Decision date / number" not in result.text
    assert "- Resmî Gazete date / issue: 26.05.2023 / 32202" in result.text
    # Stored metadata is unchanged.
    assert result.structuredContent["evidence"][0]["decision_number"] == "32202"


def test_decision_identifier_label():
    from bddk_mcp.tools.structured_outputs import decision_identifier_label

    assert decision_identifier_label("rg_32202_20230526_6") == "Resmî Gazete date / issue"
    assert decision_identifier_label("779") == "Decision date / number"
    assert decision_identifier_label("mevzuat_5411") == "Decision date / number"


@pytest.mark.asyncio
async def test_tool_written_warnings_come_before_the_verbatim_claim_and_only_stored_text_follows_it():
    pages = [f"Sayfa {n} metni\n" for n in range(1, 25)]
    pages[2] = "Giriş\n$$\n\\text{İKÇ} = \\ln\\left(\\ex"
    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(pages)))

    single = await tool("954", 3, 1)
    multi = await tool("954", 2, 3)

    for result, content in ((single, pages[2]), (multi, None)):
        text = result.text
        claim = text.index("The text below is the exact stored document content")
        assert text.rindex("⚠") < claim
        assert "Bu belge 24 sayfa" in text[:claim]
        assert "Page 3 ends inside a formula that continues on page 4" in text[:claim]
        assert "⚠" not in text[claim:]
        assert "Retrieval warnings:" not in text
        assert "current legal applicability" in text[:claim]
        if content is not None:
            assert "inside a word, a table row or a formula.\n\n" + content in text
    assert multi.text.index("tool separators") < multi.text.index("### Page 2/24")
    assert multi.text.index("### Page 2/24") > multi.text.index("The text below is the exact stored")


@pytest.mark.asyncio
async def test_page_starting_inside_a_cut_table_row_or_formula_names_the_previous_page():
    pages = [
        "Giriş\n| 030001 | YERLİ MEVDUA",
        "T BANKALARI |\n| 2 | b |\n",
        "Metin\n$$\nx = 1",
        "+ 2\n$$\nson\n",
        "Temiz\n",
    ]
    structured_warnings = lambda r: r.structuredContent["warnings"]  # noqa: E731
    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(pages)))

    first = await tool("1043", 1, 1)
    row = await tool("1043", 2, 1)
    both = await tool("1043", 1, 2)
    formula = await tool("1043", 4, 1)
    clean = await tool("1043", 5, 1)

    assert "Page 2 starts inside a table row that began on page 1" in row.text
    assert "first line of this page is the rest of a row begun there" in row.text
    assert "Retrieve page 1 as well before quoting that row" in row.text
    assert any("Page 2 starts inside a table row" in w for w in structured_warnings(row))
    # The opening fragment is before the verbatim claim, not after it.
    assert row.text.index("Page 2 starts inside") < row.text.index("The text below is the exact stored")
    # Both pages in one response: the previous page is not asked for again.
    assert "Retrieve page 1" not in both.text
    assert "begun at the end of page 1 in this response" in both.text
    # A formula opened on the previous page, on a page that is also the last one.
    assert "Page 4 starts inside a formula opened on page 3" in formula.text
    assert "first lines of this page are the rest of a formula begun there" in formula.text
    assert "Retrieve page 3 as well before quoting that formula" in formula.text
    # Clean boundary, and page 1 itself: no start warning.
    assert "starts inside" not in clean.text and "starts inside" not in first.text


@pytest.mark.asyncio
async def test_both_tools_show_the_same_status_after_a_catalog_refresh_and_sync(doc_store):
    from bddk_mcp.ingest.client import BddkApiClient
    from bddk_mcp.ingest.doc_sync import DocumentSyncer
    from bddk_mcp.store.doc_store import StoredDocument
    from bddk_mcp.tools import sections as sections_tools

    await doc_store.store_document(
        StoredDocument(
            document_id="1310",
            title="Faiz Oranı Riski Yönetmeliği",
            category="Yönetmelik",
            source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1310",
            markdown_content="MADDE 1 - (1) Bankalar faiz oranı riskini ölçer.\n\nMADDE 2 - (2) Rapor verilir.\n",
        )
    )
    # `sync --refresh-catalog` publishes a listing in which BDDK now files the instrument as repealed.
    await doc_store.replace_catalog(
        [
            {
                "document_id": "1310",
                "title": "Faiz Oranı Riski Yönetmeliği",
                "category": "Mülga Düzenleme",
                "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1310",
            }
        ]
    )
    client = BddkApiClient(doc_store)
    await client.ensure_cache()
    deps = Dependencies(db=None, doc_store=doc_store, client=client, http=None)
    mcp = MagicMock()
    sections_tools.register(mcp, deps)
    section_tool = next(
        c.args[0] for c in mcp.tool.return_value.call_args_list if c.args[0].__name__ == "get_document_section"
    )
    document_tool = _capture_get_bddk_document(deps)

    async with DocumentSyncer(doc_store, ocr_backends=[]) as syncer:
        await syncer.sync_document("1310", title="Faiz Oranı Riski Yönetmeliği", category="Mülga Düzenleme")

    document = await document_tool("1310")
    section = await section_tool("1310", "madde", "1")
    assert "MÜLGA" in document.text and "MÜLGA" in section.text


_ANNEX_PLACEHOLDER_BODY = (
    "MADDE 1 - (1) Bu Yönetmelik denetimi düzenler.\n\nMADDE 2 - (1) Ek-1 formu kullanılır.\n\nEkleri için tıklayınız."
)


def _real_store_deps(doc_store) -> Dependencies:
    return Dependencies(
        db=None, doc_store=doc_store, client=MagicMock(find_by_id=MagicMock(return_value=None)), http=None
    )


@pytest.mark.asyncio
async def test_a_placeholder_for_annexes_that_are_not_stored_is_noticed_before_the_verbatim_claim(doc_store):
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(document_id="mevzuat_39257", title="BSD Yönetmeliği", markdown_content=_ANNEX_PLACEHOLDER_BODY)
    )
    tool = _capture_get_bddk_document(_real_store_deps(doc_store))

    result = await tool("mevzuat_39257", 1)

    notice = next(w for w in result.structuredContent["warnings"] if "annexes" in w)
    assert "not stored in this corpus" in notice
    assert '"Ekleri için tıklayınız."' in notice and "source" in notice
    assert result.text.count(notice) == 1
    claim = result.text.index("The text below is the exact stored document content")
    assert result.text.index(notice) < claim
    assert "Retrieval warnings:" not in result.text
    assert result.text.count("annexes (ekler) of this instrument are not stored") == 1


@pytest.mark.asyncio
async def test_no_annex_notice_for_prose_that_mentions_ekleri_or_for_a_document_that_stores_its_annex(doc_store):
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(
            document_id="mevzuat_11111",
            title="Düz Yönetmelik",
            markdown_content="MADDE 1 - (1) Bu Yönetmeliğin ekleri bakımından düzenleme yapılır; ekleri için tıklayınız "
            "demek yeterli değildir.\n\nMADDE 2 - (1) Son.\n",
        )
    )
    await doc_store.store_document(
        StoredDocument(
            document_id="mevzuat_19498",
            title="Ekli Yönetmelik",
            markdown_content="MADDE 1 - (1) Metin.\n\nEki için tıklayınız.\n\nEK-1\n\nBİLDİRİM FORMU\n\nForm alanları burada.\n",
        )
    )
    tool = _capture_get_bddk_document(_real_store_deps(doc_store))

    for doc_id in ("mevzuat_11111", "mevzuat_19498"):
        result = await tool(doc_id, 1)
        assert "annexes" not in result.text
        assert not any("annex" in w for w in result.structuredContent["warnings"])


@pytest.mark.asyncio
async def test_formula_cut_warnings_are_exact_for_each_edge():
    # Page 1 ends inside a formula; page 2 only closes it; page 3 opens and closes one inside itself;
    # page 4 opens a formula and ends inside it; page 5 is cut at both edges; page 6 only closes it.
    pages = [
        "a\n$$ x +",
        "y $$\nb\n",
        "c $$ z $$\n",
        "d\n$$ u +",
        "v +",
        "w $$\nend\n",
    ]
    tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(pages)))

    out = {n: await tool("1240", n, 1) for n in range(1, 7)}

    for n in range(1, 7):
        assert "cannot tell which" not in out[n].text and "cut at the start or the end" not in out[n].text
    assert "Page 1 ends inside a formula that continues on page 2" in out[1].text
    assert "Retrieve page 2 as well before quoting that formula" in out[1].text
    assert "starts inside a formula" not in out[1].text
    # Page 2 only starts inside one: previous page only, no request for page 3.
    assert "Page 2 starts inside a formula opened on page 1" in out[2].text
    assert "Retrieve page 1 as well" in out[2].text
    assert "ends inside a formula" not in out[2].text and "page 3" not in out[2].text
    assert "starts inside" not in out[3].text and "ends inside" not in out[3].text
    assert "Page 4 ends inside a formula that continues on page 5" in out[4].text
    assert "starts inside" not in out[4].text
    assert "Page 5 starts inside a formula opened on page 4" in out[5].text
    assert "Page 5 ends inside a formula that continues on page 6" in out[5].text
    assert "Retrieve page 4 as well" in out[5].text and "Retrieve page 6 as well" in out[5].text
    # The last page asks for the previous page only.
    assert "Page 6 starts inside a formula opened on page 5" in out[6].text
    assert "page 7" not in out[6].text and "ends inside" not in out[6].text
    both = await tool("1240", 4, 3)
    assert "Retrieve page" not in both.text
    assert "continuation is at the start of page 5 in this response" in both.text


@pytest.mark.asyncio
async def test_a_complete_table_row_at_a_page_edge_is_no_cut_but_a_row_cut_mid_cell_warns_on_both_sides():
    complete = ["| a | b |\n| 1 | 2 |", "\n| 3 | 4 |\nson\n"]
    cut = ["| a | b |\n| 1 | 2", " |\n| 3 | 4 |\nson\n"]
    complete_tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(complete)))
    cut_tool = _capture_get_bddk_document(_make_deps(doc_store=_paged_store(cut)))

    for n in (1, 2):
        text = (await complete_tool("1043", n, 1)).text
        assert "table row" not in text.split("The text below")[0]
    first = (await cut_tool("1043", 1, 1)).text
    second = (await cut_tool("1043", 2, 1)).text
    assert "Page 1 ends inside a table row that continues on the next page" in first
    assert "Page 2 starts inside a table row that began on page 1" in second
