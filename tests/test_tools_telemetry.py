"""Tests that retrieval tools emit optional telemetry metadata."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deps import Dependencies
from doc_store import StoredDocumentSection
from tools.sections import register as register_sections


def _capture_section_tool(deps: Dependencies, name: str):
    mcp = MagicMock()
    register_sections(mcp, deps)
    for call in mcp.tool.return_value.call_args_list:
        fn = call.args[0]
        if fn.__name__ == name:
            return fn
    raise AssertionError(f"{name} not registered")


def _section(doc_id: str, section_ref: str) -> StoredDocumentSection:
    return StoredDocumentSection(
        doc_id=doc_id,
        section_type="ilke",
        section_ref=section_ref,
        heading="Model validasyonu",
        start_char=10,
        end_char=80,
        content=f"İlke {section_ref}\nModel validasyonu yapılır.",
        content_hash=f"hash-{section_ref}",
    )


@pytest.mark.asyncio
async def test_search_document_sections_records_doc_ids_and_latency(monkeypatch):
    from tools import sections as sections_mod

    recorder = AsyncMock(return_value=True)
    monkeypatch.setattr(sections_mod, "record_tool_call_trace", recorder)

    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    doc_store.search_document_sections = AsyncMock(return_value=[_section("943", "5"), _section("943", "6")])
    deps = Dependencies(pool=object(), doc_store=doc_store, client=None, http=None)

    tool = _capture_section_tool(deps, "search_document_sections")
    await tool("943 İlke 5 model validasyonu", limit=5)

    recorder.assert_awaited_once()
    kwargs = recorder.await_args.kwargs
    assert kwargs["tool_name"] == "search_document_sections"
    assert kwargs["args"]["query"] == "943 İlke 5 model validasyonu"
    assert kwargs["latency_ms"] >= 0
    assert kwargs["result_count"] == 2
    assert kwargs["doc_ids"] == ["943"]
    assert kwargs["relevance_stats"]["exact_ref_detected"] is True
