"""get_bddk_document serves only the local corpus: no live fetch, no second store."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError


def _register_and_get_tool(deps):
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP("test")

    from bddk_mcp.tools import documents

    documents.register(test_mcp, deps)

    for tool in test_mcp._tool_manager._tools.values():
        if tool.name == "get_bddk_document":
            return tool.fn
    raise AssertionError("get_bddk_document tool not registered")


class TestGetBddkDocumentAirlock:
    """get_bddk_document is airlocked — it must never call the live-fetch client path."""

    @pytest.mark.asyncio
    async def test_serves_pages_from_the_document_store(self):
        from bddk_mcp.core.deps import Dependencies
        from bddk_mcp.store.doc_store import DocumentPage

        mock_client = MagicMock()
        mock_client._cache = []
        mock_client.find_by_id.return_value = None

        mock_doc_store = MagicMock()
        mock_doc_store.get_document_quality = AsyncMock(return_value=None)
        mock_doc_store.get_document_page = AsyncMock(
            return_value=DocumentPage(
                document_id="956",
                title="Test",
                markdown_content="Local doc_store content",
                page_number=1,
                total_pages=1,
                extraction_method="markitdown",
                category="",
            )
        )

        deps = MagicMock(spec=Dependencies)
        deps.client = mock_client
        deps.vector_store = MagicMock()
        deps.doc_store = mock_doc_store

        result = await _register_and_get_tool(deps)("956")

        assert "Local doc_store content" in result
        assert result.structuredContent["served_via"] == "document_store"
        mock_doc_store.get_document_page.assert_awaited_once_with("956", 1)
        deps.vector_store.assert_not_called()
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_not_in_seed_when_the_store_misses(self):
        """When the corpus has no such document, return an error — never live-fetch."""
        from bddk_mcp.core.deps import Dependencies

        mock_client = MagicMock()
        mock_client._cache = []
        mock_client.find_by_id.return_value = None

        mock_doc_store = MagicMock()
        mock_doc_store.get_document_page = AsyncMock(return_value=None)

        deps = MagicMock(spec=Dependencies)
        deps.client = mock_client
        deps.vector_store = MagicMock()
        deps.doc_store = mock_doc_store

        with pytest.raises(ToolError) as error:
            await _register_and_get_tool(deps)("mevzuat_42626")

        assert "[ERROR:NOT_FOUND] retryable=false" in str(error.value)
        assert "airlocked" in str(error.value).lower()
        assert "mevzuat_42626" in str(error.value)
        assert "bddk-mcp build" in str(error.value)
