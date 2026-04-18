"""Tests for pgvector fallback and VectorStore initialization in server.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVectorStoreInit:
    """init_vector_store() background task behavior."""

    @pytest.mark.asyncio
    async def test_failed_init_leaves_vector_store_none(self):
        """If VectorStore.initialize() raises, deps.vector_store stays None."""
        from deps import Dependencies
        from server import init_vector_store

        deps = MagicMock(spec=Dependencies)
        deps.pool = AsyncMock()
        deps.vector_store = None

        with patch("vector_store.VectorStore") as MockVS:
            instance = MockVS.return_value
            instance.initialize = AsyncMock(side_effect=RuntimeError("pgvector extension not available"))

            await init_vector_store(deps)

        # On failure, vector_store must not be set to a broken instance
        assert deps.vector_store is None or deps.vector_store != instance

    @pytest.mark.asyncio
    async def test_successful_init_sets_vector_store(self):
        """After successful initialize(), deps.vector_store is set."""
        from deps import Dependencies
        from server import init_vector_store

        deps = MagicMock(spec=Dependencies)
        deps.pool = AsyncMock()
        deps.vector_store = None

        with patch("vector_store.VectorStore") as MockVS:
            instance = MockVS.return_value
            instance.initialize = AsyncMock()

            await init_vector_store(deps)

        assert deps.vector_store is instance


def _register_and_get_tool(deps):
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP("test")

    from tools import documents

    documents.register(test_mcp, deps)

    for tool in test_mcp._tool_manager._tools.values():
        if tool.name == "get_bddk_document":
            return tool.fn
    raise AssertionError("get_bddk_document tool not registered")


class TestGetBddkDocumentAirlock:
    """get_bddk_document is airlocked — it must never call the live-fetch client path."""

    @pytest.mark.asyncio
    async def test_falls_back_to_doc_store_on_pgvector_error(self):
        """If pgvector raises, fall back to the local PostgreSQL doc_store (still local)."""
        from deps import Dependencies
        from doc_store import DocumentPage

        mock_client = MagicMock()
        mock_client._cache = []
        mock_client.get_document_markdown = AsyncMock()

        mock_vs = MagicMock()
        mock_vs.get_document_page = AsyncMock(side_effect=Exception('relation "document_chunks" does not exist'))

        mock_doc_store = MagicMock()
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
        deps.vector_store = mock_vs
        deps.doc_store = mock_doc_store

        result = await _register_and_get_tool(deps)("956")

        assert "Local doc_store content" in result
        mock_doc_store.get_document_page.assert_awaited_once_with("956", 1)
        mock_client.get_document_markdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_pgvector_when_available(self):
        """If pgvector works, use it without falling back."""
        from deps import Dependencies

        mock_client = MagicMock()
        mock_client._cache = []
        mock_client.get_document_markdown = AsyncMock()

        mock_vs = MagicMock()
        mock_vs.get_document_page = AsyncMock(
            return_value={"content": "pgvector content here", "page_number": 1, "total_pages": 1}
        )

        deps = MagicMock(spec=Dependencies)
        deps.client = mock_client
        deps.vector_store = mock_vs
        deps.doc_store = MagicMock()
        deps.doc_store.get_document_page = AsyncMock()

        result = await _register_and_get_tool(deps)("956")

        assert "pgvector content here" in result
        deps.doc_store.get_document_page.assert_not_called()
        mock_client.get_document_markdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_not_in_seed_when_both_stores_miss(self):
        """When both pgvector and doc_store miss, return an error — never live-fetch."""
        from deps import Dependencies

        mock_client = MagicMock()
        mock_client._cache = []
        mock_client.get_document_markdown = AsyncMock()

        mock_vs = MagicMock()
        mock_vs.get_document_page = AsyncMock(return_value=None)

        mock_doc_store = MagicMock()
        mock_doc_store.get_document_page = AsyncMock(return_value=None)

        deps = MagicMock(spec=Dependencies)
        deps.client = mock_client
        deps.vector_store = mock_vs
        deps.doc_store = mock_doc_store

        result = await _register_and_get_tool(deps)("mevzuat_42626")

        assert "airlocked" in result.lower()
        assert "mevzuat_42626" in result
        mock_client.get_document_markdown.assert_not_called()
