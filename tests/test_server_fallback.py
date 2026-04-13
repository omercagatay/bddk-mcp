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


class TestGetBddkDocumentFallback:
    """get_bddk_document tool must fall back to doc store when pgvector fails."""

    @pytest.mark.asyncio
    async def test_falls_back_on_database_error(self):
        """If pgvector raises any exception, fallback to document store."""
        from deps import Dependencies
        from models import BddkDocumentMarkdown

        mock_client = MagicMock()
        mock_client._cache = []
        mock_client.get_document_markdown = AsyncMock(
            return_value=BddkDocumentMarkdown(
                document_id="956",
                markdown_content="Fallback content from doc store",
                page_number=1,
                total_pages=1,
            )
        )

        mock_vs = MagicMock()
        mock_vs.get_document_page = AsyncMock(side_effect=Exception('relation "document_chunks" does not exist'))

        deps = MagicMock(spec=Dependencies)
        deps.client = mock_client
        deps.vector_store = mock_vs

        # Import the tool via the tools module directly
        from mcp.server.fastmcp import FastMCP

        test_mcp = FastMCP("test")

        from tools import documents

        documents.register(test_mcp, deps)

        # Find the registered tool
        tool_fn = None
        for tool in test_mcp._tool_manager._tools.values():
            if tool.name == "get_bddk_document":
                tool_fn = tool.fn
                break

        assert tool_fn is not None, "get_bddk_document tool not registered"
        result = await tool_fn("956")

        assert "Fallback content from doc store" in result
        mock_client.get_document_markdown.assert_called_once_with("956", 1)

    @pytest.mark.asyncio
    async def test_uses_pgvector_when_available(self):
        """If pgvector works, use it without falling back."""
        from deps import Dependencies

        mock_client = MagicMock()
        mock_client._cache = []

        mock_vs = MagicMock()
        mock_vs.get_document_page = AsyncMock(
            return_value={
                "content": "pgvector content here",
                "page_number": 1,
                "total_pages": 1,
            }
        )

        deps = MagicMock(spec=Dependencies)
        deps.client = mock_client
        deps.vector_store = mock_vs

        from mcp.server.fastmcp import FastMCP

        test_mcp = FastMCP("test")

        from tools import documents

        documents.register(test_mcp, deps)

        tool_fn = None
        for tool in test_mcp._tool_manager._tools.values():
            if tool.name == "get_bddk_document":
                tool_fn = tool.fn
                break

        assert tool_fn is not None, "get_bddk_document tool not registered"
        result = await tool_fn("956")

        assert "pgvector content here" in result
        mock_client.get_document_markdown.assert_not_called()
