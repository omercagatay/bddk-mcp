"""Tests for pgvector fallback and singleton initialization in server.py."""

from unittest.mock import AsyncMock, patch

import pytest


class TestVectorStoreSingletonRetry:
    """_get_vector_store() must not cache a broken instance after init failure."""

    @pytest.mark.asyncio
    async def test_failed_init_does_not_cache_singleton(self):
        """If VectorStore.initialize() raises, _vector_store stays None so next call retries."""
        import server

        # Reset global state
        original = server._vector_store
        server._vector_store = None

        mock_pool = AsyncMock()

        try:
            with patch.object(server, "_get_pool", return_value=mock_pool):
                with patch("server.VectorStore") as MockVS:
                    instance = MockVS.return_value
                    instance.initialize = AsyncMock(
                        side_effect=RuntimeError("pgvector extension not available")
                    )

                    with pytest.raises(RuntimeError, match="pgvector extension not available"):
                        await server._get_vector_store()

                    # Singleton must remain None so next call retries
                    assert server._vector_store is None

                    # Second call with successful init should work
                    instance.initialize = AsyncMock()
                    result = await server._get_vector_store()
                    assert result is instance
                    assert server._vector_store is instance
        finally:
            server._vector_store = original

    @pytest.mark.asyncio
    async def test_successful_init_caches_singleton(self):
        """After successful initialize(), singleton is cached."""
        import server

        original = server._vector_store
        server._vector_store = None

        mock_pool = AsyncMock()

        try:
            with patch.object(server, "_get_pool", return_value=mock_pool):
                with patch("server.VectorStore") as MockVS:
                    instance = MockVS.return_value
                    instance.initialize = AsyncMock()

                    result = await server._get_vector_store()
                    assert result is instance
                    assert server._vector_store is instance
                    instance.initialize.assert_called_once()
        finally:
            server._vector_store = original


class TestGetBddkDocumentFallback:
    """get_bddk_document must fall back to doc store when pgvector fails."""

    @pytest.mark.asyncio
    async def test_falls_back_on_database_error(self):
        """If pgvector raises any exception, fallback to document store."""
        import server
        from models import BddkDocumentMarkdown

        mock_client = AsyncMock()
        mock_client._cache = []
        mock_client.get_document_markdown = AsyncMock(
            return_value=BddkDocumentMarkdown(
                document_id="956",
                markdown_content="Fallback content from doc store",
                page_number=1,
                total_pages=1,
            )
        )

        mock_vs = AsyncMock()
        # Simulate asyncpg.UndefinedTableError (relation does not exist)
        mock_vs.get_document_page = AsyncMock(
            side_effect=Exception('relation "document_chunks" does not exist')
        )

        with patch.object(server, "_get_client", return_value=mock_client):
            with patch.object(server, "_get_vector_store", return_value=mock_vs):
                result = await server.get_bddk_document("956")

        assert "Fallback content from doc store" in result
        mock_client.get_document_markdown.assert_called_once_with("956", 1)

    @pytest.mark.asyncio
    async def test_uses_pgvector_when_available(self):
        """If pgvector works, use it without falling back."""
        import server

        mock_client = AsyncMock()
        mock_client._cache = []

        mock_vs = AsyncMock()
        mock_vs.get_document_page = AsyncMock(
            return_value={
                "content": "pgvector content here",
                "page_number": 1,
                "total_pages": 1,
            }
        )

        with patch.object(server, "_get_client", return_value=mock_client):
            with patch.object(server, "_get_vector_store", return_value=mock_vs):
                result = await server.get_bddk_document("956")

        assert "pgvector content here" in result
        mock_client.get_document_markdown.assert_not_called()
