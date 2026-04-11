"""Document retrieval and management tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from exceptions import BddkError, BddkStorageError
from metrics import metrics

if TYPE_CHECKING:
    from deps import Dependencies

logger = logging.getLogger(__name__)


def register(mcp, deps: Dependencies) -> None:
    """Register document tools on the given MCP instance."""

    @mcp.tool()
    async def get_bddk_document(
        document_id: str,
        page_number: int = 1,
    ) -> str:
        """
        Retrieve a BDDK decision document as Markdown.

        Uses local pgvector store for instant retrieval.
        Falls back to PostgreSQL document store, then live fetch if not found locally.

        Args:
            document_id: The numeric document ID (from search results)
            page_number: Page of the markdown output (documents are split into 5000-char pages)
        """
        # Look up metadata from cache
        client = deps.client
        meta_title = document_id
        meta_date = ""
        meta_number = ""
        meta_category = ""
        source_url = ""
        for dec in client._cache:
            if dec.document_id == document_id:
                meta_title = dec.title
                meta_date = dec.decision_date
                meta_number = dec.decision_number
                meta_category = dec.category
                source_url = dec.source_url or ""
                break

        def _build_header(page_num: int, total: int) -> str:
            return (
                f"## {meta_title}\n"
                f"- Document ID: {document_id}\n"
                f"- Decision Date: {meta_date or 'N/A'}\n"
                f"- Decision Number: {meta_number or 'N/A'}\n"
                f"- Category: {meta_category or 'N/A'}\n"
                f"- Source: {source_url or 'N/A'}\n"
                f"- Page: {page_num}/{total}\n"
                f"---\n"
                f"Use ONLY the text below. Do not add information not present in this document.\n\n"
            )

        # Try pgvector first (instant)
        if deps.vector_store is not None:
            try:
                page = await deps.vector_store.get_document_page(document_id, page_number)
                if page and page["content"] and "Invalid page" not in page["content"]:
                    return _build_header(page["page_number"], page["total_pages"]) + page["content"]
            except Exception as e:
                logger.debug("pgvector lookup failed for %s, falling back: %s", document_id, e)

        # Fallback to document store → live fetch
        doc = await client.get_document_markdown(document_id, page_number)
        return _build_header(doc.page_number, doc.total_pages) + doc.markdown_content

    @mcp.tool()
    async def get_document_history(
        document_id: str,
    ) -> str:
        """
        Get version history for a BDDK document.

        Shows all previous versions with timestamps and content hashes.

        Args:
            document_id: The document ID (from search results)
        """
        store = deps.doc_store
        history = await store.get_document_history(document_id)

        if not history:
            return f"No version history found for document {document_id}."

        lines = [f"**Version History for {document_id}** ({len(history)} version(s)):\n"]
        for v in history:
            lines.append(
                f"  v{v['version']} — {v['synced_at']} (hash: {v['content_hash'][:12]}..., {v['content_length']} chars)"
            )

        return "\n".join(lines)

    @mcp.tool()
    async def document_store_stats() -> str:
        """
        Show document store statistics for PostgreSQL and pgvector stores.
        """
        lines = ["**Document Store Statistics**\n"]

        # pgvector stats
        if deps.vector_store is not None:
            try:
                vs_stats = await deps.vector_store.stats()
                lines.append("**pgvector (Vector Store):**")
                lines.append(f"  Documents: {vs_stats['total_documents']}")
                lines.append(f"  Chunks: {vs_stats['total_chunks']}")
                lines.append(f"  Embedding model: {vs_stats['embedding_model']}")
                if vs_stats.get("categories"):
                    lines.append("  Categories:")
                    for cat, count in vs_stats["categories"].items():
                        lines.append(f"    {cat}: {count}")
            except Exception as e:
                lines.append(f"  pgvector: unavailable ({e})")
        else:
            lines.append("  pgvector: unavailable (not initialized)")

        # PostgreSQL document stats
        try:
            store = deps.doc_store
            st = await store.stats()
            lines.append("\n**PostgreSQL (Document Store):**")
            lines.append(f"  Documents: {st.total_documents}")
            lines.append(f"  Size: {st.total_size_mb} MB")
        except (RuntimeError, BddkStorageError) as e:
            lines.append(f"  PostgreSQL: unavailable ({e})")

        return "\n".join(lines)
