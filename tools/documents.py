"""Document retrieval and management tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from exceptions import BddkStorageError

if TYPE_CHECKING:
    from deps import Dependencies

logger = logging.getLogger(__name__)

# Backend names whose output preserves mathematical formulas and inline images.
# Combined method strings (e.g. "mevzuat_pdf+lightocr", "html_parser+manual_latex")
# are matched by substring. "manual_latex" is the marker for documents that were
# hand-corrected to embed LaTeX where OCR failed.
_FORMULA_AWARE_TOKENS = ("lightocr", "chandra2", "pp_structure", "manual_latex")

_DEGRADED_WARNING = (
    "Bu belgedeki matematiksel formüller ve bazı görseller çıkartılamamış olabilir. "
    "Metin 'aşağıdaki formül', 'aşağıda yer alan formül' gibi bir ifadeye atıfta bulunuyorsa, "
    "formülü hafızadan veya standart literatürden yeniden kurma — kullanıcıyı kaynak PDF'e yönlendir."
)


def _is_formula_aware(method: str) -> bool:
    """True when the extraction method used a formula-preserving OCR backend."""
    if not method:
        return False
    lower = method.lower()
    return any(token in lower for token in _FORMULA_AWARE_TOKENS)


def register(mcp, deps: Dependencies) -> None:
    """Register document tools on the given MCP instance."""

    @mcp.tool()
    async def get_bddk_document(
        document_id: str,
        page_number: int = 1,
    ) -> str:
        """
        Retrieve a BDDK decision document as Markdown.

        Airlocked: serves only from local stores (pgvector chunks, then PostgreSQL
        documents). If the document is not present locally, returns a clear
        "not in seed" error rather than live-fetching from mevzuat.gov.tr / BDDK.

        A bare numeric ID (e.g. "21192") is also tried as `mevzuat_<id>` and
        `bddk_<id>` so callers don't need to know the catalog's prefix
        convention. The resolved ID is shown in the header.

        Args:
            document_id: The numeric document ID (from search results)
            page_number: Page of the markdown output (documents are split into 5000-char pages)
        """
        candidates = (
            [document_id, f"mevzuat_{document_id}", f"bddk_{document_id}"] if document_id.isdigit() else [document_id]
        )

        resolved_id: str | None = None
        page_num = 0
        total_pages = 0
        content = ""
        extraction_method = ""
        served_via_vector = False

        for cand in candidates:
            if deps.vector_store is not None:
                try:
                    vp = await deps.vector_store.get_document_page(cand, page_number)
                    if vp and vp["content"] and "Invalid page" not in vp["content"]:
                        resolved_id = cand
                        page_num, total_pages, content = (
                            vp["page_number"],
                            vp["total_pages"],
                            vp["content"],
                        )
                        served_via_vector = True
                        break
                except Exception as e:
                    logger.debug("pgvector lookup failed for %s: %s", cand, e)

            try:
                stored = await deps.doc_store.get_document_page(cand, page_number)
            except (RuntimeError, BddkStorageError) as e:
                logger.warning("doc_store lookup failed for %s: %s", cand, e)
                stored = None

            if stored and stored.markdown_content and "Invalid page" not in stored.markdown_content:
                resolved_id = cand
                page_num, total_pages, content = (
                    stored.page_number,
                    stored.total_pages,
                    stored.markdown_content,
                )
                extraction_method = stored.extraction_method or ""
                break

        if resolved_id is None:
            return (
                f"Document {document_id} is not available in the local store. "
                "This MCP server is airlocked and does not fetch from live BDDK / mevzuat.gov.tr sources at runtime. "
                "If the document should be available, re-run the seed (`seed.py import`) or sync pipeline."
            )

        meta_title = resolved_id
        meta_date = ""
        meta_number = ""
        meta_category = ""
        source_url = ""
        found = deps.client.find_by_id(resolved_id)
        if found:
            meta_title = found.title
            meta_date = found.decision_date
            meta_number = found.decision_number
            meta_category = found.category
            source_url = found.source_url or ""

        alias_line = f"- Resolved from: `{document_id}` -> `{resolved_id}`\n" if resolved_id != document_id else ""

        # The pgvector chunk rows don't carry extraction_method; look it up.
        if served_via_vector:
            try:
                extraction_method = await deps.doc_store.get_extraction_method(resolved_id) or ""
            except (RuntimeError, BddkStorageError) as e:
                logger.debug("extraction_method lookup failed for %s: %s", resolved_id, e)

        degraded = bool(extraction_method) and not _is_formula_aware(extraction_method)
        method_display = extraction_method or "unknown"
        if degraded:
            method_display = f"{method_display} (formula-unaware — equations/images may be missing)"

        warning_block = f"⚠ {_DEGRADED_WARNING}\n\n" if degraded else ""

        header = (
            f"## {meta_title}\n"
            f"- Document ID: {resolved_id}\n"
            f"{alias_line}"
            f"- Decision Date: {meta_date or 'N/A'}\n"
            f"- Decision Number: {meta_number or 'N/A'}\n"
            f"- Category: {meta_category or 'N/A'}\n"
            f"- Source: {source_url or 'N/A'}\n"
            f"- Page: {page_num}/{total_pages}\n"
            f"- Extraction: {method_display}\n"
            f"---\n"
            f"Use ONLY the text below. Do not add information not present in this document.\n\n"
            f"{warning_block}"
        )

        return header + content

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
