"""Document retrieval and management tools."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from config import ADMIN_TOOLS
from exceptions import BddkStorageError
from markdown_quality import assess_markdown_quality, sanitize_markdown_for_context
from telemetry import elapsed_ms, record_tool_call_trace
from tools.tool_logging import logged_tool

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

_LARGE_DOCUMENT_PAGE_THRESHOLD = 20
_MAX_PAGES_PER_RESPONSE = 5
_LARGE_DOCUMENT_WARNING_TEMPLATE = (
    "Bu belge {total_pages} sayfa. Tam belgeyi sayfa sayfa çekmek context'i hızla şişirebilir. "
    "Hedefli retrieval için önce search_document_sections veya get_document_section kullan; "
    "yalnızca gerekli sayfalar için get_bddk_document içinde page_number ve max_pages parametrelerini kullan."
)


def _is_formula_aware(method: str) -> bool:
    """True when the extraction method used a formula-preserving OCR backend."""
    if not method:
        return False
    lower = method.lower()
    return any(token in lower for token in _FORMULA_AWARE_TOKENS)


def _normalize_max_pages(max_pages: int) -> tuple[int, bool]:
    try:
        requested = int(max_pages)
    except (TypeError, ValueError):
        requested = 1
    normalized = max(1, min(requested, _MAX_PAGES_PER_RESPONSE))
    return normalized, requested != normalized


def register(mcp, deps: Dependencies) -> None:
    """Register document tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def get_bddk_document(
        document_id: str,
        page_number: int = 1,
        max_pages: int = 1,
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
            page_number: First page of the markdown output (documents are split into 5000-char pages)
            max_pages: Maximum consecutive pages to return, capped at 5 to avoid context overflow
        """
        start = time.perf_counter()
        max_pages, max_pages_capped = _normalize_max_pages(max_pages)
        args = {"document_id": document_id, "page_number": page_number, "max_pages": max_pages}
        candidates = [document_id] + (
            [f"mevzuat_{document_id}", f"bddk_{document_id}"] if document_id.isdigit() else []
        )

        async def _lookup_page(cand: str, page: int) -> tuple[int, int, str, str, bool] | None:
            if deps.vector_store is not None:
                try:
                    vp = await deps.vector_store.get_document_page(cand, page)
                    if vp and vp["content"] and "Invalid page" not in vp["content"]:
                        return vp["page_number"], vp["total_pages"], vp["content"], "", True
                except Exception as e:
                    logger.debug("pgvector lookup failed for %s: %s", cand, e)

            try:
                stored = await deps.doc_store.get_document_page(cand, page)
            except (RuntimeError, BddkStorageError) as e:
                logger.warning("doc_store lookup failed for %s: %s", cand, e)
                stored = None

            if stored and stored.markdown_content and "Invalid page" not in stored.markdown_content:
                return (
                    stored.page_number,
                    stored.total_pages,
                    stored.markdown_content,
                    stored.extraction_method or "",
                    False,
                )
            return None

        resolved_id: str | None = None
        page_num = 0
        total_pages = 0
        page_contents: list[tuple[int, str]] = []
        extraction_method = ""
        served_sources: set[str] = set()

        for cand in candidates:
            found_page = await _lookup_page(cand, page_number)
            if found_page:
                resolved_id = cand
                page_num, total_pages, content, extraction_method, served_via_vector = found_page
                page_contents.append((page_num, content))
                served_sources.add("vector_store" if served_via_vector else "document_store")
                break

        if resolved_id is None:
            await record_tool_call_trace(
                getattr(deps, "pool", None),
                tool_name="get_bddk_document",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "not_found"},
            )
            return f"Document {document_id} is not available in the local store. This MCP server is airlocked and does not fetch from live BDDK / mevzuat.gov.tr sources at runtime. If the document should be available, re-run the seed (`seed.py import`) or sync pipeline."

        found = deps.client.find_by_id(resolved_id)
        meta_title, meta_date, meta_number, meta_category, source_url = (
            (found.title, found.decision_date, found.decision_number, found.category, found.source_url or "")
            if found
            else (resolved_id, "", "", "", "")
        )

        alias_line = f"- Resolved from: `{document_id}` -> `{resolved_id}`\n" if resolved_id != document_id else ""

        last_page_num = page_num
        if max_pages > 1 and page_num < total_pages:
            final_page = min(page_num + max_pages - 1, total_pages)
            for next_page in range(page_num + 1, final_page + 1):
                found_page = await _lookup_page(resolved_id, next_page)
                if not found_page:
                    break
                last_page_num, total_pages, page_content, page_extraction_method, served_via_vector = found_page
                page_contents.append((last_page_num, page_content))
                served_sources.add("vector_store" if served_via_vector else "document_store")
                if page_extraction_method and not extraction_method:
                    extraction_method = page_extraction_method

        # The pgvector chunk rows don't carry extraction_method; look it up.
        if "vector_store" in served_sources:
            try:
                extraction_method = await deps.doc_store.get_extraction_method(resolved_id) or ""
            except (RuntimeError, BddkStorageError) as e:
                logger.debug("extraction_method lookup failed for %s: %s", resolved_id, e)

        if len(page_contents) > 1:
            content = "\n\n---\n\n".join(
                f"### Page {page}/{total_pages}\n\n{page_content}" for page, page_content in page_contents
            )
        else:
            content = page_contents[0][1]

        formula_aware = _is_formula_aware(extraction_method)
        quality = assess_markdown_quality(content, document_id=resolved_id)
        if formula_aware and quality.flags == ["formula_ref_without_latex_or_image"]:
            quality.label = "clean"
            quality.flags = []
            quality.warning = ""
        content = sanitize_markdown_for_context(content)

        degraded = bool(extraction_method) and not formula_aware
        method_display = extraction_method or "unknown"
        if degraded:
            method_display = f"{method_display} (formula-unaware — equations/images may be missing)"

        quality_lines = ""
        quality_warning_block = ""
        if quality.label != "clean":
            flags = ", ".join(quality.flags) if quality.flags else "none"
            quality_lines = f"- Quality: {quality.label}\n- Quality flags: {flags}\n"
            quality_warning_block = f"⚠ Quality warning: {quality.warning}\n\n" if quality.warning else ""

        large_document_warning_block = ""
        if total_pages >= _LARGE_DOCUMENT_PAGE_THRESHOLD:
            large_document_warning_block = f"⚠ {_LARGE_DOCUMENT_WARNING_TEMPLATE.format(total_pages=total_pages)}\n\n"

        page_limit_warning_block = ""
        if max_pages_capped:
            page_limit_warning_block = (
                f"⚠ Requested max_pages was capped at {_MAX_PAGES_PER_RESPONSE} pages per response.\n\n"
            )

        degraded_warning_block = f"⚠ {_DEGRADED_WARNING}\n\n" if degraded else ""
        page_display = (
            f"{page_num}/{total_pages}" if last_page_num == page_num else f"{page_num}-{last_page_num}/{total_pages}"
        )

        header = (
            f"## {meta_title}\n- Document ID: {resolved_id}\n{alias_line}"
            f"- Decision Date: {meta_date or 'N/A'}\n- Decision Number: {meta_number or 'N/A'}\n"
            f"- Category: {meta_category or 'N/A'}\n- Source: {source_url or 'N/A'}\n"
            f"- Page: {page_display}\n- Extraction: {method_display}\n{quality_lines}---\n"
            "Use ONLY the text below. Do not add information not present in this document.\n\n"
            f"{large_document_warning_block}{page_limit_warning_block}{quality_warning_block}{degraded_warning_block}"
        )

        served_via = next(iter(served_sources)) if len(served_sources) == 1 else "mixed"
        await record_tool_call_trace(
            getattr(deps, "pool", None),
            tool_name="get_bddk_document",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=1,
            doc_ids=[resolved_id],
            quality_labels={
                resolved_id: {
                    "label": quality.label,
                    "flags": quality.flags,
                    "extraction_method": extraction_method or "unknown",
                }
            },
            relevance_stats={
                "page_number": page_num,
                "last_page_number": last_page_num,
                "total_pages": total_pages,
                "pages_returned": len(page_contents),
                "served_via": served_via,
            },
        )
        return header + content

    @mcp.tool()
    @logged_tool(logger)
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

    if not ADMIN_TOOLS:
        return

    @mcp.tool()
    @logged_tool(logger)
    async def document_store_stats() -> str:
        """
        Show document store statistics for PostgreSQL and pgvector stores.
        """
        lines = ["**Document Store Statistics**\n"]

        if deps.vector_store is not None:
            try:
                vs = await deps.vector_store.stats()
                lines.append(
                    f"**pgvector (Vector Store):**\n  Documents: {vs['total_documents']}\n"
                    f"  Chunks: {vs['total_chunks']}\n  Embedding model: {vs['embedding_model']}"
                )
                if vs.get("categories"):
                    lines.append("  Categories:")
                    for cat, count in vs["categories"].items():
                        lines.append(f"    {cat}: {count}")
            except Exception as e:
                lines.append(f"  pgvector: unavailable ({e})")
        else:
            lines.append("  pgvector: unavailable (not initialized)")

        try:
            st = await deps.doc_store.stats()
            lines.append(
                f"\n**PostgreSQL (Document Store):**\n  Documents: {st.total_documents}\n  Size: {st.total_size_mb} MB"
            )
        except (RuntimeError, BddkStorageError) as e:
            lines.append(f"  PostgreSQL: unavailable ({e})")

        return "\n".join(lines)
