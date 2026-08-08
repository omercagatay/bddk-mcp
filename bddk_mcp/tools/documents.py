"""Document retrieval and management tools."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from bddk_mcp.core.exceptions import BddkStorageError
from bddk_mcp.observability.telemetry import elapsed_ms, record_tool_call_trace
from bddk_mcp.quality.markdown_quality import assess_markdown_quality, sanitize_markdown_for_context
from bddk_mcp.store.legal_ref import document_id_candidates
from bddk_mcp.tools.errors import INVALID_INPUT, NOT_FOUND, tool_error
from bddk_mcp.tools.structured_outputs import (
    UNTRUSTED_SOURCE_WARNING,
    DocumentHistoryResponse,
    DocumentHistoryToolResult,
    DocumentPageContent,
    DocumentResponse,
    DocumentToolResult,
    DocumentVersionItem,
    EvidenceReference,
    QualityMetadata,
    structured_tool_result,
)
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

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
DocumentId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
        description="Local BDDK/mevzuat document identifier returned by search.",
    ),
]
PageNumber = Annotated[int, Field(ge=1, le=10_000, description="1-based normalized document page number.")]
MaxPages = Annotated[int, Field(ge=1, le=5, description="Consecutive pages to return (1-5).")]
_PAGE_GAP_WARNING_TEMPLATE = (
    "Sayfa {missing_page} local store'dan alınamadı; yalnızca {first_page}-{last_page} arası döndürüldü. "
    "Belge toplam {total_pages} sayfa olarak kayıtlı — eksik sayfa, store tutarsızlığına işaret edebilir."
)
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


def register(
    mcp,
    deps: Dependencies,
    *,
    include_operator: bool = False,
) -> None:
    """Register document tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def get_bddk_document(
        document_id: DocumentId,
        page_number: PageNumber = 1,
        max_pages: MaxPages = 1,
    ) -> DocumentToolResult:
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
            max_pages: Consecutive pages to return (1-5)
        """
        start = time.perf_counter()
        if isinstance(max_pages, bool) or not 1 <= max_pages <= _MAX_PAGES_PER_RESPONSE:
            return tool_error(INVALID_INPUT, "max_pages must be between 1 and 5.", retryable=False)
        args = {"document_id": document_id, "page_number": page_number, "max_pages": max_pages}
        candidates = document_id_candidates(document_id)

        async def _lookup_page(cand: str, page: int) -> tuple[int, int, str, str, bool] | None:
            if deps.vector_store is not None:
                try:
                    vp = await deps.vector_store.get_document_page(cand, page)
                    if vp and vp["content"] and "Invalid page" not in vp["content"]:
                        return vp["page_number"], vp["total_pages"], vp["content"], "", True
                except Exception as exc:
                    logger.warning(
                        "Vector page lookup failed; falling back to document store",
                        extra={"error_type": type(exc).__name__},
                    )

            try:
                stored = await deps.doc_store.get_document_page(cand, page)
            except (RuntimeError, BddkStorageError) as exc:
                logger.warning("Document page lookup failed", extra={"error_type": type(exc).__name__})
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
                getattr(deps, "telemetry_pool", None),
                tool_name="get_bddk_document",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "not_found"},
            )
            return tool_error(
                NOT_FOUND,
                f"Document {document_id} is not available in the local store. This MCP server is airlocked "
                "and does not fetch from live BDDK / mevzuat.gov.tr sources at runtime.",
                retryable=False,
                hint="If the document should be available, re-run the seed (`seed.py import`) or sync pipeline.",
            )

        found = deps.client.find_by_id(resolved_id)
        meta_title, meta_date, meta_number, meta_category, source_url = (
            (found.title, found.decision_date, found.decision_number, found.category, found.source_url or "")
            if found
            else (resolved_id, "", "", "", "")
        )

        alias_line = f"- Resolved from: `{document_id}` -> `{resolved_id}`\n" if resolved_id != document_id else ""

        last_page_num = page_num
        missing_page: int | None = None
        if max_pages > 1 and page_num < total_pages:
            final_page = min(page_num + max_pages - 1, total_pages)
            for next_page in range(page_num + 1, final_page + 1):
                found_page = await _lookup_page(resolved_id, next_page)
                if not found_page:
                    missing_page = next_page
                    logger.warning(
                        "Requested page %d is missing from local stores (total_pages=%d)",
                        next_page,
                        total_pages,
                    )
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
            except (RuntimeError, BddkStorageError) as exc:
                logger.debug("Extraction method lookup failed", extra={"error_type": type(exc).__name__})

        if len(page_contents) > 1:
            raw_content = "\n\n---\n\n".join(
                f"### Page {page}/{total_pages}\n\n{page_content}" for page, page_content in page_contents
            )
        else:
            raw_content = page_contents[0][1]

        formula_aware = _is_formula_aware(extraction_method)
        quality = assess_markdown_quality(raw_content, document_id=resolved_id)
        if formula_aware and quality.flags == ["formula_ref_without_latex_or_image"]:
            quality.label = "clean"
            quality.flags = []
            quality.warning = ""
        sanitized_pages = [
            DocumentPageContent(
                page_number=page,
                content=sanitize_markdown_for_context(page_content),
            )
            for page, page_content in page_contents
        ]
        if len(sanitized_pages) > 1:
            content = "\n\n---\n\n".join(
                f"### Page {page.page_number}/{total_pages}\n\n{page.content}" for page in sanitized_pages
            )
        else:
            content = sanitized_pages[0].content

        degraded = bool(extraction_method) and not formula_aware
        method_display = extraction_method or "unknown"
        if degraded:
            method_display = f"{method_display} (formula-unaware — equations/images may be missing)"

        quality_lines = ""
        quality_warning_block = ""
        warnings: list[str] = [UNTRUSTED_SOURCE_WARNING]
        if quality.label != "clean":
            flags = ", ".join(quality.flags) if quality.flags else "none"
            quality_lines = f"- Quality: {quality.label}\n- Quality flags: {flags}\n"
            quality_warning_block = f"⚠ Quality warning: {quality.warning}\n\n" if quality.warning else ""
            if quality.warning:
                warnings.append(quality.warning)

        large_document_warning_block = ""
        if total_pages >= _LARGE_DOCUMENT_PAGE_THRESHOLD:
            large_document_warning = _LARGE_DOCUMENT_WARNING_TEMPLATE.format(total_pages=total_pages)
            large_document_warning_block = f"⚠ {large_document_warning}\n\n"
            warnings.append(large_document_warning)

        page_gap_warning_block = ""
        if missing_page is not None:
            page_gap_warning = _PAGE_GAP_WARNING_TEMPLATE.format(
                missing_page=missing_page,
                first_page=page_num,
                last_page=last_page_num,
                total_pages=total_pages,
            )
            page_gap_warning_block = "⚠ " + page_gap_warning + "\n\n"
            warnings.append(page_gap_warning)

        degraded_warning_block = f"⚠ {_DEGRADED_WARNING}\n\n" if degraded else ""
        if degraded:
            warnings.append(_DEGRADED_WARNING)
        page_display = (
            f"{page_num}/{total_pages}" if last_page_num == page_num else f"{page_num}-{last_page_num}/{total_pages}"
        )

        header = (
            f"## {meta_title}\n- Document ID: {resolved_id}\n{alias_line}"
            f"- Decision Date: {meta_date or 'N/A'}\n- Decision Number: {meta_number or 'N/A'}\n"
            f"- Category: {meta_category or 'N/A'}\n- Source: {source_url or 'N/A'}\n"
            f"- Page: {page_display}\n- Extraction: {method_display}\n{quality_lines}---\n"
            "Use ONLY the text below. Do not add information not present in this document.\n\n"
            f"{large_document_warning_block}{page_gap_warning_block}"
            f"{quality_warning_block}{degraded_warning_block}"
        )

        served_via = next(iter(served_sources)) if len(served_sources) == 1 else "mixed"
        relevance_stats = {
            "page_number": page_num,
            "last_page_number": last_page_num,
            "total_pages": total_pages,
            "pages_returned": len(page_contents),
            "served_via": served_via,
        }
        if missing_page is not None:
            relevance_stats["missing_page"] = missing_page
        await record_tool_call_trace(
            getattr(deps, "telemetry_pool", None),
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
            relevance_stats=relevance_stats,
        )
        quality_metadata = QualityMetadata(
            label=quality.label,  # type: ignore[arg-type]
            flags=list(quality.flags),
            warning=quality.warning or None,
        )
        response = DocumentResponse(
            status="partial" if missing_page is not None else "ok",
            text=header + content,
            evidence=[
                EvidenceReference(
                    document_id=resolved_id,
                    title=meta_title,
                    source_url=source_url or None,
                    decision_date=meta_date or None,
                    decision_number=meta_number or None,
                    category=meta_category or None,
                    retrieval_source=served_via,
                    page_start=page_num,
                    page_end=last_page_num,
                    extraction_method=extraction_method or "unknown",
                    quality=quality_metadata,
                )
            ],
            warnings=list(dict.fromkeys(warnings)),
            requested_document_id=document_id,
            resolved_document_id=resolved_id,
            title=meta_title,
            decision_date=meta_date,
            decision_number=meta_number,
            category=meta_category,
            source_url=source_url,
            first_page=page_num,
            last_page=last_page_num,
            total_pages=total_pages,
            pages=sanitized_pages,
            extraction_method=extraction_method or "unknown",
            served_via=served_via,  # type: ignore[arg-type]
            quality=quality_metadata,
        )
        return structured_tool_result(response)

    @mcp.tool()
    @logged_tool(logger)
    async def get_document_history(
        document_id: DocumentId,
    ) -> DocumentHistoryToolResult:
        """
        Get version history for a BDDK document.

        Shows all previous versions with timestamps and content hashes.

        Args:
            document_id: The document ID (from search results)
        """
        store = deps.doc_store
        history = await store.get_document_history(document_id)

        if not history:
            output = f"No version history found for document {document_id}."
            return structured_tool_result(
                DocumentHistoryResponse(
                    status="no_results",
                    text=output,
                    document_id=document_id,
                )
            )

        lines = [f"**Version History for {document_id}** ({len(history)} version(s)):\n"]
        versions: list[DocumentVersionItem] = []
        evidence: list[EvidenceReference] = []
        for v in history:
            lines.append(
                f"  v{v['version']} — {v['synced_at']} (hash: {v['content_hash'][:12]}..., {v['content_length']} chars)"
            )
            version = DocumentVersionItem(
                version=int(v["version"]),
                synced_at=str(v["synced_at"]),
                content_hash=str(v["content_hash"]),
                content_length=int(v["content_length"]),
            )
            versions.append(version)
            evidence.append(
                EvidenceReference(
                    document_id=document_id,
                    retrieval_source="version_store",
                    content_hash=version.content_hash,
                )
            )

        return structured_tool_result(
            DocumentHistoryResponse(
                status="ok",
                text="\n".join(lines),
                evidence=evidence,
                document_id=document_id,
                versions=versions,
            )
        )

    if not include_operator:
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
            except Exception as exc:
                logger.warning("Vector store statistics unavailable", extra={"error_type": type(exc).__name__})
                lines.append("  pgvector: unavailable")
        else:
            lines.append("  pgvector: unavailable (not initialized)")

        try:
            st = await deps.doc_store.stats()
            lines.append(
                f"\n**PostgreSQL (Document Store):**\n  Documents: {st.total_documents}\n  Size: {st.total_size_mb} MB"
            )
        except (RuntimeError, BddkStorageError) as exc:
            logger.warning("Document store statistics unavailable", extra={"error_type": type(exc).__name__})
            lines.append("  PostgreSQL: unavailable")

        return "\n".join(lines)
