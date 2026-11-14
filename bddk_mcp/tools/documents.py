"""Document retrieval and management tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bddk_mcp.core.config import PAGE_SIZE
from bddk_mcp.core.exceptions import BddkStorageError
from bddk_mcp.quality.markdown_quality import (
    FORMULA_EXTRACTION_WARNING as _DEGRADED_WARNING,
)
from bddk_mcp.quality.markdown_quality import is_formula_aware as _is_formula_aware
from bddk_mcp.quality.markdown_quality import mentions_formula as _mentions_formula
from bddk_mcp.quality.markdown_quality import quality_assessment_from_metadata, unsafe_verbatim_reason
from bddk_mcp.store.legal_ref import document_id_candidates
from bddk_mcp.tools.contract_types import (
    MAX_DOCUMENT_PAGES_PER_RESPONSE,
    DocumentId,
    DocumentMaxPages,
    DocumentPageNumber,
)
from bddk_mcp.tools.errors import INVALID_INPUT, NOT_FOUND, tool_error
from bddk_mcp.tools.structured_outputs import (
    LEGAL_APPLICABILITY_WARNING,
    UNTRUSTED_SOURCE_WARNING,
    DocumentCurrentBody,
    DocumentHistoryResponse,
    DocumentHistoryToolResult,
    DocumentPageContent,
    DocumentResponse,
    DocumentToolResult,
    DocumentVersionItem,
    EvidenceReference,
    QualityMetadata,
    decision_identifier_label,
    decision_identifier_text,
    document_status_notice,
    source_date_text,
    structured_tool_result,
    utc_date,
)
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import DocumentPage

logger = logging.getLogger(__name__)

_LARGE_DOCUMENT_PAGE_THRESHOLD = 20
_MAX_PAGES_PER_RESPONSE = MAX_DOCUMENT_PAGES_PER_RESPONSE
_HISTORY_NOTICE = (
    "This is the local text-change log of this corpus file (bodies replaced by bddk-mcp sync); it is not the "
    "legal amendment history of the instrument. Check mevzuat.gov.tr / Resmî Gazete for amendments."
)
_PAGE_GAP_WARNING_TEMPLATE = (
    "Sayfa {missing_page} local store'dan alınamadı; yalnızca {first_page}-{last_page} arası döndürüldü. "
    "Belge toplam {total_pages} sayfa olarak kayıtlı — eksik sayfa, store tutarsızlığına işaret edebilir."
)
_LARGE_DOCUMENT_WARNING_TEMPLATE = (
    "Bu belge {total_pages} sayfa. Tam belgeyi sayfa sayfa çekmek context'i hızla şişirebilir. "
    "Hedefli retrieval için önce search_document_sections veya get_document_section kullan; "
    "yalnızca gerekli sayfalar için get_bddk_document içinde page_number ve max_pages parametrelerini kullan."
)


def _page_cut_warning(page: int, content: str, *, total_pages: int, returned_pages: set[int]) -> str | None:
    """Name a clear structural cut on a page that is not the last one.

    An odd number of $$ delimiters says only that a formula crosses one of the page's two boundaries,
    not which one, so that advice names both neighbours. A last line starting with "|" and no final
    newline is an end-of-page cut, so that advice names the next page only. A neighbour already in
    this response is not asked for again.
    """
    advice = []
    if content.count("$$") % 2 == 1:
        adjacent = [n for n in (page - 1, page + 1) if 1 <= n <= total_pages]
        missing = [n for n in adjacent if n not in returned_pages]
        text = (
            f"Page {page} has an odd number of $$ formula delimiters: a formula is cut at the start or the end of "
            "this page, and the delimiter count cannot tell which."
        )
        if missing:
            text += (
                f" Retrieve page(s) {' and '.join(map(str, missing))} as well before quoting a formula at either "
                "edge of this page."
            )
        else:
            text += " Read the adjacent page(s) in this response before quoting a formula at either edge of it."
        advice.append(text)
    if not content.endswith("\n") and content.rsplit("\n", 1)[-1].startswith("|"):
        text = f"Page {page} ends inside a table row that continues on the next page."
        if page + 1 not in returned_pages:
            text += f" Retrieve page {page + 1} as well before quoting that row."
        else:
            text += f" Its continuation is at the start of page {page + 1} in this response."
        advice.append(text)
    if not advice:
        return None
    return " ".join(advice) + " The page boundary is a fixed character window, not a source boundary."


def register(
    mcp,
    deps: Dependencies,
) -> None:
    """Register document tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def get_bddk_document(
        document_id: DocumentId,
        page_number: DocumentPageNumber = 1,
        max_pages: DocumentMaxPages = 1,
    ) -> DocumentToolResult:
        """
        Retrieve a BDDK decision document as Markdown.

        Airlocked: serves only from the local corpus file. If the document is not
        present locally, returns a NOT_FOUND error rather than live-fetching from
        mevzuat.gov.tr / BDDK. A page_number beyond the end of a stored document
        is INVALID_INPUT and the error gives the document's page count.

        A bare numeric ID (e.g. "21192") is also tried as `mevzuat_<id>` and
        `bddk_<id>` so callers don't need to know the catalog's prefix
        convention. The resolved ID is shown in the header.

        Args:
            document_id: The numeric document ID (from search results)
            page_number: First page of the markdown output. Pages are fixed windows of
                BDDK_PAGE_SIZE characters (5000 by default) that can cut a word, table row or formula.
            max_pages: Consecutive pages to return (1-5)
        """
        if isinstance(max_pages, bool) or not 1 <= max_pages <= _MAX_PAGES_PER_RESPONSE:
            return tool_error(INVALID_INPUT, "max_pages must be between 1 and 5.", retryable=False)
        candidates = document_id_candidates(document_id)

        async def _lookup_page(cand: str, page: int) -> DocumentPage | None:
            stored = await deps.doc_store.get_document_page(cand, page)
            if stored and stored.markdown_content:
                return stored
            return None

        resolved_id: str | None = None
        first_found: DocumentPage | None = None
        page_num = 0
        total_pages = 0
        page_contents: list[tuple[int, str]] = []
        extraction_method = ""
        body_hash = ""
        downloaded_at: float | None = None

        for cand in candidates:
            found_page = await _lookup_page(cand, page_number)
            if found_page and not 1 <= found_page.page_number <= found_page.total_pages:
                # The store answers an out-of-range page with a sentinel page; the document exists.
                return tool_error(
                    INVALID_INPUT,
                    f"page_number {page_number} is out of range; document {cand} has {found_page.total_pages} page(s).",
                    retryable=False,
                )
            if found_page:
                resolved_id = cand
                first_found = found_page
                page_num, total_pages = found_page.page_number, found_page.total_pages
                extraction_method = found_page.extraction_method or ""
                body_hash, downloaded_at = found_page.content_hash, found_page.downloaded_at
                page_contents.append((page_num, found_page.markdown_content))
                break

        # A bare number may be a Kurul decision number rather than a document id (8970 is the decision
        # number of document 779); the alias order of document_id_candidates is not changed by this.
        bare_number = document_id.strip()
        decision_holders = (
            [
                entry
                for entry in deps.client.get_cache_items()
                if (entry.decision_number or "").strip() == bare_number and entry.document_id != resolved_id
            ]
            if bare_number.isdigit()
            else []
        )
        holders_text = "; ".join(f'document {entry.document_id} "{entry.title}"' for entry in decision_holders)

        if resolved_id is None:
            if decision_holders:
                return tool_error(
                    NOT_FOUND,
                    f"Document {document_id} is not available in the local store: {document_id} is not a stored "
                    f"document id, but it is the Kurul decision number of {holders_text}. This MCP server is "
                    "airlocked and does not fetch from live BDDK / mevzuat.gov.tr sources at runtime.",
                    retryable=False,
                    hint=f'Call get_bddk_document(document_id="{decision_holders[0].document_id}"), or use '
                    f'search_bddk_regulations(keywords="{bare_number}") to see every entry with this number.',
                )
            return tool_error(
                NOT_FOUND,
                f"Document {document_id} is not available in the local store. This MCP server is airlocked "
                "and does not fetch from live BDDK / mevzuat.gov.tr sources at runtime.",
                retryable=False,
                hint="If the document should be available, rebuild the corpus (`bddk-mcp build`) or run `bddk-mcp sync`.",
            )

        found = deps.client.find_by_id(resolved_id)
        # Without a catalog row, the stored page still carries the document's title and category, so the
        # MÜLGA/TASLAK status mark agrees with the section tools, which read it from the documents table.
        meta_title, meta_date, meta_number, meta_category, source_url = (
            (found.title, found.decision_date, found.decision_number, found.category, found.source_url or "")
            if found
            else (
                (first_found.title if first_found else "") or resolved_id,
                "",
                "",
                first_found.category if first_found else "",
                "",
            )
        )

        alias_line = f"- Resolved from: `{document_id}` -> `{resolved_id}`\n" if resolved_id != document_id else ""
        if decision_holders:
            alias_line += (
                f"- Note: {bare_number} is also the Kurul decision number of {holders_text}, a different "
                f'document (get_bddk_document(document_id="{decision_holders[0].document_id}")).\n'
            )

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
                last_page_num, total_pages = found_page.page_number, found_page.total_pages
                page_contents.append((last_page_num, found_page.markdown_content))
                if found_page.extraction_method and not extraction_method:
                    extraction_method = found_page.extraction_method

        # Page cuts can split tables/formulas, or hide a failure elsewhere in the body.
        try:
            quality = await deps.doc_store.get_document_quality(resolved_id)
        except (RuntimeError, BddkStorageError) as exc:
            logger.warning("Document quality lookup failed", extra={"error_type": type(exc).__name__})
            quality = None
        if quality is None:
            quality = quality_assessment_from_metadata(resolved_id, "unknown", ["document_quality_unavailable"])
        formula_aware = _is_formula_aware(extraction_method)
        # Verbatim contract: returned document text is the exact stored characters.
        # If returning them unchanged would leak an unsafe extraction blob, refuse
        # the document instead of silently rewriting (sanitizing) its text.
        unsafe_reason = next(
            (reason for _page, page_content in page_contents if (reason := unsafe_verbatim_reason(page_content))),
            None,
        )
        if unsafe_reason is not None:
            return tool_error(
                "VERBATIM_UNAVAILABLE",
                f"Document {resolved_id} cannot be returned verbatim: its stored text contains an unsafe "
                f"extraction artifact ({unsafe_reason}). This server does not rewrite, sanitize, or summarize "
                "document text; repair and re-publish the source instead.",
                retryable=False,
                hint="Use the source document (source_url), or repair/re-extract the document and rebuild or sync the corpus.",
            )
        sanitized_pages = [
            DocumentPageContent(page_number=page, content=page_content) for page, page_content in page_contents
        ]
        if len(sanitized_pages) > 1:
            content = "\n\n---\n\n".join(
                f"### Page {page.page_number}/{total_pages}\n\n{page.content}" for page in sanitized_pages
            )
        else:
            content = sanitized_pages[0].content

        returned_text = "\n".join(page_content for _page, page_content in page_contents)
        degraded = bool(extraction_method) and not formula_aware and _mentions_formula(returned_text)
        method_display = extraction_method or "unknown"
        if degraded:
            method_display = f"{method_display} (formula-unaware — equations/images may be missing)"

        quality_lines = ""
        quality_warning_block = ""
        warnings: list[str] = [UNTRUSTED_SOURCE_WARNING, LEGAL_APPLICABILITY_WARNING]
        status_notice = document_status_notice(meta_category)
        status_line = f"- Document status: {status_notice}\n" if status_notice else ""
        if status_notice:
            warnings.append(status_notice)
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

        separator_note = (
            " In this multi-page output the '---' and '### Page N/T' lines are tool separators between pages, "
            "not source text; never quote them."
            if len(page_contents) > 1
            else ""
        )
        page_cut_warnings = [
            warning
            for page, page_content in page_contents
            if page < total_pages
            and (
                warning := _page_cut_warning(
                    page,
                    page_content,
                    total_pages=total_pages,
                    returned_pages={returned for returned, _content in page_contents},
                )
            )
        ]
        page_cut_warning_block = "".join(f"⚠ {warning}\n\n" for warning in page_cut_warnings)
        warnings.extend(page_cut_warnings)

        degraded_warning_block = f"⚠ {_DEGRADED_WARNING}\n\n" if degraded else ""
        if degraded:
            warnings.append(_DEGRADED_WARNING)
        page_display = (
            f"{page_num}/{total_pages}" if last_page_num == page_num else f"{page_num}-{last_page_num}/{total_pages}"
        )

        decision = decision_identifier_text(meta_date, meta_number)
        decision_line = f"- {decision_identifier_label(resolved_id)}: {decision}\n" if decision else ""
        header = (
            f"## {meta_title}\n- Document ID: {resolved_id}\n{alias_line}"
            f"{decision_line}"
            f"- Category: {meta_category or 'N/A'}\n{status_line}- Source: {source_url or 'N/A'}\n"
            f"- Text obtained from the source on: {source_date_text(downloaded_at)}\n"
            f"- Page: {page_display}\n- Extraction: {method_display}\n{quality_lines}---\n"
            "The text below is the exact stored document content; no sanitization, paraphrase or "
            "summary has been applied. Quote it verbatim, character for character, and do not add information "
            "not present in this document. Verbatim equality with the stored extraction does not by itself prove "
            "equality with the official source; use source_url to verify. "
            f"Pages are fixed windows of {PAGE_SIZE:,} characters of the stored text, so a page can start or end "
            "inside a word, a table row or a formula."
            f"{separator_note}\n\n"
            f"{large_document_warning_block}{page_gap_warning_block}{page_cut_warning_block}"
            f"{quality_warning_block}{degraded_warning_block}"
        )

        served_via = "document_store"
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
                    content_hash=body_hash or None,
                    source_downloaded_on=utc_date(downloaded_at),
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
        List the local text-change log of a stored document in this corpus file.

        This is NOT the legal amendment history of the instrument: it lists only
        earlier bodies that `bddk-mcp sync` (or a rebuild) replaced in this corpus
        file, newest first, each with the UTC time it was replaced and its content
        hash. The current body is reported separately (content hash, source
        download date, extraction method) and is never one of the versions. Check
        mevzuat.gov.tr / Resmî Gazete for amendments. A bare numeric ID is also
        tried as `mevzuat_<id>` and `bddk_<id>`; an ID that is not a stored
        document is a NOT_FOUND error.

        Args:
            document_id: The document ID (from search results)
        """
        store = deps.doc_store
        resolved_id: str | None = None
        current: dict | None = None
        for candidate in document_id_candidates(document_id):
            current = await store.get_current_body(candidate)
            if current is not None:
                resolved_id = candidate
                break
        if resolved_id is None or current is None:
            return tool_error(
                NOT_FOUND,
                f"Document {document_id} is not available in the local store. This MCP server is airlocked "
                "and does not fetch from live BDDK / mevzuat.gov.tr sources at runtime.",
                retryable=False,
                hint="Check the document ID with search_bddk_regulations or search_document_store.",
            )
        history = await store.get_document_history(resolved_id)

        current_body = DocumentCurrentBody(
            content_hash=str(current["content_hash"]),
            content_length=int(current["content_length"]),
            source_downloaded_on=utc_date(current["downloaded_at"]),
            extraction_method=str(current["extraction_method"] or "unknown"),
        )
        alias_line = f"Resolved from: `{document_id}` -> `{resolved_id}`\n" if resolved_id != document_id else ""
        current_line = (
            f"Current stored body (not a version): hash {current_body.content_hash[:12]}..., "
            f"{current_body.content_length} chars, text obtained from the source on: "
            f"{source_date_text(current['downloaded_at'])}, extraction method: {current_body.extraction_method}"
        )

        if not history:
            output = (
                f"{alias_line}{current_line}\n\n"
                f"No version history found for document {resolved_id}. This corpus file records no replaced "
                "body for it; that says nothing about whether the instrument was amended."
            )
            return structured_tool_result(
                DocumentHistoryResponse(
                    status="no_results",
                    text=output,
                    document_id=document_id,
                    resolved_document_id=resolved_id,
                    current_body=current_body,
                ),
                trusted_notice=_HISTORY_NOTICE,
            )

        lines = [
            f"{alias_line}{current_line}\n",
            f"**Replaced bodies of {resolved_id}** ({len(history)} version(s), newest first):\n",
        ]
        versions: list[DocumentVersionItem] = []
        evidence: list[EvidenceReference] = []
        for v in history:
            lines.append(
                f"  v{v['version']} — replaced on {v['synced_at']} "
                f"(hash: {v['content_hash'][:12]}..., {v['content_length']} chars)"
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
                    document_id=resolved_id,
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
                resolved_document_id=resolved_id,
                current_body=current_body,
            ),
            trusted_notice=_HISTORY_NOTICE,
        )
