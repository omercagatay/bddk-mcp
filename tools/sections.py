"""Section-level document retrieval and search tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from legal_ref import parse_legal_refs

if TYPE_CHECKING:
    from deps import Dependencies
    from doc_store import StoredDocumentSection


def _format_section(section: StoredDocumentSection, *, include_content: bool = True) -> str:
    heading = f" — {section.heading}" if section.heading else ""
    lines = [
        f"### {section.doc_id} — {section.section_type} {section.section_ref}{heading}",
        f"- Document ID: {section.doc_id}",
        f"- Section: {section.section_type} {section.section_ref}",
        f"- Character range: {section.start_char}-{section.end_char}",
    ]
    if section.page_start is not None:
        page_end = section.page_end if section.page_end is not None else section.page_start
        lines.append(f"- Pages: {section.page_start}-{page_end}")
    if include_content:
        lines.extend(["", section.content])
    return "\n".join(lines)


def _section_preview(section: StoredDocumentSection, *, length: int = 220) -> str:
    return " ".join(section.content.split())[:length]


def _section_key(section: StoredDocumentSection) -> tuple[str, str, str, str]:
    return (section.doc_id, section.section_type, section.section_ref, section.content_hash)


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value.lower() if value else None


def register(mcp, deps: Dependencies) -> None:
    """Register section tools on the given MCP instance."""

    @mcp.tool()
    async def get_document_section(
        document_id: str,
        section_type: str | None = None,
        section_ref: str | None = None,
        heading: str | None = None,
    ) -> str:
        """
        Retrieve exact structural sections from a stored BDDK document.

        Use when the user asks for a specific article/principle/paragraph such as
        `943 İlke 5` or `mevzuat_22599 Madde 9`.

        Args:
            document_id: Stored document ID, e.g. `943` or `mevzuat_22599`
            section_type: Optional exact section type, e.g. madde, ilke, paragraf, ek
            section_ref: Optional exact section reference, e.g. 9 or 5
            heading: Optional heading substring filter
        """
        sections = await deps.doc_store.get_document_section(
            document_id,
            section_type=_normalize_optional(section_type),
            section_ref=_normalize_optional(section_ref),
            heading=heading,
        )
        if not sections:
            query = " ".join(
                part for part in (document_id, section_type or "", section_ref or "", heading or "") if part
            )
            return (
                f"No section found for document {document_id} with the requested filters.\n"
                f"Try search_document_sections with query: {query or document_id}"
            )

        if len(sections) == 1:
            return _format_section(sections[0])

        lines = [f"Multiple sections matched ({len(sections)}). Narrow by section_type, section_ref, or heading:\n"]
        for section in sections[:10]:
            heading_text = f" — {section.heading}" if section.heading else ""
            lines.append(
                f"- {section.doc_id} {section.section_type} {section.section_ref}{heading_text} "
                f"(start_char={section.start_char}, hash={section.content_hash[:12]})"
            )
            lines.append(f"  {_section_preview(section)}")
        if len(sections) > 10:
            lines.append(f"... {len(sections) - 10} more match(es) omitted.")
        return "\n".join(lines)

    @mcp.tool()
    async def search_document_sections(
        query: str,
        document_id: str | None = None,
        section_type: str | None = None,
        limit: int = 10,
    ) -> str:
        """
        Search section-level content in stored BDDK documents.

        Parses exact legal references from the query and uses them as filters when
        explicit arguments are not provided.

        Args:
            query: Turkish legal query, e.g. `943 İlke 5 model validasyonu`
            document_id: Optional document ID filter
            section_type: Optional section type filter
            limit: Maximum number of section results
        """
        refs = parse_legal_refs(query)
        inferred_doc_id = document_id or (refs.document_ids[0] if refs.document_ids else None)
        inferred_section_type = section_type or (refs.sections[0][0] if refs.sections else None)
        inferred_section_ref = refs.sections[0][1] if refs.sections else None

        exact_hits = []
        if inferred_doc_id and inferred_section_type and inferred_section_ref:
            exact_hits = await deps.doc_store.get_document_section(
                inferred_doc_id,
                section_type=_normalize_optional(inferred_section_type),
                section_ref=_normalize_optional(inferred_section_ref),
            )
        hits = await deps.doc_store.search_document_sections(
            query,
            document_id=inferred_doc_id,
            section_type=_normalize_optional(inferred_section_type),
            limit=limit,
        )
        if exact_hits:
            seen = {_section_key(section) for section in exact_hits}
            hits = exact_hits + [section for section in hits if _section_key(section) not in seen]
        if not hits:
            return (
                f"NO RESULTS: No document sections found matching '{query}'.\n"
                "Try a broader query, remove the document_id/section_type filter, or retrieve the full document."
            )

        lines = [f"Found {len(hits)} section result(s) for '{query}':\n"]
        for hit in hits:
            heading = f" — {hit.heading}" if hit.heading else ""
            lines.append(f"**{hit.doc_id} — {hit.section_type} {hit.section_ref}{heading}**")
            lines.append(f"  Document ID: {hit.doc_id}")
            lines.append(f"  Section: {hit.section_type} {hit.section_ref}")
            lines.append(f"  Character range: {hit.start_char}-{hit.end_char}")
            preview = _section_preview(hit)
            if preview:
                lines.append(f"  ...{preview}...")
            lines.append("")
        return "\n".join(lines)
