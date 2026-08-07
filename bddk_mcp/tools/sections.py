"""Section-level document retrieval and search tools."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from bddk_mcp.observability.telemetry import elapsed_ms, record_tool_call_trace, unique_doc_ids
from bddk_mcp.regulatory.graph_queries import one_hop_section_refs
from bddk_mcp.store.legal_ref import parse_legal_refs

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocumentSection


_LOOSE_SECTION_SEARCH_STOPWORDS = {
    "acaba",
    "altına",
    "aşağıdaki",
    "aşağıdakilerden",
    "bir",
    "biri",
    "değildir",
    "göre",
    "hangi",
    "hangisi",
    "hakkında",
    "için",
    "ilişkin",
    "olan",
    "olarak",
    "ve",
    "veya",
    "yanlıştır",
    "yer",
}


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


def _normalize_optional(value: str | int | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value.lower() if value else None


def _loose_search_terms(query: str, *, limit: int = 16) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in query.strip().split():
        term = re.sub(r"[^\w]+", "", raw_term, flags=re.UNICODE).casefold()
        if len(term) < 3 or term in _LOOSE_SECTION_SEARCH_STOPWORDS or term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= limit:
            break
    return terms


def _loose_section_score(section: StoredDocumentSection, terms: list[str]) -> int:
    text = f"{section.heading} {section.content}".casefold()
    return sum(1 for term in terms if term in text)


async def _search_sections_loose(
    deps: Dependencies,
    query: str,
    *,
    document_id: str | None,
    section_type: str | None,
    limit: int,
) -> list[StoredDocumentSection]:
    terms = _loose_search_terms(query)
    if not terms:
        return []

    merged: dict[tuple[str, str, str, str], StoredDocumentSection] = {}
    for term in terms:
        term_hits = await deps.doc_store.search_document_sections(
            term,
            document_id=document_id,
            section_type=section_type,
            limit=limit,
        )
        for section in term_hits:
            # Per-term ranks come from different tsqueries and are not
            # comparable; surfacing them would mislead rank-gating clients.
            merged[_section_key(section)] = section.model_copy(update={"rank": None})

    ranked = sorted(
        merged.values(),
        key=lambda section: (-_loose_section_score(section, terms), section.doc_id, section.start_char),
    )
    return ranked[:limit]


def register(mcp, deps: Dependencies) -> None:
    """Register section tools on the given MCP instance."""

    @mcp.tool()
    async def get_document_section(
        document_id: str,
        section_type: str | None = None,
        section_ref: str | int | None = None,
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
        start = time.perf_counter()
        args = {
            "document_id": document_id,
            "section_type": section_type,
            "section_ref": section_ref,
            "heading": heading,
        }
        sections = await deps.doc_store.get_document_section(
            document_id,
            section_type=_normalize_optional(section_type),
            section_ref=_normalize_optional(section_ref),
            heading=heading,
        )
        if not sections:
            query = " ".join(
                str(part) for part in (document_id, section_type or "", section_ref or "", heading or "") if part
            )
            await record_tool_call_trace(
                getattr(deps, "pool", None),
                tool_name="get_document_section",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "not_found"},
            )
            return (
                f"No section found for document {document_id} with the requested filters.\n"
                f"Try search_document_sections with query: {query or document_id}"
            )

        if len(sections) == 1:
            await record_tool_call_trace(
                getattr(deps, "pool", None),
                tool_name="get_document_section",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=1,
                doc_ids=[sections[0].doc_id],
                relevance_stats={"status": "exact_match"},
            )
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
        await record_tool_call_trace(
            getattr(deps, "pool", None),
            tool_name="get_document_section",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(sections),
            doc_ids=unique_doc_ids([section.doc_id for section in sections]),
            relevance_stats={"status": "disambiguation"},
        )
        return "\n".join(lines)

    @mcp.tool()
    async def search_document_sections(
        query: str,
        document_id: str | None = None,
        section_type: str | None = None,
        limit: int = 10,
        expand_references: bool = False,
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
            expand_references: Doğrulanmış çapraz referans kenarlarını 1 adım
                takip ederek ilişkili bölüm etiketleri ekle (içerik dahil edilmez)

        Each FTS hit includes a `Match rank` (length-normalized ts_rank_cd):
        higher is better; ranks are comparable within one query's results and
        can be used to gate low-confidence retrieval.
        """
        start = time.perf_counter()
        args = {
            "query": query,
            "document_id": document_id,
            "section_type": section_type,
            "limit": limit,
            "expand_references": expand_references,
        }
        refs = parse_legal_refs(query)
        inferred_doc_id = document_id or (refs.document_ids[0] if refs.document_ids else None)
        inferred_section_type = section_type or (refs.sections[0][0] if refs.sections else None)
        inferred_section_ref = refs.sections[0][1] if refs.sections else None
        exact_ref_detected = bool(inferred_doc_id and inferred_section_type and inferred_section_ref)

        exact_hits = []
        if exact_ref_detected:
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
        loose_fallback_used = False
        if not hits and not exact_hits:
            hits = await _search_sections_loose(
                deps,
                query,
                document_id=inferred_doc_id,
                section_type=_normalize_optional(inferred_section_type),
                limit=limit,
            )
            loose_fallback_used = bool(hits)
        if exact_hits:
            # Exact-ref lookups carry no FTS rank; inherit it from the FTS
            # duplicate being deduplicated away so the top hit is not the
            # only unscored result.
            fts_ranks = {_section_key(s): s.rank for s in hits if s.rank is not None}
            exact_hits = [
                s.model_copy(update={"rank": fts_ranks.get(_section_key(s))}) if s.rank is None else s
                for s in exact_hits
            ]
            seen = {_section_key(section) for section in exact_hits}
            hits = exact_hits + [section for section in hits if _section_key(section) not in seen]
        if not hits:
            await record_tool_call_trace(
                getattr(deps, "pool", None),
                tool_name="search_document_sections",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"exact_ref_detected": exact_ref_detected, "status": "no_results"},
            )
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
            if hit.rank is not None:
                # "relative" guards against conflation with the percent-scale
                # relevance gate in the server instructions (store search).
                lines.append(f"  Match rank: {hit.rank:.4f} (relative, FTS)")
            preview = _section_preview(hit)
            if preview:
                lines.append(f"  ...{preview}...")
            lines.append("")
        if expand_references and deps.pool is not None:
            expanded_blocks: list[str] = []
            for hit in hits:
                try:
                    neighbors = await one_hop_section_refs(
                        deps.pool,
                        doc_id=hit.doc_id,
                        section_type=hit.section_type,
                        section_ref=hit.section_ref,
                        limit=3,
                    )
                except Exception:  # noqa: BLE001 — expansion must never break search
                    neighbors = []
                if not neighbors:
                    continue
                block = [
                    f"#### İlişkili bölümler (doğrulanmış kenarlar) — {hit.doc_id} {hit.section_type} {hit.section_ref}"
                ]
                block.extend(
                    f"- {n['doc_id']} — {n['section_type']} {n['section_ref']} (kenar: {n['relation_type']})"
                    for n in neighbors
                )
                expanded_blocks.append("\n".join(block))
            if expanded_blocks:
                lines.append("\n\n".join(expanded_blocks))
        await record_tool_call_trace(
            getattr(deps, "pool", None),
            tool_name="search_document_sections",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(hits),
            doc_ids=unique_doc_ids([hit.doc_id for hit in hits]),
            relevance_stats={"exact_ref_detected": exact_ref_detected, "loose_fallback": loose_fallback_used},
        )
        return "\n".join(lines)
