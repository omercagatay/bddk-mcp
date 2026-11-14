"""Search tools: search_bddk_regulations, search_bddk_institutions,
search_bddk_announcements, and search_document_store.

Uses an OrderedDict-based LRU cache for O(1) eviction instead of the
O(n) min() scan in server.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING

from bddk_mcp.core.config import ANNOUNCEMENT_CATEGORY_IDS, SEARCH_CACHE_MAX, SEARCH_CACHE_TTL
from bddk_mcp.core.exceptions import BddkUpstreamError, BddkUpstreamUnreachableError
from bddk_mcp.core.models import BddkSearchRequest
from bddk_mcp.ingest.client import _turkish_lower, catalog_keyword_tokens
from bddk_mcp.ingest.data_sources import fetch_announcements, fetch_institutions_with_status
from bddk_mcp.quality.markdown_quality import (
    QualityAssessment,
    assess_markdown_quality,
    quality_assessment_from_metadata,
)
from bddk_mcp.store.vector_store import SemanticSearchReadinessError, SemanticSearchUnavailableError
from bddk_mcp.tools.contract_types import (
    ActiveOnly,
    AnnouncementCategory,
    DateFrom,
    DateTo,
    InstitutionType,
    OptionalRegulationCategory,
    OptionalSearchKeywords,
    PageNumber,
    PageSize,
    RegulationKeywords,
    ResultLimit,
    SemanticQuery,
    normalize_announcement_category,
    normalize_institution_type,
    validate_catalog_query,
    validate_date_order,
)
from bddk_mcp.tools.errors import UPSTREAM_FETCH_FAILED, tool_error
from bddk_mcp.tools.structured_outputs import (
    LEGAL_APPLICABILITY_WARNING,
    STRUCTURELESS_CITATION_HINT,
    STRUCTURELESS_GOVDE_LABEL,
    UNTRUSTED_SOURCE_WARNING,
    DocumentSearchItem,
    DocumentSearchResponse,
    DocumentSearchToolResult,
    EvidenceReference,
    QualityMetadata,
    RegulationCatalogItem,
    RegulationCatalogResponse,
    RegulationCatalogToolResult,
    decision_identifier_text,
    document_status_notice,
    frame_untrusted_source,
    structured_tool_result,
)
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)

_CATALOG_SUMMARY_WARNING = (
    "Catalog summary/excerpt is BDDK catalog metadata, not extracted document text. It is neither a "
    "verbatim provision nor a basis for a quotation; retrieve the exact document or section instead."
)
_SNIPPET_TRUNCATION_WARNING = (
    "Search snippets are exact leading excerpts of one stored chunk and may be truncated; they are "
    "discovery leads, not the complete provision text."
)


class _LRUCache:
    """In-memory LRU cache with TTL.

    Uses OrderedDict for O(1) eviction of the least-recently-used entry.
    Each value is stored as (timestamp, payload).
    """

    def __init__(self, max_size: int, ttl: int) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()

    def get(self, key: str) -> object | None:
        """Return the cached value if present and not expired, else None.

        On cache hit the entry is moved to the end (most-recently-used).
        """
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if (time.time() - ts) >= self._ttl:
            del self._data[key]
            return None
        # Move to end — most recently used
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: object) -> None:
        """Store a value. Evicts the least-recently-used entry when full."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (time.time(), value)
        while len(self._data) > self._max_size:
            # popitem(last=False) removes the front — oldest/least-recently-used
            self._data.popitem(last=False)


# Module-level cache shared across all invocations. Keys of corpus-backed
# searches carry the corpus change token, so a response computed before a
# corpus write is never served after it.
_search_cache: _LRUCache = _LRUCache(max_size=SEARCH_CACHE_MAX, ttl=SEARCH_CACHE_TTL)


async def _corpus_cache_token(deps: Dependencies) -> str:
    """Return the corpus change token as a cache-key part ("none" without a corpus)."""
    if deps.db is None:
        return "none"
    epoch, data_version = await asyncio.to_thread(deps.db.change_token)
    return f"{epoch}.{data_version}"


_MATCH_TYPES = frozenset({"hybrid", "vector", "fts", "fts_exact"})
# A keyword (FTS AND) match on a query that names a legal reference: there is no
# semantic score to show, and the match does not confirm the provision.
_FTS_EXACT_LABEL = "[keyword match on the legal reference in the query; no semantic score]"


def _match_strength(relevance: float) -> str:
    if relevance >= 0.70:
        return "strong"
    if relevance >= 0.50:
        return "moderate"
    return "weak"


def _quality_metadata(quality: QualityAssessment) -> QualityMetadata:
    return QualityMetadata(
        label=quality.label,  # type: ignore[arg-type]
        flags=list(quality.flags),
        warning=quality.warning or None,
    )


def _quality_result_lines(quality: QualityAssessment, *, indent: str = "  ") -> list[str]:
    """Return consistent user-visible quality metadata for a non-clean hit."""
    if quality.label not in {"warning", "fail"}:
        return []
    flags = ", ".join(quality.flags) if quality.flags else "none"
    lines = [f"{indent}Quality: {quality.label}", f"{indent}Quality flags: {flags}"]
    if quality.warning:
        lines.append(f"{indent}⚠ Quality warning: {quality.warning}")
    return lines


def _search_hit_quality(hit: dict) -> QualityAssessment:
    return quality_assessment_from_metadata(
        str(hit.get("doc_id") or hit.get("document_id") or ""),
        hit.get("quality_label"),
        list(hit.get("quality_flags") or []),
    )


_MAX_ANNOUNCEMENTS_SHOWN = 20


def _newest_first(announcements: list[dict]) -> list[dict]:
    """Sort announcements by their DD.MM.YYYY date, newest first; undated or unparsable ones last (stable)."""
    dated: list[tuple[datetime, int, dict]] = []
    undated: list[dict] = []
    for index, announcement in enumerate(announcements):
        try:
            parsed = datetime.strptime(str(announcement.get("date") or "").strip(), "%d.%m.%Y")
        except ValueError:
            undated.append(announcement)
            continue
        dated.append((parsed, index, announcement))
    dated.sort(key=lambda item: (-item[0].toordinal(), item[1]))
    return [announcement for _, _, announcement in dated] + undated


def register(mcp, deps: Dependencies) -> None:  # type: ignore[type-arg]
    """Register the four search tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def search_bddk_regulations(
        keywords: RegulationKeywords = "",
        page: PageNumber = 1,
        page_size: PageSize = 10,
        category: OptionalRegulationCategory = None,
        date_from: DateFrom = None,
        date_to: DateTo = None,
    ) -> RegulationCatalogToolResult:
        """
        Search the local corpus catalog (the BDDK regulations served by this corpus file) by title,
        category, decision number, and date. It is not a live search of the BDDK website.

        Covers the BDDK regulatory document types held in this corpus: yönetmelik, tebliğ, genelge,
        rehber, kurul kararı, kanun, mülga düzenleme, etc.

        This is a TITLE/METADATA search only — it does NOT search document body content.
        Use this when you know words that appear in the regulation's name itself
        (e.g. "kredilerin sınıflandırılması yönetmeliği", "8970 sayılı kurul kararı").

        For terms that appear only inside document bodies — abbreviations like "TFRS 9",
        article references ("madde 5"), defined terms, calculation formulas — use
        search_document_store instead (semantic search over full text).

        Matching: function words (sayılı, hakkında, ilişkin, dair, göre, ile, ve, ...) are
        ignored; document-type words (yönetmelik, tebliğ, genelge, rehber, karar, kanun, taslak,
        in any inflection) are not required literally but rank documents of that type first;
        every other token must match the title (exactly or by a shared word stem) or the
        category/date/number. A number equal to a decision number ranks that decision first.
        keywords may be empty when category or a date range is given: the matching catalog
        entries are then listed newest decision date first. A date bound skips every entry
        without a decision date (most yönetmelik, kanun, rehber, genelge and tebliğ entries);
        the response states how many were skipped (undated_excluded).

        Args:
            keywords: Search terms in Turkish (e.g. "elektronik para", "banka lisansı");
                optional when category, date_from or date_to is given
            page: Page number, starting from 1
            page_size: Number of results per page (max 50)
            category: Optional category filter. Available categories:
                Yönetmelik, Genelge, Tebliğ, Rehber, Bilgi Sistemleri,
                Sermaye Yeterliliği, Faizsiz Bankacılık, Tekdüzen Hesap Planı,
                Kurul Kararı, Kanun, Banka Kartları,
                Finansal Kiralama ve Faktoring, BDDK Düzenlemesi,
                Düzenleme Taslağı, Mülga Düzenleme, Muhasebe Standardı
            date_from: Optional start date filter (DD.MM.YYYY); undated entries are not considered
            date_to: Optional end date filter (DD.MM.YYYY); undated entries are not considered
        """
        validate_catalog_query(catalog_keyword_tokens(keywords), category, date_from, date_to)
        validate_date_order(date_from, date_to)
        corpus_token = await _corpus_cache_token(deps)
        cache_key = f"regulations:{corpus_token}:{keywords}:{page}:{page_size}:{category}:{date_from}:{date_to}"
        cached = _search_cache.get(cache_key)
        if isinstance(cached, RegulationCatalogResponse):
            return structured_tool_result(cached)

        request = BddkSearchRequest(
            keywords=keywords,
            page=page,
            page_size=page_size,
            category=category,
            date_from=date_from,
            date_to=date_to,
        )
        result = await deps.client.search_decisions(request)
        undated_excluded = result.undated_excluded if (date_from or date_to) else 0
        undated_note = (
            f"{undated_excluded:,} catalog entries have no decision date and were not considered by the date filter."
            if undated_excluded
            else ""
        )

        if not result.decisions:
            catalog_scope = (
                f"this corpus catalog of {result.catalog_size:,} documents"
                if result.catalog_size is not None
                else "this corpus catalog"
            )
            absence_line = (
                f"{undated_note} Absence here is NOT evidence that the instrument does not exist; repeat the "
                "search without date_from/date_to to include the undated entries."
                if undated_note
                else "Absence here is NOT evidence that the instrument does not exist; it may simply not be in "
                "this corpus."
            )
            output = f"""NO RESULTS: No entry in {catalog_scope} matches these keywords and filters.
This tool searches the local corpus catalog metadata only — not document bodies and not the BDDK website.
{absence_line}
DO NOT provide information about BDDK regulations from your own knowledge.
Try: (1) call search_document_store with the same query for full-text semantic search, or
(2) use only words you'd expect in the regulation's title."""
            return structured_tool_result(
                RegulationCatalogResponse(
                    status="no_results",
                    text=output,
                    keywords=keywords,
                    page=result.page,
                    page_size=result.page_size,
                    total_results=result.total_results,
                    undated_excluded=undated_excluded,
                    warnings=[undated_note] if undated_note else [],
                )
            )

        # Batch version count lookup — one query instead of N
        doc_ids = [d.document_id for d in result.decisions]
        version_counts = await deps.doc_store.get_version_counts(doc_ids)

        lines = [f"Found {result.total_results} result(s) (page {result.page}):\n"]
        if undated_note:
            lines.insert(0, f"Note: {undated_note}")
        items: list[RegulationCatalogItem] = []
        evidence: list[EvidenceReference] = []
        warnings: list[str] = [UNTRUSTED_SOURCE_WARNING, LEGAL_APPLICABILITY_WARNING]
        if undated_note:
            warnings.append(undated_note)
        for d in result.decisions:
            date_info = f" ({d.decision_date} - {d.decision_number})" if d.decision_date else ""
            cat_info = f" [{d.category}]" if d.category else ""
            lines.append(f"**{d.title}**{date_info}{cat_info}")
            lines.append(f"  Document ID: {d.document_id}")
            if d.source_url:
                lines.append(f"  Catalog source (not a validated citation): {d.source_url}")
            if status_notice := document_status_notice(d.category):
                lines.append(f"  Document status: {status_notice}")
                warnings.append(status_notice)
            ver_count, ver_latest = version_counts.get(d.document_id, (0, None))
            if ver_count:
                lines.append(
                    f"  Replaced bodies in this corpus file: {ver_count} (latest replaced on: {ver_latest}); "
                    "a local text-change log, not amendment history"
                )
            quality = assess_markdown_quality("", document_id=d.document_id)
            lines.extend(_quality_result_lines(quality))
            if d.content and d.content.strip() != d.title.strip():
                lines.append(
                    f"  Catalog summary/excerpt (NOT document text; never quote as the provision): {d.content}"
                )
                warnings.append(_CATALOG_SUMMARY_WARNING)
            lines.append("")
            quality_metadata = _quality_metadata(quality)
            if quality.warning:
                warnings.append(quality.warning)
            items.append(
                RegulationCatalogItem(
                    document_id=d.document_id,
                    title=d.title,
                    summary=d.content,
                    decision_date=d.decision_date,
                    decision_number=d.decision_number,
                    category=d.category,
                    source_url=d.source_url,
                    version_count=ver_count,
                    latest_version_at=ver_latest,
                    quality=quality_metadata,
                )
            )
            evidence.append(
                EvidenceReference(
                    document_id=d.document_id,
                    title=d.title,
                    source_url=d.source_url or None,
                    decision_date=d.decision_date or None,
                    decision_number=d.decision_number or None,
                    category=d.category or None,
                    retrieval_source="catalog",
                    quality=quality_metadata,
                )
            )

        output = "\n".join(lines)
        response = RegulationCatalogResponse(
            status="ok",
            text=output,
            evidence=evidence,
            warnings=list(dict.fromkeys(warnings)),
            keywords=keywords,
            page=result.page,
            page_size=result.page_size,
            total_results=result.total_results,
            undated_excluded=undated_excluded,
            results=items,
        )
        _search_cache.set(cache_key, response)
        return structured_tool_result(response)

    @mcp.tool()
    @logged_tool(logger)
    async def search_bddk_institutions(
        keywords: OptionalSearchKeywords = "",
        institution_type: InstitutionType = None,
        active_only: ActiveOnly = True,
    ) -> str:
        """
        Search the BDDK institution directory (banks, leasing, factoring, etc.).

        Banks that BDDK's directory marks as digital banks are shown with
        "(Dijital Banka)" after the name.

        Args:
            keywords: Search terms (e.g. "Ziraat", "Garanti", "katılım")
            institution_type: Filter by type: Banka, Finansal Kiralama Şirketi,
                Faktoring Şirketi, Finansman Şirketi, Varlık Yönetim Şirketi
            active_only: If true (default), only show active institutions
        """
        institution_type = normalize_institution_type(institution_type)
        try:
            directory = await fetch_institutions_with_status(deps.http, institution_type)
        except BddkUpstreamError:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "The BDDK institution directory could not be fetched (upstream or network unavailable). "
                "This is NOT evidence that an institution does not exist.",
                retryable=True,
                hint="Retry later. In restricted networks, verify egress to www.bddk.org.tr is permitted.",
            )

        institutions = directory.institutions
        # A truncated directory must never be presented as the complete one.
        partial_warning = (
            f"\nWARNING: {directory.failed_pages} of {directory.attempted_pages} directory pages could not be "
            "fetched; this listing is incomplete and absence here is not evidence that an institution is "
            "unlicensed or does not exist."
            if directory.partial
            else ""
        )

        if active_only:
            institutions = [i for i in institutions if i["status"] == "Aktif"]

        if keywords:
            kw = _turkish_lower(keywords)
            institutions = [
                i for i in institutions if kw in _turkish_lower(i["name"]) or kw in _turkish_lower(i.get("type", ""))
            ]

        if not institutions:
            return (
                """NO RESULTS: No institutions found matching these criteria.
DO NOT guess institution names, license statuses, or other details.
Suggest the user try: broader keywords or removing the type/active filter."""
                + partial_warning
            )

        lines = [f"Found {len(institutions)} institution(s):\n"]
        for i in institutions:
            status = f" ({i['status']})" if i["status"] != "Aktif" else ""
            digital = " (Dijital Banka)" if i.get("digital") else ""
            website = f" — {i['website']}" if i["website"] else ""
            lines.append(f"**{i['name']}**{digital}{status} [{i['type']}]{website}")
        if partial_warning:
            lines.append(partial_warning.strip())
        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    async def search_bddk_announcements(
        keywords: OptionalSearchKeywords = "",
        category: AnnouncementCategory = "basın",
    ) -> str:
        """
        Search BDDK announcements and press releases.

        Results are listed newest first by announcement date (undated last); at most
        the 20 most recent are shown, and the output says how many more matched.

        Args:
            keywords: Search terms in Turkish
            category: Announcement type: basın (press), mevzuat (regulation),
                insan kaynakları (HR), veri (data publication), kuruluş (institution).
                Use "tümü" or "all" to search across all categories.
        """
        category = normalize_announcement_category(category)
        category_ids = {
            "basın": [39],
            "mevzuat": [40],
            "insan kaynakları": [41],
            "veri": [42],
            "kuruluş": [48],
            "tümü": list(ANNOUNCEMENT_CATEGORY_IDS),
        }
        cat_ids = category_ids[category]

        announcements: list[dict] = []
        failed_categories: list[int] = []
        fetched_any = False
        for cat_id in cat_ids:
            try:
                announcements.extend(await fetch_announcements(deps.http, cat_id))
                fetched_any = True
            except BddkUpstreamUnreachableError:
                # The host itself is unreachable; the remaining categories live
                # on that same host and would fail just as slowly, so abort
                # rather than repeating a multi-second timeout per category.
                failed_categories.extend(cat_ids[cat_ids.index(cat_id) :])
                break
            except BddkUpstreamError:
                # A single category failed (HTTP status or layout); the others
                # are still worth attempting.
                failed_categories.append(cat_id)

        if failed_categories and not fetched_any:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "BDDK announcements could not be fetched (upstream or network unavailable). "
                "This is NOT evidence that no announcements exist.",
                retryable=True,
                hint="Retry later. In restricted networks, verify egress to www.bddk.org.tr is permitted.",
            )

        partial_warning = ""
        if failed_categories:
            partial_warning = (
                f"\nWARNING: {len(failed_categories)} of {len(cat_ids)} announcement categories "
                "could not be fetched; results may be incomplete."
            )

        if keywords:
            kw = _turkish_lower(keywords)
            announcements = [a for a in announcements if kw in _turkish_lower(a.get("title", ""))]

        if not announcements:
            return (
                """NO RESULTS: No BDDK announcements found matching these criteria.
DO NOT fabricate announcements or press releases.
Suggest the user try: different keywords or a different category (basın, mevzuat, insan kaynakları, veri, kuruluş, or tümü for all)."""
                + partial_warning
            )

        announcements = _newest_first(announcements)
        total = len(announcements)
        shown = announcements[:_MAX_ANNOUNCEMENTS_SHOWN]
        header = f"Found {total} announcement(s)"
        if total > len(shown):
            header += f"; showing the {len(shown)} most recent of {total}"
        lines = [header + ":\n"]
        for a in shown:
            date_info = f" ({a['date']})" if a.get("date") else ""
            category_info = f" [{a['category']}]" if a.get("category") else ""
            lines.append(f"**{a['title']}**{date_info}{category_info}")
            if a.get("url"):
                lines.append(f"  URL: {a['url']}")
            lines.append("")
        if total > len(shown):
            lines.append(f"... and {total - len(shown)} more (narrow the keywords or the category to see them).")
        if partial_warning:
            lines.append(partial_warning.strip())
        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    async def search_document_store(
        query: SemanticQuery,
        category: OptionalRegulationCategory = None,
        limit: ResultLimit = 10,
    ) -> DocumentSearchToolResult:
        """
        Semantic search over BDDK document BODIES (full text of the local corpus).

        Uses multilingual-e5-base embeddings on chunked document content. Use this
        when query terms might appear inside the document text rather than in the
        title — abbreviations like "TFRS 9", article references, calculation
        formulas, defined terms. Understands meaning, not just keywords.

        For title-only catalog lookups (where you know words from the regulation's
        name), use search_bddk_regulations instead.

        For legal or audit questions, use a section-first workflow and treat these
        document-level results as leads.
        Prefer search_document_sections or get_document_section for exact articles,
        principles, paragraphs, and cited legal conclusions.

        Args:
            query: Natural language query in Turkish (e.g. "faiz oranı riski nasıl hesaplanır")
            category: Optional category filter (e.g. "Yönetmelik", "Rehber", "Kurul Kararı")
            limit: Maximum results to return (default 10)
        """
        if deps.vector_store is None:
            return tool_error(
                "SEMANTIC_SEARCH_UNAVAILABLE",
                "Semantic document search is unavailable in this runtime.",
                retryable=False,
                hint="Use search_document_sections or get_document_section for corpus retrieval.",
            )

        try:
            await deps.vector_store.assert_semantic_search_ready()
        except SemanticSearchUnavailableError:
            return tool_error(
                "SEMANTIC_SEARCH_UNAVAILABLE",
                "Semantic document search failed its runtime model check.",
                retryable=False,
                hint="Use search_document_sections or get_document_section; an operator must repair this runtime.",
            )
        except SemanticSearchReadinessError:
            return tool_error(
                "SEMANTIC_SEARCH_NOT_READY",
                "Semantic document search could not complete its readiness check.",
                retryable=True,
                hint="Retry once, then use search_document_sections or get_document_section.",
            )

        cache_key = f"semantic:{await _corpus_cache_token(deps)}:{query}:{category}:{limit}"
        cached = _search_cache.get(cache_key)
        if isinstance(cached, DocumentSearchResponse):
            return structured_tool_result(cached)

        try:
            hits = await deps.vector_store.search(query, limit=limit, category=category)
        except SemanticSearchUnavailableError:
            return tool_error(
                "SEMANTIC_SEARCH_UNAVAILABLE",
                "Semantic document search failed during query encoding.",
                retryable=False,
                hint="Use search_document_sections or get_document_section; an operator must repair this runtime.",
            )

        if not hits:
            output = f"""NO RESULTS: No documents found matching '{query}'.
DO NOT provide information from your own knowledge about BDDK regulations.
Suggest the user try: different Turkish keywords, broader terms, or removing the category filter."""
            return structured_tool_result(
                DocumentSearchResponse(
                    status="no_results",
                    text=output,
                    query=query,
                    category=category,
                )
            )

        govde_doc_ids = [str(h["doc_id"]) for h in hits if h.get("section_type") == "govde"]
        structureless_ids = (
            await deps.doc_store.structureless_documents(govde_doc_ids)
            if govde_doc_ids and deps.doc_store is not None
            else set()
        )
        lines = [f"Found {len(hits)} result(s) for '{query}':\n"]
        hit_quality: dict[str, QualityAssessment] = {}
        items: list[DocumentSearchItem] = []
        evidence: list[EvidenceReference] = []
        warnings: list[str] = [UNTRUSTED_SOURCE_WARNING, _SNIPPET_TRUNCATION_WARNING, LEGAL_APPLICABILITY_WARNING]
        for h in hits:
            date_info = f" ({h['decision_date']})" if h.get("decision_date") else ""
            cat_info = f" [{h['category']}]" if h.get("category") else ""
            match_type = h.get("match_type") if h.get("match_type") in _MATCH_TYPES else None
            if match_type == "fts_exact":
                match_strength = "keyword"
                relevance = f" {_FTS_EXACT_LABEL}"
            else:
                match_strength = _match_strength(h["relevance"])
                relevance = f" [{match_strength} match, relevance {h['relevance']:.1%}]"
            lines.append(f"**{h['title']}**{date_info}{cat_info}{relevance}")
            lines.append(f"  Document ID: {h['doc_id']}")
            if status_notice := document_status_notice(str(h.get("category") or "")):
                lines.append(f"  Document status: {status_notice}")
                warnings.append(status_notice)
            section_type = str(h.get("section_type") or "")
            section_ref = str(h.get("section_ref") or "")
            source_url = str(h.get("source_url") or "")
            decision_number = str(h.get("decision_number") or "")
            decision_date = str(h.get("decision_date") or "")
            if section_type == "govde" and str(h["doc_id"]) in structureless_ids:
                lines.append(f"  Section: {STRUCTURELESS_GOVDE_LABEL}")
                lines.append(f"  {STRUCTURELESS_CITATION_HINT}")
            elif section_type == "govde":
                lines.append("  Section: govde remainder — not a legal provision identity")
            elif section_type:
                lines.append(f"  Section: {section_type} {section_ref}".rstrip())
            if source_url:
                lines.append(f"  Catalog source (not a validated citation): {source_url}")
            if decision := decision_identifier_text(decision_date, decision_number):
                lines.append(f"  Decision date / number: {decision}")
            quality = _search_hit_quality(h)
            hit_quality[str(h["doc_id"])] = quality
            lines.extend(_quality_result_lines(quality))
            if h.get("snippet"):
                lines.append(f"  ...{h['snippet'][:200]}...")
            if section_type and section_ref and section_type != "govde":
                lines.append(
                    f'  Full section: get_document_section(document_id="{h["doc_id"]}", '
                    f'section_type="{section_type}", section_ref="{section_ref}")'
                )
            lines.append("")
            quality_metadata = _quality_metadata(quality)
            if quality.warning:
                warnings.append(quality.warning)
            items.append(
                DocumentSearchItem(
                    document_id=str(h["doc_id"]),
                    title=str(h.get("title") or h["doc_id"]),
                    category=str(h.get("category") or ""),
                    decision_date=str(h.get("decision_date") or ""),
                    snippet=str(h.get("snippet") or ""),
                    relevance=float(h["relevance"]),
                    match_strength=match_strength,  # type: ignore[arg-type]
                    quality=quality_metadata,
                    match_type=match_type,
                    section_type=section_type or None,
                    section_ref=section_ref or None,
                    source_url=source_url or None,
                    decision_number=decision_number or None,
                )
            )
            evidence.append(
                EvidenceReference(
                    document_id=str(h["doc_id"]),
                    title=str(h.get("title") or h["doc_id"]),
                    decision_date=str(h.get("decision_date") or "") or None,
                    category=str(h.get("category") or "") or None,
                    retrieval_source="vector_store",
                    source_url=source_url or None,
                    decision_number=decision_number or None,
                    section_type=section_type or None,
                    section_ref=section_ref or None,
                    quality=quality_metadata,
                )
            )

        lines.append(
            "For legal/audit answers, use these results as leads; retrieve exact provisions with "
            "search_document_sections or get_document_section before making detailed conclusions."
        )

        # fts_exact hits have no semantic score, so they are not "weak".
        low_count = sum(1 for h in hits if h.get("match_type") != "fts_exact" and h.get("relevance", 0) < 0.50)
        if low_count > 0:
            lines.append(
                f"\nWARNING: {low_count} result(s) are weak matches. They may not be directly relevant. Verify before citing."
            )
            warnings.append(f"{low_count} result(s) are weak matches; verify before citing.")

        output = "\n".join(lines)
        response = DocumentSearchResponse(
            status="ok",
            text=output,
            evidence=evidence,
            warnings=list(dict.fromkeys(warnings)),
            query=query,
            category=category,
            results=items,
        )
        _search_cache.set(cache_key, response)
        return structured_tool_result(response)
