"""Search tools: search_bddk_regulations, search_bddk_institutions,
search_bddk_announcements, and search_document_store.

Uses an OrderedDict-based LRU cache for O(1) eviction instead of the
O(n) min() scan in server.py.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from bddk_mcp.core.config import ANNOUNCEMENT_CATEGORY_IDS, SEARCH_CACHE_MAX, SEARCH_CACHE_TTL
from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.core.models import BddkSearchRequest
from bddk_mcp.ingest.client import _turkish_lower
from bddk_mcp.ingest.data_sources import fetch_announcements, fetch_institutions
from bddk_mcp.observability.metrics import metrics
from bddk_mcp.observability.telemetry import (
    elapsed_ms,
    record_tool_call_trace,
    relevance_stats_from_hits,
    unique_doc_ids,
)
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
    validate_date_order,
)
from bddk_mcp.tools.errors import UPSTREAM_FETCH_FAILED, tool_error
from bddk_mcp.tools.structured_outputs import (
    UNTRUSTED_SOURCE_WARNING,
    DocumentSearchItem,
    DocumentSearchResponse,
    DocumentSearchToolResult,
    EvidenceReference,
    QualityMetadata,
    RegulationCatalogItem,
    RegulationCatalogResponse,
    RegulationCatalogToolResult,
    frame_untrusted_source,
    structured_tool_result,
)
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)


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

    def clear(self) -> None:
        """Remove every cached response when the served corpus epoch changes."""
        self._data.clear()


# Module-level cache shared across all invocations
_search_cache: _LRUCache = _LRUCache(max_size=SEARCH_CACHE_MAX, ttl=SEARCH_CACHE_TTL)


def clear_search_cache() -> None:
    """Invalidate all query results derived from the previous corpus release."""
    _search_cache.clear()


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


def register(mcp, deps: Dependencies) -> None:  # type: ignore[type-arg]
    """Register the four search tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def search_bddk_regulations(
        keywords: RegulationKeywords,
        page: PageNumber = 1,
        page_size: PageSize = 10,
        category: OptionalRegulationCategory = None,
        date_from: DateFrom = None,
        date_to: DateTo = None,
    ) -> RegulationCatalogToolResult:
        """
        Search the BDDK regulations CATALOG by title, category, decision number, and date.

        Covers all BDDK regulatory document types: yönetmelik, tebliğ, genelge, rehber,
        kurul kararı, kanun, mülga düzenleme, etc. ("decisions" is loose terminology;
        the catalog is broader than just kararlar.)

        This is a TITLE/METADATA search only — it does NOT search document body content.
        Use this when you know words that appear in the regulation's name itself
        (e.g. "kredilerin sınıflandırılması", "elektronik para", "banka kartları").

        For terms that appear only inside document bodies — abbreviations like "TFRS 9",
        article references ("madde 5"), defined terms, calculation formulas — use
        search_document_store instead (semantic search over full text).

        All keyword tokens must match somewhere in title/category/date/number; one
        missing token returns zero results.

        Args:
            keywords: Search terms in Turkish (e.g. "elektronik para", "banka lisansı")
            page: Page number, starting from 1
            page_size: Number of results per page (max 50)
            category: Optional category filter. Available categories:
                Yönetmelik, Genelge, Tebliğ, Rehber, Bilgi Sistemleri,
                Sermaye Yeterliliği, Faizsiz Bankacılık, Tekdüzen Hesap Planı,
                Kurul Kararı, Kanun, Banka Kartları,
                Finansal Kiralama ve Faktoring, BDDK Düzenlemesi,
                Düzenleme Taslağı, Mülga Düzenleme
            date_from: Optional start date filter (DD.MM.YYYY)
            date_to: Optional end date filter (DD.MM.YYYY)
        """
        start = time.perf_counter()
        args = {
            "keywords": keywords,
            "page": page,
            "page_size": page_size,
            "category": category,
            "date_from": date_from,
            "date_to": date_to,
        }
        validate_date_order(date_from, date_to)
        cache_key = f"regulations:{keywords}:{page}:{page_size}:{category}:{date_from}:{date_to}"
        cached = _search_cache.get(cache_key)
        if isinstance(cached, RegulationCatalogResponse):
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_bddk_regulations",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=None,
                doc_ids=[item.document_id for item in cached.results],
                relevance_stats={"cache": "hit"},
            )
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

        if not result.decisions:
            metrics.record_empty_search("search_bddk_regulations")
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_bddk_regulations",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "no_results"},
            )
            output = """NO RESULTS: No BDDK regulations found whose title/category/date/number matches ALL keywords.
This tool searches catalog metadata only — not document bodies.
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
                )
            )

        # Batch version count lookup — one query instead of N
        doc_ids = [d.document_id for d in result.decisions]
        version_counts = await deps.doc_store.get_version_counts(doc_ids)

        lines = [f"Found {result.total_results} result(s) (page {result.page}):\n"]
        items: list[RegulationCatalogItem] = []
        evidence: list[EvidenceReference] = []
        warnings: list[str] = [UNTRUSTED_SOURCE_WARNING]
        for d in result.decisions:
            date_info = f" ({d.decision_date} - {d.decision_number})" if d.decision_date else ""
            cat_info = f" [{d.category}]" if d.category else ""
            lines.append(f"**{d.title}**{date_info}{cat_info}")
            lines.append(f"  Document ID: {d.document_id}")
            ver_count, ver_latest = version_counts.get(d.document_id, (0, None))
            if ver_count:
                lines.append(f"  Versions: {ver_count} (latest: {ver_latest})")
            quality = assess_markdown_quality("", document_id=d.document_id)
            lines.extend(_quality_result_lines(quality))
            lines.append(f"  {d.content}\n")
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
            results=items,
        )
        _search_cache.set(cache_key, response)
        await record_tool_call_trace(
            getattr(deps, "telemetry_pool", None),
            tool_name="search_bddk_regulations",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(result.decisions),
            doc_ids=[d.document_id for d in result.decisions],
            relevance_stats={"total_results": result.total_results, "page": result.page},
        )
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

        Args:
            keywords: Search terms (e.g. "Ziraat", "Garanti", "katılım")
            institution_type: Filter by type: Banka, Finansal Kiralama Şirketi,
                Faktoring Şirketi, Finansman Şirketi, Varlık Yönetim Şirketi
            active_only: If true (default), only show active institutions
        """
        institution_type = normalize_institution_type(institution_type)
        try:
            institutions = await fetch_institutions(deps.http, institution_type)
        except BddkUpstreamError:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "The BDDK institution directory could not be fetched (upstream or network unavailable). "
                "This is NOT evidence that an institution does not exist.",
                retryable=True,
                hint="Retry later. In restricted networks, verify egress to www.bddk.org.tr is permitted.",
            )

        if active_only:
            institutions = [i for i in institutions if i["status"] == "Aktif"]

        if keywords:
            kw = _turkish_lower(keywords)
            institutions = [
                i for i in institutions if kw in _turkish_lower(i["name"]) or kw in _turkish_lower(i.get("type", ""))
            ]

        if not institutions:
            metrics.record_empty_search("search_bddk_institutions")
            return """NO RESULTS: No institutions found matching these criteria.
DO NOT guess institution names, license statuses, or other details.
Suggest the user try: broader keywords or removing the type/active filter."""

        lines = [f"Found {len(institutions)} institution(s):\n"]
        for i in institutions:
            status = f" ({i['status']})" if i["status"] != "Aktif" else ""
            website = f" — {i['website']}" if i["website"] else ""
            lines.append(f"**{i['name']}**{status} [{i['type']}]{website}")
        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    async def search_bddk_announcements(
        keywords: OptionalSearchKeywords = "",
        category: AnnouncementCategory = "basın",
    ) -> str:
        """
        Search BDDK announcements and press releases.

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
            except BddkUpstreamError:
                failed_categories.append(cat_id)
                if not fetched_any:
                    # The first category is already unreachable; the remaining
                    # categories share the same host and would fail just as
                    # slowly, so fail fast instead of retrying serially.
                    break

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
            metrics.record_empty_search("search_bddk_announcements")
            return (
                """NO RESULTS: No BDDK announcements found matching these criteria.
DO NOT fabricate announcements or press releases.
Suggest the user try: different keywords or a different category (basın, mevzuat, insan kaynakları, veri, kuruluş, or tümü for all)."""
                + partial_warning
            )

        lines = [f"Found {len(announcements)} announcement(s):\n"]
        for a in announcements[:20]:
            date_info = f" ({a['date']})" if a.get("date") else ""
            lines.append(f"**{a['title']}**{date_info}")
            if a.get("url"):
                lines.append(f"  URL: {a['url']}")
            lines.append("")
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
        Semantic search over BDDK document BODIES (full text via pgvector).

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
        start = time.perf_counter()
        args = {"query": query, "category": category, "limit": limit}
        if deps.vector_store is None:
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_document_store",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "semantic_search_unavailable", "retryable": False},
            )
            return tool_error(
                "SEMANTIC_SEARCH_UNAVAILABLE",
                "Semantic document search is unavailable in this runtime.",
                retryable=False,
                hint="Use search_document_sections or get_document_section for corpus retrieval.",
            )

        try:
            await deps.vector_store.assert_semantic_search_ready()
        except SemanticSearchUnavailableError:
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_document_store",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "semantic_search_unavailable", "retryable": False},
            )
            return tool_error(
                "SEMANTIC_SEARCH_UNAVAILABLE",
                "Semantic document search failed its runtime model check.",
                retryable=False,
                hint="Use search_document_sections or get_document_section; an operator must repair this runtime.",
            )
        except SemanticSearchReadinessError:
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_document_store",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "semantic_search_not_ready", "retryable": True},
            )
            return tool_error(
                "SEMANTIC_SEARCH_NOT_READY",
                "Semantic document search could not complete its readiness check.",
                retryable=True,
                hint="Retry once, then use search_document_sections or get_document_section.",
            )

        cache_key = f"semantic:{query}:{category}:{limit}"
        cached = _search_cache.get(cache_key)
        if isinstance(cached, DocumentSearchResponse):
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_document_store",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=None,
                doc_ids=[item.document_id for item in cached.results],
                relevance_stats={"cache": "hit"},
            )
            return structured_tool_result(cached)

        try:
            hits = await deps.vector_store.search(query, limit=limit, category=category)
        except SemanticSearchUnavailableError:
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_document_store",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "semantic_search_unavailable", "retryable": False},
            )
            return tool_error(
                "SEMANTIC_SEARCH_UNAVAILABLE",
                "Semantic document search failed during query encoding.",
                retryable=False,
                hint="Use search_document_sections or get_document_section; an operator must repair this runtime.",
            )

        if not hits:
            metrics.record_empty_search("search_document_store")
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_document_store",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "no_results"},
            )
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

        lines = [f"Found {len(hits)} result(s) for '{query}':\n"]
        hit_quality: dict[str, QualityAssessment] = {}
        items: list[DocumentSearchItem] = []
        evidence: list[EvidenceReference] = []
        warnings: list[str] = [UNTRUSTED_SOURCE_WARNING]
        for h in hits:
            date_info = f" ({h['decision_date']})" if h.get("decision_date") else ""
            cat_info = f" [{h['category']}]" if h.get("category") else ""
            match_strength = _match_strength(h["relevance"])
            relevance = f" [{match_strength} match, relevance {h['relevance']:.1%}]"
            lines.append(f"**{h['title']}**{date_info}{cat_info}{relevance}")
            lines.append(f"  Document ID: {h['doc_id']}")
            quality = _search_hit_quality(h)
            hit_quality[str(h["doc_id"])] = quality
            lines.extend(_quality_result_lines(quality))
            if h.get("snippet"):
                lines.append(f"  ...{h['snippet'][:200]}...")
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
                )
            )
            evidence.append(
                EvidenceReference(
                    document_id=str(h["doc_id"]),
                    title=str(h.get("title") or h["doc_id"]),
                    decision_date=str(h.get("decision_date") or "") or None,
                    category=str(h.get("category") or "") or None,
                    retrieval_source="vector_store",
                    quality=quality_metadata,
                )
            )

        lines.append(
            "For legal/audit answers, use these results as leads; retrieve exact provisions with "
            "search_document_sections or get_document_section before making detailed conclusions."
        )

        low_count = sum(1 for h in hits if h.get("relevance", 0) < 0.50)
        if low_count > 0:
            metrics.record_low_confidence_hit()
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
        await record_tool_call_trace(
            getattr(deps, "telemetry_pool", None),
            tool_name="search_document_store",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(hits),
            doc_ids=unique_doc_ids([hit.get("doc_id") for hit in hits]),
            quality_labels={
                doc_id: {"label": quality.label, "flags": quality.flags} for doc_id, quality in hit_quality.items()
            },
            relevance_stats=relevance_stats_from_hits(hits),
        )
        return structured_tool_result(response)
