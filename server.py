"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator

import asyncpg
from mcp.server.fastmcp import FastMCP

from analytics import analyze_trends, build_digest, check_updates, compare_metrics
from client import BddkApiClient, _turkish_lower
from config import (
    AUTO_SYNC,
    DATABASE_URL,
    PG_POOL_MAX,
    PG_POOL_MIN,
    PREFER_NOUGAT,
    SEARCH_CACHE_MAX,
    SEARCH_CACHE_TTL,
    validate_column,
    validate_currency,
    validate_metric_id,
    validate_month,
    validate_table_no,
    validate_year,
)
from data_sources import (
    fetch_announcements,
    fetch_bulletin_snapshot,
    fetch_institutions,
    fetch_weekly_bulletin,
)
from doc_store import DocumentStore
from exceptions import BddkError, BddkStorageError, BddkVectorStoreError
from logging_config import configure_logging
from metrics import metrics
from models import BddkSearchRequest
from vector_store import VectorStore

configure_logging()
logger = logging.getLogger(__name__)

# -- FastMCP instance ---------------------------------------------------------

mcp = FastMCP(
    "BDDK",
    instructions="""\
Search and retrieve BDDK (Turkish Banking Regulation) decisions, regulations, and statistical data.

GROUNDING RULES — follow these strictly:
1. ONLY use information returned by tool calls. Never supplement with your own knowledge about BDDK decisions.
2. If a search returns no results, say so explicitly. Do NOT guess or invent decisions.
3. Always include document_id, decision_date, and decision_number in your response when available.
4. If document content is paginated, do NOT speculate about content on pages you have not retrieved.
5. Never fabricate karar numarası (decision numbers), tarih (dates), or legal conclusions.
6. When quoting from a document, quote only text that appears verbatim in the tool output.
7. If relevance scores are below 50%, flag this to the user and recommend refining the query.
8. Distinguish clearly between: (a) information from BDDK tools, and (b) your general knowledge.\
""",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    stateless_http=True,
)

# -- Global state -------------------------------------------------------------

_pool: asyncpg.Pool | None = None
_client: BddkApiClient | None = None
_doc_store: DocumentStore | None = None
_vector_store: VectorStore | None = None
_server_start_time: float = time.time()
_last_sync_time: float | None = None
_sync_task: asyncio.Task | None = None

# TTL cache for search results
_search_cache: dict[str, tuple[float, str]] = {}
_SEARCH_CACHE_TTL = SEARCH_CACHE_TTL
_SEARCH_CACHE_MAX = SEARCH_CACHE_MAX


def _cached_search(cache_key: str) -> str | None:
    """Return cached result if fresh, else None."""
    entry = _search_cache.get(cache_key)
    if entry and (time.time() - entry[0]) < _SEARCH_CACHE_TTL:
        return entry[1]
    return None


def _store_search(cache_key: str, result: str) -> None:
    """Store a search result in the TTL cache."""
    if len(_search_cache) >= _SEARCH_CACHE_MAX:
        oldest_key = min(_search_cache, key=lambda k: _search_cache[k][0])
        del _search_cache[oldest_key]
    _search_cache[cache_key] = (time.time(), result)


# -- Pool + component init ----------------------------------------------------


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=PG_POOL_MIN,
            max_size=PG_POOL_MAX,
        )
        logger.info("PostgreSQL pool created: %s", DATABASE_URL.split("@")[-1])
    return _pool


async def _get_doc_store() -> DocumentStore:
    global _doc_store
    if _doc_store is None:
        pool = await _get_pool()
        _doc_store = DocumentStore(pool)
        await _doc_store.initialize()
    return _doc_store


async def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        pool = await _get_pool()
        _vector_store = VectorStore(pool)
        await _vector_store.initialize()
    return _vector_store


async def _get_client() -> BddkApiClient:
    global _client
    if _client is None:
        pool = await _get_pool()
        store = await _get_doc_store()
        _client = BddkApiClient(pool=pool, doc_store=store)
        await _client.initialize()
    return _client


# -- Tools --------------------------------------------------------------------


@mcp.tool()
async def search_bddk_decisions(
    keywords: str,
    page: int = 1,
    page_size: int = 10,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """
    Search for BDDK (Banking Regulation and Supervision Agency) decisions.

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
    cache_key = f"decisions:{keywords}:{page}:{page_size}:{category}:{date_from}:{date_to}"
    cached = _cached_search(cache_key)
    if cached:
        return cached

    client = await _get_client()
    request = BddkSearchRequest(
        keywords=keywords,
        page=page,
        page_size=page_size,
        category=category,
        date_from=date_from,
        date_to=date_to,
    )
    result = await client.search_decisions(request)

    if not result.decisions:
        metrics.record_empty_search("search_bddk_decisions")
        return (
            "NO RESULTS: No BDDK decisions found matching these keywords.\n"
            "DO NOT provide information about BDDK decisions from your own knowledge.\n"
            "Suggest the user try: different Turkish keywords, broader terms, "
            "or removing date/category filters."
        )

    store = await _get_doc_store()

    lines = [f"Found {result.total_results} result(s) (page {result.page}):\n"]
    for d in result.decisions:
        date_info = f" ({d.decision_date} - {d.decision_number})" if d.decision_date else ""
        cat_info = f" [{d.category}]" if d.category else ""
        lines.append(f"**{d.title}**{date_info}{cat_info}")
        lines.append(f"  Document ID: {d.document_id}")
        history = await store.get_document_history(d.document_id)
        if history:
            lines.append(f"  Versions: {len(history)} (latest: {history[0]['synced_at']})")
        lines.append(f"  {d.content}\n")
    output = "\n".join(lines)
    _store_search(cache_key, output)
    return output


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
    client = await _get_client()
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
    try:
        vs = await _get_vector_store()
        page = await vs.get_document_page(document_id, page_number)
        if page and page["content"] and "Invalid page" not in page["content"]:
            return _build_header(page["page_number"], page["total_pages"]) + page["content"]
    except (RuntimeError, BddkVectorStoreError) as e:
        logger.debug("pgvector lookup failed for %s, falling back: %s", document_id, e)

    # Fallback to document store → live fetch
    doc = await client.get_document_markdown(document_id, page_number)
    return _build_header(doc.page_number, doc.total_pages) + doc.markdown_content


@mcp.tool()
async def bddk_cache_status() -> str:
    """
    Show BDDK cache statistics: total items, age, categories, and any page errors.
    """
    client = await _get_client()
    status = client.cache_status()

    lines = ["**BDDK Cache Status**\n"]
    lines.append(f"  Total items: {status['total_items']}")
    lines.append(f"  Cache valid: {status['cache_valid']}")
    if status["cache_age_seconds"] is not None:
        mins = status["cache_age_seconds"] // 60
        lines.append(f"  Cache age: {mins} min ({status['cache_age_seconds']}s)")
    lines.append(f"  TTL: {status['ttl_seconds']}s")

    if status["categories"]:
        lines.append("\n**Categories:**")
        for cat, count in status["categories"].items():
            lines.append(f"  {cat}: {count}")

    if status["page_errors"]:
        lines.append("\n**Page Errors:**")
        for page_id, err in status["page_errors"].items():
            lines.append(f"  Page {page_id}: {err}")

    return "\n".join(lines)


@mcp.tool()
async def search_bddk_institutions(
    keywords: str = "",
    institution_type: str | None = None,
    active_only: bool = True,
) -> str:
    """
    Search the BDDK institution directory (banks, leasing, factoring, etc.).

    Args:
        keywords: Search terms (e.g. "Ziraat", "Garanti", "katılım")
        institution_type: Filter by type: Banka, Finansal Kiralama Şirketi,
            Faktoring Şirketi, Finansman Şirketi, Varlık Yönetim Şirketi
        active_only: If true (default), only show active institutions
    """
    client = await _get_client()
    institutions = await fetch_institutions(client._http, institution_type)

    if active_only:
        institutions = [i for i in institutions if i["status"] == "Aktif"]

    if keywords:
        kw = _turkish_lower(keywords)
        institutions = [
            i for i in institutions if kw in _turkish_lower(i["name"]) or kw in _turkish_lower(i.get("type", ""))
        ]

    if not institutions:
        metrics.record_empty_search("search_bddk_institutions")
        return (
            "NO RESULTS: No institutions found matching these criteria.\n"
            "DO NOT guess institution names, license statuses, or other details.\n"
            "Suggest the user try: broader keywords or removing the type/active filter."
        )

    lines = [f"Found {len(institutions)} institution(s):\n"]
    for i in institutions:
        status = f" ({i['status']})" if i["status"] != "Aktif" else ""
        website = f" — {i['website']}" if i["website"] else ""
        lines.append(f"**{i['name']}**{status} [{i['type']}]{website}")
    return "\n".join(lines)


@mcp.tool()
async def get_bddk_bulletin(
    metric_id: str = "1.0.1",
    currency: str = "TRY",
    column: str = "1",
    date: str = "",
    days: int = 90,
) -> str:
    """
    Get weekly banking sector bulletin time-series data from BDDK.

    Args:
        metric_id: Metric ID. Common IDs:
            1.0.1=Toplam Krediler, 1.0.2=Tüketici Kredileri,
            1.0.4=Konut Kredileri, 1.0.8=Bireysel Kredi Kartları,
            1.0.10=Ticari Krediler. Use get_bddk_bulletin_snapshot for all metrics.
        currency: TRY or USD
        column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
        date: Specific date (DD.MM.YYYY), empty for latest
        days: Number of days of history (default 90)
    """
    try:
        validate_metric_id(metric_id)
        validate_currency(currency, "weekly")
        validate_column(column)
    except ValueError as e:
        return f"Validation error: {e}"

    client = await _get_client()
    data = await fetch_weekly_bulletin(
        client._http,
        metric_id,
        currency,
        days,
        date,
        column,
    )

    if "error" in data:
        return f"Error fetching bulletin: {data['error']}"

    lines = [f"**{data.get('title', 'BDDK Weekly Bulletin')}** ({data['currency']})\n"]

    dates = data.get("dates", [])
    values = data.get("values", [])

    if dates and values:
        for d, v in zip(dates[-10:], values[-10:], strict=False):
            lines.append(f"  {d}: {v}")
    else:
        lines.append("No data returned for the given parameters.")

    return "\n".join(lines)


@mcp.tool()
async def get_bddk_bulletin_snapshot() -> str:
    """
    Get the latest weekly bulletin snapshot -- all metrics with current TP/YP values.

    Returns a table of all banking sector metrics (loans, deposits, etc.)
    with their latest TP (TL) and YP (foreign currency) values.
    """
    client = await _get_client()
    rows = await fetch_bulletin_snapshot(client._http)

    if not rows:
        return "No bulletin data available."

    lines = ["**BDDK Weekly Bulletin — Latest Snapshot**\n"]
    lines.append(f"{'#':<4} {'Metric':<50} {'TP':>15} {'YP':>15} {'ID'}")
    lines.append("-" * 100)
    for r in rows:
        lines.append(f"{r['row_number']:<4} {r['name']:<50} {r['tp']:>15} {r['yp']:>15} {r['metric_id']}")
    return "\n".join(lines)


@mcp.tool()
async def search_bddk_announcements(
    keywords: str = "",
    category: str = "basın",
) -> str:
    """
    Search BDDK announcements and press releases.

    Args:
        keywords: Search terms in Turkish
        category: Announcement type: basın (press), mevzuat (regulation),
            insan kaynakları (HR), veri (data publication), kuruluş (institution).
            Use "tümü" or "all" to search across all categories.
    """
    cat_lower = _turkish_lower(category)

    cat_map: dict[str, list[int]] = {
        "basın": [39],
        "press": [39],
        "mevzuat": [40],
        "regul": [40],
        "insan": [41],
        "hr": [41],
        "veri": [42],
        "data": [42],
        "kuruluş": [48],
        "institution": [48],
        "tümü": [39, 40, 41, 42, 48],
        "all": [39, 40, 41, 42, 48],
    }

    cat_ids = [39]  # default
    for key, ids in cat_map.items():
        if key in cat_lower:
            cat_ids = ids
            break

    client = await _get_client()
    announcements: list[dict] = []
    for cat_id in cat_ids:
        announcements.extend(await fetch_announcements(client._http, cat_id))

    if keywords:
        kw = _turkish_lower(keywords)
        announcements = [a for a in announcements if kw in _turkish_lower(a.get("title", ""))]

    if not announcements:
        metrics.record_empty_search("search_bddk_announcements")
        return (
            "NO RESULTS: No BDDK announcements found matching these criteria.\n"
            "DO NOT fabricate announcements or press releases.\n"
            "Suggest the user try: different keywords or a different category "
            "(basın, mevzuat, insan kaynakları, veri, kuruluş, or tümü for all)."
        )

    lines = [f"Found {len(announcements)} announcement(s):\n"]
    for a in announcements[:20]:
        date_info = f" ({a['date']})" if a.get("date") else ""
        lines.append(f"**{a['title']}**{date_info}")
        if a.get("url"):
            lines.append(f"  URL: {a['url']}")
        lines.append("")
    return "\n".join(lines)


# -- v4 Analytics Tools -------------------------------------------------------


@mcp.tool()
async def analyze_bulletin_trends(
    metric_id: str = "1.0.1",
    currency: str = "TRY",
    column: str = "1",
    lookback_weeks: int = 12,
) -> str:
    """
    Analyze trends in BDDK weekly bulletin data with week-over-week changes.

    Returns current value, WoW change %, trend direction, min/max over the
    lookback period, and a Turkish-language narrative summary.

    Args:
        metric_id: Metric ID (e.g. 1.0.1=Toplam Krediler, 1.0.2=Tüketici Kredileri,
            1.0.4=Konut, 1.0.8=Bireysel Kredi Kartları, 1.0.10=Ticari Krediler)
        currency: TRY or USD
        column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
        lookback_weeks: Number of weeks to analyze (default 12)
    """
    try:
        validate_metric_id(metric_id)
        validate_currency(currency, "weekly")
        validate_column(column)
    except ValueError as e:
        return f"Validation error: {e}"

    client = await _get_client()
    result = await analyze_trends(
        client._http,
        metric_id,
        currency,
        column,
        lookback_weeks,
    )

    if "error" in result:
        return f"Error: {result['error']}"

    lines = [f"**Trend Analizi: {result['title']}**\n"]
    lines.append(result["narrative"])
    lines.append("")
    lines.append(f"  Güncel ({result['current_date']}): {result['current']:,.2f}")
    lines.append(f"  Önceki ({result['previous_date']}): {result['previous']:,.2f}")
    lines.append(f"  Haftalık değişim: {result['wow_change']:+,.2f} (%{result['wow_pct']:+.2f})")
    lines.append(f"  Dönem ortalaması: {result['avg']:,.2f}")
    lines.append(f"  Dönem min: {result['min']:,.2f} ({result['min_date']})")
    lines.append(f"  Dönem max: {result['max']:,.2f} ({result['max_date']})")
    lines.append(f"  Trend: {result['trend_direction']}")
    lines.append(f"  Veri noktası: {result['data_points']}")
    return "\n".join(lines)


@mcp.tool()
async def get_regulatory_digest(
    period: str = "month",
) -> str:
    """
    Get a digest of recent BDDK regulatory changes.

    Combines: new decisions, announcements, and bulletin data into
    an executive summary.

    Args:
        period: Time period -- week (7 days), month (30 days), quarter (90 days)
    """
    period_map = {"week": 7, "month": 30, "quarter": 90}
    days = period_map.get(period, 30)

    client = await _get_client()
    await client.ensure_cache()

    digest = await build_digest(client._http, client._cache, days)

    lines = [f"**BDDK Düzenleyici Özet — Son {days} Gün**\n"]
    lines.append(digest["narrative"])
    lines.append("")

    if digest["decisions_by_category"]:
        lines.append("**Kararlar (kategoriye göre):**")
        for cat, count in sorted(digest["decisions_by_category"].items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count}")
        lines.append("")

    if digest["new_decisions"]:
        lines.append("**Son Kararlar:**")
        for d in digest["new_decisions"][:10]:
            date = d.get("decision_date", "")
            lines.append(f"  - {d['title']} ({date}) [{d.get('category', '')}]")
        lines.append("")

    if digest["announcements"]:
        lines.append(f"**Duyurular ({len(digest['announcements'])}):**")
        for a in digest["announcements"][:10]:
            lines.append(f"  - {a['title']} ({a.get('date', '')})")
        lines.append("")

    if digest["bulletin_snapshot"]:
        lines.append("**Bülten Özet (ilk 5 metrik):**")
        for r in digest["bulletin_snapshot"]:
            lines.append(f"  {r['name']}: TP={r['tp']}, YP={r['yp']}")

    return "\n".join(lines)


@mcp.tool()
async def compare_bulletin_metrics(
    metric_ids: str = "1.0.1,1.0.2",
    currency: str = "TRY",
    column: str = "1",
    days: int = 90,
) -> str:
    """
    Compare multiple BDDK bulletin metrics side-by-side.

    Args:
        metric_ids: Comma-separated metric IDs (e.g. "1.0.1,1.0.2,1.0.4")
            Common: 1.0.1=Toplam Krediler, 1.0.2=Tüketici, 1.0.4=Konut,
            1.0.8=Kredi Kartları, 1.0.10=Ticari Krediler
        currency: TRY or USD
        column: 1=TP, 2=YP, 3=Toplam
        days: Days of history (default 90)
    """
    ids = [m.strip() for m in metric_ids.split(",") if m.strip()]
    if not ids:
        return "Please provide at least one metric ID."

    try:
        for mid in ids:
            validate_metric_id(mid)
        validate_currency(currency, "weekly")
        validate_column(column)
    except ValueError as e:
        return f"Validation error: {e}"

    client = await _get_client()
    result = await compare_metrics(client._http, ids, currency, column, days)

    col_label = {"1": "TP", "2": "YP", "3": "Toplam"}.get(column, column)
    lines = [f"**Metrik Karşılaştırması** ({currency}, {col_label})\n"]
    lines.append(f"{'Metrik':<55} {'Güncel':>15} {'Haftalık %':>12}")
    lines.append("-" * 85)

    for m in result["metrics"]:
        if "error" in m:
            lines.append(f"{m['metric_id']:<55} {'HATA':>15} {'-':>12}")
        else:
            title = m["title"][:55]
            lines.append(f"{title:<55} {m['current']:>15,.2f} {m['wow_pct']:>+11.2f}%")

    return "\n".join(lines)


@mcp.tool()
async def check_bddk_updates() -> str:
    """
    Check for new BDDK announcements since last check.

    Compares current announcements with cached state to detect new items.
    Useful for monitoring regulatory changes.
    """
    client = await _get_client()

    known_urls: set[str] = set()
    if hasattr(client, "_known_announcements"):
        known_urls = client._known_announcements
    else:
        from data_sources import fetch_announcements as _fa

        for cat_id in [39, 40, 41, 42, 48]:
            anns = await _fa(client._http, cat_id)
            for a in anns:
                if a.get("url"):
                    known_urls.add(a["url"])
        client._known_announcements = known_urls
        return (
            f"Baseline oluşturuldu: {len(known_urls)} duyuru biliniyor. "
            "Bir sonraki çağrıda yeni duyurular tespit edilecek."
        )

    result = await check_updates(client._http, client._cache, known_urls)

    for a in result.get("new_announcements", []):
        if a.get("url"):
            known_urls.add(a["url"])

    if not result["new_announcements"]:
        return "Yeni duyuru yok. Her şey güncel."

    lines = [f"**{result['new_announcements_count']} Yeni Duyuru Tespit Edildi!**\n"]
    for a in result["new_announcements"]:
        date = a.get("date", "")
        lines.append(f"  - {a['title']} ({date})")
        if a.get("url"):
            lines.append(f"    {a['url']}")
    return "\n".join(lines)


@mcp.tool()
async def get_bddk_monthly(
    table_no: int = 1,
    year: int = 2025,
    month: int = 12,
    currency: str = "TL",
    party_code: str = "10001",
) -> str:
    """
    Get BDDK monthly banking sector data (more detailed than weekly bulletin).

    Args:
        table_no: Table number (1-17). Key tables:
            1=Aktif Toplamı, 2=Krediler, 3=Menkul Değerler,
            4=Mevduat, 9=Sermaye Yeterliliği, 11=Gelir Tablosu,
            14=Takipteki Alacaklar
        year: Year (e.g. 2025)
        month: Month (1-12)
        currency: TL or USD
        party_code: Bank group code. 10001=Sektör, 10002=Mevduat Bankaları,
            10003=Kalkınma ve Yatırım, 10004=Katılım Bankaları,
            20001=Kamu, 20002=Özel, 20003=Yabancı
    """
    try:
        validate_table_no(table_no)
        validate_year(year)
        validate_month(month)
        validate_currency(currency, "monthly")
    except ValueError as e:
        return f"Validation error: {e}"

    from data_sources import fetch_monthly_bulletin

    client = await _get_client()
    result = await fetch_monthly_bulletin(
        client._http,
        table_no,
        year,
        month,
        currency,
        party_code,
    )

    if "error" in result:
        return f"Error: {result['error']}"

    lines = [f"**{result.get('title', 'BDDK Aylık Bülten')}**\n"]
    lines.append(f"Dönem: {month}/{year} | Para Birimi: {currency}\n")

    rows = result.get("rows", [])
    if not rows:
        lines.append("Bu parametreler için veri bulunamadı.")
    else:
        lines.append(f"{'Kalem':<55} {'TP':>15} {'YP':>15} {'Toplam':>15}")
        lines.append("-" * 105)
        for r in rows:
            lines.append(f"{r['name']:<55} {r.get('tp', ''):>15} {r.get('yp', ''):>15} {r.get('total', ''):>15}")

    return "\n".join(lines)


# -- Document Store Tools -----------------------------------------------------


@mcp.tool()
async def refresh_bddk_cache() -> str:
    """
    Force re-scrape BDDK website and update the PostgreSQL decision cache.

    Use this when you need the latest regulations/decisions from BDDK.
    Normally the server serves from PostgreSQL without hitting BDDK.
    This tool explicitly fetches fresh data from bddk.org.tr.
    """
    client = await _get_client()
    count = await client.refresh_cache()
    return f"BDDK cache refreshed: {count} decisions/regulations scraped and saved to PostgreSQL."


@mcp.tool()
async def sync_bddk_documents(
    force: bool = False,
    document_id: str | None = None,
    concurrency: int = 5,
) -> str:
    """
    Sync BDDK documents to local storage.

    Downloads documents from BDDK and mevzuat.gov.tr, extracts content to
    Markdown, and stores in PostgreSQL database for fast offline access.

    Args:
        force: Re-download all documents even if already cached
        document_id: Sync a single document by ID (e.g. "1291" or "mevzuat_42628")
        concurrency: Number of parallel downloads (default 5)
    """
    from doc_sync import DocumentSyncer

    store = await _get_doc_store()
    client = await _get_client()
    await client.ensure_cache()

    single_report = None
    sync_report = None

    async with DocumentSyncer(store, prefer_nougat=PREFER_NOUGAT) as syncer:
        if document_id:
            source_url = ""
            title = document_id
            category = ""
            for dec in client._cache:
                if dec.document_id == document_id:
                    source_url = dec.source_url
                    title = dec.title
                    category = dec.category
                    break

            result = await syncer.sync_document(
                doc_id=document_id,
                title=title,
                category=category,
                source_url=source_url,
                force=force,
            )
            status = "OK" if result.success else "FAIL"
            single_report = f"[{status}] {result.document_id}: {result.method or result.error}"
        else:
            items = [d.model_dump() for d in client._cache]
            report = await syncer.sync_all(items, concurrency=concurrency, force=force)
            sync_report = (
                f"**Sync Report**\n"
                f"  Total: {report.total}\n"
                f"  Downloaded: {report.downloaded}\n"
                f"  Skipped: {report.skipped}\n"
                f"  Failed: {report.failed}\n"
                f"  Time: {report.elapsed_seconds}s"
            )

    # Migrate documents to pgvector for semantic search
    embed_report = ""
    try:
        await _migrate_to_pgvector(store)
        vs = await _get_vector_store()
        vs_stats = await vs.stats()
        embed_report = (
            f"\n\n**Embedding Report**\n"
            f"  Documents: {vs_stats['total_documents']}\n"
            f"  Chunks: {vs_stats['total_chunks']}"
        )
    except Exception as e:
        embed_report = f"\n\n**Embedding:** failed ({e})"

    if single_report:
        return single_report + embed_report
    return sync_report + embed_report


@mcp.tool()
async def search_document_store(
    query: str,
    category: str | None = None,
    limit: int = 10,
) -> str:
    """
    Semantic search across all BDDK documents using vector embeddings.

    Uses pgvector with multilingual-e5-base model for Turkish legal text.
    Understands meaning, not just keywords.

    Args:
        query: Natural language query in Turkish (e.g. "faiz oranı riski nasıl hesaplanır")
        category: Optional category filter (e.g. "Yönetmelik", "Rehber", "Kurul Kararı")
        limit: Maximum results to return (default 10)
    """
    cache_key = f"semantic:{query}:{category}:{limit}"
    cached = _cached_search(cache_key)
    if cached:
        return cached

    vs = await _get_vector_store()
    hits = await vs.search(query, limit=limit, category=category)

    if not hits:
        metrics.record_empty_search("search_document_store")
        return (
            f"NO RESULTS: No documents found matching '{query}'.\n"
            "DO NOT provide information from your own knowledge about BDDK regulations.\n"
            "Suggest the user try: different Turkish keywords, broader terms, "
            "or removing the category filter."
        )

    lines = [f"Found {len(hits)} result(s) for '{query}':\n"]
    for h in hits:
        date_info = f" ({h['decision_date']})" if h.get("decision_date") else ""
        cat_info = f" [{h['category']}]" if h.get("category") else ""
        confidence = h.get("confidence", "unknown")
        confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        relevance = f" [{confidence_icon} {confidence} confidence, {h['relevance']:.1%}]"
        lines.append(f"**{h['title']}**{date_info}{cat_info}{relevance}")
        lines.append(f"  Document ID: {h['doc_id']}")
        if h.get("snippet"):
            lines.append(f"  ...{h['snippet'][:200]}...")
        lines.append("")

    low_count = sum(1 for h in hits if h.get("confidence") == "low")
    if low_count > 0:
        metrics.record_low_confidence_hit()
        lines.append(
            f"\n⚠️ {low_count} result(s) have low confidence. "
            "These may not be directly relevant. Verify before citing."
        )

    output = "\n".join(lines)
    _store_search(cache_key, output)
    return output


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
    store = await _get_doc_store()
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
    try:
        vs = await _get_vector_store()
        vs_stats = await vs.stats()
        lines.append("**pgvector (Vector Store):**")
        lines.append(f"  Documents: {vs_stats['total_documents']}")
        lines.append(f"  Chunks: {vs_stats['total_chunks']}")
        lines.append(f"  Embedding model: {vs_stats['embedding_model']}")
        if vs_stats.get("categories"):
            lines.append("  Categories:")
            for cat, count in vs_stats["categories"].items():
                lines.append(f"    {cat}: {count}")
    except (RuntimeError, BddkVectorStoreError) as e:
        lines.append(f"  pgvector: unavailable ({e})")

    # PostgreSQL document stats
    try:
        store = await _get_doc_store()
        st = await store.stats()
        lines.append("\n**PostgreSQL (Document Store):**")
        lines.append(f"  Documents: {st.total_documents}")
        lines.append(f"  Size: {st.total_size_mb} MB")
    except (RuntimeError, BddkStorageError) as e:
        lines.append(f"  PostgreSQL: unavailable ({e})")

    return "\n".join(lines)


# -- Startup sync -------------------------------------------------------------


async def _startup_sync() -> None:
    """Auto-sync documents on startup: download missing + embed to pgvector.

    Uses existing PostgreSQL cache — does NOT scrape BDDK for the decision list.
    Only downloads document content that is missing from the document store.
    """
    global _last_sync_time
    logger.info("Startup sync started...")
    try:
        from doc_sync import DocumentSyncer

        store = await _get_doc_store()
        client = await _get_client()
        # Cache is already loaded from PostgreSQL in initialize() — no BDDK scraping
        logger.info("Using existing cache: %d documents", len(client._cache))
        if not client._cache:
            logger.warning("Cache is empty — skipping startup sync (run refresh_bddk_cache first)")
            return

        st = await store.stats()
        cache_size = len(client._cache)

        # Phase 1: Download missing documents
        if st.total_documents < cache_size * 0.9:
            logger.info(
                "Document store incomplete (%d/%d) — downloading...",
                st.total_documents,
                cache_size,
            )
            items = [d.model_dump() for d in client._cache]
            async with DocumentSyncer(store, prefer_nougat=PREFER_NOUGAT) as syncer:
                report = await syncer.sync_all(items, concurrency=10, force=False)
            logger.info(
                "Document sync: %d downloaded, %d failed, %.1fs",
                report.downloaded,
                report.failed,
                report.elapsed_seconds,
            )
        else:
            logger.info("Document store has %d/%d documents, OK", st.total_documents, cache_size)

        # Phase 2: Migrate to pgvector
        await _migrate_to_pgvector(store)

        _last_sync_time = time.time()

    except (BddkError, RuntimeError, OSError) as e:
        logger.error("Startup sync failed: %s", e)


async def _migrate_to_pgvector(store: DocumentStore) -> None:
    """Migrate documents from document store to pgvector if needed."""
    try:
        vs = await _get_vector_store()
        vs_stats = await vs.stats()
        sqlite_stats = await store.stats()

        if vs_stats["total_documents"] >= sqlite_stats.total_documents * 0.9:
            logger.info(
                "pgvector has %d/%d documents, skipping migration",
                vs_stats["total_documents"],
                sqlite_stats.total_documents,
            )
            return

        logger.info(
            "pgvector incomplete (%d/%d) — migrating...",
            vs_stats["total_documents"],
            sqlite_stats.total_documents,
        )

        start = time.time()
        docs = await store.list_documents(limit=2000)
        migrated = 0
        total_chunks = 0

        for i, meta in enumerate(docs):
            doc_id = meta["document_id"]
            if await vs.has_document(doc_id):
                continue

            doc = await store.get_document(doc_id)
            if not doc or not doc.markdown_content:
                continue

            chunks = await vs.add_document(
                doc_id=doc.document_id,
                title=doc.title,
                content=doc.markdown_content,
                category=doc.category,
                decision_date=doc.decision_date,
                decision_number=doc.decision_number,
                source_url=doc.source_url,
            )
            total_chunks += chunks
            migrated += 1

            if (i + 1) % 100 == 0:
                logger.info("pgvector migration: %d/%d docs", i + 1, len(docs))

        elapsed = time.time() - start
        logger.info(
            "pgvector migration complete: %d docs, %d chunks, %.1fs",
            migrated,
            total_chunks,
            elapsed,
        )
    except (BddkError, RuntimeError, OSError) as e:
        logger.error("pgvector migration failed: %s", e)


@mcp.tool()
async def trigger_startup_sync() -> str:
    """
    Manually trigger document sync if auto-sync is still running or was skipped.
    Returns current sync status.
    """
    global _sync_task
    if _sync_task and not _sync_task.done():
        return "Sync is already running in background."

    store = await _get_doc_store()
    st = await store.stats()

    # Run pgvector migration if documents exist but embeddings are missing
    embed_report = ""
    try:
        await _migrate_to_pgvector(store)
        vs = await _get_vector_store()
        vs_stats = await vs.stats()
        embed_report = (
            f"\n  Vector documents: {vs_stats['total_documents']}"
            f"\n  Vector chunks: {vs_stats['total_chunks']}"
        )
    except Exception as e:
        embed_report = f"\n  Embedding migration failed: {e}"

    return f"Store has {st.total_documents} documents.{embed_report}"


# -- Health Check Tool --------------------------------------------------------


@mcp.tool()
async def health_check() -> str:
    """
    Check server health status.

    Returns uptime, cache status, store stats, and last sync time.
    """
    uptime_s = int(time.time() - _server_start_time)
    hours, remainder = divmod(uptime_s, 3600)
    minutes, seconds = divmod(remainder, 60)

    lines = ["**BDDK MCP Server Health**\n"]
    lines.append("  Status: OK")
    lines.append(f"  Uptime: {hours}h {minutes}m {seconds}s")
    lines.append(f"  Backend: PostgreSQL + pgvector")

    if _last_sync_time:
        ago = int(time.time() - _last_sync_time)
        lines.append(f"  Last sync: {ago}s ago")
    else:
        lines.append("  Last sync: never")

    # Cache status
    try:
        client = await _get_client()
        status = client.cache_status()
        lines.append(f"  Cache items: {status['total_items']}")
        lines.append(f"  Cache valid: {status['cache_valid']}")
    except (RuntimeError, BddkError):
        lines.append("  Cache: unavailable")

    # Store status
    try:
        store = await _get_doc_store()
        st = await store.stats()
        lines.append(f"  Documents: {st.total_documents}")
    except (RuntimeError, BddkStorageError):
        lines.append("  Documents: unavailable")

    sync_status = "running" if (_sync_task and not _sync_task.done()) else "idle"
    lines.append(f"  Sync status: {sync_status}")

    return "\n".join(lines)


@mcp.tool()
async def bddk_metrics() -> str:
    """
    Show server performance metrics.

    Includes request counts, average latency per tool, error rates, and cache statistics.
    """
    m = metrics.summary()

    lines = ["**BDDK MCP Server Metrics**\n"]
    lines.append(f"  Uptime: {m['uptime_seconds']}s")
    lines.append(f"  Total requests: {m['total_requests']}")
    lines.append(f"  Total errors: {m['total_errors']}")
    lines.append(f"  Cache hit rate: {m['cache_hit_rate']}%")
    lines.append(f"  Cache hits/misses: {m['cache_hits']}/{m['cache_misses']}")

    if m["tools"]:
        lines.append("\n**Per-Tool Metrics:**")
        lines.append(f"  {'Tool':<35} {'Requests':>10} {'Errors':>8} {'Avg ms':>10}")
        lines.append("  " + "-" * 65)
        for t in m["tools"]:
            lines.append(f"  {t['tool']:<35} {t['requests']:>10} {t['errors']:>8} {t['avg_latency_ms']:>10.1f}")

    return "\n".join(lines)


# -- Graceful Shutdown --------------------------------------------------------


async def _graceful_shutdown() -> None:
    """Close connections cleanly."""
    logger.info("Graceful shutdown initiated...")

    if _sync_task and not _sync_task.done():
        _sync_task.cancel()
        try:
            await _sync_task
        except asyncio.CancelledError:
            pass

    if _client is not None:
        await _client.close()
        logger.info("HTTP client closed")

    if _pool is not None:
        await _pool.close()
        logger.info("PostgreSQL pool closed")

    logger.info("Graceful shutdown complete")


# -- Entry point --------------------------------------------------------------


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
        logger.info("uvloop installed")
    except ImportError:
        pass

    _transport = os.environ.get("MCP_TRANSPORT", "stdio")
    logger.info("Transport: %s", _transport)
    logger.info("BDDK_AUTO_SYNC=%s", os.environ.get("BDDK_AUTO_SYNC", "(not set)"))
    logger.info("DATABASE_URL=%s", DATABASE_URL.split("@")[-1])

    if _transport == "streamable-http":
        import uvicorn

        app = mcp.streamable_http_app()
        port = int(os.environ.get("PORT", 8000))

        async def _run_server():
            config = uvicorn.Config(app, host="0.0.0.0", port=port)
            server = uvicorn.Server(config)

            # Seed DB from bundled JSON if tables are empty
            try:
                from seed import SEED_DIR, import_seed
                if SEED_DIR.exists():
                    result = await import_seed()
                    if not result["skipped"]:
                        logger.info(
                            "Seed import: %d cache, %d docs, %d chunks",
                            result["decision_cache"], result["documents"], result["chunks"],
                        )
                    else:
                        logger.info("DB already populated — seed import skipped")
            except Exception as e:
                logger.warning("Seed import failed (non-fatal): %s", e)

            if AUTO_SYNC:
                print("[STARTUP] launching background sync", flush=True)
                asyncio.create_task(_startup_sync())

            await server.serve()
            await _graceful_shutdown()

        asyncio.run(_run_server())
    else:
        mcp.run(transport=_transport)
