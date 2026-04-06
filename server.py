"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from client import BddkApiClient, _turkish_lower
from doc_store import DocumentStore
from data_sources import (
    fetch_announcements,
    fetch_bulletin_snapshot,
    fetch_institutions,
    fetch_weekly_bulletin,
)
from analytics import analyze_trends, build_digest, compare_metrics, check_updates
from models import BddkSearchRequest

mcp = FastMCP(
    "BDDK",
    instructions="Search and retrieve BDDK (Turkish Banking Regulation) decisions and regulations (mevzuat)",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    stateless_http=True,
)

_client: BddkApiClient | None = None
_doc_store: DocumentStore | None = None


async def _get_doc_store() -> DocumentStore:
    global _doc_store
    if _doc_store is None:
        db_path = Path(os.environ.get("BDDK_DB_PATH", Path(__file__).parent / "bddk_docs.db"))
        _doc_store = DocumentStore(db_path=db_path)
        await _doc_store.initialize()
    return _doc_store


async def _get_client() -> BddkApiClient:
    global _client
    if _client is None:
        store = await _get_doc_store()
        _client = BddkApiClient(doc_store=store)
    return _client


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
    client = await _get_client()
    request = BddkSearchRequest(
        keywords=keywords, page=page, page_size=page_size,
        category=category, date_from=date_from, date_to=date_to,
    )
    result = await client.search_decisions(request)

    if not result.decisions:
        return "No BDDK decisions found for the given keywords."

    lines = [f"Found {result.total_results} result(s) (page {result.page}):\n"]
    for d in result.decisions:
        date_info = f" ({d.decision_date} - {d.decision_number})" if d.decision_date else ""
        cat_info = f" [{d.category}]" if d.category else ""
        lines.append(f"**{d.title}**{date_info}{cat_info}")
        lines.append(f"  Document ID: {d.document_id}")
        lines.append(f"  {d.content}\n")
    return "\n".join(lines)


@mcp.tool()
async def get_bddk_document(
    document_id: str,
    page_number: int = 1,
) -> str:
    """
    Retrieve a BDDK decision document as Markdown.

    Args:
        document_id: The numeric document ID (from search results)
        page_number: Page of the markdown output (documents are split into 5000-char pages)
    """
    client = await _get_client()
    doc = await client.get_document_markdown(document_id, page_number)

    header = f"Document {doc.document_id} — Page {doc.page_number}/{doc.total_pages}\n\n"
    return header + doc.markdown_content


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
            i for i in institutions
            if kw in _turkish_lower(i["name"]) or kw in _turkish_lower(i.get("type", ""))
        ]

    if not institutions:
        return "No institutions found."

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
    client = await _get_client()
    data = await fetch_weekly_bulletin(
        client._http, metric_id, currency, days, date, column,
    )

    if "error" in data:
        return f"Error fetching bulletin: {data['error']}"

    lines = [f"**{data.get('title', 'BDDK Weekly Bulletin')}** ({data['currency']})\n"]

    dates = data.get("dates", [])
    values = data.get("values", [])

    if dates and values:
        for d, v in zip(dates[-10:], values[-10:]):
            lines.append(f"  {d}: {v}")
    else:
        lines.append("No data returned for the given parameters.")

    return "\n".join(lines)


@mcp.tool()
async def get_bddk_bulletin_snapshot() -> str:
    """
    Get the latest weekly bulletin snapshot — all metrics with current TP/YP values.

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
        lines.append(
            f"{r['row_number']:<4} {r['name']:<50} {r['tp']:>15} {r['yp']:>15} {r['metric_id']}"
        )
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
            insan kaynakları (HR), veri (data publication)
    """
    # Map category keywords to page IDs
    cat_lower = _turkish_lower(category)
    if "basın" in cat_lower or "press" in cat_lower:
        cat_id = 39
    elif "mevzuat" in cat_lower or "regul" in cat_lower:
        cat_id = 40
    elif "insan" in cat_lower or "hr" in cat_lower:
        cat_id = 41
    elif "veri" in cat_lower or "data" in cat_lower:
        cat_id = 42
    else:
        cat_id = 39

    client = await _get_client()
    announcements = await fetch_announcements(client._http, cat_id)

    if keywords:
        kw = _turkish_lower(keywords)
        announcements = [
            a for a in announcements
            if kw in _turkish_lower(a.get("title", ""))
        ]

    if not announcements:
        return "No announcements found."

    lines = [f"Found {len(announcements)} announcement(s):\n"]
    for a in announcements[:20]:  # Limit to 20
        date_info = f" ({a['date']})" if a.get("date") else ""
        lines.append(f"**{a['title']}**{date_info}")
        if a.get("url"):
            lines.append(f"  URL: {a['url']}")
        lines.append("")
    return "\n".join(lines)


# -- v4 Analytics Tools ----------------------------------------------------


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
    client = await _get_client()
    result = await analyze_trends(
        client._http, metric_id, currency, column, lookback_weeks,
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
        period: Time period — week (7 days), month (30 days), quarter (90 days)
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
            lines.append(
                f"{title:<55} {m['current']:>15,.2f} {m['wow_pct']:>+11.2f}%"
            )

    return "\n".join(lines)


@mcp.tool()
async def check_bddk_updates() -> str:
    """
    Check for new BDDK announcements since last check.

    Compares current announcements with cached state to detect new items.
    Useful for monitoring regulatory changes.
    """
    client = await _get_client()

    # Build set of known announcement URLs from previous checks
    known_urls: set[str] = set()
    if hasattr(client, "_known_announcements"):
        known_urls = client._known_announcements
    else:
        # First run — fetch and store current state as baseline
        from data_sources import fetch_announcements as _fa
        for cat_id in [39, 40]:
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

    # Update known set
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
    from data_sources import fetch_monthly_bulletin
    client = await _get_client()
    result = await fetch_monthly_bulletin(
        client._http, table_no, year, month, currency, party_code,
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
            lines.append(
                f"{r['name']:<55} {r.get('tp',''):>15} {r.get('yp',''):>15} {r.get('total',''):>15}"
            )

    return "\n".join(lines)


# -- Document Store Tools ------------------------------------------------------


@mcp.tool()
async def sync_bddk_documents(
    force: bool = False,
    document_id: str | None = None,
    concurrency: int = 5,
) -> str:
    """
    Sync BDDK documents to local storage.

    Downloads documents from BDDK and mevzuat.gov.tr, extracts content to
    Markdown, and stores in local SQLite database for fast offline access.

    Args:
        force: Re-download all documents even if already cached
        document_id: Sync a single document by ID (e.g. "1291" or "mevzuat_42628")
        concurrency: Number of parallel downloads (default 5)
    """
    from doc_sync import DocumentSyncer, SyncResult

    store = await _get_doc_store()
    client = await _get_client()
    await client.ensure_cache()

    async with DocumentSyncer(store, prefer_nougat=False) as syncer:
        if document_id:
            # Single document
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
            return f"[{status}] {result.document_id}: {result.method or result.error}"
        else:
            # Bulk sync from cache
            items = [d.model_dump() for d in client._cache]
            report = await syncer.sync_all(items, concurrency=concurrency, force=force)
            return (
                f"**Sync Report**\n"
                f"  Total: {report.total}\n"
                f"  Downloaded: {report.downloaded}\n"
                f"  Skipped: {report.skipped}\n"
                f"  Failed: {report.failed}\n"
                f"  Time: {report.elapsed_seconds}s"
            )


@mcp.tool()
async def search_document_store(
    query: str,
    category: str | None = None,
    limit: int = 20,
) -> str:
    """
    Search locally stored BDDK documents using full-text search (FTS5).

    Searches across document titles, content, and categories. Much faster
    than live web searches and works offline.

    Args:
        query: Search terms in Turkish (e.g. "sermaye yeterliliği", "faiz oranı riski")
        category: Optional category filter (e.g. "Yönetmelik", "Rehber")
        limit: Maximum results to return (default 20)
    """
    store = await _get_doc_store()
    hits = await store.search_content(query, limit=limit, category=category)

    if not hits:
        return f"No results found for '{query}' in local document store."

    lines = [f"Found {len(hits)} result(s) in local store for '{query}':\n"]
    for h in hits:
        date_info = f" ({h.decision_date})" if h.decision_date else ""
        cat_info = f" [{h.category}]" if h.category else ""
        lines.append(f"**{h.title}**{date_info}{cat_info}")
        lines.append(f"  Document ID: {h.document_id}")
        if h.snippet:
            snippet = h.snippet.replace(">>>", "**").replace("<<<", "**")
            lines.append(f"  ...{snippet}...")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def document_store_stats() -> str:
    """
    Show local document store statistics: total documents, size, categories,
    extraction methods, and documents needing refresh.
    """
    store = await _get_doc_store()
    st = await store.stats()

    lines = ["**Document Store Statistics**\n"]
    lines.append(f"  Total documents: {st.total_documents}")
    lines.append(f"  Total size: {st.total_size_mb} MB")
    lines.append(f"  Need refresh: {st.documents_needing_refresh}")

    if st.oldest_document:
        lines.append(f"  Oldest entry: {st.oldest_document}")
        lines.append(f"  Newest entry: {st.newest_document}")

    if st.categories:
        lines.append("\n**Categories:**")
        for cat, count in st.categories.items():
            lines.append(f"  {cat}: {count}")

    if st.extraction_methods:
        lines.append("\n**Extraction Methods:**")
        for m, count in st.extraction_methods.items():
            lines.append(f"  {m}: {count}")

    return "\n".join(lines)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
