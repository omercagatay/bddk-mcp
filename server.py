"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

from mcp.server.fastmcp import FastMCP

from client import BddkApiClient, _turkish_lower
from data_sources import (
    fetch_announcements,
    fetch_bulletin_snapshot,
    fetch_institutions,
    fetch_weekly_bulletin,
)
from models import BddkSearchRequest

mcp = FastMCP(
    "BDDK",
    instructions="Search and retrieve BDDK (Turkish Banking Regulation) decisions and regulations (mevzuat)",
)

_client: BddkApiClient | None = None


def _get_client() -> BddkApiClient:
    global _client
    if _client is None:
        _client = BddkApiClient()
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
    client = _get_client()
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
    client = _get_client()
    doc = await client.get_document_markdown(document_id, page_number)

    header = f"Document {doc.document_id} — Page {doc.page_number}/{doc.total_pages}\n\n"
    return header + doc.markdown_content


@mcp.tool()
async def bddk_cache_status() -> str:
    """
    Show BDDK cache statistics: total items, age, categories, and any page errors.
    """
    client = _get_client()
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
    client = _get_client()
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
    client = _get_client()
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
    client = _get_client()
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

    client = _get_client()
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


if __name__ == "__main__":
    mcp.run()
