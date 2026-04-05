"""MCP server exposing BDDK decision search and document retrieval tools."""

from mcp.server.fastmcp import FastMCP

from client import BddkApiClient
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


if __name__ == "__main__":
    mcp.run()
