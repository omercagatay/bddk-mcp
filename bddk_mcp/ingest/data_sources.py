"""Additional BDDK data source fetchers: institutions, bulletins, announcements."""

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from bddk_mcp.core.exceptions import BddkUpstreamError, BddkUpstreamUnreachableError
from bddk_mcp.core.outbound_http import (
    BDDK_HTTPS_HOSTS,
    OutboundHttpPolicyError,
    assert_public_https_resolution,
    bounded_request_with_retry,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InstitutionDirectory:
    """Directory rows plus how much of the directory was actually retrieved.

    A truncated directory rendered without its status reads as a complete one,
    which is the same false-completeness failure as reporting a blocked fetch
    as "no results".
    """

    institutions: list[dict]
    failed_pages: int
    attempted_pages: int

    @property
    def partial(self) -> bool:
        return self.failed_pages > 0


_BDDK_BASE_URL = "https://www.bddk.org.tr"
_MAX_BDDK_HTML_BYTES = 8 * 1024 * 1024
_MAX_BDDK_API_BYTES = 8 * 1024 * 1024

# Rate limiter: max 5 concurrent outbound requests to BDDK
_request_semaphore = asyncio.Semaphore(5)


async def _get(http: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    """Bounded GET inside the exact BDDK HTTPS boundary."""
    async with _request_semaphore:
        return await bounded_request_with_retry(
            http,
            "GET",
            url,
            base_url=f"{_BDDK_BASE_URL}/",
            allowed_hosts=BDDK_HTTPS_HOSTS,
            boundary_name="BDDK",
            max_bytes=_MAX_BDDK_HTML_BYTES,
            resolve=assert_public_https_resolution,
            **kwargs,
        )


async def _post(http: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    """Bounded POST inside the exact BDDK HTTPS boundary."""
    async with _request_semaphore:
        return await bounded_request_with_retry(
            http,
            "POST",
            url,
            base_url=f"{_BDDK_BASE_URL}/",
            allowed_hosts=BDDK_HTTPS_HOSTS,
            boundary_name="BDDK",
            max_bytes=_MAX_BDDK_API_BYTES,
            resolve=assert_public_https_resolution,
            **kwargs,
        )


def _format_number(val) -> str:
    """Format a numeric value with thousands separators."""
    if isinstance(val, (int, float)):
        return f"{val:,.0f}" if val == int(val) else f"{val:,.2f}"
    return str(val)


# Institution directory page IDs and their types
_INSTITUTION_PAGES = {
    77: "Banka",
    78: "Finansal Kiralama Şirketi",
    79: "Faktoring Şirketi",
    80: "Finansman Şirketi",
    82: "Varlık Yönetim Şirketi",
}

# Pages that use card/accordion layout (page 77)
_CARD_INSTITUTION_PAGES = {77}

# Announcement category page IDs
_ANNOUNCEMENT_PAGES = {
    39: "Basın Duyurusu",
    40: "Mevzuat Duyurusu",
    41: "İnsan Kaynakları Duyurusu",
    42: "Veri Yayımlama Duyurusu",
    48: "Kuruluş Duyurusu",
}


# -- Institution Directory ------------------------------------------------


def _parse_card_institutions(soup: BeautifulSoup, inst_type: str) -> list[dict]:
    """Parse page 77 (Banka) — uses div.card accordion structure."""
    results: list[dict] = []
    for card in soup.find_all("div", class_="card"):
        h5 = card.find("h5")
        if not h5:
            continue

        raw_header = h5.get_text(strip=True)
        subcategory = re.sub(r"\s*\(\d+\)\s*$", "", raw_header).strip()

        status = "Aktif"
        lower_sub = subcategory.lower()
        if "iptal" in lower_sub or "mülga" in lower_sub or "tmsf" in lower_sub:
            status = "İptal Edilmiş"

        body = card.find("div", class_="card-body")
        if not body:
            body = card.find("div", class_="collapse")
        if not body:
            continue

        for li in body.find_all("li"):
            raw_text = li.get_text(strip=True)
            if not raw_text or len(raw_text) < 5:
                continue

            website = ""
            link = li.find("a", href=lambda h: h and h.startswith("http"))
            if link:
                website = link.get("href", "")

            name = raw_text
            if link:
                name = name.replace(link.get_text(strip=True), "")
            name = re.sub(r"\s*Detay\s*$", "", name).strip()
            name = re.sub(r"^\d+\.\s*", "", name).strip()
            is_digital = "Dijital Banka" in name
            name = name.replace("Dijital Banka", "").strip()

            if not name or len(name) < 3:
                continue

            results.append(
                {
                    "name": name,
                    "website": website,
                    "type": inst_type,
                    "subcategory": subcategory,
                    "status": status,
                    "digital": is_digital,
                }
            )
    return results


def _parse_tabpane_institutions(soup: BeautifulSoup, inst_type: str) -> list[dict]:
    """Parse pages 78-82 — uses div.tab-pane with li.row items.

    Structure:
        div.tab-pane#faaliyette > li.row > div.baslikContainer (name)
        div.tab-pane#kapanan > li.row > div.baslikContainer (closed)
    """
    results: list[dict] = []
    for pane in soup.find_all("div", class_="tab-pane"):
        pane_id = pane.get("id", "")
        status = "Aktif" if pane_id == "faaliyette" else "İptal Edilmiş"

        for li in pane.find_all("li", class_="row"):
            name_div = li.find("div", class_="baslikContainer")
            if not name_div:
                continue

            name = name_div.get_text(strip=True)
            name = re.sub(r"^\d+\.\s*", "", name).strip()

            if not name or len(name) < 3:
                continue

            website = ""
            web_div = li.find("div", class_="webAdresiContainer")
            if web_div:
                link = web_div.find("a", href=lambda h: h and h.startswith("http"))
                if link:
                    website = link.get("href", "")

            results.append(
                {
                    "name": name,
                    "website": website,
                    "type": inst_type,
                    "subcategory": inst_type,
                    "status": status,
                    "digital": False,
                }
            )
    return results


async def fetch_institutions(
    http: httpx.AsyncClient,
    institution_type: str | None = None,
) -> list[dict]:
    """Fetch the institution directory, discarding partial-fetch status.

    Prefer :func:`fetch_institutions_with_status` where the caller renders the
    result to a user: a silently truncated directory reads as a complete one.
    """

    return (await fetch_institutions_with_status(http, institution_type)).institutions


async def fetch_institutions_with_status(
    http: httpx.AsyncClient,
    institution_type: str | None = None,
) -> InstitutionDirectory:
    """Fetch institution directory from BDDK.

    Returns list of dicts with: name, website, type, subcategory, status, digital.
    """
    all_institutions: list[dict] = []

    pages = _INSTITUTION_PAGES
    if institution_type:
        pages = {pid: itype for pid, itype in _INSTITUTION_PAGES.items() if institution_type.lower() in itype.lower()}
        if not pages:
            pages = _INSTITUTION_PAGES

    failed_pages: list[int] = []
    unreachable = False
    for page_id, inst_type in pages.items():
        try:
            url = f"{_BDDK_BASE_URL}/Kurulus/Liste/{page_id}"
            response = await _get(http, url)
            soup = BeautifulSoup(response.text, "html.parser")

            if page_id in _CARD_INSTITUTION_PAGES:
                items = _parse_card_institutions(soup, inst_type)
            else:
                items = _parse_tabpane_institutions(soup, inst_type)

            all_institutions.extend(items)
            logger.info("Parsed %d institutions from page %d (%s)", len(items), page_id, inst_type)
        except (httpx.TransportError, OutboundHttpPolicyError) as exc:
            # Connect-class failure: the remaining pages share the same host, so
            # retrying them would only repeat the same slow failure serially.
            failed_pages.append(page_id)
            unreachable = True
            logger.error(
                "Failed to fetch institutions page %d; upstream unreachable, aborting remaining pages",
                page_id,
                extra={"error_type": type(exc).__name__},
            )
            break
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            failed_pages.append(page_id)
            logger.error(
                "Failed to fetch institutions page %d",
                page_id,
                extra={"error_type": type(exc).__name__},
            )

    if failed_pages and not all_institutions:
        # An empty result caused by fetch failure must never masquerade as
        # "no such institutions exist"; surface a retryable upstream error.
        if unreachable:
            raise BddkUpstreamUnreachableError(
                "BDDK institution directory could not be fetched (upstream unreachable)."
            )
        raise BddkUpstreamError(
            f"BDDK institution directory could not be fetched ({len(failed_pages)} of {len(pages)} pages failed)."
        )

    if failed_pages:
        logger.warning(
            "Institution directory fetched partially: %d of %d pages failed",
            len(failed_pages),
            len(pages),
        )

    return InstitutionDirectory(
        institutions=all_institutions,
        failed_pages=len(failed_pages),
        attempted_pages=len(pages),
    )


# -- Weekly Bulletin Data -------------------------------------------------


async def fetch_weekly_bulletin(
    http: httpx.AsyncClient,
    metric_id: str = "1.0.1",
    currency: str = "TRY",
    days: int = 90,
    date: str = "",
    column: str = "1",
) -> dict:
    """Fetch weekly bulletin data from BDDK.

    First fetches the bulletin page to get session cookies and CSRF token,
    then calls the JSON API endpoint.

    Args:
        metric_id: Metric ID (e.g. '1.0.1'=total loans, '1.0.2'=consumer loans)
        currency: TRY or USD
        days: Number of days of history
        date: Specific date (DD.MM.YYYY), empty for latest
        column: Column number (1=TP, 2=YP, 3=Toplam)

    Returns dict with: title, dates, values, currency, metric_id.
    """
    try:
        # Step 1: Visit the page to get session cookies and CSRF token
        page_url = f"{_BDDK_BASE_URL}/bultenhaftalik"
        page_resp = await _get(http, page_url)
        soup = BeautifulSoup(page_resp.text, "html.parser")

        token_input = soup.find("input", {"name": "__RequestVerificationToken"})
        token = token_input["value"] if token_input else ""

        # If no date provided, extract default from page JS
        # JS uses mixed quoting: "tarih": '27.03.2026'
        if not date:
            date_match = re.search(r"""["']tarih["']\s*:\s*['"]([^'"]+)['"]""", page_resp.text)
            if date_match:
                date = date_match.group(1)

        # Step 2: Call the API with proper headers and CSRF token
        api_url = f"{_BDDK_BASE_URL}/BultenHaftalik/tr/Home/KiyaslamaJsonGetir"

        post_data = {
            "dil": "tr",
            "tarih": date,
            "id": metric_id,
            "parabirimi": currency,
            "sutun": column,
            "tarafKodu": "10001",
            "gun": str(days),
        }
        if token:
            post_data["__RequestVerificationToken"] = token

        response = await _post(
            http,
            api_url,
            data=post_data,
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Referer": page_url,
            },
        )
        data = response.json()

        return {
            "title": data.get("Baslik", ""),
            "dates": data.get("XEkseni", []),
            "values": data.get("YEkseni", []),
            "currency": currency,
            "metric_id": metric_id,
        }
    except (httpx.HTTPError, httpx.TransportError, OutboundHttpPolicyError, KeyError, ValueError) as exc:
        logger.error("Failed to fetch weekly bulletin", extra={"error_type": type(exc).__name__})
        return {"error": "Approved BDDK upstream request failed."}


async def fetch_bulletin_snapshot(
    http: httpx.AsyncClient,
) -> list[dict]:
    """Fetch the current weekly bulletin table data (latest snapshot).

    Returns list of dicts with: row_number, name, metric_id, tp, yp.
    """
    try:
        page_url = f"{_BDDK_BASE_URL}/bultenhaftalik"
        response = await _get(http, page_url)
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", id="Tablo")
        if not table:
            raise BddkUpstreamError(
                "BDDK weekly bulletin page did not contain the expected data table; "
                "the upstream layout may have changed."
            )

        rows: list[dict] = []
        for tr in table.find_all("tr"):
            match = re.search(r"ShowModalGraph\('([^']+)'", str(tr))
            if not match:
                continue
            metric_id = match.group(1)
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            rows.append(
                {
                    "row_number": tds[0].get_text(strip=True),
                    "name": tds[1].get_text(strip=True),
                    "metric_id": metric_id,
                    "tp": tds[2].get_text(strip=True),
                    "yp": tds[3].get_text(strip=True),
                }
            )
        return rows
    except (httpx.TransportError, OutboundHttpPolicyError) as exc:
        logger.error(
            "Failed to fetch bulletin snapshot; upstream unreachable", extra={"error_type": type(exc).__name__}
        )
        raise BddkUpstreamUnreachableError(
            "BDDK weekly bulletin snapshot could not be fetched (upstream unreachable)."
        ) from exc
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        logger.error("Failed to fetch bulletin snapshot", extra={"error_type": type(exc).__name__})
        raise BddkUpstreamError(
            "BDDK weekly bulletin snapshot could not be fetched (upstream request failed)."
        ) from exc


# -- Announcements -------------------------------------------------------


async def fetch_announcements(
    http: httpx.AsyncClient,
    category_id: int = 39,
) -> list[dict]:
    """Fetch announcements from BDDK.

    Args:
        category_id: 39=press, 40=regulation, 41=HR, 42=data, 48=institution

    Returns list of dicts with: title, date, url, category.
    """
    url = f"{_BDDK_BASE_URL}/Duyuru/Liste/{category_id}"
    category_name = _ANNOUNCEMENT_PAGES.get(category_id, f"Duyuru ({category_id})")

    try:
        response = await _get(http, url)
        soup = BeautifulSoup(response.text, "html.parser")

        announcements: list[dict] = []

        for link in soup.find_all("a", href=re.compile(r"/Duyuru/Detay/\d+")):
            href = link.get("href", "")
            full_url = f"{_BDDK_BASE_URL}{href}"

            date_span = link.find("span", class_="gorunenTarih")
            date = date_span.get_text(strip=True) if date_span else ""

            text_span = link.find("span", class_="text")
            if text_span:
                title = text_span.get_text(strip=True)
                if date:
                    title = title.replace(date, "").strip()
            else:
                title = link.get_text(strip=True)
                if date:
                    title = title.replace(date, "").strip()

            if not title:
                continue

            announcements.append(
                {
                    "title": title,
                    "date": date,
                    "url": full_url,
                    "category": category_name,
                }
            )

        logger.info("Parsed %d announcements from category %d", len(announcements), category_id)
        return announcements
    except (httpx.TransportError, OutboundHttpPolicyError) as exc:
        logger.error(
            "Failed to fetch announcements category %d; upstream unreachable",
            category_id,
            extra={"error_type": type(exc).__name__},
        )
        raise BddkUpstreamUnreachableError(
            f"BDDK announcements ({category_name}) could not be fetched (upstream unreachable)."
        ) from exc
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        logger.error(
            "Failed to fetch announcements category %d",
            category_id,
            extra={"error_type": type(exc).__name__},
        )
        raise BddkUpstreamError(
            f"BDDK announcements ({category_name}) could not be fetched (upstream request failed)."
        ) from exc


# -- Monthly Bulletin Data -------------------------------------------------


async def fetch_monthly_bulletin(
    http: httpx.AsyncClient,
    table_no: int = 1,
    year: int = 2025,
    month: int = 12,
    currency: str = "TL",
    party_code: str = "10001",
) -> dict:
    """Fetch BDDK monthly banking sector statistics.

    Uses the same AJAX pattern as the weekly bulletin.

    Args:
        table_no: Table number (1-17)
        year: Year
        month: Month (1-12)
        currency: TL or USD
        party_code: Bank group code (10001=sector total)

    Returns dict with: title, rows [{name, value}], period.
    """
    try:
        page_url = f"{_BDDK_BASE_URL}/BultenAylik"
        page_resp = await _get(http, page_url)
        soup = BeautifulSoup(page_resp.text, "html.parser")

        token_input = soup.find("input", {"name": "__RequestVerificationToken"})
        token = token_input["value"] if token_input else ""

        api_url = f"{_BDDK_BASE_URL}/BultenAylik/tr/Home/BasitRaporGetir"
        post_data: dict = {
            "tabloNo": str(table_no),
            "yil": str(year),
            "ay": str(month),
            "paraBirimi": currency,
            "taraf[0]": party_code,
        }
        if token:
            post_data["__RequestVerificationToken"] = token

        response = await _post(
            http,
            api_url,
            data=post_data,
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Referer": page_url,
            },
        )
        result = response.json()

        # Response: {success, Json: {data: {rows: [{cell: [group, idx, name, font, tp, yp, total]}]}}}
        js = result.get("Json", {})
        caption = js.get("caption", f"Tablo {table_no}")
        raw_rows = js.get("data", {}).get("rows", [])

        rows: list[dict] = []
        for r in raw_rows:
            cell = r.get("cell", [])
            if len(cell) >= 7:
                rows.append(
                    {
                        "name": str(cell[2]),
                        "tp": _format_number(cell[4]),
                        "yp": _format_number(cell[5]),
                        "total": _format_number(cell[6]),
                    }
                )

        return {
            "title": caption,
            "rows": rows,
            "period": f"{month}/{year}",
            "currency": currency,
        }
    except (httpx.HTTPError, httpx.TransportError, OutboundHttpPolicyError, KeyError, ValueError) as exc:
        logger.error("Failed to fetch monthly bulletin", extra={"error_type": type(exc).__name__})
        return {"error": "Approved BDDK upstream request failed."}
