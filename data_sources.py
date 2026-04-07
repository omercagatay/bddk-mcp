"""Additional BDDK data source fetchers: institutions, bulletins, announcements."""

import asyncio
import logging
import re

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BDDK_BASE_URL = "https://www.bddk.org.tr"

# Rate limiter: max 5 concurrent outbound requests to BDDK
_request_semaphore = asyncio.Semaphore(5)


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
    """Fetch institution directory from BDDK.

    Returns list of dicts with: name, website, type, subcategory, status, digital.
    """
    all_institutions: list[dict] = []

    pages = _INSTITUTION_PAGES
    if institution_type:
        pages = {pid: itype for pid, itype in _INSTITUTION_PAGES.items() if institution_type.lower() in itype.lower()}
        if not pages:
            pages = _INSTITUTION_PAGES

    for page_id, inst_type in pages.items():
        try:
            url = f"{_BDDK_BASE_URL}/Kurulus/Liste/{page_id}"
            async with _request_semaphore:
                response = await http.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            if page_id in _CARD_INSTITUTION_PAGES:
                items = _parse_card_institutions(soup, inst_type)
            else:
                items = _parse_tabpane_institutions(soup, inst_type)

            all_institutions.extend(items)
            logger.info("Parsed %d institutions from page %d (%s)", len(items), page_id, inst_type)
        except (httpx.HTTPError, httpx.TransportError, ValueError, AttributeError) as e:
            logger.error("Failed to fetch institutions page %d: %s", page_id, e)

    return all_institutions


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
        page_resp = await http.get(page_url)
        page_resp.raise_for_status()
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

        response = await http.post(
            api_url,
            data=post_data,
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Referer": page_url,
            },
        )
        response.raise_for_status()
        data = response.json()

        return {
            "title": data.get("Baslik", ""),
            "dates": data.get("XEkseni", []),
            "values": data.get("YEkseni", []),
            "currency": currency,
            "metric_id": metric_id,
        }
    except (httpx.HTTPError, httpx.TransportError, KeyError, ValueError) as e:
        logger.error("Failed to fetch weekly bulletin: %s", e)
        return {"error": str(e)}


async def fetch_bulletin_snapshot(
    http: httpx.AsyncClient,
) -> list[dict]:
    """Fetch the current weekly bulletin table data (latest snapshot).

    Returns list of dicts with: row_number, name, metric_id, tp, yp.
    """
    try:
        page_url = f"{_BDDK_BASE_URL}/bultenhaftalik"
        response = await http.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", id="Tablo")
        if not table:
            return []

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
    except (httpx.HTTPError, httpx.TransportError, ValueError, AttributeError) as e:
        logger.error("Failed to fetch bulletin snapshot: %s", e)
        return []


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
        response = await http.get(url)
        response.raise_for_status()
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
    except (httpx.HTTPError, httpx.TransportError, ValueError, AttributeError) as e:
        logger.error("Failed to fetch announcements category %d: %s", category_id, e)
        return []


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
        page_resp = await http.get(page_url)
        page_resp.raise_for_status()
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

        response = await http.post(
            api_url,
            data=post_data,
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Referer": page_url,
            },
        )
        response.raise_for_status()
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
    except (httpx.HTTPError, httpx.TransportError, KeyError, ValueError) as e:
        logger.error("Failed to fetch monthly bulletin: %s", e)
        return {"error": str(e)}
