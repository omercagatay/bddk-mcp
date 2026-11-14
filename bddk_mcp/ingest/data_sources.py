"""Additional BDDK data source fetchers: institutions, bulletins, announcements."""

import asyncio
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone

import httpx
from bs4 import BeautifulSoup

from bddk_mcp.core.exceptions import BddkUpstreamError, BddkUpstreamUnreachableError
from bddk_mcp.core.outbound_http import (
    BDDK_HTTPS_HOSTS,
    OutboundHttpPolicyError,
    assert_public_https_resolution,
    bounded_request_with_retry,
)
from bddk_mcp.store.legal_ref import turkish_casefold

logger = logging.getLogger(__name__)

# Turkey has used UTC+3 all year since 2016. A fixed offset avoids a time-zone
# database dependency (Windows hosts have none without an extra package).
ISTANBUL_TZ = timezone(timedelta(hours=3), "Europe/Istanbul")


def istanbul_now() -> datetime:
    """Current time in Istanbul (UTC+3), whatever the host time zone is."""
    return datetime.now(UTC).astimezone(ISTANBUL_TZ)


def istanbul_today(clock: Callable[[], datetime] | None = None) -> date:
    """The Istanbul calendar date; ``clock`` (aware or naive Istanbul time) is injectable for tests."""
    now = (clock or istanbul_now)()
    if now.tzinfo is not None:
        now = now.astimezone(ISTANBUL_TZ)
    return now.date()


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
    """Render one monthly-table cell in a single explicit format.

    A JSON number uses an en-US style: comma thousands separator, dot decimal
    mark, no decimals for whole values and two decimals otherwise (1500 ->
    "1,500", 1234.5 -> "1,234.50"). A missing observation (None, an empty
    string, a non-finite number) is "-".

    Text from BDDK passes through exactly as sent, numeric-looking or not: a
    string such as "1.234" or "1.500" does not say whether its dot is a
    thousands separator or a decimal mark, and guessing would print a figure
    BDDK never published. The tool states this next to the table.
    """
    if val is None:
        return "-"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return "-"
        return f"{val:,.0f}" if val == int(val) else f"{val:,.2f}"
    text = str(val).strip()
    return text or "-"


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
        # Turkish rules: "İPTAL"/"İptal" must fold to "iptal"; ASCII "IPTAL" is
        # accepted too (dotless "ı" is mapped back to "i" for the comparison).
        lower_sub = turkish_casefold(subcategory).replace("ı", "i")
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
            # The live page writes the marker as a "(Dijital Banka)" suffix; older
            # pages put "Dijital Banka" before the name. Remove it with its brackets.
            is_digital = "Dijital Banka" in name
            name = re.sub(r"\s*\(?\s*Dijital Banka\s*\)?\s*", " ", name).strip()

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

# Commonly asked weekly metric ids with BDDK's own row label (the Krediler table
# of /bultenhaftalik, or the series title for ids from the other weekly tables)
# and, for a sub-row, the row it belongs to. Checked against captured BDDK pages
# in tests/fixtures/bddk_weekly_page.html and bddk_weekly_series.json.
WEEKLY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("1.0.1", "Toplam Krediler (2+10)", ""),
    ("1.0.2", "Tüketici Kredileri ve Bireysel Kredi Kartları (3+7)", ""),
    ("1.0.3", "Tüketici Kredileri (4+5+6)", ""),
    ("1.0.4", "a) Konut", "Tüketici Kredileri altında"),
    ("1.0.8", "Bireysel Kredi Kartları (8+9)", ""),
    ("1.0.10", "b) Taksitsiz", "Bireysel Kredi Kartları altında"),
    ("1.0.12", "Ticari ve Diğer Krediler", ""),
    ("1.0.11", "KOBİ Kredileri (Bilgi)", ""),
    ("2.0.1", "Takipteki Alacaklar (2+8)", "haftalık Takipteki Alacaklar tablosu"),
    ("4.0.1", "Mevduat (Katılım Fonu) (2+5+8)", "haftalık Mevduat tablosu"),
)


def weekly_metric_guide(indent: str) -> str:
    """The WEEKLY_METRICS key as description lines, one id per line."""
    lines = [f"{indent}{mid}={label}" + (f" ({note})" if note else "") for mid, label, note in WEEKLY_METRICS]
    return ",\n".join(lines)


# BDDK answers an id it does not know with this chart title and empty series.
_WEEKLY_PLACEHOLDER_TITLE = "Grafik Başlığı"
UNKNOWN_METRIC = "unknown_metric"
# BDDK ignores the requested window ("gun") and serves about the last 13 weeks.
WEEKLY_SERIES_LIMIT_NOTE = "BDDK haftalık bültende yalnızca son yaklaşık 13 haftayı döndürür"
WEEKLY_NUMBER_FORMAT_NOTE = "Sayı biçimi: binlik ayırıcı virgül, ondalık nokta (1,234.50)."
_TURKISH_NUMBER = re.compile(r"^-?\d{1,3}(?:\.\d{3})*(?:,(\d+))?$")


def format_bulletin_number(value: float, decimals: int = 2) -> str:
    """The one number format of the weekly tools: comma thousands, dot decimals."""
    return f"{value:,.{decimals}f}"


def format_snapshot_cell(text: str) -> str:
    """Re-render a weekly-page cell such as "18.523.337" or "1.234,56" in the weekly format.

    The page writes numbers in Turkish style (dot thousands, comma decimals), as
    its Excel copy of the same table confirms ("18.523.337,66"). Only text that
    matches that pattern exactly is converted, keeping the decimals it shows;
    anything else passes through unchanged.
    """
    raw = text.strip()
    match = _TURKISH_NUMBER.match(raw)
    if not match:
        return raw
    decimals = len(match.group(1) or "")
    return format_bulletin_number(float(raw.replace(".", "").replace(",", ".")), decimals)


def _weekly_page_header(soup: BeautifulSoup) -> tuple[str, str]:
    """The bulletin date and unit the weekly page states ("Tarih: ... | Birim: ...")."""
    for h5 in soup.find_all("h5"):
        text = h5.get_text(" ", strip=True)
        if "Tarih:" in text and "Birim:" in text:
            bold = [b.get_text(" ", strip=True) for b in h5.find_all("b")]
            if len(bold) >= 2:
                return " ".join(bold[0].split()), " ".join(bold[1].split())
    return "", ""


def _parse_weekly_date(text: str) -> date | None:
    try:
        return datetime.strptime(text.strip(), "%d.%m.%Y").date()
    except ValueError:
        return None


def _trim_to_window(dates: list[str], values: list[float], days: int) -> tuple[list[str], list[float], bool | None]:
    """Keep the observations within ``days`` of the latest one.

    Returns the kept series and whether BDDK's series reached back to the start of
    the window: False when an observation one step before the oldest returned one
    would still have fallen inside it. None when the dates do not parse (the
    series is then kept as received).
    """
    parsed = [_parse_weekly_date(d) for d in dates]
    if not parsed or any(p is None for p in parsed):
        return dates, values, None
    cutoff = parsed[-1] - timedelta(days=days)
    gaps = [(b - a).days for a, b in zip(parsed, parsed[1:], strict=False) if b > a]
    step = min(gaps) if gaps else 7
    window_met = parsed[0] - timedelta(days=step) < cutoff
    kept = [i for i, p in enumerate(parsed) if p >= cutoff]
    return [dates[i] for i in kept], [values[i] for i in kept], window_met


def _ascending_by_date(dates: list[str], values: list[float]) -> tuple[list[str], list[float]]:
    """Order observations oldest first when every date parses as DD.MM.YYYY.

    The consumers take the last point as current and the one before as previous,
    so the upstream order is not trusted. If any date does not parse the series
    is returned as received.
    """
    try:
        parsed = [datetime.strptime(d.strip(), "%d.%m.%Y") for d in dates]
    except ValueError:
        return dates, values
    if all(a <= b for a, b in zip(parsed, parsed[1:], strict=False)):
        return dates, values
    order = sorted(range(len(parsed)), key=lambda i: parsed[i])
    return [dates[i] for i in order], [values[i] for i in order]


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

    Returns dict with: title, dates, values, currency, metric_id, unit (the
    page's "Birim" for TRY; empty for USD, which the page does not state),
    bulletin_date (the page's "Tarih"), requested_days, returned_observations,
    returned_first_date, trimmed_observations and window_met. The series is
    oldest first (re-ordered when the dates parse and arrive out of order).

    BDDK ignores the requested window and returns about the last 13 weekly
    observations, so the series is trimmed here to the observations within
    ``days`` of the latest one; window_met is False when BDDK's series did not
    reach back to the start of the requested window (None when the dates do
    not parse). An empty series from BDDK comes back as empty lists, not as an
    error; BDDK's placeholder answer for an unknown id is an error with reason
    "unknown_metric".
    """
    try:
        # Step 1: Visit the page to get session cookies and CSRF token
        page_url = f"{_BDDK_BASE_URL}/bultenhaftalik"
        page_resp = await _get(http, page_url)
        soup = BeautifulSoup(page_resp.text, "html.parser")

        token_input = soup.find("input", {"name": "__RequestVerificationToken"})
        token = token_input["value"] if token_input else ""
        bulletin_date, page_unit = _weekly_page_header(soup)
        # The page states the unit of its TL view only.
        unit = page_unit if currency == "TRY" else ""

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
        if not isinstance(data, dict):
            raise ValueError("Invalid bulletin response")
        dates, values = data.get("XEkseni"), data.get("YEkseni")
        if isinstance(dates, list) and isinstance(values, list) and not dates and not values:
            title = str(data.get("Baslik") or "").strip()
            if not title or title == _WEEKLY_PLACEHOLDER_TITLE:
                # BDDK's placeholder chart: it does not know this metric id.
                return {
                    "error": f"BDDK does not know the weekly metric id {metric_id}.",
                    "reason": UNKNOWN_METRIC,
                    "metric_id": metric_id,
                }
            # Both series present and empty: BDDK answered "no observations",
            # which is not an upstream failure.
            return {
                "title": data.get("Baslik", ""),
                "dates": [],
                "values": [],
                "currency": currency,
                "metric_id": metric_id,
                "unit": unit,
                "bulletin_date": bulletin_date,
                "requested_days": days,
                "returned_observations": 0,
                "returned_first_date": "",
                "trimmed_observations": 0,
                "window_met": None,
            }
        if (
            not isinstance(dates, list)
            or not isinstance(values, list)
            or not values
            or len(dates) != len(values)
            or any(not isinstance(date, str) or not date.strip() for date in dates)
            or any(isinstance(value, bool) or not isinstance(value, (int, float, str)) for value in values)
        ):
            raise ValueError("Missing or misaligned bulletin observations")
        values = [float(value) for value in values]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite bulletin observations")
        dates, values = _ascending_by_date(dates, values)
        returned, returned_first = len(dates), dates[0]
        dates, values, window_met = _trim_to_window(dates, values, days)

        return {
            "title": data.get("Baslik", ""),
            "dates": dates,
            "values": values,
            "currency": currency,
            "metric_id": metric_id,
            "unit": unit,
            "bulletin_date": bulletin_date,
            "requested_days": days,
            "returned_observations": returned,
            "returned_first_date": returned_first,
            "trimmed_observations": returned - len(dates),
            "window_met": window_met,
        }
    except (httpx.HTTPError, httpx.TransportError, OutboundHttpPolicyError, KeyError, ValueError, OverflowError) as exc:
        logger.error("Failed to fetch weekly bulletin", extra={"error_type": type(exc).__name__})
        return {"error": "Approved BDDK upstream request failed."}


async def fetch_bulletin_snapshot(
    http: httpx.AsyncClient,
) -> list[dict]:
    """Fetch the current weekly bulletin table data (latest snapshot).

    Returns list of dicts with: row_number, name, metric_id, tp, yp.
    """
    return (await fetch_bulletin_snapshot_table(http))["rows"]


async def fetch_bulletin_snapshot_table(
    http: httpx.AsyncClient,
) -> dict:
    """Fetch the table the weekly bulletin page shows, with what the page says about it.

    The /bultenhaftalik page carries one table, the Krediler table of the sector
    (the other weekly tables need separate requests and are not fetched).
    Returns dict with: heading (e.g. "Krediler"), party (e.g. "Sektör"),
    bulletin_date (the page's "Tarih"), unit (the page's "Birim") and rows
    [{row_number, name, metric_id, tp, yp}], cells as the page writes them.
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
        heading_tag = table.find_previous("h4")
        party_tag = heading_tag.find_next("h5") if heading_tag else None
        bulletin_date, unit = _weekly_page_header(soup)
        return {
            "heading": " ".join(heading_tag.get_text(" ", strip=True).split()) if heading_tag else "",
            "party": " ".join(party_tag.get_text(" ", strip=True).split()) if party_tag else "",
            "bulletin_date": bulletin_date,
            "unit": unit,
            "rows": rows,
        }
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


def recent_complete_months(count: int, clock: Callable[[], datetime] | None = None) -> list[tuple[int, int]]:
    """Return the ``count`` most recent complete calendar months as (year, month), newest first.

    "Today" is the Istanbul date (see ``istanbul_today``), not the host's.
    """
    first_day = istanbul_today(clock).replace(day=1)
    months: list[tuple[int, int]] = []
    for _ in range(count):
        first_day = (first_day - timedelta(days=1)).replace(day=1)
        months.append((first_day.year, first_day.month))
    return months


# Cells 0-3 of a monthly row are [group, index, name, font]; the value cells follow.
_MONTHLY_LABEL_CELLS = 4
MONTHLY_CLASSIC_COLUMNS = ("TP", "YP", "Toplam")
MONTHLY_SHAPE_ERROR = "unexpected_shape"
MONTHLY_REFUSED = "bddk_refused"
# BDDK's footnote (Json.uyari) is an HTML string whose only markup is <br/>.
_MONTHLY_FOOTNOTE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _is_monthly_not_offered(message: str) -> bool:
    # Only "not offered" says these parameters can never be served; any other BDDK error may be transient.
    return "sunulmamaktad" in turkish_casefold(message)


class _MonthlyShapeError(ValueError):
    """BDDK answered, but the rows do not match the column list it sent."""


async def fetch_monthly_bulletin(
    http: httpx.AsyncClient,
    table_no: int,
    year: int,
    month: int,
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

    Returns dict with: title (BDDK's caption), columns (the value-column labels
    from BDDK's ``Json.colNames[4:]``), rows [{name, values}], period, currency,
    and footnote when BDDK's ``Json.uyari`` is a non-empty string (its <br/>
    breaks and runs of whitespace collapsed to single spaces, otherwise as sent).
    ``values`` holds one cell per label, in BDDK's order. A table whose value
    columns are exactly TP, YP, Toplam also carries tp/yp/total on each row.
    Numeric cells are rendered en-US style by ``_format_number`` ("1,500",
    "1,234.50"); a missing cell is "-". A response without a ``rows`` list is
    an error. A row whose cell count differs from ``len(colNames)``, or rows
    that are not the classic 7-cell layout when colNames is missing, are an
    upstream shape error: {"error": ..., "reason": "unexpected_shape"}, which
    retrying does not fix.
    BDDK refuses a period it holds no data for (typically a month it has not
    published yet) with success=false and a "... veri bulunma..." message; that
    is an answer, so rows is empty and the message is passed on as notice.
    BDDK saying the requested data is not offered (seen live: "Bu tabloya
    ilişkin USD cinsinden veri sunulmamaktadır.") is also its answer:
    {"error": ..., "reason": "bddk_refused", "notice": <BDDK's message>}, which
    retrying does not change. Any other non-empty BDDK message is an upstream
    error that carries the message: {"error": ..., "notice": <BDDK's message>}.
    Returns {"error": ...} when the request fails, BDDK refuses without a
    message, or the response is not the expected object.
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

        # Response: {success, Json: {caption, colNames, data: {rows: [{cell: [group, idx, name, font, values...]}]}}}
        if not isinstance(result, dict):
            raise ValueError("Unexpected monthly bulletin response")
        if result.get("success") is False:
            # Seen live: "2026 yılının en son 9 ayına ait veri bulunmaktadır!" and
            # "2031 yılına ait veri bulunmamaktadır!". Any other answer is not
            # evidence of missing data: "not offered" is BDDK's refusal of these
            # parameters, another message is BDDK's own error, and no message is
            # a failed call.
            notice = str(result.get("error") or "")
            if "veri bulunma" not in notice.lower():
                if not notice.strip():
                    raise ValueError("Unexpected monthly bulletin response")
                if _is_monthly_not_offered(notice):
                    return {
                        "error": "BDDK refused this monthly bulletin request.",
                        "reason": MONTHLY_REFUSED,
                        "notice": notice.strip()[:500],
                    }
                return {
                    "error": "BDDK answered the monthly bulletin request with an error of its own.",
                    "notice": notice.strip()[:500],
                }
            return {
                "title": f"Tablo {table_no}",
                "rows": [],
                "period": f"{month}/{year}",
                "currency": currency,
                "notice": notice[:200],
            }
        if not isinstance(result.get("Json"), dict):
            raise ValueError("Unexpected monthly bulletin response")
        js = result["Json"]
        caption = js.get("caption", f"Tablo {table_no}")
        data = js.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("Unexpected monthly bulletin response")
        raw_rows = data.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError("Monthly bulletin response has no rows list")

        # The value columns are BDDK's own: colNames is [group, idx, name, font,
        # value columns...] and every row has one cell per colNames entry (1, 3,
        # 6 or 9 value columns, depending on the table). Without usable colNames
        # only the classic [group, idx, name, font, TP, YP, Toplam] row is read.
        col_names = js.get("colNames")
        if (
            isinstance(col_names, list)
            and len(col_names) > _MONTHLY_LABEL_CELLS
            and all(isinstance(name, str) for name in col_names)
        ):
            columns = [
                " ".join(name.split()) or f"Sütun {i}"
                for i, name in enumerate(col_names[_MONTHLY_LABEL_CELLS:], start=1)
            ]
            width = len(col_names)
        else:
            columns = list(MONTHLY_CLASSIC_COLUMNS)
            width = _MONTHLY_LABEL_CELLS + len(MONTHLY_CLASSIC_COLUMNS)
        classic = tuple(columns) == MONTHLY_CLASSIC_COLUMNS

        # An unexpected row shape is an upstream error, never "no data": only an
        # empty rows list (or BDDK's own refusal above) means no data.
        rows: list[dict] = []
        for r in raw_rows:
            cell = r.get("cell") if isinstance(r, dict) else None
            if not isinstance(cell, list) or len(cell) != width:
                raise _MonthlyShapeError("Monthly bulletin row does not match its column list")
            values = [_format_number(v) for v in cell[_MONTHLY_LABEL_CELLS:]]
            row: dict = {"name": str(cell[2]), "values": values}
            if classic:
                row.update({"tp": values[0], "yp": values[1], "total": values[2]})
            rows.append(row)

        answer: dict = {
            "title": caption,
            "columns": columns,
            "rows": rows,
            "period": f"{month}/{year}",
            "currency": currency,
        }
        uyari = js.get("uyari")
        if isinstance(uyari, str):
            footnote = " ".join(_MONTHLY_FOOTNOTE_BREAK.sub(" ", uyari).split())
            if footnote:
                answer["footnote"] = footnote
        return answer
    except _MonthlyShapeError as exc:
        logger.error("Monthly bulletin response not understood", extra={"error_type": type(exc).__name__})
        return {"error": "BDDK monthly bulletin response format was not understood.", "reason": MONTHLY_SHAPE_ERROR}
    except (
        httpx.HTTPError,
        httpx.TransportError,
        OutboundHttpPolicyError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.error("Failed to fetch monthly bulletin", extra={"error_type": type(exc).__name__})
        return {"error": "Approved BDDK upstream request failed."}
