"""Tests for data_sources.py — institution, bulletin, and announcement parsers."""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from bs4 import BeautifulSoup

from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.ingest.data_sources import (
    _parse_card_institutions,
    _parse_tabpane_institutions,
    fetch_announcements,
    fetch_bulletin_snapshot,
    fetch_institutions_with_status,
    fetch_weekly_bulletin,
)

# -- Institution parsing tests --


CARD_HTML = """
<div class="card">
  <h5>Mevduat Bankaları (3)</h5>
  <div class="card-body">
    <ul>
      <li>
        1. AKBANK T.A.Ş.
        <a href="http://www.akbank.com">http://www.akbank.com</a>
        Detay
      </li>
      <li>
        2. Dijital Banka HAYAT FİNANS A.Ş.
        <a href="http://www.hayatfinans.com.tr">http://www.hayatfinans.com.tr</a>
        Detay
      </li>
      <li>3. ZİRAAT BANKASI A.Ş. Detay</li>
    </ul>
  </div>
</div>
<div class="card">
  <h5>TMSF'ye Devredilen Bankalar (1)</h5>
  <div class="card-body">
    <ul>
      <li>1. TEST BANK A.Ş. Detay</li>
    </ul>
  </div>
</div>
"""


def test_parse_card_institutions():
    soup = BeautifulSoup(CARD_HTML, "html.parser")
    result = _parse_card_institutions(soup, "Banka")
    assert len(result) == 4

    akbank = result[0]
    assert akbank["name"] == "AKBANK T.A.Ş."
    assert akbank["website"] == "http://www.akbank.com"
    assert akbank["type"] == "Banka"
    assert akbank["subcategory"] == "Mevduat Bankaları"
    assert akbank["status"] == "Aktif"
    assert akbank["digital"] is False

    hayat = result[1]
    assert hayat["digital"] is True
    assert "Dijital Banka" not in hayat["name"]

    tmsf = result[3]
    assert tmsf["status"] == "İptal Edilmiş"
    assert tmsf["subcategory"] == "TMSF'ye Devredilen Bankalar"


@pytest.mark.parametrize(
    "heading", ["Faaliyet İzni İptal Edilen Bankalar (1)", "İPTAL EDİLEN BANKALAR (1)", "IPTAL EDILEN BANKALAR (1)"]
)
def test_parse_card_institutions_recognises_turkish_cased_cancelled_headings(heading):
    html = (
        f'<div class="card"><h5>{heading}</h5><div class="card-body">'
        "<ul><li>1. ESKI BANKA A.Ş. Detay</li></ul></div></div>"
    )
    result = _parse_card_institutions(BeautifulSoup(html, "html.parser"), "Banka")
    assert [i["status"] for i in result] == ["İptal Edilmiş"]


async def test_active_only_search_leaves_out_turkish_cased_cancelled_cards():
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.ingest.data_sources import InstitutionDirectory
    from bddk_mcp.tools.search import register

    def card(heading: str, name: str) -> str:
        return (
            f'<div class="card"><h5>{heading}</h5><div class="card-body"><ul><li>1. {name} Detay</li></ul></div></div>'
        )

    html = (
        card("Mevduat Bankaları (1)", "AKTİF BANKA A.Ş.")
        + card("Faaliyet İzni İptal Edilen Bankalar (1)", "ESKİ BANKA A.Ş.")
        + card("İPTAL EDİLEN BANKALAR (1)", "KAPANAN BANKA A.Ş.")
    )
    items = _parse_card_institutions(BeautifulSoup(html, "html.parser"), "Banka")
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=MagicMock()))
    tool = {c.args[0].__name__: c.args[0] for c in mcp.tool.return_value.call_args_list}["search_bddk_institutions"]
    directory = InstitutionDirectory(institutions=items, failed_pages=0, attempted_pages=1)
    with patch("bddk_mcp.tools.search.fetch_institutions_with_status", new=AsyncMock(return_value=directory)):
        out = await tool(active_only=True)
    assert "AKTİF BANKA" in out
    assert "ESKİ BANKA" not in out and "KAPANAN BANKA" not in out


TABPANE_HTML = """
<div class="tab-pane active" id="faaliyette">
  <li class="row">
    <div class="col-md-4 col-sm-12 baslikContainer">
      <div class="satirNo">1. </div>A&amp;T FİNANSAL KİRALAMA A.Ş.
    </div>
    <div class="col-md-4 col-sm-12 webAdresiContainer">
      <a href="https://www.atleasing.com.tr/" target="_blank">https://www.atleasing.com.tr/</a>
    </div>
  </li>
  <li class="row">
    <div class="col-md-4 col-sm-12 baslikContainer">
      <div class="satirNo">2. </div>DE LAGE LANDEN FİNANSAL KİRALAMA A.Ş.
    </div>
    <div class="col-md-4 col-sm-12 webAdresiContainer">
      <a>-</a>
    </div>
  </li>
</div>
<div class="tab-pane" id="kapanan">
  <li class="row">
    <div class="col-md-4 col-sm-12 baslikContainer">
      <div class="satirNo">1. </div>ATA FİNANSAL KİRALAMA A.Ş.
    </div>
    <div class="col-md-4 col-sm-12 webAdresiContainer">
      <a>-</a>
    </div>
  </li>
</div>
"""


def test_parse_tabpane_institutions():
    soup = BeautifulSoup(TABPANE_HTML, "html.parser")
    result = _parse_tabpane_institutions(soup, "Finansal Kiralama Şirketi")
    assert len(result) == 3

    first = result[0]
    assert first["name"] == "A&T FİNANSAL KİRALAMA A.Ş."
    assert first["website"] == "https://www.atleasing.com.tr/"
    assert first["status"] == "Aktif"

    second = result[1]
    assert second["website"] == ""  # "-" link has no http href

    closed = result[2]
    assert closed["status"] == "İptal Edilmiş"
    assert closed["name"] == "ATA FİNANSAL KİRALAMA A.Ş."


def test_parse_card_institutions_empty():
    soup = BeautifulSoup("<div></div>", "html.parser")
    assert _parse_card_institutions(soup, "Banka") == []


def test_parse_tabpane_institutions_empty():
    soup = BeautifulSoup("<div></div>", "html.parser")
    assert _parse_tabpane_institutions(soup, "Test") == []


# -- fetch_institutions_with_status integration (mocked HTTP) --


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Skip the exponential-backoff sleeps in request_with_retry so 5xx paths stay fast."""
    monkeypatch.setattr("bddk_mcp.core.outbound_http.asyncio.sleep", AsyncMock())
    monkeypatch.setattr("bddk_mcp.ingest.data_sources.assert_public_https_resolution", AsyncMock())


class _MockStreamContext:
    def __init__(self, response_awaitable) -> None:
        self._response_awaitable = response_awaitable
        self._response = None

    async def __aenter__(self):
        self._response = await self._response_awaitable
        return self._response

    async def __aexit__(self, *_exc) -> None:
        await self._response.aclose()


@pytest.fixture
def mock_http():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)

    def stream(method: str, url: str, **kwargs):
        send = getattr(client, method.lower())
        return _MockStreamContext(send(url, **kwargs))

    client.stream = MagicMock(side_effect=stream)
    return client


def _make_response(text: str = "", status_code: int = 200, *, json_data=None) -> httpx.Response:
    request = httpx.Request("GET", "https://www.bddk.org.tr/test")
    if json_data is not None:
        return httpx.Response(status_code, content=json.dumps(json_data).encode(), request=request)
    return httpx.Response(status_code, text=text, request=request)


async def test_fetch_institutions_with_status_card_page(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(CARD_HTML))
    result = (await fetch_institutions_with_status(mock_http, institution_type="Banka")).institutions
    assert len(result) == 4
    assert all(r["type"] == "Banka" for r in result)


async def test_fetch_institutions_with_status_type_filter(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(TABPANE_HTML))
    result = (await fetch_institutions_with_status(mock_http, institution_type="Finansal Kiralama")).institutions
    assert len(result) == 3


async def test_fetch_institutions_with_status_total_failure_raises_upstream_error(mock_http):
    """A fully failed directory fetch must never look like an empty directory."""
    mock_http.get = AsyncMock(return_value=_make_response("", status_code=500))
    with pytest.raises(BddkUpstreamError):
        await fetch_institutions_with_status(mock_http)


async def test_fetch_institutions_with_status_partial_failure_returns_partial_results(mock_http):
    """One good page plus failing pages still returns the fetched institutions."""
    # Page 77 (card layout) succeeds; every later page returns 500.
    mock_http.get = AsyncMock(side_effect=[_make_response(CARD_HTML)] + [_make_response("", status_code=500)] * 20)
    result = (await fetch_institutions_with_status(mock_http)).institutions
    assert len(result) > 0


async def test_fetch_institutions_with_status_unreachable_fails_fast(mock_http):
    """A connect-class failure aborts the remaining serial page fetches.

    The retry layer may attempt one page a few times, but the loop must not
    move on to the other directory pages once the host is unreachable.
    """
    mock_http.get = AsyncMock(side_effect=httpx.ConnectError("blocked egress"))
    with pytest.raises(BddkUpstreamError):
        await fetch_institutions_with_status(mock_http)
    assert mock_http.get.await_count <= 3  # one page's retries, not pages x retries


# -- Bulletin tests --


BULLETIN_PAGE_HTML = """
<html>
<body>
<input name="__RequestVerificationToken" value="test_token_123" />
<script>
    "tarih": '27.03.2026', "id": id
</script>
<table id="Tablo">
  <tr>
    <td></td><td>Krediler</td><td>TP</td><td>YP</td>
  </tr>
  <tr onclick="ShowModalGraph('1.0.1','TRY',1)">
    <td>1</td><td>Toplam Krediler (2+10)</td><td>15.550.362</td><td>8.995.594</td>
  </tr>
  <tr onclick="ShowModalGraph('1.0.2','TRY',1)">
    <td>2</td><td>Tüketici Kredileri</td><td>6.049.214</td><td>10.718</td>
  </tr>
</table>
</body>
</html>
"""

BULLETIN_API_RESPONSE = {
    "Baslik": "Toplam Krediler (TRY) [TP] [Sektör]",
    "XEkseni": ["2.01.2026", "9.01.2026", "16.01.2026"],
    "YEkseni": [14000000.0, 14500000.0, 15000000.0],
}


async def test_fetch_weekly_bulletin(mock_http):
    page_resp = _make_response(BULLETIN_PAGE_HTML)
    api_resp = _make_response(json_data=BULLETIN_API_RESPONSE)

    mock_http.get = AsyncMock(return_value=page_resp)
    mock_http.post = AsyncMock(return_value=api_resp)

    result = await fetch_weekly_bulletin(mock_http, metric_id="1.0.1", currency="TRY")
    assert result["title"] == "Toplam Krediler (TRY) [TP] [Sektör]"
    assert len(result["dates"]) == 3
    assert len(result["values"]) == 3
    assert result["currency"] == "TRY"

    # Verify the POST included the token and date
    call_kwargs = mock_http.post.call_args
    post_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
    assert post_data["__RequestVerificationToken"] == "test_token_123"
    assert post_data["tarih"] == "27.03.2026"
    assert post_data["tarafKodu"] == "10001"


async def test_fetch_weekly_bulletin_error(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(BULLETIN_PAGE_HTML))
    mock_http.post = AsyncMock(return_value=_make_response("", status_code=500))
    result = await fetch_weekly_bulletin(mock_http)
    assert "error" in result


async def test_fetch_weekly_bulletin_empty_series_is_no_data_not_an_upstream_error(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(BULLETIN_PAGE_HTML))
    mock_http.post = AsyncMock(return_value=_make_response(json_data={"Baslik": "T", "XEkseni": [], "YEkseni": []}))
    result = await fetch_weekly_bulletin(mock_http)
    assert "error" not in result
    assert result["dates"] == [] and result["values"] == []


async def test_fetch_bulletin_snapshot(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(BULLETIN_PAGE_HTML))
    result = await fetch_bulletin_snapshot(mock_http)
    assert len(result) == 2
    assert result[0]["name"] == "Toplam Krediler (2+10)"
    assert result[0]["metric_id"] == "1.0.1"
    assert result[0]["tp"] == "15.550.362"
    assert result[0]["yp"] == "8.995.594"
    assert result[1]["metric_id"] == "1.0.2"


async def test_fetch_bulletin_snapshot_no_table_raises_upstream_error(mock_http):
    """A page without the bulletin table is an extraction failure, not empty data."""
    mock_http.get = AsyncMock(return_value=_make_response("<html></html>"))
    with pytest.raises(BddkUpstreamError):
        await fetch_bulletin_snapshot(mock_http)


async def test_fetch_bulletin_snapshot_transport_error_raises_upstream_error(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response("", status_code=500))
    with pytest.raises(BddkUpstreamError):
        await fetch_bulletin_snapshot(mock_http)


# -- Announcement tests --


ANNOUNCEMENT_HTML = """
<html><body>
<a href="/Duyuru/Detay/2152">
  <span class="text">
    <span class="gorunenTarih">13.02.2026</span>
    Dolandırıcılık Hakkında Basın Duyurusu
  </span>
</a>
<a href="/Duyuru/Detay/2100">
  <span class="text">
    <span class="gorunenTarih">01.01.2026</span>
    Bankacılık Sektörü Verileri
  </span>
</a>
<a href="/some/other/link">Unrelated link</a>
</body></html>
"""


async def test_fetch_announcements(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(ANNOUNCEMENT_HTML))
    result = await fetch_announcements(mock_http, category_id=39)
    assert len(result) == 2

    first = result[0]
    assert first["title"] == "Dolandırıcılık Hakkında Basın Duyurusu"
    assert first["date"] == "13.02.2026"
    assert first["url"] == "https://www.bddk.org.tr/Duyuru/Detay/2152"
    assert first["category"] == "Basın Duyurusu"


async def test_fetch_announcements_empty(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response("<html></html>"))
    result = await fetch_announcements(mock_http, category_id=39)
    assert result == []


async def test_fetch_announcements_error_raises_upstream_error(mock_http):
    """A failed announcements fetch must never look like zero announcements."""
    mock_http.get = AsyncMock(return_value=_make_response("", status_code=500))
    with pytest.raises(BddkUpstreamError):
        await fetch_announcements(mock_http, category_id=39)


# -- Weekly metric key pinned against captured BDDK pages --

_FIXTURES = Path(__file__).parent / "fixtures"


def _weekly_page_rows() -> dict[str, str]:
    soup = BeautifulSoup((_FIXTURES / "bddk_weekly_page.html").read_text(encoding="utf-8"), "html.parser")
    rows = {}
    for tr in soup.find("table", id="Tablo").find_all("tr"):
        match = re.search(r"ShowModalGraph\('([^']+)'", str(tr))
        if match:
            rows[match.group(1)] = tr.find_all("td")[1].get_text(strip=True)
    return rows


def test_weekly_metric_key_uses_bddk_row_labels():
    from bddk_mcp.ingest.data_sources import WEEKLY_METRICS

    page_rows = _weekly_page_rows()
    series = json.loads((_FIXTURES / "bddk_weekly_series.json").read_text(encoding="utf-8"))["series"]
    for metric_id, label, _note in WEEKLY_METRICS:
        if metric_id in page_rows:
            assert page_rows[metric_id] == label, metric_id
        else:
            assert series[metric_id]["Baslik"].startswith(f"{label} (TRY)"), metric_id
    ids = {metric_id for metric_id, _label, _note in WEEKLY_METRICS}
    assert {"1.0.1", "1.0.2", "1.0.4", "1.0.8", "1.0.10", "1.0.12", "2.0.1", "4.0.1"} <= ids
    assert page_rows["1.0.10"] == "b) Taksitsiz" and page_rows["1.0.12"] == "Ticari ve Diğer Krediler"


def _weekly_series_fixture(metric_id: str) -> dict:
    return json.loads((_FIXTURES / "bddk_weekly_series.json").read_text(encoding="utf-8"))["series"][metric_id]


def _mock_weekly(mock_http, metric_id: str = "1.0.1") -> None:
    page = (_FIXTURES / "bddk_weekly_page.html").read_text(encoding="utf-8")
    mock_http.get = AsyncMock(return_value=_make_response(page))
    mock_http.post = AsyncMock(return_value=_make_response(json_data=_weekly_series_fixture(metric_id)))


@pytest.mark.parametrize(
    ("days", "expected_dates", "window_met"),
    [
        (14, ["23.10.2026", "30.10.2026", "6.11.2026"], True),
        (7, ["30.10.2026", "6.11.2026"], True),
        (90, None, True),  # 14.08-6.11 is every weekly point a 90-day window can hold
        (91, None, False),  # a 7.08.2026 observation would have been inside the window
        (3650, None, False),
    ],
)
async def test_weekly_series_is_trimmed_to_the_requested_window(mock_http, days, expected_dates, window_met):
    """BDDK ignores the requested window and always returns its last 13 weeks."""
    _mock_weekly(mock_http)
    result = await fetch_weekly_bulletin(mock_http, "1.0.1", "TRY", days)
    returned = _weekly_series_fixture("1.0.1")["XEkseni"]
    assert result["dates"] == (expected_dates or returned)
    assert result["values"][-1] == 18523337.65624
    assert result["returned_observations"] == 13 and result["returned_first_date"] == "14.08.2026"
    assert result["trimmed_observations"] == 13 - len(result["dates"])
    assert result["window_met"] is window_met
    assert result["requested_days"] == days


async def test_weekly_series_carries_the_unit_and_date_the_page_states(mock_http):
    _mock_weekly(mock_http)
    result = await fetch_weekly_bulletin(mock_http, "1.0.1", "TRY")
    assert result["unit"] == "Milyon TL"
    assert result["bulletin_date"] == "6 Kasım 2026 Cuma"
    assert mock_http.post.call_args.kwargs["data"]["tarih"] == "6.11.2026"
    # The page states the unit of its TL view only; a USD series gets no guessed unit.
    _mock_weekly(mock_http)
    assert (await fetch_weekly_bulletin(mock_http, "1.0.1", "USD"))["unit"] == ""


async def test_unknown_weekly_metric_is_an_unknown_metric_error_not_an_empty_period(mock_http):
    from bddk_mcp.ingest.data_sources import UNKNOWN_METRIC

    _mock_weekly(mock_http, "9.9.9")
    result = await fetch_weekly_bulletin(mock_http, "9.9.9")
    assert result["reason"] == UNKNOWN_METRIC and "9.9.9" in result["error"]


async def test_bulletin_snapshot_table_reads_heading_date_and_unit(mock_http):
    from bddk_mcp.ingest.data_sources import fetch_bulletin_snapshot_table

    _mock_weekly(mock_http)
    table = await fetch_bulletin_snapshot_table(mock_http)
    assert (table["heading"], table["party"]) == ("Krediler", "Sektör")
    assert (table["bulletin_date"], table["unit"]) == ("6 Kasım 2026 Cuma", "Milyon TL")
    assert len(table["rows"]) == 22
    assert table["rows"][0] == {
        "row_number": "1",
        "name": "Toplam Krediler (2+10)",
        "metric_id": "1.0.1",
        "tp": "18.523.337",
        "yp": "10.517.881",
    }
    assert {row["metric_id"] for row in table["rows"]} == set(_weekly_page_rows())
    assert await fetch_bulletin_snapshot(mock_http) == table["rows"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18.523.337", "18,523,337"),
        (" 10.517.881", "10,517,881"),
        ("280", "280"),
        ("0", "0"),
        ("18.523.337,66", "18,523,337.66"),
        ("-1.234,5", "-1,234.5"),
        ("n/a", "n/a"),
        ("1,234.56", "1,234.56"),  # not the page's pattern: passed through unchanged
    ],
)
def test_snapshot_cells_use_the_weekly_number_format(text, expected):
    from bddk_mcp.ingest.data_sources import format_snapshot_cell

    assert format_snapshot_cell(text) == expected


def test_parse_card_institutions_live_digital_bank_suffix():
    """The live page writes '(Dijital Banka)' after the name; neither '()' nor the flag may be lost."""
    soup = BeautifulSoup((_FIXTURES / "bddk_kurulus_77_katilim.html").read_text(encoding="utf-8"), "html.parser")
    result = {item["name"]: item for item in _parse_card_institutions(soup, "Banka")}
    assert list(result) == [
        "ADİL KATILIM BANKASI A.Ş.",
        "ALBARAKA TÜRK KATILIM BANKASI A.Ş.",
        "DÜNYA KATILIM BANKASI A.Ş.",
        "HAYAT FİNANS KATILIM BANKASI A.Ş.",
    ]
    assert result["ADİL KATILIM BANKASI A.Ş."]["digital"] is True
    assert result["HAYAT FİNANS KATILIM BANKASI A.Ş."]["digital"] is True
    assert result["DÜNYA KATILIM BANKASI A.Ş."]["digital"] is False
    assert result["ADİL KATILIM BANKASI A.Ş."]["subcategory"] == "Katılım Bankaları"
    assert all("()" not in name and "Dijital" not in name for name in result)
