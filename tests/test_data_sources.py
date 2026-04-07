"""Tests for data_sources.py — institution, bulletin, and announcement parsers."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from bs4 import BeautifulSoup

from data_sources import (
    _parse_card_institutions,
    _parse_tabpane_institutions,
    fetch_announcements,
    fetch_bulletin_snapshot,
    fetch_institutions,
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


# -- fetch_institutions integration (mocked HTTP) --


@pytest.fixture
def mock_http():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


def _make_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=resp)
    return resp


async def test_fetch_institutions_card_page(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(CARD_HTML))
    result = await fetch_institutions(mock_http, institution_type="Banka")
    assert len(result) == 4
    assert all(r["type"] == "Banka" for r in result)


async def test_fetch_institutions_type_filter(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(TABPANE_HTML))
    result = await fetch_institutions(mock_http, institution_type="Finansal Kiralama")
    assert len(result) == 3


async def test_fetch_institutions_error_handling(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response("", status_code=500))
    result = await fetch_institutions(mock_http)
    assert result == []


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
    api_resp = MagicMock(spec=httpx.Response)
    api_resp.status_code = 200
    api_resp.raise_for_status = MagicMock()
    api_resp.json = MagicMock(return_value=BULLETIN_API_RESPONSE)

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


async def test_fetch_bulletin_snapshot(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(BULLETIN_PAGE_HTML))
    result = await fetch_bulletin_snapshot(mock_http)
    assert len(result) == 2
    assert result[0]["name"] == "Toplam Krediler (2+10)"
    assert result[0]["metric_id"] == "1.0.1"
    assert result[0]["tp"] == "15.550.362"
    assert result[0]["yp"] == "8.995.594"
    assert result[1]["metric_id"] == "1.0.2"


async def test_fetch_bulletin_snapshot_no_table(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response("<html></html>"))
    result = await fetch_bulletin_snapshot(mock_http)
    assert result == []


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


async def test_fetch_announcements_error(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response("", status_code=500))
    result = await fetch_announcements(mock_http, category_id=39)
    assert result == []
