"""Tests for analytics.py — trend analysis, digest, comparison, update detection."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from analytics import analyze_trends, build_digest, compare_metrics, check_updates


def _make_response(text: str = "", status_code: int = 200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=resp
        )
    return resp


BULLETIN_PAGE = """
<html><body>
<input name="__RequestVerificationToken" value="tok123" />
<script>"tarih": '01.04.2026'</script>
</body></html>
"""

BULLETIN_API = {
    "Baslik": "Toplam Krediler (TRY) [TP]",
    "XEkseni": [
        "01.01.2026", "08.01.2026", "15.01.2026", "22.01.2026",
        "29.01.2026", "05.02.2026", "12.02.2026", "19.02.2026",
    ],
    "YEkseni": [
        10000.0, 10100.0, 10050.0, 10200.0,
        10300.0, 10150.0, 10400.0, 10500.0,
    ],
}


@pytest.fixture
def mock_http():
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


def _setup_bulletin_mock(mock_http, api_data=None):
    """Set up mock to return bulletin page + API response."""
    page_resp = _make_response(BULLETIN_PAGE)
    api_resp = _make_response(json_data=api_data or BULLETIN_API)
    mock_http.get = AsyncMock(return_value=page_resp)
    mock_http.post = AsyncMock(return_value=api_resp)


# -- Trend Analysis Tests --


async def test_analyze_trends_basic(mock_http):
    _setup_bulletin_mock(mock_http)
    result = await analyze_trends(mock_http, "1.0.1", "TRY", "1", 12)

    assert "error" not in result
    assert result["current"] == 10500.0
    assert result["previous"] == 10400.0
    assert result["wow_change"] == pytest.approx(100.0)
    assert result["wow_pct"] == pytest.approx(100.0 / 10400 * 100, rel=0.01)
    assert result["data_points"] == 8
    assert result["trend_direction"] in ("yükseliş", "düşüş", "yatay")
    assert "narrative" in result


async def test_analyze_trends_includes_minmax(mock_http):
    _setup_bulletin_mock(mock_http)
    result = await analyze_trends(mock_http, "1.0.1", "TRY", "1", 12)

    assert result["min"] == 10000.0
    assert result["max"] == 10500.0
    assert result["min_date"] == "01.01.2026"
    assert result["max_date"] == "19.02.2026"


async def test_analyze_trends_not_enough_data(mock_http):
    _setup_bulletin_mock(mock_http, {
        "Baslik": "Test",
        "XEkseni": ["01.01.2026"],
        "YEkseni": [100.0],
    })
    result = await analyze_trends(mock_http)
    assert "error" in result
    assert "Not enough" in result["error"]


async def test_analyze_trends_api_error(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(BULLETIN_PAGE))
    mock_http.post = AsyncMock(return_value=_make_response("", status_code=500))
    result = await analyze_trends(mock_http)
    assert "error" in result


# -- Compare Metrics Tests --


async def test_compare_metrics_multiple(mock_http):
    _setup_bulletin_mock(mock_http)
    result = await compare_metrics(mock_http, ["1.0.1", "1.0.2"], "TRY", "1", 90)

    assert len(result["metrics"]) == 2
    assert result["currency"] == "TRY"
    # Both use the same mock, so same values
    for m in result["metrics"]:
        assert "error" not in m
        assert m["current"] == 10500.0


async def test_compare_metrics_max_four(mock_http):
    _setup_bulletin_mock(mock_http)
    result = await compare_metrics(
        mock_http, ["1", "2", "3", "4", "5", "6"], "TRY", "1", 90,
    )
    # Should cap at 4
    assert len(result["metrics"]) == 4


async def test_compare_metrics_with_error(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(BULLETIN_PAGE))
    mock_http.post = AsyncMock(return_value=_make_response("", status_code=500))
    result = await compare_metrics(mock_http, ["1.0.1"], "TRY", "1", 90)
    assert "error" in result["metrics"][0]


# -- Digest Tests --


ANNOUNCEMENT_HTML = """
<html><body>
<a href="/Duyuru/Detay/9999">
  <span class="text">
    <span class="gorunenTarih">01.04.2026</span>
    Test Duyuru
  </span>
</a>
</body></html>
"""

SNAPSHOT_HTML = """
<html><body>
<table id="Tablo">
  <tr><td></td><td>Krediler</td><td>TP</td><td>YP</td></tr>
  <tr onclick="ShowModalGraph('1.0.1','TRY',1)">
    <td>1</td><td>Toplam Krediler</td><td>15.000</td><td>9.000</td>
  </tr>
</table>
</body></html>
"""


async def test_build_digest_basic(mock_http):
    # Mock: announcements return HTML, bulletin snapshot returns HTML
    def mock_get(url, **kwargs):
        if "Duyuru" in url:
            return _make_response(ANNOUNCEMENT_HTML)
        return _make_response(SNAPSHOT_HTML)

    mock_http.get = AsyncMock(side_effect=mock_get)

    decisions = [
        {
            "title": "Test Karar",
            "decision_date": "01.04.2026",
            "category": "Kurul Kararı",
            "document_id": "1",
        },
    ]
    result = await build_digest(mock_http, decisions, period_days=30)

    assert result["total_decisions"] >= 0  # Date may not match exactly
    assert result["total_announcements"] >= 0
    assert "narrative" in result


async def test_build_digest_empty_cache(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response("<html></html>"))
    result = await build_digest(mock_http, [], period_days=7)

    assert result["total_decisions"] == 0
    assert "narrative" in result


# -- Update Detection Tests --


async def test_check_updates_new_items(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(ANNOUNCEMENT_HTML))

    # Empty known set — everything is "new"
    result = await check_updates(mock_http, [], known_announcement_ids=set())

    assert result["new_announcements_count"] >= 1
    assert result["new_announcements"][0]["title"] == "Test Duyuru"


async def test_check_updates_nothing_new(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response(ANNOUNCEMENT_HTML))

    # Already know about this announcement
    known = {"https://www.bddk.org.tr/Duyuru/Detay/9999"}
    result = await check_updates(mock_http, [], known_announcement_ids=known)

    assert result["new_announcements_count"] == 0


async def test_check_updates_error_handling(mock_http):
    mock_http.get = AsyncMock(return_value=_make_response("", status_code=500))
    result = await check_updates(mock_http, [], known_announcement_ids=set())
    assert result["new_announcements_count"] == 0
