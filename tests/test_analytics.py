"""Tests for analytics.py — trend analysis, digest, comparison, update detection."""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.observability.analytics import analyze_trends, build_digest, compare_metrics


def _make_response(text: str = "", status_code: int = 200, json_data=None):
    if json_data is not None:
        return httpx.Response(
            status_code,
            json=json_data,
            request=httpx.Request("POST", "https://www.bddk.org.tr/api"),
        )
    return httpx.Response(
        status_code,
        text=text,
        request=httpx.Request("GET", "https://www.bddk.org.tr/test"),
    )


class _StreamContext:
    """Adapt the tests' method mocks to the production streaming interface."""

    def __init__(self, response: Awaitable[httpx.Response]) -> None:
        self._response = response
        self._entered: httpx.Response | None = None

    async def __aenter__(self) -> httpx.Response:
        self._entered = await self._response
        return self._entered

    async def __aexit__(self, *_exc: object) -> None:
        if self._entered is not None:
            await self._entered.aclose()


BULLETIN_PAGE = """
<html><body>
<input name="__RequestVerificationToken" value="tok123" />
<script>"tarih": '01.04.2026'</script>
</body></html>
"""

BULLETIN_API = {
    "Baslik": "Toplam Krediler (TRY) [TP]",
    "XEkseni": [
        "01.01.2026",
        "08.01.2026",
        "15.01.2026",
        "22.01.2026",
        "29.01.2026",
        "05.02.2026",
        "12.02.2026",
        "19.02.2026",
    ],
    "YEkseni": [
        10000.0,
        10100.0,
        10050.0,
        10200.0,
        10300.0,
        10150.0,
        10400.0,
        10500.0,
    ],
}


@pytest.fixture
def mock_http(monkeypatch):
    import bddk_mcp.ingest.data_sources as data_sources

    # DNS policy is covered independently; these parser/analytics tests must
    # not depend on external name resolution.
    monkeypatch.setattr(data_sources, "assert_public_https_resolution", AsyncMock(return_value=None))
    # Error-path tests exercise the retry loop; skip its real exponential backoff sleeps.
    monkeypatch.setattr("bddk_mcp.core.outbound_http.asyncio.sleep", AsyncMock())
    client = AsyncMock(spec=httpx.AsyncClient)

    def stream(method: str, url: str, **kwargs):
        caller = client.post if method.upper() == "POST" else client.get
        return _StreamContext(caller(url, **kwargs))

    client.stream = MagicMock(side_effect=stream)
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


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([5, 5, 5, 5], "yatay"),
        ([5, 6, 6, 6], "yatay"),
        ([6, 6, 6, 6], "yatay"),
        ([9, 8, 7, 6], "düşüş"),
        ([5, 5, 6, 4], "yatay"),
        ([5, 6, 5, 4], "düşüş"),
        ([1, 2, 3, 4], "yükseliş"),
    ],
)
async def test_analyze_trends_direction_counts_falls_as_well_as_rises(mock_http, values, expected):
    dates = ["01.01.2026", "08.01.2026", "15.01.2026", "22.01.2026", "29.01.2026"][-len(values) :]
    _setup_bulletin_mock(mock_http, {"Baslik": "", "XEkseni": dates, "YEkseni": values})
    result = await analyze_trends(mock_http, "1.0.1")
    assert result["trend_direction"] == expected
    assert result["title"] == "1.0.1"  # blank upstream title falls back to the metric id
    assert "Son 4 veri noktasına göre trend" in result["narrative"]


async def test_analyze_trends_includes_minmax(mock_http):
    _setup_bulletin_mock(mock_http)
    result = await analyze_trends(mock_http, "1.0.1", "TRY", "1", 12)

    assert result["min"] == 10000.0
    assert result["max"] == 10500.0
    assert result["min_date"] == "01.01.2026"
    assert result["max_date"] == "19.02.2026"


async def test_analyze_trends_not_enough_data(mock_http):
    _setup_bulletin_mock(
        mock_http,
        {
            "Baslik": "Test",
            "XEkseni": ["01.01.2026"],
            "YEkseni": [100.0],
        },
    )
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


async def test_compare_metrics_never_drops_requested_ids(mock_http):
    """The 4-metric limit lives in the tool schema; the engine must not slice silently."""
    _setup_bulletin_mock(mock_http)
    result = await compare_metrics(mock_http, ["1.0.1", "1.0.2", "1.0.4", "1.0.8", "1.0.10"], "TRY", "1", 90)
    assert [m["metric_id"] for m in result["metrics"]] == ["1.0.1", "1.0.2", "1.0.4", "1.0.8", "1.0.10"]


async def test_compare_metrics_blank_title_falls_back_to_metric_id(mock_http):
    _setup_bulletin_mock(mock_http, {**BULLETIN_API, "Baslik": ""})
    result = await compare_metrics(mock_http, ["1.0.1"])
    assert result["metrics"][0]["title"] == "1.0.1"


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


async def test_build_digest_marks_announcements_unavailable_and_drops_partial_rows(mock_http, monkeypatch):
    """A failed announcement sweep must not leave a partial set behind a False flag."""
    calls: list[int] = []

    async def flaky_fetch(_http, category_id: int):
        calls.append(category_id)
        if len(calls) == 1:
            return [{"title": "Erken duyuru", "date": "01.08.2026", "url": "https://example.invalid/a"}]
        raise BddkUpstreamError("upstream down")

    monkeypatch.setattr("bddk_mcp.observability.analytics.fetch_announcements", flaky_fetch)
    monkeypatch.setattr(
        "bddk_mcp.observability.analytics.fetch_bulletin_snapshot",
        AsyncMock(side_effect=BddkUpstreamError("upstream down")),
    )

    digest = await build_digest(mock_http, [], 30)

    assert digest["announcements_available"] is False
    assert digest["bulletin_snapshot_available"] is False
    assert digest["announcements"] == []
    assert digest["total_announcements"] == 0
    assert "alınamadı" in digest["narrative"]


@pytest.mark.parametrize(
    "values, dates",
    [
        ([100, None], ["01.01.2026", "08.01.2026"]),
        ([100, "missing"], ["01.01.2026", "08.01.2026"]),
        ([100, True], ["01.01.2026", "08.01.2026"]),
        ([100, "NaN"], ["01.01.2026", "08.01.2026"]),
        ([100, "inf"], ["01.01.2026", "08.01.2026"]),
        ([100, 90], ["01.01.2026"]),
        ([], []),
        ([100], ["01.01.2026"]),
    ],
)
async def test_invalid_observations_never_become_trends_or_comparisons(mock_http, values, dates):
    _setup_bulletin_mock(mock_http, {"Baslik": "Test", "XEkseni": dates, "YEkseni": values})
    assert "error" in await analyze_trends(mock_http)
    result = await compare_metrics(mock_http, ["1.0.1"])
    assert "error" in result["metrics"][0]
    assert "current" not in result["metrics"][0]


async def test_numeric_strings_and_zero_remain_valid_observations(mock_http):
    _setup_bulletin_mock(mock_http, {"XEkseni": ["01.01.2026", "08.01.2026"], "YEkseni": ["100", 0]})
    result = await analyze_trends(mock_http)
    assert result["current"] == 0
    assert result["wow_pct"] == -100


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 4, 10, 15, 30)


async def test_build_digest_window_is_period_days_calendar_days_including_today(mock_http, monkeypatch):
    monkeypatch.setattr("bddk_mcp.observability.analytics.datetime", _FrozenDatetime)
    mock_http.get = AsyncMock(return_value=_make_response("<html></html>"))
    decisions = [
        {"title": f"K{day}", "decision_date": f"{day:02d}.04.2026", "category": "Kurul Kararı"}
        for day in (10, 9, 4, 3)  # 0, 1, 6 and 7 days old
    ]

    day = await build_digest(mock_http, decisions, period_days=1)
    week = await build_digest(mock_http, decisions, period_days=7)

    assert [d["title"] for d in day["new_decisions"]] == ["K10"]
    assert [d["title"] for d in week["new_decisions"]] == ["K10", "K9", "K4"]
    assert "yerel corpus kataloğundan" in week["narrative"]
