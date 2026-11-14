"""Tests for tools/analytics.py — tool registration and digest period mapping."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.analytics import register


def _registered_tools(mcp: MagicMock) -> dict:
    """Return {tool_name: tool_fn} for all @mcp.tool()-registered functions."""
    return {call.args[0].__name__: call.args[0] for call in mcp.tool.return_value.call_args_list}


def test_public_register_adds_only_stateless_analytics_tools():
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    assert set(_registered_tools(mcp).keys()) == {
        "analyze_bulletin_trends",
        "get_regulatory_digest",
        "compare_bulletin_metrics",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "period,expected_days",
    [("day", 1), ("week", 7), ("month", 30), ("quarter", 90)],
)
async def test_regulatory_digest_period_mapping(period, expected_days):
    """get_regulatory_digest must map 'day' → 1 day and fall back to 30 on unknown."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=[])
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with patch(
        "bddk_mcp.tools.analytics.build_digest",
        new=AsyncMock(
            return_value={
                "narrative": "ok",
                "decisions_by_category": {},
                "new_decisions": [],
                "announcements": [],
                "bulletin_snapshot": [],
            }
        ),
    ) as mock_digest:
        out = await get_regulatory_digest(period=period)

    mock_digest.assert_awaited_once()
    called_period_days = mock_digest.await_args.args[2]
    assert called_period_days == expected_days
    assert f"Son {expected_days} Gün" in out


@pytest.mark.asyncio
async def test_regulatory_digest_rejects_unknown_period():
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await get_regulatory_digest(period="unknown")


# -- Upstream failure surfacing ----------------------------------------------


@pytest.mark.asyncio
async def test_regulatory_digest_marks_unavailable_upstream_sections():
    """The digest must say announcement/bulletin data was unavailable, not omit it."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=[])
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with patch(
        "bddk_mcp.tools.analytics.build_digest",
        new=AsyncMock(
            return_value={
                "narrative": "Duyuru verileri alınamadı.",
                "decisions_by_category": {},
                "new_decisions": [],
                "announcements": [],
                "announcements_available": False,
                "bulletin_snapshot": [],
                "bulletin_snapshot_available": False,
            }
        ),
    ):
        out = await get_regulatory_digest(period="week")

    assert "Duyurular:" in out
    assert "veri alınamadı" in out
    assert "Bülten Özet:" in out


@pytest.mark.asyncio
async def test_regulatory_digest_header_shows_true_totals_and_cut_lines():
    """Capped lists must say how many entries were left out; the header prints the true total."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=[])
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with patch(
        "bddk_mcp.tools.analytics.build_digest",
        new=AsyncMock(
            return_value={
                "narrative": "ok",
                "decisions_by_category": {"Kurul Kararı": 25},
                "new_decisions": [{"title": f"K{i}", "decision_date": "01.01.2026"} for i in range(20)],
                "total_decisions": 25,
                "announcements": [{"title": f"D{i}", "date": "01.01.2026"} for i in range(20)],
                "total_announcements": 57,
                "bulletin_snapshot": [],
            }
        ),
    ):
        out = await get_regulatory_digest(period="month")

    assert "**Duyurular (57):**" in out
    assert "... ve 47 daha" in out
    assert "... ve 15 daha" in out


def _analytics_tool(name: str):
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock()))
    return _registered_tools(mcp)[name]


@pytest.mark.asyncio
async def test_trend_with_too_few_points_is_a_non_retryable_error_not_an_upstream_failure():
    tool = _analytics_tool("analyze_bulletin_trends")
    result = {"error": "En az 2 gözlem gerekir (bulunan: 1).", "reason": "insufficient_data"}
    with patch("bddk_mcp.tools.analytics.analyze_trends", new=AsyncMock(return_value=result)):
        with pytest.raises(ToolError) as excinfo:
            await tool()
    message = str(excinfo.value)
    assert "[ERROR:INSUFFICIENT_DATA] retryable=false" in message
    assert "UPSTREAM_FETCH_FAILED" not in message and "En az 2 gözlem" in message


@pytest.mark.asyncio
async def test_trend_upstream_error_stays_retryable():
    tool = _analytics_tool("analyze_bulletin_trends")
    with patch("bddk_mcp.tools.analytics.analyze_trends", new=AsyncMock(return_value={"error": "x"})):
        with pytest.raises(ToolError, match="UPSTREAM_FETCH_FAILED"):
            await tool()


@pytest.mark.asyncio
async def test_compare_table_shows_the_reason_and_the_missing_percentage():
    tool = _analytics_tool("compare_bulletin_metrics")
    result = {
        "metrics": [
            {"metric_id": "1.0.1", "error": "En az 2 gözlem gerekir (bulunan: 1).", "reason": "insufficient_data"},
            {"metric_id": "1.0.2", "error": "Approved BDDK upstream request failed."},
            {"metric_id": "1.0.4", "title": "Konut", "current": 5.0, "wow_pct": None},
        ]
    }
    with patch("bddk_mcp.tools.analytics.compare_metrics", new=AsyncMock(return_value=result)):
        out = await tool(metric_ids="1.0.1,1.0.2,1.0.4")
    assert "En az 2 gözlem gerekir" in out
    assert "HATA" in out
    assert "n/a" in out and "yüzde değişim hesaplanamadı (önceki değer 0)" in out


@pytest.mark.asyncio
async def test_trend_text_for_a_zero_base_does_not_print_a_percentage():
    tool = _analytics_tool("analyze_bulletin_trends")
    result = {
        "title": "T",
        "narrative": "n",
        "current": 50.0,
        "current_date": "b",
        "previous": 0.0,
        "previous_date": "a",
        "wow_change": 50.0,
        "wow_pct": None,
        "avg": 25.0,
        "min": 0.0,
        "min_date": "a",
        "max": 50.0,
        "max_date": "b",
        "trend_direction": "yükseliş",
        "data_points": 2,
    }
    with patch("bddk_mcp.tools.analytics.analyze_trends", new=AsyncMock(return_value=result)):
        out = await tool()
    assert "yüzde değişim hesaplanamadı (önceki değer 0)" in out and "%+" not in out


@pytest.mark.parametrize("tool_name", ["analyze_bulletin_trends", "compare_bulletin_metrics"])
def test_weekly_metric_key_in_descriptions_is_bddk_labels(tool_name):
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=None))
    doc = " ".join(_registered_tools(mcp)[tool_name].__doc__.split())
    assert "1.0.12=Ticari ve Diğer Krediler" in doc
    assert "1.0.10=b) Taksitsiz (Bireysel Kredi Kartları altında)" in doc
    assert "1.0.2=Tüketici Kredileri ve Bireysel Kredi Kartları (3+7)" in doc
    assert "4.0.1=Mevduat (Katılım Fonu)" in doc and "2.0.1=Takipteki Alacaklar" in doc
    assert "1.0.10=Ticari" not in doc


# -- Weekly coverage, unit and unknown ids through the real engine (captured BDDK page) --

_FIXTURES = Path(__file__).parent / "fixtures"


def _patch_weekly_upstream(monkeypatch, series_by_id: dict[str, str] | None = None):
    import bddk_mcp.ingest.data_sources as data_sources

    series = json.loads((_FIXTURES / "bddk_weekly_series.json").read_text(encoding="utf-8"))["series"]
    page = (_FIXTURES / "bddk_weekly_page.html").read_text(encoding="utf-8")
    request = httpx.Request("GET", "https://www.bddk.org.tr/bultenhaftalik")

    async def post(_http, _url, **kwargs):
        metric_id = kwargs["data"]["id"]
        payload = series[(series_by_id or {}).get(metric_id, metric_id)]
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(data_sources, "_get", AsyncMock(return_value=httpx.Response(200, text=page, request=request)))
    monkeypatch.setattr(data_sources, "_post", post)


@pytest.mark.asyncio
async def test_trend_over_a_longer_window_than_bddk_serves_states_the_covered_period(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    out = await _analytics_tool("analyze_bulletin_trends")(metric_id="1.0.1", lookback_weeks=52)
    assert "Kapsanan dönem: 14.08.2026 – 6.11.2026 (13 gözlem; istenen pencere: son 52 hafta (364 gün))" in out
    assert "istenen pencere karşılanamadı" in out
    assert "Birim: Milyon TL" in out
    assert "Güncel (6.11.2026): 18,523,337.66" in out


@pytest.mark.asyncio
async def test_trend_statistics_use_only_the_requested_weeks(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    out = await _analytics_tool("analyze_bulletin_trends")(metric_id="1.0.1", lookback_weeks=4)
    # 6.11.2026 - 28 days = 9.10.2026: five weekly points, the lowest of them on 16.10.2026.
    assert "Kapsanan dönem: 9.10.2026 – 6.11.2026 (5 gözlem" in out
    assert "Veri noktası: 5" in out
    assert "Dönem min: 18,246,309.67 (16.10.2026)" in out
    assert "karşılanamadı" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trimmed", "suggests_longer_window"),
    [(3, True), (0, False)],
)
async def test_insufficient_data_hint_suggests_a_longer_window_only_when_trimming_caused_it(
    trimmed, suggests_longer_window
):
    tool = _analytics_tool("analyze_bulletin_trends")
    result = {
        "error": "En az 2 gözlem gerekir (bulunan: 1).",
        "reason": "insufficient_data",
        "trimmed_observations": trimmed,
    }
    with patch("bddk_mcp.tools.analytics.analyze_trends", new=AsyncMock(return_value=result)):
        with pytest.raises(ToolError) as excinfo:
            await tool()
    message = str(excinfo.value)
    assert ("increase lookback_weeks" in message) is suggests_longer_window
    if not suggests_longer_window:
        assert "a longer lookback_weeks will not help" in message


@pytest.mark.asyncio
async def test_trend_and_comparison_report_an_unknown_metric_as_invalid_input(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    with pytest.raises(ToolError) as excinfo:
        await _analytics_tool("analyze_bulletin_trends")(metric_id="9.9.9")
    assert "[ERROR:INVALID_INPUT]" in str(excinfo.value) and "INSUFFICIENT_DATA" not in str(excinfo.value)
    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await _analytics_tool("compare_bulletin_metrics")(metric_ids="9.9.9")
    out = await _analytics_tool("compare_bulletin_metrics")(metric_ids="1.0.1,9.9.9")
    assert "BİLİNMEYEN ID" in out


@pytest.mark.asyncio
async def test_comparison_states_date_unit_coverage_and_number_format(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    out = await _analytics_tool("compare_bulletin_metrics")(metric_ids="1.0.1,1.0.12", days=3650)
    row = next(line for line in out.splitlines() if line.startswith("Ticari ve Diğer Krediler"))
    assert "6.11.2026" in row and "11,509,666.16" in row
    assert "Kapsanan dönem (1.0.12): 14.08.2026 – 6.11.2026 (13 gözlem; istenen pencere: son 3650 gün)" in out
    assert "Birim: Milyon TL" in out
    assert "istenen pencere karşılanamadı" in out
    assert out.count("Sayı biçimi:") == 1


@pytest.mark.asyncio
async def test_comparison_uses_only_the_requested_window(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    out = await _analytics_tool("compare_bulletin_metrics")(metric_ids="1.0.1,1.0.12", days=14)
    # 6.11.2026 - 14 days = 23.10.2026: three weekly points per metric.
    assert "Kapsanan dönem (1.0.12): 23.10.2026 – 6.11.2026 (3 gözlem; istenen pencere: son 14 gün)" in out
    assert "karşılanamadı" not in out


# -- Digest: catalog count is not a fact about BDDK; announcements newest first --


def _digest_tool_with(decisions):
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=decisions)
    register(mcp, deps)
    return _registered_tools(mcp)["get_regulatory_digest"]


@pytest.mark.asyncio
async def test_digest_prints_the_bulletin_block_with_date_unit_and_format():
    tool = _digest_tool_with([])
    digest = {
        "narrative": "n",
        "decisions_by_category": {},
        "new_decisions": [],
        "announcements": [],
        "bulletin_snapshot": [{"name": "Toplam Krediler (2+10)", "tp": "18.523.337", "yp": "10.517.881"}],
        "bulletin_heading": "Krediler",
        "bulletin_date": "6 Kasım 2026 Cuma",
        "bulletin_unit": "Milyon TL",
    }
    with patch("bddk_mcp.tools.analytics.build_digest", new=AsyncMock(return_value=digest)):
        out = await tool(period="week")
    assert "Krediler tablosu" in out and "Bülten tarihi: 6 Kasım 2026 Cuma" in out and "Birim: Milyon TL" in out
    assert "Toplam Krediler (2+10): TP=18,523,337, YP=10,517,881" in out
    assert "Sayı biçimi:" in out
