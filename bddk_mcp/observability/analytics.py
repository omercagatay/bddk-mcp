"""BDDK analytics: trend analysis, digest, comparison."""

from collections.abc import Callable
from datetime import datetime, timedelta

import httpx

from bddk_mcp.core.config import ANNOUNCEMENT_CATEGORY_IDS
from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.ingest.data_sources import (
    fetch_announcements,
    fetch_bulletin_snapshot_table,
    fetch_weekly_bulletin,
    format_bulletin_number,
    istanbul_today,
)

# -- Trend Analysis --------------------------------------------------------

REASON_INSUFFICIENT_DATA = "insufficient_data"
PCT_UNAVAILABLE_TEXT = "yüzde değişim hesaplanamadı (önceki değer 0)"


def _pct_change(current: float, previous: float) -> float | None:
    """Percent change from ``previous`` to ``current``; None when ``previous`` is 0.

    The base enters as its absolute value so the sign of the result always
    follows the direction of the move, also for negative series.
    """
    if previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def _coverage(data: dict) -> dict:
    """What the weekly fetch says about the window it actually covers."""
    dates = data.get("dates") or []
    return {
        "first_date": dates[0] if dates else "",
        "unit": data.get("unit", ""),
        "bulletin_date": data.get("bulletin_date", ""),
        "requested_days": data.get("requested_days"),
        "returned_observations": data.get("returned_observations", len(dates)),
        "returned_first_date": data.get("returned_first_date", ""),
        "trimmed_observations": data.get("trimmed_observations", 0),
        "window_met": data.get("window_met"),
    }


def _trend_label(points: list[float]) -> str:
    """Label the movement over up to the last 4 points by net change.

    With fewer than 4 points only the last step is used. With 4, the label
    follows the net move over the window (last minus first).
    """
    if len(points) < 4:
        net = points[-1] - points[-2]
    else:
        net = points[-1] - points[-4]
    return "yükseliş" if net > 0 else "düşüş" if net < 0 else "yatay"


async def analyze_trends(
    http: httpx.AsyncClient,
    metric_id: str = "1.0.1",
    currency: str = "TRY",
    column: str = "1",
    lookback_weeks: int = 12,
) -> dict:
    """Fetch a metric's time-series and compute trend statistics.

    Returns dict with: title, current, previous, wow_change, wow_pct,
    avg, min, max, min_date, max_date, trend_direction, narrative, series.
    wow_pct is None when the previous value is 0 (no percentage exists). A series
    with fewer than 2 points returns an error with reason "insufficient_data" and
    trimmed_observations (how many of BDDK's points fell outside the window). The
    result also carries the coverage of the window (first_date, unit,
    returned_observations, returned_first_date, window_met); an unknown metric id
    is an error with reason "unknown_metric".
    """
    data = await fetch_weekly_bulletin(
        http,
        metric_id=metric_id,
        currency=currency,
        days=lookback_weeks * 7,
        column=column,
    )

    if "error" in data:
        return {key: data[key] for key in ("error", "reason") if key in data}

    dates = data.get("dates", [])
    values = data.get("values", [])

    if len(values) < 2:
        return {
            "error": f"En az 2 gözlem gerekir (bulunan: {len(values)}).",
            "reason": REASON_INSUFFICIENT_DATA,
            "trimmed_observations": data.get("trimmed_observations", 0),
        }

    parsed = values

    current = parsed[-1]
    previous = parsed[-2]
    wow_change = current - previous
    wow_pct = _pct_change(current, previous)

    avg_val = sum(parsed) / len(parsed)
    min_val = min(parsed)
    max_val = max(parsed)
    min_idx = parsed.index(min_val)
    max_idx = parsed.index(max_val)

    trend = _trend_label(parsed)
    points_used = min(len(parsed), 4)
    basis = (
        f"Son {points_used} veri noktasına göre"
        if len(parsed) >= 4
        else "Yalnızca son adıma göre (4'ten az veri noktası)"
    )

    # Narrative
    title = data.get("title") or metric_id
    direction = "arttı" if wow_change > 0 else "azaldı" if wow_change < 0 else "değişmedi"
    change_text = f"%{abs(wow_pct):.2f} {direction}" if wow_pct is not None else f"{direction}; {PCT_UNAVAILABLE_TEXT}"

    narrative = (
        f"{title}: Son haftada {change_text}. "
        f"{basis} trend: {trend}. "
        f"Mevcut değer: {format_bulletin_number(current)}, ortalama: {format_bulletin_number(avg_val)}."
    )

    return {
        **_coverage(data),
        "title": title,
        "current": current,
        "current_date": dates[-1] if dates else "",
        "previous": previous,
        "previous_date": dates[-2] if len(dates) >= 2 else "",
        "wow_change": wow_change,
        "wow_pct": wow_pct,
        "avg": avg_val,
        "min": min_val,
        "min_date": dates[min_idx] if min_idx < len(dates) else "",
        "max": max_val,
        "max_date": dates[max_idx] if max_idx < len(dates) else "",
        "trend_direction": trend,
        "narrative": narrative,
        "data_points": len(parsed),
        "series": list(zip(dates, parsed, strict=False)),
    }


# -- Regulatory Digest -----------------------------------------------------


async def build_digest(
    http: httpx.AsyncClient,
    decisions_cache: list[dict],
    period_days: int = 30,
    clock: Callable[[], datetime] | None = None,
) -> dict:
    """Build a regulatory change digest for the given period.

    Combines: recent decisions from cache + announcements + bulletin snapshot.

    Args:
        decisions_cache: The current list of cached decision dicts from BddkApiClient.
        period_days: How many days back to look.
        clock: Injectable "now" for tests; the default is the Istanbul (UTC+3) time.

    Returns dict with: period, new_decisions, announcements, bulletin_summary, narrative.
    The decision count is a count of the local corpus catalog, not of BDDK:
    latest_catalog_decision_date is the newest decision date in that catalog and
    catalog_covers_period is False when it is older than the start of the period
    (the count then says nothing about what BDDK published). Announcements are
    newest first across categories. The bulletin block is the weekly page's
    Krediler table with its bulletin_heading, bulletin_date and bulletin_unit.
    """
    today = istanbul_today(clock)  # Istanbul date, not the host's
    cutoff_str = today.strftime("%d.%m.%Y")
    # window = period_days calendar days, today included
    window_start = today - timedelta(days=period_days - 1)

    # -- Recent decisions from cache --
    # Cache items may be Pydantic models or dicts; normalize access.
    recent_decisions: list[dict] = []
    latest_catalog_date = None
    for d in decisions_cache:
        if hasattr(d, "model_dump"):
            dd = d.model_dump()
        elif isinstance(d, dict):
            dd = d
        else:
            continue
        date_str = dd.get("decision_date", "")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y")
        except ValueError:
            continue
        if dt.date() <= today and (latest_catalog_date is None or dt.date() > latest_catalog_date):
            latest_catalog_date = dt.date()
        if 0 <= (today - dt.date()).days < period_days:
            recent_decisions.append(dd)
    catalog_covers_period = latest_catalog_date is not None and latest_catalog_date >= window_start

    # Sort by date descending
    recent_decisions.sort(
        key=lambda x: datetime.strptime(x["decision_date"], "%d.%m.%Y"),
        reverse=True,
    )

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for d in recent_decisions:
        cat = d.get("category", "Diğer")
        by_category.setdefault(cat, []).append(d)

    # -- Recent announcements (all 5 categories) --
    all_announcements: list[dict] = []
    announcements_available = True
    for cat_id in ANNOUNCEMENT_CATEGORY_IDS:
        try:
            anns = await fetch_announcements(http, cat_id)
        except BddkUpstreamError:
            # Upstream failed; the digest must say so instead of silently
            # reporting zero announcements. Discard whatever earlier categories
            # returned: a partial set reported alongside an unavailable flag
            # would still be read as a complete count.
            announcements_available = False
            all_announcements = []
            break
        for a in anns:
            date_str = a.get("date", "")
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%d.%m.%Y")
                if 0 <= (today - dt.date()).days < period_days:
                    all_announcements.append(a)
            except ValueError:
                continue

    # Newest first across categories, so a cut keeps the most recent ones.
    all_announcements.sort(key=lambda a: datetime.strptime(a["date"], "%d.%m.%Y"), reverse=True)

    # -- Bulletin snapshot --
    bulletin_snapshot_available = True
    try:
        snapshot_table = await fetch_bulletin_snapshot_table(http)
    except BddkUpstreamError:
        snapshot_table = {}
        bulletin_snapshot_available = False
    snapshot = snapshot_table.get("rows") or []

    # -- Build narrative --
    latest_text = latest_catalog_date.strftime("%d.%m.%Y") if latest_catalog_date else ""
    narrative_parts = [
        f"Yerel karar kataloğunda son {period_days} gün için {len(recent_decisions)} karar var "
        f"(katalogdaki en yeni karar tarihi: {latest_text or 'katalogda tarihli karar yok'})."
    ]
    if not catalog_covers_period:
        narrative_parts.append(
            "Katalog bu dönemi kapsamıyor; karar sayısı BDDK'nın bu dönemde yayımladığı kararlar hakkında "
            "bir şey söylemez (kararlar için BDDK'nın sitesine ve duyurulara bakın)."
        )
    if announcements_available:
        narrative_parts.append(f"BDDK sitesinden bu dönem için {len(all_announcements)} duyuru alındı.")
    else:
        narrative_parts.append(
            "Duyuru verileri alınamadı (BDDK üst kaynağına erişilemedi); duyuru sayıları bu özette eksiktir."
        )

    narrative_parts.append(
        "Kararlar yerel corpus kataloğundan gelir, canlı bir akış değildir; katalog güncel değilse yeni kararlar görünmez."
    )

    if by_category:
        top_cats = sorted(by_category.items(), key=lambda x: -len(x[1]))[:3]
        cats_summary = ", ".join(f"{cat} ({len(items)})" for cat, items in top_cats)
        narrative_parts.append(f"En aktif kategoriler: {cats_summary}.")

    return {
        "period_days": period_days,
        "cutoff_date": cutoff_str,
        "new_decisions": recent_decisions[:20],
        "decisions_by_category": {k: len(v) for k, v in by_category.items()},
        "announcements": all_announcements[:20],
        "announcements_available": announcements_available,
        "bulletin_snapshot": snapshot[:5] if snapshot else [],
        "bulletin_snapshot_available": bulletin_snapshot_available,
        "bulletin_heading": snapshot_table.get("heading", ""),
        "bulletin_date": snapshot_table.get("bulletin_date", ""),
        "bulletin_unit": snapshot_table.get("unit", ""),
        "latest_catalog_decision_date": latest_text,
        "catalog_covers_period": catalog_covers_period,
        "narrative": " ".join(narrative_parts),
        "total_decisions": len(recent_decisions),
        "total_announcements": len(all_announcements),
    }


# -- Compare Metrics -------------------------------------------------------


async def compare_metrics(
    http: httpx.AsyncClient,
    metric_ids: list[str],
    currency: str = "TRY",
    column: str = "1",
    days: int = 90,
) -> dict:
    """Compare two or more bulletin metrics side-by-side.

    Returns dict with: metrics (list of {title, current, wow_pct, trend}), comparison.
    """
    results = []
    for mid in metric_ids:  # the tool schema caps the list (MAX_METRIC_IDS)
        data = await fetch_weekly_bulletin(
            http,
            metric_id=mid,
            currency=currency,
            days=days,
            column=column,
        )
        if "error" in data:
            results.append({"metric_id": mid, **{key: data[key] for key in ("error", "reason") if key in data}})
            continue

        values = data.get("values", [])
        dates = data.get("dates", [])

        if len(values) < 2:
            results.append(
                {
                    "metric_id": mid,
                    "error": f"En az 2 gözlem gerekir (bulunan: {len(values)}).",
                    "reason": REASON_INSUFFICIENT_DATA,
                    "trimmed_observations": data.get("trimmed_observations", 0),
                }
            )
            continue
        current = values[-1]
        previous = values[-2]
        wow_pct = _pct_change(current, previous)

        results.append(
            {
                **_coverage(data),
                "metric_id": mid,
                "title": data.get("title") or mid,
                "current": current,
                "current_date": dates[-1] if dates else "",
                "wow_pct": wow_pct,
                "data_points": len(values),
            }
        )

    return {"metrics": results, "currency": currency, "column": column}
