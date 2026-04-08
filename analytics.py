"""BDDK analytics: trend analysis, digest, comparison, update detection."""

from datetime import datetime

import httpx

from data_sources import (
    fetch_announcements,
    fetch_bulletin_snapshot,
    fetch_weekly_bulletin,
)

# -- Trend Analysis --------------------------------------------------------


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
    """
    data = await fetch_weekly_bulletin(
        http,
        metric_id=metric_id,
        currency=currency,
        days=lookback_weeks * 7,
        column=column,
    )

    if "error" in data:
        return {"error": data["error"]}

    dates = data.get("dates", [])
    values = data.get("values", [])

    if len(values) < 2:
        return {"error": "Not enough data points for trend analysis."}

    # Parse numeric values (they may come as strings or floats)
    parsed = []
    for v in values:
        try:
            parsed.append(float(v) if not isinstance(v, (int, float)) else v)
        except (ValueError, TypeError):
            parsed.append(0.0)

    current = parsed[-1]
    previous = parsed[-2]
    wow_change = current - previous
    wow_pct = (wow_change / previous * 100) if previous != 0 else 0.0

    avg_val = sum(parsed) / len(parsed)
    min_val = min(parsed)
    max_val = max(parsed)
    min_idx = parsed.index(min_val)
    max_idx = parsed.index(max_val)

    # Trend direction over last 4 data points
    if len(parsed) >= 4:
        recent = parsed[-4:]
        ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        trend = "yükseliş" if ups >= 2 else "düşüş" if ups == 0 else "yatay"
    else:
        trend = "yükseliş" if wow_change > 0 else "düşüş" if wow_change < 0 else "yatay"

    # Narrative
    title = data.get("title", metric_id)
    direction = "arttı" if wow_change > 0 else "azaldı" if wow_change < 0 else "değişmedi"

    narrative = (
        f"{title}: Son haftada %{abs(wow_pct):.2f} {direction}. "
        f"Son {len(parsed)} haftalık trend: {trend}. "
        f"Mevcut değer: {current:,.2f}, ortalama: {avg_val:,.2f}."
    )

    return {
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
) -> dict:
    """Build a regulatory change digest for the given period.

    Combines: recent decisions from cache + announcements + bulletin snapshot.

    Args:
        decisions_cache: The current list of cached decision dicts from BddkApiClient.
        period_days: How many days back to look.

    Returns dict with: period, new_decisions, announcements, bulletin_summary, narrative.
    """
    cutoff = datetime.now()
    cutoff_str = cutoff.strftime("%d.%m.%Y")

    # -- Recent decisions from cache --
    # Cache items may be Pydantic models or dicts; normalize access.
    recent_decisions: list[dict] = []
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
            diff = (cutoff - dt).days
            if 0 <= diff <= period_days:
                recent_decisions.append(dd)
        except ValueError:
            continue

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
    for cat_id in [39, 40, 41, 42, 48]:  # Press, regulation, HR, data, institution
        anns = await fetch_announcements(http, cat_id)
        for a in anns:
            date_str = a.get("date", "")
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%d.%m.%Y")
                if 0 <= (cutoff - dt).days <= period_days:
                    all_announcements.append(a)
            except ValueError:
                continue

    # -- Bulletin snapshot --
    snapshot = await fetch_bulletin_snapshot(http)

    # -- Build narrative --
    narrative_parts = []
    narrative_parts.append(
        f"Son {period_days} günde {len(recent_decisions)} yeni karar ve {len(all_announcements)} duyuru yayımlandı."
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
        "bulletin_snapshot": snapshot[:5] if snapshot else [],
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
    for mid in metric_ids[:4]:  # Max 4 metrics
        data = await fetch_weekly_bulletin(
            http,
            metric_id=mid,
            currency=currency,
            days=days,
            column=column,
        )
        if "error" in data:
            results.append({"metric_id": mid, "error": data["error"]})
            continue

        values = data.get("values", [])
        dates = data.get("dates", [])

        if len(values) >= 2:
            current = float(values[-1]) if not isinstance(values[-1], (int, float)) else values[-1]
            previous = float(values[-2]) if not isinstance(values[-2], (int, float)) else values[-2]
            wow_pct = ((current - previous) / previous * 100) if previous else 0
        else:
            current = float(values[-1]) if values else 0
            previous = 0
            wow_pct = 0

        results.append(
            {
                "metric_id": mid,
                "title": data.get("title", mid),
                "current": current,
                "current_date": dates[-1] if dates else "",
                "wow_pct": wow_pct,
                "data_points": len(values),
            }
        )

    return {"metrics": results, "currency": currency, "column": column}


# -- Update Detection -------------------------------------------------------


async def check_updates(
    http: httpx.AsyncClient,
    known_decisions: list[dict],
    known_announcement_ids: set[str] | None = None,
) -> dict:
    """Compare current BDDK state with known state to detect new items.

    Args:
        known_decisions: Previously cached decisions list.
        known_announcement_ids: Set of known announcement URLs.

    Returns dict with: new_decisions_count, new_announcements, details.
    """
    # Check announcements (fast — just one page per category)
    new_announcements: list[dict] = []
    if known_announcement_ids is None:
        known_announcement_ids = set()

    for cat_id in [39, 40, 41, 42, 48]:  # All 5 categories
        anns = await fetch_announcements(http, cat_id)
        for a in anns:
            if a.get("url") and a["url"] not in known_announcement_ids:
                new_announcements.append(a)

    # For decisions, we compare snapshot counts
    # (Full re-fetch is expensive, so we just report announcement changes)
    return {
        "new_announcements": new_announcements[:10],
        "new_announcements_count": len(new_announcements),
        "checked_categories": [
            "Basın Duyurusu",
            "Mevzuat Duyurusu",
            "İnsan Kaynakları Duyurusu",
            "Veri Yayımlama Duyurusu",
            "Kuruluş Duyurusu",
        ],
    }
