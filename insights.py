"""Actionable insights generation from scored forecast data.

Produces structured insights for the dashboard and notifications:
headline, best day, action items, snowpack status, 7-day snow total.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _fmt_date_short(iso_date: str) -> str:
    """Format ISO date as 'Wed 25/02'."""
    try:
        d = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")
        return f"{DAY_NAMES[d.weekday()]} {d.strftime('%d/%m')}"
    except (ValueError, TypeError):
        return str(iso_date)


def generate_insights(scores, patterns, current=None):
    """Generate actionable insights from scored forecast data.

    Args:
        scores: list of score dicts (from scoring.py)
        patterns: list of pattern dicts (from patterns.py)
        current: dict of current conditions (optional)

    Returns dict with:
    - headline: str — one-line summary
    - best_day: dict — {date, score, label, reason}
    - action_items: list of str
    - snowpack_status: str
    - seven_day_snow_total: float
    """
    week = scores[:7]
    if not week:
        return None

    # Find best day
    best = max(week, key=lambda s: s["total"])
    best_date = best.get("date", "")
    best_date_short = _fmt_date_short(best_date)
    best_score = best["total"]
    best_label = best["label"]

    # Build reason string for best day
    cond = best.get("conditions", {})
    reason_parts = []
    snow = cond.get("snowfall_24h_cm", 0)
    if snow > 0:
        reason_parts.append(f"{snow:.0f}cm fresh")
    temp = cond.get("temperature_c")
    if temp is not None:
        reason_parts.append(f"{temp:.0f}C")
    wind = cond.get("wind_speed_kmh")
    if wind is not None:
        reason_parts.append(f"{wind:.0f}km/h")
    sky_class = best.get("sky", {}).get("classification", "")
    if sky_class == "BLUEBIRD":
        reason_parts.append("Bluebird")
    elif sky_class == "MOSTLY_SUNNY":
        reason_parts.append("Mostly Sunny")
    reason = " / ".join(reason_parts) if reason_parts else ""

    best_day = {
        "date": best_date,
        "score": best_score,
        "label": best_label,
        "reason": reason,
    }

    # 7-day snow total
    seven_day_snow_total = sum(
        s.get("conditions", {}).get("snowfall_24h_cm", 0) for s in week
    )

    # Headline
    # Check for relevant patterns
    pattern_str = ""
    for p in patterns:
        if p["type"] == "storm_then_clear":
            pattern_str = "Bluebird setup"
            break
        elif p["type"] == "multi_day_storm":
            pattern_str = "Multi-day storm"
            break

    if best_score >= 60:
        headline = f"Fresh {snow:.0f}cm {best_date_short}"
        if pattern_str:
            headline += f" — {pattern_str}"
    elif seven_day_snow_total > 10:
        snow_days = sum(1 for s in week if s.get("conditions", {}).get("snowfall_24h_cm", 0) >= 3)
        headline = f"{seven_day_snow_total:.0f}cm building over {snow_days} day{'s' if snow_days != 1 else ''}"
    else:
        depth_str = _get_depth_str(current)
        headline = f"Dry week ahead, {depth_str} base"

    # Action items
    action_items = []
    if best_score >= 40:
        action_items.append(f"Best skiing {best_date_short}")

    for i, s in enumerate(week):
        gust = s.get("conditions", {}).get("wind_gust_kmh", 0)
        if gust > 50:
            day_str = _fmt_date_short(s.get("date", ""))
            action_items.append(f"Avoid {day_str} ({gust:.0f} km/h gusts)")

    for p in patterns:
        if p["type"] == "warming_trend":
            idx = p.get("day_index", 0)
            if idx < len(scores):
                day_str = _fmt_date_short(scores[idx].get("date", ""))
                action_items.append(f"Watch freezing level {day_str}")
        elif p["type"] == "wind_slab_risk":
            idx = p.get("day_index", 0)
            if idx < len(scores):
                day_str = _fmt_date_short(scores[idx].get("date", ""))
                action_items.append(f"Caution: wind slab risk {day_str}")

    # Snowpack status
    snowpack_status = _compute_snowpack_status(scores, current)

    return {
        "headline": headline,
        "best_day": best_day if best_score >= 20 else None,
        "action_items": action_items,
        "snowpack_status": snowpack_status,
        "seven_day_snow_total": round(seven_day_snow_total, 1),
    }


def _get_depth_str(current):
    """Get snow depth string from current conditions."""
    if current and current.get("snow_depth") is not None:
        depth_cm = current["snow_depth"] * 100
        return f"{depth_cm:.0f}cm"
    return "unknown"


def _compute_snowpack_status(scores, current):
    """Compute snowpack status string: '{depth}cm ({trend})'."""
    depth_cm = 0
    if current and current.get("snow_depth") is not None:
        depth_cm = current["snow_depth"] * 100

    # Determine trend from first 3 days of snowfall and temperature
    trend = "stable"
    if len(scores) >= 3:
        snow_3d = sum(
            s.get("conditions", {}).get("snowfall_24h_cm", 0) for s in scores[:3]
        )
        avg_temp = sum(
            s.get("conditions", {}).get("temperature_c", 0) for s in scores[:3]
        ) / 3

        if snow_3d > 5:
            trend = "building"
        elif avg_temp > 0:
            trend = "melting"
        else:
            trend = "stable"

    if depth_cm > 0:
        return f"{depth_cm:.0f}cm ({trend})"
    return f"unknown ({trend})"
