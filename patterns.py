"""Pattern detection for multi-day weather sequences.

Detects high-value skiing patterns like "storm then clear" (bluebird powder day),
multi-day storms, warming trends, and other actionable sequences.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def detect_storm_then_clear(scores: list) -> list:
    """Detect the holy grail: storm day followed by clearing/bluebird day.

    Looks for:
    - Day N: snowfall_24h > 10cm, cloud_cover > 70%
    - Day N+1: cloud_cover < 40%, wind < 20 km/h, temp < -2°C

    Returns list of dicts with day indices and details.
    """
    patterns = []

    for i in range(len(scores) - 1):
        day_n = scores[i]
        day_n1 = scores[i + 1]

        cond_n = day_n.get("conditions", {})
        cond_n1 = day_n1.get("conditions", {})
        sky_n1 = day_n1.get("sky", {})

        snow_n = cond_n.get("snowfall_24h_cm", 0)
        cloud_n = day_n.get("sky", {}).get("cloud_cover", 100)
        cloud_n1 = sky_n1.get("cloud_cover", 100)
        wind_n1 = cond_n1.get("wind_speed_kmh", 50)
        temp_n1 = cond_n1.get("temperature_c", 0)
        sun_n1 = sky_n1.get("sunshine_hours", 0)

        is_storm_day = snow_n >= 10 and cloud_n >= 70
        is_clear_day = cloud_n1 < 40 and wind_n1 < 20 and temp_n1 < -2

        if is_storm_day and is_clear_day:
            quality = "PERFECT" if cloud_n1 < 20 and sun_n1 > 6 else "GOOD"
            patterns.append({
                "type": "storm_then_clear",
                "quality": quality,
                "storm_day_index": i,
                "clear_day_index": i + 1,
                "storm_snow_cm": snow_n,
                "clear_day_cloud": cloud_n1,
                "clear_day_sunshine": sun_n1,
                "clear_day_wind": wind_n1,
                "clear_day_temp": temp_n1,
                "message": f"{'Bluebird' if quality == 'PERFECT' else 'Clearing'} powder day! "
                           f"{snow_n:.0f}cm falls Day {i+1}, clears Day {i+2}.",
            })

    return patterns


def detect_multi_day_storm(scores: list) -> list:
    """Detect multi-day storm sequences (3+ consecutive days with snowfall).

    These are the big accumulation events — 40-80+ cm over several days.
    """
    patterns = []
    streak_start = None
    streak_total = 0

    for i, day in enumerate(scores):
        snow = day.get("conditions", {}).get("snowfall_24h_cm", 0)

        if snow >= 3:  # Counting days with at least 3cm
            if streak_start is None:
                streak_start = i
                streak_total = snow
            else:
                streak_total += snow
        else:
            if streak_start is not None and (i - streak_start) >= 3:
                patterns.append({
                    "type": "multi_day_storm",
                    "start_index": streak_start,
                    "end_index": i - 1,
                    "duration_days": i - streak_start,
                    "total_snow_cm": round(streak_total, 1),
                    "message": f"Multi-day storm: {streak_total:.0f}cm over "
                               f"{i - streak_start} days (Day {streak_start+1}-{i}).",
                })
            streak_start = None
            streak_total = 0

    # Check if streak continues to end of forecast
    if streak_start is not None and (len(scores) - streak_start) >= 3:
        patterns.append({
            "type": "multi_day_storm",
            "start_index": streak_start,
            "end_index": len(scores) - 1,
            "duration_days": len(scores) - streak_start,
            "total_snow_cm": round(streak_total, 1),
            "message": f"Multi-day storm building: {streak_total:.0f}cm over "
                       f"{len(scores) - streak_start}+ days.",
        })

    return patterns


def detect_warming_trend(scores: list) -> list:
    """Detect dangerous warming trends where freezing level approaches freeride zone.

    Warns when freezing level is rising toward 2000m+ over consecutive days.
    """
    patterns = []

    for i in range(2, len(scores)):
        fl_2 = scores[i-2].get("conditions", {}).get("freezing_level_m", 0)
        fl_1 = scores[i-1].get("conditions", {}).get("freezing_level_m", 0)
        fl_0 = scores[i].get("conditions", {}).get("freezing_level_m", 0)

        if fl_2 and fl_1 and fl_0:
            rising = fl_0 > fl_1 > fl_2
            high = fl_0 > 2000

            if rising and high:
                patterns.append({
                    "type": "warming_trend",
                    "day_index": i,
                    "freezing_levels": [fl_2, fl_1, fl_0],
                    "message": f"Warning: Freezing level rising ({fl_2:.0f} → "
                               f"{fl_1:.0f} → {fl_0:.0f}m). Rain risk at freeride elevations.",
                })

    return patterns


def detect_cold_snap(scores: list) -> list:
    """Detect incoming cold snap — good for powder preservation.

    When temps drop significantly (>5°C) over 2 days, existing snow will be preserved.
    """
    patterns = []

    for i in range(1, len(scores)):
        temp_prev = scores[i-1].get("conditions", {}).get("temperature_c", 0)
        temp_now = scores[i].get("conditions", {}).get("temperature_c", 0)

        if temp_prev is not None and temp_now is not None:
            drop = temp_prev - temp_now
            if drop > 5 and temp_now < -5:
                patterns.append({
                    "type": "cold_snap",
                    "day_index": i,
                    "temp_drop": round(drop, 1),
                    "new_temp": round(temp_now, 1),
                    "message": f"Cold snap Day {i+1}: {drop:.0f}°C drop to {temp_now:.0f}°C. "
                               f"Great for powder preservation.",
                })

    return patterns


def detect_all_patterns(scores: list) -> list:
    """Run all pattern detectors and return combined list."""
    all_patterns = []
    all_patterns.extend(detect_storm_then_clear(scores))
    all_patterns.extend(detect_multi_day_storm(scores))
    all_patterns.extend(detect_warming_trend(scores))
    all_patterns.extend(detect_cold_snap(scores))

    # Sort by day index
    all_patterns.sort(key=lambda p: p.get("storm_day_index", p.get("start_index", p.get("day_index", 0))))

    return all_patterns
