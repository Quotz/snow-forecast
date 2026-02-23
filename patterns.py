"""Pattern detection for multi-day weather sequences.

Detects high-value skiing patterns like "storm then clear" (bluebird powder day),
multi-day storms, warming trends, cold snaps, upslope events, wind slab risk,
and melt-freeze crust. Supports near-miss detection and combined patterns.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default thresholds — overridden by config["patterns"] when provided
_DEFAULTS = {
    "storm_then_clear": {
        "min_snow_cm": 10,
        "min_cloud_storm": 70,
        "max_cloud_clear": 40,
        "max_wind_clear": 20,
        "max_temp_clear": -2,
        "perfect_cloud": 20,
        "perfect_sunshine_hours": 6,
    },
    "multi_day_storm": {
        "min_daily_snow_cm": 3,
        "min_consecutive_days": 3,
    },
    "warming_trend": {
        "min_freezing_level": 2000,
    },
    "cold_snap": {
        "min_temp_drop": 5,
        "max_temp": -5,
    },
    "upslope_event": {
        "wind_dir_min_deg": 120,
        "wind_dir_max_deg": 220,
        "min_humidity": 70,
        "max_temp": 0,
    },
    "wind_slab_risk": {
        "min_wind_kmh": 25,
        "min_recent_snow_cm": 5,
        "recent_snow_days": 2,
    },
    "melt_freeze_crust": {
        "min_day_temp": 0,
        "max_night_temp": -3,
        "min_snow_depth_m": 0.3,
        "max_fresh_snow_cm": 2,
        "night_temp_offset": 5,
    },
    "near_miss_relax_pct": 25,
}


def _get_cfg(config, pattern_name, key):
    """Get a config value with fallback to defaults."""
    if config:
        section = config.get("patterns", {}).get(pattern_name, {})
        if isinstance(section, dict) and key in section:
            return section[key]
    default_section = _DEFAULTS.get(pattern_name)
    if isinstance(default_section, dict):
        return default_section.get(key, _DEFAULTS.get(key))
    # Top-level key (e.g. near_miss_relax_pct)
    return default_section if default_section is not None else _DEFAULTS.get(key)


def _relax_threshold(value, relax_pct, direction="lower"):
    """Relax a threshold by a percentage. direction='lower' reduces it, 'higher' increases it."""
    factor = relax_pct / 100.0
    if direction == "lower":
        return value * (1 - factor)
    else:
        return value * (1 + factor)


def detect_storm_then_clear(scores: list, config=None) -> list:
    """Detect the holy grail: storm day followed by clearing/bluebird day.

    Looks for:
    - Day N: snowfall_24h > min_snow_cm, cloud_cover > min_cloud_storm
    - Day N+1: cloud_cover < max_cloud_clear, wind < max_wind_clear, temp < max_temp_clear
    """
    cfg = lambda k: _get_cfg(config, "storm_then_clear", k)
    min_snow = cfg("min_snow_cm")
    min_cloud_storm = cfg("min_cloud_storm")
    max_cloud_clear = cfg("max_cloud_clear")
    max_wind_clear = cfg("max_wind_clear")
    max_temp_clear = cfg("max_temp_clear")
    perfect_cloud = cfg("perfect_cloud")
    perfect_sun = cfg("perfect_sunshine_hours")

    patterns = _detect_storm_then_clear_with_thresholds(
        scores, min_snow, min_cloud_storm, max_cloud_clear,
        max_wind_clear, max_temp_clear, perfect_cloud, perfect_sun
    )

    # Near-miss detection
    relax_pct = _get_cfg(config, "near_miss_relax_pct", "near_miss_relax_pct")
    if relax_pct is None:
        relax_pct = _DEFAULTS["near_miss_relax_pct"]
    strict_indices = {(p["storm_day_index"], p["clear_day_index"]) for p in patterns}

    relaxed = _detect_storm_then_clear_with_thresholds(
        scores,
        _relax_threshold(min_snow, relax_pct, "lower"),
        _relax_threshold(min_cloud_storm, relax_pct, "lower"),
        _relax_threshold(max_cloud_clear, relax_pct, "higher"),
        _relax_threshold(max_wind_clear, relax_pct, "higher"),
        max_temp_clear + (abs(max_temp_clear) * relax_pct / 100),
        perfect_cloud, perfect_sun
    )

    for p in relaxed:
        key = (p["storm_day_index"], p["clear_day_index"])
        if key not in strict_indices:
            p["quality"] = "NEAR_MISS"
            snow = p.get("storm_snow_cm", 0)
            p["message"] = (
                f"Close to bluebird setup: {snow:.0f}cm (needs {min_snow}) "
                f"with clearing Day {p['clear_day_index'] + 1}."
            )
            patterns.append(p)

    return patterns


def _detect_storm_then_clear_with_thresholds(
    scores, min_snow, min_cloud_storm, max_cloud_clear,
    max_wind_clear, max_temp_clear, perfect_cloud, perfect_sun
):
    """Core storm-then-clear detection with given thresholds."""
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

        is_storm_day = snow_n >= min_snow and cloud_n >= min_cloud_storm
        is_clear_day = cloud_n1 < max_cloud_clear and wind_n1 < max_wind_clear and temp_n1 < max_temp_clear

        if is_storm_day and is_clear_day:
            quality = "PERFECT" if cloud_n1 < perfect_cloud and sun_n1 > perfect_sun else "GOOD"
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


def detect_multi_day_storm(scores: list, config=None) -> list:
    """Detect multi-day storm sequences (3+ consecutive days with snowfall).

    These are the big accumulation events — 40-80+ cm over several days.
    """
    min_daily = _get_cfg(config, "multi_day_storm", "min_daily_snow_cm")
    min_consec = _get_cfg(config, "multi_day_storm", "min_consecutive_days")

    patterns = _detect_multi_day_storm_with_thresholds(scores, min_daily, min_consec)

    # Near-miss detection
    relax_pct = _get_cfg(config, "near_miss_relax_pct", "near_miss_relax_pct")
    if relax_pct is None:
        relax_pct = _DEFAULTS["near_miss_relax_pct"]
    strict_keys = {(p["start_index"], p["end_index"]) for p in patterns}

    relaxed = _detect_multi_day_storm_with_thresholds(
        scores,
        _relax_threshold(min_daily, relax_pct, "lower"),
        min_consec,  # Don't relax day count
    )

    for p in relaxed:
        key = (p["start_index"], p["end_index"])
        if key not in strict_keys:
            p["quality"] = "NEAR_MISS"
            p["message"] = (
                f"Close to multi-day storm: {p['total_snow_cm']:.0f}cm over "
                f"{p['duration_days']} days (needs {min_daily}cm/day minimum)."
            )
            patterns.append(p)

    return patterns


def _detect_multi_day_storm_with_thresholds(scores, min_daily, min_consec):
    """Core multi-day storm detection with given thresholds."""
    patterns = []
    streak_start = None
    streak_total = 0

    for i, day in enumerate(scores):
        snow = day.get("conditions", {}).get("snowfall_24h_cm", 0)

        if snow >= min_daily:
            if streak_start is None:
                streak_start = i
                streak_total = snow
            else:
                streak_total += snow
        else:
            if streak_start is not None and (i - streak_start) >= min_consec:
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
    if streak_start is not None and (len(scores) - streak_start) >= min_consec:
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


def detect_warming_trend(scores: list, config=None) -> list:
    """Detect dangerous warming trends where freezing level approaches freeride zone.

    Warns when freezing level is rising toward 2000m+ over consecutive days.
    """
    min_fl = _get_cfg(config, "warming_trend", "min_freezing_level")
    patterns = []

    for i in range(2, len(scores)):
        fl_2 = scores[i-2].get("conditions", {}).get("freezing_level_m", 0)
        fl_1 = scores[i-1].get("conditions", {}).get("freezing_level_m", 0)
        fl_0 = scores[i].get("conditions", {}).get("freezing_level_m", 0)

        if fl_2 and fl_1 and fl_0:
            rising = fl_0 > fl_1 > fl_2
            high = fl_0 > min_fl

            if rising and high:
                patterns.append({
                    "type": "warming_trend",
                    "day_index": i,
                    "freezing_levels": [fl_2, fl_1, fl_0],
                    "message": f"Warning: Freezing level rising ({fl_2:.0f} → "
                               f"{fl_1:.0f} → {fl_0:.0f}m). Rain risk at freeride elevations.",
                })

    # Near-miss: relaxed freezing level
    relax_pct = _get_cfg(config, "near_miss_relax_pct", "near_miss_relax_pct")
    if relax_pct is None:
        relax_pct = _DEFAULTS["near_miss_relax_pct"]
    relaxed_fl = _relax_threshold(min_fl, relax_pct, "lower")
    strict_indices = {p["day_index"] for p in patterns}

    for i in range(2, len(scores)):
        if i in strict_indices:
            continue
        fl_2 = scores[i-2].get("conditions", {}).get("freezing_level_m", 0)
        fl_1 = scores[i-1].get("conditions", {}).get("freezing_level_m", 0)
        fl_0 = scores[i].get("conditions", {}).get("freezing_level_m", 0)

        if fl_2 and fl_1 and fl_0:
            rising = fl_0 > fl_1 > fl_2
            high = fl_0 > relaxed_fl

            if rising and high:
                patterns.append({
                    "type": "warming_trend",
                    "quality": "NEAR_MISS",
                    "day_index": i,
                    "freezing_levels": [fl_2, fl_1, fl_0],
                    "message": f"Close to warming concern: Freezing level rising to "
                               f"{fl_0:.0f}m (threshold {min_fl:.0f}m).",
                })

    return patterns


def detect_cold_snap(scores: list, config=None) -> list:
    """Detect incoming cold snap — good for powder preservation.

    When temps drop significantly (>5C) over 2 days, existing snow will be preserved.
    """
    min_drop = _get_cfg(config, "cold_snap", "min_temp_drop")
    max_temp = _get_cfg(config, "cold_snap", "max_temp")
    patterns = []

    for i in range(1, len(scores)):
        temp_prev = scores[i-1].get("conditions", {}).get("temperature_c", 0)
        temp_now = scores[i].get("conditions", {}).get("temperature_c", 0)

        if temp_prev is not None and temp_now is not None:
            drop = temp_prev - temp_now
            if drop > min_drop and temp_now < max_temp:
                patterns.append({
                    "type": "cold_snap",
                    "day_index": i,
                    "temp_drop": round(drop, 1),
                    "new_temp": round(temp_now, 1),
                    "message": f"Cold snap Day {i+1}: {drop:.0f}°C drop to {temp_now:.0f}°C. "
                               f"Great for powder preservation.",
                })

    # Near-miss detection
    relax_pct = _get_cfg(config, "near_miss_relax_pct", "near_miss_relax_pct")
    if relax_pct is None:
        relax_pct = _DEFAULTS["near_miss_relax_pct"]
    strict_indices = {p["day_index"] for p in patterns}

    relaxed_drop = _relax_threshold(min_drop, relax_pct, "lower")
    relaxed_temp = max_temp + (abs(max_temp) * relax_pct / 100)

    for i in range(1, len(scores)):
        if i in strict_indices:
            continue
        temp_prev = scores[i-1].get("conditions", {}).get("temperature_c", 0)
        temp_now = scores[i].get("conditions", {}).get("temperature_c", 0)

        if temp_prev is not None and temp_now is not None:
            drop = temp_prev - temp_now
            if drop > relaxed_drop and temp_now < relaxed_temp:
                patterns.append({
                    "type": "cold_snap",
                    "quality": "NEAR_MISS",
                    "day_index": i,
                    "temp_drop": round(drop, 1),
                    "new_temp": round(temp_now, 1),
                    "message": f"Close to cold snap Day {i+1}: {drop:.0f}°C drop to "
                               f"{temp_now:.0f}°C (needs >{min_drop}°C drop below {max_temp}°C).",
                })

    return patterns


def detect_upslope_event(scores: list, config=None) -> list:
    """Detect upslope precipitation events.

    S/SE/SW wind (120-220 deg) + humidity > 70% + temp < 0C at summit.
    """
    wind_dir_min = _get_cfg(config, "upslope_event", "wind_dir_min_deg")
    wind_dir_max = _get_cfg(config, "upslope_event", "wind_dir_max_deg")
    min_humidity = _get_cfg(config, "upslope_event", "min_humidity")
    max_temp = _get_cfg(config, "upslope_event", "max_temp")
    patterns = []

    for i, day in enumerate(scores):
        cond = day.get("conditions", {})
        wind_dir = cond.get("wind_direction_deg")
        humidity = cond.get("humidity_avg")
        temp = cond.get("temperature_c")

        if wind_dir is None or humidity is None or temp is None:
            continue

        is_upslope_wind = wind_dir_min <= wind_dir <= wind_dir_max
        is_humid = humidity >= min_humidity
        is_cold = temp < max_temp

        if is_upslope_wind and is_humid and is_cold:
            patterns.append({
                "type": "upslope_event",
                "day_index": i,
                "wind_direction_deg": wind_dir,
                "humidity": humidity,
                "temperature_c": temp,
                "message": f"Upslope event Day {i+1}: S/SE wind at {wind_dir:.0f}°, "
                           f"humidity {humidity:.0f}%, temp {temp:.0f}°C. "
                           f"Enhanced orographic snowfall likely.",
            })

    return patterns


def detect_wind_slab_risk(scores: list, config=None) -> list:
    """Detect wind slab risk conditions.

    Wind > 25 km/h sustained + fresh snow > 5cm in previous 48h + wind loading
    on the primary north-facing aspect.
    """
    min_wind = _get_cfg(config, "wind_slab_risk", "min_wind_kmh")
    min_snow = _get_cfg(config, "wind_slab_risk", "min_recent_snow_cm")
    recent_days = _get_cfg(config, "wind_slab_risk", "recent_snow_days")
    patterns = []

    for i, day in enumerate(scores):
        cond = day.get("conditions", {})
        wind = cond.get("wind_speed_kmh", 0)
        wind_dir = cond.get("wind_direction_deg")

        if wind < min_wind:
            continue

        # Sum snowfall over current + previous days (up to recent_days)
        recent_snow = 0
        for j in range(max(0, i - recent_days + 1), i + 1):
            recent_snow += scores[j].get("conditions", {}).get("snowfall_24h_cm", 0)

        if recent_snow < min_snow:
            continue

        # Check if wind loads north-facing aspect (wind from S/SW/SE = 90-270 loads north)
        loads_north = False
        if wind_dir is not None:
            loads_north = 90 <= wind_dir <= 270
        else:
            # Without direction data, flag if wind + snow thresholds met
            loads_north = True

        if loads_north:
            patterns.append({
                "type": "wind_slab_risk",
                "day_index": i,
                "wind_speed_kmh": wind,
                "wind_direction_deg": wind_dir,
                "recent_snow_cm": round(recent_snow, 1),
                "message": f"Wind slab risk Day {i+1}: {wind:.0f} km/h wind with "
                           f"{recent_snow:.0f}cm recent snow. Wind loading on north aspects.",
            })

    return patterns


def detect_melt_freeze_crust(scores: list, config=None) -> list:
    """Detect melt-freeze crust conditions.

    Day temp > 0C, night temp < -3C, snow_depth > 30cm, no fresh snow (<2cm).
    """
    min_day_temp = _get_cfg(config, "melt_freeze_crust", "min_day_temp")
    max_night_temp = _get_cfg(config, "melt_freeze_crust", "max_night_temp")
    min_depth = _get_cfg(config, "melt_freeze_crust", "min_snow_depth_m")
    max_fresh = _get_cfg(config, "melt_freeze_crust", "max_fresh_snow_cm")
    night_offset = _get_cfg(config, "melt_freeze_crust", "night_temp_offset")
    patterns = []

    for i, day in enumerate(scores):
        cond = day.get("conditions", {})
        temp = cond.get("temperature_c")
        temp_min = cond.get("temperature_min_c")
        snow_depth = cond.get("snow_depth_m", 0)
        fresh_snow = cond.get("snowfall_24h_cm", 0)

        if temp is None:
            continue

        day_temp = temp
        night_temp = temp_min if temp_min is not None else (temp - night_offset)

        is_day_warm = day_temp > min_day_temp
        is_night_cold = night_temp < max_night_temp
        has_base = snow_depth >= min_depth
        no_fresh = fresh_snow < max_fresh

        if is_day_warm and is_night_cold and has_base and no_fresh:
            patterns.append({
                "type": "melt_freeze_crust",
                "day_index": i,
                "day_temp": round(day_temp, 1),
                "night_temp": round(night_temp, 1),
                "snow_depth_m": snow_depth,
                "message": f"Melt-freeze crust Day {i+1}: Day {day_temp:.0f}°C / "
                           f"Night {night_temp:.0f}°C with {snow_depth*100:.0f}cm base. "
                           f"Expect firm/icy morning, softening afternoon.",
            })

    return patterns


def _detect_combined_patterns(all_patterns: list, scores: list) -> list:
    """Detect combined patterns from individual detections."""
    combined = []

    storm_clear = [p for p in all_patterns if p["type"] == "storm_then_clear" and p.get("quality") != "NEAR_MISS"]
    cold_snaps = [p for p in all_patterns if p["type"] == "cold_snap" and p.get("quality") != "NEAR_MISS"]

    for sc in storm_clear:
        clear_day = sc["clear_day_index"]
        for cs in cold_snaps:
            cs_day = cs["day_index"]
            # Cold snap on the clear day or the day after
            if cs_day == clear_day or cs_day == clear_day + 1:
                combined.append({
                    "type": "legendary_setup",
                    "storm_day_index": sc["storm_day_index"],
                    "clear_day_index": clear_day,
                    "cold_snap_day_index": cs_day,
                    "storm_snow_cm": sc.get("storm_snow_cm", 0),
                    "temp": cs.get("new_temp", 0),
                    "message": f"Legendary setup: storm dumps {sc.get('storm_snow_cm', 0):.0f}cm, "
                               f"cold locks powder in, then clears Day {clear_day + 1}.",
                })
                break  # One legendary per storm_then_clear

    return combined


def detect_all_patterns(scores: list, config=None) -> list:
    """Run all pattern detectors and return combined list."""
    all_patterns = []
    all_patterns.extend(detect_storm_then_clear(scores, config))
    all_patterns.extend(detect_multi_day_storm(scores, config))
    all_patterns.extend(detect_warming_trend(scores, config))
    all_patterns.extend(detect_cold_snap(scores, config))
    all_patterns.extend(detect_upslope_event(scores, config))
    all_patterns.extend(detect_wind_slab_risk(scores, config))
    all_patterns.extend(detect_melt_freeze_crust(scores, config))

    # Combined pattern detection
    all_patterns.extend(_detect_combined_patterns(all_patterns, scores))

    # Sort by day index
    all_patterns.sort(key=lambda p: p.get(
        "storm_day_index",
        p.get("start_index", p.get("day_index", 0))
    ))

    return all_patterns
