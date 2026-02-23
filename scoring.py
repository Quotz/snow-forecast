"""Powder Score algorithm.

Calculates a composite score (0-100) for each forecast day based on
snow conditions, temperature, wind, sky conditions, and model agreement.
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _safe(val, default=0):
    """Return val if not None, else default."""
    return val if val is not None else default


def _clamp(val, lo=0, hi=100):
    return max(lo, min(hi, val))


def _linear_score(val, zero_at, full_at, max_pts):
    """Linear interpolation between zero_at (0 points) and full_at (max_pts)."""
    if full_at == zero_at:
        return max_pts if val >= full_at else 0
    ratio = (val - zero_at) / (full_at - zero_at)
    return _clamp(ratio * max_pts, 0, max_pts)


def score_snow_quantity(snowfall_24h_cm: float, cfg: dict) -> float:
    """Score based on fresh snowfall amount (0-40 points).

    Diminishing returns above epic threshold.
    """
    if snowfall_24h_cm <= 0:
        return 0

    good = cfg.get("good_cm", 10)
    great = cfg.get("great_cm", 20)
    epic = cfg.get("epic_cm", 30)

    if snowfall_24h_cm <= good:
        return _linear_score(snowfall_24h_cm, 0, good, 20)
    elif snowfall_24h_cm <= great:
        return 20 + _linear_score(snowfall_24h_cm, good, great, 10)
    elif snowfall_24h_cm <= epic:
        return 30 + _linear_score(snowfall_24h_cm, great, epic, 5)
    else:
        # Diminishing returns: 35 + small bonus up to 40
        extra = min((snowfall_24h_cm - epic) / 20, 1.0) * 5
        return 35 + extra


def score_temperature(temp_c: float, cfg: dict) -> float:
    """Score based on summit temperature (0-15 points).

    Sweet spot is ideal_min to ideal_max (e.g. -8 to -3°C).
    """
    ideal_min = cfg.get("ideal_min", -8)
    ideal_max = cfg.get("ideal_max", -3)
    warm_pen = cfg.get("warm_penalty", -2)
    cold_pen = cfg.get("cold_penalty", -20)

    # In the sweet spot
    if ideal_min <= temp_c <= ideal_max:
        return 15.0

    # Warmer than ideal
    if temp_c > ideal_max:
        if temp_c >= warm_pen:
            return 0  # Too warm, wet snow
        return _linear_score(temp_c, warm_pen, ideal_max, 15)

    # Colder than ideal
    if temp_c < ideal_min:
        if temp_c <= cold_pen:
            return 5  # Very cold gets a floor — snow is still dry
        return 15 - _linear_score(ideal_min - temp_c, 0, ideal_min - cold_pen, 10)

    return 0


def score_wind(wind_speed_kmh: float, wind_gust_kmh: float, cfg: dict) -> float:
    """Score based on wind conditions (0-10 points).

    Low wind = powder stays put. High wind = transport and slab risk.
    """
    ideal_max = cfg.get("ideal_max_kmh", 10)
    poor = cfg.get("poor_kmh", 25)

    # Sustained wind score
    if wind_speed_kmh <= ideal_max:
        wind_pts = 10.0
    elif wind_speed_kmh >= poor:
        wind_pts = 0
    else:
        wind_pts = _linear_score(poor - wind_speed_kmh, 0, poor - ideal_max, 10)

    # Gust penalty — even if sustained is OK, gusts damage powder
    gust_danger = cfg.get("gust_danger_kmh", 50)
    if wind_gust_kmh > gust_danger:
        wind_pts *= 0.3  # Severe penalty
    elif wind_gust_kmh > poor:
        gust_factor = 1.0 - 0.7 * ((wind_gust_kmh - poor) / (gust_danger - poor))
        wind_pts *= max(gust_factor, 0.3)

    return wind_pts


def score_freezing_level(freezing_level_m: float, cfg: dict) -> float:
    """Score based on freezing level height (0-5 points).

    Below ideal = all snow. Above danger = rain at freeride elevations.
    """
    ideal = cfg.get("ideal_below_m", 1800)
    danger = cfg.get("danger_above_m", 2200)

    if freezing_level_m <= ideal:
        return 5.0
    elif freezing_level_m >= danger:
        return 0
    else:
        return _linear_score(danger - freezing_level_m, 0, danger - ideal, 5)


def score_storm_timing(hourly_snowfall: list, hourly_times: list) -> float:
    """Score based on when snowfall stops relative to morning (0-5 points).

    Best: storm ends overnight (02:00-07:00). Powder is fresh but settled by morning.
    Good: storm ends in afternoon (still falling when you ski).
    """
    if not hourly_snowfall or not hourly_times:
        return 0

    # Find the last hour with snowfall > 0.1 cm
    last_snow_hour = None
    for i in range(len(hourly_snowfall) - 1, -1, -1):
        if _safe(hourly_snowfall[i]) > 0.1:
            last_snow_hour = hourly_times[i]
            break

    if last_snow_hour is None:
        return 0  # No snowfall at all

    # Extract hour from time string
    try:
        if "T" in str(last_snow_hour):
            hour = int(str(last_snow_hour).split("T")[1][:2])
        else:
            hour = 12  # Default midday
    except (ValueError, IndexError):
        return 2.5  # Can't parse, give middle score

    # Overnight ending (02:00-07:00) = best
    if 2 <= hour <= 7:
        return 5.0
    # Early morning (07:00-10:00) = good, snow just stopping
    elif 7 < hour <= 10:
        return 4.0
    # Afternoon (still snowing during ski hours)
    elif 10 < hour <= 16:
        return 3.0
    # Evening/night (tomorrow will be the day)
    else:
        return 2.0


def score_snow_depth_trend(current_depth: float, depths_3day: list) -> float:
    """Score based on snow depth trend (0-5 points).

    Increasing = good base building. Decreasing = melt cycle.
    """
    if not depths_3day or len(depths_3day) < 2:
        return 2.5  # Unknown, give middle score

    valid = [d for d in depths_3day if d is not None]
    if len(valid) < 2:
        return 2.5

    trend = valid[-1] - valid[0]  # positive = increasing

    if trend > 0.05:  # Increasing by 5+ cm
        return 5.0
    elif trend > 0:
        return 4.0
    elif trend > -0.05:
        return 2.0  # Slight decrease (normal settling)
    else:
        return 0  # Significant melt


def score_model_agreement(snowfall_values: list) -> float:
    """Score based on how well models agree (0-20 points).

    Low standard deviation across models = high confidence.
    """
    valid = [v for v in snowfall_values if v is not None and v >= 0]

    if len(valid) < 2:
        return 10  # Only one model, middle confidence

    mean = sum(valid) / len(valid)
    if mean < 0.5:
        # All models agree there's basically no snow — high agreement
        return 18

    variance = sum((v - mean) ** 2 for v in valid) / len(valid)
    stdev = math.sqrt(variance)

    # Coefficient of variation (relative agreement)
    cv = stdev / mean if mean > 0 else 0

    if cv < 0.2:
        return 20  # Models agree within 20% — very high confidence
    elif cv < 0.4:
        return 15
    elif cv < 0.6:
        return 10
    elif cv < 0.8:
        return 5
    else:
        return 0  # Models wildly disagree


def score_sky_conditions(cloud_cover: float, cloud_cover_low: float,
                         sunshine_hours: float, visibility_m: float,
                         cfg: dict) -> dict:
    """Score sky/experience conditions. Returns dict with classification and details.

    Not part of the main powder score (since stormy days ARE powder days),
    but tracked for the "day after" bluebird detection.
    """
    bluebird_max = cfg.get("bluebird_max_cloud", 20)
    low_warning = cfg.get("low_cloud_warning", 60)
    good_vis = cfg.get("good_visibility_m", 30000)

    cloud = _safe(cloud_cover, 50)
    low = _safe(cloud_cover_low, 50)
    sun = _safe(sunshine_hours, 0) / 3600 if sunshine_hours and sunshine_hours > 100 else _safe(sunshine_hours, 0)
    vis = _safe(visibility_m, 10000)

    if cloud < 10 and sun > 8:
        classification = "BLUEBIRD"
    elif cloud < 30 and sun > 6:
        classification = "MOSTLY_SUNNY"
    elif cloud < 60:
        classification = "PARTLY_CLOUDY"
    elif cloud < 80:
        classification = "MOSTLY_OVERCAST"
    else:
        classification = "OVERCAST"

    return {
        "classification": classification,
        "cloud_cover": cloud,
        "cloud_cover_low": low,
        "sunshine_hours": sun,
        "visibility_m": vis,
        "flat_light_warning": low > low_warning,
        "good_visibility": vis > good_vis,
    }


def score_source_confidence(om_snow: float, scraper_snow_values: list) -> dict:
    """Score cross-source confidence and compute blended snowfall estimate.

    Compares Open-Meteo's snowfall prediction against scraper sources
    (Snow-Forecast.com, Mountain-Forecast.com) to adjust confidence.

    Logic:
        - If all sources agree on snow → boost confidence (up to +8 pts)
        - If all sources agree on no snow → boost confidence (up to +5 pts)
        - If Open-Meteo predicts snow but scrapers don't → dampen (-3 pts)
        - If scrapers predict snow but Open-Meteo doesn't → slight boost (+2 pts)
        - Blended snowfall is weighted: 60% Open-Meteo, 20% each scraper

    Returns:
        {
            "adjustment": float,       # Points to add/subtract from total
            "blended_snow_cm": float,  # Weighted snowfall estimate
            "confidence": str,         # HIGH / MEDIUM / LOW
            "detail": str,             # Human-readable explanation
        }
    """
    if not scraper_snow_values:
        return {
            "adjustment": 0,
            "blended_snow_cm": om_snow,
            "confidence": "MEDIUM",
            "detail": "No scraper data available",
        }

    valid_scraper = [v for v in scraper_snow_values if v is not None]
    if not valid_scraper:
        return {
            "adjustment": 0,
            "blended_snow_cm": om_snow,
            "confidence": "MEDIUM",
            "detail": "Scraper data incomplete",
        }

    scraper_avg = sum(valid_scraper) / len(valid_scraper)

    # Weighted blend: Open-Meteo 60%, scrapers split remaining 40%
    scraper_weight = 0.4 / len(valid_scraper)
    blended = om_snow * 0.6 + sum(v * scraper_weight for v in valid_scraper)

    om_has_snow = om_snow >= 2
    scrapers_have_snow = scraper_avg >= 2
    om_no_snow = om_snow < 1
    scrapers_no_snow = scraper_avg < 1

    if om_has_snow and scrapers_have_snow:
        # All agree: snow is coming — high confidence boost
        # Scale boost by how close the predictions are
        spread = abs(om_snow - scraper_avg) / max(om_snow, scraper_avg, 1)
        if spread < 0.3:
            adj = 8  # Very close agreement
            confidence = "HIGH"
            detail = f"All sources agree: ~{blended:.0f}cm snow"
        else:
            adj = 4  # Agree on snow but differ on amount
            confidence = "HIGH"
            detail = f"Sources agree on snow, differ on amount ({om_snow:.0f} vs {scraper_avg:.0f}cm)"

    elif om_no_snow and scrapers_no_snow:
        # All agree: dry day
        adj = 5
        confidence = "HIGH"
        detail = "All sources agree: dry"

    elif om_has_snow and not scrapers_have_snow:
        # Open-Meteo says snow, scrapers disagree — dampen confidence
        adj = -3
        confidence = "LOW"
        detail = f"Open-Meteo predicts {om_snow:.0f}cm but scrapers show {scraper_avg:.0f}cm"

    elif not om_has_snow and scrapers_have_snow:
        # Scrapers say snow, Open-Meteo doesn't — interesting signal
        adj = 2
        confidence = "LOW"
        detail = f"Scrapers predict {scraper_avg:.0f}cm but Open-Meteo shows {om_snow:.0f}cm"

    else:
        # Mixed/borderline
        adj = 0
        confidence = "MEDIUM"
        detail = f"Sources borderline (OM:{om_snow:.1f} vs scrapers:{scraper_avg:.1f}cm)"

    return {
        "adjustment": adj,
        "blended_snow_cm": round(blended, 1),
        "confidence": confidence,
        "detail": detail,
    }


def calculate_powder_score(day_data: dict, scoring_cfg: dict, sky_cfg: dict,
                           scraper_snow_values: list = None) -> dict:
    """Calculate the full powder score for a single day.

    Args:
        day_data: Dict containing all forecast variables for one day:
            - snowfall_24h_cm: float (sum of hourly snowfall)
            - temperature_summit: float (°C)
            - wind_speed_kmh: float
            - wind_gust_kmh: float
            - freezing_level_m: float
            - hourly_snowfall: list (hourly cm values)
            - hourly_times: list (hour timestamps)
            - snow_depth_m: float (current)
            - snow_depths_3day: list (last 3 days)
            - model_snowfall_values: list (snowfall from each model)
            - cloud_cover: float (%)
            - cloud_cover_low: float (%)
            - sunshine_hours: float
            - visibility_m: float
            - dew_point_depression: float (°C)
            - slr: float (snow-to-liquid ratio)

        scoring_cfg: Scoring section from config
        sky_cfg: Sky section from config

    Returns:
        Dict with total score, breakdown, classification, and metadata.
    """
    snow_24h = _safe(day_data.get("snowfall_24h_cm"), 0)
    temp = _safe(day_data.get("temperature_summit"), -5)
    wind = _safe(day_data.get("wind_speed_kmh"), 10)
    gust = _safe(day_data.get("wind_gust_kmh"), 20)
    freeze = _safe(day_data.get("freezing_level_m"), 1500)
    depth = _safe(day_data.get("snow_depth_m"), 0)

    # Gate check
    gate_limited = snow_24h < scoring_cfg.get("min_snow_gate_cm", 5)

    # Calculate component scores
    snow_pts = score_snow_quantity(snow_24h, scoring_cfg.get("snow", {}))
    temp_pts = score_temperature(temp, scoring_cfg.get("temperature", {}))
    wind_pts = score_wind(wind, gust, scoring_cfg.get("wind", {}))
    freeze_pts = score_freezing_level(freeze, scoring_cfg.get("freezing_level", {}))
    timing_pts = score_storm_timing(
        day_data.get("hourly_snowfall", []),
        day_data.get("hourly_times", []),
    )
    depth_pts = score_snow_depth_trend(
        depth,
        day_data.get("snow_depths_3day", []),
    )
    agreement_pts = score_model_agreement(
        day_data.get("model_snowfall_values", [])
    )

    # Cross-source confidence adjustment
    source_conf = score_source_confidence(snow_24h, scraper_snow_values or [])

    # Sum components
    raw_total = snow_pts + temp_pts + wind_pts + freeze_pts + timing_pts + depth_pts + agreement_pts

    # Apply source confidence adjustment (before gate)
    raw_total += source_conf["adjustment"]

    # Apply gate: if below snow threshold, cap at 30
    # But use blended snow for the gate check when scraper data available
    effective_snow = source_conf["blended_snow_cm"] if scraper_snow_values else snow_24h
    gate_limited = effective_snow < scoring_cfg.get("min_snow_gate_cm", 5)

    total = min(raw_total, 30) if gate_limited else raw_total
    total = _clamp(total, 0, 100)

    # Classification
    if total >= 80:
        label = "EPIC"
    elif total >= 60:
        label = "GOOD"
    elif total >= 40:
        label = "FAIR"
    elif total >= 20:
        label = "MARGINAL"
    else:
        label = "SKIP"

    # Sky conditions (separate from powder score)
    sky = score_sky_conditions(
        day_data.get("cloud_cover", 50),
        day_data.get("cloud_cover_low", 50),
        day_data.get("sunshine_hours", 0),
        day_data.get("visibility_m", 10000),
        sky_cfg,
    )

    # Derived metrics
    slr = day_data.get("slr")
    dpd = day_data.get("dew_point_depression")

    return {
        "total": round(total, 1),
        "label": label,
        "gate_limited": gate_limited,
        "breakdown": {
            "snow_quantity": round(snow_pts, 1),
            "temperature": round(temp_pts, 1),
            "wind": round(wind_pts, 1),
            "freezing_level": round(freeze_pts, 1),
            "storm_timing": round(timing_pts, 1),
            "snow_depth_trend": round(depth_pts, 1),
            "model_agreement": round(agreement_pts, 1),
            "source_confidence": round(source_conf["adjustment"], 1),
        },
        "source_confidence": {
            "confidence": source_conf["confidence"],
            "blended_snow_cm": source_conf["blended_snow_cm"],
            "detail": source_conf["detail"],
        },
        "sky": sky,
        "conditions": {
            "snowfall_24h_cm": round(snow_24h, 1),
            "temperature_c": round(temp, 1),
            "wind_speed_kmh": round(wind, 1),
            "wind_gust_kmh": round(gust, 1),
            "freezing_level_m": round(freeze, 0),
            "snow_depth_m": round(depth, 2),
            "snow_to_liquid_ratio": round(slr, 1) if slr else None,
            "dew_point_depression": round(dpd, 1) if dpd else None,
        },
    }
