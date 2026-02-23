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
    """Score based on fresh snowfall amount (0-35 points).

    Diminishing returns above epic threshold.
    """
    if snowfall_24h_cm <= 0:
        return 0

    good = cfg.get("good_cm", 10)
    great = cfg.get("great_cm", 20)
    epic = cfg.get("epic_cm", 30)

    if snowfall_24h_cm <= good:
        return _linear_score(snowfall_24h_cm, 0, good, 18)
    elif snowfall_24h_cm <= great:
        return 18 + _linear_score(snowfall_24h_cm, good, great, 9)
    elif snowfall_24h_cm <= epic:
        return 27 + _linear_score(snowfall_24h_cm, great, epic, 5)
    else:
        # Diminishing returns: 32 + small bonus up to 35
        extra = min((snowfall_24h_cm - epic) / 20, 1.0) * 3
        return 32 + extra


def score_temperature(temp_c: float, cfg: dict) -> float:
    """Score based on summit temperature (0-15 points).

    Sweet spot is ideal_min to ideal_max (e.g. -8 to -3C).
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
            return 5  # Very cold gets a floor -- snow is still dry
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

    # Gust penalty -- even if sustained is OK, gusts damage powder
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
    """Score based on accumulation curve analysis (0-5 points).

    Finds t_75pct -- the hour when 75% of daily total has fallen.
    Overnight dumps score highest; storms still building score lower.
    """
    if not hourly_snowfall or not hourly_times:
        return 0

    valid_snow = [_safe(v) for v in hourly_snowfall]
    total = sum(valid_snow)

    if total < 0.1:
        return 0  # No meaningful snowfall

    threshold = total * 0.75
    cumulative = 0
    t_75_hour = None

    for i, amount in enumerate(valid_snow):
        cumulative += amount
        if cumulative >= threshold:
            # Extract hour from the time string
            try:
                t_str = str(hourly_times[i])
                if "T" in t_str:
                    t_75_hour = int(t_str.split("T")[1][:2])
                else:
                    t_75_hour = None
            except (ValueError, IndexError):
                t_75_hour = None
            break

    if t_75_hour is None:
        # Uniform distribution or can't determine -- middle score
        return 3.0

    # Overnight dump (02:00-07:00) = best
    if 2 <= t_75_hour <= 7:
        return 5.0
    # Pre-opening (07:00-10:00) = good
    elif 7 < t_75_hour <= 10:
        return 4.0
    # Storm still building (after 16:00)
    elif t_75_hour > 16:
        return 2.0
    # Midday (10:00-16:00) = moderate
    else:
        return 3.0


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
    """Score based on how well models agree (0-15 points).

    Hybrid approach:
    - Direction agreement (5 pts): fraction agreeing on snow vs no-snow
    - Pairwise closeness (5 pts): fraction of pairs within 30% of each other
    - Range tightness (5 pts): 1 - ((max-min)/mean), scaled
    """
    valid = [v for v in snowfall_values if v is not None and v >= 0]

    if len(valid) < 2:
        return 7.5  # Only one model, middle confidence

    snow_threshold = 1.0  # cm
    mean = sum(valid) / len(valid)

    # --- Direction agreement (5 pts) ---
    snow_count = sum(1 for v in valid if v >= snow_threshold)
    no_snow_count = len(valid) - snow_count
    majority = max(snow_count, no_snow_count)
    direction_frac = majority / len(valid)
    direction_pts = direction_frac * 5.0

    # --- Pairwise closeness (5 pts) ---
    n_pairs = 0
    close_pairs = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            n_pairs += 1
            pair_max = max(valid[i], valid[j])
            if pair_max < 0.5:
                # Both near zero -- consider them close
                close_pairs += 1
            else:
                diff_ratio = abs(valid[i] - valid[j]) / pair_max
                if diff_ratio <= 0.3:
                    close_pairs += 1
    closeness_frac = close_pairs / n_pairs if n_pairs > 0 else 0
    closeness_pts = closeness_frac * 5.0

    # --- Range tightness (5 pts) ---
    val_range = max(valid) - min(valid)
    if mean > 0.5:
        tightness = max(0, 1.0 - (val_range / mean))
    else:
        # All near zero -- tight by definition
        tightness = 1.0 if val_range < 1.0 else 0.0
    tightness_pts = tightness * 5.0

    return round(direction_pts + closeness_pts + tightness_pts, 2)


def score_snow_quality(day_data: dict, cfg: dict) -> float:
    """Score powder quality (0-10 points).

    Components:
    - SLR / snow-to-liquid ratio (4 pts)
    - Dew point depression (3 pts)
    - Humidity (2 pts)
    - Temperature modifier (1 pt)

    Only scored when snowfall > 2cm.
    """
    snow_24h = _safe(day_data.get("snowfall_24h_cm"), 0)
    if snow_24h <= 2:
        return 0

    temp = _safe(day_data.get("temperature_summit"), -5)

    # --- SLR (4 pts) ---
    slr = day_data.get("slr")
    if slr is not None:
        if slr > 15:
            slr_pts = 4.0
        elif slr >= 12:
            slr_pts = 3.0
        elif slr >= 8:
            slr_pts = 2.0
        else:
            slr_pts = 0
    else:
        # Estimate SLR from temperature
        if temp < -12:
            est_slr = 20
        elif temp < -8:
            est_slr = 15
        elif temp < -4:
            est_slr = 12
        elif temp <= 0:
            est_slr = 8
        else:
            est_slr = 5

        if est_slr > 15:
            slr_pts = 4.0
        elif est_slr >= 12:
            slr_pts = 3.0
        elif est_slr >= 8:
            slr_pts = 2.0
        else:
            slr_pts = 0

    # --- Dew point depression (3 pts) ---
    dpd = day_data.get("dew_point_depression")
    if dpd is not None:
        if dpd > 12:
            dpd_pts = 3.0
        elif dpd < 3:
            dpd_pts = 0
        else:
            # Linear 0-3 over dpd range 3-12
            dpd_pts = _linear_score(dpd, 3, 12, 3)
    else:
        dpd_pts = 1.0  # Unknown, give low-middle

    # --- Humidity (2 pts) ---
    humidity = day_data.get("humidity_avg")
    if humidity is not None:
        if humidity < 50:
            hum_pts = 2.0
        elif humidity > 90:
            hum_pts = 0
        else:
            # Linear: 2 at 50%, 0 at 80%+
            hum_pts = _linear_score(80 - humidity, 0, 30, 2)
            if humidity > 80:
                hum_pts = 0
    else:
        hum_pts = 0.5  # Unknown, conservative

    # --- Temperature modifier (1 pt) ---
    if temp < -8:
        temp_mod = 1.0
    elif temp <= -2:
        temp_mod = 0.5
    else:
        temp_mod = 0

    return round(slr_pts + dpd_pts + hum_pts + temp_mod, 2)


def score_wind_loading(day_data: dict, cfg: dict, location_cfg: dict = None) -> float:
    """Score wind loading bonus (0-5 points).

    Wind depositing snow on lee slopes improves conditions.
    Requires moderate wind from a favorable direction with active snowfall.
    """
    snow_24h = _safe(day_data.get("snowfall_24h_cm"), 0)
    if snow_24h <= 2:
        return 0

    wind_dir = day_data.get("wind_direction_deg")
    wind_speed = _safe(day_data.get("wind_speed_kmh"), 0)

    if wind_dir is None:
        return 0

    # Very high wind scours snow away
    scour_threshold = cfg.get("scour_speed_kmh", cfg.get("scour_kmh", 40))
    if wind_speed > scour_threshold:
        return 0

    # Get primary aspect from location config
    if location_cfg is None:
        location_cfg = {}
    aspects = location_cfg.get("aspects", {})
    primary_degrees = aspects.get("primary_degrees", 0)

    # Lee direction is opposite of primary aspect
    lee_direction = (primary_degrees + 180) % 360

    # Compute angle between wind direction and lee direction
    angle_diff = abs(wind_dir - lee_direction)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Direction factor: 0-30 degrees off lee = full, decays to 0 at 90 degrees
    if angle_diff <= 30:
        direction_factor = 1.0
    elif angle_diff >= 90:
        direction_factor = 0
    else:
        direction_factor = 1.0 - (angle_diff - 30) / 60.0

    # Wind speed sweet spot: 15-35 km/h
    min_wind = cfg.get("ideal_speed_min_kmh", cfg.get("min_wind_kmh", 15))
    max_wind = cfg.get("ideal_speed_max_kmh", cfg.get("max_wind_kmh", 35))
    if wind_speed < min_wind:
        speed_factor = wind_speed / min_wind if min_wind > 0 else 0
    elif wind_speed <= max_wind:
        speed_factor = 1.0
    else:
        # Between max_wind and scour -- rapid decay
        speed_factor = max(0, 1.0 - (wind_speed - max_wind) / (scour_threshold - max_wind))

    return round(5.0 * direction_factor * speed_factor, 2)


def score_ensemble_confidence(ensemble_data: dict, day_data: dict) -> float:
    """Score based on ensemble spread (range: -5 to +5 points).

    Tight ensemble = high confidence bonus.
    Wide ensemble = confidence penalty.
    """
    if not ensemble_data:
        return 0

    p10 = ensemble_data.get("p10")
    p25 = ensemble_data.get("p25")
    p50 = ensemble_data.get("p50")
    p75 = ensemble_data.get("p75")
    p90 = ensemble_data.get("p90")

    if p50 is None or p25 is None or p75 is None:
        return 0

    iqr = p75 - p25

    # Normalize IQR
    if p50 > 1:
        niqr = iqr / p50
    else:
        niqr = iqr

    if niqr < 0.3:
        return 5.0
    elif niqr <= 0.6:
        return 2.0
    elif niqr <= 1.0:
        return 0
    else:
        return -3.0


def compute_confidence_pct(forecast_day_index: int) -> float:
    """Compute forecast confidence percentage based on day index.

    Days 0-2: 100%
    Days 3-6: linear decay 100% -> 70%
    Days 7-10: linear decay 70% -> 40%
    Days 11-15: linear decay 40% -> 20%
    """
    if forecast_day_index <= 2:
        return 100.0
    elif forecast_day_index <= 6:
        # Linear 100 -> 70 over days 3-6
        return 100.0 - (forecast_day_index - 2) * (30.0 / 4)
    elif forecast_day_index <= 10:
        # Linear 70 -> 40 over days 7-10
        return 70.0 - (forecast_day_index - 6) * (30.0 / 4)
    elif forecast_day_index <= 15:
        # Linear 40 -> 20 over days 11-15
        return 40.0 - (forecast_day_index - 10) * (20.0 / 5)
    else:
        return 20.0


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
        - If all sources agree on snow -> boost confidence (up to +8 pts)
        - If all sources agree on no snow -> boost confidence (up to +5 pts)
        - If Open-Meteo predicts snow but scrapers don't -> dampen (-3 pts)
        - If scrapers predict snow but Open-Meteo doesn't -> slight boost (+2 pts)
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
        # All agree: snow is coming -- high confidence boost
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
        # Open-Meteo says snow, scrapers disagree -- dampen confidence
        adj = -3
        confidence = "LOW"
        detail = f"Open-Meteo predicts {om_snow:.0f}cm but scrapers show {scraper_avg:.0f}cm"

    elif not om_has_snow and scrapers_have_snow:
        # Scrapers say snow, Open-Meteo doesn't -- interesting signal
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
                           scraper_snow_values: list = None,
                           ensemble_day_data: dict = None,
                           location_cfg: dict = None) -> dict:
    """Calculate the full powder score for a single day.

    Args:
        day_data: Dict containing all forecast variables for one day:
            - snowfall_24h_cm: float (sum of hourly snowfall)
            - temperature_summit: float (C)
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
            - dew_point_depression: float (C)
            - slr: float (snow-to-liquid ratio)
            - humidity_avg: float (%)
            - wind_direction_deg: float (degrees)
            - forecast_day_index: int (0 = today)

        scoring_cfg: Scoring section from config
        sky_cfg: Sky section from config
        scraper_snow_values: Optional list of scraper snowfall values
        ensemble_day_data: Optional dict with percentiles (p10, p25, p50, p75, p90)
        location_cfg: Optional dict with location info including aspects.primary_degrees

    Returns:
        Dict with total score, breakdown, classification, and metadata.
    """
    snow_24h = _safe(day_data.get("snowfall_24h_cm"), 0)
    temp = _safe(day_data.get("temperature_summit"), -5)
    wind = _safe(day_data.get("wind_speed_kmh"), 10)
    gust = _safe(day_data.get("wind_gust_kmh"), 20)
    freeze = _safe(day_data.get("freezing_level_m"), 1500)
    depth = _safe(day_data.get("snow_depth_m"), 0)

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

    # New: snow quality scoring
    quality_pts = score_snow_quality(day_data, scoring_cfg.get("snow_quality", {}))

    # New: wind loading bonus
    loading_pts = score_wind_loading(
        day_data, scoring_cfg.get("wind_loading", {}), location_cfg
    )

    # Cross-source confidence adjustment
    source_conf = score_source_confidence(snow_24h, scraper_snow_values or [])

    # New: ensemble confidence
    ensemble_pts = score_ensemble_confidence(ensemble_day_data, day_data) if ensemble_day_data else 0

    # Sum components
    raw_total = (snow_pts + temp_pts + wind_pts + freeze_pts + timing_pts
                 + depth_pts + agreement_pts + quality_pts + loading_pts)

    # Apply source confidence adjustment
    raw_total += source_conf["adjustment"]

    # Apply ensemble confidence adjustment
    raw_total += ensemble_pts

    # Soft gate: use blended snow for the gate check when scraper data available
    effective_snow = source_conf["blended_snow_cm"] if scraper_snow_values else snow_24h

    gate_floor = scoring_cfg.get("gate_floor_cm", 3)
    gate_ceiling = scoring_cfg.get("gate_ceiling_cm", 7)
    gate_limited = False

    if effective_snow < gate_floor:
        total = min(raw_total, 30)
        gate_limited = True
    elif effective_snow < gate_ceiling:
        gate_factor = (effective_snow - gate_floor) / (gate_ceiling - gate_floor)
        cap = 30 + gate_factor * (raw_total - 30)
        total = min(raw_total, cap)
        gate_limited = True
    else:
        total = raw_total

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

    # Forecast confidence
    forecast_day_index = day_data.get("forecast_day_index", 0)
    confidence_pct = compute_confidence_pct(forecast_day_index)

    # Derived metrics
    slr = day_data.get("slr")
    dpd = day_data.get("dew_point_depression")

    return {
        "total": round(total, 1),
        "label": label,
        "gate_limited": gate_limited,
        "confidence_pct": round(confidence_pct, 1),
        "breakdown": {
            "snow_quantity": round(snow_pts, 1),
            "temperature": round(temp_pts, 1),
            "wind": round(wind_pts, 1),
            "freezing_level": round(freeze_pts, 1),
            "storm_timing": round(timing_pts, 1),
            "snow_depth_trend": round(depth_pts, 1),
            "model_agreement": round(agreement_pts, 1),
            "snow_quality": round(quality_pts, 1),
            "wind_loading": round(loading_pts, 1),
            "ensemble_confidence": round(ensemble_pts, 1),
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
            "temperature_min_c": day_data.get("temperature_min_c"),
            "temperature_max_c": day_data.get("temperature_max_c"),
            "wind_speed_kmh": round(wind, 1),
            "wind_gust_kmh": round(gust, 1),
            "wind_direction_deg": day_data.get("wind_direction_deg"),
            "freezing_level_m": round(freeze, 0),
            "snow_depth_m": round(depth, 2),
            "snow_to_liquid_ratio": round(slr, 1) if slr else None,
            "dew_point_depression": round(dpd, 1) if dpd else None,
            "humidity_avg": day_data.get("humidity_avg"),
            "rain_mm": _safe(day_data.get("rain_mm"), 0),
            "snow_hours": day_data.get("snow_hours", 0),
            "rain_hours": day_data.get("rain_hours", 0),
        },
    }


def smooth_scores(scores: list) -> list:
    """Post-processing: smooth out single-day spikes in scored results.

    Rules:
    - If a day's score differs from neighbor average by >25 points,
      pull 30% toward the neighbor average.
    - Never smooth away EPIC scores (>=80).
    - Never smooth UP a SKIP score (<20 and snow < 3cm).
    - Single pass with forward/backward neighbor averaging.

    Args:
        scores: List of result dicts from calculate_powder_score().

    Returns:
        New list of result dicts with smoothed totals and updated labels.
    """
    if len(scores) <= 1:
        return scores

    # Work on copies to avoid mutating originals
    smoothed = []
    for s in scores:
        smoothed.append(dict(s))

    for i in range(len(smoothed)):
        original_total = smoothed[i]["total"]

        # Never smooth EPIC scores down
        if original_total >= 80:
            continue

        # Never smooth SKIP scores up (when snow is < 3cm)
        snow_cm = smoothed[i].get("conditions", {}).get("snowfall_24h_cm", 0)
        if original_total < 20 and snow_cm < 3:
            continue

        # Compute neighbor average
        neighbors = []
        if i > 0:
            neighbors.append(scores[i - 1]["total"])
        if i < len(scores) - 1:
            neighbors.append(scores[i + 1]["total"])

        if not neighbors:
            continue

        neighbor_avg = sum(neighbors) / len(neighbors)
        diff = original_total - neighbor_avg

        if abs(diff) > 25:
            # Pull 30% toward neighbor average
            new_total = original_total - 0.3 * diff
            new_total = _clamp(new_total, 0, 100)
            smoothed[i]["total"] = round(new_total, 1)

            # Update label based on new total
            t = smoothed[i]["total"]
            if t >= 80:
                smoothed[i]["label"] = "EPIC"
            elif t >= 60:
                smoothed[i]["label"] = "GOOD"
            elif t >= 40:
                smoothed[i]["label"] = "FAIR"
            elif t >= 20:
                smoothed[i]["label"] = "MARGINAL"
            else:
                smoothed[i]["label"] = "SKIP"

    return smoothed
