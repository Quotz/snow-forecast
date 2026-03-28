"""Snow physics and atmospheric analysis for powder quality prediction.

Provides Dendritic Growth Zone detection, surface hoar risk assessment,
and bluebird day classification from upper-atmosphere data.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dendritic Growth Zone (DGZ) Detection
# ---------------------------------------------------------------------------

# The DGZ is the temperature range where stellar dendrite crystals form,
# producing the lightest, fluffiest powder snow. Occurs at 700hPa level
# when temperature is between -12 and -16°C (Nakaya diagram sweet spot).

def detect_dgz(temperature_700hpa_hourly: list, snowfall_hourly: list = None,
               cfg: dict = None) -> dict:
    """Detect Dendritic Growth Zone activity from 700hPa temperature.

    Args:
        temperature_700hpa_hourly: Hourly 700hPa temperatures (°C), 24 values.
        snowfall_hourly: Optional hourly snowfall (cm) to check concurrent snow.
        cfg: Optional config with ideal_700hpa_min, ideal_700hpa_max.

    Returns:
        Dict with: active (bool), quality (str), hours_in_dgz (int),
        avg_temp_700 (float).
    """
    if not temperature_700hpa_hourly:
        return {"active": False, "quality": "unknown", "hours_in_dgz": 0,
                "avg_temp_700": None}

    cfg = cfg or {}
    dgz_min = cfg.get("ideal_700hpa_min", -16)
    dgz_max = cfg.get("ideal_700hpa_max", -12)
    # Extended range for "good" quality (not champagne but still nice)
    extended_min = cfg.get("extended_700hpa_min", -20)
    extended_max = cfg.get("extended_700hpa_max", -8)

    valid_temps = [t for t in temperature_700hpa_hourly if t is not None]
    if not valid_temps:
        return {"active": False, "quality": "unknown", "hours_in_dgz": 0,
                "avg_temp_700": None}

    avg_temp = sum(valid_temps) / len(valid_temps)

    # Count hours in the ideal DGZ range
    hours_ideal = sum(1 for t in valid_temps if dgz_min <= t <= dgz_max)
    hours_extended = sum(1 for t in valid_temps if extended_min <= t <= extended_max)

    # Check if snow is actually falling during DGZ hours
    has_concurrent_snow = True  # Default assume yes if no snowfall data
    if snowfall_hourly:
        snow_during_dgz = 0
        for i, t in enumerate(temperature_700hpa_hourly):
            if t is not None and dgz_min <= t <= dgz_max:
                if i < len(snowfall_hourly) and snowfall_hourly[i]:
                    snow_during_dgz += snowfall_hourly[i]
        has_concurrent_snow = snow_during_dgz > 0.5  # At least 0.5cm during DGZ

    # Classify quality
    if hours_ideal >= 6 and has_concurrent_snow:
        quality = "champagne"
        active = True
    elif hours_ideal >= 3 and has_concurrent_snow:
        quality = "good"
        active = True
    elif hours_extended >= 6 and has_concurrent_snow:
        quality = "good"
        active = True
    elif hours_extended >= 3:
        quality = "marginal"
        active = hours_extended >= 3 and has_concurrent_snow
    else:
        quality = "none"
        active = False

    return {
        "active": active,
        "quality": quality,
        "hours_in_dgz": hours_ideal,
        "hours_in_extended": hours_extended,
        "avg_temp_700": round(avg_temp, 1),
        "has_concurrent_snow": has_concurrent_snow,
    }


# ---------------------------------------------------------------------------
# Surface Hoar Risk Assessment
# ---------------------------------------------------------------------------

def assess_surface_hoar_risk(night_cloud_pct: float, humidity: float,
                              wind_speed_kmh: float, temperature_c: float,
                              cfg: dict = None) -> dict:
    """Assess risk of surface hoar crystal formation.

    Surface hoar forms on clear, calm, humid nights when radiative cooling
    drives vapor deposition onto the snow surface. It creates a persistent
    weak layer that's an avalanche hazard when buried by subsequent snowfall.

    Args:
        night_cloud_pct: Average cloud cover during night hours (18:00-06:00).
        humidity: Average relative humidity (%).
        wind_speed_kmh: Average wind speed (km/h).
        temperature_c: Average temperature (°C).
        cfg: Optional config with thresholds.

    Returns:
        Dict with: risk (str), score (float 0-100), message (str), factors (dict).
    """
    cfg = cfg or {}
    max_cloud = cfg.get("max_cloud_pct", 20)
    min_humidity = cfg.get("min_humidity", 70)
    max_wind = cfg.get("max_wind_kmh", 10)
    max_temp = cfg.get("max_temp_c", -5)

    score = 0
    factors = {}

    # Clear skies enable radiative cooling (most important factor)
    if night_cloud_pct is not None:
        if night_cloud_pct <= max_cloud:
            score += 35
            factors["clear_sky"] = True
        elif night_cloud_pct <= 40:
            score += 15
            factors["clear_sky"] = False
    else:
        score += 10  # Unknown, partial credit

    # High humidity provides moisture for crystal growth
    if humidity is not None:
        if humidity >= min_humidity:
            score += 25
            factors["high_humidity"] = True
        elif humidity >= 50:
            score += 10
            factors["high_humidity"] = False
    else:
        score += 5

    # Calm winds allow undisturbed crystal growth
    if wind_speed_kmh is not None:
        if wind_speed_kmh <= max_wind:
            score += 20
            factors["calm_wind"] = True
        elif wind_speed_kmh <= 20:
            score += 5
            factors["calm_wind"] = False
        else:
            factors["calm_wind"] = False
    else:
        score += 5

    # Cold temperature needed for ice crystal formation
    if temperature_c is not None:
        if temperature_c <= max_temp:
            score += 20
            factors["cold_enough"] = True
        elif temperature_c <= -2:
            score += 10
            factors["cold_enough"] = False
        else:
            factors["cold_enough"] = False
    else:
        score += 5

    # Risk classification
    if score >= 75:
        risk = "high"
        message = "High surface hoar risk: clear, calm, humid, cold night. Avalanche weak layer may form."
    elif score >= 50:
        risk = "moderate"
        message = "Moderate surface hoar risk. Some conditions favorable."
    else:
        risk = "low"
        message = "Low surface hoar risk."

    return {
        "risk": risk,
        "score": score,
        "message": message,
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# Bluebird Day Classification
# ---------------------------------------------------------------------------

def classify_bluebird(gph_500hpa_hourly: list, cloud_cover_hourly: list,
                       wind_speed_hourly: list, pressure_hourly: list = None,
                       cfg: dict = None) -> dict:
    """Classify bluebird day potential from upper-atmosphere indicators.

    A bluebird day is characterized by an upper-level ridge (high 500hPa
    geopotential height) that produces sustained subsidence and clearing.
    This is more reliable than just checking cloud cover, because it
    predicts WHY the sky is clear.

    Args:
        gph_500hpa_hourly: Hourly 500hPa geopotential height (m), 24 values.
        cloud_cover_hourly: Hourly cloud cover (%), 24 values.
        wind_speed_hourly: Hourly wind speed (km/h), 24 values.
        pressure_hourly: Optional surface pressure (hPa), 24 values.
        cfg: Optional config with thresholds.

    Returns:
        Dict with: confidence (0-100), type (str), clearing_hour (int|None),
        ridge_strength (float), message (str).
    """
    cfg = cfg or {}
    # Typical 500hPa GPH at 42°N in winter: ~5400-5500m
    # Ridge: >5550m, strong ridge: >5600m
    ridge_threshold = cfg.get("ridge_gph_threshold", 5520)
    strong_ridge = cfg.get("strong_ridge_gph_threshold", 5580)

    result = {
        "confidence": 0,
        "type": "none",
        "clearing_hour": None,
        "ridge_strength": 0,
        "message": "",
    }

    # Analyze 500hPa geopotential height (primary indicator)
    if gph_500hpa_hourly:
        valid_gph = [g for g in gph_500hpa_hourly if g is not None]
        if valid_gph:
            avg_gph = sum(valid_gph) / len(valid_gph)
            max_gph = max(valid_gph)
            # Ridge trending: is GPH increasing through the day?
            if len(valid_gph) >= 12:
                first_half = sum(valid_gph[:len(valid_gph)//2]) / (len(valid_gph)//2)
                second_half = sum(valid_gph[len(valid_gph)//2:]) / (len(valid_gph) - len(valid_gph)//2)
                gph_trend = second_half - first_half  # Positive = ridge building
            else:
                gph_trend = 0

            result["ridge_strength"] = round(avg_gph - ridge_threshold, 1)

            if avg_gph >= strong_ridge:
                result["confidence"] += 45
                result["type"] = "ridge"
            elif avg_gph >= ridge_threshold:
                result["confidence"] += 30
                result["type"] = "ridge"
            elif gph_trend > 20:
                # Ridge building even if not fully established
                result["confidence"] += 15
                result["type"] = "building"

    # Analyze cloud cover (skiing hours 08-16)
    if cloud_cover_hourly:
        skiing_clouds = []
        for i, cc in enumerate(cloud_cover_hourly):
            if cc is not None and 8 <= i <= 16:
                skiing_clouds.append(cc)
        if skiing_clouds:
            avg_cloud = sum(skiing_clouds) / len(skiing_clouds)
            if avg_cloud < 15:
                result["confidence"] += 30
            elif avg_cloud < 30:
                result["confidence"] += 20
            elif avg_cloud < 50:
                result["confidence"] += 5

            # Detect clearing: find the hour when clouds drop below 30%
            for i, cc in enumerate(cloud_cover_hourly):
                if cc is not None and cc < 30 and i >= 6:
                    result["clearing_hour"] = i
                    break

    # Analyze wind (calm = better bluebird experience)
    if wind_speed_hourly:
        skiing_wind = []
        for i, ws in enumerate(wind_speed_hourly):
            if ws is not None and 8 <= i <= 16:
                skiing_wind.append(ws)
        if skiing_wind:
            avg_wind = sum(skiing_wind) / len(skiing_wind)
            if avg_wind < 10:
                result["confidence"] += 15
            elif avg_wind < 20:
                result["confidence"] += 5

    # Pressure trend (rising = clearing)
    if pressure_hourly:
        valid_pressure = [p for p in pressure_hourly if p is not None]
        if len(valid_pressure) >= 12:
            p_trend = valid_pressure[-1] - valid_pressure[0]
            if p_trend > 3:  # Rising pressure
                result["confidence"] += 10

    # Clamp confidence
    result["confidence"] = min(100, result["confidence"])

    # Generate message
    conf = result["confidence"]
    if conf >= 70:
        result["message"] = "Strong bluebird potential: upper ridge with clear skies and calm winds."
    elif conf >= 45:
        result["message"] = "Good bluebird chance: ridge pattern developing, clouds expected to clear."
    elif conf >= 25:
        result["message"] = "Partial clearing possible, but conditions uncertain."
    else:
        result["message"] = ""

    return result


# ---------------------------------------------------------------------------
# Crystal type estimation (simplified)
# ---------------------------------------------------------------------------

def estimate_crystal_type(temp_700hpa: float, temp_surface: float,
                           humidity: float = None) -> str:
    """Estimate snow crystal type from temperature profile.

    Based on the Nakaya snow crystal morphology diagram.
    Returns a simplified category for dashboard display.

    Args:
        temp_700hpa: Temperature at 700hPa (~3000m), °C.
        temp_surface: Surface temperature, °C.
        humidity: Optional relative humidity (%).

    Returns:
        One of: "powder", "light", "mixed", "wet", "ice"
    """
    if temp_700hpa is None and temp_surface is None:
        return "mixed"

    # Use 700hPa temp as primary indicator (where crystals form)
    growth_temp = temp_700hpa if temp_700hpa is not None else temp_surface

    # Check if surface is too warm (melting on contact)
    if temp_surface is not None and temp_surface > 1:
        return "wet"
    if temp_surface is not None and temp_surface > -1:
        # Borderline — could be wet or mixed
        if humidity is not None and humidity > 85:
            return "wet"
        return "mixed"

    # Classify by growth temperature (Nakaya diagram simplified)
    if growth_temp is None:
        return "mixed"

    if -16 <= growth_temp <= -12:
        return "powder"  # Dendrites — champagne powder
    elif -20 <= growth_temp < -12 or -12 < growth_temp <= -8:
        return "light"   # Plates/columns — still good
    elif -8 < growth_temp <= -4:
        return "mixed"   # Needles/columns — variable
    elif growth_temp > -4:
        return "wet"     # Too warm
    else:
        return "light"   # Very cold — small crystals but dry
