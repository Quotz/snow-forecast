"""Data extraction — transform raw API responses into scoring-ready structures."""

import math
import logging

logger = logging.getLogger(__name__)


def _safe(val, default=0):
    """Return val if not None, else default."""
    return val if val is not None else default


def extract_daily_data_from_open_meteo(om_data: dict, elevation: int,
                                        models: list) -> list:
    """Extract per-day scoring data from Open-Meteo response.

    Returns a list of dicts, one per forecast day, with all variables
    needed by the scoring engine.
    """
    elev_data = om_data.get("elevations", {}).get(elevation, {})
    if not elev_data:
        logger.warning(f"No data for elevation {elevation}m")
        return []

    daily = elev_data.get("daily", {})
    dates = daily.get("time", [])

    # Get hourly data from first model for timing analysis
    first_model = models[0] if models else None
    model_hourly = {}
    if first_model and first_model in elev_data.get("models", {}):
        model_hourly = elev_data["models"][first_model].get("hourly", {})

    days = []
    for i, date in enumerate(dates):
        # Collect snowfall from each model for agreement scoring
        model_snowfall_values = []
        for model in models:
            key = f"snowfall_sum_{model}"
            if key in daily:
                val = daily[key][i] if i < len(daily.get(key, [])) else None
                model_snowfall_values.append(_safe(val, 0))

        # Use best available snowfall (average of models, or plain key)
        if model_snowfall_values:
            avg_snow = sum(model_snowfall_values) / len(model_snowfall_values)
        elif "snowfall_sum" in daily:
            avg_snow = _safe(daily["snowfall_sum"][i], 0)
        else:
            avg_snow = 0

        # Temperature (use the first model or plain daily)
        temp_max = None
        temp_min = None
        for model in models:
            tmax_key = f"temperature_2m_max_{model}"
            tmin_key = f"temperature_2m_min_{model}"
            if tmax_key in daily and i < len(daily[tmax_key]):
                temp_max = temp_max or daily[tmax_key][i]
            if tmin_key in daily and i < len(daily[tmin_key]):
                temp_min = temp_min or daily[tmin_key][i]
        if temp_max is None and "temperature_2m_max" in daily:
            temp_max = daily["temperature_2m_max"][i] if i < len(daily.get("temperature_2m_max", [])) else None
        if temp_min is None and "temperature_2m_min" in daily:
            temp_min = daily["temperature_2m_min"][i] if i < len(daily.get("temperature_2m_min", [])) else None

        avg_temp = ((temp_max or -5) + (temp_min or -5)) / 2

        # Wind
        wind_max = None
        gust_max = None
        for model in models:
            wkey = f"wind_speed_10m_max_{model}"
            gkey = f"wind_gusts_10m_max_{model}"
            if wkey in daily and i < len(daily[wkey]):
                val = daily[wkey][i]
                wind_max = max(wind_max or 0, _safe(val, 0))
            if gkey in daily and i < len(daily[gkey]):
                val = daily[gkey][i]
                gust_max = max(gust_max or 0, _safe(val, 0))
        if wind_max is None and "wind_speed_10m_max" in daily:
            wind_max = _safe(daily["wind_speed_10m_max"][i], 10)
        if gust_max is None and "wind_gusts_10m_max" in daily:
            gust_max = _safe(daily["wind_gusts_10m_max"][i], 20)

        # Freezing level, cloud cover, etc. from hourly data (midday value)
        midday_idx = i * 24 + 12  # Approximate midday hour index
        hourly_times = model_hourly.get("time", [])
        hourly_snowfall = model_hourly.get("snowfall", [])

        freezing = None
        cloud = None
        cloud_low = None
        visibility = None
        sunshine = None
        dew_point = None
        temp_2m = None

        if first_model in elev_data.get("models", {}):
            mh = elev_data["models"][first_model]["hourly"]
            if midday_idx < len(mh.get("freezing_level_height", [])):
                freezing = mh["freezing_level_height"][midday_idx]
            if midday_idx < len(mh.get("cloud_cover", [])):
                cloud = mh["cloud_cover"][midday_idx]
            if midday_idx < len(mh.get("cloud_cover_low", [])):
                cloud_low = mh["cloud_cover_low"][midday_idx]
            if midday_idx < len(mh.get("visibility", [])):
                visibility = mh["visibility"][midday_idx]
            if midday_idx < len(mh.get("dew_point_2m", [])):
                dew_point = mh["dew_point_2m"][midday_idx]
            if midday_idx < len(mh.get("temperature_2m", [])):
                temp_2m = mh["temperature_2m"][midday_idx]

        # Daily sunshine from daily data
        for model in models:
            skey = f"sunshine_duration_{model}"
            if skey in daily and i < len(daily[skey]):
                sunshine = daily[skey][i]
                break
        if sunshine is None and "sunshine_duration" in daily:
            sunshine = daily["sunshine_duration"][i] if i < len(daily.get("sunshine_duration", [])) else None

        # Convert sunshine from seconds to hours
        sunshine_hours = (sunshine / 3600) if sunshine else 0

        # Snow depth
        snow_depth = None
        for model in models:
            sdkey = f"snow_depth_max_{model}"
            if sdkey in daily and i < len(daily[sdkey]):
                snow_depth = daily[sdkey][i]
                break
        if snow_depth is None and "snow_depth_max" in daily:
            snow_depth = daily["snow_depth_max"][i] if i < len(daily.get("snow_depth_max", [])) else None

        # Snow depths for trend (last 3 days including current)
        snow_depths_3day = []
        for j in range(max(0, i - 2), i + 1):
            sd = None
            for model in models:
                sdkey = f"snow_depth_max_{model}"
                if sdkey in daily and j < len(daily[sdkey]):
                    sd = daily[sdkey][j]
                    break
            if sd is None and "snow_depth_max" in daily and j < len(daily.get("snow_depth_max", [])):
                sd = daily["snow_depth_max"][j]
            snow_depths_3day.append(sd)

        # SLR (snow-to-liquid ratio) - from hourly data
        slr = None
        swe_key = "snowfall_water_equivalent"
        sf_key = "snowfall"
        if first_model in elev_data.get("models", {}):
            mh = elev_data["models"][first_model]["hourly"]
            # Sum daily snowfall and SWE
            day_start = i * 24
            day_end = min(day_start + 24, len(mh.get(sf_key, [])))
            total_sf = sum(_safe(mh.get(sf_key, [])[j]) for j in range(day_start, day_end))
            total_swe = sum(_safe(mh.get(swe_key, [])[j]) for j in range(day_start, min(day_end, len(mh.get(swe_key, [])))))
            if total_swe > 0 and total_sf > 0:
                slr = total_sf / total_swe  # cm per mm

        # Dew point depression
        dpd = None
        if temp_2m is not None and dew_point is not None:
            dpd = temp_2m - dew_point

        # Get hourly snowfall for this day (for storm timing)
        day_hourly_snow = []
        day_hourly_times = []
        if first_model in elev_data.get("models", {}):
            mh = elev_data["models"][first_model]["hourly"]
            times = mh.get("time", [])
            snow_vals = mh.get("snowfall", [])
            day_start = i * 24
            day_end = min(day_start + 24, len(times))
            day_hourly_times = times[day_start:day_end]
            day_hourly_snow = snow_vals[day_start:day_end]

        days.append({
            "date": date,
            "snowfall_24h_cm": avg_snow,
            "temperature_summit": avg_temp,
            "wind_speed_kmh": _safe(wind_max, 10),
            "wind_gust_kmh": _safe(gust_max, 20),
            "freezing_level_m": _safe(freezing, 1500),
            "snow_depth_m": _safe(snow_depth, 0),
            "snow_depths_3day": snow_depths_3day,
            "model_snowfall_values": model_snowfall_values,
            "hourly_snowfall": day_hourly_snow,
            "hourly_times": day_hourly_times,
            "cloud_cover": _safe(cloud, 50),
            "cloud_cover_low": _safe(cloud_low, 50),
            "sunshine_hours": sunshine_hours,
            "visibility_m": _safe(visibility, 10000),
            "slr": slr,
            "dew_point_depression": dpd,
        })

    return days


def build_model_comparison(om_data: dict, elevation: int, models: list) -> list:
    """Build model comparison table data."""
    elev_data = om_data.get("elevations", {}).get(elevation, {})
    if not elev_data:
        return []

    daily = elev_data.get("daily", {})
    dates = daily.get("time", [])

    rows = []
    for i, date in enumerate(dates[:16]):
        values = []
        for model in models:
            key = f"snowfall_sum_{model}"
            if key in daily and i < len(daily[key]):
                values.append(_safe(daily[key][i], 0))
            else:
                values.append(None)

        # Calculate agreement
        valid = [v for v in values if v is not None]
        if len(valid) >= 2:
            mean = sum(valid) / len(valid)
            if mean < 0.5:
                agreement = "HIGH"
            else:
                variance = sum((v - mean) ** 2 for v in valid) / len(valid)
                cv = math.sqrt(variance) / mean if mean > 0 else 0
                if cv < 0.3:
                    agreement = "HIGH"
                elif cv < 0.6:
                    agreement = "MED"
                else:
                    agreement = "LOW"
        else:
            agreement = "-"

        rows.append({
            "date": date,
            "snowfall_values": values,
            "agreement": agreement,
        })

    return rows
