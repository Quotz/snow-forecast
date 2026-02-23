"""Data extraction — transform raw API responses into scoring-ready structures."""

import math
import logging

logger = logging.getLogger(__name__)


def _safe(val, default=0):
    """Return val if not None, else default."""
    return val if val is not None else default


def _skiing_hours_average(hourly_array, day_index, start_hour=8, end_hour=16):
    """Average a hourly array over skiing hours (08:00-16:00) for a given day."""
    base = day_index * 24
    values = [hourly_array[h] for h in range(base + start_hour, min(base + end_hour, len(hourly_array)))
              if hourly_array[h] is not None]
    return sum(values) / len(values) if values else None


# WMO weather codes for precipitation type classification
_SNOW_CODES = {71, 73, 75, 77, 85, 86}
_RAIN_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82}


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

        # Freezing level, cloud cover, etc. from hourly data (skiing-hours average 08-16)
        hourly_times = model_hourly.get("time", [])
        hourly_snowfall = model_hourly.get("snowfall", [])

        freezing = None
        cloud = None
        cloud_low = None
        visibility = None
        sunshine = None
        dew_point = None
        temp_2m = None
        snow_hours = 0
        rain_hours = 0
        rain_mm = 0.0
        wind_direction_deg = None
        humidity_avg = None
        hourly_temp_min = None
        hourly_temp_max = None

        if first_model in elev_data.get("models", {}):
            mh = elev_data["models"][first_model]["hourly"]
            freezing = _skiing_hours_average(mh.get("freezing_level_height", []), i)
            cloud = _skiing_hours_average(mh.get("cloud_cover", []), i)
            cloud_low = _skiing_hours_average(mh.get("cloud_cover_low", []), i)
            visibility = _skiing_hours_average(mh.get("visibility", []), i)
            dew_point = _skiing_hours_average(mh.get("dew_point_2m", []), i)
            temp_2m = _skiing_hours_average(mh.get("temperature_2m", []), i)
            humidity_avg = _skiing_hours_average(mh.get("relative_humidity_2m", []), i)

            # Full-day extraction for weather codes, rain, temp range
            day_start = i * 24
            day_end = min(day_start + 24, len(mh.get("time", [])))

            # Weather code: count snow-hours and rain-hours
            wc_arr = mh.get("weather_code", [])
            for h in range(day_start, min(day_end, len(wc_arr))):
                code = wc_arr[h]
                if code is not None:
                    if code in _SNOW_CODES:
                        snow_hours += 1
                    elif code in _RAIN_CODES:
                        rain_hours += 1

            # Rain daily sum
            rain_arr = mh.get("rain", [])
            for h in range(day_start, min(day_end, len(rain_arr))):
                rain_mm += _safe(rain_arr[h], 0)

            # Temperature min/max from hourly (full 24h)
            temp_arr = mh.get("temperature_2m", [])
            day_temps = [temp_arr[h] for h in range(day_start, min(day_end, len(temp_arr)))
                         if temp_arr[h] is not None]
            if day_temps:
                hourly_temp_min = min(day_temps)
                hourly_temp_max = max(day_temps)

            # Wind direction: skiing-hours vector average using sin/cos
            wd_arr = mh.get("wind_direction_10m", [])
            sin_sum = 0.0
            cos_sum = 0.0
            wd_count = 0
            for h in range(day_start + 8, min(day_start + 16, len(wd_arr))):
                d = wd_arr[h]
                if d is not None:
                    rad = math.radians(d)
                    sin_sum += math.sin(rad)
                    cos_sum += math.cos(rad)
                    wd_count += 1
            if wd_count > 0:
                avg_rad = math.atan2(sin_sum / wd_count, cos_sum / wd_count)
                wind_direction_deg = math.degrees(avg_rad) % 360

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
            "snow_hours": snow_hours,
            "rain_hours": rain_hours,
            "rain_mm": rain_mm,
            "wind_direction_deg": wind_direction_deg,
            "humidity_avg": _safe(humidity_avg, 50),
            "temperature_min_c": hourly_temp_min if hourly_temp_min is not None else (temp_min or -5),
            "temperature_max_c": hourly_temp_max if hourly_temp_max is not None else (temp_max or -5),
            "forecast_day_index": i,
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


def infer_yr_snowfall(temperature, precipitation, symbol_code):
    """Infer snowfall from Yr.no data for a single hour.

    Yr.no provides precipitation but not snowfall directly. We infer using:
    1. Symbol code mapping (snow/sleet/rain)
    2. Temperature-based SLR
    3. Transition zone graduated snow fraction

    Args:
        temperature: Air temperature in Celsius (or None)
        precipitation: Precipitation amount in mm (or None)
        symbol_code: Yr.no symbol_code string (or None)

    Returns:
        Tuple of (snow_cm, precip_type, confidence) where:
        - snow_cm: Estimated snowfall in cm
        - precip_type: "snow", "sleet", "rain", or "none"
        - confidence: 0.0-1.0 estimate of inference quality
    """
    precip = _safe(precipitation, 0)
    if precip <= 0:
        return (0.0, "none", 1.0)

    temp = temperature if temperature is not None else 0.0

    # Determine precip type from symbol code
    precip_type = "unknown"
    snow_fraction = None
    if symbol_code:
        code_lower = symbol_code.lower()
        if "snow" in code_lower:
            precip_type = "snow"
            snow_fraction = 1.0
        elif "sleet" in code_lower:
            precip_type = "sleet"
            snow_fraction = 0.5
        elif "rain" in code_lower:
            precip_type = "rain"
            snow_fraction = 0.0

    # Temperature-based snow fraction (transition zone -2C to +3C)
    if snow_fraction is None:
        if temp <= -2:
            snow_fraction = 1.0
            precip_type = "snow"
        elif temp >= 3:
            snow_fraction = 0.0
            precip_type = "rain"
        else:
            # Linear interpolation from 1.0 at -2C to 0.0 at +3C
            snow_fraction = (3 - temp) / 5.0
            precip_type = "sleet" if snow_fraction > 0.1 else "rain"

    # Temperature-based SLR (snow-to-liquid ratio)
    if temp <= -12:
        slr = 20
    elif temp <= -8:
        slr = 20 - (temp + 12) * (5 / 4)  # 20 at -12, 15 at -8
    elif temp <= -4:
        slr = 15 - (temp + 8) * (5 / 4)   # 15 at -8, 10 at -4
    elif temp <= 0:
        slr = 10 - (temp + 4) * (5 / 4)   # 10 at -4, 5 at 0
    else:
        slr = 5

    # snow_cm = precip_mm * snow_fraction * SLR / 10
    snow_cm = precip * snow_fraction * slr / 10.0

    # Confidence based on available data
    confidence = 0.5
    if symbol_code:
        confidence += 0.3
    if temperature is not None:
        confidence += 0.2
    confidence = min(confidence, 1.0)

    return (snow_cm, precip_type, confidence)


def extract_yr_daily_data(yr_data: dict, summit_elev: int, mid_elev: int = None) -> list:
    """Aggregate hourly Yr.no data into daily structure for scoring.

    Args:
        yr_data: Raw Yr.no collector output
        summit_elev: Summit elevation in meters
        mid_elev: Mid-mountain elevation (for freezing level inference)

    Returns:
        List of day dicts compatible with scoring engine (same keys as
        extract_daily_data_from_open_meteo output).
    """
    elev_data = yr_data.get("elevations", {}).get(summit_elev, {})
    if not elev_data:
        logger.warning(f"No Yr.no data for elevation {summit_elev}m")
        return []

    hourly = elev_data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return []

    # Group hours by date
    from collections import defaultdict
    day_hours = defaultdict(list)
    for idx, t in enumerate(times):
        date = t[:10]  # YYYY-MM-DD
        day_hours[date].append(idx)

    # Also get mid-elevation data for freezing level inference
    mid_hourly = {}
    if mid_elev and mid_elev in yr_data.get("elevations", {}):
        mid_hourly = yr_data["elevations"][mid_elev].get("hourly", {})

    sorted_dates = sorted(day_hours.keys())
    days = []

    for day_idx, date in enumerate(sorted_dates):
        indices = day_hours[date]

        # Infer snowfall for each hour
        total_snow_cm = 0.0
        snow_hours = 0
        rain_hours = 0
        rain_mm = 0.0

        temps = []
        winds = []
        clouds = []
        humidities = []
        dew_points = []
        wind_dirs_sin = []
        wind_dirs_cos = []
        skiing_clouds = []
        skiing_humidities = []
        skiing_dew_points = []
        skiing_temps = []

        for idx in indices:
            temp = hourly["temperature"][idx] if idx < len(hourly.get("temperature", [])) else None
            precip_1h = hourly["precipitation_1h"][idx] if idx < len(hourly.get("precipitation_1h", [])) else None
            symbol = hourly["symbol_code_1h"][idx] if idx < len(hourly.get("symbol_code_1h", [])) else None

            snow_cm, precip_type, _ = infer_yr_snowfall(temp, precip_1h, symbol)
            total_snow_cm += snow_cm
            if precip_type == "snow":
                snow_hours += 1
            elif precip_type == "rain":
                rain_hours += 1
                rain_mm += _safe(precip_1h, 0)

            if temp is not None:
                temps.append(temp)

            wind_sp = hourly["wind_speed"][idx] if idx < len(hourly.get("wind_speed", [])) else None
            if wind_sp is not None:
                winds.append(wind_sp * 3.6)  # m/s -> km/h

            cloud = hourly["cloud_cover"][idx] if idx < len(hourly.get("cloud_cover", [])) else None
            if cloud is not None:
                clouds.append(cloud)

            hum = hourly["humidity"][idx] if idx < len(hourly.get("humidity", [])) else None
            if hum is not None:
                humidities.append(hum)

            dp = hourly["dew_point"][idx] if idx < len(hourly.get("dew_point", [])) else None
            if dp is not None:
                dew_points.append(dp)

            wd = hourly["wind_direction"][idx] if idx < len(hourly.get("wind_direction", [])) else None
            if wd is not None:
                rad = math.radians(wd)
                wind_dirs_sin.append(math.sin(rad))
                wind_dirs_cos.append(math.cos(rad))

            # Determine which hour of day this is
            time_str = times[idx]
            hour = int(time_str[11:13]) if len(time_str) >= 13 else 0

            # Skiing hours (8-16)
            if 8 <= hour < 16:
                if cloud is not None:
                    skiing_clouds.append(cloud)
                if hum is not None:
                    skiing_humidities.append(hum)
                if dp is not None:
                    skiing_dew_points.append(dp)
                if temp is not None:
                    skiing_temps.append(temp)

        # Aggregations
        temp_min = min(temps) if temps else -5
        temp_max = max(temps) if temps else -5
        avg_temp = (temp_min + temp_max) / 2
        wind_max_kmh = max(winds) if winds else 10
        cloud_avg = sum(skiing_clouds) / len(skiing_clouds) if skiing_clouds else 50
        humidity_avg = sum(skiing_humidities) / len(skiing_humidities) if skiing_humidities else 50

        # Wind direction vector average
        wind_direction = None
        if wind_dirs_sin and wind_dirs_cos:
            avg_sin = sum(wind_dirs_sin) / len(wind_dirs_sin)
            avg_cos = sum(wind_dirs_cos) / len(wind_dirs_cos)
            wind_direction = math.degrees(math.atan2(avg_sin, avg_cos)) % 360

        # Dew point depression
        dpd = None
        if skiing_temps and skiing_dew_points:
            avg_t = sum(skiing_temps) / len(skiing_temps)
            avg_dp = sum(skiing_dew_points) / len(skiing_dew_points)
            dpd = avg_t - avg_dp

        # Infer freezing level from temperature difference between mid and summit
        freezing_level = 1500  # default
        if mid_hourly and mid_elev:
            mid_temps = mid_hourly.get("temperature", [])
            summit_temps = hourly.get("temperature", [])
            # Use skiing-hours temps for both elevations
            mid_skiing_temps = []
            for idx in indices:
                time_str = times[idx]
                hour = int(time_str[11:13]) if len(time_str) >= 13 else 0
                if 8 <= hour < 16 and idx < len(mid_temps) and mid_temps[idx] is not None:
                    mid_skiing_temps.append(mid_temps[idx])

            if mid_skiing_temps and skiing_temps:
                avg_mid_t = sum(mid_skiing_temps) / len(mid_skiing_temps)
                avg_summit_t = sum(skiing_temps) / len(skiing_temps)
                elev_diff = summit_elev - mid_elev
                if elev_diff > 0:
                    lapse_rate = (avg_mid_t - avg_summit_t) / elev_diff  # C per meter
                    if lapse_rate > 0 and avg_summit_t < 0:
                        # Freezing level = summit_elev + (0 - summit_temp) / lapse_rate
                        freezing_level = summit_elev + (0 - avg_summit_t) / lapse_rate
                    elif avg_summit_t >= 0:
                        freezing_level = summit_elev + 500  # Above summit
                    else:
                        freezing_level = summit_elev - abs(avg_summit_t) * 150  # Rough estimate

        days.append({
            "date": date,
            "snowfall_24h_cm": total_snow_cm,
            "temperature_summit": avg_temp,
            "wind_speed_kmh": wind_max_kmh,
            "wind_gust_kmh": wind_max_kmh * 1.5,  # Yr.no doesn't provide gusts directly
            "freezing_level_m": freezing_level,
            "snow_depth_m": 0,  # Yr.no doesn't provide snow depth
            "snow_depths_3day": [],
            "model_snowfall_values": [total_snow_cm],
            "hourly_snowfall": [],
            "hourly_times": [],
            "cloud_cover": cloud_avg,
            "cloud_cover_low": cloud_avg,  # Use total as approximation
            "sunshine_hours": 0,
            "visibility_m": 10000,  # Yr.no doesn't provide visibility
            "slr": None,
            "dew_point_depression": dpd,
            "snow_hours": snow_hours,
            "rain_hours": rain_hours,
            "rain_mm": rain_mm,
            "wind_direction_deg": wind_direction,
            "humidity_avg": humidity_avg,
            "temperature_min_c": temp_min,
            "temperature_max_c": temp_max,
            "forecast_day_index": day_idx,
        })

    return days
