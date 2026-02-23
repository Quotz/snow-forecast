#!/usr/bin/env python3
"""Backtesting script for validating the powder scoring algorithm against historical data.

Fetches archived model predictions and ERA5 ground truth from Open-Meteo,
runs them through calculate_powder_score(), and analyzes accuracy.

Usage:
    python3 backtest.py [--days 90] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""

import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import requests
import yaml

from scoring import calculate_powder_score

logger = logging.getLogger(__name__)

# --- Configuration ---

LOCATION = {"lat": 42.0, "lon": 20.87, "elevation": 2400}

# Physics models only — AI models (ecmwf_aifs025_single, gfs_graphcast025) are excluded
# because the Historical Forecast API only archives physics-based NWP model runs
MODELS = [
    "icon_seamless",
    "ecmwf_ifs025",
    "gfs_seamless",
    "arpege_seamless",
    "ukmo_seamless",
]

MODEL_DISPLAY = {
    "icon_seamless": "ICON",
    "ecmwf_ifs025": "ECMWF",
    "gfs_seamless": "GFS",
    "arpege_seamless": "ARPEGE",
    "ukmo_seamless": "UKMO",
}

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_PARAMS = [
    "snowfall",
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "freezing_level_height",
    "cloud_cover",
    "snow_depth",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_direction_10m",
    "rain",
    "weather_code",
    "snowfall_water_equivalent",
]

DAILY_PARAMS = [
    "snowfall_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "snow_depth_max",
    "precipitation_sum",
    "sunshine_duration",
]

ERA5_DAILY_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "snowfall_sum",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# WMO weather codes for precipitation type classification
_SNOW_CODES = {71, 73, 75, 77, 85, 86}
_RAIN_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82}


def _safe(val, default=0):
    """Return val if not None, else default."""
    return val if val is not None else default


def _skiing_hours_average(hourly_array, day_index, start_hour=8, end_hour=16):
    """Average a hourly array over skiing hours (08:00-16:00) for a given day."""
    base = day_index * 24
    values = []
    for h in range(base + start_hour, min(base + end_hour, len(hourly_array))):
        if hourly_array[h] is not None:
            values.append(hourly_array[h])
    return sum(values) / len(values) if values else None


def load_config():
    """Load config.yaml from project root."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# --- API Fetchers ---

def fetch_historical_forecasts(start_date, end_date):
    """Fetch archived model predictions from Open-Meteo historical forecast API.

    Batches requests by month to avoid excessive API calls.
    Returns dict keyed by date string -> model data.
    """
    all_data = {}

    # Split into monthly batches
    batches = _monthly_batches(start_date, end_date)

    for batch_start, batch_end in batches:
        logger.info(f"Fetching historical forecasts: {batch_start} to {batch_end}")

        params = {
            "latitude": LOCATION["lat"],
            "longitude": LOCATION["lon"],
            "elevation": LOCATION["elevation"],
            "start_date": batch_start.strftime("%Y-%m-%d"),
            "end_date": batch_end.strftime("%Y-%m-%d"),
            "hourly": ",".join(HOURLY_PARAMS),
            "daily": ",".join(DAILY_PARAMS),
            "models": ",".join(MODELS),
            "timezone": "Europe/Belgrade",
        }

        try:
            resp = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.warning(f"API error for {batch_start}-{batch_end}: {data.get('reason', data['error'])}")
                time.sleep(0.5)
                continue

            _merge_forecast_data(all_data, data)

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch forecasts for {batch_start}-{batch_end}: {e}")

        time.sleep(0.5)  # Rate limiting

    return all_data


def fetch_era5_actuals(start_date, end_date):
    """Fetch ERA5 ground truth data from Open-Meteo archive API.

    Returns dict keyed by date string -> actual conditions.
    """
    all_data = {}

    batches = _monthly_batches(start_date, end_date)

    for batch_start, batch_end in batches:
        logger.info(f"Fetching ERA5 actuals: {batch_start} to {batch_end}")

        params = {
            "latitude": LOCATION["lat"],
            "longitude": LOCATION["lon"],
            "elevation": LOCATION["elevation"],
            "start_date": batch_start.strftime("%Y-%m-%d"),
            "end_date": batch_end.strftime("%Y-%m-%d"),
            "daily": ",".join(ERA5_DAILY_PARAMS),
            "timezone": "Europe/Belgrade",
        }

        try:
            resp = requests.get(ERA5_ARCHIVE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.warning(f"ERA5 API error for {batch_start}-{batch_end}: {data.get('reason', data['error'])}")
                time.sleep(0.5)
                continue

            daily = data.get("daily", {})
            dates = daily.get("time", [])

            for i, date_str in enumerate(dates):
                all_data[date_str] = {
                    "temperature_max": _get_daily_val(daily, "temperature_2m_max", i),
                    "temperature_min": _get_daily_val(daily, "temperature_2m_min", i),
                    "snowfall_sum": _get_daily_val(daily, "snowfall_sum", i),
                    "precipitation_sum": _get_daily_val(daily, "precipitation_sum", i),
                    "wind_speed_max": _get_daily_val(daily, "wind_speed_10m_max", i),
                }

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch ERA5 for {batch_start}-{batch_end}: {e}")

        time.sleep(0.5)

    return all_data


def _monthly_batches(start_date, end_date):
    """Split a date range into monthly batches."""
    batches = []
    current = start_date
    while current <= end_date:
        # End of current month or end_date, whichever is earlier
        if current.month == 12:
            month_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)
        batch_end = min(month_end, end_date)
        batches.append((current, batch_end))
        current = batch_end + timedelta(days=1)
    return batches


def _get_daily_val(daily, key, index):
    """Safely get a value from daily data arrays."""
    arr = daily.get(key, [])
    if index < len(arr):
        return arr[index]
    return None


def _merge_forecast_data(all_data, api_response):
    """Merge API response data into the all_data dict, keyed by date.

    Parses per-model data from the response, similar to the main collector.
    """
    hourly = api_response.get("hourly", {})
    daily = api_response.get("daily", {})
    hourly_times = hourly.get("time", [])
    daily_times = daily.get("time", [])

    if not daily_times:
        return

    for day_idx, date_str in enumerate(daily_times):
        if date_str in all_data:
            continue  # Already have this date

        # Per-model daily snowfall
        model_snow = {}
        model_temp_max = {}
        model_temp_min = {}
        model_wind_max = {}
        model_gust_max = {}

        for model in MODELS:
            # Snowfall
            key = f"snowfall_sum_{model}"
            val = _get_daily_val(daily, key, day_idx)
            if val is not None:
                model_snow[model] = val

            # Temperature
            tmax_key = f"temperature_2m_max_{model}"
            tmin_key = f"temperature_2m_min_{model}"
            tmax = _get_daily_val(daily, tmax_key, day_idx)
            tmin = _get_daily_val(daily, tmin_key, day_idx)
            if tmax is not None:
                model_temp_max[model] = tmax
            if tmin is not None:
                model_temp_min[model] = tmin

            # Wind
            wkey = f"wind_speed_10m_max_{model}"
            gkey = f"wind_gusts_10m_max_{model}"
            wval = _get_daily_val(daily, wkey, day_idx)
            gval = _get_daily_val(daily, gkey, day_idx)
            if wval is not None:
                model_wind_max[model] = wval
            if gval is not None:
                model_gust_max[model] = gval

        # Fallback to plain keys
        if not model_snow:
            val = _get_daily_val(daily, "snowfall_sum", day_idx)
            if val is not None:
                model_snow["default"] = val
        if not model_temp_max:
            val = _get_daily_val(daily, "temperature_2m_max", day_idx)
            if val is not None:
                model_temp_max["default"] = val
        if not model_temp_min:
            val = _get_daily_val(daily, "temperature_2m_min", day_idx)
            if val is not None:
                model_temp_min["default"] = val
        if not model_wind_max:
            val = _get_daily_val(daily, "wind_speed_10m_max", day_idx)
            if val is not None:
                model_wind_max["default"] = val
        if not model_gust_max:
            val = _get_daily_val(daily, "wind_gusts_10m_max", day_idx)
            if val is not None:
                model_gust_max["default"] = val

        # Snow depth
        snow_depth = None
        for model in MODELS:
            sdkey = f"snow_depth_max_{model}"
            val = _get_daily_val(daily, sdkey, day_idx)
            if val is not None:
                snow_depth = val
                break
        if snow_depth is None:
            snow_depth = _get_daily_val(daily, "snow_depth_max", day_idx)

        # Sunshine duration
        sunshine = None
        for model in MODELS:
            skey = f"sunshine_duration_{model}"
            val = _get_daily_val(daily, skey, day_idx)
            if val is not None:
                sunshine = val
                break
        if sunshine is None:
            sunshine = _get_daily_val(daily, "sunshine_duration", day_idx)

        # Precipitation sum
        precip_sum = None
        for model in MODELS:
            pkey = f"precipitation_sum_{model}"
            val = _get_daily_val(daily, pkey, day_idx)
            if val is not None:
                precip_sum = val
                break
        if precip_sum is None:
            precip_sum = _get_daily_val(daily, "precipitation_sum", day_idx)

        # Extract hourly data for this day (from the first available model)
        hourly_data = _extract_day_hourly(hourly, hourly_times, day_idx)

        all_data[date_str] = {
            "model_snow": model_snow,
            "model_temp_max": model_temp_max,
            "model_temp_min": model_temp_min,
            "model_wind_max": model_wind_max,
            "model_gust_max": model_gust_max,
            "snow_depth": snow_depth,
            "sunshine": sunshine,
            "precipitation_sum": precip_sum,
            "hourly": hourly_data,
        }


def _extract_day_hourly(hourly, hourly_times, day_idx):
    """Extract hourly data for a single day from the API response."""
    day_start = day_idx * 24
    day_end = min(day_start + 24, len(hourly_times))

    if day_start >= len(hourly_times):
        return {}

    result = {"times": hourly_times[day_start:day_end]}

    # Try to get data from the first model that has it
    for param in HOURLY_PARAMS:
        values = None
        for model in MODELS:
            key = f"{param}_{model}"
            if key in hourly:
                arr = hourly[key]
                values = arr[day_start:day_end] if day_start < len(arr) else None
                break
        if values is None and param in hourly:
            arr = hourly[param]
            values = arr[day_start:day_end] if day_start < len(arr) else None
        if values is not None:
            result[param] = values

    return result


# --- Day Data Builder ---

def build_day_data(date_str, forecast_data, all_forecast_data, config):
    """Build a day_data dict compatible with scoring.py from forecast data."""
    fd = forecast_data

    # Average model snowfall
    model_snow_values = list(fd["model_snow"].values())
    avg_snow = sum(model_snow_values) / len(model_snow_values) if model_snow_values else 0

    # Temperature: average of available model max/min
    temp_maxes = list(fd["model_temp_max"].values())
    temp_mins = list(fd["model_temp_min"].values())
    temp_max = sum(temp_maxes) / len(temp_maxes) if temp_maxes else -5
    temp_min = sum(temp_mins) / len(temp_mins) if temp_mins else -5
    avg_temp = (temp_max + temp_min) / 2

    # Wind: take the max across models
    wind_maxes = list(fd["model_wind_max"].values())
    gust_maxes = list(fd["model_gust_max"].values())
    wind_max = max(wind_maxes) if wind_maxes else 10
    gust_max = max(gust_maxes) if gust_maxes else 20

    # Hourly-derived fields
    hourly = fd.get("hourly", {})
    hourly_times = hourly.get("times", [])
    hourly_snowfall = hourly.get("snowfall", [])

    # Freezing level (skiing hours average)
    freezing = _hourly_skiing_avg(hourly.get("freezing_level_height", []))

    # Cloud cover
    cloud = _hourly_skiing_avg(hourly.get("cloud_cover", []))

    # Humidity
    humidity = _hourly_skiing_avg(hourly.get("relative_humidity_2m", []))

    # Dew point and temperature for DPD
    dew_point = _hourly_skiing_avg(hourly.get("dew_point_2m", []))
    temp_2m = _hourly_skiing_avg(hourly.get("temperature_2m", []))
    dpd = (temp_2m - dew_point) if (temp_2m is not None and dew_point is not None) else None

    # Wind direction (vector average over skiing hours)
    wind_direction = _hourly_wind_direction_avg(hourly.get("wind_direction_10m", []))

    # Weather code: count snow/rain hours
    snow_hours = 0
    rain_hours = 0
    rain_mm = 0.0
    wc_arr = hourly.get("weather_code", [])
    rain_arr = hourly.get("rain", [])
    for h in range(len(wc_arr)):
        code = wc_arr[h]
        if code is not None:
            if code in _SNOW_CODES:
                snow_hours += 1
            elif code in _RAIN_CODES:
                rain_hours += 1
    for h in range(len(rain_arr)):
        rain_mm += _safe(rain_arr[h], 0)

    # SLR from hourly snowfall and SWE
    slr = None
    sf_arr = hourly.get("snowfall", [])
    swe_arr = hourly.get("snowfall_water_equivalent", [])
    if sf_arr and swe_arr:
        total_sf = sum(_safe(v) for v in sf_arr)
        total_swe = sum(_safe(v) for v in swe_arr)
        if total_swe > 0 and total_sf > 0:
            slr = total_sf / total_swe

    # Snow depth and 3-day trend
    snow_depth = _safe(fd.get("snow_depth"), 0)
    snow_depths_3day = _get_snow_depths_3day(date_str, all_forecast_data)

    # Sunshine hours
    sunshine = fd.get("sunshine")
    sunshine_hours = (sunshine / 3600) if sunshine else 0

    return {
        "date": date_str,
        "snowfall_24h_cm": avg_snow,
        "temperature_summit": avg_temp,
        "wind_speed_kmh": _safe(wind_max, 10),
        "wind_gust_kmh": _safe(gust_max, 20),
        "freezing_level_m": _safe(freezing, 1500),
        "snow_depth_m": snow_depth,
        "snow_depths_3day": snow_depths_3day,
        "model_snowfall_values": model_snow_values,
        "hourly_snowfall": hourly_snowfall,
        "hourly_times": hourly_times,
        "cloud_cover": _safe(cloud, 50),
        "cloud_cover_low": _safe(cloud, 50),
        "sunshine_hours": sunshine_hours,
        "visibility_m": 10000,
        "slr": slr,
        "dew_point_depression": dpd,
        "snow_hours": snow_hours,
        "rain_hours": rain_hours,
        "rain_mm": rain_mm,
        "wind_direction_deg": wind_direction,
        "humidity_avg": _safe(humidity, 50),
        "temperature_min_c": temp_min,
        "temperature_max_c": temp_max,
        "forecast_day_index": 2,  # Simulate 2-day lead time
    }


def _hourly_skiing_avg(arr):
    """Average over skiing hours (08:00-16:00 = indices 8-15 in a 24h array)."""
    if not arr:
        return None
    values = []
    for h in range(8, min(16, len(arr))):
        if arr[h] is not None:
            values.append(arr[h])
    return sum(values) / len(values) if values else None


def _hourly_wind_direction_avg(arr):
    """Vector-averaged wind direction over skiing hours."""
    if not arr:
        return None
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0
    for h in range(8, min(16, len(arr))):
        d = arr[h]
        if d is not None:
            rad = math.radians(d)
            sin_sum += math.sin(rad)
            cos_sum += math.cos(rad)
            count += 1
    if count == 0:
        return None
    avg_rad = math.atan2(sin_sum / count, cos_sum / count)
    return math.degrees(avg_rad) % 360


def _get_snow_depths_3day(date_str, all_forecast_data):
    """Get snow depth values for the current day and 2 days prior."""
    target = datetime.strptime(date_str, "%Y-%m-%d")
    depths = []
    for offset in range(-2, 1):
        d = (target + timedelta(days=offset)).strftime("%Y-%m-%d")
        fd = all_forecast_data.get(d)
        if fd and fd.get("snow_depth") is not None:
            depths.append(fd["snow_depth"])
        else:
            depths.append(None)
    return depths


# --- Analysis ---

def run_backtest(start_date, end_date, config):
    """Main backtesting loop."""
    print(f"\n{'='*70}")
    print(f"  POWDER SCORE BACKTEST")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Location: Popova Shapka ({LOCATION['lat']}, {LOCATION['lon']}) @ {LOCATION['elevation']}m")
    print(f"{'='*70}\n")

    # Fetch data
    print("Fetching historical forecast data...")
    forecast_data = fetch_historical_forecasts(start_date, end_date)
    print(f"  Got forecast data for {len(forecast_data)} days")

    print("Fetching ERA5 ground truth...")
    era5_data = fetch_era5_actuals(start_date, end_date)
    print(f"  Got ERA5 data for {len(era5_data)} days")

    if not forecast_data or not era5_data:
        print("\nERROR: Insufficient data to run backtest.")
        return []

    # Scoring config
    scoring_cfg = config.get("scoring", {})
    sky_cfg = config.get("sky", {})
    location_cfg = config.get("locations", {}).get("popova_shapka", {})

    # Run scoring for each day
    results = []
    dates_processed = 0
    dates_skipped = 0

    for date_str in sorted(forecast_data.keys()):
        if date_str not in era5_data:
            dates_skipped += 1
            continue

        fd = forecast_data[date_str]
        actuals = era5_data[date_str]

        # Skip if we have no model snow data
        if not fd.get("model_snow"):
            dates_skipped += 1
            continue

        # Build day_data for scoring
        day_data = build_day_data(date_str, fd, forecast_data, config)

        # Run scoring
        score_result = calculate_powder_score(
            day_data, scoring_cfg, sky_cfg,
            scraper_snow_values=None,
            ensemble_day_data=None,
            location_cfg=location_cfg,
        )

        results.append({
            "date": date_str,
            "predicted_score": score_result["total"],
            "predicted_label": score_result["label"],
            "predicted_snow": day_data["snowfall_24h_cm"],
            "predicted_temp": day_data["temperature_summit"],
            "predicted_wind": day_data["wind_speed_kmh"],
            "actual_snow": _safe(actuals.get("snowfall_sum"), 0),
            "actual_temp_max": actuals.get("temperature_max"),
            "actual_temp_min": actuals.get("temperature_min"),
            "actual_precip": _safe(actuals.get("precipitation_sum"), 0),
            "actual_wind_max": actuals.get("wind_speed_max"),
            "breakdown": score_result["breakdown"],
            "per_model_snow": fd.get("model_snow", {}),
            "per_model_temp_max": fd.get("model_temp_max", {}),
        })
        dates_processed += 1

    print(f"\nProcessed {dates_processed} days, skipped {dates_skipped} (missing data)")

    if not results:
        print("\nNo results to analyze.")
        return []

    # Run analysis
    print_calibration_table(results)
    print_detection_rates(results)
    print_per_model_accuracy(results, era5_data)
    print_component_correlation(results)

    return results


def print_calibration_table(results):
    """Print calibration table: score buckets vs actual conditions."""
    print(f"\n{'='*70}")
    print("  CALIBRATION TABLE")
    print(f"{'='*70}")

    buckets = {
        "0-20": {"scores": [], "actual_snow": [], "actual_temp": []},
        "20-40": {"scores": [], "actual_snow": [], "actual_temp": []},
        "40-60": {"scores": [], "actual_snow": [], "actual_temp": []},
        "60-80": {"scores": [], "actual_snow": [], "actual_temp": []},
        "80-100": {"scores": [], "actual_snow": [], "actual_temp": []},
    }

    for r in results:
        score = r["predicted_score"]
        if score < 20:
            bucket = "0-20"
        elif score < 40:
            bucket = "20-40"
        elif score < 60:
            bucket = "40-60"
        elif score < 80:
            bucket = "60-80"
        else:
            bucket = "80-100"

        buckets[bucket]["scores"].append(score)
        buckets[bucket]["actual_snow"].append(r["actual_snow"])
        t_max = r.get("actual_temp_max")
        t_min = r.get("actual_temp_min")
        if t_max is not None and t_min is not None:
            buckets[bucket]["actual_temp"].append((t_max + t_min) / 2)

    print(f"\n  {'Score Bucket':<14} {'Days':>5}   {'Avg Actual Snow':>16}   {'Avg Actual Temp':>16}   {'Avg Score':>10}")
    print(f"  {'-'*14} {'-'*5}   {'-'*16}   {'-'*16}   {'-'*10}")

    for bucket_name in ["0-20", "20-40", "40-60", "60-80", "80-100"]:
        b = buckets[bucket_name]
        days = len(b["scores"])
        if days == 0:
            print(f"  {bucket_name:<14} {0:>5}   {'N/A':>16}   {'N/A':>16}   {'N/A':>10}")
            continue

        avg_snow = sum(b["actual_snow"]) / len(b["actual_snow"])
        avg_temp = sum(b["actual_temp"]) / len(b["actual_temp"]) if b["actual_temp"] else float("nan")
        avg_score = sum(b["scores"]) / len(b["scores"])

        temp_str = f"{avg_temp:+.1f} C" if not math.isnan(avg_temp) else "N/A"
        print(f"  {bucket_name:<14} {days:>5}   {avg_snow:>13.1f} cm   {temp_str:>16}   {avg_score:>10.1f}")


def print_detection_rates(results):
    """Print detection rates for real powder days and false alarms."""
    print(f"\n{'='*70}")
    print("  DETECTION RATES")
    print(f"{'='*70}")

    # Real powder days: actual snowfall > 10cm AND avg temp < -2
    real_powder = []
    for r in results:
        t_max = r.get("actual_temp_max")
        t_min = r.get("actual_temp_min")
        if t_max is not None and t_min is not None:
            avg_t = (t_max + t_min) / 2
        else:
            avg_t = 0
        if r["actual_snow"] > 10 and avg_t < -2:
            real_powder.append(r)

    if real_powder:
        detected = sum(1 for r in real_powder if r["predicted_score"] >= 60)
        missed = len(real_powder) - detected
        print(f"\n  When actual snowfall > 10cm AND temp < -2 C (real powder days): {len(real_powder)} days")
        print(f"    Correctly detected (score >= 60): {detected}/{len(real_powder)} = {100*detected/len(real_powder):.0f}%")
        print(f"    Missed (score < 60):              {missed}/{len(real_powder)} = {100*missed/len(real_powder):.0f}%")
    else:
        print("\n  No real powder days (>10cm snow, <-2 C) found in period.")

    # Predicted powder: score >= 60
    predicted_powder = [r for r in results if r["predicted_score"] >= 60]
    if predicted_powder:
        true_powder = sum(1 for r in predicted_powder if r["actual_snow"] > 5)
        false_alarm = len(predicted_powder) - true_powder
        print(f"\n  When score >= 60 (predicted powder): {len(predicted_powder)} days")
        print(f"    True powder day (actual > 5cm):    {true_powder}/{len(predicted_powder)} = {100*true_powder/len(predicted_powder):.0f}%")
        print(f"    False alarm (actual < 5cm):        {false_alarm}/{len(predicted_powder)} = {100*false_alarm/len(predicted_powder):.0f}%")
    else:
        print("\n  No days with predicted score >= 60 in period.")

    # Additional threshold analysis
    print(f"\n  Threshold sensitivity:")
    for threshold in [40, 50, 60, 70]:
        predicted = [r for r in results if r["predicted_score"] >= threshold]
        if predicted:
            hits = sum(1 for r in predicted if r["actual_snow"] > 5)
            precision = 100 * hits / len(predicted) if predicted else 0
            # Recall: of all days with >5cm actual snow, how many predicted >= threshold?
            actual_snow_days = [r for r in results if r["actual_snow"] > 5]
            recall = 100 * hits / len(actual_snow_days) if actual_snow_days else 0
            print(f"    Score >= {threshold}: {len(predicted)} days, precision={precision:.0f}%, recall={recall:.0f}%")


def print_per_model_accuracy(results, era5_data):
    """Print per-model accuracy metrics."""
    print(f"\n{'='*70}")
    print("  PER-MODEL ACCURACY")
    print(f"{'='*70}")

    model_stats = defaultdict(lambda: {
        "snow_errors": [], "snow_biases": [],
        "temp_errors": [], "temp_biases": [],
    })

    for r in results:
        date = r["date"]
        actual_snow = r["actual_snow"]
        actual_temp_max = r.get("actual_temp_max")

        for model, predicted_snow in r["per_model_snow"].items():
            if model == "default":
                continue
            error = abs(predicted_snow - actual_snow)
            bias = predicted_snow - actual_snow
            model_stats[model]["snow_errors"].append(error)
            model_stats[model]["snow_biases"].append(bias)

        if actual_temp_max is not None:
            for model, predicted_temp in r["per_model_temp_max"].items():
                if model == "default" or predicted_temp is None:
                    continue
                error = abs(predicted_temp - actual_temp_max)
                bias = predicted_temp - actual_temp_max
                model_stats[model]["temp_errors"].append(error)
                model_stats[model]["temp_biases"].append(bias)

    if not model_stats:
        print("\n  No per-model data available.")
        return

    print(f"\n  {'Model':<14} {'Snow MAE':>10} {'Snow Bias':>11} {'Temp MAE':>10} {'Temp Bias':>11} {'Days':>6}")
    print(f"  {'-'*14} {'-'*10} {'-'*11} {'-'*10} {'-'*11} {'-'*6}")

    for model in MODELS:
        display = MODEL_DISPLAY.get(model, model)
        stats = model_stats.get(model)
        if not stats or not stats["snow_errors"]:
            print(f"  {display:<14} {'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>11} {'0':>6}")
            continue

        snow_mae = sum(stats["snow_errors"]) / len(stats["snow_errors"])
        snow_bias = sum(stats["snow_biases"]) / len(stats["snow_biases"])
        days = len(stats["snow_errors"])

        if stats["temp_errors"]:
            temp_mae = sum(stats["temp_errors"]) / len(stats["temp_errors"])
            temp_bias = sum(stats["temp_biases"]) / len(stats["temp_biases"])
            temp_mae_str = f"{temp_mae:.1f} C"
            temp_bias_str = f"{temp_bias:+.1f} C"
        else:
            temp_mae_str = "N/A"
            temp_bias_str = "N/A"

        print(f"  {display:<14} {snow_mae:>7.1f} cm {snow_bias:>+8.1f} cm {temp_mae_str:>10} {temp_bias_str:>11} {days:>6}")


def print_component_correlation(results):
    """Print which score components correlate best with actual snowfall."""
    print(f"\n{'='*70}")
    print("  COMPONENT CORRELATION WITH ACTUAL SNOWFALL")
    print(f"{'='*70}")

    components = [
        "snow_quantity", "temperature", "wind", "freezing_level",
        "storm_timing", "snow_depth_trend", "model_agreement",
        "snow_quality", "wind_loading", "ensemble_confidence",
        "source_confidence",
    ]

    actual_snow = [r["actual_snow"] for r in results]
    mean_actual = sum(actual_snow) / len(actual_snow) if actual_snow else 0
    var_actual = sum((s - mean_actual) ** 2 for s in actual_snow) / len(actual_snow) if actual_snow else 0

    if var_actual < 0.001:
        print("\n  Not enough variance in actual snowfall to compute correlations.")
        return

    print(f"\n  {'Component':<22} {'Correlation':>12} {'Avg Value':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*10}")

    correlations = []
    for comp in components:
        comp_values = [r["breakdown"].get(comp, 0) for r in results]
        mean_comp = sum(comp_values) / len(comp_values)
        var_comp = sum((c - mean_comp) ** 2 for c in comp_values) / len(comp_values)

        if var_comp < 0.001:
            correlations.append((comp, 0.0, mean_comp))
            continue

        covariance = sum((c - mean_comp) * (a - mean_actual)
                         for c, a in zip(comp_values, actual_snow)) / len(comp_values)
        correlation = covariance / math.sqrt(var_comp * var_actual)
        correlations.append((comp, correlation, mean_comp))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for comp, corr, avg_val in correlations:
        print(f"  {comp:<22} {corr:>+10.3f}   {avg_val:>10.1f}")

    # Also show total score correlation
    total_scores = [r["predicted_score"] for r in results]
    mean_total = sum(total_scores) / len(total_scores)
    var_total = sum((s - mean_total) ** 2 for s in total_scores) / len(total_scores)
    if var_total > 0.001:
        cov = sum((s - mean_total) * (a - mean_actual)
                  for s, a in zip(total_scores, actual_snow)) / len(total_scores)
        corr_total = cov / math.sqrt(var_total * var_actual)
        print(f"\n  {'TOTAL SCORE':<22} {corr_total:>+10.3f}   {mean_total:>10.1f}")


def save_results(results, output_dir):
    """Save backtest results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "backtest_results.json"

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "location": {
            "name": "Popova Shapka",
            "lat": LOCATION["lat"],
            "lon": LOCATION["lon"],
            "elevation": LOCATION["elevation"],
        },
        "period": {
            "start": results[0]["date"] if results else None,
            "end": results[-1]["date"] if results else None,
            "days_analyzed": len(results),
        },
        "results": results,
        "summary": _compute_summary(results),
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")


def _compute_summary(results):
    """Compute summary statistics for the JSON output."""
    if not results:
        return {}

    scores = [r["predicted_score"] for r in results]
    actual_snow = [r["actual_snow"] for r in results]
    predicted_snow = [r["predicted_snow"] for r in results]

    snow_errors = [abs(p - a) for p, a in zip(predicted_snow, actual_snow)]

    return {
        "avg_score": round(sum(scores) / len(scores), 1),
        "max_score": max(scores),
        "min_score": min(scores),
        "snow_mae_cm": round(sum(snow_errors) / len(snow_errors), 1),
        "snow_bias_cm": round(sum(p - a for p, a in zip(predicted_snow, actual_snow)) / len(results), 1),
        "days_with_score_60plus": sum(1 for s in scores if s >= 60),
        "days_with_actual_snow_10plus": sum(1 for s in actual_snow if s > 10),
        "total_predicted_snow_cm": round(sum(predicted_snow), 1),
        "total_actual_snow_cm": round(sum(actual_snow), 1),
    }


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest powder scoring algorithm against historical data"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of days to backtest (default: 90, ending 5 days ago)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date (YYYY-MM-DD). Overrides --days."
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date (YYYY-MM-DD). Default: 5 days ago (ERA5 lag)."
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    config = load_config()

    # Determine date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now() - timedelta(days=5)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.days)

    # Run backtest
    results = run_backtest(start_date, end_date, config)

    # Save results
    if results:
        output_dir = Path(__file__).parent / "docs" / "verification"
        save_results(results, output_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
