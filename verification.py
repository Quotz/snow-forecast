"""Forecast verification system.

Compares historical forecast predictions against ERA5 reanalysis ground truth
from the Open-Meteo Archive API. Tracks per-model accuracy metrics over time
using exponentially weighted moving averages.
"""

import json
import logging
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import requests

from kalman import kalman_update, initialize_from_ewma
from ensemble_stats import (compute_crps, compute_brier, store_analog,
                             compute_model_correlations)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. ERA5 reanalysis fetcher
# ---------------------------------------------------------------------------

def fetch_era5_reanalysis(lat, lon, elevation, start_date, end_date):
    """Fetch ERA5 reanalysis data from the Open-Meteo Archive API.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        elevation: Elevation in metres.
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        dict with keys: dates, temperature_max, temperature_min, snowfall,
        precipitation, wind_max, wind_gust_max. Values are parallel lists.
        Returns None on failure.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "elevation": elevation,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "snowfall_sum",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
        ]),
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("ERA5 fetch failed: %s", exc)
        return None
    except ValueError as exc:
        logger.error("ERA5 JSON decode failed: %s", exc)
        return None

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        logger.warning("ERA5 returned no daily data for %s to %s", start_date, end_date)
        return None

    return {
        "dates": dates,
        "temperature_max": daily.get("temperature_2m_max", []),
        "temperature_min": daily.get("temperature_2m_min", []),
        "snowfall": daily.get("snowfall_sum", []),
        "precipitation": daily.get("precipitation_sum", []),
        "wind_max": daily.get("wind_speed_10m_max", []),
        "wind_gust_max": daily.get("wind_gusts_10m_max", []),
    }


# ---------------------------------------------------------------------------
# 2. Historical forecast finder
# ---------------------------------------------------------------------------

def find_historical_forecast(history_dir, target_date, max_lead_days=5):
    """Find a stored forecast that predicted *target_date* with lead time 1-5 days.

    Prefers forecasts with 2-3 day lead time as the most representative balance
    between freshness and typical usage.

    Args:
        history_dir: Path to docs/history/ directory.
        target_date: The date we want actuals for, as "YYYY-MM-DD".
        max_lead_days: Maximum lead-time to accept (default 5).

    Returns:
        dict with keys from the matching forecast file's score entry for
        target_date, plus 'lead_days' and 'forecast_generated_at', and
        'model_comparison_entry' if available. Returns None if no suitable
        forecast found.
    """
    history_path = Path(history_dir)
    if not history_path.is_dir():
        logger.warning("History directory does not exist: %s", history_dir)
        return None

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    candidates = []  # (lead_days, file_path, generated_date)

    # Pattern: forecast_YYYYMMDD_HHMMSS.json
    pattern = re.compile(r"forecast_(\d{8})_\d{6}\.json$")

    for entry in history_path.iterdir():
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        forecast_date_str = m.group(1)
        try:
            forecast_dt = datetime.strptime(forecast_date_str, "%Y%m%d")
        except ValueError:
            continue

        lead_days = (target_dt - forecast_dt).days
        if 1 <= lead_days <= max_lead_days:
            candidates.append((lead_days, entry, forecast_dt))

    if not candidates:
        return None

    # Prefer lead_days 2-3, then closest to that range
    def lead_preference(item):
        ld = item[0]
        if 2 <= ld <= 3:
            return (0, ld)  # best tier, prefer 2 over 3
        return (1, abs(ld - 2.5))

    candidates.sort(key=lead_preference)
    best_lead, best_file, forecast_dt = candidates[0]

    try:
        with open(best_file) as f:
            forecast_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read forecast file %s: %s", best_file, exc)
        return None

    # Find the target_date in the scores list
    scores = forecast_data.get("scores", [])
    score_entry = None
    for s in scores:
        if s.get("date") == target_date:
            score_entry = s
            break

    if score_entry is None:
        logger.debug("Forecast %s does not contain date %s", best_file.name, target_date)
        return None

    # Also grab per-model snowfall from model_comparison
    model_comparison_entry = None
    models = forecast_data.get("models", [])
    model_comparison = forecast_data.get("model_comparison", [])
    for mc in model_comparison:
        if mc.get("date") == target_date:
            model_comparison_entry = mc
            break

    # Also grab per-model snowfall from chart_data if available
    chart_model_snow = {}
    chart_data = forecast_data.get("chart_data", {})
    dates_list = forecast_data.get("dates", [])
    if target_date in dates_list:
        date_idx = dates_list.index(target_date)
        for ds in chart_data.get("datasets", []):
            label = ds.get("label", "")
            data_vals = ds.get("data", [])
            if date_idx < len(data_vals):
                chart_model_snow[label.lower()] = data_vals[date_idx]

    result = {
        "score": score_entry,
        "lead_days": best_lead,
        "forecast_generated_at": forecast_data.get("generated_at", ""),
        "forecast_file": best_file.name,
        "models": models,
    }

    if model_comparison_entry:
        result["model_comparison_entry"] = model_comparison_entry
    if chart_model_snow:
        result["chart_model_snow"] = chart_model_snow

    return result


# ---------------------------------------------------------------------------
# 3. Statistical metrics
# ---------------------------------------------------------------------------

def compute_model_metrics(forecast_values, observed_values):
    """Compute bias, MAE, and RMSE for parallel predicted/observed lists.

    Args:
        forecast_values: List of predicted values.
        observed_values: List of observed values.

    Returns:
        dict with bias, mae, rmse, n_samples. Returns None if no valid pairs.
    """
    pairs = [
        (f, o)
        for f, o in zip(forecast_values, observed_values)
        if f is not None and o is not None
    ]
    if not pairs:
        return None

    n = len(pairs)
    errors = [f - o for f, o in pairs]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e * e for e in errors]

    bias = sum(errors) / n
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)

    return {
        "bias": round(bias, 3),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# 4. Powder day accuracy
# ---------------------------------------------------------------------------

def compute_powder_day_accuracy(forecast_scores, era5_data):
    """Assess how well powder score predictions match actual conditions.

    A "good actual day" is defined as ERA5 snowfall > 10 cm AND temperature
    max < -2 C (cold enough for quality snow preservation).

    Args:
        forecast_scores: List of dicts, each with 'date' and 'total' (score).
        era5_data: dict from fetch_era5_reanalysis.

    Returns:
        dict with hit_rate, false_alarm_rate, miss_rate, n_days.
        Returns None if insufficient data.
    """
    if not era5_data or not forecast_scores:
        return None

    era5_by_date = {}
    for i, d in enumerate(era5_data["dates"]):
        era5_by_date[d] = {
            "snowfall": _safe_val(era5_data["snowfall"], i),
            "temp_max": _safe_val(era5_data["temperature_max"], i),
        }

    hits = 0
    false_alarms = 0
    misses = 0
    correct_negatives = 0
    n = 0

    for score_entry in forecast_scores:
        date = score_entry.get("date") if isinstance(score_entry, dict) else None
        if not date or date not in era5_by_date:
            continue

        actual = era5_by_date[date]
        if actual["snowfall"] is None or actual["temp_max"] is None:
            continue

        n += 1
        predicted_good = (score_entry.get("total", 0) >= 60)
        actual_good = (actual["snowfall"] > 10 and actual["temp_max"] < -2)

        if predicted_good and actual_good:
            hits += 1
        elif predicted_good and not actual_good:
            false_alarms += 1
        elif not predicted_good and actual_good:
            misses += 1
        else:
            correct_negatives += 1

    if n == 0:
        return None

    return {
        "hit_rate": round(hits / n, 3) if n else 0,
        "false_alarm_rate": round(false_alarms / n, 3) if n else 0,
        "miss_rate": round(misses / n, 3) if n else 0,
        "correct_negative_rate": round(correct_negatives / n, 3) if n else 0,
        "n_days": n,
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
    }


def _safe_val(lst, idx):
    """Safely get a value from a list by index."""
    if lst is None or idx >= len(lst):
        return None
    return lst[idx]


# ---------------------------------------------------------------------------
# 5. Persistent stats with EWMA
# ---------------------------------------------------------------------------

def update_verification_stats(new_metrics, stats_path):
    """Merge new verification metrics into persistent stats using EWMA.

    Uses exponentially weighted moving average with alpha=0.1 so that
    recent performance is weighted but long-term trends are preserved.

    Args:
        new_metrics: dict with keys 'per_model', 'overall', 'powder_accuracy',
                     'verified_dates'.
        stats_path: Path to docs/verification/stats.json.

    Returns:
        The updated stats dict.
    """
    alpha = 0.1

    # Load existing or initialise
    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read existing stats at %s, starting fresh", stats_path)
            stats = {}

    if not stats:
        stats = {
            "updated_at": None,
            "n_verifications": 0,
            "per_model": {},
            "overall": {},
            "powder_accuracy": {},
            "last_verified_dates": [],
        }

    stats["updated_at"] = datetime.utcnow().isoformat() + "Z"
    stats["n_verifications"] = stats.get("n_verifications", 0) + 1

    # --- Merge per-model metrics ---
    new_per_model = new_metrics.get("per_model", {})
    for model_name, variables in new_per_model.items():
        if model_name not in stats["per_model"]:
            stats["per_model"][model_name] = {}
        for var_name, new_vals in variables.items():
            if new_vals is None:
                continue
            existing = stats["per_model"][model_name].get(var_name, {})
            stats["per_model"][model_name][var_name] = _ewma_merge(
                existing, new_vals, alpha
            )

    # --- Merge overall metrics ---
    new_overall = new_metrics.get("overall", {})
    for var_name, new_vals in new_overall.items():
        if new_vals is None:
            continue
        existing = stats.get("overall", {}).get(var_name, {})
        if "overall" not in stats:
            stats["overall"] = {}
        stats["overall"][var_name] = _ewma_merge(existing, new_vals, alpha)

    # --- Merge powder accuracy ---
    new_pa = new_metrics.get("powder_accuracy")
    if new_pa and new_pa.get("n_days", 0) > 0:
        existing_pa = stats.get("powder_accuracy", {})
        for key in ("hit_rate", "false_alarm_rate", "miss_rate", "correct_negative_rate"):
            if key in new_pa:
                old_val = existing_pa.get(key)
                if old_val is not None:
                    existing_pa[key] = round(alpha * new_pa[key] + (1 - alpha) * old_val, 4)
                else:
                    existing_pa[key] = new_pa[key]
        existing_pa["n_days"] = existing_pa.get("n_days", 0) + new_pa.get("n_days", 0)
        stats["powder_accuracy"] = existing_pa

    # --- Update verified dates ---
    new_dates = new_metrics.get("verified_dates", [])
    existing_dates = stats.get("last_verified_dates", [])
    all_dates = sorted(set(existing_dates + new_dates))
    stats["last_verified_dates"] = all_dates[-30:]  # keep last 30

    # Save
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Verification stats updated: %d total verifications", stats["n_verifications"])
    return stats


def _ewma_merge(existing, new_vals, alpha):
    """Merge a new metrics dict into an existing one using EWMA.

    Both dicts have keys like 'bias', 'mae', 'rmse', 'n_samples'.
    """
    if not existing:
        return dict(new_vals)

    merged = {}
    for key in ("bias", "mae", "rmse"):
        new_v = new_vals.get(key)
        old_v = existing.get(key)
        if new_v is not None and old_v is not None:
            merged[key] = round(alpha * new_v + (1 - alpha) * old_v, 4)
        elif new_v is not None:
            merged[key] = new_v
        else:
            merged[key] = old_v

    merged["n_samples"] = existing.get("n_samples", 0) + new_vals.get("n_samples", 0)
    return merged


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def run_verification(config, docs_dir="docs"):
    """Run forecast verification against ERA5 reanalysis.

    Determines a verification window of T-5 to T-3 (ERA5 has ~5 day latency),
    fetches actual conditions, compares against stored historical forecasts,
    and updates persistent accuracy statistics.

    Args:
        config: Loaded config dict (from config.yaml).
        docs_dir: Path to docs/ directory (default "docs").

    Returns:
        dict with verification results, or None on failure.
    """
    logger.info("Starting forecast verification")

    # Location config
    loc_name = list(config["locations"].keys())[0]
    loc_cfg = config["locations"][loc_name]
    lat = loc_cfg["lat"]
    lon = loc_cfg["lon"]
    summit_elev = max(loc_cfg["elevations"].values())
    models = config.get("models", [])

    # Verification window: T-5 to T-3 (ERA5 reanalysis typically has 5-day lag)
    today = datetime.utcnow().date()
    end_date = today - timedelta(days=3)
    start_date = today - timedelta(days=5)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    logger.info("Verification window: %s to %s", start_str, end_str)

    # Fetch ERA5 ground truth
    era5 = fetch_era5_reanalysis(lat, lon, summit_elev, start_str, end_str)
    if not era5:
        logger.warning("Could not fetch ERA5 data, skipping verification")
        return None

    logger.info("ERA5 data: %d days retrieved", len(era5["dates"]))

    # Find historical forecasts for each date in the window
    history_dir = os.path.join(docs_dir, "history")
    per_model_snow_forecast = {m: [] for m in models}
    per_model_snow_observed = {m: [] for m in models}
    overall_snow_forecast = []
    overall_snow_observed = []
    overall_temp_forecast = []
    overall_temp_observed = []
    overall_wind_forecast = []
    overall_wind_observed = []
    forecast_scores_for_powder = []
    verified_dates = []

    era5_by_date = {}
    for i, d in enumerate(era5["dates"]):
        era5_by_date[d] = {
            "snowfall": _safe_val(era5["snowfall"], i),
            "temperature_max": _safe_val(era5["temperature_max"], i),
            "temperature_min": _safe_val(era5["temperature_min"], i),
            "precipitation": _safe_val(era5["precipitation"], i),
            "wind_max": _safe_val(era5["wind_max"], i),
            "wind_gust_max": _safe_val(era5["wind_gust_max"], i),
        }

    for target_date in era5["dates"]:
        hist = find_historical_forecast(history_dir, target_date)
        if hist is None:
            logger.debug("No historical forecast found for %s", target_date)
            continue

        verified_dates.append(target_date)
        score_entry = hist["score"]
        conditions = score_entry.get("conditions", {})
        actual = era5_by_date[target_date]

        logger.info(
            "Verifying %s (lead=%dd, file=%s): forecast snow=%.1fcm, actual snow=%.1fcm",
            target_date,
            hist["lead_days"],
            hist["forecast_file"],
            conditions.get("snowfall_24h_cm", 0),
            actual["snowfall"] or 0,
        )

        # Overall composite snowfall
        fc_snow = conditions.get("snowfall_24h_cm")
        obs_snow = actual["snowfall"]
        if fc_snow is not None and obs_snow is not None:
            overall_snow_forecast.append(fc_snow)
            overall_snow_observed.append(obs_snow)

        # Temperature (use max temp for comparison)
        fc_temp = conditions.get("temperature_max_c")
        obs_temp = actual["temperature_max"]
        if fc_temp is not None and obs_temp is not None:
            overall_temp_forecast.append(fc_temp)
            overall_temp_observed.append(obs_temp)

        # Wind
        fc_wind = conditions.get("wind_speed_kmh")
        obs_wind = actual["wind_max"]
        if fc_wind is not None and obs_wind is not None:
            overall_wind_forecast.append(fc_wind)
            overall_wind_observed.append(obs_wind)

        # Per-model snowfall from model_comparison
        mc_entry = hist.get("model_comparison_entry")
        if mc_entry:
            snowfall_values = mc_entry.get("snowfall_values", [])
            hist_models = hist.get("models", [])
            for j, model_name in enumerate(hist_models):
                if model_name in per_model_snow_forecast and j < len(snowfall_values):
                    val = snowfall_values[j]
                    if val is not None and obs_snow is not None:
                        per_model_snow_forecast[model_name].append(val)
                        per_model_snow_observed[model_name].append(obs_snow)

        # Powder day accuracy tracking
        forecast_scores_for_powder.append(score_entry)

    if not verified_dates:
        logger.warning("No dates could be verified (no matching historical forecasts)")
        return None

    logger.info("Verified %d dates: %s", len(verified_dates), ", ".join(verified_dates))

    # Compute metrics
    per_model_metrics = {}
    for model_name in models:
        fc = per_model_snow_forecast[model_name]
        obs = per_model_snow_observed[model_name]
        snow_metrics = compute_model_metrics(fc, obs) if fc else None
        if snow_metrics:
            per_model_metrics[model_name] = {"snowfall": snow_metrics}

    overall_metrics = {}
    snow_m = compute_model_metrics(overall_snow_forecast, overall_snow_observed)
    if snow_m:
        overall_metrics["snowfall"] = snow_m
    temp_m = compute_model_metrics(overall_temp_forecast, overall_temp_observed)
    if temp_m:
        overall_metrics["temperature"] = temp_m
    wind_m = compute_model_metrics(overall_wind_forecast, overall_wind_observed)
    if wind_m:
        overall_metrics["wind"] = wind_m

    # Powder day accuracy
    powder_acc = compute_powder_day_accuracy(forecast_scores_for_powder, era5)

    # --- Kalman filter updates ---
    kalman_cfg = config.get("scoring", {}).get("kalman", {})
    Q = kalman_cfg.get("process_noise_q", 0.01)
    R = kalman_cfg.get("observation_noise_r", 1.0)
    kalman_state_path = os.path.join(docs_dir, "verification", "kalman_state.json")
    weights_path = os.path.join(docs_dir, "verification", "model_weights.json")

    # Bootstrap from EWMA if Kalman state doesn't exist yet
    initialize_from_ewma(weights_path, kalman_state_path)

    # Update Kalman filter with each verified forecast-observation pair
    for target_date in verified_dates:
        hist = find_historical_forecast(history_dir, target_date)
        if hist is None:
            continue
        actual = era5_by_date.get(target_date, {})
        mc_entry = hist.get("model_comparison_entry")
        if mc_entry and actual.get("snowfall") is not None:
            snowfall_values = mc_entry.get("snowfall_values", [])
            hist_models = hist.get("models", [])
            for j, m_name in enumerate(hist_models):
                if j < len(snowfall_values) and snowfall_values[j] is not None:
                    kalman_update(m_name, "snowfall",
                                  snowfall_values[j], actual["snowfall"],
                                  kalman_state_path, Q=Q, R=R)

    # --- Lead-time-specific metrics ---
    lead_time_buckets = kalman_cfg.get("lead_time_buckets",
                                        [[0, 2], [3, 5], [6, 10], [11, 16]])
    per_lead_per_model = {}  # {bucket_label: {model: {fc: [], obs: []}}}
    for bucket in lead_time_buckets:
        label = f"{bucket[0]}-{bucket[1]}"
        per_lead_per_model[label] = {m: {"fc": [], "obs": []} for m in models}

    for target_date in verified_dates:
        hist = find_historical_forecast(history_dir, target_date)
        if hist is None:
            continue
        lead_days = hist["lead_days"]
        actual = era5_by_date.get(target_date, {})
        mc_entry = hist.get("model_comparison_entry")
        if not mc_entry or actual.get("snowfall") is None:
            continue
        snowfall_values = mc_entry.get("snowfall_values", [])
        hist_models = hist.get("models", [])
        # Find which bucket this lead time falls into
        for bucket in lead_time_buckets:
            if bucket[0] <= lead_days <= bucket[1]:
                label = f"{bucket[0]}-{bucket[1]}"
                for j, m_name in enumerate(hist_models):
                    if m_name in per_lead_per_model[label] and j < len(snowfall_values):
                        if snowfall_values[j] is not None:
                            per_lead_per_model[label][m_name]["fc"].append(snowfall_values[j])
                            per_lead_per_model[label][m_name]["obs"].append(actual["snowfall"])
                break

    # --- CRPS, Brier, and analog storage ---
    analogs_path = os.path.join(docs_dir, "verification", "analogs.json")
    crps_values = []
    brier_values = []

    for target_date in verified_dates:
        hist = find_historical_forecast(history_dir, target_date)
        if hist is None:
            continue
        actual = era5_by_date.get(target_date, {})
        if actual.get("snowfall") is None:
            continue

        score_entry = hist["score"]
        mc_entry = hist.get("model_comparison_entry")
        hist_models = hist.get("models", [])

        # CRPS from model ensemble
        if mc_entry:
            sf_values = mc_entry.get("snowfall_values", [])
            valid_sf = [v for v in sf_values if v is not None]
            if len(valid_sf) >= 2:
                crps = compute_crps(valid_sf, actual["snowfall"])
                if crps is not None:
                    crps_values.append(crps)

        # Brier score for powder day prediction
        predicted_score = score_entry.get("total", 0)
        # Convert score to probability (60+ = "good day predicted")
        powder_prob = min(1.0, max(0.0, (predicted_score - 30) / 50.0))
        actual_good = (actual["snowfall"] > 10 and
                       (actual.get("temperature_max") or 0) < -2)
        brier = compute_brier(powder_prob, actual_good)
        brier_values.append(brier)

        # Store analog for ML training
        if mc_entry:
            sf_values = mc_entry.get("snowfall_values", [])
            valid_sf = [v for v in sf_values if v is not None]
            if valid_sf:
                features = {
                    "model_predictions": sf_values,
                    "model_names": hist_models,
                    "mean": sum(valid_sf) / len(valid_sf),
                    "spread": max(valid_sf) - min(valid_sf) if len(valid_sf) > 1 else 0,
                    "lead_time": hist["lead_days"],
                    "day_of_year": datetime.strptime(target_date, "%Y-%m-%d").timetuple().tm_yday,
                }
                observed = {
                    "snowfall": actual["snowfall"],
                    "temperature_max": actual.get("temperature_max"),
                    "wind_max": actual.get("wind_max"),
                }
                store_analog(features, observed, analogs_path)

    # Compute model correlations from accumulated analogs
    correlations_path = os.path.join(docs_dir, "verification", "model_correlations.json")
    all_hist = []
    for target_date in verified_dates:
        hist = find_historical_forecast(history_dir, target_date)
        if hist:
            all_hist.append(hist)
    if all_hist:
        corr_result = compute_model_correlations(all_hist)
        if corr_result.get("matrix"):
            os.makedirs(os.path.dirname(correlations_path), exist_ok=True)
            with open(correlations_path, "w") as f:
                json.dump(corr_result, f, indent=2)

    # Update persistent stats
    stats_path = os.path.join(docs_dir, "verification", "stats.json")
    new_metrics = {
        "per_model": per_model_metrics,
        "overall": overall_metrics,
        "powder_accuracy": powder_acc,
        "verified_dates": verified_dates,
        "crps": {
            "mean": round(sum(crps_values) / len(crps_values), 4) if crps_values else None,
            "n_samples": len(crps_values),
        },
        "brier": {
            "mean": round(sum(brier_values) / len(brier_values), 4) if brier_values else None,
            "n_samples": len(brier_values),
        },
    }
    stats = update_verification_stats(new_metrics, stats_path)

    # Build result summary
    result = {
        "verification_window": {"start": start_str, "end": end_str},
        "verified_dates": verified_dates,
        "era5_data": era5,
        "per_model": per_model_metrics,
        "overall": overall_metrics,
        "powder_accuracy": powder_acc,
        "cumulative_stats": stats,
        "kalman_state_path": kalman_state_path,
    }

    logger.info("Verification complete: %d dates, overall snow MAE=%.2f cm",
                len(verified_dates),
                overall_metrics.get("snowfall", {}).get("mae", float("nan")))

    return result
