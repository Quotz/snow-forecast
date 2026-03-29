#!/usr/bin/env python3
"""Generate training data and train the local ML model from historical data.

Uses Open-Meteo Historical Forecast API + ERA5 reanalysis to build a dataset
of forecast-observation pairs for the entire winter season. Then trains a
Random Forest that learns which model combinations predict best for
Popova Shapka specifically.

Usage:
    python3 train_model.py                          # Default: last 90 days
    python3 train_model.py --days 180               # Last 180 days
    python3 train_model.py --start 2025-11-01 --end 2026-03-15  # Specific range

The trained model is saved to docs/verification/ml_model.pkl and the training
data to docs/verification/analogs.json.
"""
from __future__ import annotations

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")

# --- Configuration ---

LOCATION = {"lat": 42.0, "lon": 20.87, "elevation": 2400}

# Physics models only — Historical Forecast API doesn't archive AI models
MODELS = [
    "icon_seamless",
    "ecmwf_ifs025",
    "gfs_seamless",
    "arpege_seamless",
    "ukmo_seamless",
]

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_PARAMS = ["snowfall_sum", "temperature_2m_max", "temperature_2m_min",
                "wind_speed_10m_max", "wind_gusts_10m_max", "precipitation_sum",
                "sunshine_duration"]

ERA5_PARAMS = ["temperature_2m_max", "temperature_2m_min", "snowfall_sum",
               "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max"]


def _monthly_batches(start_date, end_date):
    """Split a date range into monthly batches."""
    batches = []
    current = start_date
    while current <= end_date:
        month_end = min(
            current.replace(day=28) + timedelta(days=4),
            end_date
        )
        month_end = min(month_end.replace(day=1) - timedelta(days=1), end_date)
        if month_end < current:
            month_end = end_date
        batches.append((current, month_end))
        current = month_end + timedelta(days=1)
    return batches


def fetch_historical_forecasts(start_date, end_date):
    """Fetch archived model predictions."""
    results = {}  # date -> {model: snowfall}

    batches = _monthly_batches(start_date, end_date)
    for batch_start, batch_end in batches:
        logger.info(f"Fetching forecasts: {batch_start} to {batch_end}")

        params = {
            "latitude": LOCATION["lat"],
            "longitude": LOCATION["lon"],
            "elevation": LOCATION["elevation"],
            "start_date": batch_start.strftime("%Y-%m-%d"),
            "end_date": batch_end.strftime("%Y-%m-%d"),
            "daily": ",".join(DAILY_PARAMS),
            "models": ",".join(MODELS),
            "timezone": "UTC",
        }

        try:
            resp = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.warning(f"API error: {data.get('reason', data['error'])}")
                time.sleep(1)
                continue

            daily = data.get("daily", {})
            dates = daily.get("time", [])

            for i, date_str in enumerate(dates):
                if date_str not in results:
                    results[date_str] = {"models": {}, "daily": {}}

                # Per-model snowfall
                for model in MODELS:
                    key = f"snowfall_sum_{model}"
                    if key in daily and i < len(daily[key]):
                        val = daily[key][i]
                        if val is not None:
                            results[date_str]["models"][model] = val

                # Daily aggregates
                for param in DAILY_PARAMS:
                    if param in daily and i < len(daily[param]):
                        results[date_str]["daily"][param] = daily[param][i]
                    # Also check per-model versions
                    for model in MODELS:
                        mkey = f"{param}_{model}"
                        if mkey in daily and i < len(daily[mkey]) and daily[mkey][i] is not None:
                            if param not in results[date_str]["daily"] or results[date_str]["daily"][param] is None:
                                results[date_str]["daily"][param] = daily[mkey][i]

        except requests.RequestException as e:
            logger.warning(f"Fetch failed: {e}")

        time.sleep(0.5)

    return results


def fetch_era5(start_date, end_date):
    """Fetch ERA5 ground truth."""
    results = {}  # date -> {snowfall, temp_max, ...}

    batches = _monthly_batches(start_date, end_date)
    for batch_start, batch_end in batches:
        logger.info(f"Fetching ERA5: {batch_start} to {batch_end}")

        params = {
            "latitude": LOCATION["lat"],
            "longitude": LOCATION["lon"],
            "elevation": LOCATION["elevation"],
            "start_date": batch_start.strftime("%Y-%m-%d"),
            "end_date": batch_end.strftime("%Y-%m-%d"),
            "daily": ",".join(ERA5_PARAMS),
            "timezone": "UTC",
        }

        try:
            resp = requests.get(ERA5_ARCHIVE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.warning(f"ERA5 error: {data.get('reason', data['error'])}")
                time.sleep(1)
                continue

            daily = data.get("daily", {})
            dates = daily.get("time", [])

            for i, date_str in enumerate(dates):
                results[date_str] = {
                    "snowfall": daily.get("snowfall_sum", [None])[i] if i < len(daily.get("snowfall_sum", [])) else None,
                    "temperature_max": daily.get("temperature_2m_max", [None])[i] if i < len(daily.get("temperature_2m_max", [])) else None,
                    "temperature_min": daily.get("temperature_2m_min", [None])[i] if i < len(daily.get("temperature_2m_min", [])) else None,
                    "wind_max": daily.get("wind_speed_10m_max", [None])[i] if i < len(daily.get("wind_speed_10m_max", [])) else None,
                    "precipitation": daily.get("precipitation_sum", [None])[i] if i < len(daily.get("precipitation_sum", [])) else None,
                }

        except requests.RequestException as e:
            logger.warning(f"ERA5 fetch failed: {e}")

        time.sleep(0.5)

    return results


def build_training_data(forecasts, era5):
    """Build analog training pairs from historical forecasts + ERA5.

    Each pair has:
    - features: model predictions, mean, spread, day_of_year
    - observed: ERA5 actual values
    """
    from ensemble_stats import store_analog

    analogs_path = str(Path(__file__).parent / "docs" / "verification" / "analogs.json")
    pairs_created = 0
    skipped = 0

    for date_str in sorted(forecasts.keys()):
        if date_str not in era5:
            skipped += 1
            continue

        fc = forecasts[date_str]
        obs = era5[date_str]

        if obs.get("snowfall") is None:
            skipped += 1
            continue

        model_preds = []
        model_names = []
        for model in MODELS:
            if model in fc["models"]:
                model_preds.append(fc["models"][model])
                model_names.append(model)

        if len(model_preds) < 3:
            skipped += 1
            continue

        mean_pred = sum(model_preds) / len(model_preds)
        spread = max(model_preds) - min(model_preds)

        try:
            doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
        except ValueError:
            doy = 180

        features = {
            "model_predictions": model_preds,
            "model_names": model_names,
            "mean": round(mean_pred, 2),
            "spread": round(spread, 2),
            "lead_time": 2,  # Historical API gives ~2-day lead equivalent
            "day_of_year": doy,
            "date": date_str,
        }

        observed = {
            "snowfall": obs["snowfall"],
            "temperature_max": obs.get("temperature_max"),
            "temperature_min": obs.get("temperature_min"),
            "wind_max": obs.get("wind_max"),
        }

        store_analog(features, observed, analogs_path, max_entries=1000)
        pairs_created += 1

    logger.info(f"Created {pairs_created} training pairs ({skipped} skipped)")
    return pairs_created


def train(analogs_path):
    """Train the ML model on accumulated analog data."""
    from ml_postprocess import train_model, should_use_ml

    model_path = str(Path(__file__).parent / "docs" / "verification" / "ml_model.pkl")

    if not should_use_ml(analogs_path):
        logger.warning("Not enough analog data for training")
        return None

    result = train_model(analogs_path, model_path)
    if result:
        logger.info(f"Model trained: CV MAE = {result['cv_mae']:.2f} cm on {result['n_samples']} samples")
        logger.info(f"Feature importances: {json.dumps(result['feature_importances'], indent=2)}")

        # Save training report
        report_path = str(Path(__file__).parent / "docs" / "verification" / "ml_training_report.json")
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate training data and train local ML model")
    parser.add_argument("--days", type=int, default=90, help="Number of days to look back")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train-only", action="store_true", help="Skip data fetch, just train on existing analogs")
    args = parser.parse_args()

    if args.train_only:
        analogs_path = str(Path(__file__).parent / "docs" / "verification" / "analogs.json")
        result = train(analogs_path)
        if result:
            print(f"\nModel trained: CV MAE = {result['cv_mae']:.2f} cm")
        else:
            print("\nTraining failed — not enough data?")
        return

    # Determine date range
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end = datetime.utcnow() - timedelta(days=5)  # ERA5 has 5-day lag
        start = end - timedelta(days=args.days)

    print(f"Generating training data: {start.date()} to {end.date()}")
    print(f"Models: {', '.join(MODELS)}")
    print()

    # Fetch data
    forecasts = fetch_historical_forecasts(start, end)
    print(f"Fetched forecasts for {len(forecasts)} dates")

    era5 = fetch_era5(start, end)
    print(f"Fetched ERA5 for {len(era5)} dates")

    # Build training pairs
    n_pairs = build_training_data(forecasts, era5)
    print(f"Created {n_pairs} training pairs")

    # Train model
    analogs_path = str(Path(__file__).parent / "docs" / "verification" / "analogs.json")
    result = train(analogs_path)

    if result:
        print(f"\nModel trained successfully!")
        print(f"  CV MAE: {result['cv_mae']:.2f} cm")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Feature importances:")
        for feat, imp in sorted(result['feature_importances'].items(), key=lambda x: -x[1]):
            print(f"    {feat:15s}: {imp:.4f}")
    else:
        print(f"\nTraining failed — need at least 30 pairs, have {n_pairs}")


if __name__ == "__main__":
    main()
