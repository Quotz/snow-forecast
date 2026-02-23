#!/usr/bin/env python3
"""Snow Monitor — Powder forecast monitoring system.

Usage:
    python main.py                  # Full run: fetch, score, report, notify
    python main.py --no-notify      # Skip notifications (for testing)
    python main.py --dashboard-only # Only regenerate dashboard from latest.json
    python main.py --json-only      # Fetch and score, output JSON only
"""

import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Ensure the project root is on the path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from collectors import (OpenMeteoCollector, YrNoCollector,
                        SnowForecastCollector, MountainForecastCollector,
                        OpenMeteoEnsembleCollector)
from scoring import calculate_powder_score, smooth_scores
from patterns import detect_all_patterns
from report import generate_dashboard, save_latest_json, save_history
from data_extract import (extract_daily_data_from_open_meteo, build_model_comparison,
                          _safe, extract_yr_daily_data)
from analysis import (build_chart_data, build_safety_flags, _build_source_comparison,
                      _format_chart_date, estimate_avalanche_danger,
                      build_multi_chart_data, build_model_spread)
from notify import notify_if_needed
from validation import validate_sources
from insights import generate_insights
from forecast_diff import compute_forecast_diff
from history import build_history_summary

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("snow_monitor")


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(PROJECT_DIR, "config.yaml")

    with open(config_path) as f:
        return yaml.safe_load(f)


def run(config: dict, no_notify: bool = False, dashboard_only: bool = False,
        json_only: bool = False):
    """Main execution flow."""

    # Get first location config
    loc_name = list(config["locations"].keys())[0]
    loc_cfg = config["locations"][loc_name]
    location = {
        "name": loc_cfg["name"],
        "lat": loc_cfg["lat"],
        "lon": loc_cfg["lon"],
        "timezone": loc_cfg.get("timezone", "UTC"),
        "elevations": loc_cfg["elevations"],
    }
    models = config.get("models", ["icon_seamless", "ecmwf_ifs025", "gfs_seamless"])
    scoring_cfg = config.get("scoring", {})
    sky_cfg = config.get("sky", {})
    data_cfg = config.get("data", {})

    # Use summit elevation for scoring
    summit_elev = max(location["elevations"].values())
    mid_elev = min(location["elevations"].values())

    if dashboard_only:
        # Regenerate dashboard from existing latest.json
        latest_path = os.path.join(PROJECT_DIR, data_cfg.get("latest_file", "docs/latest.json"))
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                report_data = json.load(f)
            dashboard_path = os.path.join(PROJECT_DIR, data_cfg.get("dashboard_file", "docs/index.html"))
            generate_dashboard(report_data, os.path.join(PROJECT_DIR, "templates"), dashboard_path)
            logger.info("Dashboard regenerated from latest.json")
        else:
            logger.error("No latest.json found. Run a full fetch first.")
        return

    # ===== FETCH DATA =====
    logger.info(f"Starting data collection for {location['name']}")

    collectors_list = [
        OpenMeteoCollector(config),
        YrNoCollector(config),
    ]

    # Add web scrapers if enabled
    if config.get("scrapers", {}).get("enabled", False):
        collectors_list.append(SnowForecastCollector(config))
        collectors_list.append(MountainForecastCollector(config))

    results = {}
    for collector in collectors_list:
        data = collector.safe_fetch(location)
        results[collector.name] = data
        if data.get("error"):
            logger.warning(f"{collector.name}: {data['error']}")

    # Ensemble collector (separate — single request for uncertainty data)
    ensemble_collector = OpenMeteoEnsembleCollector(config)
    ensemble_data = ensemble_collector.safe_fetch(location)
    if ensemble_data.get("error"):
        logger.warning(f"Ensemble: {ensemble_data['error']}")
        ensemble_data = None
    else:
        results["open_meteo_ensemble"] = ensemble_data

    om_data = results.get("open_meteo", {})
    yr_data = results.get("yr_no", {})
    sf_data = results.get("snow_forecast", {})
    mf_data = results.get("mountain_forecast", {})

    if om_data.get("error") and not om_data.get("elevations"):
        logger.error("Open-Meteo failed and is our primary source. Aborting.")
        return

    # ===== EXTRACT DATA =====
    logger.info("Extracting daily data")
    daily_data = extract_daily_data_from_open_meteo(om_data, summit_elev, models)

    # Extract Yr.no daily data (now activated!)
    yr_daily_data = []
    yr_by_date = {}
    if yr_data and not yr_data.get("error"):
        yr_daily_data = extract_yr_daily_data(yr_data, summit_elev, mid_elev)
        for d in yr_daily_data:
            yr_by_date[d["date"]] = d
        logger.info(f"Yr.no: extracted {len(yr_daily_data)} daily records")

    # Index scraper daily data by date for cross-source scoring
    sf_by_date = {}
    if sf_data and not sf_data.get("error"):
        for d in sf_data.get("daily", []):
            sf_by_date[d["date"]] = d
    mf_by_date = {}
    if mf_data and not mf_data.get("error"):
        for d in mf_data.get("daily", []):
            mf_by_date[d["date"]] = d

    # ===== VALIDATE =====
    logger.info("Running cross-source validation")
    validation_result = validate_sources(daily_data, yr_daily_data, sf_by_date, mf_by_date)
    if validation_result.get("flags"):
        for flag in validation_result["flags"]:
            logger.warning(f"Validation: {flag['type']} on {flag['date']}: {flag['detail']}")

    # ===== SCORE =====
    logger.info("Calculating powder scores")

    scores = []
    for day in daily_data:
        # Collect scraper + Yr.no snowfall values for this date
        scraper_snow = []
        sf_day = sf_by_date.get(day["date"])
        if sf_day and sf_day.get("snow_total_cm") is not None:
            scraper_snow.append(sf_day["snow_total_cm"])
        mf_day = mf_by_date.get(day["date"])
        if mf_day and mf_day.get("snow_total_cm") is not None:
            scraper_snow.append(mf_day["snow_total_cm"])
        yr_day = yr_by_date.get(day["date"])
        if yr_day and yr_day.get("snowfall_24h_cm") is not None:
            scraper_snow.append(yr_day["snowfall_24h_cm"])

        # Build ensemble day data if available
        ensemble_day = None
        if ensemble_data and not ensemble_data.get("error"):
            ens_dates = ensemble_data.get("daily", {}).get("dates", [])
            if day["date"] in ens_dates:
                idx = ens_dates.index(day["date"])
                ens_sf = ensemble_data["daily"].get("snowfall", {})
                ensemble_day = {}
                for pct in ("p10", "p25", "p50", "p75", "p90"):
                    vals = ens_sf.get(pct, [])
                    ensemble_day[pct] = vals[idx] if idx < len(vals) else None

        score = calculate_powder_score(day, scoring_cfg, sky_cfg,
                                        scraper_snow_values=scraper_snow or None,
                                        ensemble_day_data=ensemble_day,
                                        location_cfg=loc_cfg)
        score["date"] = day["date"]
        scores.append(score)

    # Apply post-processing smoothing
    scores = smooth_scores(scores)

    # ===== DETECT PATTERNS =====
    patterns = detect_all_patterns(scores, config=config)
    if patterns:
        logger.info(f"Detected {len(patterns)} pattern(s)")
        for p in patterns:
            logger.info(f"  {p['type']}: {p['message']}")

    # ===== BUILD REPORT DATA =====
    dates = [s["date"] for s in scores]
    model_comparison = build_model_comparison(om_data, summit_elev, models)
    chart_data = build_chart_data(model_comparison, models)
    safety_flags = build_safety_flags(scores)

    # Current conditions (first hour of first model at summit)
    current = {}
    if summit_elev in om_data.get("elevations", {}):
        first_model = models[0] if models else None
        if first_model:
            mh = om_data["elevations"][summit_elev].get("models", {}).get(first_model, {}).get("hourly", {})
            if mh.get("temperature_2m"):
                current = {
                    "elevation": summit_elev,
                    "temperature": _safe(mh["temperature_2m"][0]),
                    "wind_speed": _safe(mh.get("wind_speed_10m", [None])[0]),
                    "snow_depth": _safe(mh.get("snow_depth", [None])[0]),
                    "freezing_level": _safe(mh.get("freezing_level_height", [None])[0]),
                    "cloud_cover": _safe(mh.get("cloud_cover", [None])[0]),
                    "humidity": _safe(mh.get("relative_humidity_2m", [None])[0]),
                }

    # Cross-source snowfall comparison
    source_comparison = _build_source_comparison(scores, sf_data, mf_data)

    # Avalanche danger estimation
    avalanche_danger = estimate_avalanche_danger(scores)

    # Multi-chart data (temperature, wind, freezing level, snow depth)
    multi_charts = build_multi_chart_data(scores, models)

    # Model spread (min/max across models per day)
    model_spread = build_model_spread(model_comparison)

    # Generate actionable insights
    insights = generate_insights(scores, patterns, current)
    if insights:
        logger.info(f"Insight: {insights.get('headline', '')}")

    # Historical trends
    history_dir = os.path.join(PROJECT_DIR, data_cfg.get("history_dir", "docs/history"))
    history_summary = build_history_summary(history_dir)

    # Forecast diff (MUST run before save_latest_json overwrites the file)
    latest_path = os.path.join(PROJECT_DIR, data_cfg.get("latest_file", "docs/latest.json"))
    forecast_diff_result = None
    if os.path.exists(latest_path):
        forecast_diff_result = compute_forecast_diff(scores, latest_path)
        if forecast_diff_result and forecast_diff_result.get("should_alert"):
            logger.info(f"Forecast change: {forecast_diff_result.get('summary', '')}")

    # Source health tracking
    source_health = []
    for name, data in results.items():
        status = "ok" if not data.get("error") else "error"
        fetched_at = data.get("fetched_at", "")
        source_health.append({
            "name": name,
            "status": status,
            "fetched_at": fetched_at,
            "age": "just now",
        })

    # Persist source health
    health_path = os.path.join(PROJECT_DIR, "docs", ".source_health.json")
    try:
        os.makedirs(os.path.dirname(health_path), exist_ok=True)
        with open(health_path, "w") as f:
            json.dump(source_health, f)
    except Exception:
        pass

    report_data = {
        "location": location["name"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sources": [k for k, v in results.items() if not v.get("error")],
        "scores": scores,
        "dates": dates,
        "patterns": patterns,
        "model_comparison": model_comparison,
        "chart_data": chart_data,
        "models": models,
        "current": current,
        "safety_flags": safety_flags,
        "source_comparison": source_comparison,
        "insights": insights,
        "avalanche_danger": avalanche_danger,
        "multi_charts": multi_charts,
        "model_spread": model_spread,
        "validation": validation_result,
        "ensemble": ensemble_data if ensemble_data and not ensemble_data.get("error") else None,
        "source_health": source_health,
        "history_summary": history_summary,
        "forecast_diff": forecast_diff_result,
        "score_weights": {
            "snow": 35, "quality": 10, "temperature": 15, "wind": 10,
            "freezing_level": 5, "timing": 5, "depth": 5, "models": 15,
            "loading": 5, "ensemble": 5,
        },
    }

    # ===== OUTPUT =====
    save_latest_json(report_data, latest_path)
    save_history(report_data, history_dir)

    if not json_only:
        dashboard_path = os.path.join(PROJECT_DIR, data_cfg.get("dashboard_file", "docs/index.html"))
        generate_dashboard(report_data, os.path.join(PROJECT_DIR, "templates"), dashboard_path)

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"  SNOW MONITOR — {location['name']}")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    if insights:
        print(f"  {insights.get('headline', '')}\n")

    for s in scores[:7]:
        emoji = {"EPIC": "***", "GOOD": "** ", "FAIR": "*  ", "MARGINAL": "   ", "SKIP": "   "}
        e = emoji.get(s["label"], "   ")
        cond = s["conditions"]
        sky_class = s["sky"]["classification"].replace("_", " ").title()
        date_fmt = _format_chart_date(s["date"])
        conf = s.get("source_confidence", {}).get("confidence", "")
        conf_pct = s.get("confidence_pct", 100)
        print(f"  {e} {date_fmt:>10s}  Score: {s['total']:5.1f} ({s['label']:8s})  "
              f"Snow: {cond['snowfall_24h_cm']:5.1f}cm  "
              f"Temp: {cond['temperature_c']:5.1f}\u00b0C  "
              f"Wind: {cond['wind_speed_kmh']:4.0f}km/h  "
              f"{sky_class:16s} [{conf}] {conf_pct:.0f}%")

    if patterns:
        print(f"\n  Patterns:")
        for p in patterns:
            print(f"    {p['message']}")

    if avalanche_danger:
        danger_strs = [f"{ad['date'][-5:]}: L{ad['level']}" for ad in avalanche_danger[:7]]
        print(f"\n  Avalanche: {' | '.join(danger_strs)}")

    # Show source comparison if scrapers provided data
    if source_comparison and any(r.get("snow_forecast") or r.get("mountain_forecast") for r in source_comparison):
        print(f"\n  Source Comparison (Snow cm):")
        print(f"  {'Date':>12s}  {'Open-Meteo':>10s}  {'Snow-Fcst':>10s}  {'Mtn-Fcst':>10s}  {'Consensus':>10s}")
        for row in source_comparison:
            om = f"{row['open_meteo']['snow_cm']:.1f}"
            sf = f"{row['snow_forecast']['snow_cm']:.1f}" if row.get("snow_forecast") else "\u2014"
            mf = f"{row['mountain_forecast']['snow_cm']:.1f}" if row.get("mountain_forecast") else "\u2014"
            con = row.get("consensus", "\u2014")
            print(f"  {row['date']:>12s}  {om:>10s}  {sf:>10s}  {mf:>10s}  {con:>10s}")

    print(f"\n  Sources: {', '.join(report_data['sources'])}")
    print(f"  Dashboard: {os.path.abspath(os.path.join(PROJECT_DIR, data_cfg.get('dashboard_file', 'docs/index.html')))}")
    print()

    # ===== NOTIFY =====
    if not no_notify:
        notify_if_needed(scores, dates, patterns, location["name"], scoring_cfg,
                         insights=insights, avalanche_danger=avalanche_danger,
                         forecast_diff=forecast_diff_result)


def main():
    parser = argparse.ArgumentParser(description="Snow Monitor \u2014 Powder forecast system")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--no-notify", action="store_true", help="Skip notifications")
    parser.add_argument("--dashboard-only", action="store_true", help="Regenerate dashboard from latest.json")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only, no dashboard")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    run(config, no_notify=args.no_notify, dashboard_only=args.dashboard_only,
        json_only=args.json_only)


if __name__ == "__main__":
    main()
