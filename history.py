"""Historical trend aggregation for the dashboard.

Reads history files and builds summary data for snow depth charts
and score evolution tracking.
"""

import json
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def build_history_summary(history_dir):
    """Read history files and build summary for dashboard.

    Args:
        history_dir: path to docs/history/ directory

    Returns dict with:
    - snow_depth_series: list of {date, depth_cm} — last 30 days of depth readings
    - score_evolution: dict of {forecast_date: [{run_date, score}]} — how scores changed
    """
    result = {
        "snow_depth_series": [],
        "score_evolution": {},
    }

    if not os.path.isdir(history_dir):
        logger.debug("History directory does not exist yet")
        return result

    # List and sort history files (newest first), limit to 90 (30 days * 3/day)
    try:
        files = sorted(
            [f for f in os.listdir(history_dir) if f.startswith("forecast_") and f.endswith(".json")],
            reverse=True,
        )
    except OSError as e:
        logger.warning(f"Could not list history files: {e}")
        return result

    files = files[:90]  # Last 30 days

    depth_series = []
    score_evolution = {}

    for filename in reversed(files):  # Process oldest first for chronological order
        filepath = os.path.join(history_dir, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Extract run timestamp from filename: forecast_20260223_120000.json
        run_date = _parse_run_date(filename)
        if not run_date:
            continue

        # Snow depth from current conditions
        current = data.get("current", {})
        snow_depth = current.get("snow_depth")
        if snow_depth is not None:
            depth_series.append({
                "date": run_date,
                "depth_cm": round(snow_depth * 100, 1),
            })

        # Score evolution: track how each forecast date's score changed across runs
        scores = data.get("scores", [])
        for s in scores[:7]:  # Only 7-day window
            fdate = s.get("date", "")
            if not fdate:
                continue
            if fdate not in score_evolution:
                score_evolution[fdate] = []
            score_evolution[fdate].append({
                "run_date": run_date,
                "score": s.get("total", 0),
            })

    # Deduplicate depth series (keep one per day — the last run)
    seen_dates = {}
    for entry in depth_series:
        day_key = entry["date"][:10]
        seen_dates[day_key] = entry
    result["snow_depth_series"] = list(seen_dates.values())

    result["score_evolution"] = score_evolution

    return result


def _parse_run_date(filename):
    """Parse run date from history filename: forecast_20260223_120000.json -> 2026-02-23T12:00:00."""
    try:
        # forecast_20260223_120000.json
        parts = filename.replace("forecast_", "").replace(".json", "")
        dt = datetime.strptime(parts, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except (ValueError, IndexError):
        return None
