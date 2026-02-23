"""Cross-source validation — sanity checks across weather data sources.

Runs after data collection, before scoring. Flags temperature discrepancies,
snowfall outliers, rain-at-altitude contradictions, and temporal discontinuities.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUMMIT_ELEVATION = 2400  # meters


def _median(values):
    """Compute median of a list of numbers."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _extract_daily_by_date(source_data, source_name):
    """Extract daily data indexed by date from a source's data dict."""
    if not source_data or source_data.get("error"):
        return {}

    daily = {}
    for d in source_data.get("daily", []):
        date = d.get("date")
        if date:
            daily[date] = d

    return daily


def validate_sources(om_daily_data, yr_daily_data=None, sf_by_date=None, mf_by_date=None):
    """Cross-source validation. Returns dict with flags and quality indicators.

    Args:
        om_daily_data: Open-Meteo daily data (list of dicts with date, temp, snow, etc.)
        yr_daily_data: Yr.no daily data (list of dicts, optional)
        sf_by_date: Snow-Forecast raw data dict (optional)
        mf_by_date: Mountain-Forecast raw data dict (optional)

    Returns:
        dict with "flags" (list of issues) and "source_quality" (per-source status).
    """
    flags = []
    source_issues = {
        "open_meteo": 0,
        "yr_no": 0,
        "snow_forecast": 0,
        "mountain_forecast": 0,
    }

    # Build per-date index for each source
    om_by_date = {}
    if om_daily_data:
        for d in (om_daily_data if isinstance(om_daily_data, list) else []):
            date = d.get("date")
            if date:
                om_by_date[date] = d

    yr_by_date = {}
    if yr_daily_data:
        for d in (yr_daily_data if isinstance(yr_daily_data, list) else []):
            date = d.get("date")
            if date:
                yr_by_date[date] = d

    # sf_by_date and mf_by_date are already dicts indexed by date
    sf_dates = sf_by_date if isinstance(sf_by_date, dict) else {}
    mf_dates = mf_by_date if isinstance(mf_by_date, dict) else {}

    # Collect all dates across sources
    all_dates = sorted(set(
        list(om_by_date.keys()) +
        list(yr_by_date.keys()) +
        list(sf_dates.keys()) +
        list(mf_dates.keys())
    ))

    for date in all_dates:
        # --- Temperature consistency ---
        temps = {}
        om = om_by_date.get(date, {})
        yr = yr_by_date.get(date, {})
        sf = sf_dates.get(date, {})
        mf = mf_dates.get(date, {})

        if om.get("temperature_c") is not None:
            temps["open_meteo"] = om["temperature_c"]
        elif om.get("temp_c") is not None:
            temps["open_meteo"] = om["temp_c"]

        if yr.get("temperature_c") is not None:
            temps["yr_no"] = yr["temperature_c"]
        elif yr.get("temp_c") is not None:
            temps["yr_no"] = yr["temp_c"]

        # Scrapers often have min/max — use average
        if sf:
            sf_min = sf.get("temp_min_c")
            sf_max = sf.get("temp_max_c")
            if sf_min is not None and sf_max is not None:
                temps["snow_forecast"] = (sf_min + sf_max) / 2
            elif sf.get("temperature_c") is not None:
                temps["snow_forecast"] = sf["temperature_c"]

        if mf:
            mf_min = mf.get("temp_min_c")
            mf_max = mf.get("temp_max_c")
            if mf_min is not None and mf_max is not None:
                temps["mountain_forecast"] = (mf_min + mf_max) / 2
            elif mf.get("temperature_c") is not None:
                temps["mountain_forecast"] = mf["temperature_c"]

        if len(temps) >= 2:
            temp_values = list(temps.values())
            med = _median(temp_values)
            for src, val in temps.items():
                if abs(val - med) > 5:
                    flags.append({
                        "type": "temp_discrepancy",
                        "date": date,
                        "detail": f"{src} reports {val:.1f}°C, median is {med:.1f}°C (diff: {abs(val - med):.1f}°C)",
                        "severity": "warning",
                        "source": src,
                    })
                    source_issues[src] = source_issues.get(src, 0) + 1

        # --- Snowfall outlier ---
        snows = {}
        if om.get("snowfall_24h_cm") is not None:
            snows["open_meteo"] = om["snowfall_24h_cm"]
        elif om.get("snow_cm") is not None:
            snows["open_meteo"] = om["snow_cm"]

        if yr.get("snowfall_24h_cm") is not None:
            snows["yr_no"] = yr["snowfall_24h_cm"]
        elif yr.get("snow_cm") is not None:
            snows["yr_no"] = yr["snow_cm"]

        if sf.get("snow_total_cm") is not None:
            snows["snow_forecast"] = sf["snow_total_cm"]

        if mf.get("snow_total_cm") is not None:
            snows["mountain_forecast"] = mf["snow_total_cm"]

        if len(snows) >= 2:
            snow_values = list(snows.values())
            med_snow = _median(snow_values)
            for src, val in snows.items():
                # Other sources' values (excluding current)
                others = [v for k, v in snows.items() if k != src]
                other_med = _median(others)
                if other_med is not None and other_med > 0 and val > 3 * other_med:
                    flags.append({
                        "type": "snowfall_outlier",
                        "date": date,
                        "detail": f"{src} reports {val:.1f}cm snow, others' median is {other_med:.1f}cm ({val/other_med:.1f}x)",
                        "severity": "warning",
                        "source": src,
                    })
                    source_issues[src] = source_issues.get(src, 0) + 1

        # --- Rain-at-altitude contradiction ---
        # Check if freezing level > summit AND snowfall reported > 5cm
        freeze_levels = []
        snow_reports = []

        om_fl = om.get("freezing_level_m")
        if om_fl is not None:
            freeze_levels.append(om_fl)
        if sf.get("freezing_level_avg_m") is not None:
            freeze_levels.append(sf["freezing_level_avg_m"])
        if mf.get("freezing_level_avg_m") is not None:
            freeze_levels.append(mf["freezing_level_avg_m"])

        for src, val in snows.items():
            if val > 5:
                snow_reports.append((src, val))

        if freeze_levels and snow_reports:
            max_fl = max(freeze_levels)
            if max_fl > SUMMIT_ELEVATION:
                for src, val in snow_reports:
                    flags.append({
                        "type": "rain_at_altitude",
                        "date": date,
                        "detail": f"Freezing level {max_fl:.0f}m (above summit {SUMMIT_ELEVATION}m) "
                                  f"but {src} reports {val:.1f}cm snow — likely rain, not snow",
                        "severity": "warning",
                        "source": src,
                    })
                    source_issues[src] = source_issues.get(src, 0) + 1

    # --- Temporal continuity ---
    # For each source, check for sharp troughs between high-snow days
    for src_name, src_dates in [("open_meteo", om_by_date), ("yr_no", yr_by_date),
                                 ("snow_forecast", sf_dates), ("mountain_forecast", mf_dates)]:
        if not src_dates:
            continue

        sorted_dates = sorted(src_dates.keys())
        for idx in range(1, len(sorted_dates) - 1):
            prev_date = sorted_dates[idx - 1]
            curr_date = sorted_dates[idx]
            next_date = sorted_dates[idx + 1]

            prev_d = src_dates[prev_date]
            curr_d = src_dates[curr_date]
            next_d = src_dates[next_date]

            # Get snow values with various key names
            prev_snow = prev_d.get("snowfall_24h_cm", prev_d.get("snow_total_cm", prev_d.get("snow_cm", 0))) or 0
            curr_snow = curr_d.get("snowfall_24h_cm", curr_d.get("snow_total_cm", curr_d.get("snow_cm", 0))) or 0
            next_snow = next_d.get("snowfall_24h_cm", next_d.get("snow_total_cm", next_d.get("snow_cm", 0))) or 0

            # Sharp trough: both neighbors > 15cm more than current
            if prev_snow - curr_snow > 15 and next_snow - curr_snow > 15:
                flags.append({
                    "type": "temporal_discontinuity",
                    "date": curr_date,
                    "detail": f"{src_name} has sharp trough: {prev_snow:.0f}cm → {curr_snow:.0f}cm → {next_snow:.0f}cm",
                    "severity": "warning",
                    "source": src_name,
                })
                source_issues[src_name] = source_issues.get(src_name, 0) + 1

    # Build source quality summary
    source_quality = {}
    for src, issues in source_issues.items():
        if issues == 0:
            source_quality[src] = {"status": "ok", "issues": 0}
        elif issues <= 2:
            source_quality[src] = {"status": "degraded", "issues": issues}
        else:
            source_quality[src] = {"status": "poor", "issues": issues}

    return {
        "flags": flags,
        "source_quality": source_quality,
    }
