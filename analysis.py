"""Analysis utilities — chart data, safety flags, source comparison,
avalanche danger estimation, multi-chart data builder, model spread."""

import math
from datetime import datetime


def _format_chart_date(iso_date: str) -> str:
    """Format ISO date for chart labels: 'Wed 19/02'."""
    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    try:
        d = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")
        return f"{DAY_NAMES[d.weekday()]} {d.strftime('%d/%m')}"
    except (ValueError, TypeError):
        return str(iso_date)


def build_chart_data(model_comparison: list, models: list) -> dict:
    """Build Chart.js-compatible data structure."""
    colors = [
        "rgba(77,141,247,0.7)",    # ICON - blue
        "rgba(60,192,104,0.7)",    # ECMWF - green
        "rgba(240,194,48,0.7)",    # GFS - yellow
        "rgba(187,134,252,0.7)",   # ARPEGE - purple
        "rgba(255,138,101,0.7)",   # UKMO - orange
        "rgba(0,188,212,0.7)",     # AIFS - cyan
        "rgba(233,30,99,0.7)",     # GraphCast - pink
    ]

    # AI model name formatting
    AI_NAME_MAP = {
        "ecmwf_aifs025": "AIFS",
        "ecmwf_aifs025_single": "AIFS",
        "graphcast025": "GraphCast",
        "gfs_graphcast025": "GraphCast",
    }

    labels = [_format_chart_date(row["date"]) for row in model_comparison[:7]]

    datasets = []
    for j, model in enumerate(models):
        name = AI_NAME_MAP.get(model, model.replace("_seamless", "").replace("_ifs025", "").upper())
        data = [row["snowfall_values"][j] if j < len(row["snowfall_values"]) else 0 for row in model_comparison[:7]]
        datasets.append({
            "label": name,
            "data": data,
            "color": colors[j % len(colors)],
        })

    return {"labels": labels, "datasets": datasets}


def build_safety_flags(scores: list) -> list:
    """Generate safety warning flags."""
    flags = []

    for i, s in enumerate(scores[:7]):
        cond = s.get("conditions", {})
        date = s.get("date", f"Day {i+1}")

        if cond.get("wind_gust_kmh", 0) > 50:
            flags.append(f"{date}: Extreme wind gusts ({cond['wind_gust_kmh']:.0f} km/h). Exposed terrain dangerous.")

        if cond.get("freezing_level_m", 0) > 2200:
            flags.append(f"{date}: Freezing level above freeride zone ({cond['freezing_level_m']:.0f}m). Rain risk at summit.")

        if cond.get("snowfall_24h_cm", 0) > 30:
            flags.append(f"{date}: Heavy snowfall ({cond['snowfall_24h_cm']:.0f}cm). Elevated avalanche risk. Natural avalanches likely.")

    return flags


def _build_source_comparison(scores: list, sf_data: dict, mf_data: dict) -> list:
    """Build a cross-source snowfall comparison table.

    Compares Open-Meteo (from scores) with Snow-Forecast and Mountain-Forecast
    scraped data, showing where sources agree or disagree.
    """
    comparison = []

    # Index scraper daily data by date
    sf_daily = {}
    if sf_data and not sf_data.get("error"):
        for d in sf_data.get("daily", []):
            sf_daily[d["date"]] = d

    mf_daily = {}
    if mf_data and not mf_data.get("error"):
        for d in mf_data.get("daily", []):
            mf_daily[d["date"]] = d

    for s in scores[:7]:
        date = s.get("date", "")
        om_snow = s.get("conditions", {}).get("snowfall_24h_cm", 0)
        om_temp = s.get("conditions", {}).get("temperature_c")
        om_wind = s.get("conditions", {}).get("wind_speed_kmh")

        sf = sf_daily.get(date, {})
        mf = mf_daily.get(date, {})

        row = {
            "date": date,
            "open_meteo": {
                "snow_cm": round(om_snow, 1),
                "temp_c": round(om_temp, 1) if om_temp is not None else None,
                "wind_kmh": round(om_wind, 0) if om_wind is not None else None,
            },
            "snow_forecast": {
                "snow_cm": sf.get("snow_total_cm"),
                "temp_min_c": sf.get("temp_min_c"),
                "temp_max_c": sf.get("temp_max_c"),
                "wind_kmh": sf.get("wind_max_kmh"),
                "freeze_m": sf.get("freezing_level_avg_m"),
            } if sf else None,
            "mountain_forecast": {
                "snow_cm": mf.get("snow_total_cm"),
                "temp_min_c": mf.get("temp_min_c"),
                "temp_max_c": mf.get("temp_max_c"),
                "wind_kmh": mf.get("wind_max_kmh"),
                "freeze_m": mf.get("freezing_level_avg_m"),
            } if mf else None,
        }

        # Calculate consensus (do sources agree on snow?)
        snow_values = [om_snow]
        if sf:
            snow_values.append(sf.get("snow_total_cm", 0))
        if mf:
            snow_values.append(mf.get("snow_total_cm", 0))

        all_snow = all(v > 2 for v in snow_values)
        no_snow = all(v < 1 for v in snow_values)
        if all_snow:
            row["consensus"] = "ALL_SNOW"
        elif no_snow:
            row["consensus"] = "ALL_DRY"
        else:
            row["consensus"] = "MIXED"

        comparison.append(row)

    return comparison


def estimate_avalanche_danger(scores: list) -> list:
    """Estimate avalanche danger per day using European 1-5 scale.

    This is an automated heuristic estimate. Always consult local avalanche bulletins.
    """
    DISCLAIMER = "Automated estimate only. Always consult local avalanche bulletins."
    results = []

    for i, day in enumerate(scores[:7]):
        cond = day.get("conditions", {})
        date = day.get("date", f"Day {i+1}")

        snow_24h = cond.get("snowfall_24h_cm", 0) or 0
        wind = cond.get("wind_speed_kmh", 0) or 0
        freezing_level = cond.get("freezing_level_m", 0) or 0
        snow_depth = cond.get("snow_depth_m", 0) or 0
        rain_mm = cond.get("rain_mm", 0) or 0

        # --- New snow 24h score (linear interpolation) ---
        if snow_24h <= 0:
            new_snow_24h_pts = 0
        elif snow_24h >= 30:
            new_snow_24h_pts = 30
        else:
            # Linear: 5cm=5, 10=10, 15=15, 20=20, 30=30
            new_snow_24h_pts = snow_24h

        # --- New snow 72h score ---
        snow_72h = snow_24h
        for j in range(1, 3):
            if i - j >= 0:
                snow_72h += (scores[i - j].get("conditions", {}).get("snowfall_24h_cm", 0) or 0)

        if snow_72h > 50:
            new_snow_72h_pts = 15
        elif snow_72h > 30:
            new_snow_72h_pts = 10
        elif snow_72h > 15:
            new_snow_72h_pts = 5
        else:
            new_snow_72h_pts = 0

        # --- Wind transport score ---
        # Recent snow for wind transport check (48h)
        recent_snow_48h = snow_24h
        if i - 1 >= 0:
            recent_snow_48h += (scores[i - 1].get("conditions", {}).get("snowfall_24h_cm", 0) or 0)

        wind_transport_pts = 0
        if wind > 35:
            wind_transport_pts = 15  # Strong wind even without snow
        if wind > 25 and recent_snow_48h > 5:
            wind_transport_pts = 20  # Wind + snow = significant transport

        # --- Rapid warming score ---
        rapid_warming_pts = 0
        if i > 0:
            prev_fl = scores[i - 1].get("conditions", {}).get("freezing_level_m", 0) or 0
            if prev_fl > 0 and freezing_level > 0:
                fl_rise = freezing_level - prev_fl
                if fl_rise > 500:
                    rapid_warming_pts = 15

        # --- Rain on snow score ---
        rain_on_snow_pts = 0
        if rain_mm > 0 and snow_depth > 0.1:
            rain_on_snow_pts = 20
            if rain_mm > 5:
                rain_on_snow_pts += 15

        # Total and level mapping
        total = min(100, new_snow_24h_pts + new_snow_72h_pts + wind_transport_pts +
                    rapid_warming_pts + rain_on_snow_pts)

        if total >= 81:
            level, level_name = 5, "EXTREME"
        elif total >= 61:
            level, level_name = 4, "HIGH"
        elif total >= 41:
            level, level_name = 3, "CONSIDERABLE"
        elif total >= 21:
            level, level_name = 2, "MODERATE"
        else:
            level, level_name = 1, "LOW"

        results.append({
            "date": date,
            "level": level,
            "level_name": level_name,
            "total_score": round(total),
            "factors": {
                "new_snow_24h": round(new_snow_24h_pts),
                "new_snow_72h": round(new_snow_72h_pts),
                "wind_transport": round(wind_transport_pts),
                "rapid_warming": round(rapid_warming_pts),
                "rain_on_snow": round(rain_on_snow_pts),
            },
            "disclaimer": DISCLAIMER,
        })

    return results


def build_multi_chart_data(scores: list, models: list = None) -> dict:
    """Build Chart.js-compatible data for temperature, wind, freezing level, and snow depth charts.

    Returns dict with keys: temperature_chart, wind_chart, freezing_level_chart, snow_depth_chart.
    """
    days = scores[:7]
    labels = [_format_chart_date(d.get("date", "")) for d in days]

    # Temperature chart
    temp_avg = []
    temp_max = []
    temp_min = []
    for d in days:
        cond = d.get("conditions", {})
        t = cond.get("temperature_c")
        t_min = cond.get("temperature_min_c")
        t_max = cond.get("temperature_max_c")

        temp_avg.append(round(t, 1) if t is not None else None)
        # Approximate min/max if not provided
        if t_max is not None:
            temp_max.append(round(t_max, 1))
        elif t is not None:
            temp_max.append(round(t + 3, 1))
        else:
            temp_max.append(None)

        if t_min is not None:
            temp_min.append(round(t_min, 1))
        elif t is not None:
            temp_min.append(round(t - 3, 1))
        else:
            temp_min.append(None)

    temperature_chart = {
        "labels": labels,
        "datasets": [
            {"label": "Max", "data": temp_max},
            {"label": "Min", "data": temp_min},
            {"label": "Avg", "data": temp_avg},
        ],
    }

    # Wind chart
    wind_sustained = []
    wind_gusts = []
    for d in days:
        cond = d.get("conditions", {})
        ws = cond.get("wind_speed_kmh")
        wg = cond.get("wind_gust_kmh")
        wind_sustained.append(round(ws, 1) if ws is not None else None)
        wind_gusts.append(round(wg, 1) if wg is not None else None)

    wind_chart = {
        "labels": labels,
        "datasets": [
            {"label": "Sustained", "data": wind_sustained},
            {"label": "Gusts", "data": wind_gusts},
        ],
    }

    # Freezing level chart
    fl_data = []
    for d in days:
        fl = d.get("conditions", {}).get("freezing_level_m")
        fl_data.append(round(fl) if fl is not None else None)

    freezing_level_chart = {
        "labels": labels,
        "datasets": [
            {"label": "Freezing Level", "data": fl_data},
        ],
        "reference_lines": [
            {"value": 1900, "label": "Mid-mountain (1900m)"},
            {"value": 2400, "label": "Summit (2400m)"},
        ],
    }

    # Snow depth chart (meters -> cm)
    depth_data = []
    for d in days:
        sd = d.get("conditions", {}).get("snow_depth_m")
        if sd is not None:
            depth_data.append(round(sd * 100, 1))
        else:
            depth_data.append(None)

    snow_depth_chart = {
        "labels": labels,
        "datasets": [
            {"label": "Snow Depth (cm)", "data": depth_data},
        ],
    }

    return {
        "temperature_chart": temperature_chart,
        "wind_chart": wind_chart,
        "freezing_level_chart": freezing_level_chart,
        "snow_depth_chart": snow_depth_chart,
    }


def build_model_spread(model_comparison: list) -> list:
    """Compute min/max snowfall spread across models for each day.

    Returns list of dicts with date, min_cm, max_cm, spread_cm.
    """
    results = []

    for row in model_comparison[:7]:
        date = row.get("date", "")
        values = row.get("snowfall_values", [])
        # Filter out None/missing values
        valid = [v for v in values if v is not None]

        if valid:
            min_v = min(valid)
            max_v = max(valid)
            results.append({
                "date": date,
                "min_cm": round(min_v, 1),
                "max_cm": round(max_v, 1),
                "spread_cm": round(max_v - min_v, 1),
            })
        else:
            results.append({
                "date": date,
                "min_cm": 0,
                "max_cm": 0,
                "spread_cm": 0,
            })

    return results


def build_spaghetti_data(model_comparison: list, models: list) -> dict:
    """Build per-model snowfall traces for spaghetti plot visualization.

    Each model gets its own line, showing 16-day snowfall predictions.
    This makes model agreement/disagreement visually obvious.

    Args:
        model_comparison: List of daily rows with snowfall_values.
        models: List of model names.

    Returns:
        Dict with 'labels' (dates) and 'traces' (list of {model, values, color}).
    """
    AI_NAME_MAP = {
        "ecmwf_aifs025": "AIFS",
        "ecmwf_aifs025_single": "AIFS",
        "graphcast025": "GraphCast",
        "gfs_graphcast025": "GraphCast",
    }
    colors = [
        "rgba(77,141,247,0.8)",    # ICON - blue
        "rgba(60,192,104,0.8)",    # ECMWF - green
        "rgba(240,194,48,0.8)",    # GFS - yellow
        "rgba(187,134,252,0.8)",   # ARPEGE - purple
        "rgba(255,138,101,0.8)",   # UKMO - orange
        "rgba(0,188,212,0.8)",     # AIFS - cyan
        "rgba(233,30,99,0.8)",     # GraphCast - pink
    ]

    labels = [_format_chart_date(row["date"]) for row in model_comparison[:16]]

    traces = []
    for j, model in enumerate(models):
        name = AI_NAME_MAP.get(model, model.replace("_seamless", "").replace("_ifs025", "").upper())
        values = []
        for row in model_comparison[:16]:
            sv = row.get("snowfall_values", [])
            values.append(sv[j] if j < len(sv) and sv[j] is not None else None)
        traces.append({
            "model": name,
            "values": values,
            "color": colors[j % len(colors)],
        })

    return {"labels": labels, "traces": traces}


def build_probability_fan_data(ensemble_data: dict) -> dict:
    """Build probability fan chart data from ensemble percentiles.

    Shows uncertainty bands widening with lead time:
    - p50 as solid line
    - p25-p75 as dark band
    - p10-p90 as light band

    Args:
        ensemble_data: Dict with daily.dates, daily.snowfall.{p10,p25,p50,p75,p90}.

    Returns:
        Dict with 'labels' (dates), 'p50', 'p25', 'p75', 'p10', 'p90' arrays.
    """
    if not ensemble_data or ensemble_data.get("error"):
        return {}

    daily = ensemble_data.get("daily", {})
    dates = daily.get("dates", [])
    snowfall = daily.get("snowfall", {})

    if not dates:
        return {}

    labels = [_format_chart_date(d) for d in dates[:16]]

    return {
        "labels": labels,
        "p10": snowfall.get("p10", [])[:16],
        "p25": snowfall.get("p25", [])[:16],
        "p50": snowfall.get("p50", [])[:16],
        "p75": snowfall.get("p75", [])[:16],
        "p90": snowfall.get("p90", [])[:16],
    }


def build_season_stats(history_dir: str) -> dict:
    """Build season statistics from historical forecast files.

    Scans all history files to compute: total powder days, total snowfall,
    biggest single day, longest dry spell, current streak.

    Args:
        history_dir: Path to docs/history/ directory.

    Returns:
        Dict with season-level statistics.
    """
    import os
    import json

    stats = {
        "total_powder_days": 0,
        "total_snowfall_cm": 0,
        "biggest_day_cm": 0,
        "biggest_day_date": "",
        "longest_dry_spell": 0,
        "current_dry_spell": 0,
        "season_days_tracked": 0,
    }

    if not os.path.isdir(history_dir):
        return stats

    # Read the most recent forecast file for each unique date
    files = sorted([f for f in os.listdir(history_dir) if f.startswith("forecast_")])
    if not files:
        return stats

    # Use the latest file to get the most up-to-date view
    latest_file = files[-1]
    try:
        with open(os.path.join(history_dir, latest_file)) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return stats

    scores = data.get("scores", [])
    if not scores:
        return stats

    dry_spell = 0
    max_dry = 0

    for s in scores:
        cond = s.get("conditions", {})
        snow = cond.get("snowfall_24h_cm", 0)
        score = s.get("total", 0)
        date = s.get("date", "")

        stats["total_snowfall_cm"] += snow
        stats["season_days_tracked"] += 1

        if score >= 60:
            stats["total_powder_days"] += 1

        if snow > stats["biggest_day_cm"]:
            stats["biggest_day_cm"] = round(snow, 1)
            stats["biggest_day_date"] = date

        if snow < 1:
            dry_spell += 1
            max_dry = max(max_dry, dry_spell)
        else:
            dry_spell = 0

    stats["longest_dry_spell"] = max_dry
    stats["current_dry_spell"] = dry_spell
    stats["total_snowfall_cm"] = round(stats["total_snowfall_cm"], 1)

    return stats
