"""Analysis utilities — chart data, safety flags, source comparison."""

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
    colors = ["rgba(77,141,247,0.7)", "rgba(60,192,104,0.7)", "rgba(240,194,48,0.7)"]
    labels = [_format_chart_date(row["date"]) for row in model_comparison[:7]]

    datasets = []
    for j, model in enumerate(models):
        name = model.replace("_seamless", "").replace("_ifs025", "").upper()
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
