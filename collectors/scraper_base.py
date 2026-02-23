"""Shared HTML table parser for Snow-Forecast.com and Mountain-Forecast.com.

Both sites use near-identical table structures with `data-row` attributes:
  data-row="days", "time", "snow", "rain", "temperature-max", "temperature-min",
  "temperature-chill", "wind", "humidity", "freezing-level", "cloud-base",
  "phrases", "sunrise", "sunset"
"""

import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup


def parse_forecast_table(html: str) -> dict:
    """Parse the shared forecast table format used by snow-forecast.com
    and mountain-forecast.com.

    Returns:
        {
            "periods": [
                {
                    "date": "2026-02-19",
                    "time_of_day": "AM" | "PM" | "night",
                    "snow_cm": float or None,
                    "rain_mm": float or None,
                    "temp_max_c": float or None,
                    "temp_min_c": float or None,
                    "wind_chill_c": float or None,
                    "wind_speed_kmh": float or None,
                    "wind_direction": str or None,
                    "humidity_pct": float or None,
                    "freezing_level_m": float or None,
                    "cloud_base_m": float or None,
                    "phrase": str,
                }
            ],
            "daily": [
                {
                    "date": "2026-02-19",
                    "snow_total_cm": float,
                    "rain_total_mm": float,
                    "temp_max_c": float,
                    "temp_min_c": float,
                    "wind_max_kmh": float,
                    "freezing_level_avg_m": float,
                    "humidity_avg_pct": float,
                }
            ]
        }
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the forecast table
    table = soup.find("table", class_=lambda x: x and "forecast" in str(x).lower())
    if not table:
        logger.warning("No forecast table found in HTML")
        return {"periods": [], "daily": []}

    rows = {}
    for tr in table.find_all("tr", class_="forecast-table__row"):
        data_row = tr.get("data-row", "")
        if data_row:
            cells = tr.find_all(["td", "th"])
            # Skip first cell (label) and extract values
            values = []
            for cell in cells[1:]:
                values.append(cell.get_text(strip=True))
            rows[data_row] = values

    if not rows:
        logger.warning("No data-row elements found in forecast table")
        return {"periods": [], "daily": []}

    # Parse dates from "days" row
    # Format: "Thursday19", "Friday20", etc.
    raw_days = rows.get("days", [])
    raw_times = rows.get("time", [])

    # Determine dates: days row has day names with dates
    # Each day has 3 periods (AM, PM, night)
    today = datetime.utcnow().date()
    current_year = today.year

    dates_for_periods = []
    current_date = None

    for i, time_label in enumerate(raw_times):
        time_label = time_label.strip()
        if not time_label:
            continue

        # Check if there's a corresponding day header
        if i < len(raw_days):
            day_text = raw_days[i].strip()
            if day_text:
                # Extract day number: "Thursday19" -> 19
                match = re.search(r"(\d+)", day_text)
                if match:
                    day_num = int(match.group(1))
                    # Figure out the month — use today as reference
                    # If day_num < today.day and we're near month end, it's next month
                    candidate = today.replace(day=day_num)
                    if candidate < today - timedelta(days=1):
                        # Must be next month
                        if today.month == 12:
                            candidate = candidate.replace(year=today.year + 1, month=1)
                        else:
                            candidate = candidate.replace(month=today.month + 1)
                    current_date = candidate

        if current_date:
            dates_for_periods.append({
                "date": current_date.isoformat(),
                "time_of_day": time_label,
            })

    # Now extract data for each period
    n_periods = len(dates_for_periods)

    def get_row_values(key, count):
        """Get values from a row, padded to count length."""
        raw = rows.get(key, [])
        result = []
        for v in raw[:count]:
            result.append(v)
        while len(result) < count:
            result.append("")
        return result

    def parse_num(val):
        """Parse a numeric value, returning None for dashes/empty."""
        if not val or val in ("—", "-", "–", ""):
            return None
        val = re.sub(r"[^\d.\-]", "", val)
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def parse_wind(val):
        """Parse wind value like '20WSW' -> (20, 'WSW')."""
        if not val or val in ("—", "-", "–", ""):
            return None, None
        match = re.match(r"(\d+)\s*([A-Z]+)?", val)
        if match:
            speed = float(match.group(1))
            direction = match.group(2) or ""
            return speed, direction
        return None, None

    snow_vals = get_row_values("snow", n_periods)
    rain_vals = get_row_values("rain", n_periods)
    tmax_vals = get_row_values("temperature-max", n_periods)
    tmin_vals = get_row_values("temperature-min", n_periods)
    chill_vals = get_row_values("temperature-chill", n_periods)
    wind_vals = get_row_values("wind", n_periods)
    humid_vals = get_row_values("humidity", n_periods)
    freeze_vals = get_row_values("freezing-level", n_periods)
    cloud_base_vals = get_row_values("cloud-base", n_periods)
    phrase_vals = get_row_values("phrases", n_periods)

    periods = []
    for i in range(n_periods):
        wind_speed, wind_dir = parse_wind(wind_vals[i])
        periods.append({
            "date": dates_for_periods[i]["date"],
            "time_of_day": dates_for_periods[i]["time_of_day"],
            "snow_cm": parse_num(snow_vals[i]),
            "rain_mm": parse_num(rain_vals[i]),
            "temp_max_c": parse_num(tmax_vals[i]),
            "temp_min_c": parse_num(tmin_vals[i]),
            "wind_chill_c": parse_num(chill_vals[i]),
            "wind_speed_kmh": wind_speed,
            "wind_direction": wind_dir,
            "humidity_pct": parse_num(humid_vals[i]),
            "freezing_level_m": parse_num(freeze_vals[i]),
            "cloud_base_m": parse_num(cloud_base_vals[i]),
            "phrase": phrase_vals[i] if i < len(phrase_vals) else "",
        })

    # Aggregate to daily
    daily_map = {}
    for p in periods:
        d = p["date"]
        if d not in daily_map:
            daily_map[d] = []
        daily_map[d].append(p)

    daily = []
    for date_str in sorted(daily_map.keys()):
        day_periods = daily_map[date_str]

        snow_total = sum(p["snow_cm"] or 0 for p in day_periods)
        rain_total = sum(p["rain_mm"] or 0 for p in day_periods)

        temps_max = [p["temp_max_c"] for p in day_periods if p["temp_max_c"] is not None]
        temps_min = [p["temp_min_c"] for p in day_periods if p["temp_min_c"] is not None]
        winds = [p["wind_speed_kmh"] for p in day_periods if p["wind_speed_kmh"] is not None]
        freezes = [p["freezing_level_m"] for p in day_periods if p["freezing_level_m"] is not None]
        humids = [p["humidity_pct"] for p in day_periods if p["humidity_pct"] is not None]

        daily.append({
            "date": date_str,
            "snow_total_cm": snow_total,
            "rain_total_mm": rain_total,
            "temp_max_c": max(temps_max) if temps_max else None,
            "temp_min_c": min(temps_min) if temps_min else None,
            "wind_max_kmh": max(winds) if winds else None,
            "freezing_level_avg_m": sum(freezes) / len(freezes) if freezes else None,
            "humidity_avg_pct": sum(humids) / len(humids) if humids else None,
        })

    return {"periods": periods, "daily": daily}
