"""Meteoblue API collector — terrain-aware alpine downscaling."""

import os
import requests
from datetime import datetime
from .base import BaseCollector

BASE_URL = "https://my.meteoblue.com/packages/basic-day"

# Standard atmospheric lapse rate: 6.5°C per 1000m
LAPSE_RATE_C_PER_M = 6.5 / 1000


def _estimate_slr(temp_c: float) -> float:
    """Estimate snow-to-liquid ratio from temperature.

    Colder temperatures produce lighter, fluffier snow with higher SLR.
    Uses a simple piecewise approximation:
        <= -15°C  -> 20:1 (very dry cold)
        -15 to -10 -> 15:1 (cold/dry)
        -10 to -5  -> 12:1 (typical)
        -5 to -1   -> 8:1  (damp)
        > -1°C     -> 5:1  (wet/heavy)
    """
    if temp_c <= -15:
        return 20.0
    elif temp_c <= -10:
        return 15.0
    elif temp_c <= -5:
        return 12.0
    elif temp_c <= -1:
        return 8.0
    else:
        return 5.0


def _estimate_freezing_level(temp_min: float, temp_max: float,
                             elevation: int) -> float:
    """Estimate freezing level from surface temps using standard lapse rate.

    Uses mean temperature at the station elevation and solves for the altitude
    where temperature reaches 0°C.
    """
    temp_mean = (temp_min + temp_max) / 2.0
    # Positive temp_mean → freezing level above station; negative → below
    return elevation + (temp_mean / LAPSE_RATE_C_PER_M)


class MeteoblueCollector(BaseCollector):
    """Fetches terrain-aware forecast data from Meteoblue basic-day package."""

    def __init__(self, config: dict):
        super().__init__("meteoblue", config)
        self.api_key = os.environ.get("METEOBLUE_API_KEY")
        # Check enabled in config
        scrapers_cfg = config.get("scrapers", {})
        mb_cfg = scrapers_cfg.get("meteoblue", {})
        self.enabled = mb_cfg.get("enabled", False)

    def _fetch_data(self, lat: float, lon: float, elevation: int) -> dict:
        """Fetch basic-day package from Meteoblue API."""
        if not self.api_key:
            raise ValueError("METEOBLUE_API_KEY environment variable not set")

        params = {
            "lat": lat,
            "lon": lon,
            "asl": elevation,
            "apikey": self.api_key,
            "format": "json",
            "timeformat": "timestamp_unix",
        }

        self.logger.debug(f"Fetching Meteoblue at {elevation}m")
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _parse_response(self, data: dict, elevation: int) -> list:
        """Parse Meteoblue basic-day response into standard daily list."""
        data_day = data.get("data_day", {})

        times = data_day.get("time", [])
        temp_maxs = data_day.get("temperature_max", [])
        temp_mins = data_day.get("temperature_min", [])
        precips = data_day.get("precipitation", [])
        snow_fracs = data_day.get("snowfraction", [])
        wind_maxs = data_day.get("windspeed_max", [])
        gust_maxs = data_day.get("windgust_max", [])
        wind_dirs = data_day.get("winddirection_dominant", [])

        daily = []
        for i, time_val in enumerate(times):
            # Parse date — Meteoblue with timestamp_unix returns epoch strings
            if isinstance(time_val, (int, float)):
                date_str = datetime.utcfromtimestamp(time_val).strftime("%Y-%m-%d")
            else:
                date_str = str(time_val)[:10]

            temp_min = temp_mins[i] if i < len(temp_mins) else None
            temp_max = temp_maxs[i] if i < len(temp_maxs) else None
            precip_mm = precips[i] if i < len(precips) else 0.0
            snow_frac = snow_fracs[i] if i < len(snow_fracs) else 0.0
            wind_max = wind_maxs[i] if i < len(wind_maxs) else None
            gust_max = gust_maxs[i] if i < len(gust_maxs) else None
            wind_dir = wind_dirs[i] if i < len(wind_dirs) else None

            # Calculate snow in cm:
            # precip_mm * snowfraction gives snow water equivalent in mm
            # Multiply by SLR to get snow depth, divide by 10 to convert mm->cm
            mean_temp = ((temp_min or 0) + (temp_max or 0)) / 2.0
            slr = _estimate_slr(mean_temp)
            snow_cm = (precip_mm * (snow_frac or 0) * slr) / 10.0

            # Estimate freezing level
            freezing_level = None
            if temp_min is not None and temp_max is not None:
                freezing_level = round(
                    _estimate_freezing_level(temp_min, temp_max, elevation)
                )

            daily.append({
                "date": date_str,
                "snow_total_cm": round(snow_cm, 1),
                "temp_min_c": temp_min,
                "temp_max_c": temp_max,
                "wind_max_kmh": wind_max,
                "wind_gust_kmh": gust_max,
                "freezing_level_avg_m": freezing_level,
                "wind_direction_deg": wind_dir,
            })

        return daily

    def fetch(self, location: dict) -> dict:
        """Fetch Meteoblue data for summit elevation only."""
        if not self.enabled:
            raise ValueError("Meteoblue collector is not enabled in config")

        lat = location["lat"]
        lon = location["lon"]
        elevations = location.get("elevations", {})

        # Use summit elevation only (like ensemble collector)
        summit_elev = elevations.get("summit", max(elevations.values()))

        self.logger.info(
            f"Fetching Meteoblue for {location['name']} at {summit_elev}m"
        )

        raw_data = self._fetch_data(lat, lon, summit_elev)
        daily = self._parse_response(raw_data, summit_elev)

        return {
            "source": "meteoblue",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "elevations": {summit_elev: {"elevation": summit_elev, "daily": daily}},
            "daily": daily,
            "models": ["meteoblue"],
            "error": None,
        }
