"""Open-Meteo Seasonal Forecast API collector — 46-day weekly anomaly outlook."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from .base import BaseCollector

BASE_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"

# Daily parameters available from the seasonal API
DAILY_PARAMS = [
    "temperature_2m_mean",
    "precipitation_sum",
]


class SeasonalCollector(BaseCollector):
    """Fetches 46-day seasonal outlook from Open-Meteo and aggregates into weekly anomalies."""

    def __init__(self, config: dict):
        super().__init__("seasonal", config)
        self.forecast_days = 46

    def _aggregate_weeks(self, daily: dict) -> list[dict]:
        """Aggregate daily data into weekly periods with anomalies.

        Returns a list of week dicts with period label, mean temp anomaly,
        and total precipitation.
        """
        times = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])

        if not times:
            return []

        # Split into ~7-day chunks
        weeks: list[dict] = []
        chunk_size = 7
        for i in range(0, len(times), chunk_size):
            chunk_times = times[i : i + chunk_size]
            chunk_temps = [t for t in temps[i : i + chunk_size] if t is not None]
            chunk_precip = [p for p in precip[i : i + chunk_size] if p is not None]

            if not chunk_times:
                continue

            week_num = (i // chunk_size) + 1
            total_weeks = -(-len(times) // chunk_size)  # ceiling division
            period = f"Week {week_num}" if week_num <= total_weeks else f"Week {week_num}"

            # Friendly period label using date range
            start_date = chunk_times[0]
            end_date = chunk_times[-1]
            period_label = f"Week {week_num} ({start_date} to {end_date})"

            week_data = {
                "period": period_label,
                "week_number": week_num,
                "start_date": start_date,
                "end_date": end_date,
                "temp_anomaly_c": round(sum(chunk_temps) / len(chunk_temps), 1) if chunk_temps else None,
                "precip_total_mm": round(sum(chunk_precip), 1) if chunk_precip else None,
                "days_in_period": len(chunk_times),
            }
            weeks.append(week_data)

        return weeks

    def fetch(self, location: dict) -> dict:
        """Fetch 46-day seasonal outlook and return weekly anomaly summary."""

        lat = location["lat"]
        lon = location["lon"]
        tz = location.get("timezone", "UTC")

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(DAILY_PARAMS),
            "forecast_days": self.forecast_days,
            "timezone": tz,
        }

        self.logger.info(f"Fetching seasonal outlook for {location.get('name', 'unknown')}")
        self.logger.debug(f"Seasonal API params: {params}")

        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise ValueError(f"Seasonal API error: {data.get('reason', data['error'])}")

        daily = data.get("daily", {})
        weeks = self._aggregate_weeks(daily)

        return {
            "source": "seasonal",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "weeks": weeks,
            "forecast_days": self.forecast_days,
            "elevations": {},
            "daily": daily,
            "models": [],
            "error": None,
        }
