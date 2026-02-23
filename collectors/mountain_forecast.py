"""Mountain-Forecast.com scraper collector.

Fetches 6-day forecast for mountain huts/lodges.
Popova Shapka available at 1825m.
Bonus: provides cloud base height data not available from other sources.
"""

import requests
from datetime import datetime

from .base import BaseCollector
from .scraper_base import parse_forecast_table

# Mountain-Forecast.com location mapping
LOCATION_CONFIGS = {
    "popova_shapka": {
        "slug": "Popova-Shapka",
        "type": "huts-and-lodges",  # could also be "peaks"
        "elevation": 1825,
    },
}

BASE_URL = "https://www.mountain-forecast.com/{type}/{slug}/forecasts/{elevation}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


class MountainForecastCollector(BaseCollector):
    """Scrapes mountain-forecast.com for 6-day mountain forecasts.

    Unique data: cloud base height (m) — useful for visibility/flat light estimation.
    """

    def __init__(self, config: dict):
        super().__init__("mountain_forecast", config)
        self.timeout = config.get("scrapers", {}).get("timeout", 15)

    def fetch(self, location: dict) -> dict:
        """Fetch forecast from mountain-forecast.com."""

        loc_key = self.config.get("scrapers", {}).get("mountain_forecast", {}).get(
            "location_key", "popova_shapka"
        )
        loc_cfg = LOCATION_CONFIGS.get(loc_key, LOCATION_CONFIGS["popova_shapka"])

        url = BASE_URL.format(
            type=loc_cfg["type"],
            slug=loc_cfg["slug"],
            elevation=loc_cfg["elevation"],
        )

        self.logger.info(f"Fetching Mountain-Forecast.com ({loc_cfg['elevation']}m): {url}")

        result = {
            "source": self.name,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "elevations": {},
            "daily": {},
            "models": ["mountain_forecast"],
            "error": None,
        }

        try:
            resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            result["error"] = str(e)
            return result

        parsed = parse_forecast_table(resp.text)

        if not parsed["daily"]:
            result["error"] = "No daily data could be parsed from HTML"
            return result

        elevation = loc_cfg["elevation"]
        result["elevations"][elevation] = {
            "level": "mid",
            "periods": parsed["periods"],
            "daily": parsed["daily"],
        }
        result["daily"] = parsed["daily"]

        # Extract cloud base data for unique value-add
        cloud_base_data = []
        for period in parsed["periods"]:
            if period.get("cloud_base_m") is not None:
                cloud_base_data.append({
                    "date": period["date"],
                    "time_of_day": period["time_of_day"],
                    "cloud_base_m": period["cloud_base_m"],
                })

        if cloud_base_data:
            result["cloud_base"] = cloud_base_data

        return result
