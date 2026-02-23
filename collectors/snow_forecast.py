"""Snow-Forecast.com scraper collector.

Fetches 6-day forecasts at multiple elevations (bot/mid/top).
Popova Sapka elevations: 1670m (bot), 2035m (mid), 2400m (top).
"""

import requests
from datetime import datetime

from .base import BaseCollector
from .scraper_base import parse_forecast_table

# Snow-Forecast.com resort slug and elevation mapping
RESORT_CONFIGS = {
    "popova_shapka": {
        "slug": "PopovaShapka",
        "elevations": {
            "bot": 1670,
            "mid": 2035,
            "top": 2400,
        },
    },
}

BASE_URL = "https://www.snow-forecast.com/resorts/{slug}/6day/{level}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


class SnowForecastCollector(BaseCollector):
    """Scrapes snow-forecast.com for 6-day mountain forecasts."""

    def __init__(self, config: dict):
        super().__init__("snow_forecast", config)
        self.timeout = config.get("scrapers", {}).get("timeout", 15)
        self.levels = config.get("scrapers", {}).get("snow_forecast", {}).get(
            "levels", ["mid", "top"]
        )

    def fetch(self, location: dict) -> dict:
        """Fetch forecast from snow-forecast.com at configured elevation levels."""

        # Determine resort config
        resort_key = self.config.get("scrapers", {}).get("snow_forecast", {}).get(
            "resort_key", "popova_shapka"
        )
        resort = RESORT_CONFIGS.get(resort_key, RESORT_CONFIGS["popova_shapka"])
        slug = resort["slug"]

        result = {
            "source": self.name,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "elevations": {},
            "daily": {},
            "models": ["snow_forecast"],
            "error": None,
        }

        for level in self.levels:
            elevation = resort["elevations"].get(level)
            if not elevation:
                continue

            url = BASE_URL.format(slug=slug, level=level)
            self.logger.info(f"Fetching Snow-Forecast.com {level} ({elevation}m): {url}")

            try:
                resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as e:
                self.logger.warning(f"Failed to fetch {level}: {e}")
                continue

            parsed = parse_forecast_table(resp.text)

            if not parsed["daily"]:
                self.logger.warning(f"No daily data parsed for {level}")
                continue

            result["elevations"][elevation] = {
                "level": level,
                "periods": parsed["periods"],
                "daily": parsed["daily"],
            }

            # Use mid or top level daily data as the main daily summary
            if level in ("mid", "top") and not result["daily"]:
                result["daily"] = parsed["daily"]

        if not result["elevations"]:
            result["error"] = "No elevation data could be scraped"

        return result
