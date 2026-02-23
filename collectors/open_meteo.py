"""Open-Meteo multi-model weather data collector."""

import requests
from datetime import datetime
from .base import BaseCollector

# Hourly parameters we want from Open-Meteo
HOURLY_PARAMS = [
    "snowfall",
    "snowfall_water_equivalent",
    "snow_depth",
    "temperature_2m",
    "apparent_temperature",
    "dew_point_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "freezing_level_height",
    "visibility",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_high",
    "direct_radiation",
    "sunshine_duration",
    "surface_pressure",
]

DAILY_PARAMS = [
    "snowfall_sum",
    "snow_depth_max",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
    "precipitation_probability_max",
    "sunrise",
    "sunset",
    "sunshine_duration",
    "uv_index_max",
    "weather_code",
]

BASE_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoCollector(BaseCollector):
    """Fetches multi-model forecast data from Open-Meteo."""

    def __init__(self, config: dict):
        super().__init__("open_meteo", config)
        self.models = config.get("models", ["icon_seamless", "ecmwf_ifs025", "gfs_seamless"])
        self.forecast_days = config.get("forecast", {}).get("days", 16)

    def _fetch_elevation(self, lat: float, lon: float, elevation: int,
                         timezone: str) -> dict:
        """Fetch multi-model data for a single elevation."""

        # Build hourly param list with model suffixes
        # When requesting multiple models, Open-Meteo appends model name to each param
        hourly_str = ",".join(HOURLY_PARAMS)
        daily_str = ",".join(DAILY_PARAMS)
        models_str = ",".join(self.models)

        params = {
            "latitude": lat,
            "longitude": lon,
            "elevation": elevation,
            "hourly": hourly_str,
            "daily": daily_str,
            "models": models_str,
            "forecast_days": self.forecast_days,
            "timezone": timezone,
        }

        self.logger.debug(f"Fetching Open-Meteo at {elevation}m: {params}")
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise ValueError(f"Open-Meteo API error: {data.get('reason', data['error'])}")

        return self._parse_response(data, elevation)

    def _parse_response(self, data: dict, elevation: int) -> dict:
        """Parse Open-Meteo response into normalized structure."""

        hourly = data.get("hourly", {})
        daily = data.get("daily", {})
        times = hourly.get("time", [])
        daily_times = daily.get("time", [])

        result = {
            "elevation": elevation,
            "api_elevation": data.get("elevation"),
            "models": {},
            "daily": {},
        }

        # Parse per-model hourly data
        # Open-Meteo returns keys like "snowfall_icon_seamless", "snowfall_ecmwf_ifs025"
        for model in self.models:
            model_data = {"hourly": {"time": times}}

            for param in HOURLY_PARAMS:
                # Try model-specific key first, then fallback to plain key
                model_key = f"{param}_{model}"
                if model_key in hourly:
                    model_data["hourly"][param] = hourly[model_key]
                elif param in hourly:
                    # Some params (like freezing_level_height) aren't per-model
                    model_data["hourly"][param] = hourly[param]

            result["models"][model] = model_data

        # Parse daily data (usually from the default/best model blend)
        daily_data = {"time": daily_times}
        for param in DAILY_PARAMS:
            # Daily params may also have model suffixes
            found = False
            for model in self.models:
                model_key = f"{param}_{model}"
                if model_key in daily:
                    daily_data[f"{param}_{model}"] = daily[model_key]
                    found = True
            # Also grab the plain version if available
            if param in daily:
                daily_data[param] = daily[param]

        result["daily"] = daily_data

        return result

    def fetch(self, location: dict) -> dict:
        """Fetch data for all elevations at a location."""

        lat = location["lat"]
        lon = location["lon"]
        tz = location.get("timezone", "UTC")
        elevations = location.get("elevations", {})

        result = {
            "source": "open_meteo",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "elevations": {},
            "daily": {},
            "models": self.models,
            "error": None,
        }

        for elev_name, elev_meters in elevations.items():
            self.logger.info(f"Fetching Open-Meteo for {location['name']} at {elev_meters}m ({elev_name})")
            elev_data = self._fetch_elevation(lat, lon, elev_meters, tz)
            result["elevations"][elev_meters] = elev_data

            # Use the first elevation's daily data as the primary daily view
            if not result["daily"]:
                result["daily"] = elev_data["daily"]

        return result
