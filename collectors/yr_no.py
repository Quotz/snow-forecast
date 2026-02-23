"""Yr.no (MET Norway) weather data collector."""

import requests
from datetime import datetime, timedelta
from .base import BaseCollector

BASE_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
USER_AGENT = "SnowMonitor/1.0 github.com/snow-monitor"


class YrNoCollector(BaseCollector):
    """Fetches forecast data from MET Norway (Yr.no) API."""

    def __init__(self, config: dict):
        super().__init__("yr_no", config)

    def _fetch_elevation(self, lat: float, lon: float, altitude: int) -> dict:
        """Fetch forecast for a single altitude."""

        headers = {"User-Agent": USER_AGENT}
        params = {"lat": lat, "lon": lon, "altitude": altitude}

        self.logger.debug(f"Fetching Yr.no at altitude={altitude}m")
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        return self._parse_response(data, altitude)

    def _parse_response(self, data: dict, altitude: int) -> dict:
        """Parse Yr.no response into normalized hourly structure."""

        timeseries = data.get("properties", {}).get("timeseries", [])

        hourly = {
            "time": [],
            "temperature": [],
            "wind_speed": [],        # m/s
            "wind_direction": [],    # degrees
            "humidity": [],          # %
            "dew_point": [],         # °C
            "cloud_cover": [],       # %
            "cloud_cover_high": [],
            "cloud_cover_mid": [],
            "cloud_cover_low": [],
            "fog": [],               # %
            "pressure": [],          # hPa
            "precipitation_1h": [],  # mm (next 1 hour)
            "precipitation_6h": [],  # mm (next 6 hours)
            "symbol_code_1h": [],
            "symbol_code_6h": [],
        }

        for ts in timeseries:
            time_str = ts["time"]
            instant = ts["data"]["instant"]["details"]

            hourly["time"].append(time_str)
            hourly["temperature"].append(instant.get("air_temperature"))
            hourly["wind_speed"].append(instant.get("wind_speed"))
            hourly["wind_direction"].append(instant.get("wind_from_direction"))
            hourly["humidity"].append(instant.get("relative_humidity"))
            hourly["dew_point"].append(instant.get("dew_point_temperature"))
            hourly["cloud_cover"].append(instant.get("cloud_area_fraction"))
            hourly["cloud_cover_high"].append(instant.get("cloud_area_fraction_high"))
            hourly["cloud_cover_mid"].append(instant.get("cloud_area_fraction_medium"))
            hourly["cloud_cover_low"].append(instant.get("cloud_area_fraction_low"))
            hourly["fog"].append(instant.get("fog_area_fraction"))
            hourly["pressure"].append(instant.get("air_pressure_at_sea_level"))

            # Precipitation from next_1_hours or next_6_hours blocks
            precip_1h = None
            precip_6h = None
            if "next_1_hours" in ts["data"]:
                precip_1h = ts["data"]["next_1_hours"]["details"].get("precipitation_amount")
            if "next_6_hours" in ts["data"]:
                precip_6h = ts["data"]["next_6_hours"]["details"].get("precipitation_amount")

            hourly["precipitation_1h"].append(precip_1h)
            hourly["precipitation_6h"].append(precip_6h)

            # Symbol codes for snowfall inference
            symbol_1h = None
            symbol_6h = None
            if "next_1_hours" in ts["data"]:
                symbol_1h = ts["data"]["next_1_hours"].get("summary", {}).get("symbol_code")
            if "next_6_hours" in ts["data"]:
                symbol_6h = ts["data"]["next_6_hours"].get("summary", {}).get("symbol_code")
            hourly["symbol_code_1h"].append(symbol_1h)
            hourly["symbol_code_6h"].append(symbol_6h)

        return {
            "altitude": altitude,
            "hourly": hourly,
            "timeseries_count": len(timeseries),
        }

    def fetch(self, location: dict) -> dict:
        """Fetch data for all elevations."""

        lat = location["lat"]
        lon = location["lon"]
        elevations = location.get("elevations", {})

        result = {
            "source": "yr_no",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "elevations": {},
            "daily": {},
            "models": ["met_norway"],
            "error": None,
        }

        for elev_name, elev_meters in elevations.items():
            self.logger.info(f"Fetching Yr.no for {location['name']} at {elev_meters}m ({elev_name})")
            elev_data = self._fetch_elevation(lat, lon, elev_meters)
            result["elevations"][elev_meters] = elev_data

        return result
