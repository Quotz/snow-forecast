"""Open-Meteo Ensemble API collector for probabilistic snowfall forecasts."""

import requests
from datetime import datetime
from .base import BaseCollector

BASE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

ENSEMBLE_MODELS = ["icon_eu_eps", "ecmwf_ifs025", "gfs025"]
MEMBER_COUNTS = {"icon_eu_eps": 40, "ecmwf_ifs025": 51, "gfs025": 31}

HOURLY_PARAMS = ["snowfall", "precipitation", "temperature_2m"]


def _percentile(sorted_vals, pct):
    """Compute percentile from a sorted list using nearest-rank method."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    d = k - f
    return sorted_vals[f] * (1 - d) + sorted_vals[c] * d


def _compute_percentiles(values):
    """Compute p10, p25, p50, p75, p90 from a list of values."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return {p: None for p in ("p10", "p25", "p50", "p75", "p90")}
    s = sorted(cleaned)
    return {
        "p10": _percentile(s, 10),
        "p25": _percentile(s, 25),
        "p50": _percentile(s, 50),
        "p75": _percentile(s, 75),
        "p90": _percentile(s, 90),
    }


class OpenMeteoEnsembleCollector(BaseCollector):
    """Fetches ensemble forecast data from Open-Meteo Ensemble API."""

    def __init__(self, config: dict):
        super().__init__("open_meteo_ensemble", config)
        self.forecast_days = config.get("forecast", {}).get("days", 7)
        # Ensemble API has limited range; cap at 7 days
        if self.forecast_days > 7:
            self.forecast_days = 7

    def _fetch_elevation(self, lat: float, lon: float, elevation: int,
                         timezone: str) -> dict:
        """Fetch ensemble data for a single elevation."""
        hourly_str = ",".join(HOURLY_PARAMS)
        models_str = ",".join(ENSEMBLE_MODELS)

        params = {
            "latitude": lat,
            "longitude": lon,
            "elevation": elevation,
            "hourly": hourly_str,
            "models": models_str,
            "forecast_days": self.forecast_days,
            "timezone": timezone,
        }

        self.logger.debug(f"Fetching Open-Meteo Ensemble at {elevation}m")
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise ValueError(
                f"Open-Meteo Ensemble API error: {data.get('reason', data['error'])}"
            )

        return self._parse_response(data, elevation)

    def _parse_response(self, data: dict, elevation: int) -> dict:
        """Parse ensemble response into daily percentiles and probabilities."""
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        # Figure out how many days we have
        n_hours = len(times)
        n_days = n_hours // 24 if n_hours else 0

        # Build dates list from hourly times
        dates = []
        for d in range(n_days):
            # Take date from the first hour of each day
            if d * 24 < len(times):
                dates.append(times[d * 24][:10])

        # Collect all member data per model per param
        # Ensemble API returns keys like "snowfall_member01_icon_eu_eps"
        # or "snowfall_icon_eu_eps_member01" depending on version
        # We detect the pattern from available keys
        all_member_daily = {param: [] for param in HOURLY_PARAMS}

        for model in ENSEMBLE_MODELS:
            n_members = MEMBER_COUNTS.get(model, 30)
            for member_idx in range(n_members):
                member_str = f"member{member_idx:02d}"
                for param in HOURLY_PARAMS:
                    # Try common key patterns
                    key = None
                    for pattern in [
                        f"{param}_{model}_{member_str}",
                        f"{param}_{member_str}_{model}",
                    ]:
                        if pattern in hourly:
                            key = pattern
                            break

                    if key is None:
                        continue

                    member_hourly = hourly[key]

                    # Aggregate hourly to daily
                    for d in range(n_days):
                        start = d * 24
                        end = min(start + 24, len(member_hourly))
                        day_vals = [
                            v for v in member_hourly[start:end]
                            if v is not None
                        ]
                        if not day_vals:
                            continue

                        if param == "temperature_2m":
                            daily_val = sum(day_vals) / len(day_vals)
                        else:
                            daily_val = sum(day_vals)

                        # Ensure we have enough slots
                        while len(all_member_daily[param]) <= d:
                            all_member_daily[param].append([])
                        all_member_daily[param][d].append(daily_val)

        # Compute percentiles and probabilities per day
        result_daily = {"dates": dates}
        for param in HOURLY_PARAMS:
            param_result = {
                "p10": [], "p25": [], "p50": [], "p75": [], "p90": [],
            }
            if param in ("snowfall", "precipitation"):
                param_result["prob_5cm"] = []
                param_result["prob_15cm"] = []

            for d in range(n_days):
                if d < len(all_member_daily[param]):
                    day_members = all_member_daily[param][d]
                else:
                    day_members = []

                pcts = _compute_percentiles(day_members)
                for p_key in ("p10", "p25", "p50", "p75", "p90"):
                    param_result[p_key].append(pcts[p_key])

                if param in ("snowfall", "precipitation"):
                    if day_members:
                        n_total = len(day_members)
                        prob_5 = sum(1 for v in day_members if v >= 5) / n_total
                        prob_15 = sum(1 for v in day_members if v >= 15) / n_total
                    else:
                        prob_5 = None
                        prob_15 = None
                    param_result["prob_5cm"].append(prob_5)
                    param_result["prob_15cm"].append(prob_15)

            result_daily[param] = param_result

        return {
            "elevation": elevation,
            "daily": result_daily,
            "n_days": n_days,
        }

    def fetch(self, location: dict) -> dict:
        """Fetch ensemble data for summit elevation only."""
        lat = location["lat"]
        lon = location["lon"]
        tz = location.get("timezone", "UTC")
        elevations = location.get("elevations", {})

        # Use summit elevation only for ensemble
        summit_elev = elevations.get("summit", max(elevations.values()))

        result = {
            "source": "open_meteo_ensemble",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "elevations": {},
            "daily": {},
            "models": list(ENSEMBLE_MODELS),
            "member_counts": dict(MEMBER_COUNTS),
            "error": None,
        }

        self.logger.info(
            f"Fetching Open-Meteo Ensemble for {location['name']} at {summit_elev}m"
        )
        elev_data = self._fetch_elevation(lat, lon, summit_elev, tz)
        result["elevations"][summit_elev] = elev_data
        result["daily"] = elev_data["daily"]

        return result
