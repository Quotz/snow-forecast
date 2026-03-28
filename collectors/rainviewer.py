"""RainViewer radar tile collector — free radar imagery and nowcast frames."""

from __future__ import annotations

import requests
from datetime import datetime
from .base import BaseCollector

API_URL = "https://api.rainviewer.com/public/weather-maps.json"

# Default tile parameters for the ski area
# z/x/y for zoom level 6, covering the Balkans region
DEFAULT_ZOOM = 6
DEFAULT_X = 35
DEFAULT_Y = 23

# Frames older than this many seconds are excluded (2 hours)
MAX_PAST_AGE_SECONDS = 2 * 60 * 60


class RainViewerCollector(BaseCollector):
    """Fetches radar frame index from RainViewer for animated radar display."""

    def __init__(self, config: dict):
        super().__init__("rainviewer", config)
        rainviewer_cfg = config.get("rainviewer", {})
        self.zoom = rainviewer_cfg.get("zoom", DEFAULT_ZOOM)
        self.tile_x = rainviewer_cfg.get("tile_x", DEFAULT_X)
        self.tile_y = rainviewer_cfg.get("tile_y", DEFAULT_Y)

    def _build_tile_url(self, host: str, path: str) -> str:
        """Build a tile URL from host and frame path.

        Format: {host}{path}/256/{z}/{x}/{y}/2/1_1.png
        """
        return f"{host}{path}/256/{self.zoom}/{self.tile_x}/{self.tile_y}/2/1_1.png"

    def _filter_past_frames(self, frames: list[dict], generated: int) -> list[dict]:
        """Keep only frames from the last 2 hours relative to generated timestamp."""
        cutoff = generated - MAX_PAST_AGE_SECONDS
        return [f for f in frames if f.get("time", 0) >= cutoff]

    def fetch(self, location: dict) -> dict:
        """Fetch radar map index and extract recent past + nowcast frames."""

        self.logger.info("Fetching RainViewer radar map index")

        resp = requests.get(API_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        host = data.get("host", "")
        generated = data.get("generated", 0)
        generated_iso = datetime.utcfromtimestamp(generated).isoformat() + "Z" if generated else None

        radar = data.get("radar", {})
        past_raw = radar.get("past", [])
        nowcast_raw = radar.get("nowcast", [])

        # Filter past frames to last 2 hours
        past_filtered = self._filter_past_frames(past_raw, generated)

        frames: list[dict] = []

        for frame in past_filtered:
            ts = frame.get("time", 0)
            path = frame.get("path", "")
            frames.append({
                "timestamp": ts,
                "url": self._build_tile_url(host, path),
                "type": "past",
            })

        for frame in nowcast_raw:
            ts = frame.get("time", 0)
            path = frame.get("path", "")
            frames.append({
                "timestamp": ts,
                "url": self._build_tile_url(host, path),
                "type": "nowcast",
            })

        return {
            "source": "rainviewer",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "location": location.get("name", "unknown"),
            "frames": frames,
            "generated_at": generated_iso,
            "past_frame_count": len(past_filtered),
            "nowcast_frame_count": len(nowcast_raw),
            "elevations": {},
            "daily": {},
            "models": [],
            "error": None,
        }
