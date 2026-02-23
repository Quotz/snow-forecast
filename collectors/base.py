"""Base collector class for weather data sources."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for all weather data collectors."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"collector.{name}")

    @abstractmethod
    def fetch(self, location: dict) -> dict:
        """Fetch forecast data for a location.

        Args:
            location: Dict with keys: name, lat, lon, timezone, elevations (dict)

        Returns:
            Dict with structure:
            {
                "source": str,
                "fetched_at": str (ISO),
                "location": str,
                "elevations": {
                    1900: { ... data ... },
                    2400: { ... data ... },
                },
                "daily": { ... daily aggregated data ... },
                "models": [str],  # model names if multi-model
                "error": str or None
            }
        """
        pass

    def safe_fetch(self, location: dict) -> dict:
        """Fetch with error handling — never crashes the pipeline."""
        try:
            result = self.fetch(location)
            self.logger.info(f"Successfully fetched from {self.name}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to fetch from {self.name}: {e}")
            return {
                "source": self.name,
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "location": location.get("name", "unknown"),
                "elevations": {},
                "daily": {},
                "models": [],
                "error": str(e),
            }
