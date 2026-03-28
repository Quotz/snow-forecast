"""Crowd-sourced condition reports from Telegram subscribers.

Parses /report commands from Telegram users, stores structured condition
reports, and provides recent reports for dashboard display and verification.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Keyword patterns for parsing free-text condition reports
_SNOW_DEPTH_RE = re.compile(r'(\d+)\s*cm', re.IGNORECASE)
_QUALITY_KEYWORDS = {
    "powder": "powder",
    "champagne": "powder",
    "fluffy": "powder",
    "light": "light",
    "hero": "powder",
    "wet": "wet",
    "heavy": "wet",
    "cement": "wet",
    "ice": "ice",
    "icy": "ice",
    "hard": "ice",
    "crusty": "crust",
    "crust": "crust",
    "tracked": "tracked",
    "tracked out": "tracked",
    "moguls": "tracked",
}
_SKY_KEYWORDS = {
    "bluebird": "bluebird",
    "sunny": "sunny",
    "cloudy": "cloudy",
    "overcast": "overcast",
    "fog": "fog",
    "whiteout": "whiteout",
    "windy": "windy",
    "flat light": "flat_light",
}


def parse_report(text: str) -> dict:
    """Parse a free-text condition report into structured data.

    Examples:
        "/report powder 30cm bluebird" -> {depth_cm: 30, quality: "powder", sky: "bluebird"}
        "/report ice windy" -> {quality: "ice", sky: "windy"}
        "/report 15cm tracked out" -> {depth_cm: 15, quality: "tracked"}

    Args:
        text: Raw message text (may include /report prefix).

    Returns:
        Dict with parsed fields. Always includes 'raw_text'.
    """
    # Strip /report prefix
    clean = text.strip()
    if clean.lower().startswith("/report"):
        clean = clean[7:].strip()

    result = {
        "raw_text": clean,
        "depth_cm": None,
        "quality": None,
        "sky": None,
    }

    if not clean:
        return result

    text_lower = clean.lower()

    # Extract snow depth
    depth_match = _SNOW_DEPTH_RE.search(clean)
    if depth_match:
        result["depth_cm"] = int(depth_match.group(1))

    # Extract snow quality
    for keyword, quality in _QUALITY_KEYWORDS.items():
        if keyword in text_lower:
            result["quality"] = quality
            break

    # Extract sky condition
    for keyword, sky in _SKY_KEYWORDS.items():
        if keyword in text_lower:
            result["sky"] = sky
            break

    return result


def store_report(chat_id: str, report: dict, reports_path: str) -> None:
    """Store a parsed condition report.

    Args:
        chat_id: Telegram chat ID of the reporter.
        report: Parsed report dict from parse_report().
        reports_path: Path to crowd_reports.json file.
    """
    reports = _load_reports(reports_path)

    entry = {
        "chat_id": str(chat_id),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "report": report,
    }
    reports.append(entry)

    # Keep last 200 reports (rolling window)
    if len(reports) > 200:
        reports = reports[-200:]

    os.makedirs(os.path.dirname(reports_path) if os.path.dirname(reports_path) else ".", exist_ok=True)
    with open(reports_path, "w") as f:
        json.dump(reports, f, indent=1)

    logger.info("Stored crowd report from %s: %s", chat_id, report.get("raw_text", ""))


def get_recent_reports(reports_path: str, days: int = 7) -> list:
    """Get condition reports from the last N days.

    Args:
        reports_path: Path to crowd_reports.json.
        days: Number of days to look back.

    Returns:
        List of report entries sorted by timestamp (newest first).
    """
    reports = _load_reports(reports_path)
    if not reports:
        return []

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    recent = [r for r in reports if r.get("timestamp", "") >= cutoff]
    recent.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return recent


def _load_reports(reports_path: str) -> list:
    """Load reports from disk."""
    if not os.path.exists(reports_path):
        return []
    try:
        with open(reports_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
