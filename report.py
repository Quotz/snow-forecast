"""HTML dashboard report generator using Jinja2."""

import os
import json
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Score label to color mapping
LABEL_COLORS = {
    "EPIC": {"bg": "#1a73e8", "text": "#ffffff"},
    "GOOD": {"bg": "#34a853", "text": "#ffffff"},
    "FAIR": {"bg": "#fbbc04", "text": "#333333"},
    "MARGINAL": {"bg": "#9e9e9e", "text": "#ffffff"},
    "SKIP": {"bg": "#ea4335", "text": "#ffffff"},
}

SKY_LABELS = {
    "BLUEBIRD": "Bluebird",
    "MOSTLY_SUNNY": "Mostly Sunny",
    "PARTLY_CLOUDY": "Partly Cloudy",
    "MOSTLY_OVERCAST": "Mostly Overcast",
    "OVERCAST": "Overcast",
}

AVALANCHE_COLORS = {
    1: {"bg": "#4CAF50", "name": "LOW"},
    2: {"bg": "#FFEB3B", "name": "MODERATE"},
    3: {"bg": "#FF9800", "name": "CONSIDERABLE"},
    4: {"bg": "#F44336", "name": "HIGH"},
    5: {"bg": "#212121", "name": "EXTREME"},
}

VERDICT_COLORS = {
    "YES": {"bg": "#34a853", "text": "#ffffff", "icon": "✓"},
    "MAYBE": {"bg": "#fbbc04", "text": "#333333", "icon": "?"},
    "WAIT": {"bg": "#6b7280", "text": "#ffffff", "icon": "—"},
}

CRYSTAL_ICONS = {
    "powder": "❄️",
    "light": "❅",
    "mixed": "❆",
    "wet": "💧",
    "ice": "🧊",
    "crust": "🪨",
}


def generate_dashboard(report_data: dict, template_dir: str, output_path: str):
    """Generate the HTML dashboard from forecast data.

    Args:
        report_data: Full report dict containing scores, patterns, raw data, etc.
        template_dir: Path to Jinja2 templates directory
        output_path: Where to write the generated HTML
    """
    env = Environment(loader=FileSystemLoader(template_dir))
    env.globals["LABEL_COLORS"] = LABEL_COLORS
    env.globals["SKY_LABELS"] = SKY_LABELS
    env.globals["AVALANCHE_COLORS"] = AVALANCHE_COLORS
    env.globals["AI_NAME_MAP"] = {
        "ecmwf_aifs025": "AIFS",
        "ecmwf_aifs025_single": "AIFS",
        "graphcast025": "GraphCast",
        "gfs_graphcast025": "GraphCast",
    }
    env.globals["VERDICT_COLORS"] = VERDICT_COLORS
    env.globals["CRYSTAL_ICONS"] = CRYSTAL_ICONS
    env.globals["now"] = datetime.utcnow

    # Custom date filters: "2026-02-19" → "Wed 19/02/2026"
    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def fmt_date_full(iso_date: str) -> str:
        """Format ISO date as 'Wed 19/02/2026'."""
        try:
            d = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")
            return f"{DAY_NAMES[d.weekday()]} {d.strftime('%d/%m/%Y')}"
        except (ValueError, TypeError):
            return str(iso_date)

    def fmt_date_short(iso_date: str) -> str:
        """Format ISO date as 'Wed 19/02'."""
        try:
            d = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")
            return f"{DAY_NAMES[d.weekday()]} {d.strftime('%d/%m')}"
        except (ValueError, TypeError):
            return str(iso_date)

    def fmt_date_card(iso_date: str) -> str:
        """Format for score cards: 'Wed' on line 1, '19/02' on line 2."""
        try:
            d = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")
            return f"{DAY_NAMES[d.weekday()]}<br>{d.strftime('%d/%m')}"
        except (ValueError, TypeError):
            return str(iso_date)

    env.filters["datefull"] = fmt_date_full
    env.filters["dateshort"] = fmt_date_short
    env.filters["datecard"] = fmt_date_card

    template = env.get_template("dashboard.html")

    html = template.render(
        data=report_data,
        generated_at=datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC"),
        label_colors=LABEL_COLORS,
        sky_labels=SKY_LABELS,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard written to {output_path}")


def save_latest_json(report_data: dict, output_path: str):
    """Save the latest forecast data as JSON for OpenClaw/NanoClaw integration."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info(f"Latest JSON written to {output_path}")


def save_history(report_data: dict, history_dir: str):
    """Save a timestamped copy of the forecast data for trend tracking."""
    os.makedirs(history_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(history_dir, f"forecast_{ts}.json")
    with open(path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info(f"History saved to {path}")

    # Cleanup: keep only last 30 days of history (90 files at 3x/day)
    _cleanup_history(history_dir, max_files=90)


def _cleanup_history(history_dir: str, max_files: int = 90):
    """Remove old history files, keeping most recent max_files."""
    try:
        files = sorted(
            [f for f in os.listdir(history_dir) if f.startswith("forecast_")],
            reverse=True,
        )
        for old_file in files[max_files:]:
            os.remove(os.path.join(history_dir, old_file))
    except Exception as e:
        logger.warning(f"History cleanup failed: {e}")
