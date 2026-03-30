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


def build_narrative(report_data: dict) -> str:
    """Build a weather narrative string from forecast data.

    Returns a short paragraph summarizing the key weather story:
    when snow arrives, how much, quality, when it clears, hazards.
    """
    scores = report_data.get("scores", [])[:7]
    safety_flags = report_data.get("safety_flags", [])
    model_spread = report_data.get("model_spread", [])
    patterns = report_data.get("patterns", [])

    if not scores:
        return ""

    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def day_name(iso_date):
        try:
            from datetime import datetime as dt
            d = dt.strptime(str(iso_date)[:10], "%Y-%m-%d")
            return DAY_NAMES[d.weekday()]
        except (ValueError, TypeError):
            return str(iso_date)

    # Find snow days and dry days
    snow_days = []
    dry_days = []
    best_day = None
    best_snow = 0
    total_snow = 0

    for i, s in enumerate(scores):
        snow_cm = s.get("conditions", {}).get("snowfall_24h_cm", 0) or 0
        total_snow += snow_cm
        if snow_cm >= 1:
            snow_days.append((i, s, snow_cm))
        else:
            dry_days.append((i, s))
        if snow_cm > best_snow:
            best_snow = snow_cm
            best_day = (i, s)

    parts = []

    if not snow_days:
        parts.append("Dry week ahead with no significant snowfall expected.")
    elif len(snow_days) == 1:
        i, s, cm = snow_days[0]
        name = day_name(s["date"])
        # Check if there's a spread range
        spread = model_spread[i] if i < len(model_spread) else None
        if spread and spread.get("spread_cm", 0) > 2:
            parts.append(f"Snow on <b>{name}</b> with <b>{spread['min_cm']:.0f}-{spread['max_cm']:.0f}cm</b> expected.")
        else:
            parts.append(f"Light snow on <b>{name}</b> with <b>{cm:.0f}cm</b> expected.")
    else:
        # Multiple snow days - find the sequence
        first_i, first_s, _ = snow_days[0]
        last_i, last_s, _ = snow_days[-1]
        first_name = day_name(first_s["date"])
        last_name = day_name(last_s["date"])

        # Check total range from model spread
        total_min = sum(model_spread[i].get("min_cm", 0) for i, _, _ in snow_days if i < len(model_spread))
        total_max = sum(model_spread[i].get("max_cm", 0) for i, _, _ in snow_days if i < len(model_spread))

        if first_i == last_i:
            parts.append(f"Snow on <b>{first_name}</b>.")
        elif last_i - first_i == len(snow_days) - 1:
            # Consecutive
            parts.append(f"Snow from <b>{first_name}</b> through <b>{last_name}</b>.")
        else:
            parts.append(f"Snow on multiple days, <b>{first_name}</b> to <b>{last_name}</b>.")

        if total_snow > 0:
            parts.append(f"Total accumulation around <b>{total_snow:.0f}cm</b>.")

    # Best day
    if best_day:
        i, s = best_day
        name = day_name(s["date"])
        crystal = s.get("crystal_type", "")
        quality_str = f" ({crystal})" if crystal and crystal != "mixed" else ""
        label = s.get("label", "")
        if label in ("EPIC", "GOOD"):
            parts.append(f"Best day: <b>{name}</b> with {best_snow:.0f}cm{quality_str} — {label}.")

    # Wind warnings
    wind_warnings = [f for f in safety_flags if "wind" in f.lower()]
    if wind_warnings:
        parts.append(f"<span style='color:var(--skip)'>Strong winds expected — check conditions before heading out.</span>")

    # Pattern highlights
    for p in patterns[:2]:
        ptype = p.get("type", "")
        if ptype == "storm_then_clear":
            parts.append("Storm-then-clear setup may produce excellent post-storm conditions.")
        elif ptype == "bluebird_day":
            parts.append("Bluebird conditions in the outlook.")

    return " ".join(parts)


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

    # Build weather narrative
    narrative = build_narrative(report_data)

    html = template.render(
        data=report_data,
        narrative=narrative,
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
