"""Notification system — Telegram alerts only when something is worth watching.

Sends notifications only when upcoming conditions are noteworthy:
- Powder Alert: high scores (GOOD/EPIC days) in the next 7 days
- Condition Change: significant forecast changes between runs
- Snow Watch: significant snowfall building but not yet powder-grade
- Pattern Alert: bluebird setups, multi-day storms, etc.

Stays silent when nothing interesting is happening.
Uses a state file to avoid re-alerting about the same situation.
"""

import os
import hashlib
import json
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_EMOJI = {
    "EPIC": "\u2744\ufe0f\u2744\ufe0f\u2744\ufe0f",
    "GOOD": "\u2744\ufe0f\u2744\ufe0f",
    "FAIR": "\u2744\ufe0f",
    "MARGINAL": "\u2601\ufe0f",
    "SKIP": "\u274c",
}

SKY_EMOJI = {
    "BLUEBIRD": "\u2600\ufe0f",
    "MOSTLY_SUNNY": "\U0001f324\ufe0f",
    "PARTLY_CLOUDY": "\u26c5",
    "MOSTLY_OVERCAST": "\U0001f325\ufe0f",
    "OVERCAST": "\u2601\ufe0f",
}

STATE_FILE = Path(__file__).parent / "docs" / ".last_alert"


def send_telegram(text: str) -> bool:
    """Send a Telegram message using env vars. Returns True on success."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.debug("Telegram credentials not set, skipping")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        logger.info(f"Telegram message sent to {chat_id}")
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def _alert_fingerprint(alert_type: str, key_dates: list, snow_total: float) -> str:
    """Generate a fingerprint for an alert situation.

    Changes when the situation meaningfully changes (new days involved,
    significantly different snow totals), stays the same for minor updates.
    """
    # Bucket snow to nearest 5cm so small fluctuations don't re-trigger
    snow_bucket = round(snow_total / 5) * 5
    raw = f"{alert_type}:{','.join(sorted(key_dates))}:{snow_bucket}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _was_already_sent(fingerprint: str) -> bool:
    """Check if this alert fingerprint was already sent."""
    try:
        if STATE_FILE.exists():
            stored = STATE_FILE.read_text().strip()
            return fingerprint in stored.split("\n")
    except Exception:
        pass
    return False


def _mark_sent(fingerprint: str):
    """Record that this alert was sent. Keeps last 10 fingerprints."""
    try:
        existing = []
        if STATE_FILE.exists():
            existing = STATE_FILE.read_text().strip().split("\n")
            existing = [f for f in existing if f]
        existing.append(fingerprint)
        # Keep only the last 10 to allow re-alerting after situation changes
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text("\n".join(existing[-10:]) + "\n")
    except Exception as e:
        logger.warning(f"Could not write alert state: {e}")


def _format_powder_alert(powder_days: list, dates: list, patterns: list,
                         dashboard_url: str, insights=None,
                         avalanche_danger=None) -> str:
    """Format a powder alert with progressive disclosure layout."""
    lines = ["\u2744\ufe0f\u2744\ufe0f *POWDER ALERT*"]
    lines.append("Popova Shapka\n")

    # Best day highlight
    best = max(powder_days, key=lambda s: s["total"])
    best_date = best.get("date", "?")
    cond = best["conditions"]
    sky_class = best["sky"]["classification"].replace("_", " ").title()
    sky_emoji = SKY_EMOJI.get(best["sky"]["classification"], "")

    lines.append(f"\U0001f3d4 *Best Day: {best_date}*")
    lines.append(f"Score: {best['total']:.0f} ({best['label']})")
    lines.append(
        f"{cond['snowfall_24h_cm']:.0f}cm fresh \u00b7 "
        f"{cond['temperature_c']:.0f}\u00b0C \u00b7 "
        f"{cond['wind_speed_kmh']:.0f}km/h \u00b7 "
        f"{sky_emoji} {sky_class}"
    )

    # 7-day outlook
    week = [s for s in (powder_days[0:1] if len(powder_days) == 1 else [])]  # placeholder
    lines.append(f"\n\U0001f4ca *7-Day Outlook*")
    # Show all powder days with the best one marked
    for s in powder_days[:5]:
        date = s.get("date", "?")
        emoji = LABEL_EMOJI.get(s["label"], "")
        marker = "  \u2190 best" if s is best else ""
        lines.append(f"{emoji} {date}: {s['total']:.0f} {s['label']}{marker}")

    # Snowpack and avalanche info
    if insights and insights.get("snowpack_status"):
        lines.append(f"\n\U0001f4c9 Snowpack: {insights['snowpack_status']}")

    if avalanche_danger and isinstance(avalanche_danger, list) and avalanche_danger:
        max_danger = max(avalanche_danger[:7], key=lambda a: a.get("level", 0))
        if max_danger.get("level", 0) >= 3:
            lines.append(f"\u26a0\ufe0f Avalanche: Level {max_danger['level']} ({max_danger.get('level_name', 'N/A')})")

    # Patterns
    for p in patterns:
        if p["type"] == "storm_then_clear":
            lines.append(f"\n\u26a1 {p['message']}")
        elif p["type"] == "multi_day_storm":
            lines.append(f"\n\U0001f328\ufe0f {p['message']}")

    lines.append(f"\n\U0001f4f1 [Dashboard]({dashboard_url})")
    return "\n".join(lines)


def _format_snow_watch(snow_days: list, total_7d: float, patterns: list,
                       dashboard_url: str) -> str:
    """Format a snow watch — snow on the radar, worth monitoring."""
    lines = ["\U0001f4a8 *Snow Watch \u2014 Popova Shapka*\n"]
    lines.append(f"~{total_7d:.0f}cm in the next 7 days\n")

    for s in snow_days[:4]:
        date = s.get("date", "?")
        cond = s["conditions"]
        lines.append(f"\u2022 *{date}*: {cond['snowfall_24h_cm']:.0f}cm forecast")

    for p in patterns:
        if p["type"] == "multi_day_storm":
            lines.append(f"\n\U0001f328\ufe0f {p['message']}")
        elif p["type"] == "storm_then_clear":
            lines.append(f"\n\u26a1 {p['message']}")

    lines.append(f"\n[Monitor on dashboard]({dashboard_url})")
    return "\n".join(lines)


def _format_condition_change(diff_data, dashboard_url):
    """Format condition change alert from forecast diff data."""
    lines = ["\U0001f4ca *Conditions Update*"]

    summary = diff_data.get("summary")
    if summary:
        lines.append(summary)
    else:
        for c in diff_data.get("changes", [])[:3]:
            if c["field"] == "score":
                lines.append(f"{c['date']}: {c['old']:.0f} \u2192 {c['new']:.0f} ({c['detail']})")

    lines.append(f"\n[Dashboard]({dashboard_url})")
    return "\n".join(lines)


def _format_pattern_alert(patterns: list, dashboard_url: str) -> str:
    """Format a pattern-only alert — noteworthy setup detected."""
    type_emoji = {
        "storm_then_clear": "\u26a1",
        "multi_day_storm": "\U0001f328\ufe0f",
        "cold_snap": "\u2744\ufe0f",
        "warming_trend": "\u26a0\ufe0f",
        "upslope_event": "\U0001f3d4",
        "wind_slab_risk": "\u26a0\ufe0f",
        "melt_freeze_crust": "\U0001f9ca",
        "legendary_setup": "\U0001f3c6",
        "near_miss_warm": "\U0001f440",
        "near_miss_wind": "\U0001f440",
        "near_miss_low_snow": "\U0001f440",
    }

    lines = ["\U0001f4cc *Conditions Update \u2014 Popova Shapka*\n"]
    for p in patterns:
        emoji = type_emoji.get(p["type"], "\u2022")
        lines.append(f"{emoji} {p['message']}")

    lines.append(f"\n[Dashboard]({dashboard_url})")
    return "\n".join(lines)


def notify_if_needed(scores: list, dates: list, patterns: list,
                     location_name: str, scoring_cfg: dict,
                     insights=None, avalanche_danger=None,
                     forecast_diff=None):
    """Send a Telegram notification only if something noteworthy is coming.

    Priority:
    1. Powder Alert — any day scores >= 60 (GOOD/EPIC)
    2. Snow Watch — significant snow (>5cm days) but below powder threshold
    2.5. Condition Change — significant forecast changes between runs
    3. Pattern Alert — bluebird setup, multi-day storm, etc. with no snow alert
    4. Silent — nothing interesting, don't bother the user
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logger.debug("Telegram not configured, skipping notifications")
        return

    dashboard_url = os.environ.get(
        "DASHBOARD_URL", "https://quotz.github.io/snow-forecast/"
    )

    alert_min = scoring_cfg.get("alerts", {}).get("min_score", 60)
    week = scores[:7]

    # Confidence filtering: exclude days with confidence_pct < 50%
    def _is_confident(s):
        conf_pct = s.get("confidence_pct")
        if conf_pct is not None and conf_pct < 50:
            return False
        return True

    # 1. Powder Alert — high-score days (with confidence filter)
    powder_days = [s for s in week if s["total"] >= alert_min and _is_confident(s)]
    if powder_days:
        key_dates = [s["date"] for s in powder_days]
        total_snow = sum(s["conditions"]["snowfall_24h_cm"] for s in powder_days)
        fp = _alert_fingerprint("powder", key_dates, total_snow)

        if not _was_already_sent(fp):
            text = _format_powder_alert(
                powder_days, dates, patterns, dashboard_url,
                insights=insights, avalanche_danger=avalanche_danger,
            )
            if send_telegram(text):
                _mark_sent(fp)
                logger.info(f"Powder alert sent (fp={fp})")
        else:
            logger.info(f"Powder alert already sent (fp={fp}), skipping")
        return

    # 2. Snow Watch — notable snowfall building
    snow_days = [s for s in week if s["conditions"]["snowfall_24h_cm"] >= 5]
    total_7d = sum(s["conditions"]["snowfall_24h_cm"] for s in week)

    if snow_days and total_7d >= 10:
        key_dates = [s["date"] for s in snow_days]
        fp = _alert_fingerprint("snow_watch", key_dates, total_7d)

        if not _was_already_sent(fp):
            text = _format_snow_watch(snow_days, total_7d, patterns, dashboard_url)
            if send_telegram(text):
                _mark_sent(fp)
                logger.info(f"Snow watch sent (fp={fp})")
        else:
            logger.info(f"Snow watch already sent (fp={fp}), skipping")
        return

    # 2.5. Condition Change — significant forecast changes
    if forecast_diff and forecast_diff.get("should_alert"):
        summary = forecast_diff.get("summary", "")
        fp = _alert_fingerprint("condition_change", [summary[:50]], 0)

        if not _was_already_sent(fp):
            text = _format_condition_change(forecast_diff, dashboard_url)
            if send_telegram(text):
                _mark_sent(fp)
                logger.info(f"Condition change alert sent (fp={fp})")
        else:
            logger.info(f"Condition change already sent (fp={fp}), skipping")
        return

    # 3. Pattern Alert — noteworthy pattern with no snow alert
    notable_patterns = [p for p in patterns if p["type"] in
                        ("storm_then_clear", "multi_day_storm")]
    if notable_patterns:
        key = "|".join(p["type"] for p in notable_patterns)
        fp = _alert_fingerprint("pattern", [key], 0)

        if not _was_already_sent(fp):
            text = _format_pattern_alert(notable_patterns, dashboard_url)
            if send_telegram(text):
                _mark_sent(fp)
                logger.info(f"Pattern alert sent (fp={fp})")
        else:
            logger.info(f"Pattern alert already sent (fp={fp}), skipping")
        return

    # 4. Nothing noteworthy — stay silent
    logger.info("No noteworthy conditions, skipping notification")
