"""Notification system — Telegram alerts via environment variables."""

import os
import logging
import requests

logger = logging.getLogger(__name__)

# Score label to emoji mapping
LABEL_EMOJI = {
    "EPIC": "\u2744\ufe0f\u2744\ufe0f\u2744\ufe0f",  # snowflakes
    "GOOD": "\u2744\ufe0f\u2744\ufe0f",
    "FAIR": "\u2744\ufe0f",
    "MARGINAL": "\u2601\ufe0f",
    "SKIP": "\u274c",
}

SKY_EMOJI = {
    "BLUEBIRD": "\u2600\ufe0f",
    "MOSTLY_SUNNY": "\ud83c\udf24\ufe0f",
    "PARTLY_CLOUDY": "\u26c5",
    "MOSTLY_OVERCAST": "\ud83c\udf25\ufe0f",
    "OVERCAST": "\u2601\ufe0f",
}


def send_telegram_alert(title: str, body: str) -> bool:
    """Send a Telegram message using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.debug("Telegram credentials not set in environment, skipping")
        return False

    text = f"*{title}*\n\n{body}"
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


def format_powder_alert(scores: list, dates: list, patterns: list) -> tuple:
    """Format a powder alert message. Returns (title, body)."""

    # Find best days
    alerts = []
    for i, score in enumerate(scores):
        if score["total"] >= 60:
            date = dates[i] if i < len(dates) else f"Day {i+1}"
            emoji = LABEL_EMOJI.get(score["label"], "")
            sky_emoji = SKY_EMOJI.get(score["sky"]["classification"], "")
            cond = score["conditions"]

            alerts.append(
                f"{emoji} *{date}* — Score: {score['total']:.0f} ({score['label']})\n"
                f"  Snow: {cond['snowfall_24h_cm']:.0f}cm | "
                f"Temp: {cond['temperature_c']:.0f}\u00b0C | "
                f"Wind: {cond['wind_speed_kmh']:.0f}km/h\n"
                f"  Sky: {sky_emoji} {score['sky']['classification'].replace('_', ' ').title()}"
            )

    if not alerts:
        return None, None

    title = "Powder Alert \u2014 Popova Shapka"
    body_parts = alerts[:5]  # Max 5 days in one alert

    # Add pattern insights
    for p in patterns:
        if p["type"] == "storm_then_clear":
            body_parts.append(f"\n\u26a1 {p['message']}")
        elif p["type"] == "multi_day_storm":
            body_parts.append(f"\n\ud83c\udf28\ufe0f {p['message']}")

    body = "\n\n".join(body_parts)
    return title, body


def format_daily_briefing(scores: list, dates: list, patterns: list,
                          location_name: str) -> tuple:
    """Format a daily morning briefing. Returns (title, body)."""

    title = f"Snow Briefing \u2014 {location_name}"

    lines = ["*7-Day Outlook*\n"]
    for i in range(min(7, len(scores))):
        s = scores[i]
        date = dates[i] if i < len(dates) else f"Day {i+1}"
        emoji = LABEL_EMOJI.get(s["label"], "")
        sky_emoji = SKY_EMOJI.get(s["sky"]["classification"], "")
        cond = s["conditions"]
        lines.append(
            f"{emoji} *{date}*: {s['total']:.0f}pts "
            f"| {cond['snowfall_24h_cm']:.0f}cm "
            f"| {cond['temperature_c']:.0f}\u00b0C "
            f"| {cond['wind_speed_kmh']:.0f}km/h "
            f"{sky_emoji}"
        )

    # Patterns summary
    if patterns:
        lines.append("\n*Patterns Detected:*")
        for p in patterns[:3]:
            lines.append(f"  \u2022 {p['message']}")

    body = "\n".join(lines)
    return title, body


def notify_if_needed(scores: list, dates: list, patterns: list,
                     location_name: str, scoring_cfg: dict):
    """Check thresholds and send Telegram notification if warranted."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.debug("Telegram not configured, skipping notifications")
        return

    alert_min = scoring_cfg.get("alerts", {}).get("min_score", 60)
    alert_days = [s for s in scores[:7] if s["total"] >= alert_min]

    if alert_days:
        title, body = format_powder_alert(scores, dates, patterns)
        if title:
            send_telegram_alert(title, body)
    else:
        # Send daily briefing instead
        title, body = format_daily_briefing(scores, dates, patterns, location_name)
        send_telegram_alert(title, body)
