"""Community condition reports via Telegram.

Handles /report and /status commands from Telegram users.
Reports are parsed into structured data and stored for forecast verification.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SALT = "popova-shapka-2024"

QUALITY_KEYWORDS = {
    5: ["powder", "great", "epic", "amazing", "incredible", "perfect"],
    4: ["good", "nice", "solid"],
    3: ["ok", "okay", "decent", "average", "fine"],
    2: ["heavy", "wet", "crusty", "sticky", "dense"],
    1: ["ice", "icy", "terrible", "awful", "dangerous", "hard"],
}

WIND_KEYWORDS = {
    "light": ["light", "calm", "no wind", "still"],
    "moderate": ["moderate"],
    "strong": ["strong", "gusty", "windy", "gale", "howling"],
}


def parse_report(text: str) -> dict:
    """Parse a free-text condition report into structured data.

    Extracts snow depth, quality rating, wind conditions from natural language.
    Always stores raw text in notes.
    """
    # Strip the /report command prefix
    clean = re.sub(r"^/report\s*", "", text, flags=re.IGNORECASE).strip()

    result = {
        "snow_depth_cm": None,
        "quality_rating": None,
        "wind_conditions": None,
        "notes": clean if clean else text,
    }

    # Extract snow depth: "20cm", "20 cm", "20cm fresh"
    depth_match = re.search(r"(\d+)\s*cm", clean, re.IGNORECASE)
    if depth_match:
        result["snow_depth_cm"] = int(depth_match.group(1))

    # Check for "no new snow" / "no snow" -> 0cm
    if re.search(r"no\s+(new\s+)?snow", clean, re.IGNORECASE):
        result["snow_depth_cm"] = 0

    # Extract explicit quality rating: "quality 3" or "3/5"
    quality_match = re.search(r"quality\s+(\d)", clean, re.IGNORECASE)
    if quality_match:
        val = int(quality_match.group(1))
        if 1 <= val <= 5:
            result["quality_rating"] = val

    if result["quality_rating"] is None:
        slash_match = re.search(r"(\d)/5", clean)
        if slash_match:
            val = int(slash_match.group(1))
            if 1 <= val <= 5:
                result["quality_rating"] = val

    # Infer quality from keywords if not explicitly set
    if result["quality_rating"] is None:
        lower = clean.lower()
        for rating, keywords in QUALITY_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                result["quality_rating"] = rating
                break

    # Extract wind conditions
    lower = clean.lower()
    for condition, keywords in WIND_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            result["wind_conditions"] = condition
            break

    return result


def anonymize_user(user_id) -> str:
    """Hash a Telegram user ID to protect privacy."""
    raw = f"{SALT}:{user_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_reports(reports_path: str) -> list:
    """Load existing reports from JSON file."""
    path = Path(reports_path)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load reports from {reports_path}: {e}")
        return []


def save_reports(reports: list, reports_path: str):
    """Save reports list to JSON file, creating directory if needed."""
    path = Path(reports_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(reports, f, indent=2)


def _load_update_id(state_path: str):
    """Load last processed update ID from state file."""
    path = Path(state_path)
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (ValueError, IOError):
        return None


def _save_update_id(update_id: int, state_path: str):
    """Save last processed update ID to state file."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(update_id))


def _send_reply(bot_token: str, chat_id, text: str):
    """Send a reply message to a Telegram chat."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Failed to send reply: {e}")


def _handle_status(bot_token: str, chat_id):
    """Reply with current scores summary from latest.json."""
    latest_path = Path(__file__).parent / "docs" / "latest.json"
    if not latest_path.exists():
        _send_reply(bot_token, chat_id, "No forecast data available yet.")
        return

    try:
        with open(latest_path) as f:
            data = json.load(f)

        scores = data.get("scores", [])
        if not scores:
            _send_reply(bot_token, chat_id, "No scores available.")
            return

        lines = ["*Current Forecast — Popova Shapka*\n"]
        for s in scores[:7]:
            date = s.get("date", "?")
            total = s.get("total", 0)
            label = s.get("label", "?")
            snow = s.get("conditions", {}).get("snowfall_24h_cm", 0)
            lines.append(f"{date}: {total:.0f} ({label}) — {snow:.0f}cm snow")

        lines.append("\nhttps://quotz.github.io/snow-forecast/")
        _send_reply(bot_token, chat_id, "\n".join(lines))
    except Exception as e:
        logger.error(f"Error handling /status: {e}")
        _send_reply(bot_token, chat_id, "Error loading forecast data.")


def process_reports(bot_token: str, reports_path: str, state_path: str = None) -> list:
    """Process new Telegram messages for /report and /status commands.

    Uses getUpdates API with long polling. Tracks last processed message
    via state file to avoid reprocessing.

    Returns list of new report entries.
    """
    if state_path is None:
        # Derive state path relative to reports_path (e.g. docs/verification/reports.json → docs/.last_report_update_id)
        state_path = str(Path(reports_path).parent.parent / ".last_report_update_id")

    last_update_id = _load_update_id(state_path)

    params = {"timeout": 5}
    if last_update_id is not None:
        params["offset"] = last_update_id + 1

    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getUpdates",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch Telegram updates: {e}")
        return []

    if not data.get("ok"):
        logger.error(f"Telegram API error: {data}")
        return []

    updates = data.get("result", [])
    if not updates:
        return []

    existing_reports = load_reports(reports_path)
    new_reports = []
    max_update_id = last_update_id

    for update in updates:
        update_id = update.get("update_id", 0)
        if max_update_id is None or update_id > max_update_id:
            max_update_id = update_id

        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")

        if not text or not chat_id:
            continue

        if text.strip().lower().startswith("/status"):
            _handle_status(bot_token, chat_id)
            continue

        if text.strip().lower().startswith("/report"):
            try:
                parsed = parse_report(text)
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_id": anonymize_user(user_id) if user_id else "unknown",
                    "raw_text": text,
                    "parsed": parsed,
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
                new_reports.append(entry)
                _send_reply(
                    bot_token, chat_id,
                    f"Report received. Snow: {parsed['snow_depth_cm']}cm, "
                    f"Quality: {parsed['quality_rating']}/5"
                )
            except Exception as e:
                logger.error(f"Error processing report: {e}")
                _send_reply(bot_token, chat_id, "Sorry, couldn't process that report.")

    if new_reports:
        existing_reports.extend(new_reports)
        save_reports(existing_reports, reports_path)
        logger.info(f"Saved {len(new_reports)} new report(s)")

    if max_update_id is not None:
        _save_update_id(max_update_id, state_path)

    return new_reports


if __name__ == "__main__":
    import os
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        reports = process_reports(token, "docs/verification/reports.json")
        print(f"Processed {len(reports)} new reports")
    else:
        print("Set TELEGRAM_BOT_TOKEN to process reports")
