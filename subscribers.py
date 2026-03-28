"""Subscriber management — handles /start, /stop, and /report via Telegram getUpdates.

Polls for new messages each cron run, maintains a subscriber list in
.subscribers.json (project root, NOT docs/ to avoid leaking chat IDs).
All subscribers receive the same powder alerts,
snow watches, and pattern alerts.
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

from crowd_reports import parse_report, store_report

logger = logging.getLogger(__name__)

SUBSCRIBERS_FILE = Path(__file__).parent / ".subscribers.json"


def _load_subscribers() -> dict:
    """Load subscriber data from JSON file, creating default if missing."""
    if SUBSCRIBERS_FILE.exists():
        try:
            with open(SUBSCRIBERS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read subscribers file: {e}")
    return {"subscribers": {}, "last_update_id": 0}


def _save_subscribers(data: dict):
    """Persist subscriber data to JSON file."""
    SUBSCRIBERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def process_subscriber_updates() -> list:
    """Poll Telegram for /start and /stop commands, return list of subscriber chat_ids.

    Called during each cron run. Uses getUpdates with offset to only
    process new messages since last run.

    Returns:
        List of chat_id strings for all active subscribers.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    owner_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not owner_chat_id:
        logger.debug("Telegram credentials not set, skipping subscriber updates")
        return [owner_chat_id] if owner_chat_id else []

    data = _load_subscribers()

    # Ensure owner is always subscribed
    if owner_chat_id not in data["subscribers"]:
        data["subscribers"][owner_chat_id] = {
            "subscribed_at": datetime.now(timezone.utc).isoformat(),
            "username": "owner",
        }

    # Poll for new messages
    offset = data.get("last_update_id", 0)
    if offset:
        offset += 1  # getUpdates offset = last_id + 1 to skip already-processed

    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getUpdates",
            params={"offset": offset, "timeout": 5, "allowed_updates": '["message"]'},
            timeout=10,
        )
        resp.raise_for_status()
        updates = resp.json().get("result", [])
    except Exception as e:
        logger.error(f"Failed to poll Telegram updates: {e}")
        _save_subscribers(data)
        return list(data["subscribers"].keys())

    for update in updates:
        data["last_update_id"] = max(data.get("last_update_id", 0), update["update_id"])

        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        username = msg.get("from", {}).get("username", "")

        if not chat_id or not text:
            continue

        cmd = text.split()[0].split("@")[0]
        if cmd == "/start":
            if chat_id not in data["subscribers"]:
                data["subscribers"][chat_id] = {
                    "subscribed_at": datetime.now(timezone.utc).isoformat(),
                    "username": username,
                }
                logger.info(f"New subscriber: {chat_id} (@{username})")
            else:
                # Update username if changed (e.g. owner auto-enrolled as "owner")
                if username:
                    data["subscribers"][chat_id]["username"] = username
            _send_reply(bot_token, chat_id,
                        "Welcome to Powder Bot! You'll receive alerts when "
                        "conditions look good at Popova Shapka.\n\n"
                        "Send /stop to unsubscribe.")

        elif cmd == "/stop":
            if chat_id == owner_chat_id:
                _send_reply(bot_token, chat_id,
                            "Owner account can't unsubscribe.")
                continue
            if chat_id in data["subscribers"]:
                del data["subscribers"][chat_id]
                logger.info(f"Unsubscribed: {chat_id} (@{username})")
            _send_reply(bot_token, chat_id,
                        "You've been unsubscribed. Send /start to re-subscribe.")

        elif cmd == "/report":
            # Crowd-sourced condition report
            reports_path = str(Path(__file__).parent / "docs" / "verification" / "crowd_reports.json")
            report_data = parse_report(text)
            store_report(chat_id, report_data, reports_path)
            parts = []
            if report_data.get("depth_cm"):
                parts.append(f"{report_data['depth_cm']}cm")
            if report_data.get("quality"):
                parts.append(report_data["quality"])
            if report_data.get("sky"):
                parts.append(report_data["sky"])
            summary = ", ".join(parts) if parts else "noted"
            _send_reply(bot_token, chat_id,
                        f"Thanks for the report! Logged: {summary}\n"
                        "Your reports help improve forecast accuracy.")

    _save_subscribers(data)
    chat_ids = list(data["subscribers"].keys())
    logger.info(f"Active subscribers: {len(chat_ids)}")
    return chat_ids


def _send_reply(bot_token: str, chat_id: str, text: str):
    """Send a simple text reply to a chat."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Failed to send reply to {chat_id}: {e}")
