"""Forecast change detection — compare current forecast with previous run.

Detects significant changes in scores, snowfall amounts, label boundaries,
and peak day shifts. Used to trigger condition-change notifications.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_forecast_diff(current_scores, previous_latest_path):
    """Compare current forecast with previous run.

    Args:
        current_scores: list of current score dicts
        previous_latest_path: path to previous latest.json

    Returns dict with:
    - changes: list of significant changes
    - summary: str describing the changes
    - should_alert: bool
    - alert_type: str or None ("upgrade" / "downgrade")
    """
    result = {
        "changes": [],
        "summary": None,
        "should_alert": False,
        "alert_type": None,
    }

    # Load previous data
    prev_scores = _load_previous_scores(previous_latest_path)
    if not prev_scores:
        return result

    week_current = current_scores[:7]
    changes = []

    # Index previous scores by date
    prev_by_date = {s["date"]: s for s in prev_scores}

    for s in week_current:
        date = s.get("date", "")
        prev = prev_by_date.get(date)
        if not prev:
            continue

        # Score change >= 15 points
        old_score = prev.get("total", 0)
        new_score = s.get("total", 0)
        score_diff = new_score - old_score

        if abs(score_diff) >= 15:
            old_label = prev.get("label", "?")
            new_label = s.get("label", "?")
            label_changed = old_label != new_label
            detail = f"{old_label} -> {new_label}" if label_changed else f"{score_diff:+.0f}pts"
            changes.append({
                "date": date,
                "field": "score",
                "old": old_score,
                "new": new_score,
                "detail": detail,
            })

        # Snowfall change >= 40%
        old_snow = prev.get("conditions", {}).get("snowfall_24h_cm", 0)
        new_snow = s.get("conditions", {}).get("snowfall_24h_cm", 0)
        if old_snow > 3 or new_snow > 3:
            max_snow = max(old_snow, new_snow, 1)
            pct_change = abs(new_snow - old_snow) / max_snow
            if pct_change >= 0.4:
                pct_str = f"{((new_snow - old_snow) / max_snow) * 100:+.0f}%"
                changes.append({
                    "date": date,
                    "field": "snowfall",
                    "old": old_snow,
                    "new": new_snow,
                    "detail": pct_str,
                })

    # Peak day shift
    if week_current and prev_scores:
        current_peak = max(week_current, key=lambda s: s["total"])
        prev_week = [s for s in prev_scores if s.get("date") in {x.get("date") for x in week_current}]
        if prev_week:
            prev_peak = max(prev_week, key=lambda s: s["total"])
            try:
                current_dates = [s["date"] for s in week_current]
                if current_peak["date"] in current_dates and prev_peak["date"] in current_dates:
                    current_idx = current_dates.index(current_peak["date"])
                    prev_idx = current_dates.index(prev_peak["date"])
                    if abs(current_idx - prev_idx) >= 2:
                        changes.append({
                            "date": current_peak["date"],
                            "field": "peak_shift",
                            "old": prev_peak["date"],
                            "new": current_peak["date"],
                            "detail": f"Peak shifted from {prev_peak['date']} to {current_peak['date']}",
                        })
            except (ValueError, KeyError):
                pass

    if not changes:
        return result

    # Determine if we should alert
    score_changes = [c for c in changes if c["field"] == "score"]
    should_alert = False
    alert_type = None

    if score_changes:
        # Alert on label boundary crossings
        for c in score_changes:
            if "->" in c.get("detail", ""):
                should_alert = True
                if c["new"] > c["old"]:
                    alert_type = "upgrade"
                else:
                    alert_type = "downgrade"
                break

        # Also alert on large score swings even without label change
        if not should_alert:
            max_change = max(abs(c["new"] - c["old"]) for c in score_changes)
            if max_change >= 20:
                should_alert = True
                avg_direction = sum(c["new"] - c["old"] for c in score_changes) / len(score_changes)
                alert_type = "upgrade" if avg_direction > 0 else "downgrade"

    # Build summary
    summary_parts = []
    for c in score_changes[:2]:
        from insights import _fmt_date_short
        day = _fmt_date_short(c["date"])
        summary_parts.append(f"{day} {'upgraded' if c['new'] > c['old'] else 'downgraded'}: {c['old']:.0f} -> {c['new']:.0f} ({c['detail']})")
    summary = "; ".join(summary_parts) if summary_parts else None

    result["changes"] = changes
    result["summary"] = summary
    result["should_alert"] = should_alert
    result["alert_type"] = alert_type
    return result


def _load_previous_scores(path):
    """Load previous forecast scores from latest.json."""
    try:
        p = Path(path)
        if not p.exists():
            logger.debug("No previous latest.json found (first run)")
            return None

        with open(p) as f:
            data = json.load(f)

        scores = data.get("scores")
        if not scores or not isinstance(scores, list):
            logger.warning("Previous latest.json has no valid scores")
            return None

        return scores

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load previous forecast: {e}")
        return None
