"""Automatic weekly recalibration of model weights and Kalman filter parameters.

Runs on Sundays to:
1. Recompute model weights from last 30 days of verification stats
2. Adjust Kalman filter noise parameters based on innovation sequence
3. Detect seasonal regime changes (spring warming increases process noise)
4. Log recalibration decisions
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def should_recalibrate() -> bool:
    """Check if today is Sunday (recalibration day)."""
    return datetime.utcnow().weekday() == 6  # Sunday = 6


def run_weekly_recalibration(config: dict, docs_dir: str = "docs") -> dict:
    """Run weekly recalibration of model weights and Kalman parameters.

    Args:
        config: Full config dict.
        docs_dir: Path to docs/ directory.

    Returns:
        Dict summarizing recalibration actions taken.
    """
    logger.info("Running weekly recalibration")
    actions = []

    stats_path = os.path.join(docs_dir, "verification", "stats.json")
    weights_path = os.path.join(docs_dir, "verification", "model_weights.json")
    kalman_path = os.path.join(docs_dir, "verification", "kalman_state.json")
    log_path = os.path.join(docs_dir, "verification", "recalibration_log.json")

    # Load verification stats
    stats = _load_json(stats_path)
    if not stats:
        logger.info("No verification stats available, skipping recalibration")
        return {"actions": [], "skipped": True}

    # 1. Recompute model weights from recent 30-day window
    weights_updated = _recompute_weights(stats, weights_path)
    if weights_updated:
        actions.append("Updated model weights from 30-day verification window")

    # 2. Adjust Kalman filter noise parameters
    kalman_adjusted = _adjust_kalman_parameters(stats, kalman_path, config)
    if kalman_adjusted:
        actions.append(f"Adjusted Kalman parameters: {kalman_adjusted}")

    # 2.5. Train ML models for all targets (snowfall, temp, wind, sunshine)
    analogs_path = os.path.join(docs_dir, "verification", "analogs.json")
    models_dir = os.path.join(docs_dir, "verification")
    try:
        from ml_postprocess import should_use_ml, train_all_models
        if should_use_ml(analogs_path):
            ml_results = train_all_models(analogs_path, models_dir)
            for target, info in ml_results.items():
                actions.append(f"ML {target}: CV MAE={info['cv_mae']:.2f}{info['unit']} ({info['n_samples']} samples)")
    except ImportError:
        pass  # scikit-learn not available
    except Exception as e:
        logger.warning("ML training failed: %s", e)

    # 3. Detect seasonal regime change
    regime = _detect_regime_change(stats)
    if regime:
        actions.append(f"Regime change detected: {regime}")

    # Log recalibration
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "actions": actions,
        "n_verifications": stats.get("n_verifications", 0),
    }
    _append_log(log_path, log_entry)

    logger.info("Recalibration complete: %d actions", len(actions))
    return log_entry


def _recompute_weights(stats: dict, weights_path: str) -> bool:
    """Recompute model weights from verification statistics.

    Uses the same inverse-MAE approach as update_model_weights() but
    operates on the accumulated stats rather than a single verification run.
    """
    per_model = stats.get("per_model", {})
    if not per_model:
        return False

    inv_mae = {}
    bias_corrections = {}

    for model, variables in per_model.items():
        snow_metrics = variables.get("snowfall", {})
        mae = snow_metrics.get("mae")
        bias = snow_metrics.get("bias")

        if mae is not None and mae > 0:
            inv_mae[model] = 1.0 / mae
        elif mae == 0:
            inv_mae[model] = 10.0

        if bias is not None:
            bias_corrections[model] = {"snowfall": bias}

    if not inv_mae:
        return False

    total_inv = sum(inv_mae.values())
    raw_weights = {m: v / total_inv for m, v in inv_mae.items()}

    # Apply 10% floor and renormalize
    min_weight = 0.1
    weights = {}
    for m, w in raw_weights.items():
        weights[m] = max(w, min_weight)
    total_w = sum(weights.values())
    weights = {m: round(w / total_w, 4) for m, w in weights.items()}

    result = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "weights": weights,
        "bias_corrections": bias_corrections,
        "ewma_mae": {m: v.get("snowfall", {}) for m, v in per_model.items()},
        "recalibrated": True,
    }

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Weights recomputed: %s", weights)
    return True


def _adjust_kalman_parameters(stats: dict, kalman_path: str, config: dict) -> str | None:
    """Adjust Kalman filter Q/R based on innovation sequence variance.

    If recent errors are larger than expected (innovations too big),
    increase process noise Q to allow the filter to adapt faster.
    If errors are stable, keep current parameters.
    """
    kalman_state = _load_json(kalman_path)
    if not kalman_state:
        return None

    base_Q = config.get("scoring", {}).get("kalman", {}).get("process_noise_q", 0.01)
    adjustments = []

    for key, state in kalman_state.items():
        if not isinstance(state, dict) or "P" not in state:
            continue

        n_updates = state.get("n_updates", 0)
        if n_updates < 5:
            continue  # Not enough data to adjust

        # If the estimate variance P is very low, the filter may be too rigid
        # Increase Q slightly to allow adaptation
        P = state.get("P", 1.0)
        if P < base_Q * 0.1:
            # Filter is too confident — increase Q
            state["P"] = base_Q * 2
            adjustments.append(f"{key}: P reset (was too confident)")

    if adjustments:
        with open(kalman_path, "w") as f:
            json.dump(kalman_state, f, indent=2)
        return "; ".join(adjustments)

    return None


def _detect_regime_change(stats: dict) -> str | None:
    """Detect if we've entered a different weather regime.

    Spring warming: if overall temperature bias is increasing (models
    predicting colder than reality), we're entering spring and should
    increase process noise to allow faster adaptation.
    """
    overall = stats.get("overall", {})
    temp_metrics = overall.get("temperature", {})
    temp_bias = temp_metrics.get("bias")

    if temp_bias is not None:
        if temp_bias < -3:
            return "spring_warming (models cold-biased, temperatures rising)"
        elif temp_bias > 3:
            return "cold_spell (models warm-biased, temperatures dropping)"

    return None


def _load_json(path: str) -> dict:
    """Load JSON from path, return empty dict on failure."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _append_log(log_path: str, entry: dict) -> None:
    """Append an entry to the recalibration log (keep last 52 weeks)."""
    log = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            log = []

    log.append(entry)
    if len(log) > 52:
        log = log[-52:]

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
