"""Kalman filter bias correction for forecast models.

Implements a univariate Kalman filter per model per variable that adaptively
corrects systematic forecast biases. Converges faster and handles non-stationary
bias better than simple subtraction.

State persists to docs/verification/kalman_state.json.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def _load_state(state_path: str) -> dict:
    """Load Kalman filter state from disk."""
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load Kalman state: %s", e)
        return {}


def _save_state(state: dict, state_path: str) -> None:
    """Persist Kalman filter state to disk."""
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def kalman_correct(model_name: str, variable: str, forecast_value: float,
                   state_path: str, Q: float = 0.01, R: float = 1.0,
                   initial_variance: float = 10.0) -> float:
    """Apply Kalman filter correction to a single forecast value.

    The filter tracks a bias state x_k (how much the model over-predicts)
    and subtracts it from the raw forecast.

    State model:
        x_k = x_{k-1} + w_k,  w_k ~ N(0, Q)    (bias evolves slowly)
        z_k = x_k + v_k,      v_k ~ N(0, R)     (observed error is noisy)

    On each observation (forecast_value - observed), the filter updates.
    For correction (no observation yet), we use the current bias estimate.

    Args:
        model_name: Name of the forecast model.
        variable: Variable being corrected (e.g., "snowfall", "temperature").
        forecast_value: Raw forecast value to correct.
        state_path: Path to Kalman state JSON file.
        Q: Process noise variance (how fast bias can change).
        R: Observation noise variance (how noisy errors are).
        initial_variance: Initial estimate uncertainty.

    Returns:
        Corrected forecast value (forecast - estimated_bias), clamped >= 0
        for snowfall.
    """
    state = _load_state(state_path)

    key = f"{model_name}.{variable}"
    model_state = state.get(key)

    if model_state is None:
        # Cold start: no bias estimate yet, return unchanged
        return forecast_value

    bias_estimate = model_state.get("x", 0.0)
    corrected = forecast_value - bias_estimate

    # Clamp non-negative variables
    if variable in ("snowfall", "rain"):
        corrected = max(0.0, corrected)

    return corrected


def kalman_batch_correct(model_values: list, model_names: list,
                         variable: str, state_path: str,
                         Q: float = 0.01, R: float = 1.0) -> list:
    """Apply Kalman correction to a list of model values.

    Drop-in replacement for apply_bias_correction() with Kalman filter.

    Args:
        model_values: List of forecast values from each model.
        model_names: List of model names (same order).
        variable: Variable name (e.g., "snowfall").
        state_path: Path to Kalman state JSON.
        Q: Process noise.
        R: Observation noise.

    Returns:
        List of corrected values. None values pass through unchanged.
    """
    if not model_names:
        return model_values

    corrected = []
    for i, val in enumerate(model_values):
        if val is None:
            corrected.append(val)
            continue
        name = model_names[i] if i < len(model_names) else None
        if name:
            corrected.append(kalman_correct(name, variable, val, state_path,
                                            Q=Q, R=R))
        else:
            corrected.append(val)
    return corrected


def kalman_update(model_name: str, variable: str,
                  forecast_value: float, observed_value: float,
                  state_path: str, Q: float = 0.01, R: float = 1.0,
                  initial_variance: float = 10.0) -> dict:
    """Update the Kalman filter with a new forecast-observation pair.

    Called during verification when we have ground truth (ERA5).

    Args:
        model_name: Forecast model name.
        variable: Variable name.
        forecast_value: What the model predicted.
        observed_value: What actually happened (ERA5).
        state_path: Path to Kalman state JSON.
        Q: Process noise variance.
        R: Observation noise variance.
        initial_variance: Initial P if cold-starting.

    Returns:
        Updated state dict for this model/variable.
    """
    state = _load_state(state_path)
    key = f"{model_name}.{variable}"

    model_state = state.get(key)
    if model_state is None:
        # Initialize from first observation
        model_state = {
            "x": 0.0,  # bias estimate
            "P": initial_variance,  # estimate variance
            "n_updates": 0,
        }

    x = model_state["x"]  # prior bias estimate
    P = model_state["P"]  # prior estimate variance

    # Prediction step: bias evolves with process noise
    x_pred = x
    P_pred = P + Q

    # Observation: the "measurement" of bias is (forecast - observed)
    z = forecast_value - observed_value  # observed bias

    # Innovation
    y = z - x_pred  # innovation (surprise)
    S = P_pred + R   # innovation variance

    # Kalman gain
    K = P_pred / S

    # Update step
    x_new = x_pred + K * y
    P_new = (1 - K) * P_pred

    model_state["x"] = round(x_new, 6)
    model_state["P"] = round(P_new, 6)
    model_state["n_updates"] = model_state.get("n_updates", 0) + 1

    # Save
    state[key] = model_state
    _save_state(state, state_path)

    return model_state


def initialize_from_ewma(weights_path: str, state_path: str,
                         initial_variance: float = 5.0) -> None:
    """Bootstrap Kalman state from existing EWMA bias corrections.

    If model_weights.json exists with bias_corrections, use those values
    to warm-start the Kalman filter instead of cold-starting from zero.

    Args:
        weights_path: Path to existing model_weights.json.
        state_path: Path to Kalman state JSON to create.
        initial_variance: Initial uncertainty for the warm-started state.
    """
    if os.path.exists(state_path):
        # Already initialized
        return

    if not os.path.exists(weights_path):
        return

    try:
        with open(weights_path) as f:
            weights_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    bias_corrections = weights_data.get("bias_corrections", {})
    if not bias_corrections:
        return

    state = {}
    for model_name, corrections in bias_corrections.items():
        for variable, bias_value in corrections.items():
            key = f"{model_name}.{variable}"
            state[key] = {
                "x": bias_value,
                "P": initial_variance,
                "n_updates": 0,
                "initialized_from": "ewma",
            }

    if state:
        _save_state(state, state_path)
        logger.info("Kalman state initialized from EWMA for %d model/variable pairs",
                     len(state))
