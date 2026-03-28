"""Ensemble statistics: model clustering, CRPS, Brier score, analog storage.

Provides probabilistic verification metrics and model independence analysis
to improve ensemble combination and forecast calibration.
"""
from __future__ import annotations

import json
import logging
import math
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Correlation / Clustering
# ---------------------------------------------------------------------------

def compute_model_correlations(history_forecasts: list) -> dict:
    """Compute pairwise Pearson correlation of model snowfall predictions.

    Args:
        history_forecasts: List of dicts with 'model_comparison_entry' containing
            'snowfall_values' and model names.

    Returns:
        Dict with 'matrix' (model_a -> model_b -> correlation),
        'n_samples', 'effective_dof'.
    """
    # Collect per-model time series
    model_series = {}
    for entry in history_forecasts:
        mc = entry.get("model_comparison_entry", {})
        models = entry.get("models", [])
        values = mc.get("snowfall_values", [])
        for j, model in enumerate(models):
            if j < len(values) and values[j] is not None:
                model_series.setdefault(model, []).append(values[j])

    model_names = sorted(model_series.keys())
    if len(model_names) < 2:
        return {"matrix": {}, "n_samples": 0, "effective_dof": len(model_names)}

    # Align series to common length
    min_len = min(len(model_series[m]) for m in model_names)
    if min_len < 5:
        return {"matrix": {}, "n_samples": min_len, "effective_dof": len(model_names)}

    # Compute correlation matrix
    matrix = {}
    for m_a in model_names:
        matrix[m_a] = {}
        for m_b in model_names:
            if m_a == m_b:
                matrix[m_a][m_b] = 1.0
            else:
                corr = _pearson(model_series[m_a][:min_len],
                                model_series[m_b][:min_len])
                matrix[m_a][m_b] = round(corr, 4) if corr is not None else 0.0

    # Compute effective degrees of freedom
    n_eff = effective_degrees_of_freedom(matrix, model_names)

    return {
        "matrix": matrix,
        "n_samples": min_len,
        "effective_dof": round(n_eff, 2),
        "model_names": model_names,
    }


def _pearson(x: list, y: list) -> float | None:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    if n < 3:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-10:
        return 0.0

    return cov / denom


def effective_degrees_of_freedom(corr_matrix: dict,
                                  model_names: list = None) -> float:
    """Compute effective degrees of freedom from correlation matrix.

    When models are correlated, the effective number of independent pieces
    of information is less than the raw model count. Uses the trace-based
    approximation: N_eff = N^2 / sum(r_ij^2).

    Args:
        corr_matrix: Dict mapping model_a -> model_b -> correlation.
        model_names: List of model names (keys of corr_matrix).

    Returns:
        N_eff (float), always >= 1 and <= N.
    """
    if not corr_matrix:
        return float(len(model_names)) if model_names else 1.0

    if model_names is None:
        model_names = sorted(corr_matrix.keys())

    n = len(model_names)
    if n <= 1:
        return float(n)

    sum_r_sq = 0.0
    for m_a in model_names:
        for m_b in model_names:
            r = corr_matrix.get(m_a, {}).get(m_b, 0.0)
            sum_r_sq += r * r

    if sum_r_sq < 1e-10:
        return float(n)

    n_eff = (n * n) / sum_r_sq
    return max(1.0, min(float(n), n_eff))


# ---------------------------------------------------------------------------
# CRPS (Continuous Ranked Probability Score)
# ---------------------------------------------------------------------------

def compute_crps(ensemble_values: list, observed: float) -> float | None:
    """Compute CRPS for a single observation against ensemble members.

    CRPS measures the quality of a probabilistic forecast. Lower is better.
    Uses the exact formula for a finite ensemble:
        CRPS = (1/N) * sum|x_i - obs| - (1/2N^2) * sum_ij|x_i - x_j|

    Args:
        ensemble_values: List of ensemble member predictions.
        observed: Observed value (ground truth).

    Returns:
        CRPS value (float), or None if insufficient data.
    """
    valid = [v for v in ensemble_values if v is not None]
    if len(valid) < 2:
        return None

    n = len(valid)

    # First term: mean absolute error to observation
    term1 = sum(abs(v - observed) for v in valid) / n

    # Second term: mean absolute pairwise difference
    term2 = 0.0
    for i in range(n):
        for j in range(n):
            term2 += abs(valid[i] - valid[j])
    term2 /= (2.0 * n * n)

    return round(term1 - term2, 4)


def compute_crps_from_percentiles(p10: float, p25: float, p50: float,
                                    p75: float, p90: float,
                                    observed: float) -> float | None:
    """Approximate CRPS from ensemble percentiles.

    When we don't have individual ensemble members, approximate CRPS
    from the quantile function using 5-point quadrature.

    Args:
        p10-p90: Ensemble percentiles.
        observed: Observed value.

    Returns:
        Approximate CRPS value.
    """
    if any(v is None for v in [p10, p25, p50, p75, p90]):
        return None

    # Approximate using quantile decomposition (5-point)
    quantiles = [(0.10, p10), (0.25, p25), (0.50, p50), (0.75, p75), (0.90, p90)]

    crps_approx = 0.0
    for tau, q_tau in quantiles:
        indicator = 1.0 if observed <= q_tau else 0.0
        crps_approx += (indicator - tau) ** 2

    # Scale by interval width (trapezoidal approximation)
    return round(crps_approx * 0.2, 4)


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

def compute_brier(probability: float, outcome: bool) -> float:
    """Compute Brier score for a binary prediction.

    Brier = (probability - outcome)^2
    Range: 0 (perfect) to 1 (worst).

    Args:
        probability: Predicted probability of the event (0 to 1).
        outcome: Whether the event occurred (True/False).

    Returns:
        Brier score (float).
    """
    probability = max(0.0, min(1.0, probability))
    outcome_val = 1.0 if outcome else 0.0
    return round((probability - outcome_val) ** 2, 4)


# ---------------------------------------------------------------------------
# Analog Storage
# ---------------------------------------------------------------------------

def store_analog(features: dict, observed: dict, analogs_path: str,
                 max_entries: int = 500) -> None:
    """Store a forecast-observation pair for ML training.

    Each analog entry contains feature values (model predictions, metadata)
    and the observed outcome. These accumulate over time for Phase 4 ML.

    Args:
        features: Dict with model predictions, spread, lead_time, day_of_year.
        observed: Dict with ERA5 observed values (snowfall, temp, wind).
        analogs_path: Path to analogs JSON file.
        max_entries: Maximum stored entries (rolling window).
    """
    analogs = []
    if os.path.exists(analogs_path):
        try:
            with open(analogs_path) as f:
                analogs = json.load(f)
        except (json.JSONDecodeError, OSError):
            analogs = []

    entry = {
        "features": features,
        "observed": observed,
    }
    analogs.append(entry)

    # Rolling window
    if len(analogs) > max_entries:
        analogs = analogs[-max_entries:]

    os.makedirs(os.path.dirname(analogs_path), exist_ok=True)
    with open(analogs_path, "w") as f:
        json.dump(analogs, f, indent=1)


def load_analogs(analogs_path: str) -> list:
    """Load accumulated analog pairs."""
    if not os.path.exists(analogs_path):
        return []
    try:
        with open(analogs_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
