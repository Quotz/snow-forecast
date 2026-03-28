"""Machine learning post-processing for snowfall prediction.

Trains a lightweight Random Forest on accumulated analog pairs (forecast-observation)
to learn a meta-model that outperforms simple weighted averaging. Requires scikit-learn
and at least 30 analog pairs before activation.

Training runs weekly during recalibration. Model serialized via joblib.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

MIN_ANALOGS = 30  # Minimum training samples


def should_use_ml(analogs_path: str) -> bool:
    """Check if enough analog data exists for ML training."""
    if not os.path.exists(analogs_path):
        return False
    try:
        with open(analogs_path) as f:
            analogs = json.load(f)
        return len(analogs) >= MIN_ANALOGS
    except (json.JSONDecodeError, OSError):
        return False


def train_model(analogs_path: str, model_path: str) -> dict | None:
    """Train a Random Forest on accumulated analog pairs.

    Features (per analog):
        - Individual model predictions (up to 7)
        - Mean of model predictions
        - Spread (max - min)
        - Lead time (days)
        - Day of year (1-366)

    Target: ERA5 observed snowfall.

    Args:
        analogs_path: Path to analogs.json with training data.
        model_path: Path to save trained model (.pkl).

    Returns:
        Dict with training metrics, or None on failure.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        import joblib
    except ImportError:
        logger.warning("scikit-learn not installed, skipping ML training")
        return None

    if not os.path.exists(analogs_path):
        return None

    try:
        with open(analogs_path) as f:
            analogs = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if len(analogs) < MIN_ANALOGS:
        logger.info("Only %d analogs (need %d), skipping ML training", len(analogs), MIN_ANALOGS)
        return None

    # Build feature matrix and target vector
    X = []
    y = []

    for entry in analogs:
        features = entry.get("features", {})
        observed = entry.get("observed", {})

        obs_snow = observed.get("snowfall")
        if obs_snow is None:
            continue

        # Model predictions (pad to 7 if fewer)
        preds = features.get("model_predictions", [])
        preds_clean = [p if p is not None else 0 for p in preds]
        while len(preds_clean) < 7:
            preds_clean.append(0)
        preds_clean = preds_clean[:7]

        row = preds_clean + [
            features.get("mean", 0),
            features.get("spread", 0),
            features.get("lead_time", 3),
            features.get("day_of_year", 180),
        ]
        X.append(row)
        y.append(obs_snow)

    if len(X) < MIN_ANALOGS:
        return None

    logger.info("Training ML model on %d analog pairs", len(X))

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # Cross-validation score (negative MAE)
    cv_scores = cross_val_score(rf, X, y, cv=min(5, len(X) // 5), scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(rf, model_path)

    result = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": len(X),
        "cv_mae": round(cv_mae, 3),
        "feature_importances": {
            f"model_{i}": round(imp, 4) for i, imp in enumerate(rf.feature_importances_[:7])
        },
    }
    result["feature_importances"]["mean"] = round(rf.feature_importances_[7], 4)
    result["feature_importances"]["spread"] = round(rf.feature_importances_[8], 4)
    result["feature_importances"]["lead_time"] = round(rf.feature_importances_[9], 4)
    result["feature_importances"]["day_of_year"] = round(rf.feature_importances_[10], 4)

    logger.info("ML model trained: CV MAE=%.2f cm, %d samples", cv_mae, len(X))
    return result


def predict(features: dict, model_path: str) -> float | None:
    """Make an ML-corrected snowfall prediction.

    Args:
        features: Dict with model_predictions, mean, spread, lead_time, day_of_year.
        model_path: Path to trained model (.pkl).

    Returns:
        Predicted snowfall (cm), or None if model unavailable.
    """
    if not os.path.exists(model_path):
        return None

    try:
        import joblib
    except ImportError:
        return None

    try:
        rf = joblib.load(model_path)
    except Exception:
        return None

    preds = features.get("model_predictions", [])
    preds_clean = [p if p is not None else 0 for p in preds]
    while len(preds_clean) < 7:
        preds_clean.append(0)
    preds_clean = preds_clean[:7]

    row = preds_clean + [
        features.get("mean", 0),
        features.get("spread", 0),
        features.get("lead_time", 3),
        features.get("day_of_year", 180),
    ]

    try:
        prediction = rf.predict([row])[0]
        return max(0, round(prediction, 2))
    except Exception:
        return None
