"""Self-learning ML post-processing for all forecast variables.

Trains separate Random Forest models for snowfall, temperature, wind, cloud
cover, and sunshine from accumulated analog pairs. Retrains automatically
during weekly recalibration or when enough new data accumulates.

Each model learns which NWP model combinations predict best for Popova Shapka
for each variable independently.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

MIN_ANALOGS = 30  # Minimum training samples per target

# All targets we train models for
TARGETS = {
    "snowfall": {"unit": "cm", "min_val": 0},
    "temperature_max": {"unit": "°C", "min_val": None},
    "temperature_min": {"unit": "°C", "min_val": None},
    "wind_max": {"unit": "km/h", "min_val": 0},
    "sunshine_hours": {"unit": "h", "min_val": 0},
}


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


def _build_features(entry: dict) -> list | None:
    """Build feature vector from an analog entry.

    Features (11 total):
        0-6: Individual model predictions (padded to 7)
        7:   Mean of model predictions
        8:   Spread (max - min)
        9:   Lead time (days)
        10:  Day of year (1-366)
    """
    features = entry.get("features", {})
    preds = features.get("model_predictions", [])
    preds_clean = [p if p is not None else 0 for p in preds]
    while len(preds_clean) < 7:
        preds_clean.append(0)
    preds_clean = preds_clean[:7]

    valid = [p for p in preds if p is not None]
    if not valid:
        return None

    return preds_clean + [
        features.get("mean", sum(valid) / len(valid)),
        features.get("spread", max(valid) - min(valid) if len(valid) > 1 else 0),
        features.get("lead_time", 3),
        features.get("day_of_year", 180),
    ]


def train_all_models(analogs_path: str, models_dir: str) -> dict:
    """Train ML models for all forecast targets.

    Trains a separate Random Forest for each target variable (snowfall,
    temperature, wind, sunshine). Each model learns independently which
    NWP combinations work best for that variable.

    Args:
        analogs_path: Path to analogs.json.
        models_dir: Directory to save trained models (e.g., docs/verification/).

    Returns:
        Dict with training results per target.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        import joblib
    except ImportError:
        logger.warning("scikit-learn not installed, skipping ML training")
        return {}

    if not os.path.exists(analogs_path):
        return {}

    try:
        with open(analogs_path) as f:
            analogs = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if len(analogs) < MIN_ANALOGS:
        logger.info("Only %d analogs (need %d), skipping ML training", len(analogs), MIN_ANALOGS)
        return {}

    os.makedirs(models_dir, exist_ok=True)
    results = {}

    for target_name, target_cfg in TARGETS.items():
        X = []
        y = []

        for entry in analogs:
            row = _build_features(entry)
            if row is None:
                continue

            observed = entry.get("observed", {})
            obs_val = observed.get(target_name)
            if obs_val is None:
                continue

            X.append(row)
            y.append(obs_val)

        if len(X) < MIN_ANALOGS:
            logger.info("Target %s: only %d samples (need %d), skipping",
                        target_name, len(X), MIN_ANALOGS)
            continue

        logger.info("Training %s model on %d samples", target_name, len(X))

        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)

        n_folds = min(5, max(2, len(X) // 10))
        cv_scores = cross_val_score(rf, X, y, cv=n_folds,
                                     scoring="neg_mean_absolute_error")
        cv_mae = -cv_scores.mean()

        model_path = os.path.join(models_dir, f"ml_model_{target_name}.pkl")
        joblib.dump(rf, model_path)

        # Feature importances
        feat_names = [f"model_{i}" for i in range(7)] + ["mean", "spread", "lead_time", "day_of_year"]
        importances = {feat_names[i]: round(imp, 4)
                       for i, imp in enumerate(rf.feature_importances_)}

        results[target_name] = {
            "cv_mae": round(cv_mae, 3),
            "n_samples": len(X),
            "model_path": model_path,
            "importances": importances,
            "unit": target_cfg["unit"],
        }

        logger.info("  %s: CV MAE = %.2f %s (%d samples)",
                     target_name, cv_mae, target_cfg["unit"], len(X))

    # Also save the old single-model format for backward compat
    if "snowfall" in results:
        compat_path = os.path.join(models_dir, "ml_model.pkl")
        snow_path = results["snowfall"]["model_path"]
        if os.path.exists(snow_path):
            import shutil
            shutil.copy2(snow_path, compat_path)

    # Save training report
    report = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_analogs": len(analogs),
        "targets": {k: {kk: vv for kk, vv in v.items() if kk != "model_path"}
                    for k, v in results.items()},
    }
    report_path = os.path.join(models_dir, "ml_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Trained %d ML models", len(results))
    return results


# Keep backward-compat single-target functions
def train_model(analogs_path: str, model_path: str) -> dict | None:
    """Train snowfall model only (backward compatible)."""
    models_dir = os.path.dirname(model_path)
    results = train_all_models(analogs_path, models_dir)
    return results.get("snowfall")


def predict(features: dict, model_path: str) -> float | None:
    """Predict snowfall using trained ML model."""
    return predict_target("snowfall", features, os.path.dirname(model_path))


def predict_target(target_name: str, features: dict, models_dir: str) -> float | None:
    """Predict a specific target variable using the trained ML model.

    Args:
        target_name: One of TARGETS keys (snowfall, temperature_max, etc.)
        features: Dict with model_predictions, mean, spread, lead_time, day_of_year.
        models_dir: Directory containing trained models.

    Returns:
        Predicted value, or None if model unavailable.
    """
    model_path = os.path.join(models_dir, f"ml_model_{target_name}.pkl")
    if not os.path.exists(model_path):
        # Fallback to old single-model format for snowfall
        if target_name == "snowfall":
            model_path = os.path.join(models_dir, "ml_model.pkl")
            if not os.path.exists(model_path):
                return None
        else:
            return None

    try:
        import joblib
    except ImportError:
        return None

    try:
        rf = joblib.load(model_path)
    except Exception:
        return None

    entry = {"features": features}
    row = _build_features(entry)
    if row is None:
        return None

    try:
        prediction = rf.predict([row])[0]
        min_val = TARGETS.get(target_name, {}).get("min_val")
        if min_val is not None:
            prediction = max(min_val, prediction)
        return round(prediction, 2)
    except Exception:
        return None
