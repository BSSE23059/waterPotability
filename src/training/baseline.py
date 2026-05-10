"""
Random Forest baseline model training for the Water Potability Prediction System.

Trains a regularised Random Forest as a performance baseline before XGBoost tuning.
XGBoost must outperform this baseline on the validation set to justify its added
complexity and longer training time.
"""

from typing import Dict, Tuple

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a Random Forest baseline model and evaluate on the validation set.

    The baseline uses conservative hyperparameters chosen to avoid severe
    overfitting:
        n_estimators=100   — sufficient trees for stable predictions
        max_depth=6        — moderate depth to prevent memorisation
        min_samples_split=10 — requires substantial support to split a node
        min_samples_leaf=5   — requires substantial support in leaf nodes
        class_weight='balanced' — compensates for class imbalance natively

    Results are logged to MLflow under run name "RandomForest_Baseline".

    Args:
        X_train: Preprocessed + SMOTE-resampled training features.
        y_train: Resampled training labels.
        X_val:   Preprocessed validation features (NOT resampled).
        y_val:   Validation labels (original distribution).

    Returns:
        Tuple of (trained RandomForestClassifier, val_metrics dict).
    """
    logger.info("Training Random Forest baseline model...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest baseline training complete.")

    val_proba = rf.predict_proba(X_val)[:, 1]
    val_preds = rf.predict(X_val)

    val_metrics = {
        "accuracy":  float(accuracy_score(y_val, val_preds)),
        "precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "recall":    float(recall_score(y_val, val_preds, zero_division=0)),
        "f1":        float(f1_score(y_val, val_preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_val, val_proba)),
    }

    logger.info("Baseline validation metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    try:
        with mlflow.start_run(run_name="RandomForest_Baseline", nested=True):
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
            }
            mlflow.log_params(params)
            for metric_name, metric_val in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_val)
            mlflow.sklearn.log_model(rf, "random_forest_baseline")
    except Exception as exc:
        logger.warning(f"Skipping baseline MLflow logging: {exc}")

    return rf, val_metrics
