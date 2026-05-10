"""
Shared scoring utilities for threshold selection and model selection.

The rescue target is balanced generalisation: keep accuracy, F1, ROC-AUC,
precision, and recall in play instead of optimising a single fragile metric.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_array(values: Any) -> np.ndarray:
    """Return a flat numpy array without assuming pandas input."""
    return np.asarray(values).ravel()


def metrics_at_threshold(
    y_true: Any,
    proba: Any,
    threshold: float,
    target_positive_rate: float | None = None,
) -> Dict[str, float]:
    """
    Compute binary metrics and the balanced rescue score at one threshold.

    The score is:
        0.35 * accuracy
      + 0.35 * f1
      + 0.20 * roc_auc
      + 0.10 * min(precision, recall)
      - 0.10 * abs(predicted_positive_rate - target_positive_rate)
    """
    y_arr = _as_array(y_true)
    p_arr = _as_array(proba)
    preds = (p_arr >= threshold).astype(int)

    positive_rate = float(preds.mean())
    if target_positive_rate is None:
        target_positive_rate = float(y_arr.mean())

    accuracy = float(accuracy_score(y_arr, preds))
    precision = float(precision_score(y_arr, preds, zero_division=0))
    recall = float(recall_score(y_arr, preds, zero_division=0))
    f1 = float(f1_score(y_arr, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_arr, p_arr))
    positive_rate_penalty = abs(positive_rate - float(target_positive_rate))

    balanced_score = (
        0.35 * accuracy
        + 0.35 * f1
        + 0.20 * roc_auc
        + 0.10 * min(precision, recall)
        - 0.10 * positive_rate_penalty
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "predicted_positive_rate": positive_rate,
        "target_positive_rate": float(target_positive_rate),
        "positive_rate_penalty": float(positive_rate_penalty),
        "balanced_score": float(balanced_score),
    }


def find_best_threshold(
    y_true: Any,
    proba: Any,
    min_threshold: float = 0.30,
    max_threshold: float = 0.70,
    step: float = 0.01,
    target_positive_rate: float | None = None,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Search a threshold range and return the best balanced-score threshold.
    """
    thresholds = np.arange(min_threshold, max_threshold + (step / 2), step)
    rows = [
        metrics_at_threshold(
            y_true,
            proba,
            float(round(threshold, 4)),
            target_positive_rate=target_positive_rate,
        )
        for threshold in thresholds
    ]
    results = pd.DataFrame(rows)
    best_idx = int(results["balanced_score"].idxmax())
    best_row = results.loc[best_idx].to_dict()
    return float(best_row["threshold"]), best_row, results
