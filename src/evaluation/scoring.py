"""
Shared precision-first scoring utilities.

For water potability, a class-1 false positive is the highest-risk error:
predicting water is potable when it is actually not potable. Threshold
selection therefore starts with validation precision, then keeps recall/F1 and
generalization gaps from collapsing.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.85
THRESHOLD_STEP = 0.01

MIN_PRECISION_FIRST_F1 = 0.50
MIN_USEFUL_RECALL = 0.30
MIN_PREDICTED_POSITIVE_RATE = 0.10
MIN_PREDICTED_POSITIVES = 30
REFERENCE_FALSE_POSITIVE_THRESHOLD = 0.37
MIN_FALSE_POSITIVE_REDUCTION = 0.15
IDEAL_GENERALIZATION_GAP = 0.10
MAX_GENERALIZATION_GAP = 0.12


def _as_array(values: Any) -> np.ndarray:
    """Return a flat numpy array without assuming pandas input."""
    return np.asarray(values).ravel()


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return ROC-AUC, falling back to 0.5 when a split is degenerate."""
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def get_positive_scores(model: Any, X: Any) -> np.ndarray:
    """
    Return positive-class scores from a fitted classifier.

    Probabilities are preferred. A decision_function fallback is converted with
    a sigmoid only for compatibility with classifiers that do not expose
    predict_proba.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = _as_array(model.decision_function(X))
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        raise AttributeError(
            f"{model.__class__.__name__} does not expose predict_proba or decision_function."
        )
    return _as_array(scores).astype(float)


def metrics_at_threshold(
    y_true: Any,
    proba: Any,
    threshold: float,
    target_positive_rate: float | None = None,
) -> Dict[str, float]:
    """Compute binary metrics and confusion-matrix counts at one threshold."""
    y_arr = _as_array(y_true).astype(int)
    p_arr = _as_array(proba).astype(float)
    preds = (p_arr >= threshold).astype(int)

    positive_rate = float(preds.mean())
    if target_positive_rate is None:
        target_positive_rate = float(y_arr.mean())

    cm = confusion_matrix(y_arr, preds, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]

    accuracy = float(accuracy_score(y_arr, preds))
    precision = float(precision_score(y_arr, preds, zero_division=0))
    recall = float(recall_score(y_arr, preds, zero_division=0))
    f1 = float(f1_score(y_arr, preds, zero_division=0))
    roc_auc = _safe_roc_auc(y_arr, p_arr)
    positive_rate_penalty = abs(positive_rate - float(target_positive_rate))

    balanced_score = (
        0.35 * accuracy
        + 0.35 * f1
        + 0.20 * roc_auc
        + 0.10 * min(precision, recall)
        - 0.10 * positive_rate_penalty
    )
    precision_compromise_score = 0.70 * precision + 0.20 * f1 + 0.10 * roc_auc

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
        "precision_compromise_score": float(precision_compromise_score),
        "predicted_positives": int(preds.sum()),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
    }


def threshold_sweep(
    y_true: Any,
    proba: Any,
    min_threshold: float = THRESHOLD_MIN,
    max_threshold: float = THRESHOLD_MAX,
    step: float = THRESHOLD_STEP,
    target_positive_rate: float | None = None,
) -> pd.DataFrame:
    """Return metrics for every threshold in the configured search range."""
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
    return pd.DataFrame(rows)


def find_best_threshold(
    y_true: Any,
    proba: Any,
    min_threshold: float = THRESHOLD_MIN,
    max_threshold: float = THRESHOLD_MAX,
    step: float = THRESHOLD_STEP,
    target_positive_rate: float | None = None,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Backward-compatible threshold search using the precision-first policy."""
    return select_precision_threshold(
        y_true,
        proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
        target_positive_rate=target_positive_rate,
    )


def _add_train_val_gap_columns(
    results: pd.DataFrame,
    train_y_true: Any | None,
    train_proba: Any | None,
    min_threshold: float,
    max_threshold: float,
    step: float,
    target_positive_rate: float | None,
) -> pd.DataFrame:
    """Attach train-validation gap columns for threshold tie-breaking."""
    results = results.copy()
    if train_y_true is None or train_proba is None:
        results["train_val_precision_gap"] = 0.0
        results["train_val_f1_gap"] = 0.0
        results["train_val_roc_auc_gap"] = 0.0
        return results

    train_df = threshold_sweep(
        train_y_true,
        train_proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
        target_positive_rate=target_positive_rate,
    )
    train_lookup = train_df.set_index("threshold")
    results["train_precision"] = results["threshold"].map(train_lookup["precision"])
    results["train_f1"] = results["threshold"].map(train_lookup["f1"])
    results["train_roc_auc"] = results["threshold"].map(train_lookup["roc_auc"])
    results["train_val_precision_gap"] = (
        results["train_precision"] - results["precision"]
    ).clip(lower=0)
    results["train_val_f1_gap"] = (results["train_f1"] - results["f1"]).clip(lower=0)
    results["train_val_roc_auc_gap"] = (
        results["train_roc_auc"] - results["roc_auc"]
    ).clip(lower=0)
    return results


def select_precision_threshold(
    y_true: Any,
    proba: Any,
    train_y_true: Any | None = None,
    train_proba: Any | None = None,
    min_threshold: float = THRESHOLD_MIN,
    max_threshold: float = THRESHOLD_MAX,
    step: float = THRESHOLD_STEP,
    target_positive_rate: float | None = None,
    min_f1: float = MIN_PRECISION_FIRST_F1,
    min_recall: float = MIN_USEFUL_RECALL,
    min_predicted_positive_rate: float = MIN_PREDICTED_POSITIVE_RATE,
    min_predicted_positives: int = MIN_PREDICTED_POSITIVES,
    reference_threshold: float = REFERENCE_FALSE_POSITIVE_THRESHOLD,
    min_false_positive_reduction: float = MIN_FALSE_POSITIVE_REDUCTION,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Select a validation threshold with precision as the first priority.

    A threshold must remain useful: F1 and recall cannot collapse, and it must
    still predict a non-trivial number of positives. When possible, it must also
    reduce validation false positives versus the old 0.37-style threshold.
    """
    results = threshold_sweep(
        y_true,
        proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
        target_positive_rate=target_positive_rate,
    )
    results = _add_train_val_gap_columns(
        results,
        train_y_true,
        train_proba,
        min_threshold,
        max_threshold,
        step,
        target_positive_rate,
    )

    y_arr = _as_array(y_true)
    min_positive_count = max(
        int(min_predicted_positives),
        int(np.ceil(len(y_arr) * min_predicted_positive_rate)),
    )

    reference_metrics = metrics_at_threshold(y_true, proba, reference_threshold)
    target_false_positives = reference_metrics["false_positives"] * (
        1.0 - min_false_positive_reduction
    )
    results["reference_false_positives"] = reference_metrics["false_positives"]
    results["false_positive_reduction"] = (
        reference_metrics["false_positives"] - results["false_positives"]
    )
    results["false_positive_reduction_rate"] = (
        results["false_positive_reduction"]
        / max(1, reference_metrics["false_positives"])
    )

    useful = results[
        (results["f1"] >= min_f1)
        & (results["recall"] >= min_recall)
        & (results["predicted_positives"] >= min_positive_count)
    ].copy()
    strict = useful[useful["false_positives"] <= target_false_positives].copy()

    if not strict.empty:
        candidates = strict
        reason = "highest_validation_precision_with_fp_reduction_and_useful_f1"
    elif not useful.empty:
        candidates = useful
        reason = "highest_validation_precision_with_useful_f1_no_fp_reduction_candidate"
    else:
        candidates = results[
            (results["recall"] >= min_recall)
            & (results["predicted_positives"] >= min_positive_count)
        ].copy()
        if candidates.empty:
            candidates = results.copy()
        reason = "precision_compromise_because_no_threshold_met_useful_f1_floor"

    candidates = candidates.sort_values(
        by=[
            "precision",
            "f1",
            "false_positives",
            "train_val_precision_gap",
            "train_val_f1_gap",
            "train_val_roc_auc_gap",
            "roc_auc",
            "threshold",
        ],
        ascending=[False, False, True, True, True, True, False, False],
    )

    best_row = candidates.iloc[0].to_dict()
    best_row["threshold_selection_reason"] = reason
    best_row["min_f1_floor"] = min_f1
    best_row["min_recall_floor"] = min_recall
    best_row["min_predicted_positives_floor"] = min_positive_count
    return float(best_row["threshold"]), best_row, results


def precision_focused_selection_score(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    min_f1: float = MIN_PRECISION_FIRST_F1,
    min_recall: float = MIN_USEFUL_RECALL,
    max_gap: float = MAX_GENERALIZATION_GAP,
) -> Tuple[float, Dict[str, float]]:
    """Compute the validation precision-focused model-selection score."""
    base_score = (
        0.65 * val_metrics["precision"]
        + 0.20 * val_metrics["f1"]
        + 0.10 * val_metrics["roc_auc"]
        + 0.05 * val_metrics["accuracy"]
    )

    precision_gap = max(0.0, train_metrics["precision"] - val_metrics["precision"])
    f1_gap = max(0.0, train_metrics["f1"] - val_metrics["f1"])
    auc_gap = max(0.0, train_metrics["roc_auc"] - val_metrics["roc_auc"])

    penalties = {
        "val_f1_penalty": 0.60 * max(0.0, min_f1 - val_metrics["f1"]),
        "val_recall_penalty": 0.60 * max(0.0, min_recall - val_metrics["recall"]),
        "precision_gap_penalty": 0.60 * max(0.0, precision_gap - max_gap),
        "f1_gap_penalty": 0.80 * max(0.0, f1_gap - max_gap),
        "roc_auc_gap_penalty": 0.80 * max(0.0, auc_gap - max_gap),
        "ideal_precision_gap_penalty": 0.15
        * max(0.0, precision_gap - IDEAL_GENERALIZATION_GAP),
        "ideal_f1_gap_penalty": 0.20
        * max(0.0, f1_gap - IDEAL_GENERALIZATION_GAP),
        "ideal_roc_auc_gap_penalty": 0.20
        * max(0.0, auc_gap - IDEAL_GENERALIZATION_GAP),
    }
    total_penalty = float(sum(penalties.values()))
    final_score = float(base_score - total_penalty)

    details = {
        "base_selection_score": float(base_score),
        "final_selection_score": final_score,
        "selection_score": final_score,
        "total_penalty": total_penalty,
        "train_val_precision_gap": float(precision_gap),
        "train_val_f1_gap": float(f1_gap),
        "train_val_roc_auc_gap": float(auc_gap),
        **{key: float(value) for key, value in penalties.items()},
    }
    return final_score, details
