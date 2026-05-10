"""
Metrics computation and evaluation for all data splits.

Computes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
for train, validation, and test splits. Also saves confusion matrix plots.
"""

import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    output_dir: str = "artifacts/plots",
) -> None:
    """
    Generate and save a confusion matrix heatmap for a given split.

    Args:
        y_true:     True target labels.
        y_pred:     Predicted labels (already threshold-applied).
        split_name: 'train' | 'validation' | 'test' — used in filename.
        output_dir: Directory to save the PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Potable", "Potable"],
        yticklabels=["Not Potable", "Potable"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {split_name.capitalize()} Split")
    plt.tight_layout()

    path = os.path.join(output_dir, f"confusion_matrix_{split_name}.png")
    try:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        logger.warning(f"Confusion matrix plot was not written: {exc}")
    plt.close(fig)
    logger.debug(f"Confusion matrix plot saved → {path}")


def evaluate_split(
    model: Any,
    X: Any,
    y: Any,
    split_name: str,
    threshold: float,
    plots_dir: str = "artifacts/plots",
) -> Dict[str, Any]:
    """
    Compute and return all evaluation metrics for a given data split.

    Metrics computed:
        - accuracy, precision, recall, f1, roc_auc
        - confusion_matrix (as a list-of-lists for JSON serialisability)

    Side effects:
        - Saves a confusion matrix PNG to ``plots_dir``.

    Args:
        model:      Trained XGBoost or sklearn model.
        X:          Feature matrix (preprocessed).
        y:          True labels.
        split_name: 'train' | 'validation' | 'test'.
        threshold:  Optimised decision threshold (applied to probabilities).
        plots_dir:  Directory to save confusion matrix plots.

    Returns:
        Dict with all computed metrics.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "split":            split_name,
        "accuracy":         float(accuracy_score(y, preds)),
        "precision":        float(precision_score(y, preds, zero_division=0)),
        "recall":           float(recall_score(y, preds, zero_division=0)),
        "f1":               float(f1_score(y, preds, zero_division=0)),
        "roc_auc":          float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }

    logger.info(f"[{split_name.upper()}] Evaluation @ threshold={threshold:.2f}:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        logger.info(f"  {metric}: {metrics[metric]:.4f}")
    logger.info(f"  confusion_matrix: {metrics['confusion_matrix']}")

    save_confusion_matrix_plot(y, preds, split_name, output_dir=plots_dir)
    return metrics
