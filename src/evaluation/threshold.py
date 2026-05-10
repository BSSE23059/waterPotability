"""
Balanced threshold selection for the Water Potability model.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.evaluation.scoring import find_best_threshold
from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def optimize_threshold(
    y_val: np.ndarray,
    val_proba: np.ndarray,
    output_csv: str = "artifacts/threshold_analysis.csv",
) -> Tuple[float, pd.DataFrame]:
    """
    Find the optimal decision threshold using the shared balanced rescue score.

    Args:
        y_val: True validation labels.
        val_proba: Predicted probabilities for the positive class on validation.
        output_csv: Path to save the full threshold analysis CSV.

    Returns:
        Tuple of (best_threshold float, results DataFrame with all thresholds).
    """
    best_threshold, best_metrics, results_df = find_best_threshold(
        y_val,
        val_proba,
        min_threshold=0.30,
        max_threshold=0.70,
        step=0.01,
    )

    logger.info(
        "Threshold optimisation complete - "
        f"optimal threshold: {best_threshold:.4f} "
        f"(balanced_score={best_metrics['balanced_score']:.4f}, "
        f"f1={best_metrics['f1']:.4f}, "
        f"roc_auc={best_metrics['roc_auc']:.4f})"
    )

    try:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        logger.debug(f"Threshold analysis saved -> {output_csv}")
    except OSError as exc:
        logger.warning(
            "Threshold analysis CSV was not written because the file is unavailable: %s",
            exc,
        )

    return best_threshold, results_df


def diagnose_probability_distribution(model, X_val, y_val):
    """
    Print the full probability distribution to understand
    what the model actually thinks before threshold is applied.
    """
    proba = model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 50)
    print("PROBABILITY DISTRIBUTION DIAGNOSIS")
    print("=" * 50)
    print(f"Min probability:    {proba.min():.4f}")
    print(f"Max probability:    {proba.max():.4f}")
    print(f"Mean probability:   {proba.mean():.4f}")
    print(f"Median probability: {np.median(proba):.4f}")
    print(f"Std probability:    {proba.std():.4f}")
    print()

    buckets = [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    print("Distribution buckets:")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = ((proba >= lo) & (proba < hi)).sum()
        pct = count / len(proba) * 100
        print(f"  [{lo:.2f} - {hi:.2f}): {count:4d} samples ({pct:.1f}%)")

    print()
    print(
        "Samples BELOW 0.50 threshold: "
        f"{(proba < 0.50).sum()} ({(proba < 0.50).mean() * 100:.1f}%)"
    )
    print(
        "Samples ABOVE 0.50 threshold: "
        f"{(proba >= 0.50).sum()} ({(proba >= 0.50).mean() * 100:.1f}%)"
    )

    y_arr = np.asarray(y_val)
    pos_proba = proba[y_arr == 1]
    neg_proba = proba[y_arr == 0]
    print(f"\nMean proba for TRUE positives: {pos_proba.mean():.4f}")
    print(f"Mean proba for TRUE negatives: {neg_proba.mean():.4f}")
    print(
        "Separation (delta):            "
        f"{(pos_proba.mean() - neg_proba.mean()):.4f}"
    )
    print("=" * 50)
