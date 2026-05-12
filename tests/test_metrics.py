"""
Metrics computation tests — evaluate_split correctness and output structure.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate_split
from src.evaluation.threshold import optimize_threshold


class TestEvaluateSplitOutputStructure:
    """Verify evaluate_split returns correct keys and value ranges."""

    def test_returns_all_required_keys(self, tmp_path):
        """evaluate_split must return all 7 required keys."""
        # Create dummy model
        class DummyModel:
            def predict_proba(self, X):
                n = len(X) if hasattr(X, '__len__') else X.shape[0]
                proba = np.random.uniform(0.2, 0.8, n)
                return np.column_stack([1 - proba, proba])

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)

        model = DummyModel()
        metrics = evaluate_split(model, X, y, "test", threshold=0.5, plots_dir=str(tmp_path))

        required_keys = {"split", "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}
        assert required_keys.issubset(set(metrics.keys())), (
            f"Missing keys: {required_keys - set(metrics.keys())}"
        )

    def test_metrics_in_valid_ranges(self, tmp_path):
        """All metric values must be in [0, 1]."""
        class DummyModel:
            def predict_proba(self, X):
                n = X.shape[0]
                proba = np.random.uniform(0.2, 0.8, n)
                return np.column_stack([1 - proba, proba])

        np.random.seed(0)
        X = np.random.randn(150, 5)
        y = np.random.randint(0, 2, 150)

        model = DummyModel()
        metrics = evaluate_split(model, X, y, "validation", threshold=0.5, plots_dir=str(tmp_path))

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            val = metrics[metric]
            assert 0.0 <= val <= 1.0, (
                f"Metric '{metric}' = {val} is outside [0, 1]."
            )

    def test_confusion_matrix_shape(self, tmp_path):
        """Confusion matrix must be 2×2 for binary classification."""
        class DummyModel:
            def predict_proba(self, X):
                n = X.shape[0]
                proba = np.where(np.arange(n) % 2 == 0, 0.7, 0.3)
                return np.column_stack([1 - proba, proba])

        X = np.random.randn(80, 5)
        y = np.array([0, 1] * 40)

        model = DummyModel()
        metrics = evaluate_split(model, X, y, "train", threshold=0.5, plots_dir=str(tmp_path))

        cm = metrics["confusion_matrix"]
        assert len(cm) == 2, f"Expected 2 rows in confusion matrix, got {len(cm)}"
        assert len(cm[0]) == 2, f"Expected 2 columns in confusion matrix, got {len(cm[0])}"


class TestOptimizeThreshold:
    """Verify threshold optimisation returns valid results."""

    def test_threshold_in_search_range(self):
        """Returned threshold must be within the specified search range."""
        np.random.seed(1)
        y_val = np.random.randint(0, 2, 200)
        val_proba = np.random.uniform(0.2, 0.8, 200)

        threshold, _ = optimize_threshold(y_val, val_proba)

        assert 0.30 <= threshold <= 0.85, (
            f"Threshold {threshold:.2f} outside [0.30, 0.85]."
        )

    def test_results_dataframe_non_empty(self):
        """Results DataFrame must have one row per threshold candidate."""
        np.random.seed(2)
        y_val = np.random.randint(0, 2, 200)
        val_proba = np.random.uniform(0.25, 0.75, 200)

        _, df = optimize_threshold(y_val, val_proba)

        assert len(df) > 0, f"Expected at least 1 row in threshold analysis, got {len(df)}"
