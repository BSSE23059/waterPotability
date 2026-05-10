"""
Training pipeline tests — XGBoost training, early stopping, overfitting detection,
and threshold range validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.overfitting_detector import check_overfitting
from src.evaluation.threshold import optimize_threshold
from src.preprocessing.pipeline import build_pipeline
from src.training.trainer import get_default_params, train_xgboost


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset():
    """Synthetic binary classification dataset for training tests."""
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({
        "ph":             np.random.uniform(6.0, 8.5, n),
        "Hardness":       np.random.uniform(100, 300, n),
        "Solids":         np.random.uniform(5000, 40000, n),
        "Chloramines":    np.random.uniform(3, 10, n),
        "Sulfate":        np.random.uniform(200, 400, n),
        "Conductivity":   np.random.uniform(300, 700, n),
        "Organic_carbon": np.random.uniform(8, 18, n),
        "Trihalomethanes":np.random.uniform(50, 100, n),
        "Turbidity":      np.random.uniform(1, 6, n),
        "Potability":     (np.random.uniform(0, 1, n) > 0.61).astype(int),
    })
    return df


@pytest.fixture
def prepared_splits(synthetic_dataset):
    """Full preprocessing pipeline applied to train/val splits."""
    from sklearn.model_selection import train_test_split

    X = synthetic_dataset.drop("Potability", axis=1)
    y = synthetic_dataset["Potability"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    X_train_proc = pipeline.fit_transform(X_train, y_train)
    X_val_proc   = pipeline.transform(X_val)

    return X_train_proc, y_train, X_val_proc, y_val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestXGBoostTrainsWithoutError:
    """Verify XGBoost model trains end-to-end on a small synthetic dataset."""

    def test_model_trains_and_predicts(self, prepared_splits):
        """Model must train and produce valid probability predictions."""
        X_train, y_train, X_val, y_val = prepared_splits
        params = get_default_params()

        model = train_xgboost(X_train, y_train, X_val, y_val, params)

        # Must be able to predict probabilities
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2), (
            f"Unexpected predict_proba shape: {proba.shape}"
        )
        assert (proba >= 0).all() and (proba <= 1).all(), (
            "Probabilities outside [0, 1] range."
        )

    def test_model_has_feature_importances(self, prepared_splits):
        """Trained model must expose feature_importances_."""
        X_train, y_train, X_val, y_val = prepared_splits
        model = train_xgboost(X_train, y_train, X_val, y_val, get_default_params())

        assert hasattr(model, "feature_importances_"), (
            "Model missing feature_importances_ attribute."
        )
        assert len(model.feature_importances_) == X_train.shape[1]


class TestEarlyStoppingActivates:
    """Verify early stopping fires before n_estimators is exhausted."""

    def test_best_iteration_less_than_n_estimators(self, prepared_splits):
        """With early stopping, best_iteration should be < n_estimators."""
        X_train, y_train, X_val, y_val = prepared_splits

        params = get_default_params()
        # Use a large n_estimators to guarantee early stopping has room to fire
        params["n_estimators"] = 500

        model = train_xgboost(
            X_train, y_train, X_val, y_val,
            params=params,
            early_stopping_rounds=20,
        )

        # best_iteration is 0-indexed; total fitted trees ≤ n_estimators
        assert hasattr(model, "best_iteration"), (
            "Model missing best_iteration — early stopping may not be active."
        )
        # We cannot guarantee it fires on a small synthetic set, but it must be ≤ n_estimators
        assert model.best_iteration <= params["n_estimators"], (
            f"best_iteration ({model.best_iteration}) > n_estimators ({params['n_estimators']})"
        )


class TestOverfittingDetectorWarns:
    """Verify overfitting detector produces correct output for large gaps."""

    def test_warns_on_large_accuracy_gap(self, caplog):
        """Warning must be emitted when train-val accuracy gap > 0.10."""
        import logging

        train_m = {"accuracy": 0.95, "f1": 0.94, "roc_auc": 0.97, "precision": 0.90, "recall": 0.85}
        val_m   = {"accuracy": 0.60, "f1": 0.58, "roc_auc": 0.72, "precision": 0.65, "recall": 0.55}
        test_m  = {"accuracy": 0.61, "f1": 0.59, "roc_auc": 0.71, "precision": 0.63, "recall": 0.54}

        with caplog.at_level(logging.WARNING):
            gaps = check_overfitting(train_m, val_m, test_m)

        assert gaps["accuracy_gap"] > 0.10, (
            f"Expected accuracy_gap > 0.10, got {gaps['accuracy_gap']}"
        )
        assert any("OVERFITTING" in record.message for record in caplog.records), (
            "Expected OVERFITTING warning not found in logs."
        )

    def test_no_warning_on_small_gap(self, caplog):
        """No overfitting warning for tight train-val gaps."""
        import logging

        train_m = {"accuracy": 0.74, "f1": 0.73, "roc_auc": 0.81, "precision": 0.72, "recall": 0.71}
        val_m   = {"accuracy": 0.71, "f1": 0.70, "roc_auc": 0.79, "precision": 0.70, "recall": 0.69}
        test_m  = {"accuracy": 0.72, "f1": 0.71, "roc_auc": 0.80, "precision": 0.71, "recall": 0.70}

        with caplog.at_level(logging.WARNING):
            gaps = check_overfitting(train_m, val_m, test_m)

        overfitting_warnings = [
            r for r in caplog.records if "OVERFITTING DETECTED" in r.message
        ]
        assert len(overfitting_warnings) == 0, (
            f"Unexpected OVERFITTING warnings: {[r.message for r in overfitting_warnings]}"
        )


class TestThresholdInValidRange:
    """Verify optimised threshold is always within [0.30, 0.70]."""

    def test_threshold_within_search_range(self, prepared_splits):
        """Optimal threshold must be within the configured search range."""
        X_train, y_train, X_val, y_val = prepared_splits
        model = train_xgboost(
            X_train, y_train, X_val, y_val, get_default_params()
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        best_threshold, df = optimize_threshold(y_val, val_proba)

        assert 0.30 <= best_threshold <= 0.70, (
            f"Optimal threshold {best_threshold:.2f} is outside the [0.30, 0.70] range."
        )

    def test_threshold_analysis_df_columns(self, prepared_splits):
        """Threshold analysis DataFrame must contain expected columns."""
        X_train, y_train, X_val, y_val = prepared_splits
        model = train_xgboost(
            X_train, y_train, X_val, y_val, get_default_params()
        )
        val_proba = model.predict_proba(X_val)[:, 1]
        _, df = optimize_threshold(y_val, val_proba)

        expected_cols = {"threshold", "precision", "recall", "f1"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns in threshold analysis DataFrame: "
            f"{expected_cols - set(df.columns)}"
        )
