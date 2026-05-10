"""
Full preprocessing pipeline assembly for the Water Potability Prediction System.

The rescue training path defaults to imputation plus robust scaling only.
Interaction feature engineering is preserved as an opt-in experiment because the
current holdout diagnostics showed those interactions can increase overfit risk.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.preprocessing.feature_engineer import (
    ENGINEERED_FEATURE_NAMES,
    engineer_features,
)
from src.preprocessing.feature_selector import build_selector, fit_selector
from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Append binary missingness flags before imputation.

    XGBoost can learn missing directions natively, but sklearn models cannot.
    These flags preserve the fact that pH, Sulfate, and Trihalomethanes were
    missing before median imputation replaces their values.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns_ = [col for col in X.columns if X[col].isna().any()]
        else:
            self.columns_ = [col for col in self.columns if col in X.columns]
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.columns_:
            X_out[f"{col}_missing"] = X[col].isna().astype(float)
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        return np.array(
            list(input_features) + [f"{col}_missing" for col in self.columns_]
        )


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around engineer_features.
    Stateless: no fitting required, always safe to apply to any split.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            raise ValueError("FeatureEngineerTransformer requires a pandas DataFrame.")
        return engineer_features(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        return np.array(list(input_features) + ENGINEERED_FEATURE_NAMES)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Sklearn imputer wrapper that preserves DataFrame column names.
    """

    def __init__(self):
        self.strategy = "median"

    def fit(self, X, y=None):
        from sklearn.impute import SimpleImputer

        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X)

        logger.info("DataFrameImputer fitted - medians learned:")
        for col, median_val in zip(self.columns_, self.imputer_.statistics_):
            logger.info(f"  {col}: {median_val:.4f}")

        return self

    def transform(self, X):
        arr = self.imputer_.transform(X)
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    RobustScaler wrapper that preserves DataFrame column names.
    """

    def fit(self, X, y=None):
        from sklearn.preprocessing import RobustScaler

        self.columns_ = list(X.columns)
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X)

        logger.info("DataFrameScaler (RobustScaler) fitted on training data.")
        return self

    def transform(self, X):
        arr = self.scaler_.transform(X)
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    SelectFromModel wrapper that preserves feature names after selection.

    This transformer is kept for experiments, but the rescue path does not add
    it to build_pipeline by default.
    """

    def fit(self, X, y=None):
        self.selector_ = build_selector()
        self.selector_ = fit_selector(self.selector_, X, y)

        mask = self.selector_.get_support()
        cols = list(X.columns) if hasattr(X, "columns") else [
            f"f{i}" for i in range(X.shape[1])
        ]

        self.selected_columns_ = [c for c, m in zip(cols, mask) if m]
        return self

    def transform(self, X):
        arr = self.selector_.transform(X)
        index = X.index if hasattr(X, "index") else None
        return pd.DataFrame(arr, columns=self.selected_columns_, index=index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_columns_)


def build_pipeline(
    add_engineered_features: bool = False,
    add_missing_indicators: bool = True,
) -> Pipeline:
    """
    Construct the preprocessing Pipeline.

    Args:
        add_engineered_features: When True, append the interaction features
            from engineer_features. Defaults to False for the rescue model.
        add_missing_indicators: When True, append binary missingness flags
            before median imputation.

    Returns:
        Unfitted sklearn Pipeline.
    """
    steps = []
    if add_missing_indicators:
        steps.append(("missing_flags", MissingIndicatorTransformer()))
    steps.append(("imputer", DataFrameImputer()))
    if add_engineered_features:
        steps.append(("engineer", FeatureEngineerTransformer()))
    steps.append(("scaler", DataFrameScaler()))

    pipeline = Pipeline(steps=steps)
    step_names = " -> ".join(name for name, _ in steps)
    logger.info(f"Preprocessing pipeline constructed with steps: {step_names}")
    return pipeline


def get_pipeline_feature_names(pipeline: Pipeline) -> List[str]:
    """
    Retrieve the final feature names after all pipeline transformations.
    """
    scaler_step: DataFrameScaler = pipeline.named_steps["scaler"]
    return list(scaler_step.get_feature_names_out())
