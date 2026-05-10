"""
XGBoost model training with early stopping and anti-overfitting constraints.

Design philosophy (per Section 7 of the master prompt):
    - Shallow trees (max_depth ≤ 4) prevent memorisation of training noise
    - Early stopping (rounds=30) halts training when validation loss stagnates
    - Low learning rate (0.01–0.1) forces gradual, generalisable learning
    - L1 + L2 regularisation explicitly penalise model complexity
    - Low learning rate (0.01–0.1) forces gradual, generalisable learning
    - L1 + L2 regularisation explicitly penalise model complexity
    - Row and column subsampling add stochasticity, reducing overfitting
"""

from typing import Any, Dict

import pandas as pd
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    early_stopping_rounds: int = 30,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with anti-overfitting constraints and early stopping.

    Anti-overfitting design decisions:
        max_depth ≤ 4:
            Shallow trees prevent memorisation of training noise.
            Deep trees (depth 6+) on small datasets are the primary overfitting cause.
        early_stopping_rounds=30:
            Halts training when validation logloss hasn't improved for 30 rounds.
            The model with the best validation score is automatically restored.
        learning_rate in [0.01, 0.1]:
            Low learning rate forces gradual learning. Combined with early stopping,
            prevents aggressive fitting to noise.
        min_child_weight ≥ 3:
            Higher values prevent leaves that represent only 1–2 samples,
            which are highly sensitive to noise.
        subsample in [0.6, 0.85]:
            Row subsampling per tree adds stochasticity and reduces overfitting.
        colsample_bytree in [0.6, 0.85]:
            Feature subsampling per tree forces feature diversity across trees.
        reg_alpha (L1) + reg_lambda (L2):
            Explicit regularisation terms — critical for this noisy, small dataset.
        reg_alpha (L1) + reg_lambda (L2):
            Explicit regularisation terms — critical for this noisy, small dataset.

    Args:
        X_train:               Preprocessed + SMOTE-resampled training features.
        y_train:               Resampled training labels.
        X_val:                 Preprocessed validation features (not resampled).
        y_val:                 Validation labels (original distribution).
        params:                XGBoost hyperparameters dict (from Optuna or defaults).
        early_stopping_rounds: Rounds without improvement before halting.

    Returns:
        Trained XGBClassifier with best_iteration and best_score attributes set.
    """
    params = dict(params)
    params.setdefault("n_jobs", 1)

    model = xgb.XGBClassifier(
        **params,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    logger.info(f"XGBoost training complete.")
    logger.info(f"  Best iteration:  {model.best_iteration}")
    logger.info(f"  Best val logloss: {model.best_score:.6f}")

    return model


def get_default_params() -> Dict[str, Any]:
    """
    Return conservative default XGBoost hyperparameters.

    These defaults are intentionally constrained to avoid overfitting
    when Optuna optimisation is skipped (e.g. for quick tests).

    Returns:
        Dict of XGBoost hyperparameter key-value pairs.
    """
    return {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.035,
        "min_child_weight": 25,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "reg_alpha": 8.0,
        "reg_lambda": 30.0,
        "gamma": 1.5,
        "scale_pos_weight": 1.2,
        "n_jobs": 1,
    }
