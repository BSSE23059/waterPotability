"""
Optuna hyperparameter optimisation for conservative XGBoost models.

The objective uses 3-fold StratifiedKFold cross-validation on the training split
so model choice is less dependent on one lucky validation partition.
"""

from typing import Any, Dict, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.scoring import find_best_threshold
from src.training.trainer import train_xgboost
from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _take_rows(data, indices):
    """Index pandas or numpy-like data by row positions."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def run_optuna_study(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_trials: int = 100,
    random_state: int = 42,
    early_stopping_rounds: int = 15,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run Optuna CV optimisation to find conservative XGBoost hyperparameters.

    X_val and y_val are accepted for backward compatibility with older callers,
    but the objective only uses cross-validation inside X_train/y_train.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 180),
            "max_depth": trial.suggest_int("max_depth", 1, 2),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 50),
            "subsample": trial.suggest_float("subsample", 0.55, 0.80),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.80),
            "reg_alpha": trial.suggest_float("reg_alpha", 4.0, 16.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 15.0, 50.0),
            "gamma": trial.suggest_float("gamma", 0.5, 4.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 1.35),
            "n_jobs": 1,
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        fold_scores = []

        for fold_idx, (fit_idx, score_idx) in enumerate(cv.split(X_train, y_train)):
            X_fit = _take_rows(X_train, fit_idx)
            y_fit = _take_rows(y_train, fit_idx)
            X_score = _take_rows(X_train, score_idx)
            y_score = _take_rows(y_train, score_idx)

            model = train_xgboost(
                X_fit,
                y_fit,
                X_score,
                y_score,
                params=params,
                early_stopping_rounds=early_stopping_rounds,
            )

            val_proba = model.predict_proba(X_score)[:, 1]
            _, metrics, _ = find_best_threshold(y_score, val_proba)

            train_proba = model.predict_proba(X_fit)[:, 1]
            train_auc = float(roc_auc_score(y_fit, train_proba))
            overfit_penalty = max(0.0, train_auc - metrics["roc_auc"] - 0.10)

            fold_score = metrics["balanced_score"] - (0.10 * overfit_penalty)
            fold_scores.append(fold_score)

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="water_potability_xgboost_cv",
    )

    logger.info(
        "Starting Optuna CV study - "
        f"{n_trials} trials, 3 folds, direction=maximize"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["n_jobs"] = 1
    logger.info(f"Optuna CV complete - best balanced score: {study.best_value:.4f}")
    logger.info(f"Best XGBoost params: {best_params}")

    return best_params, study
