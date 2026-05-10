"""
Validation-set model selection for the rescue training path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

from src.evaluation.scoring import find_best_threshold
from src.evaluation.scoring import metrics_at_threshold
from src.training.trainer import train_xgboost
from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


@dataclass
class ModelSelectionResult:
    model: Any
    model_type: str
    threshold: float
    candidate_scores: Dict[str, Dict[str, float]]
    selected_params: Dict[str, Any]
    selection_metric: str = "balanced_f1_auc"


def _rounded_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        key: round(float(value), 6)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _score_model(model: Any, X_train, y_train, X_val, y_val) -> tuple[float, Dict[str, float]]:
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, metrics, _ = find_best_threshold(y_val, val_proba)
    train_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = metrics_at_threshold(y_train, train_proba, threshold)

    f1_gap = max(0.0, train_metrics["f1"] - metrics["f1"])
    auc_gap = max(0.0, train_metrics["roc_auc"] - metrics["roc_auc"])
    gap_penalty = 0.25 * max(0.0, f1_gap - 0.10) + 0.25 * max(0.0, auc_gap - 0.10)

    metrics = dict(metrics)
    metrics["threshold"] = threshold
    metrics["train_f1"] = train_metrics["f1"]
    metrics["train_roc_auc"] = train_metrics["roc_auc"]
    metrics["f1_gap"] = f1_gap
    metrics["auc_gap"] = auc_gap
    metrics["gap_penalty"] = gap_penalty
    metrics["selection_score"] = metrics["balanced_score"] - gap_penalty
    return float(metrics["selection_score"]), _rounded_metrics(metrics)


def select_best_model(
    X_train,
    y_train,
    X_val,
    y_val,
    xgb_params: Dict[str, Any],
    early_stopping_rounds: int = 15,
    random_state: int = 42,
) -> ModelSelectionResult:
    """
    Train the rescue candidates and select by validation balanced score.
    """
    candidate_scores: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}
    candidate_params: Dict[str, Dict[str, Any]] = {}

    logger.info("Training candidate model: xgboost_cv")
    xgb_model = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        params=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
    )
    trained_models["xgboost_cv"] = xgb_model
    candidate_params["xgboost_cv"] = dict(xgb_params)

    logger.info("Training candidate model: extra_trees_shallow")
    extra_trees_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": 1,
    }
    extra_trees = ExtraTreesClassifier(**extra_trees_params)
    extra_trees.fit(X_train, y_train)
    trained_models["extra_trees_shallow"] = extra_trees
    candidate_params["extra_trees_shallow"] = extra_trees_params

    logger.info("Training candidate model: random_forest_shallow")
    random_forest_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_split": 30,
        "min_samples_leaf": 20,
        "class_weight": "balanced_subsample",
        "random_state": random_state,
        "n_jobs": 1,
    }
    random_forest = RandomForestClassifier(**random_forest_params)
    random_forest.fit(X_train, y_train)
    trained_models["random_forest_shallow"] = random_forest
    candidate_params["random_forest_shallow"] = random_forest_params

    logger.info("Training candidate model: polynomial_logistic")
    polynomial_logistic_params = {
        "poly_degree": 2,
        "include_bias": False,
        "C": 0.2,
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": 3000,
    }
    polynomial_logistic = Pipeline(steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        (
            "logreg",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.2,
                solver="liblinear",
            ),
        ),
    ])
    polynomial_logistic.fit(X_train, y_train)
    trained_models["polynomial_logistic"] = polynomial_logistic
    candidate_params["polynomial_logistic"] = polynomial_logistic_params

    logger.info("Training candidate model: svc_rbf_balanced")
    svc_params = {
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
        "random_state": random_state,
    }
    svc = SVC(**svc_params)
    svc.fit(X_train, y_train)
    trained_models["svc_rbf_balanced"] = svc
    candidate_params["svc_rbf_balanced"] = svc_params

    for name, model in trained_models.items():
        score, metrics = _score_model(model, X_train, y_train, X_val, y_val)
        candidate_scores[name] = metrics
        logger.info(
            f"Candidate {name}: selection_score={score:.4f}, "
            f"balanced_score={metrics['balanced_score']:.4f}, "
            f"threshold={metrics['threshold']:.2f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"gap_penalty={metrics['gap_penalty']:.4f}"
        )

    selected_name = max(
        candidate_scores,
        key=lambda name: candidate_scores[name]["selection_score"],
    )
    logger.info(
        f"Selected model: {selected_name} "
        f"(selection_score={candidate_scores[selected_name]['selection_score']:.4f})"
    )

    return ModelSelectionResult(
        model=trained_models[selected_name],
        model_type=selected_name,
        threshold=float(candidate_scores[selected_name]["threshold"]),
        candidate_scores=candidate_scores,
        selected_params=candidate_params[selected_name],
    )
