"""
Validation-set model selection with class-1 precision as the main objective.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC, SVC

from src.evaluation.scoring import (
    get_positive_scores,
    metrics_at_threshold,
    precision_focused_selection_score,
    select_precision_threshold,
    threshold_sweep,
)
from src.training.trainer import train_xgboost
from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


PREFERRED_GENERALIZATION_GAP = 0.10
ACCEPTABLE_GENERALIZATION_GAP = 0.12
MIN_SELECTION_F1 = 0.50
MIN_SELECTION_RECALL = 0.30


try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:  # pragma: no cover - optional dependency guard
    BalancedRandomForestClassifier = None


@dataclass
class ModelSelectionResult:
    model: Any
    model_type: str
    threshold: float
    candidate_scores: Dict[str, Dict[str, float]]
    selected_params: Dict[str, Any]
    trained_models: Dict[str, Any]
    threshold_sweep: pd.DataFrame
    selection_metric: str = "precision_focused_validation_score_with_generalization_guard"


class DropMissingIndicatorColumns(BaseEstimator, TransformerMixin):
    """Drop *_missing columns for candidates that test no missingness flags."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.keep_columns_ = [col for col in self.columns_ if not col.endswith("_missing")]
        return self

    def transform(self, X):
        return X.loc[:, self.keep_columns_]

    def get_feature_names_out(self, input_features=None):
        return self.keep_columns_


def _rounded_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        key: round(float(value), 6)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p").replace("-", "neg").replace("'", "")


def _has_useful_validation_metrics(metrics: Dict[str, float]) -> bool:
    return (
        metrics["val_f1"] >= MIN_SELECTION_F1
        and metrics["val_recall"] >= MIN_SELECTION_RECALL
        and metrics["val_predicted_positives"] >= 30
    )


def _within_generalization_gap(metrics: Dict[str, float], limit: float) -> bool:
    return (
        metrics["train_val_precision_gap"] <= limit
        and metrics["train_val_f1_gap"] <= limit
        and metrics["train_val_roc_auc_gap"] <= limit
    )


def _select_candidate_pool(
    candidate_scores: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Prefer precision-useful models whose train-validation gaps are controlled.

    This prevents a model with a tiny validation-precision edge from winning
    when it has clearly worse precision/F1/AUC generalization gaps.
    """
    useful = {
        name: metrics
        for name, metrics in candidate_scores.items()
        if _has_useful_validation_metrics(metrics)
    }
    if not useful:
        return candidate_scores, "fallback_all_candidates_no_useful_precision_model"

    preferred = {
        name: metrics
        for name, metrics in useful.items()
        if _within_generalization_gap(metrics, PREFERRED_GENERALIZATION_GAP)
    }
    if preferred:
        return preferred, "preferred_gap_guard_all_gaps_le_0p10"

    acceptable = {
        name: metrics
        for name, metrics in useful.items()
        if _within_generalization_gap(metrics, ACCEPTABLE_GENERALIZATION_GAP)
    }
    if acceptable:
        return acceptable, "acceptable_gap_guard_all_gaps_le_0p12"

    return useful, "fallback_useful_candidates_no_gap_guard_match"


def _make_calibrated(estimator: Any, cv: int = 3) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
    except TypeError:  # scikit-learn < 1.2 compatibility
        return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=cv)


def _maybe_drop_missing(model: Any, use_missing_indicators: bool) -> Any:
    if use_missing_indicators:
        return model
    return Pipeline(steps=[
        ("drop_missing_flags", DropMissingIndicatorColumns()),
        ("model", model),
    ])


def _logistic_model(
    penalty: str,
    C: float,
    class_weight: str | None,
    random_state: int,
) -> LogisticRegression:
    kwargs: Dict[str, Any] = {
        "C": C,
        "penalty": penalty,
        "class_weight": class_weight,
        "solver": "saga",
        "max_iter": 8000,
        "random_state": random_state,
    }
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = 0.5
    return LogisticRegression(**kwargs)


def _candidate_rows(
    xgb_params: Dict[str, Any],
    random_state: int,
) -> Iterable[Tuple[str, Any, Dict[str, Any]]]:
    """Yield candidate model definitions."""
    yield "xgboost_conservative", "xgboost", dict(xgb_params)

    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        for penalty in ["l1", "l2", "elasticnet"]:
            for class_weight in [None, "balanced"]:
                for use_missing in [True, False]:
                    name = (
                        f"logreg_{penalty}_C{_format_value(C)}_"
                        f"cw{_format_value(class_weight)}_"
                        f"missing_{'on' if use_missing else 'off'}"
                    )
                    model = _maybe_drop_missing(
                        _logistic_model(penalty, C, class_weight, random_state),
                        use_missing,
                    )
                    params = {
                        "family": "logistic_regression",
                        "C": C,
                        "penalty": penalty,
                        "class_weight": class_weight,
                        "missing_indicators": use_missing,
                    }
                    yield name, model, params

    for C in [0.01, 0.1, 1.0]:
        for penalty in ["l2", "elasticnet"]:
            for class_weight in [None, "balanced"]:
                for use_missing in [True, False]:
                    for interaction_only in [True, False]:
                        name = (
                            f"poly_logreg_{penalty}_C{_format_value(C)}_"
                            f"cw{_format_value(class_weight)}_"
                            f"missing_{'on' if use_missing else 'off'}_"
                            f"{'interactions' if interaction_only else 'full'}"
                        )
                        logreg = _logistic_model(penalty, C, class_weight, random_state)
                        model = Pipeline(steps=[
                            ("maybe_drop_missing", DropMissingIndicatorColumns())
                            if not use_missing
                            else ("keep_missing_flags", "passthrough"),
                            (
                                "poly",
                                PolynomialFeatures(
                                    degree=2,
                                    include_bias=False,
                                    interaction_only=interaction_only,
                                ),
                            ),
                            ("logreg", logreg),
                        ])
                        params = {
                            "family": "polynomial_logistic_regression",
                            "C": C,
                            "penalty": penalty,
                            "class_weight": class_weight,
                            "missing_indicators": use_missing,
                            "poly_degree": 2,
                            "interaction_only": interaction_only,
                        }
                        yield name, model, params

    for max_depth in [3, 4]:
        for min_leaf in [15, 30]:
            for class_weight in [None, "balanced"]:
                model_params = {
                    "n_estimators": 350,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "class_weight": class_weight,
                    "random_state": random_state,
                    "n_jobs": 1,
                }
                name = (
                    f"extra_trees_depth{max_depth}_leaf{min_leaf}_"
                    f"cw{_format_value(class_weight)}"
                )
                params = {"family": "extra_trees", **model_params}
                yield name, ExtraTreesClassifier(**model_params), params

    for max_depth in [3, 4]:
        for min_leaf in [15, 30]:
            for class_weight in [None, "balanced_subsample"]:
                model_params = {
                    "n_estimators": 350,
                    "max_depth": max_depth,
                    "min_samples_split": 30,
                    "min_samples_leaf": min_leaf,
                    "class_weight": class_weight,
                    "random_state": random_state,
                    "n_jobs": 1,
                }
                name = (
                    f"random_forest_depth{max_depth}_leaf{min_leaf}_"
                    f"cw{_format_value(class_weight)}"
                )
                params = {"family": "random_forest", **model_params}
                yield name, RandomForestClassifier(**model_params), params

    for learning_rate in [0.025, 0.04]:
        for l2_regularization in [0.5, 2.0]:
            model_params = {
                "max_iter": 180,
                "learning_rate": learning_rate,
                "max_leaf_nodes": 8,
                "max_depth": 2,
                "l2_regularization": l2_regularization,
                "random_state": random_state,
            }
            name = (
                f"hist_gb_lr{_format_value(learning_rate)}_"
                f"l2{_format_value(l2_regularization)}"
            )
            params = {"family": "hist_gradient_boosting", **model_params}
            yield name, HistGradientBoostingClassifier(**model_params), params

    for C in [0.5, 1.0]:
        for class_weight in [None, "balanced"]:
            estimator = LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=8000,
                random_state=random_state,
                dual="auto",
            )
            params = {
                "family": "calibrated_linear_svc",
                "C": C,
                "class_weight": class_weight,
                "calibration": "sigmoid_cv3",
            }
            name = f"calibrated_linear_svc_C{_format_value(C)}_cw{_format_value(class_weight)}"
            yield name, _make_calibrated(estimator), params

    for C in [0.5, 1.0]:
        for gamma in ["scale", 0.1]:
            estimator = SVC(
                C=C,
                gamma=gamma,
                class_weight="balanced",
                probability=False,
                random_state=random_state,
            )
            params = {
                "family": "calibrated_rbf_svc",
                "C": C,
                "gamma": gamma,
                "class_weight": "balanced",
                "calibration": "sigmoid_cv3",
            }
            name = f"calibrated_rbf_svc_C{_format_value(C)}_gamma{_format_value(gamma)}"
            yield name, _make_calibrated(estimator), params

    if BalancedRandomForestClassifier is not None:
        for max_depth in [3, 4]:
            for min_leaf in [15, 30]:
                model_params = {
                    "n_estimators": 250,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "random_state": random_state,
                    "n_jobs": 1,
                    "replacement": True,
                }
                name = f"balanced_rf_depth{max_depth}_leaf{min_leaf}"
                params = {"family": "balanced_random_forest", **model_params}
                yield name, BalancedRandomForestClassifier(**model_params), params


def _threshold_report_rows(
    model_name: str,
    model: Any,
    splits: List[Tuple[str, Any, Any]],
) -> pd.DataFrame:
    rows = []
    for split_name, X_split, y_split in splits:
        scores = get_positive_scores(model, X_split)
        split_df = threshold_sweep(y_split, scores)
        split_df.insert(0, "split", split_name)
        split_df.insert(0, "model_name", model_name)
        rows.append(split_df)
    return pd.concat(rows, ignore_index=True)


def _score_model(
    model: Any,
    X_train,
    y_train,
    X_val,
    y_val,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    val_scores = get_positive_scores(model, X_val)
    train_scores = get_positive_scores(model, X_train)
    threshold, val_metrics, _ = select_precision_threshold(
        y_val,
        val_scores,
        train_y_true=y_train,
        train_proba=train_scores,
    )
    train_metrics = metrics_at_threshold(y_train, train_scores, threshold)

    score, score_details = precision_focused_selection_score(
        train_metrics,
        val_metrics,
    )

    metrics = {
        "threshold": threshold,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_roc_auc": train_metrics["roc_auc"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_predicted_positives": val_metrics["predicted_positives"],
        "val_false_positives": val_metrics["false_positives"],
        "val_false_negatives": val_metrics["false_negatives"],
        "val_false_positive_reduction_rate": val_metrics["false_positive_reduction_rate"],
        "threshold_selection_reason": val_metrics["threshold_selection_reason"],
        **score_details,
    }

    sweep_df = _threshold_report_rows(
        "candidate",
        model,
        [("train", X_train, y_train), ("validation", X_val, y_val)],
    )
    return score, _rounded_metrics(metrics), sweep_df


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
    Train candidates and select by validation precision-focused objective.
    """
    candidate_scores: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}
    candidate_params: Dict[str, Dict[str, Any]] = {}
    sweep_frames: List[pd.DataFrame] = []

    for name, candidate, params in _candidate_rows(xgb_params, random_state):
        logger.info("Training candidate model: %s", name)
        try:
            if candidate == "xgboost":
                model = train_xgboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    params=params,
                    early_stopping_rounds=early_stopping_rounds,
                )
            else:
                model = candidate
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    model.fit(X_train, y_train)

            score, metrics, sweep_df = _score_model(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            sweep_df["model_name"] = name

            trained_models[name] = model
            candidate_scores[name] = metrics
            candidate_params[name] = params
            sweep_frames.append(sweep_df)

            logger.info(
                "Candidate %s: score=%.4f, val_precision=%.4f, "
                "val_f1=%.4f, val_recall=%.4f, val_auc=%.4f, threshold=%.2f, "
                "gaps(p/f1/auc)=%.4f/%.4f/%.4f",
                name,
                score,
                metrics["val_precision"],
                metrics["val_f1"],
                metrics["val_recall"],
                metrics["val_roc_auc"],
                metrics["threshold"],
                metrics["train_val_precision_gap"],
                metrics["train_val_f1_gap"],
                metrics["train_val_roc_auc_gap"],
            )
        except Exception as exc:
            logger.warning("Candidate %s failed and was skipped: %s", name, exc)

    if not candidate_scores:
        raise RuntimeError("No candidate model trained successfully.")

    selection_pool, selection_pool_reason = _select_candidate_pool(candidate_scores)
    selected_name = max(
        selection_pool,
        key=lambda model_name: selection_pool[model_name]["final_selection_score"],
    )
    candidate_scores[selected_name]["selected_by_gap_guard"] = 1.0
    candidate_scores[selected_name]["selection_pool_size"] = float(len(selection_pool))
    logger.info(
        "Selected model: %s (score=%.4f, val_precision=%.4f, threshold=%.2f, pool=%s)",
        selected_name,
        candidate_scores[selected_name]["final_selection_score"],
        candidate_scores[selected_name]["val_precision"],
        candidate_scores[selected_name]["threshold"],
        selection_pool_reason,
    )

    ordered_scores = dict(
        sorted(
            candidate_scores.items(),
            key=lambda item: item[1]["final_selection_score"],
            reverse=True,
        )
    )
    threshold_report = (
        pd.concat(sweep_frames, ignore_index=True)
        if sweep_frames
        else pd.DataFrame()
    )

    return ModelSelectionResult(
        model=trained_models[selected_name],
        model_type=selected_name,
        threshold=float(candidate_scores[selected_name]["threshold"]),
        candidate_scores=ordered_scores,
        selected_params=candidate_params[selected_name],
        trained_models=trained_models,
        threshold_sweep=threshold_report,
    )


def save_threshold_sweep_report(
    threshold_sweep_df: pd.DataFrame,
    output_csv: str = "artifacts/reports/threshold_sweep.csv",
) -> None:
    """Persist the train/validation threshold sweep for all candidates."""
    if threshold_sweep_df.empty:
        logger.warning("Threshold sweep report was empty and was not written.")
        return

    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except OSError as exc:
        logger.warning("Threshold sweep report directory was not created: %s", exc)
        return
    try:
        threshold_sweep_df.to_csv(output_csv, index=False)
        logger.info("Threshold sweep report saved -> %s", output_csv)
    except OSError as exc:
        logger.warning("Threshold sweep report was not written: %s", exc)


def save_model_comparison_report(
    selection: ModelSelectionResult,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    output_csv: str = "artifacts/reports/model_comparison.csv",
) -> pd.DataFrame:
    """
    Save model comparison metrics.

    Test metrics are computed only after the selected model is fixed by
    validation score. They are reported for transparency, not for selection.
    """
    rows = []
    for model_name, model in selection.trained_models.items():
        threshold = float(selection.candidate_scores[model_name]["threshold"])
        split_metrics = {}
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            scores = get_positive_scores(model, X_split)
            split_metrics[split_name] = metrics_at_threshold(y_split, scores, threshold)

        train_m = split_metrics["train"]
        val_m = split_metrics["val"]
        test_m = split_metrics["test"]
        score_row = selection.candidate_scores[model_name]
        rows.append({
            "model_name": model_name,
            "selected_threshold": threshold,
            "train_accuracy": train_m["accuracy"],
            "val_accuracy": val_m["accuracy"],
            "test_accuracy": test_m["accuracy"],
            "train_precision": train_m["precision"],
            "val_precision": val_m["precision"],
            "test_precision": test_m["precision"],
            "train_recall": train_m["recall"],
            "val_recall": val_m["recall"],
            "test_recall": test_m["recall"],
            "train_f1": train_m["f1"],
            "val_f1": val_m["f1"],
            "test_f1": test_m["f1"],
            "train_roc_auc": train_m["roc_auc"],
            "val_roc_auc": val_m["roc_auc"],
            "test_roc_auc": test_m["roc_auc"],
            "train_val_precision_gap": score_row["train_val_precision_gap"],
            "train_val_f1_gap": score_row["train_val_f1_gap"],
            "train_val_roc_auc_gap": score_row["train_val_roc_auc_gap"],
            "final_selection_score": score_row["final_selection_score"],
            "reason_selected": (
                "selected_using_validation_precision_with_generalization_guard"
                if model_name == selection.model_type
                else (
                    "not_selected_generalization_gap_above_guard"
                    if not _within_generalization_gap(score_row, ACCEPTABLE_GENERALIZATION_GAP)
                    else "not_selected_lower_guarded_validation_score"
                )
            ),
        })

    report_df = pd.DataFrame(rows).sort_values(
        by="final_selection_score",
        ascending=False,
    )
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except OSError as exc:
        logger.warning("Model comparison report directory was not created: %s", exc)
        return report_df
    try:
        report_df.to_csv(output_csv, index=False)
        logger.info("Model comparison report saved -> %s", output_csv)
    except OSError as exc:
        logger.warning("Model comparison report was not written: %s", exc)
    return report_df
