"""
MLflow integration utilities for the Water Potability Prediction System.

Logs parameters, metrics, model artifacts, SHAP plots, and Optuna results
to a single parent MLflow run under the "water_potability" experiment.
"""

import os
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """
    Configure MLflow tracking URI and set/create the experiment.

    Args:
        tracking_uri:    Local path (e.g. "mlruns") or remote URI.
        experiment_name: Name of the MLflow experiment to use.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {tracking_uri} | Experiment: {experiment_name}")
    except Exception as exc:
        logger.warning(f"MLflow setup failed; continuing without tracking: {exc}")


def log_full_experiment(
    run_name: str,
    model: Any,
    params: Dict[str, Any],
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    threshold: float,
    optuna_study: Any,
    shap_plots_dir: str,
    confusion_matrix_plots_dir: str,
    model_type: str = "xgboost_cv",
    candidate_scores: Dict[str, Any] | None = None,
    selection_metric: str = "balanced_f1_auc",
) -> None:
    """
    Log a complete experiment run to MLflow.

    Logs:
        - Hyperparameters (best Optuna params + optimal threshold)
        - All metrics for train / validation / test splits (5 metrics × 3 = 15)
        - Train-val overfitting gaps
        - XGBoost model artifact
        - SHAP visualisation plots
        - Confusion matrix plots
        - Optuna trials CSV
        - Threshold analysis CSV

    Args:
        run_name:                      Human-readable MLflow run name.
        model:                         Trained XGBoost classifier.
        params:                        Best hyperparameters from Optuna.
        train_metrics:                 Metrics dict for training split.
        val_metrics:                   Metrics dict for validation split.
        test_metrics:                  Metrics dict for test split.
        threshold:                     Optimised decision threshold.
        optuna_study:                  Completed Optuna Study object.
        shap_plots_dir:                Directory containing SHAP PNG files.
        confusion_matrix_plots_dir:    Directory containing confusion matrix PNGs.
    """
    # End any existing active run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    try:
        run_context = mlflow.start_run(run_name=run_name)
    except Exception as exc:
        logger.warning(f"Skipping MLflow experiment logging: {exc}")
        return

    with run_context:
        # 1. Hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("optimal_threshold", threshold)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("selection_metric", selection_metric)

        # 2. Metrics — all three splits
        for split, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ]:
            mlflow.log_metric(f"{split}_accuracy", metrics["accuracy"])
            mlflow.log_metric(f"{split}_precision", metrics["precision"])
            mlflow.log_metric(f"{split}_recall", metrics["recall"])
            mlflow.log_metric(f"{split}_f1", metrics["f1"])
            mlflow.log_metric(f"{split}_roc_auc", metrics["roc_auc"])

        # 3. Overfitting gaps
        mlflow.log_metric(
            "train_val_accuracy_gap",
            train_metrics["accuracy"] - val_metrics["accuracy"],
        )
        mlflow.log_metric(
            "train_val_f1_gap",
            train_metrics["f1"] - val_metrics["f1"],
        )

        # 4. Model artifact
        if model.__class__.__module__.startswith("xgboost"):
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        if candidate_scores:
            for candidate_name, scores in candidate_scores.items():
                for metric_name, metric_value in scores.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(
                            f"candidate_{candidate_name}_{metric_name}",
                            float(metric_value),
                        )

        # 5. SHAP plots
        if os.path.isdir(shap_plots_dir):
            mlflow.log_artifacts(shap_plots_dir, artifact_path="shap")

        # 6. Confusion matrices
        if os.path.isdir(confusion_matrix_plots_dir):
            mlflow.log_artifacts(
                confusion_matrix_plots_dir, artifact_path="confusion_matrices"
            )

        # 7. Optuna trials CSV
        os.makedirs("artifacts", exist_ok=True)
        optuna_csv = "artifacts/optuna_trials.csv"
        if optuna_study is not None:
            optuna_df = optuna_study.trials_dataframe()
            optuna_df.to_csv(optuna_csv, index=False)
            mlflow.log_artifact(optuna_csv, artifact_path="optuna")

        # 8. Threshold analysis CSV
        threshold_csv = "artifacts/threshold_analysis.csv"
        if os.path.exists(threshold_csv):
            mlflow.log_artifact(threshold_csv, artifact_path="threshold")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run logged successfully — run_id: {run_id}")
