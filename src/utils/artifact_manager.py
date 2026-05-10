"""
Artifact manager: save and load pipeline, model, threshold, and metadata.
"""

import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def save_artifacts(
    preprocessing_pipeline: Any,
    model: Any,
    optimal_threshold: float,
    config: SimpleNamespace,
    model_type: str = "xgboost_cv",
    selection_metric: str = "balanced_f1_auc",
    candidate_scores: Optional[Dict[str, Any]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist all inference artifacts to disk.

    Saves:
        - preprocessing_pipeline.pkl: fitted sklearn Pipeline
        - model.pkl: selected trained model
        - xgboost_model.pkl: compatibility copy for older loaders
        - threshold.json: selected decision threshold
        - metadata.json: model selection and training metadata
    """
    artifacts_dir: str = config.api.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    pipeline_path = os.path.join(artifacts_dir, "preprocessing_pipeline.pkl")
    joblib.dump(preprocessing_pipeline, pipeline_path)
    logger.info(f"Preprocessing pipeline saved -> {pipeline_path}")

    model_path = os.path.join(artifacts_dir, "model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Selected model saved -> {model_path}")

    legacy_model_path = os.path.join(artifacts_dir, "xgboost_model.pkl")
    joblib.dump(model, legacy_model_path)
    logger.info(f"Compatibility model copy saved -> {legacy_model_path}")

    threshold_path = os.path.join(artifacts_dir, "threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)
    logger.info(f"Threshold saved -> {threshold_path} (value={optimal_threshold:.4f})")

    metadata = {
        "version": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "optimal_threshold": optimal_threshold,
        "selected_threshold": optimal_threshold,
        "model_type": model_type,
        "selection_metric": selection_metric,
        "candidate_scores": candidate_scores or {},
        "model_params": model_params or {},
        "config": {
            "training_random_state": config.training.random_state,
        },
    }
    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved -> {metadata_path}")


def load_artifact(path: str) -> Any:
    """
    Load a joblib-serialized artifact from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_processed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Save the preprocessed and split data to CSV for auditability.
    """
    data_dir = "artifacts/data"
    os.makedirs(data_dir, exist_ok=True)

    splits = [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]

    for name, X, y in splits:
        df = pd.concat([X, y], axis=1)
        path = os.path.join(data_dir, f"{name}_processed.csv")
        try:
            df.to_csv(path, index=False)
            logger.info(f"Processed {name} data saved -> {path}")
        except OSError as exc:
            logger.warning(f"Processed {name} data was not written: {exc}")
