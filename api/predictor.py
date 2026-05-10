"""
Model loading and inference logic for the Water Potability Prediction API.

Implements a thread-safe singleton predictor that loads all trained artifacts
at startup and handles feature engineering + preprocessing at inference time.
"""

import json
import os
from typing import Dict, Any

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__, "api.log")


class WaterPotabilityPredictor:
    """
    Singleton predictor that wraps the trained preprocessing pipeline and model.

    Lifecycle:
        1. Instantiated once at FastAPI startup via lifespan or module-level.
        2. Loads preprocessing_pipeline.pkl, xgboost_model.pkl, threshold.json,
           and metadata.json from the artifacts directory.
        3. Each call to predict() applies the full preprocessing pipeline
           and returns a structured prediction dict.

    Thread safety:
        The predict() method is stateless (read-only operations on loaded objects),
        making it safe for concurrent FastAPI requests without locking.
    """

    def __init__(self, artifacts_dir: str):
        """
        Load all inference artifacts from disk.

        Args:
            artifacts_dir: Path to the directory containing model artifacts.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        self._artifacts_dir = artifacts_dir
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load pipeline, model, threshold, and metadata from the artifacts directory."""
        pipeline_path  = os.path.join(self._artifacts_dir, "preprocessing_pipeline.pkl")
        model_path     = os.path.join(self._artifacts_dir, "model.pkl")
        legacy_model_path = os.path.join(self._artifacts_dir, "xgboost_model.pkl")
        threshold_path = os.path.join(self._artifacts_dir, "threshold.json")
        metadata_path  = os.path.join(self._artifacts_dir, "metadata.json")

        if not os.path.exists(model_path) and os.path.exists(legacy_model_path):
            model_path = legacy_model_path

        for path in [pipeline_path, model_path, threshold_path, metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required artifact not found: {path}. "
                    f"Run 'python main.py --mode train' first."
                )

        self.pipeline  = joblib.load(pipeline_path)
        self.model     = joblib.load(model_path)

        with open(threshold_path, "r", encoding="utf-8") as f:
            self.threshold = json.load(f)["optimal_threshold"]

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.model_version = metadata.get("version", "unknown")
        self.model_type = metadata.get("model_type", self.model.__class__.__name__)

        logger.info(
            f"Predictor loaded — version={self.model_version}, "
            f"model_type={self.model_type}, threshold={self.threshold:.4f}"
        )

    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full inference pipeline on a single water sample.

        Pipeline applied at inference:
            1. Convert dict to DataFrame (preserving column names).
            2. Apply fitted preprocessing pipeline (impute → engineer → scale → select).
            3. Predict class probability using the trained XGBoost model.
            4. Apply optimal threshold to obtain binary prediction.
            5. Compute confidence based on distance from threshold.

        Confidence levels:
            HIGH:   |probability - threshold| > 0.20
            MEDIUM: |probability - threshold| > 0.10
            LOW:    otherwise (prediction is uncertain)

        Args:
            sample: Dict with water quality feature values (9 features).

        Returns:
            Dict with keys: potability, probability, threshold_used,
                            interpretation, confidence.
        """
        X = pd.DataFrame([sample])
        X_processed = self.pipeline.transform(X)
        proba = float(self.model.predict_proba(X_processed)[0][1])
        prediction = int(proba >= self.threshold)

        dist = abs(proba - self.threshold)
        if dist > 0.20:
            confidence = "HIGH"
        elif dist > 0.10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "potability":     prediction,
            "probability":    round(proba, 4),
            "threshold_used": self.threshold,
            "interpretation": "POTABLE" if prediction == 1 else "NOT POTABLE",
            "confidence":     confidence,
        }
