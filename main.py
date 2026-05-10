"""
Water Potability Prediction System — Full Pipeline Entrypoint

Usage:
    python main.py --mode train     # Full training pipeline
    python main.py --mode evaluate  # Evaluate saved model
    python main.py --mode serve     # Start API server
    python main.py --mode all       # Train + evaluate + serve
"""

import argparse
import os
import sys
from contextlib import nullcontext

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
import pandas as pd

from src.config import CONFIG
from src.data.loader import load_and_validate
from src.data.splitter import stratified_split
from src.data.validator import validate_split_distributions
from src.evaluation.metrics import evaluate_split
from src.evaluation.overfitting_detector import check_overfitting
from src.evaluation.threshold import optimize_threshold, diagnose_probability_distribution
from src.explainability.shap_explainer import generate_shap_artifacts
from src.preprocessing.pipeline import build_pipeline, get_pipeline_feature_names
from src.training.baseline import train_baseline
from src.training.model_selection import select_best_model
from src.training.tuner import run_optuna_study
from src.utils.artifact_manager import save_artifacts, save_processed_data
from src.utils.logger import get_logger
from src.utils.mlflow_utils import log_full_experiment, setup_mlflow

logger = get_logger(__name__, "training.log")


def run_training_pipeline() -> None:
    """
    Execute the complete end-to-end training pipeline.

    Follows the mandatory 18-step sequence defined in Section 3 and
    Section 19 of the master prompt. No step may be reordered.
    """
    logger.info("=" * 60)
    logger.info("WATER POTABILITY — TRAINING PIPELINE")
    logger.info("=" * 60)

    # Setup MLflow
    setup_mlflow(
        tracking_uri=CONFIG.mlflow.tracking_uri,
        experiment_name=CONFIG.mlflow.experiment_name,
    )

    try:
        run_context = mlflow.start_run(
            run_name=f"Pipeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
    except Exception as e:
        logger.warning(f"MLflow parent run disabled; continuing without tracking: {e}")
        run_context = nullcontext()

    with run_context:

        # STEP 1: Load data
        logger.info("STEP 1: Loading dataset...")
        df = load_and_validate(CONFIG.data.filepath)

        # STEP 2: Separate features and target
        logger.info("STEP 2: Separating features and target...")
        X = df.drop(CONFIG.data.target_column, axis=1)
        y = df[CONFIG.data.target_column]

        # STEP 3: Stratified split
        logger.info("STEP 3: Stratified train/val/test split...")
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, CONFIG)
        validate_split_distributions(y_train, y_val, y_test)

        # STEPS 4–7: Build and fit preprocessing pipeline (train only)
        logger.info("STEPS 4-7: Building and fitting preprocessing pipeline...")
        preprocessing_pipeline = build_pipeline(add_engineered_features=False)
        X_train_proc = preprocessing_pipeline.fit_transform(X_train, y_train)
        X_val_proc   = preprocessing_pipeline.transform(X_val)
        X_test_proc  = preprocessing_pipeline.transform(X_test)
        logger.info(f"Preprocessed shapes — Train: {X_train_proc.shape}, "
                    f"Val: {X_val_proc.shape}, Test: {X_test_proc.shape}")

        # STEP 8: SMOTE removed (relying on scale_pos_weight via Optuna)

        # NEW STEP: Save processed data splits to CSV
        logger.info("Saving preprocessed data splits to CSV for auditability...")
        save_processed_data(
            X_train_proc, y_train,
            X_val_proc, y_val,
            X_test_proc, y_test
        )


        # STEP 9: Baseline Random Forest
        logger.info("STEP 9: Training Random Forest baseline...")
        baseline_model, baseline_metrics = train_baseline(
            X_train_proc, y_train, X_val_proc, y_val
        )
        logger.info(f"Baseline Val Accuracy: {baseline_metrics['accuracy']:.4f} | "
                    f"F1: {baseline_metrics['f1']:.4f}")

        # STEP 10: Optuna hyperparameter optimisation
        logger.info(f"STEP 10: Running Optuna ({CONFIG.training.optuna_trials} trials)...")
        best_params, study = run_optuna_study(
            X_train_proc, y_train, X_val_proc, y_val,
            n_trials=CONFIG.training.optuna_trials,
            random_state=CONFIG.training.random_state,
            early_stopping_rounds=CONFIG.training.early_stopping_rounds,
        )

        # STEP 11: Train and select the best rescue candidate
        logger.info("STEP 11: Training rescue candidates and selecting best model...")
        selection = select_best_model(
            X_train_proc,
            y_train,
            X_val_proc,
            y_val,
            xgb_params=best_params,
            early_stopping_rounds=CONFIG.training.early_stopping_rounds,
            random_state=CONFIG.training.random_state,
        )
        final_model = selection.model

        # DIAGNOSIS: Check probability distribution
        diagnose_probability_distribution(final_model, X_val_proc, y_val)


        # STEP 12: Threshold optimisation (validation only, selected model)
        logger.info("STEP 12: Optimising decision threshold for selected model...")
        val_proba = final_model.predict_proba(X_val_proc)[:, 1]
        optimal_threshold, threshold_df = optimize_threshold(y_val, val_proba)

        # STEP 13: Evaluate all splits
        logger.info("STEP 13: Evaluating all splits...")
        os.makedirs("artifacts/plots", exist_ok=True)
        train_metrics = evaluate_split(
            final_model, X_train_proc, y_train, "train", optimal_threshold
        )
        val_metrics = evaluate_split(
            final_model, X_val_proc, y_val, "validation", optimal_threshold
        )
        test_metrics = evaluate_split(
            final_model, X_test_proc, y_test, "test", optimal_threshold
        )

        # STEP 14: Overfitting detection
        logger.info("STEP 14: Running overfitting detection...")
        gaps = check_overfitting(train_metrics, val_metrics, test_metrics)

        # STEP 15: SHAP explanations
        logger.info("STEP 15: Generating SHAP explainability artifacts...")
        feature_names = get_pipeline_feature_names(preprocessing_pipeline)
        try:
            generate_shap_artifacts(
                final_model, X_train_proc, X_val_proc,
                feature_names=feature_names,
                output_dir="artifacts/plots",
            )
        except Exception as e:
            logger.warning(f"SHAP generation failed (non-critical): {e}")

        # STEP 16: Save artifacts
        logger.info("STEP 16: Saving model artifacts...")
        save_artifacts(
            preprocessing_pipeline,
            final_model,
            optimal_threshold,
            CONFIG,
            model_type=selection.model_type,
            selection_metric=selection.selection_metric,
            candidate_scores=selection.candidate_scores,
            model_params=selection.selected_params,
        )

        # STEP 17: Log to MLflow
        logger.info("STEP 17: Logging experiment to MLflow...")
        log_full_experiment(
            run_name=f"ModelSelection_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            model=final_model,
            params=selection.selected_params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            threshold=optimal_threshold,
            optuna_study=study,
            shap_plots_dir="artifacts/plots",
            confusion_matrix_plots_dir="artifacts/plots",
            model_type=selection.model_type,
            candidate_scores=selection.candidate_scores,
            selection_metric=selection.selection_metric,
        )

        # STEP 18: Final report
        print_final_report(train_metrics, val_metrics, test_metrics, optimal_threshold, gaps)


def run_evaluate() -> None:
    """
    Load saved model artifacts and evaluate on the dataset.

    Loads preprocessing_pipeline.pkl, xgboost_model.pkl, and threshold.json
    from the artifacts directory and reports performance metrics.
    """
    import joblib
    import json

    logger.info("=" * 60)
    logger.info("WATER POTABILITY — EVALUATION MODE")
    logger.info("=" * 60)

    artifacts_dir = CONFIG.api.artifacts_dir
    pipeline = joblib.load(os.path.join(artifacts_dir, "preprocessing_pipeline.pkl"))
    model_path = os.path.join(artifacts_dir, "model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(artifacts_dir, "xgboost_model.pkl")
    model = joblib.load(model_path)

    with open(os.path.join(artifacts_dir, "threshold.json")) as f:
        threshold = json.load(f)["optimal_threshold"]

    df = load_and_validate(CONFIG.data.filepath)
    X  = df.drop(CONFIG.data.target_column, axis=1)
    y  = df[CONFIG.data.target_column]

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, CONFIG)
    X_test_proc = pipeline.transform(X_test)

    test_metrics = evaluate_split(model, X_test_proc, y_test, "test", threshold)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1:       {test_metrics['f1']:.4f}")
    logger.info(f"Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")


def run_serve() -> None:
    """Start the FastAPI server using uvicorn."""
    import uvicorn

    logger.info("=" * 60)
    logger.info("WATER POTABILITY — API SERVER")
    logger.info(f"Starting on http://{CONFIG.api.host}:{CONFIG.api.port}")
    logger.info(f"Docs: http://{CONFIG.api.host}:{CONFIG.api.port}/docs")
    logger.info(f"UI:   http://{CONFIG.api.host}:{CONFIG.api.port}/ui")
    logger.info("=" * 60)

    uvicorn.run(
        "api.main:app",
        host=CONFIG.api.host,
        port=CONFIG.api.port,
        reload=False,
        log_level="info",
    )


def print_final_report(
    train_m: dict,
    val_m: dict,
    test_m: dict,
    threshold: float,
    gaps: dict,
) -> None:
    """Print a formatted performance table to stdout."""
    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE REPORT")
    print("=" * 60)
    print(f"{'Metric':<15} {'Train':>10} {'Val':>10} {'Test':>10}")
    print("-" * 50)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(
            f"{metric:<15} "
            f"{train_m[metric]:>10.4f} "
            f"{val_m[metric]:>10.4f} "
            f"{test_m[metric]:>10.4f}"
        )
    print("-" * 50)
    print(f"Optimal Threshold:      {threshold:.2f}")
    print(f"Train-Val Accuracy Gap: {gaps['accuracy_gap']:+.4f}")
    print(f"Train-Val F1 Gap:       {gaps['f1_gap']:+.4f}")
    print("=" * 60)

    if abs(gaps["accuracy_gap"]) < 0.08 and abs(gaps["f1_gap"]) < 0.08:
        assessment = "Excellent - train-val gaps within target (<0.08)"
    elif abs(gaps["accuracy_gap"]) < 0.12:
        assessment = "Acceptable - some overfitting present"
    else:
        assessment = "Poor - significant overfitting detected"

    print(f"GENERALISATION: {assessment}")
    print("=" * 60 + "\n")
    return

    # Generalisation assessment
    if abs(gaps["accuracy_gap"]) < 0.08 and abs(gaps["f1_gap"]) < 0.08:
        print("✓ GENERALISATION: Excellent — train-val gaps within target (<0.08)")
    elif abs(gaps["accuracy_gap"]) < 0.12:
        print("⚠ GENERALISATION: Acceptable — some overfitting present")
    else:
        print("✗ GENERALISATION: Poor — significant overfitting detected")
    print("=" * 60 + "\n")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate pipeline mode."""
    parser = argparse.ArgumentParser(
        description="Water Potability Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train     # Full training pipeline
  python main.py --mode evaluate  # Evaluate saved model on test set
  python main.py --mode serve     # Start the FastAPI API server
  python main.py --mode all       # Train, then serve
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "serve", "all"],
        default="train",
        help="Pipeline mode to run (default: train)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training_pipeline()
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "serve":
        run_serve()
    elif args.mode == "all":
        run_training_pipeline()
        run_serve()


if __name__ == "__main__":
    main()
