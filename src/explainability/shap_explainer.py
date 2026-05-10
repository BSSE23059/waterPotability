"""
SHAP explainability artifacts for the Water Potability XGBoost model.

Generates three visualisation artifacts on the validation set:
    1. Summary beeswarm plot  — global feature impact distribution
    2. Feature importance bar — mean |SHAP| per feature
    3. Dependence plot        — SHAP values vs feature value for top feature

Uses TreeExplainer which computes exact SHAP values for tree-based models,
unlike KernelExplainer which approximates via sampling.

Explanations are generated on the VALIDATION set (not training):
    - Validation set is representative of unseen data distribution.
    - Not contaminated by SMOTE synthetic samples.
    - Provides unbiased feature importance estimates.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__, "training.log")


def generate_shap_artifacts(
    model,
    X_train,
    X_val,
    feature_names: List[str],
    output_dir: str = "artifacts/plots",
) -> None:
    """
    Generate and save three SHAP visualisation plots.

    Plots generated:
        shap_summary.png     — beeswarm plot showing per-sample feature impacts
        shap_importance.png  — bar chart of mean absolute SHAP values
        shap_dependence_top.png — dependence plot for the most impactful feature

    Args:
        model:         Trained XGBoost classifier (xgb.XGBClassifier).
        X_train:       Training features (used to build the SHAP explainer background).
        X_val:         Validation features (used to compute SHAP values).
        feature_names: List of feature name strings corresponding to X_val columns.
        output_dir:    Directory path to save PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating SHAP explanations using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_val)
    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1] if len(raw_shap_values) > 1 else raw_shap_values[0]
    else:
        shap_values = np.asarray(raw_shap_values)
        if shap_values.ndim == 3:
            class_index = 1 if shap_values.shape[2] > 1 else 0
            shap_values = shap_values[:, :, class_index]

    # Convert to numpy if DataFrame
    X_val_arr = X_val.values if hasattr(X_val, "values") else X_val

    # 1. Summary beeswarm plot
    logger.debug("Generating SHAP summary (beeswarm) plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val_arr, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Impact (Beeswarm)", fontsize=14, pad=12)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved → {summary_path}")

    # 2. Feature importance bar chart
    logger.debug("Generating SHAP feature importance (bar) plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_val_arr,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, pad=12)
    plt.tight_layout()
    importance_path = os.path.join(output_dir, "shap_importance.png")
    plt.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP importance plot saved → {importance_path}")

    # 3. Dependence plot for the most important feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_idx = int(np.argmax(mean_abs_shap))
    top_feature_name = feature_names[top_feature_idx] if feature_names else str(top_feature_idx)

    logger.debug(f"Generating SHAP dependence plot for top feature: '{top_feature_name}'...")
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        top_feature_idx, shap_values, X_val_arr,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Dependence — {top_feature_name}", fontsize=14, pad=12)
    plt.tight_layout()
    dependence_path = os.path.join(output_dir, "shap_dependence_top.png")
    plt.savefig(dependence_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP dependence plot saved → {dependence_path}")

    logger.info(
        f"All SHAP artifacts generated successfully in {output_dir}. "
        f"Top feature: '{top_feature_name}' (mean |SHAP|={mean_abs_shap[top_feature_idx]:.4f})"
    )
