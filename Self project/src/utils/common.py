"""Common utilities shared across training and visualization scripts."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance
from sklearn.metrics import mean_squared_error


def resolve_device(device_arg: str) -> torch.device:
    """Prefer CUDA and prompt before falling back to CPU."""
    device_arg = device_arg or "cuda"
    wants_cuda = device_arg.lower().startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        message = (
            "CUDA device requested but no GPU is available. "
            "Type 'Y' to continue on CPU (anything else aborts): "
        )
        if input(message).strip().lower() != "y":
            raise SystemExit("Aborted: GPU unavailable and CPU fallback declined.")
        logging.warning("Falling back to CPU because CUDA was requested but unavailable.")
        return torch.device("cpu")
    return torch.device(device_arg)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics between true and predicted values.
    
    Args:
        y_true: True values (flattened for proper correlation computation)
        y_pred: Predicted values (flattened for proper correlation computation)
        
    Returns:
        Dictionary with RMSE, Pearson correlation, K-S stat, and Wasserstein distance
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    pearson_r, pearson_p = pearsonr(y_true_flat, y_pred_flat)
    ks_stat, ks_p = ks_2samp(y_true_flat, y_pred_flat)
    wass = float(wasserstein_distance(y_true_flat, y_pred_flat))
    
    return {
        "RMSE": rmse,
        "MSE": float(mean_squared_error(y_true_flat, y_pred_flat)),
        "Pearson": float(pearson_r),
        "Pearson_r": float(pearson_r),
        "Pearson_p": float(pearson_p),
        "PCC": float(pearson_r),  # Alias for visualization scripts
        "KS_stat": float(ks_stat),
        "KS_p": float(ks_p),
        "K-S": float(ks_stat),  # Alias for optimization scripts
        "Wasserstein": wass,
    }


def compute_single_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """
    Compute a single specified metric.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric: Metric name (RMSE, PCC, K-S, or Wasserstein)
        
    Returns:
        Metric value
    """
    all_metrics = compute_regression_metrics(y_true, y_pred)
    if metric in all_metrics:
        return all_metrics[metric]
    raise ValueError(f"Unknown metric: {metric}. Available: {list(all_metrics.keys())}")


# Constants
# All available halo properties that can be used as features
ALL_HALO_PROPERTIES = [
    "M_h", "R_h", "V_h",  # Original halo properties
    "Omega_m", "sigma_8",  # Cosmological parameters
    "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"  # Astrophysical feedback parameters
]
# All available galaxy properties that can be predicted
ALL_GALAXY_PROPERTIES = ["SM", "SFR", "SR", "Colour", "Delta_SM"]

# Default feature and target columns (for backwards compatibility)
FEATURE_COLUMNS = ["M_h", "R_h", "V_h"]
TARGET_COLUMNS = ["SM", "SFR", "SR", "Colour"]


def validate_features(features: list[str]) -> list[str]:
    """
    Validate that requested features are available in the dataset.
    
    Args:
        features: List of feature column names
        
    Returns:
        Validated list of features
        
    Raises:
        ValueError: If any feature is not in ALL_HALO_PROPERTIES
    """
    if not features:
        raise ValueError("At least one feature must be specified")
    
    invalid_features = [f for f in features if f not in ALL_HALO_PROPERTIES]
    if invalid_features:
        raise ValueError(
            f"Invalid features: {invalid_features}. "
            f"Available features: {ALL_HALO_PROPERTIES}"
        )
    
    return features


def validate_targets(targets: list[str]) -> list[str]:
    """
    Validate that requested targets are available in the dataset.
    
    Args:
        targets: List of target column names
        
    Returns:
        Validated list of targets
        
    Raises:
        ValueError: If any target is not in ALL_GALAXY_PROPERTIES
    """
    if not targets:
        raise ValueError("At least one target must be specified")
    
    invalid_targets = [t for t in targets if t not in ALL_GALAXY_PROPERTIES]
    if invalid_targets:
        raise ValueError(
            f"Invalid targets: {invalid_targets}. "
            f"Available targets: {ALL_GALAXY_PROPERTIES}"
        )
    
    return targets

# Color scheme for consistent visualizations
MODEL_COLORS = {
    # NN Raw - Blue
    "NN_Raw": "#0173B2",
    "NN Raw": "#0173B2",
    "NN (Raw)": "#0173B2",
    "Raw": "#0173B2",
    # NN SMOGN - Yellow/Orange
    "NN_SMOGN": "#DE8F05",
    "NN SMOGN": "#DE8F05",
    "NN (SMOGN)": "#DE8F05",
    "SMOGN": "#DE8F05",
    # NN Optuna - Red
    "OPTUNA_Raw": "#DC143C",
    "NN_Optuna": "#DC143C",
    "NN Optuna": "#DC143C",
    "NN (Optuna)": "#DC143C",
    "optuna_best": "#DC143C",
    "Optuna": "#DC143C",
    # NF (mean) - Green
    "NF_Mean": "#029E73",
    "NF Mean": "#029E73",
    "NF (Mean)": "#029E73",
    "NF (mean)": "#029E73",
    # NF (mode) - Purple
    "NF_Mode": "#8B5CF6",
    "NF Mode": "#8B5CF6",
    "NF (Mode)": "#8B5CF6",
    "NF (mode)": "#8B5CF6",
    # NF (random) - Pink
    "NF_Random": "#CC78BC",
    "NF Random": "#CC78BC",
    "NF (Random)": "#CC78BC",
    "NF (random)": "#CC78BC",
}


def get_model_color(label: str) -> str:
    """Get consistent color for a model label."""
    # Try exact match first
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    
    # Try keyword matching
    label_lower = label.lower()
    if "optuna" in label_lower:
        return "#DC143C"  # Red
    elif "smogn" in label_lower:
        return "#DE8F05"  # Yellow/Orange
    elif "raw" in label_lower:
        return "#0173B2"  # Blue
    elif "nf" in label_lower:
        if "mean" in label_lower:
            return "#029E73"  # Green
        elif "mode" in label_lower:
            return "#8B5CF6"  # Purple
        elif "random" in label_lower:
            return "#CC78BC"  # Pink
    
    return "#888888"  # Default grey

