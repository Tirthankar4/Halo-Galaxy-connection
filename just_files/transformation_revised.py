"""
Revised dataset transformations for CAMELS ASTRID with proper flow-compatible preprocessing.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def apply_shifted_log_transform(
    vals: np.ndarray, feature_name: str
) -> np.ndarray:
    """
    Apply log transform with automatic shift for zero-heavy columns.
    Deterministic and invertible - suitable for normalizing flows.
    
    Args:
        vals: Array of values to transform
        feature_name: Name of feature (for logging)
        
    Returns:
        Log-transformed values with consistent handling of zeros
    """
    # Find minimum positive value
    positive_mask = vals > 0
    
    if not positive_mask.any():
        raise ValueError(f"All values are zero or negative in {feature_name}")
    
    min_positive = vals[positive_mask].min()
    
    # Shift by 1 order of magnitude below minimum positive value
    # This ensures zeros map to a physically plausible value
    shift = min_positive / 10.0
    
    # Apply shifted log transform
    log_vals = np.log10(vals + shift)
    
    num_zeros = np.sum(vals == 0)
    if num_zeros > 0:
        zero_mapped_value = np.log10(shift)
        print(f"  {feature_name}: {num_zeros} zeros mapped to log10({shift:.2e}) = {zero_mapped_value:.4f}")
    
    return log_vals


def apply_mixture_model_transform(
    vals: np.ndarray, feature_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform zero-heavy features into indicator + log-transformed values.
    Best for modeling bimodal (zero vs non-zero) distributions.
    
    Args:
        vals: Array of values to transform
        feature_name: Name of feature
        
    Returns:
        indicator: Binary array (1 if zero, 0 if non-zero)
        log_vals: Log-transformed values (zeros mapped to min non-zero)
    """
    # Binary indicator for zeros
    indicator = (vals == 0).astype(float)
    
    # Transform non-zeros
    positive_mask = vals > 0
    if not positive_mask.any():
        raise ValueError(f"All values are zero in {feature_name}")
    
    min_positive = vals[positive_mask].min()
    
    # Map zeros to minimum non-zero value (deterministic)
    vals_for_log = np.where(vals > 0, vals, min_positive)
    log_vals = np.log10(vals_for_log)
    
    num_zeros = int(indicator.sum())
    pct_zeros = 100 * num_zeros / len(vals)
    print(f"  {feature_name}: {pct_zeros:.1f}% zeros (n={num_zeros})")
    
    return indicator, log_vals


def load_data(data_path: Path) -> pd.DataFrame:
    """Load data and filter for central galaxies only."""
    df = pd.read_parquet(data_path)
    
    # Filter for central galaxies
    df = df[df["is_central"] == True].copy()
    
    # Compute velocity modulus V
    df["V"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2 + df["vel_z"] ** 2)
    
    print(f"Loaded {len(df)} central galaxies")
    return df


def transform_features_shifted_log(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply shifted log transforms for all features.
    DETERMINISTIC - suitable for normalizing flows.
    
    Strategy:
    - Zero-heavy columns: shifted log (maps zeros to consistent value)
    - Positive-only columns: standard log10
    """
    zero_heavy_cols = ["SFR", "Mg", "MBH", "Zg"]
    log10_cols = ["Mstar", "Mt", "Vmax", "sigma_v", "Zstar", "J", "Rstar", "Rt", "Rmax", "V"]
    
    print("\nApplying SHIFTED LOG transformation (deterministic)...")
    print("=" * 60)
    
    # Zero-heavy columns get shifted log
    for col in zero_heavy_cols:
        if col in feature_cols:
            df[f"log_{col}"] = apply_shifted_log_transform(df[col].values, col)
    
    print("\nApplying standard LOG10 transforms...")
    print("=" * 60)
    
    # Positive-only columns get standard log
    for col in log10_cols:
        if col in feature_cols:
            if (df[col] <= 0).any():
                raise ValueError(f"Log10 transform requires positive values in '{col}'.")
            df[f"log_{col}"] = np.log10(df[col].values)
            print(f"  log_{col}: standard log10 (all positive)")
    
    # Build transformed column list
    transformed_feature_cols = []
    for col in feature_cols:
        if col in zero_heavy_cols or col in log10_cols:
            transformed_feature_cols.append(f"log_{col}")
        else:
            transformed_feature_cols.append(col)
    
    return df, transformed_feature_cols


def transform_features_mixture(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply mixture model transformation for zero-heavy features.
    Creates indicator variables + log-transformed values.
    BEST for modeling physical bimodality (quenched vs star-forming, etc.)
    """
    zero_heavy_cols = ["SFR", "Mg", "MBH", "Zg"]
    log10_cols = ["Mstar", "Mt", "Vmax", "sigma_v", "Zstar", "J", "Rstar", "Rt", "Rmax", "V"]
    
    print("\nApplying MIXTURE MODEL transformation...")
    print("=" * 60)
    
    # Zero-heavy columns get indicator + log transform
    for col in zero_heavy_cols:
        if col in feature_cols:
            indicator, log_vals = apply_mixture_model_transform(df[col].values, col)
            df[f"{col}_is_zero"] = indicator
            df[f"log_{col}"] = log_vals
    
    print("\nApplying standard LOG10 transforms...")
    print("=" * 60)
    
    # Positive-only columns get standard log
    for col in log10_cols:
        if col in feature_cols:
            if (df[col] <= 0).any():
                raise ValueError(f"Log10 transform requires positive values in '{col}'.")
            df[f"log_{col}"] = np.log10(df[col].values)
            print(f"  log_{col}: standard log10")
    
    # Build transformed column list
    transformed_feature_cols = []
    for col in feature_cols:
        if col in zero_heavy_cols:
            # Add both indicator and log-transformed value
            transformed_feature_cols.append(f"{col}_is_zero")
            transformed_feature_cols.append(f"log_{col}")
        elif col in log10_cols:
            transformed_feature_cols.append(f"log_{col}")
        else:
            transformed_feature_cols.append(col)
    
    return df, transformed_feature_cols


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardization for stable flow training.
    
    Returns:
        standardized_features: (features - mean) / std
        means: Feature means (for inverse transform)
        stds: Feature stds (for inverse transform)
    """
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    
    # Avoid division by zero for constant features
    stds = np.where(stds == 0, 1.0, stds)
    
    standardized = (features - means) / stds
    
    print("\nStandardization statistics:")
    print(f"  Mean of means: {means.mean():.4f}")
    print(f"  Mean of stds: {stds.mean():.4f}")
    print(f"  Standardized range: [{standardized.min():.2f}, {standardized.max():.2f}]")
    
    return standardized, means, stds


def prepare_features_and_params(
    data_path: Path,
    galaxy_properties: List[str],
    param_columns: List[str],
    method: str = "shifted_log"  # or "mixture"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load parquet data and return transformed, standardized features/params for training.
    
    Args:
        data_path: Path to parquet file
        galaxy_properties: List of galaxy property column names
        param_columns: List of cosmological parameter column names
        method: "shifted_log" or "mixture"
        
    Returns:
        features: Transformed and standardized features
        params: Raw parameters (not transformed)
        feature_means: Means for inverse transform
        feature_stds: Stds for inverse transform
        feature_cols: Original feature column names
        transformed_feature_cols: Transformed feature column names
    """
    df = load_data(data_path)
    
    # All galaxy properties including computed V
    feature_cols = galaxy_properties + ["V"]
    
    # Apply transformation
    if method == "shifted_log":
        df, transformed_feature_cols = transform_features_shifted_log(df, feature_cols)
    elif method == "mixture":
        df, transformed_feature_cols = transform_features_mixture(df, feature_cols)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract transformed features
    features = df[transformed_feature_cols].values
    params = df[param_columns].values
    
    # Standardize features for stable flow training
    print("\n" + "=" * 60)
    features_standardized, feature_means, feature_stds = standardize_features(features)
    
    print("\n" + "=" * 60)
    print(f"FINAL SHAPES:")
    print(f"  Features: {features_standardized.shape}")
    print(f"  Params: {params.shape}")
    print(f"  Original feature cols: {len(feature_cols)}")
    print(f"  Transformed feature cols: {len(transformed_feature_cols)}")
    
    return (
        features_standardized,
        params,
        feature_means,
        feature_stds,
        feature_cols,
        transformed_feature_cols
    )


# Inverse transformations for predictions
def inverse_transform_shifted_log(log_vals: np.ndarray, shift: float) -> np.ndarray:
    """Invert shifted log transform."""
    return 10**log_vals - shift


def inverse_standardize(
    standardized_vals: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """Invert z-score standardization."""
    return standardized_vals * stds + means
