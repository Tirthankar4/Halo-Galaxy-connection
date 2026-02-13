"""
Dataset transformations for CAMELS ASTRID parquet files.
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
    # This ensures zeros map to a value below all non-zeros
    shift = min_positive / 10.0

    # Apply shifted log transform: log10(x + shift)
    log_vals = np.log10(vals + shift)

    num_zeros = np.sum(vals == 0)
    if num_zeros > 0:
        zero_mapped_value = np.log10(shift)
        print(f"  {feature_name}: {num_zeros} zeros -> log10({shift:.2e}) = {zero_mapped_value:.4f}")

    return log_vals


def load_data(data_path: Path) -> pd.DataFrame:
    """Load data and filter for central galaxies only."""
    df = pd.read_parquet(data_path)

    # Filter for central galaxies
    df = df[df["is_central"] == True].copy()

    # Compute velocity modulus V
    df["V"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2 + df["vel_z"] ** 2)

    print(f"Loaded {len(df)} central galaxies")
    return df


def transform_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply shifted log for zero-heavy columns and standard log10 transforms
    for positive-only columns. Returns updated features.
    
    """
    zero_heavy_cols = ["SFR", "Mg", "MBH", "Zg"]
    log10_cols = [
        "Mstar",
        "Mt",
        "Vmax",
        "sigma_v",
        "Zstar",
        "J",
        "Rstar",
        "Rt",
        "Rmax",
        "V",
    ]

    print("\nApplying SHIFTED LOG transformation (deterministic)...")
    for col in zero_heavy_cols:
        if col in feature_cols:
            df[f"log_{col}"] = apply_shifted_log_transform(df[col].values, col)

    print("\nApplying standard log10 transforms for positive-only columns...")
    for col in log10_cols:
        if col in feature_cols:
            if (df[col] <= 0).any():
                raise ValueError(
                    f"Log10 transform requires positive values in '{col}'."
                )
            df[f"log_{col}"] = np.log10(df[col].values)

    transformed_feature_cols = []
    for col in feature_cols:
        if col in zero_heavy_cols or col in log10_cols:
            transformed_feature_cols.append(f"log_{col}")
        else:
            transformed_feature_cols.append(col)

    print(
        "\nTransformed columns: "
        "SFR -> log_SFR, Mg -> log_Mg, MBH -> log_MBH, Zg -> log_Zg, "
        "Mstar -> log_Mstar, Mt -> log_Mt, Vmax -> log_Vmax, "
        "sigma_v -> log_sigma_v, Zstar -> log_Zstar, J -> log_J, "
        "Rstar -> log_Rstar, Rt -> log_Rt, Rmax -> log_Rmax, V -> log_V"
    )
    return df, transformed_feature_cols


def prepare_features_and_params(
    data_path: Path, galaxy_properties: List[str], param_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load parquet data and return transformed features/params for training.
    """
    df = load_data(data_path)

    # All galaxy properties including computed V
    feature_cols = galaxy_properties + ["V"]

    df, transformed_feature_cols = transform_features(df, feature_cols)

    features = df[transformed_feature_cols].values
    params = df[param_columns].values

    return features, params, feature_cols, transformed_feature_cols
