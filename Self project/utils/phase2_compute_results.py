#!/usr/bin/env python3
"""
Phase 2: Compute per-simulation metrics for plotting (results CSV).

This evaluates the Phase 2 delta NF model on all test simulations and writes:
simulation, distance_sigma, a_sn1, rmse, pearson_r, num_galaxies
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure local src is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Local imports
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_universal_scatter_model"

TRAINING_SETS = ["LH_135", "LH_473", "LH_798"]
COSMO_PARAMS_FILE = RAW_DATA_DIR / "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
SHMR_JSON = OUTPUT_DIR / "shmr_fit_parameters.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_cosmo_params() -> pd.DataFrame:
    df = pd.read_csv(
        COSMO_PARAMS_FILE,
        sep=r"\s+",
        skiprows=1,
        names=[
            "Name",
            "Omega_m",
            "sigma_8",
            "A_SN1",
            "A_AGN1",
            "A_SN2",
            "A_AGN2",
            "seed",
        ],
    )
    df["Name"] = df["Name"].str.strip()
    return df


def get_available_processed_datasets() -> List[str]:
    return sorted(
        d.name
        for d in PROCESSED_DATA_DIR.iterdir()
        if d.is_dir() and (d / "halo_galaxy.parquet").exists()
    )


def get_test_simulations(cosmo_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return list of (LH_###, lh###) for sims that are processed and not training."""
    test_sims: List[Tuple[str, str]] = []
    for proc_dir in get_available_processed_datasets():
        if not proc_dir.startswith("lh"):
            continue
        num = proc_dir[2:]
        upper = f"LH_{num}"
        if upper in TRAINING_SETS:
            continue
        if len(cosmo_df[cosmo_df["Name"] == upper]) == 0:
            continue
        test_sims.append((upper, proc_dir))
    return test_sims


def compute_multivariate_distance(test_row: pd.Series, training_rows: pd.DataFrame) -> float:
    params = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1"]
    means = training_rows[params].mean()
    stds = training_rows[params].std().replace(0, 1.0)
    diffs = (test_row[params] - means) / stds
    return float(np.sqrt(np.sum(diffs.values ** 2)))


def load_shmr_params() -> Tuple[float, float]:
    with open(SHMR_JSON, "r") as f:
        data = json.load(f)
    params = data["parameters"]
    return float(params["alpha"]), float(params["beta"])


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) < 2:
        return np.nan
    return float(np.corrcoef(y_t, y_p)[0, 1])


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean bias (pred - true)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return np.nan
    return float(np.mean(y_p - y_t))


# --------------------------------------------------------------------------- #
# Bias correction model
# --------------------------------------------------------------------------- #
def apply_bias_correction(sm_pred: np.ndarray, a_sn1_values: float) -> np.ndarray:
    """Apply quadratic bias correction model."""
    bias_pred = -0.1056 * a_sn1_values**2 + 0.7652 * a_sn1_values - 1.1497
    return sm_pred - bias_pred


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #
def evaluate_sims(model_run: str, num_draws: int) -> pd.DataFrame:
    from src.plots.nf_visualizer import load_nf_artifacts, sample_nf_posterior_for_halos
    from src.utils.common import resolve_device

    cosmo_df = load_cosmo_params()
    training_rows = cosmo_df[cosmo_df["Name"].isin(TRAINING_SETS)]
    test_sims = get_test_simulations(cosmo_df)
    alpha, beta = load_shmr_params()

    device = resolve_device("cpu")
    artifacts = load_nf_artifacts(model_run, device)
    features = artifacts["features"]
    targets = artifacts["targets"]
    if "Delta_SM" in targets:
        delta_idx = targets.index("Delta_SM")
    elif "SM_delta" in targets:
        delta_idx = targets.index("SM_delta")
    else:
        raise ValueError("Model must predict Delta_SM or SM_delta.")

    rows: List[Dict] = []

    for sim_upper, sim_dir in test_sims:
        parquet_path = PROCESSED_DATA_DIR / sim_dir / "halo_galaxy.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        if not set(features).issubset(df.columns) or "SM" not in df.columns:
            continue

        halos = df[features].values
        post = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=num_draws,
            device=device,
            batch_size=256,
        )
        delta_pred = np.median(post[:, :, delta_idx], axis=1)

        sm_shmr = alpha * df["M_h"].values + beta
        sm_pred = sm_shmr + delta_pred

        cosmo_row = cosmo_df[cosmo_df["Name"] == sim_upper].iloc[0]
        distance = compute_multivariate_distance(cosmo_row, training_rows)

        # Apply bias correction using the quadratic model
        sm_pred_corrected = apply_bias_correction(
            sm_pred, float(cosmo_row["A_SN1"])
        )

        # Original metrics
        rmse = compute_rmse(df["SM"].values, sm_pred)
        pearson = compute_pearson_r(df["SM"].values, sm_pred)
        bias = compute_bias(df["SM"].values, sm_pred)

        # Corrected metrics
        rmse_corrected = compute_rmse(df["SM"].values, sm_pred_corrected)
        bias_corrected = compute_bias(df["SM"].values, sm_pred_corrected)

        rows.append(
            {
                "simulation": sim_upper,
                "distance_sigma": distance,
                "a_sn1": float(cosmo_row["A_SN1"]),
                "rmse": rmse,
                "pearson_r": pearson,
                "bias": bias,
                "rmse_corrected": rmse_corrected,
                "bias_corrected": bias_corrected,
                "num_galaxies": len(df),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Phase 2 results CSV")
    parser.add_argument(
        "--model",
        default="delta_sm",
        help="NF run name that predicts Delta_SM (default: delta_sm).",
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=500,
        help="Posterior draws for NF sampling (default: 500).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "phase2_results.csv",
        help="Output CSV path (default: outputs/phase2_universal_scatter_model/phase2_results.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = evaluate_sims(args.model, args.num_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()

