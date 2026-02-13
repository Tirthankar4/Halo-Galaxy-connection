#!/usr/bin/env python3
"""
Phase 3: Compute calibration learning‑curve metrics (results CSV).

This evaluates the Phase 3 few‑shot calibration procedure on all test
simulations and writes a CSV suitable for `utils/phase3_plots.py`.

Each row corresponds to a (simulation, N_calib) pair and contains:
  - simulation : simulation name (e.g., "LH_122")
  - N_calib   : number of calibration galaxies used
  - rmse      : RMSE after calibration (dex)
  - bias      : bias after calibration (dex, pred - true)
  - a_sn1     : feedback parameter A_SN1 for that simulation
  - distance_sigma : multivariate distance from training (optional, for analysis)
  - num_galaxies  : number of galaxies in the simulation

The calibration rule matches Phase 3:
  1) Draw N_calib galaxies without replacement.
  2) Estimate a constant bias delta = mean(SM_true - SM_pred_nf) on them.
  3) Apply this constant to all galaxies: SM_cal = SM_pred_nf + delta.
  4) Measure RMSE and bias over the full galaxy set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Ensure local src is importable
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Local imports and paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase3_calibration"

TRAINING_SETS = ["LH_135", "LH_473", "LH_798"]
COSMO_PARAMS_FILE = RAW_DATA_DIR / "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"

# Re‑use SHMR fit from Phase 2
SHMR_JSON = (
    BASE_DIR
    / "outputs"
    / "phase2_universal_scatter_model"
    / "shmr_fit_parameters.json"
)


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
    """
    Return list of (LH_###, lh###) for sims that are processed and not training.
    """
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


def compute_multivariate_distance(
    test_row: pd.Series, training_rows: pd.DataFrame
) -> float:
    """
    Mahalanobis‑like distance in standardized units across four parameters.

    Parameters used: Omega_m, sigma_8, A_SN1, A_AGN1.
    If any std is zero, fall back to absolute difference (std=1 guard).
    """
    params = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1"]
    means = training_rows[params].mean()
    stds = training_rows[params].std().replace(0, 1.0)
    diffs = (test_row[params] - means) / stds
    return float(np.sqrt(np.sum(diffs.values**2)))


def load_shmr_params() -> Tuple[float, float]:
    if not SHMR_JSON.exists():
        raise FileNotFoundError(
            f"SHMR parameters not found at {SHMR_JSON}. "
            "Run the Phase 2 SHMR / universal scatter script first."
        )
    with SHMR_JSON.open("r") as f:
        data = json.load(f)
    params = data["parameters"]
    return float(params["alpha"]), float(params["beta"])


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean bias (pred - true)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return np.nan
    return float(np.mean(y_p - y_t))


def _parse_calib_ns(values: Iterable[int]) -> List[int]:
    ns = sorted(set(int(v) for v in values if int(v) > 0))
    if not ns:
        raise ValueError("At least one positive N_calib value is required.")
    return ns


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #


def evaluate_learning_curves(
    model_run: str,
    num_draws: int,
    calib_ns: List[int],
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate calibration learning curves across all test simulations.

    `calib_ns` is the list of N_calib values (numbers of calibration galaxies)
    to probe for each simulation.
    """
    from src.plots.nf_visualizer import load_nf_artifacts, sample_nf_posterior_for_halos
    from src.utils.common import resolve_device

    cosmo_df = load_cosmo_params()
    training_rows = cosmo_df[cosmo_df["Name"].isin(TRAINING_SETS)]
    test_sims = get_test_simulations(cosmo_df)
    alpha, beta = load_shmr_params()

    if len(test_sims) == 0:
        raise RuntimeError("No test simulations found in processed data.")

    device = resolve_device("cpu")
    artifacts = load_nf_artifacts(model_run, device)
    features = artifacts["features"]
    targets = artifacts["targets"]

    # Determine which target we have
    if "Delta_SM" in targets:
        sm_idx = targets.index("Delta_SM")
        is_delta = True
    elif "SM_delta" in targets:
        sm_idx = targets.index("SM_delta")
        is_delta = True
    elif "SM" in targets:
        sm_idx = targets.index("SM")
        is_delta = False
    else:
        raise ValueError(
            f"Model must predict 'Delta_SM'/'SM_delta' or 'SM'; got targets={targets}"
        )

    rows: List[Dict] = []
    base_rng = np.random.default_rng(rng_seed)

    print(f"Evaluating Phase 3 learning curves for {len(test_sims)} test simulations...")
    print(f"Calibration N values: {calib_ns}")

    for sim_idx, (sim_upper, sim_dir) in enumerate(test_sims):
        parquet_path = PROCESSED_DATA_DIR / sim_dir / "halo_galaxy.parquet"
        if not parquet_path.exists():
            print(f"  [SKIP] Missing parquet for {sim_upper} at {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)
        if not set(features).issubset(df.columns) or "SM" not in df.columns:
            print(f"  [SKIP] Missing required columns for {sim_upper}")
            continue

        halos = df[features].values
        sm_true = df["SM"].values

        # Sample NF posterior once per simulation
        post = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=num_draws,
            device=device,
            batch_size=256,
        )

        target_samples = post[:, :, sm_idx]
        target_median = np.median(target_samples, axis=1)

        if is_delta:
            # Convert Delta_SM -> SM via SHMR
            sm_shmr = alpha * df["M_h"].values + beta
            sm_pred_nf = sm_shmr + target_median
        else:
            sm_pred_nf = target_median

        cosmo_row = cosmo_df[cosmo_df["Name"] == sim_upper].iloc[0]
        distance = compute_multivariate_distance(cosmo_row, training_rows)
        a_sn1 = float(cosmo_row["A_SN1"])

        n_gal = len(df)
        # Derive a per‑sim RNG to keep draws reproducible but independent
        rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))

        for n_calib in calib_ns:
            if n_gal < n_calib:
                # Not enough galaxies; skip this N for this sim
                continue

            idx = rng.choice(n_gal, size=n_calib, replace=False)
            sm_true_obs = sm_true[idx]
            sm_pred_obs = sm_pred_nf[idx]

            # Constant calibration offset (few‑shot bias correction)
            delta = float(np.mean(sm_true_obs - sm_pred_obs))
            sm_cal = sm_pred_nf + delta

            rmse_cal = compute_rmse(sm_true, sm_cal)
            bias_cal = compute_bias(sm_true, sm_cal)

            rows.append(
                {
                    "simulation": sim_upper,
                    "N_calib": int(n_calib),
                    "rmse": rmse_cal,
                    "bias": bias_cal,
                    "a_sn1": a_sn1,
                    "distance_sigma": distance,
                    "num_galaxies": n_gal,
                }
            )

        print(
            f"  {sim_upper}: processed {len(calib_ns)} N values "
            f"(n_gal={n_gal}, A_SN1={a_sn1:.3f}, distance={distance:.2f})"
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Phase 3 few‑shot calibration learning‑curve CSV."
    )
    parser.add_argument(
        "--model",
        default="delta_sm",
        help="NF run name that predicts Delta_SM / SM (default: delta_sm).",
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=500,
        help="Posterior draws for NF sampling (default: 500).",
    )
    parser.add_argument(
        "--calib-Ns",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 50, 100],
        help=(
            "List of N calibration galaxies to evaluate (default: 1 3 5 10 20 50 100). "
            "Use log‑spaced or custom values if desired."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "phase3_learning_curves.csv",
        help=(
            "Output CSV path "
            "(default: outputs/phase3_calibration/phase3_learning_curves.csv)."
        ),
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Base RNG seed for calibration sampling (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    calib_ns = _parse_calib_ns(args.calib_Ns)
    df = evaluate_learning_curves(
        model_run=args.model,
        num_draws=args.num_draws,
        calib_ns=calib_ns,
        rng_seed=args.rng_seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()


