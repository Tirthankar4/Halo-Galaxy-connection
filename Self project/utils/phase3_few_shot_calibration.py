#!/usr/bin/env python3
"""
Phase 3: Few-Shot Calibration (Core Contribution)

A unified script for few-shot calibration with reconstruction tests.

Parts:
  3.1: Observation Step - Sample N galaxies from target
  3.2: Bootstrap Calibration - Robustness check via bootstrapping
  3.3: Calibration Bias - Compute constant + apply to full set
  3.4: Cheat Reconstruction - Test using target's own SHMR (upper bound)
  3.5: Blind Reconstruction - Test using training SHMR (realistic scenario)

Formula: M*,calib = SHMR_train + Δ_pred + δ

Goal: Demonstrate that calibrating with just 10 observations dramatically
improves predictions and enables cross-simulation transfer.

Discovery: Few-shot calibration is the key to making universal emulators practical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance, gaussian_kde

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.plots.nf_visualizer import load_nf_artifacts, sample_nf_posterior_for_halos
from src.utils.common import resolve_device
from src.config import PROCESSED_DATA_DIR

# ============================================================================
# Configuration
# ============================================================================
TEST_DATASET = "lh1"
OUTPUT_DIR = BASE_DIR / "outputs" / f"phase3_few_shot_calibration_{TEST_DATASET}"

MODEL_RUN_NAME = "delta_sm"  # Change to "baseline_multi" if available, or "robustness_check/LOO_0"
NUM_DRAWS = 1000
DEVICE = "cpu"
N_BOOT = 100
N_SAMPLE = 50
N_OBS = 10
RNG_SEED = 42


# ============================================================================
# PART 3.1: Observation Sampling
# ============================================================================

def load_test_data() -> pd.DataFrame:
    """Load test dataset."""
    test_path = PROCESSED_DATA_DIR / TEST_DATASET / "halo_galaxy.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset '{TEST_DATASET}' not found: {test_path}")
    
    df = pd.read_parquet(test_path)
    print(f"Loaded {TEST_DATASET.upper()} data: {len(df):,} galaxies")
    return df


def sample_observations(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sample N_OBS galaxies from the dataset."""
    if len(df) < N_OBS:
        raise ValueError(f"Dataset too small: {len(df)} rows")
    
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(df), size=N_OBS, replace=False)
    
    sample = df.iloc[idx].copy()
    sample.insert(0, "sample_idx", sample.index)
    sample = sample[["sample_idx", "M_h", "SM"]]
    return sample, idx


def save_observation_sample(sample: pd.DataFrame) -> Tuple[Path, Path]:
    """Save sampled galaxies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / f"observed_galaxies_{TEST_DATASET}.parquet"
    csv_path = OUTPUT_DIR / f"observed_galaxies_{TEST_DATASET}.csv"
    
    sample.to_parquet(parquet_path, index=False)
    sample.to_csv(csv_path, index=False)
    return parquet_path, csv_path


# ============================================================================
# PART 3.2: Bootstrap Calibration
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_bootstrap_calibration(
    df: pd.DataFrame,
    sm_true: np.ndarray,
    sm_pred_nf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform bootstrap calibration."""
    rng = np.random.default_rng(RNG_SEED)
    biases = []
    rmses_boot = []
    
    for i in range(N_BOOT):
        idx = rng.choice(len(df), size=N_SAMPLE, replace=False)
        bias = float(np.mean(sm_true[idx] - sm_pred_nf[idx]))
        sm_cal = sm_pred_nf + bias
        rmses_boot.append(rmse(sm_true, sm_cal))
        biases.append(bias)
    
    return np.array(biases), np.array(rmses_boot)


def save_bootstrap_results(
    biases: np.ndarray,
    rmses: np.ndarray,
) -> Tuple[Path, Path, Path]:
    """Save bootstrap results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_path = OUTPUT_DIR / "bootstrap_calibration_results.csv"
    pd.DataFrame({"bias": biases, "rmse": rmses}).to_csv(results_path, index=False)
    
    metrics = {
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std()),
        "rmse_min": float(rmses.min()),
        "rmse_max": float(rmses.max()),
        "bias_mean": float(biases.mean()),
        "bias_std": float(biases.std()),
    }
    metrics_path = OUTPUT_DIR / "bootstrap_calibration_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    plt.figure(figsize=(6, 4))
    plt.hist(rmses, bins=15, color="#1f77b4", edgecolor="black", alpha=0.75)
    plt.xlabel("RMSE (dex)")
    plt.ylabel("Count")
    plt.title(f"Bootstrap RMSE ({N_BOOT} runs, sample size {N_SAMPLE})")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    hist_path = OUTPUT_DIR / "bootstrap_rmse_hist.png"
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return results_path, metrics_path, hist_path


# ============================================================================
# PART 3.3: Calibration Bias
# ============================================================================

def compute_calibration_bias(
    sample_obs: pd.DataFrame,
    df: pd.DataFrame,
    sm_true: np.ndarray,
    sm_pred_nf: np.ndarray,
) -> Tuple[float, Dict]:
    """Compute calibration constant from observations."""
    obs_indices = sample_obs["sample_idx"].values
    
    sm_true_obs = sm_true[obs_indices]
    sm_pred_obs = sm_pred_nf[obs_indices]
    
    bias_constant = float(np.mean(sm_true_obs - sm_pred_obs))
    
    details = {
        "n_observations": int(len(obs_indices)),
        "bias_constant": bias_constant,
        "per_galaxy": []
    }
    
    for idx in obs_indices:
        details["per_galaxy"].append({
            "original_index": int(idx),
            "SM_true": float(sm_true[idx]),
            "SM_pred": float(sm_pred_nf[idx]),
            "residual": float(sm_true[idx] - sm_pred_nf[idx])
        })
    
    residuals = sm_true_obs - sm_pred_obs
    details["residual_mean"] = float(np.mean(residuals))
    details["residual_std"] = float(np.std(residuals))
    details["residual_min"] = float(np.min(residuals))
    details["residual_max"] = float(np.max(residuals))
    
    return bias_constant, details


def apply_calibration(df: pd.DataFrame, sm_true: np.ndarray, sm_pred_nf: np.ndarray, bias_constant: float) -> pd.DataFrame:
    """Apply calibration to full dataset."""
    result = df[["M_h", "SM"]].copy()
    
    sm_calibrated = sm_pred_nf + bias_constant
    
    result["SM_pred_uncalibrated"] = sm_pred_nf
    result["SM_pred_calibrated"] = sm_calibrated
    result["residual_uncalibrated"] = sm_true - sm_pred_nf
    result["residual_calibrated"] = sm_true - sm_calibrated
    
    return result


# ============================================================================
# PART 3.4: Cheat Reconstruction
# ============================================================================

def cheat_reconstruction(df: pd.DataFrame, sm_true: np.ndarray):
    """Test reconstruction using target's own SHMR (upper bound)."""
    print("\n" + "="*70)
    print("PART 3.4: Cheat Reconstruction (Using Target's Own SHMR)")
    print("="*70)
    
    # Compute target's own SHMR
    mask = np.isfinite(df['M_h']) & np.isfinite(df['SM'])
    log_mh = df.loc[mask, 'M_h'].values
    log_sm = df.loc[mask, 'SM'].values
    
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_mh, log_sm)
    
    # Predict SM using cheat SHMR
    sm_shmr_cheat = slope * df['M_h'].values + intercept
    
    # Compute metrics
    rmse_cheat = rmse(sm_true, sm_shmr_cheat)
    print(f"  Cheat SHMR RMSE: {rmse_cheat:.4f} dex")
    print(f"  (This is the upper bound - best possible with target's SHMR)")


# ============================================================================
# PART 3.5: Blind Reconstruction
# ============================================================================

def blind_reconstruction(df: pd.DataFrame, sm_true: np.ndarray):
    """Test reconstruction using training SHMR (realistic scenario)."""
    print("\n" + "="*70)
    print("PART 3.5: Blind Reconstruction (Using Training SHMR)")
    print("="*70)
    
    # Load training SHMR fit
    shmr_path = BASE_DIR / "outputs" / "phase2_universal_scatter_model" / "shmr_fit_parameters.json"
    if not shmr_path.exists():
        print(f"  Skipping: SHMR parameters not found at {shmr_path}")
        return
    
    with open(shmr_path) as f:
        shmr_data = json.load(f)
    
    alpha = shmr_data['parameters']['alpha']
    beta = shmr_data['parameters']['beta']
    
    # Predict SM using training SHMR
    sm_shmr_train = alpha * df['M_h'].values + beta
    
    # Compute metrics
    rmse_blind = rmse(sm_true, sm_shmr_train)
    print(f"  Blind SHMR RMSE: {rmse_blind:.4f} dex")
    print(f"  (This is the baseline - using only training SHMR)")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("Phase 3: Few-Shot Calibration (Core Contribution)")
    print(f"Test Dataset: {TEST_DATASET.upper()}")
    print("="*80)
    
    # Load data
    df = load_test_data()
    
    # PART 3.1: Sample observations
    print("\n" + "="*80)
    print("PART 3.1: Observation Step")
    print("="*80)
    
    print(f"\nSampling {N_OBS} 'observed' galaxies...")
    sample_obs, obs_indices = sample_observations(df)
    parquet_path, csv_path = save_observation_sample(sample_obs)
    print(f"✓ Saved sampled galaxies to {parquet_path}")
    
    # Load model and compute predictions
    print("\n" + "="*80)
    print("Loading NF Model and Computing Predictions")
    print("="*80)
    
    device = resolve_device(DEVICE)
    print(f"Device: {device}")
    print(f"Model: {MODEL_RUN_NAME}")
    
    artifacts = load_nf_artifacts(MODEL_RUN_NAME, device)
    features = artifacts["features"]
    targets = artifacts["targets"]
    
    # Handle both "SM" and "Delta_SM" targets
    if "SM" in targets:
        sm_idx = targets.index("SM")
        target_name = "SM"
        is_delta = False
    elif "Delta_SM" in targets:
        sm_idx = targets.index("Delta_SM")
        target_name = "Delta_SM"
        is_delta = True
    else:
        raise ValueError(f"Expected 'SM' or 'Delta_SM' in targets, but got: {targets}")
    
    halos = df[features].values
    
    print(f"Sampling NF posteriors ({NUM_DRAWS} draws per halo)...")
    posterior = sample_nf_posterior_for_halos(
        halos=halos,
        artifacts=artifacts,
        num_draws=NUM_DRAWS,
        device=device,
        batch_size=256,
    )
    
    delta_samples = posterior[:, :, sm_idx]
    delta_pred_nf = np.median(delta_samples, axis=1)
    
    # If model predicts Delta_SM, convert to SM by adding SHMR prediction
    if is_delta:
        # Load SHMR parameters to convert Delta_SM to SM
        shmr_path = BASE_DIR / "outputs" / "phase2_universal_scatter_model" / "shmr_fit_parameters.json"
        if not shmr_path.exists():
            # Try alternative path
            shmr_path = BASE_DIR / "outputs" / "phase2_shmr" / "shmr_fit_parameters.json"
        
        if shmr_path.exists():
            with open(shmr_path) as f:
                shmr_data = json.load(f)
            alpha = shmr_data['parameters']['alpha']
            beta = shmr_data['parameters']['beta']
            
            # Compute SHMR prediction: SM_shmr = alpha * M_h + beta
            sm_shmr = alpha * df['M_h'].values + beta
            
            # Convert Delta_SM to SM: SM = SHMR + Delta_SM
            sm_pred_nf = sm_shmr + delta_pred_nf
        else:
            raise FileNotFoundError(
                f"SHMR parameters not found. Tried:\n"
                f"  - {BASE_DIR / 'outputs' / 'phase2_universal_scatter_model' / 'shmr_fit_parameters.json'}\n"
                f"  - {BASE_DIR / 'outputs' / 'phase2_shmr' / 'shmr_fit_parameters.json'}\n"
                f"Please run phase2_universal_scatter_model.py first."
            )
    else:
        sm_pred_nf = delta_pred_nf
    
    sm_true = df["SM"].values
    
    print(f"  SM true range: [{sm_true.min():.3f}, {sm_true.max():.3f}]")
    print(f"  SM pred range: [{sm_pred_nf.min():.3f}, {sm_pred_nf.max():.3f}]")
    
    # PART 3.2: Bootstrap calibration
    print("\n" + "="*80)
    print("PART 3.2: Bootstrap Calibration Robustness")
    print("="*80)
    
    print(f"Running {N_BOOT} bootstrap iterations (sample size {N_SAMPLE})...")
    biases, rmses_boot = compute_bootstrap_calibration(df, sm_true, sm_pred_nf)
    results_path, metrics_path, hist_path = save_bootstrap_results(biases, rmses_boot)
    
    print(f"\n  RMSE: {rmses_boot.mean():.4f} ± {rmses_boot.std():.4f}")
    print(f"  Bias: {biases.mean():.4f} ± {biases.std():.4f} dex")
    print(f"✓ Saved bootstrap results to {results_path}")
    
    # PART 3.3: Compute calibration constant
    print("\n" + "="*80)
    print("PART 3.3: Compute Calibration Constant")
    print("="*80)
    
    bias_constant, calibration_details = compute_calibration_bias(
        sample_obs, df, sm_true, sm_pred_nf
    )
    
    print(f"\n  Calibration Constant: {bias_constant:.4f} dex")
    print(f"  Per-galaxy residuals:")
    for item in calibration_details["per_galaxy"]:
        print(f"    idx {item['original_index']}: {item['residual']:+.4f} dex")
    
    details_path = OUTPUT_DIR / "calibration_details.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with details_path.open("w") as f:
        json.dump(calibration_details, f, indent=2)
    print(f"✓ Saved calibration details")
    
    # Apply calibration to full set
    print(f"\nApplying calibration to full {TEST_DATASET.upper()}...")
    predictions_df = apply_calibration(df, sm_true, sm_pred_nf, bias_constant)
    
    # Compute metrics
    rmse_uncal = rmse(sm_true, sm_pred_nf)
    rmse_cal = rmse(sm_true, sm_pred_nf + bias_constant)
    
    print(f"\n  Uncalibrated RMSE: {rmse_uncal:.4f} dex")
    print(f"  Calibrated RMSE:   {rmse_cal:.4f} dex")
    print(f"  Improvement: {100 * (1 - rmse_cal / rmse_uncal):+.1f}%")
    
    pred_path = OUTPUT_DIR / "calibrated_predictions.parquet"
    predictions_df.to_parquet(pred_path, index=False)
    print(f"✓ Saved calibrated predictions")
    
    # PART 3.4: Cheat reconstruction
    print("\n" + "="*80)
    print("PART 3.4: Cheat Reconstruction")
    print("="*80)
    cheat_reconstruction(df, sm_true)
    
    # PART 3.5: Blind reconstruction
    print("\n" + "="*80)
    print("PART 3.5: Blind Reconstruction")
    print("="*80)
    blind_reconstruction(df, sm_true)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 3 COMPLETE!")
    print("="*80)
    print(f"Key Results:")
    print(f"  - Few-shot calibration with {N_OBS} observations")
    print(f"  - Calibration constant: {bias_constant:.4f} dex")
    print(f"  - RMSE improvement: {100 * (1 - rmse_cal / rmse_uncal):+.1f}%")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
