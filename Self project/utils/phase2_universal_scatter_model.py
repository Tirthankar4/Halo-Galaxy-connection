#!/usr/bin/env python3
"""
Phase 2: Universal Scatter Model

A unified script for computing SHMR, residuals, and training NF on scatter.

Parts:
  2.1: Compute Stellar-Halo Mass Relation (SHMR) from training set
  2.2: Compute residuals (deltas) = actual SM - SHMR prediction
  2.3: Create Phase 2 training dataset with residuals as target
  2.4: Validate NF-predicted SHMR vs empirical truth
  2.5: Validate NF-predicted size-mass relations
  2.6: Predict deltas for test dataset (utility)

Formula: M* = SHMR(Mh) + Δ_NF(Halo)

Goal: Create a universal scatter emulator that works across simulations
without conditioning on feedback parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_DIR
from src.plots.nf_visualizer import load_nf_artifacts, sample_nf_posterior_for_halos
from src.utils.common import resolve_device

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_universal_scatter_model"

GOLDEN_SIMS = ["lh135", "lh473", "lh798", "lh844"]
N_BINS = 50
TEST_DATASET = "lh135"  # Configurable test dataset

# Optional NF models for validation and predictions
# Part 2.4: validate_nf_shmr() needs "baseline_multi" model
# Part 2.5: validate_nf_size_mass() needs "baseline_multi" model
# Part 2.6: predict_delta_sm_test() needs "delta_sm" model (available in models/nf/)


# ============================================================================
# PART 2.1: SHMR Computation
# ============================================================================

def load_master_training_set() -> pd.DataFrame:
    """Load the master training parquet file."""
    parquet_path = PROCESSED_DATA_DIR / "master_training_set.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Master training set not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded master training set: {len(df):,} galaxies")
    return df


def compute_binned_median_shmr(df: pd.DataFrame) -> tuple:
    """Compute median stellar mass in bins of halo mass."""
    mask = np.isfinite(df['M_h']) & np.isfinite(df['SM'])
    log_mh = df.loc[mask, 'M_h'].values
    log_sm = df.loc[mask, 'SM'].values
    
    mh_min = np.percentile(log_mh, 1)
    mh_max = np.percentile(log_mh, 99)
    
    bin_edges = np.linspace(mh_min, mh_max, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    median_sm = np.zeros(N_BINS)
    std_sm = np.zeros(N_BINS)
    percentile_16 = np.zeros(N_BINS)
    percentile_84 = np.zeros(N_BINS)
    counts = np.zeros(N_BINS, dtype=int)
    
    for i in range(N_BINS):
        mask_bin = (log_mh >= bin_edges[i]) & (log_mh < bin_edges[i + 1])
        sm_in_bin = log_sm[mask_bin]
        
        if len(sm_in_bin) > 0:
            median_sm[i] = np.median(sm_in_bin)
            std_sm[i] = np.std(sm_in_bin)
            percentile_16[i] = np.percentile(sm_in_bin, 16)
            percentile_84[i] = np.percentile(sm_in_bin, 84)
            counts[i] = len(sm_in_bin)
        else:
            median_sm[i] = np.nan
            std_sm[i] = np.nan
            percentile_16[i] = np.nan
            percentile_84[i] = np.nan
            counts[i] = 0
    
    return bin_centers, median_sm, std_sm, percentile_16, percentile_84, counts, bin_edges


def power_law_func(log_mh: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Power law: log(M*) = α * log(Mh) + β"""
    return alpha * log_mh + beta


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple:
    """Fit a power law to raw data: y = α*x + β.
    
    Args:
        x: Independent variable (e.g., log(M_h))
        y: Dependent variable (e.g., log(M*))
    
    Returns:
        (alpha, beta, r_squared) - power law parameters and R²
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) == 0:
        return np.nan, np.nan, 0.0
    
    # Use linear regression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_clean, y_clean)
    r_squared = r_value ** 2
    
    return slope, intercept, r_squared


def fit_shmr_power_law(bin_centers: np.ndarray, median_sm: np.ndarray, counts: np.ndarray) -> tuple:
    """Fit a power law to the binned SHMR data."""
    mask = np.isfinite(median_sm) & (counts > 0)
    x = bin_centers[mask]
    y = median_sm[mask]
    weights = np.sqrt(counts[mask])
    
    popt, pcov = curve_fit(power_law_func, x, y, sigma=1/weights, p0=[1.0, 0.0])
    alpha, beta = popt
    
    y_pred = power_law_func(x, alpha, beta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return alpha, beta, r_squared


def save_shmr_fit(alpha: float, beta: float, r_squared: float):
    """Save SHMR fit parameters."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fit_params = {
        'model': 'power_law',
        'equation': 'log(M_star) = alpha * log(M_h) + beta',
        'parameters': {
            'alpha': float(alpha),
            'beta': float(beta)
        },
        'statistics': {
            'r_squared': float(r_squared),
        }
    }
    
    json_path = OUTPUT_DIR / "shmr_fit_parameters.json"
    with open(json_path, 'w') as f:
        json.dump(fit_params, f, indent=2)
    print(f"✓ Saved SHMR fit parameters")


# ============================================================================
# PART 2.2: Delta Computation
# ============================================================================

def compute_deltas(df: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    """Compute residuals (deltas) for all galaxies."""
    df_out = df.copy()
    df_out['SM_SHMR_pred'] = power_law_func(df_out['M_h'], alpha, beta)
    df_out['SM_delta'] = df_out['SM'] - df_out['SM_SHMR_pred']
    return df_out


# ============================================================================
# PART 2.3: Create Phase 2 Dataset
# ============================================================================

def create_phase2_dataset(df_with_deltas: pd.DataFrame) -> pd.DataFrame:
    """Create Phase 2 dataset with residuals as target."""
    features = ['M_h', 'R_h', 'V_h', 'sigma_8']
    missing = [f for f in features + ['SM_delta'] if f not in df_with_deltas.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    phase2_df = df_with_deltas[features + ['SM_delta']].copy()
    phase2_df = phase2_df.rename(columns={'SM_delta': 'Delta_SM'})
    phase2_df = phase2_df.dropna()
    
    return phase2_df


# ============================================================================
# PART 2.4: Validate NF-Predicted SHMR
# ============================================================================

def validate_nf_shmr(test_dataset: str = "lh1"):
    """Validate NF-predicted SHMR vs empirical truth.
    
    Requires: models/nf/baseline_multi/ (trained on SM, SFR, SR, Colour targets)
    """
    print("\n" + "="*70)
    print("PART 2.4: NF-Predicted SHMR Validation")
    print("="*70)
    
    # Load test data
    df_test = pd.read_parquet(PROCESSED_DATA_DIR / test_dataset / "halo_galaxy.parquet")
    print(f"Loaded {test_dataset}: {len(df_test)} galaxies")
    
    # Load NF model and predict
    device = resolve_device("cpu")
    try:
        artifacts = load_nf_artifacts("baseline_multi", device)
        features = artifacts["features"]
        targets = artifacts["targets"]
        
        halos = df_test[features].values
        posteriors = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=500,
            device=device,
            batch_size=256,
        )
        
        sm_idx = targets.index("SM")
        sm_pred_nf = np.median(posteriors[:, :, sm_idx], axis=1)
        
        # Fit SHMR to NF predictions
        alpha_nf, beta_nf, r2_nf = fit_power_law(df_test['M_h'].values, sm_pred_nf)
        
        # Fit SHMR to true data
        alpha_true, beta_true, r2_true = fit_power_law(df_test['M_h'].values, df_test['SM'].values)
        
        print(f"\n  NF-predicted SHMR: log(M*) = {alpha_nf:.6f} × log(Mh) + {beta_nf:.6f}")
        print(f"  Empirical SHMR:   log(M*) = {alpha_true:.6f} × log(Mh) + {beta_true:.6f}")
        print(f"  Difference: Δα = {abs(alpha_nf - alpha_true):.6f}, Δβ = {abs(beta_nf - beta_true):.6f}")
        
    except FileNotFoundError:
        print(f"  Skipped: NF model 'baseline_multi' not found in models/nf/")
        print(f"           This is optional - baseline_multi should predict SM, SFR, SR, Colour")
    except Exception as e:
        print(f"  Skipped: Could not validate NF SHMR ({type(e).__name__})")


# ============================================================================
# PART 2.5: Validate NF Size-Mass Relations
# ============================================================================

def validate_nf_size_mass(test_dataset: str = "lh1"):
    """Validate NF-predicted size-mass relations.
    
    Requires: models/nf/baseline_multi/ (trained on SM, SFR, SR, Colour targets)
    """
    print("\n" + "="*70)
    print("PART 2.5: NF Size-Mass Relation Validation")
    print("="*70)
    
    # Load test data
    df_test = pd.read_parquet(PROCESSED_DATA_DIR / test_dataset / "halo_galaxy.parquet")
    print(f"Loaded {test_dataset}: {len(df_test)} galaxies")
    
    # Check if size column exists
    if 'SR' not in df_test.columns:
        print(f"  Skipped: SR column not found in {test_dataset}")
        return
    
    # Load NF model and predict
    device = resolve_device("cpu")
    try:
        artifacts = load_nf_artifacts("baseline_multi", device)
        features = artifacts["features"]
        targets = artifacts["targets"]
        
        if 'SR' not in targets:
            print(f"  Skipped: SR not in NF targets")
            return
        
        halos = df_test[features].values
        posteriors = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=500,
            device=device,
            batch_size=256,
        )
        
        sr_idx = targets.index("SR")
        sr_pred_nf = np.median(posteriors[:, :, sr_idx], axis=1)
        
        # Fit power law: log(SR) vs log(M*)
        alpha_sm, beta_sm, r2_sm = fit_power_law(df_test['SM'].values, sr_pred_nf)
        
        print(f"\n  NF Size-Stellar Mass: log(SR) = {alpha_sm:.6f} × log(M*) + {beta_sm:.6f}")
        
    except FileNotFoundError:
        print(f"  Skipped: NF model 'baseline_multi' not found in models/nf/")
        print(f"           This is optional - baseline_multi should predict SM, SFR, SR, Colour")
    except Exception as e:
        print(f"  Skipped: Could not validate NF size-mass ({type(e).__name__})")


# ============================================================================
# PART 2.6: Predict Deltas for Test Dataset
# ============================================================================

def predict_delta_sm_test(test_dataset: str = "lh1", model_run_name: str = "delta_sm"):
    """Predict Delta_SM for test dataset using trained model.
    
    Requires: models/nf/delta_sm/ 
    - Features: M_h, R_h, V_h, sigma_8
    - Target: Delta_SM (residuals after subtracting SHMR)
    """
    print("\n" + "="*70)
    print(f"PART 2.6: Predict Deltas for {test_dataset.upper()}")
    print("="*70)
    
    # Load test data
    df_test = pd.read_parquet(PROCESSED_DATA_DIR / test_dataset / "halo_galaxy.parquet")
    print(f"Loaded {test_dataset}: {len(df_test)} galaxies")
    
    # Load NF model
    device = resolve_device("cpu")
    try:
        artifacts = load_nf_artifacts(model_run_name, device)
        features = artifacts["features"]
        targets = artifacts["targets"]
        
        if 'Delta_SM' not in targets and 'SM_delta' not in targets:
            print(f"  Skipped: Model does not predict deltas")
            return
        
        halos = df_test[features].values
        posteriors = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=500,
            device=device,
            batch_size=256,
        )
        
        # Get delta predictions
        if 'Delta_SM' in targets:
            delta_idx = targets.index("Delta_SM")
        else:
            delta_idx = targets.index("SM_delta")
        
        delta_median = np.median(posteriors[:, :, delta_idx], axis=1)
        
        # Save predictions
        df_test['Delta_SM_pred'] = delta_median
        
        output_path = OUTPUT_DIR / f"delta_predictions_{test_dataset}.parquet"
        df_test[['M_h', 'SM', 'Delta_SM_pred']].to_parquet(output_path)
        print(f"✓ Saved delta predictions to {output_path}")
        
    except FileNotFoundError:
        print(f"  Skipped: NF model '{model_run_name}' not found in models/nf/")
        print(f"           Available models: delta_sm, nf_halo_*, robustness_check/*")
        print(f"           This is optional - used for predicting stellar mass residuals")
    except Exception as e:
        print(f"  Skipped: Could not predict deltas ({type(e).__name__})")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("Phase 2: Universal Scatter Model")
    print("="*80)
    
    # Part 2.1: Compute SHMR
    print("\n" + "="*80)
    print("PART 2.1: Compute SHMR")
    print("="*80)
    
    df = load_master_training_set()
    
    print("\nComputing binned median SHMR...")
    bin_centers, median_sm, std_sm, p16, p84, counts, bin_edges = compute_binned_median_shmr(df)
    
    mask_valid = np.isfinite(median_sm) & (counts > 0)
    print(f"  Bins with data: {np.sum(mask_valid)}")
    print(f"  M_h range: [{bin_centers[mask_valid].min():.3f}, {bin_centers[mask_valid].max():.3f}]")
    
    print("\nFitting power law to binned SHMR...")
    alpha, beta, r_squared = fit_shmr_power_law(bin_centers, median_sm, counts)
    print(f"  Power law: log(M*) = {alpha:.6f} × log(Mh) + {beta:.6f}")
    print(f"  R² = {r_squared:.6f}")
    
    save_shmr_fit(alpha, beta, r_squared)
    
    # Part 2.2: Compute Deltas
    print("\n" + "="*80)
    print("PART 2.2: Compute Residuals (Deltas)")
    print("="*80)
    
    print("Computing deltas for all galaxies...")
    df_with_deltas = compute_deltas(df, alpha, beta)
    print(f"  Delta statistics:")
    print(f"    Mean: {df_with_deltas['SM_delta'].mean():.6f}")
    print(f"    Std: {df_with_deltas['SM_delta'].std():.6f}")
    print(f"    Range: [{df_with_deltas['SM_delta'].min():.3f}, {df_with_deltas['SM_delta'].max():.3f}]")
    
    # Part 2.3: Create Phase 2 Dataset
    print("\n" + "="*80)
    print("PART 2.3: Create Phase 2 Training Dataset")
    print("="*80)
    
    print("Creating Phase 2 dataset with residuals as target...")
    phase2_df = create_phase2_dataset(df_with_deltas)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2_path = OUTPUT_DIR / "phase2_training_set.parquet"
    phase2_df.to_parquet(phase2_path, index=False)
    print(f"✓ Saved Phase 2 training set: {len(phase2_df):,} samples")
    
    # Save full dataset with deltas
    full_path = OUTPUT_DIR / "training_set_with_deltas.parquet"
    df_with_deltas.to_parquet(full_path, index=False)
    print(f"✓ Saved full dataset with deltas")
    
    # Part 2.4: Validate NF SHMR
    print("\n" + "="*80)
    print("PART 2.4: NF SHMR Validation")
    print("="*80)
    validate_nf_shmr(TEST_DATASET)
    
    # Part 2.5: Validate NF Size-Mass
    print("\n" + "="*80)
    print("PART 2.5: NF Size-Mass Validation")
    print("="*80)
    validate_nf_size_mass(TEST_DATASET)
    
    # Part 2.6: Predict Deltas
    print("\n" + "="*80)
    print("PART 2.6: Predict Deltas for Test Datasets")
    print("="*80)
    predict_delta_sm_test(TEST_DATASET)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 2 COMPLETE!")
    print("="*80)
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
