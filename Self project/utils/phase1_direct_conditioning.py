#!/usr/bin/env python3
"""
Phase 1: Direct Conditioning (Baseline Failure)

A unified script demonstrating why direct conditioning fails for feedback parameters.

Parts:
  1.1: Train and test NF models conditioned on [Halo + single parameter]
  1.2: Fit empirical SHMR baseline for comparison
  1.3: Generate diagnostic plots and metrics

Models tested (all conditioned on Halo + one parameter):
  - Model A: Halo + Ωₘ (cosmology)
  - Model B: Halo + σ₈ (cosmology)
  - Model C: Halo + A_SN1 (feedback)
  - Model D: Halo + A_AGN1 (feedback)

Discovery: Cosmological parameters are safe, but feedback parameters cause severe failure.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plots.nf_visualizer import load_nf_artifacts, sample_nf_posterior_for_halos
from src.utils.common import resolve_device
from src.config import PROCESSED_DATA_DIR

# ============================================================================
# Configuration
# ============================================================================
TEST_DATASET = "lh500"  # Change to test with different datasets
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs" / f"phase1_direct_conditioning_{TEST_DATASET}"


# ============================================================================
# PART 1.1: Direct Conditioning Tests
# ============================================================================

def load_test_data(features, targets):
    """Load test dataset with specified features and targets."""
    data_path = PROCESSED_DATA_DIR / TEST_DATASET / "halo_galaxy.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Test dataset '{TEST_DATASET}' not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    missing_features = [f for f in features if f not in df.columns]
    missing_targets = [t for t in targets if t not in df.columns]
    
    if missing_features or missing_targets:
        raise ValueError(f"Missing columns in {TEST_DATASET}")
    
    halos = df[features].values
    galaxies = df[targets].values
    
    return df, halos, galaxies


def predict_sm(halos, model_artifacts, n_samples=1000):
    """Predict stellar mass from halos using NF model."""
    device = model_artifacts["device"]
    
    posteriors = sample_nf_posterior_for_halos(
        halos=halos,
        artifacts=model_artifacts,
        num_draws=n_samples,
        device=device,
        batch_size=256,
    )
    
    targets = model_artifacts["targets"]
    if "SM" not in targets:
        raise ValueError(f"SM not found in model targets: {targets}")
    
    sm_index = targets.index("SM")
    sm_samples = posteriors[:, :, sm_index]
    sm_median = np.median(sm_samples, axis=1)
    
    return sm_median, sm_samples


def compute_metrics(true_sm, pred_sm):
    """Compute RMSE, Pearson R, and Bias."""
    mask = np.isfinite(true_sm) & np.isfinite(pred_sm)
    true_clean = true_sm[mask]
    pred_clean = pred_sm[mask]
    
    rmse = float(np.sqrt(np.mean((pred_clean - true_clean) ** 2)))
    correlation = float(np.corrcoef(true_clean, pred_clean)[0, 1])
    bias = float(np.mean(pred_clean - true_clean))
    
    return {
        "RMSE": rmse,
        "Pearson_R": correlation,
        "Bias": bias,
    }


def plot_crash_panel(models_data, output_path):
    """Create 1x4 crash plot showing True vs Predicted SM for all models."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for idx, model in enumerate(models_data):
        ax = axes[idx]
        true_sm = model["true_sm"]
        pred_sm = model["pred_sm"]
        metrics = model["metrics"]
        
        ax.scatter(true_sm, pred_sm, s=8, alpha=0.4, color="#1f77b4", rasterized=True)
        
        lims = [
            min(true_sm.min(), pred_sm.min()) - 0.2,
            max(true_sm.max(), pred_sm.max()) + 0.2
        ]
        ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.7, label='1:1 line')
        
        metrics_text = (
            f"RMSE: {metrics['RMSE']:.3f}\n"
            f"R: {metrics['Pearson_R']:.3f}\n"
            f"Bias: {metrics['Bias']:.3f}"
        )
        ax.text(
            0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(r'True $\log_{10}(M_\star/M_\odot)$', fontsize=12)
        if idx == 0:
            ax.set_ylabel(r'Predicted $\log_{10}(M_\star/M_\odot)$', fontsize=12)
        ax.set_title(model["label"], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved crash plot: {output_path}")


def save_scoreboard(models_data, output_path):
    """Save metrics table as CSV."""
    rows = []
    for model in models_data:
        rows.append({
            "Model": model["label"],
            "RMSE (dex)": f"{model['metrics']['RMSE']:.4f}",
            "Pearson R": f"{model['metrics']['Pearson_R']:.4f}",
            "Bias (dex)": f"{model['metrics']['Bias']:.4f}",
        })
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved scoreboard: {output_path}")
    
    print("\n" + "="*70)
    print(f"SCOREBOARD: Phase 1 Direct Conditioning ({TEST_DATASET.upper()})")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


# ============================================================================
# PART 1.2: Empirical SHMR Baseline
# ============================================================================

def fit_power_law(log_mh: np.ndarray, log_sm: np.ndarray) -> tuple:
    """Fit a power law: log(M*) = α*log(Mh) + β"""
    mask = np.isfinite(log_mh) & np.isfinite(log_sm)
    log_mh_clean = log_mh[mask]
    log_sm_clean = log_sm[mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_mh_clean, log_sm_clean)
    predicted = slope * log_mh_clean + intercept
    residuals = log_sm_clean - predicted
    residuals_std = np.std(residuals)
    r_squared = r_value ** 2
    
    return slope, intercept, r_squared, residuals_std


def fit_empirical_shmr(output_dir: Path):
    """Fit empirical SHMR for test dataset as baseline comparison."""
    print("\n" + "="*70)
    print("PART 1.2: Empirical SHMR Baseline")
    print("="*70)
    
    print(f"\nFitting empirical SHMR for {TEST_DATASET.upper()}...")
    df = pd.read_parquet(PROCESSED_DATA_DIR / TEST_DATASET / "halo_galaxy.parquet")
    
    alpha, beta, r_squared, residuals_std = fit_power_law(df['M_h'].values, df['SM'].values)
    
    print(f"  Power law: log(M*) = {alpha:.6f} × log(Mh) + {beta:.6f}")
    print(f"  R² = {r_squared:.6f}")
    print(f"  Residual σ = {residuals_std:.6f}")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    n_sample = min(5000, len(df))
    df_sample = df.sample(n=n_sample, random_state=42)
    
    ax.scatter(df_sample['M_h'], df_sample['SM'], s=5, alpha=0.3, color='#1f77b4', label='Data')
    
    log_mh_range = np.linspace(df['M_h'].min(), df['M_h'].max(), 100)
    log_sm_fit = alpha * log_mh_range + beta
    ax.plot(log_mh_range, log_sm_fit, 'r-', linewidth=2.5, label='Power law fit')
    
    ax.set_xlabel('log₁₀(M_h) [M☉/h]', fontsize=13, fontweight='bold')
    ax.set_ylabel('log₁₀(M_*) [M☉]', fontsize=13, fontweight='bold')
    ax.set_title(f'Empirical SHMR: {TEST_DATASET.upper()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "empirical_shmr.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved empirical SHMR plot")
    
    return alpha, beta, r_squared


# ============================================================================
# PART 1.3: Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("Phase 1: Direct Conditioning (Baseline Failure)")
    print(f"Test Dataset: {TEST_DATASET.upper()}")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Part 1.1: Test direct conditioning models
    print("\n" + "="*80)
    print("PART 1.1: Direct Conditioning Tests")
    print("="*80)
    
    device = resolve_device("cpu")
    
    models = [
        {
            "run_name": "nf_halo_omega",
            "label": "Halo + Ωₘ",
            "param_name": "Omega_m",
            "type": "cosmo"
        },
        {
            "run_name": "nf_halo_sigma8",
            "label": "Halo + σ₈",
            "param_name": "sigma_8",
            "type": "cosmo"
        },
        {
            "run_name": "nf_halo_asn1",
            "label": "Halo + A_SN1",
            "param_name": "A_SN1",
            "type": "feedback"
        },
        {
            "run_name": "nf_halo_aagn1",
            "label": "Halo + A_AGN1",
            "param_name": "A_AGN1",
            "type": "feedback"
        },
    ]
    
    models_data = []
    
    for model_info in models:
        print(f"\n{'='*70}")
        print(f"Testing: {model_info['label']}")
        print(f"{'='*70}")
        
        try:
            print(f"Loading model: {model_info['run_name']}...")
            artifacts = load_nf_artifacts(model_info["run_name"], device)
            artifacts["device"] = device
            features = artifacts["features"]
            targets = artifacts["targets"]
            print(f"  Features: {features}")
            print(f"  Targets: {targets}")
            
            print(f"Loading {TEST_DATASET.upper()} data...")
            df_test, halos, galaxies = load_test_data(features, targets)
            print(f"  Loaded {len(halos)} halos/galaxies")
            
            sm_index = targets.index("SM")
            true_sm = galaxies[:, sm_index]
            print(f"  True SM range: [{true_sm.min():.3f}, {true_sm.max():.3f}]")
            
            print("Generating predictions (1000 samples per halo)...")
            pred_sm, pred_samples = predict_sm(halos, artifacts, n_samples=1000)
            print(f"  Pred SM range: [{pred_sm.min():.3f}, {pred_sm.max():.3f}]")
            
            metrics = compute_metrics(true_sm, pred_sm)
            print(f"  RMSE: {metrics['RMSE']:.4f} dex")
            print(f"  Pearson R: {metrics['Pearson_R']:.4f}")
            print(f"  Bias: {metrics['Bias']:.4f} dex")
            
            models_data.append({
                "name": model_info["run_name"],
                "label": model_info["label"],
                "type": model_info["type"],
                "true_sm": true_sm,
                "pred_sm": pred_sm,
                "metrics": metrics,
            })
            
        except Exception as e:
            print(f"❌ Error processing {model_info['run_name']}: {e}")
            continue
    
    # Generate combined outputs
    if len(models_data) >= 2:
        print(f"\n{'='*70}")
        print("Generating combined outputs...")
        print(f"{'='*70}")
        
        crash_plot_path = OUTPUT_DIR / "crash_plot_1x4.png"
        plot_crash_panel(models_data, crash_plot_path)
        
        scoreboard_path = OUTPUT_DIR / "scoreboard.csv"
        save_scoreboard(models_data, scoreboard_path)
    
    # Part 1.2: Fit empirical SHMR baseline
    print("\n" + "="*80)
    print("PART 1.2: Empirical SHMR Baseline")
    print("="*80)
    
    fit_empirical_shmr(OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE!")
    print("="*80)
    #print(f"Key findings:")
    #print(f"  - Cosmological models show tight diagonal (safe)")
    #print(f"  - Feedback models show 'The Tilt' or 'The Blob' (dangerous)")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
