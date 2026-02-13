"""
Phase 6: Feedback Meter - Inverse Design / Parameter Inference

Goal: Determine if the "Calibration Bias" (delta) is a direct proxy for the 
Supernova Feedback parameter (A_SN1).

Steps:
1. Load master training set and SHMR fit parameters
2. Compute per-simulation bias delta for each Golden sim and test dataset
3. Correlate bias with A_SN1 and predict test dataset's A_SN1

Configuration:
  - Set TEST_DATASET at top of script to change test dataset (e.g., "lh1", "lh2", "lh3")
  - Dataset should be in: data/processed/{TEST_DATASET}/halo_galaxy.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
# Try phase2_universal_scatter_model first (current), then phase2_shmr (legacy)
SHMR_PARAMS_PATH = BASE_DIR / "outputs" / "phase2_universal_scatter_model" / "shmr_fit_parameters.json"
SHMR_PARAMS_PATH_ALT = BASE_DIR / "outputs" / "phase2_shmr" / "shmr_fit_parameters.json"

# ============================================================================
# Configuration: Test Dataset
# ============================================================================
# Set the test dataset name (e.g., "lh1", "lh2", "lh3", etc.)
# The dataset should be in: data/processed/{TEST_DATASET}/halo_galaxy.parquet
TEST_DATASET = "lh500"  # Default: LH_1, change this to test with different datasets

OUTPUTS_DIR = BASE_DIR / "outputs" / f"phase6_feedback_meter_{TEST_DATASET}"

# Golden training simulations
GOLDEN_SIMS = ["lh135", "lh473", "lh798", "lh844"]


def load_shmr_params(path: Path) -> Tuple[float, float]:
    """Load SHMR power-law parameters (alpha, beta)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    params = data["parameters"]
    return float(params["alpha"]), float(params["beta"])


def load_feedback_params(path: Path) -> Dict[str, Tuple[float, float]]:
    """
    Load feedback parameters from the LH parameter file.
    Returns dict: {sim_name (lowercase): (A_SN1, A_AGN1)}
    """
    params = {}
    with path.open("r") as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 8:
            sim_name = parts[0].strip().lower()  # e.g., "lh_1" -> "lh_1"
            a_sn1 = float(parts[3])
            a_agn1 = float(parts[4])
            params[sim_name] = (a_sn1, a_agn1)
    
    return params


def assign_sim_ids_from_feedback_params(
    df: pd.DataFrame,
    feedback_params: Dict[str, Tuple[float, float]],
    allowed_sims: list
) -> pd.Series:
    """
    Map each row to a Simulation ID using its (A_SN1, A_AGN1) pair.
    We round to 5 decimals to avoid floating noise and only keep allowed_sims.
    """
    allowed = {sim.lower(): feedback_params[sim] for sim in feedback_params if sim.lower() in allowed_sims}
    if not allowed:
        allowed = {sim.lower(): params for sim, params in feedback_params.items()}  # fallback to all sims

    sim_names = list(allowed.keys())
    param_array = np.array([allowed[s] for s in sim_names])  # shape (n_sims, 2)

    def map_row(row):
        pair = np.array([row["A_SN1"], row["A_AGN1"]])
        dists = np.linalg.norm(param_array - pair, axis=1)
        idx = int(np.argmin(dists))
        min_dist = dists[idx]
        if min_dist > 1e-2:  # tolerance
            return f"asn1_{row['A_SN1']:.5f}_agn1_{row['A_AGN1']:.5f}"
        return sim_names[idx]

    return df.apply(map_row, axis=1)


def compute_per_sim_bias(
    master_df: pd.DataFrame,
    alpha: float,
    beta: float,
    sim_ids: list,
    feedback_params: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Compute bias delta for each simulation.
    
    delta = mean(SM_true - (alpha*M_h + beta))
    
    Args:
        master_df: Master training set with 'Simulation', 'M_h', 'SM' columns
        alpha, beta: SHMR power-law parameters
        sim_ids: List of simulation IDs to process
        feedback_params: Dict of feedback parameters keyed by sim name
    
    Returns:
        DataFrame with columns: Simulation, bias_delta, A_SN1, A_AGN1
    """
    results = []
    
    for sim_id in sim_ids:
        # Handle both "lh_135" and "lh135" naming conventions
        sim_name_underscore = f"lh_{sim_id.replace('lh', '')}"
        sim_name_no_underscore = sim_id.lower()

        # Get data for this simulation (SimID column already assigned)
        mask = master_df["SimID"].str.lower() == sim_id.lower()
        df_sim = master_df[mask]
        
        if len(df_sim) == 0:
            print(f"  Warning: No data found for {sim_id}")
            continue
        
        # Compute SHMR prediction and bias
        log_mh = df_sim['M_h'].values
        sm_true = df_sim['SM'].values
        shmr_pred = alpha * log_mh + beta
        
        # Bias: mean(SM_true - SHMR_pred)
        bias_delta = np.mean(sm_true - shmr_pred)
        
        # Look up feedback parameters
        a_sn1, a_agn1 = None, None
        for key in [sim_name_no_underscore, sim_name_underscore]:
            if key in feedback_params:
                a_sn1, a_agn1 = feedback_params[key]
                break
        
        if a_sn1 is None:
            print(f"  Warning: Feedback params not found for {sim_id}")
            continue
        
        results.append({
            'Simulation': sim_id,
            'N_galaxies': len(df_sim),
            'bias_delta': bias_delta,
            'A_SN1': a_sn1,
            'A_AGN1': a_agn1
        })
        
        print(f"  {sim_id}: bias_delta = {bias_delta:+.4f} dex, A_SN1 = {a_sn1:.5f}")
    
    return pd.DataFrame(results)


def fit_and_predict(bias_data: pd.DataFrame, lh1_bias: float) -> Tuple[float, float, float, float]:
    """
    Fit linear regression: A_SN1 = m * delta + c
    Predict LH_1's A_SN1 from its bias.
    
    Returns:
        (predicted_a_sn1, slope, intercept, r_value)
    """
    x = bias_data['bias_delta'].values
    y = bias_data['A_SN1'].values
    
    # Fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Predict LH_1
    predicted_a_sn1 = slope * lh1_bias + intercept
    
    print(f"\nLinear fit: A_SN1 = {slope:.6f} * delta + {intercept:.6f}")
    print(f"  R² = {r_value**2:.6f}")
    print(f"  p-value = {p_value:.6e}")
    print(f"\nLH_1 bias = {lh1_bias:+.4f} dex")
    print(f"Predicted A_SN1 = {predicted_a_sn1:.5f}")
    
    return predicted_a_sn1, slope, intercept, r_value


def plot_feedback_correlation(
    bias_data: pd.DataFrame,
    slope: float,
    intercept: float,
    r_value: float,
    test_bias: float,
    test_true_asn1: float,
    test_pred_asn1: float,
    output_dir: Path,
    test_dataset: str = None
):
    """Create scatter plot of bias vs A_SN1 with fit line."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    ax.scatter(
        bias_data['bias_delta'],
        bias_data['A_SN1'],
        s=150,
        alpha=0.7,
        color='#2E86AB',
        edgecolor='black',
        linewidth=1.5,
        label='Golden training sims',
        zorder=5
    )
    
    # Fit line
    x_range = np.array([bias_data['bias_delta'].min() - 0.1, bias_data['bias_delta'].max() + 0.1])
    y_fit = slope * x_range + intercept
    ax.plot(x_range, y_fit, 'r--', linewidth=2.5, label=f'Linear fit (R² = {r_value**2:.4f})')
    
    if test_dataset is None:
        test_dataset = TEST_DATASET
    
    # Test dataset point (if true value available)
    if test_true_asn1 is not None:
        ax.scatter(
            [test_bias],
            [test_true_asn1],
            s=200,
            marker='*',
            color='gold',
            edgecolor='red',
            linewidth=2,
            label=f'{test_dataset.upper()} (true)',
            zorder=6
        )
    
    # Test dataset prediction
    ax.scatter(
        [test_bias],
        [test_pred_asn1],
        s=200,
        marker='X',
        color='green',
        edgecolor='darkgreen',
        linewidth=2,
        label=f'{test_dataset.upper()} (predicted)',
        zorder=6
    )
    
    # Add annotations
    for idx, row in bias_data.iterrows():
        ax.annotate(
            row['Simulation'].upper(),
            (row['bias_delta'], row['A_SN1']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.7
        )
    
    ax.set_xlabel('Bias (delta) [dex]', fontsize=13, fontweight='bold')
    ax.set_ylabel('A_SN1 (Supernova Feedback)', fontsize=13, fontweight='bold')
    ax.set_title('Feedback Meter: Calibration Bias vs A_SN1', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add equation and error text
    textstr = f'Fit: A_SN1 = {slope:.4f} × delta + {intercept:.4f}\n'
    textstr += f'{test_dataset.upper()} predicted A_SN1 = {test_pred_asn1:.5f}\n'
    if test_true_asn1 is not None:
        textstr += f'{test_dataset.upper()} true A_SN1 = {test_true_asn1:.5f}\n'
        textstr += f'Error = {abs(test_pred_asn1 - test_true_asn1):.5f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "feedback_bias_correlation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Phase 6: Feedback Meter - Parameter Inference")
    print("=" * 80)
    
    # Step 1: Inspect inputs
    print("\n[1] Loading inputs...")
    
    # Load master training set
    master_path = DATA_DIR / "master_training_set.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Master training set not found: {master_path}")
    master_df = pd.read_parquet(master_path)
    print(f"  Master training set: {len(master_df):,} galaxies")
    print(f"  Columns: {master_df.columns.tolist()}")
    
    # Load SHMR parameters (try both possible locations)
    if SHMR_PARAMS_PATH.exists():
        shmr_path = SHMR_PARAMS_PATH
    elif SHMR_PARAMS_PATH_ALT.exists():
        shmr_path = SHMR_PARAMS_PATH_ALT
    else:
        raise FileNotFoundError(
            f"SHMR fit parameters not found. Tried:\n"
            f"  - {SHMR_PARAMS_PATH}\n"
            f"  - {SHMR_PARAMS_PATH_ALT}\n"
            f"Please run phase2_universal_scatter_model.py first."
        )
    alpha, beta = load_shmr_params(shmr_path)
    print(f"  SHMR fit: log(M*) = {alpha:.6f} × log(M_h) + {beta:.6f}")
    
    # Load feedback parameters
    param_file = RAW_DATA_DIR / "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    feedback_params = load_feedback_params(param_file)
    print(f"  Loaded feedback parameters for {len(feedback_params)} simulations")

    # Ensure SimID is present (prefer existing columns, fall back to parameter matching)
    if "SimID" in master_df.columns:
        master_df["SimID"] = master_df["SimID"].str.lower()
    elif "Simulation" in master_df.columns:
        master_df["SimID"] = master_df["Simulation"].str.lower()
    else:
        master_df["SimID"] = assign_sim_ids_from_feedback_params(master_df, feedback_params, GOLDEN_SIMS)
    unique_sims = master_df["SimID"].unique().tolist()
    print(f"  Sim IDs found: {unique_sims}")
    print("  Unique (A_SN1, A_AGN1) pairs:", master_df[["A_SN1", "A_AGN1"]].drop_duplicates().values.tolist())
    
    # Step 2: Compute per-simulation biases
    print("\n[2] Computing per-simulation SHMR biases...")
    
    # Compute for golden sims (using SimID column)
    bias_data = compute_per_sim_bias(master_df, alpha, beta, GOLDEN_SIMS, feedback_params)
    
    # Compute for test dataset
    print(f"\n  Computing {TEST_DATASET.upper()} bias from test parquet...")
    test_path = DATA_DIR / TEST_DATASET / "halo_galaxy.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset '{TEST_DATASET}' parquet not found: {test_path}")
    df_test = pd.read_parquet(test_path)
    log_mh_test = df_test['M_h'].values
    sm_true_test = df_test['SM'].values
    shmr_pred_test = alpha * log_mh_test + beta
    test_bias = np.mean(sm_true_test - shmr_pred_test)
    print(f"  {TEST_DATASET.upper()}: bias_delta = {test_bias:+.4f} dex")
    
    # Get true test dataset A_SN1 (try multiple naming conventions)
    test_name_lower = TEST_DATASET.lower()
    test_key_variants = [
        test_name_lower,  # e.g., "lh500"
        test_name_lower.replace('_', ''),  # e.g., "lh500" (if input was "lh_500")
        f"lh_{test_name_lower.replace('lh', '').replace('_', '')}",  # e.g., "lh_500" (add underscore)
    ]
    
    test_true_asn1 = None
    for variant in test_key_variants:
        if variant in feedback_params:
            test_true_asn1 = feedback_params[variant][0]
            print(f"  Found {TEST_DATASET.upper()} in params as '{variant}'")
            break
    
    if test_true_asn1 is None:
        print(f"  WARNING: {TEST_DATASET.upper()} not found in feedback params (tried: {test_key_variants}), skipping true value check")
    if test_true_asn1 is not None:
        print(f"  {TEST_DATASET.upper()}: true A_SN1 = {test_true_asn1:.5f}")
    
    # Step 3: Fit and predict
    print(f"\n[3] Fitting correlation and predicting {TEST_DATASET.upper()}...")
    pred_asn1, slope, intercept, r_value = fit_and_predict(bias_data, test_bias)
    
    # Step 4: Create plots
    print("\n[4] Creating visualization...")
    plot_feedback_correlation(
        bias_data, slope, intercept, r_value,
        test_bias, test_true_asn1, pred_asn1,
        OUTPUTS_DIR, TEST_DATASET
    )
    
    # Step 5: Save results
    print("\n[5] Saving results...")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save bias table
    csv_path = OUTPUTS_DIR / "feedback_bias_table.csv"
    bias_data.to_csv(csv_path, index=False)
    print(f"  Saved bias table to: {csv_path}")
    
    # Save summary
    summary = {
        'model': 'linear',
        'equation': 'A_SN1 = slope × bias_delta + intercept',
        'parameters': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'r_value': float(r_value)
        },
        f'{TEST_DATASET}_inference': {
            'bias_delta': float(test_bias),
            'predicted_a_sn1': float(pred_asn1),
            'true_a_sn1': float(test_true_asn1) if test_true_asn1 is not None else None,
            'prediction_error': float(abs(pred_asn1 - test_true_asn1)) if test_true_asn1 is not None else None
        },
        'golden_sims': bias_data.to_dict('records')
    }
    
    json_path = OUTPUTS_DIR / "feedback_inference_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to: {json_path}")
    
    print("\n" + "=" * 80)
    print(f"INFERENCE RESULT: {TEST_DATASET.upper()} A_SN1 = {pred_asn1:.5f}", end="")
    if test_true_asn1 is not None:
        print(f" (true: {test_true_asn1:.5f})")
        print(f"Absolute error: {abs(pred_asn1 - test_true_asn1):.5f}")
        print(f"Relative error: {100 * abs(pred_asn1 - test_true_asn1) / test_true_asn1:.2f}%")
    else:
        print(" (true value not available)")
    print("=" * 80)


if __name__ == "__main__":
    main()

