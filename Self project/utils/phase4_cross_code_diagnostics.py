"""
Phase 5: The "Outlier Detective" - SIMBA LH_1 Analysis

This script applies a TNG-trained model to SIMBA data to fingerprint
physics differences between the two simulations through failure patterns.

The goal is NOT to get good predictions (we expect failure), but to use
the pattern of failure to understand physics differences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.stats import pearsonr
import sys
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.shmr import load_shmr
from src.plots.nf_visualizer import load_nf_artifacts, sample_nf_posterior_for_halos
from src.utils.common import resolve_device
from src.data.preprocess import load_catalogs, preprocess_dataframe

# Constants
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs" / "phase5_outlier_detective"

# Speed of light in m/s (used in old preprocessing method)
C_LIGHT = 3.0 * 10**8

# CAMELS parameters for LH_1 (from CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt)
LH_1_SIGMA_8 = 0.93940


def load_simba_lh1_data() -> pd.DataFrame:
    """
    Load SIMBA LH_1 data from HDF5 file using unified preprocessing.
    
    Returns:
        DataFrame with columns: M_h, R_h, V_h, SM, V_max (log10 scaled)
    """
    print("=" * 80)
    print("Loading SIMBA LH_1 data using unified preprocessing...")
    print("=" * 80)
    
    h5_path = DATA_DIR / "groups_090_lh1_simba.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"SIMBA LH_1 data not found: {h5_path}")
    
    print(f"Loading from: {h5_path}")
    
    # Use unified preprocessing function
    # This is SIMBA data, so explicitly specify simulation type
    sim_type = "SIMBA"
    halos, gals, sim_type = load_catalogs(h5_path, sim_type)
    print(f"Using simulation type: {sim_type}")
    
    # Preprocess with sigma_8 parameter
    # Create a minimal CAMELS params dict for LH_1
    # Format: Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2
    # We only need sigma_8, others can be defaults
    camels_params = {
        "LH_1": np.array([0.3, LH_1_SIGMA_8, 1.0, 1.0, 1.0, 1.0])  # Defaults except sigma_8
    }
    
    df = preprocess_dataframe(halos, gals, camels_params=camels_params, sim_name="LH_1", sim_type=sim_type)
    
    print(f"  Final dataframe shape: {df.shape}")
    print(f"  M_h range: [{df['M_h'].min():.3f}, {df['M_h'].max():.3f}]")
    print(f"  SM range: [{df['SM'].min():.3f}, {df['SM'].max():.3f}]")
    
    return df


def load_simba_lh1_data_old() -> pd.DataFrame:
    """
    OLD METHOD - Kept for reference but not used.
    Load SIMBA LH_1 data from HDF5 file.
    
    Returns:
        DataFrame with columns: M_h, R_h, V_h, SM, V_max (log10 scaled)
    """
    print("=" * 80)
    print("Loading SIMBA LH_1 data...")
    print("=" * 80)
    
    h5_path = DATA_DIR / "groups_090_lh1_simba.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"SIMBA LH_1 data not found: {h5_path}")
    
    print(f"Loading from: {h5_path}")
    
    with h5py.File(h5_path, "r") as f:
        # === LOAD GROUP (HALO) DATA ===
        m_h = f["Group/Group_M_Crit200"][:] * 1e10  # Msun/h
        r_h = f["Group/Group_R_Crit200"][:] / C_LIGHT  # kpc/h
        v_h = f["Group/GroupVel"][:]
        v_h_mag = np.linalg.norm(v_h, axis=1)  # km/s
        group_firstsub = f["Group/GroupFirstSub"][:]  # Index of first subhalo
        
        # === LOAD SUBHALO (GALAXY) DATA ===
        sm = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Stellar mass (Msun/h)
        sfr = f["Subhalo/SubhaloSFR"][:] * 1e10  # Star formation rate
        sr = f["Subhalo/SubhaloHalfmassRadType"][:, 4]  # Half-mass radius (kpc/h)
        subhalo_grnr = f["Subhalo/SubhaloGrNr"][:]  # Parent group global ID
        
        # Try to load V_max if available
        try:
            v_max = f["Subhalo/SubhaloVmax"][:]  # km/s
            has_vmax = True
        except KeyError:
            v_max = None
            has_vmax = False
            print("  Warning: SubhaloVmax not found, will compute from M_h and R_h")
        
        # === MATCH CENTRAL GALAXIES TO HALOS ===
        # Strategy: Group subhalos by their parent group ID (SubhaloGrNr),
        # find the first subhalo (lowest index) for each group - this is the central.
        
        halos = []
        galaxies = []
        
        # Create mapping from global group ID to list of (subhalo_index, stellar_mass)
        grnr_to_subhalos = {}
        for sub_idx in range(len(subhalo_grnr)):
            global_grnr = subhalo_grnr[sub_idx]
            if global_grnr not in grnr_to_subhalos:
                grnr_to_subhalos[global_grnr] = []
            grnr_to_subhalos[global_grnr].append(sub_idx)
        
        # For each group, find the central (first) subhalo
        matched_groups = set()
        for global_grnr in sorted(grnr_to_subhalos.keys()):
            subhalo_indices = sorted(grnr_to_subhalos[global_grnr])
            central_sub_idx = subhalo_indices[0]  # First subhalo is the central
            
            # Skip if no stellar mass
            if sm[central_sub_idx] <= 0:
                continue
            
            # Skip unresolved galaxies (resolution limit)
            sr_kpc = sr[central_sub_idx]
            MIN_RESOLVED_SR = 1.0  # kpc
            if sr_kpc < MIN_RESOLVED_SR:
                continue
            
            # Map global group ID to local group index
            # Try offset: assume local_idx = global_grnr - min_grnr
            min_grnr = min(grnr_to_subhalos.keys()) if grnr_to_subhalos else 0
            local_grp_idx = global_grnr - min_grnr
            
            # Check if this local index is valid and not already matched
            if 0 <= local_grp_idx < len(m_h) and local_grp_idx not in matched_groups:
                matched_groups.add(local_grp_idx)
                
                # Store halo properties
                halos.append({
                    "M_h": m_h[local_grp_idx],
                    "R_h": r_h[local_grp_idx],
                    "V_h": v_h_mag[local_grp_idx],
                    "ID": global_grnr
                })
                
                # Store central galaxy properties
                gal_dict = {
                    "SM": sm[central_sub_idx],
                    "SFR": sfr[central_sub_idx],
                    "SR": sr[central_sub_idx],
                    "ID": global_grnr
                }
                
                # Add V_max if available
                if has_vmax:
                    gal_dict["V_max"] = v_max[central_sub_idx]
                
                galaxies.append(gal_dict)
    
    # Convert to DataFrames
    halos_df = pd.DataFrame(halos)
    gals_df = pd.DataFrame(galaxies)
    
    print(f"  Loaded {len(halos_df)} halos and {len(gals_df)} galaxies")
    
    # Merge halos and galaxies
    df = pd.merge(halos_df, gals_df, on="ID", how="inner")
    
    print(f"  Matched {len(df)} halo-galaxy pairs")
    
    # Log transforms (matching TNG preprocessing)
    # Ensure all values are positive before log transform
    for col in ["M_h", "R_h", "V_h", "SM"]:
        df[col] = df[col].clip(lower=1e-10)
        df[col] = np.log10(df[col])
    
    # Handle SFR zeros before log scaling
    df["SFR"] = df["SFR"].replace(0, 1)
    df["SFR"] = np.log10(df["SFR"])
    
    # Log transform SR
    df["SR"] = np.log10(df["SR"] + 0.001)
    
    # Log transform V_max if available
    if "V_max" in df.columns:
        df["V_max"] = df["V_max"].clip(lower=1e-10)
        df["V_max"] = np.log10(df["V_max"])
    
    # Add sigma_8 (from CAMELS parameters)
    df["sigma_8"] = LH_1_SIGMA_8
    
    # Replace any inf or -inf values with NaN, then fill with reasonable defaults
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().any():
            fill_value = df[col].median() if not np.isnan(df[col].median()) else df[col].mean()
            if not np.isnan(fill_value):
                df[col] = df[col].fillna(fill_value)
    
    print(f"  Final dataframe shape: {df.shape}")
    print(f"  M_h range: [{df['M_h'].min():.3f}, {df['M_h'].max():.3f}]")
    print(f"  SM range: [{df['SM'].min():.3f}, {df['SM'].max():.3f}]")
    
    return df


def predict_delta_simba(df: pd.DataFrame, model_run_name: str = "robustness_check/LOO_3", 
                        device: str = "cuda", n_samples: int = 1000) -> np.ndarray:
    """
    Apply TNG-trained LOO_3 model to SIMBA halos.
    
    Args:
        df: DataFrame with SIMBA halo properties
        model_run_name: Name of the trained model run
        device: Device to use ("cuda" or "cpu")
        n_samples: Number of posterior samples per halo
    
    Returns:
        Array of predicted delta_sm values (mean over samples)
    """
    print("\n" + "=" * 80)
    print("Predicting Delta_SM using TNG-trained model...")
    print("=" * 80)
    
    # Resolve device
    device = resolve_device(device)
    print(f"Using device: {device}")
    
    # Load trained model
    print(f"Loading trained model: {model_run_name}...")
    artifacts = load_nf_artifacts(model_run_name, device)
    features = artifacts["features"]
    targets = artifacts["targets"]
    print(f"  Features: {features}")
    print(f"  Targets: {targets}")
    
    # Check that we have all required features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in SIMBA data: {missing_features}")
    
    # Extract halo features
    halos = df[features].values
    print(f"  Halo features shape: {halos.shape}")
    
    # Generate predictions
    print(f"Generating predictions ({n_samples} samples per halo)...")
    posterior_samples = sample_nf_posterior_for_halos(
        halos=halos,
        artifacts=artifacts,
        num_draws=n_samples,
        device=device,
        batch_size=256
    )
    
    # posterior_samples shape: (N, n_samples, gal_dim)
    # LOO_3 model predicts all 4 targets: ['SM', 'SFR', 'SR', 'Colour']
    # We need to extract SM (index 0) and compute delta_sm = SM_pred - SHMR(M_h)
    
    # Extract SM samples (index 0)
    sm_samples = posterior_samples[:, :, 0]  # (N, n_samples)
    
    # Compute mean SM prediction over samples
    sm_pred_mean = np.mean(sm_samples, axis=1)  # (N,)
    
    # Load SHMR to compute delta_sm = SM_pred - SHMR(M_h)
    from src.utils.shmr import load_shmr
    shmr = load_shmr()
    
    # Get M_h from dataframe
    log_mh = df["M_h"].values
    
    # Predict SHMR
    sm_shmr = shmr.predict(log_mh, method='fit')
    
    # Compute delta_sm = SM_pred - SHMR_pred
    delta_pred = sm_pred_mean - sm_shmr
    
    print(f"  Predictions shape: {delta_pred.shape}")
    print(f"  Mean Delta_SM_pred: {delta_pred.mean():.4f} ± {delta_pred.std():.4f}")
    
    return delta_pred


def compute_true_scatter(df: pd.DataFrame) -> np.ndarray:
    """
    Compute true scatter using TNG SHMR baseline.
    
    Δ_true = log M_*, SIMBA - SHMR_TNG(M_h_SIMBA)
    
    Args:
        df: DataFrame with SIMBA halo and stellar masses
    
    Returns:
        Array of true scatter values
    """
    print("\n" + "=" * 80)
    print("Computing true scatter using TNG SHMR baseline...")
    print("=" * 80)
    
    # Load TNG SHMR
    shmr = load_shmr()
    print(f"  Loaded TNG SHMR: {shmr}")
    
    # Predict stellar mass from TNG SHMR
    sm_shmr_tng = shmr.predict(df["M_h"].values, method='fit')
    
    # Compute true scatter: Δ_true = log M_*, SIMBA - SHMR_TNG(M_h_SIMBA)
    delta_true = df["SM"].values - sm_shmr_tng
    
    print(f"  True scatter shape: {delta_true.shape}")
    print(f"  Mean Δ_true: {delta_true.mean():.4f} ± {delta_true.std():.4f}")
    print(f"  Range: [{delta_true.min():.4f}, {delta_true.max():.4f}]")
    
    return delta_true


def compute_residuals(delta_true: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    """
    Compute residuals: ε = Δ_true - Δ_pred
    
    Args:
        delta_true: True scatter values
        delta_pred: Predicted scatter values
    
    Returns:
        Array of residuals
    """
    residuals = delta_true - delta_pred
    
    print("\n" + "=" * 80)
    print("Residual Statistics:")
    print("=" * 80)
    print(f"  Mean ε: {residuals.mean():.4f}")
    print(f"  Std ε: {residuals.std():.4f}")
    print(f"  Range: [{residuals.min():.4f}, {residuals.max():.4f}]")
    print(f"  Median |ε|: {np.median(np.abs(residuals)):.4f}")
    
    return residuals


def compute_concentration(df: pd.DataFrame) -> np.ndarray:
    """
    Compute halo concentration: V_max / V_vir
    
    Args:
        df: DataFrame with halo properties
    
    Returns:
        Array of concentration values (log10 scale)
    """
    print("\n" + "=" * 80)
    print("Computing halo concentration...")
    print("=" * 80)
    
    # Convert from log10 to linear for computation
    m_h_linear = 10**df["M_h"].values  # Msun/h
    r_h_linear = 10**df["R_h"].values  # kpc/h
    
    # Compute V_vir from M_h and R_h
    # V_vir = sqrt(G * M_h / R_h)
    # G = 4.301e-9 (Msun/h)^-1 (km/s)^2 (kpc/h)
    G = 4.301e-9
    v_vir = np.sqrt(G * m_h_linear / r_h_linear)  # km/s
    
    # Get V_max
    if "V_max" in df.columns:
        v_max_linear = 10**df["V_max"].values  # km/s
        print("  Using V_max from data")
    else:
        # Use V_h as proxy for V_max
        v_max_linear = 10**df["V_h"].values  # km/s
        print("  Using V_h as proxy for V_max")
    
    # Compute concentration: V_max / V_vir
    concentration = v_max_linear / v_vir
    
    # Log transform
    concentration_log = np.log10(concentration)
    
    print(f"  Concentration range: [{concentration_log.min():.4f}, {concentration_log.max():.4f}]")
    print(f"  Mean concentration: {concentration_log.mean():.4f} ± {concentration_log.std():.4f}")
    
    return concentration_log


def plot_mass_bias(df: pd.DataFrame, residuals: np.ndarray, output_dir: Path):
    """
    Plot A: The "Mass Bias" (Global Offset)
    
    X-axis: log₁₀(M_halo)
    Y-axis: Residual ε (True - Pred)
    """
    print("\n" + "=" * 80)
    print("Creating Plot A: Mass Bias...")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Clean data
    mask = np.isfinite(df["M_h"].values) & np.isfinite(residuals)
    m_h_clean = df.loc[mask, "M_h"].values
    residuals_clean = residuals[mask]
    
    # Subsample for visualization if too many points
    if len(m_h_clean) > 10000:
        idx = np.random.choice(len(m_h_clean), 10000, replace=False)
        m_h_clean = m_h_clean[idx]
        residuals_clean = residuals_clean[idx]
    
    # Scatter plot with density coloring
    scatter = ax.scatter(
        m_h_clean,
        residuals_clean,
        c=residuals_clean,
        cmap='coolwarm',
        s=5,
        alpha=0.4,
        rasterized=True,
        vmin=-0.5,
        vmax=0.5
    )
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero residual')
    
    # Add ±1σ lines
    sigma = residuals_clean.std()
    ax.axhline(y=sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'±1σ = ±{sigma:.3f}')
    ax.axhline(y=-sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Residual ε', fontsize=12)
    
    ax.set_xlabel('log₁₀(M_halo) [M☉/h]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Residual ε = Δ_true - Δ_pred', fontsize=13, fontweight='bold')
    ax.set_title('Plot A: The "Mass Bias" (Global Offset)\nSIMBA LH_1: TNG Model Failure Pattern', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    textstr = f'Mean(ε) = {residuals_clean.mean():.4f}\n'
    textstr += f'Std(ε) = {residuals_clean.std():.4f}\n'
    textstr += f'Median(|ε|) = {np.median(np.abs(residuals_clean)):.4f}\n'
    textstr += f'N = {len(residuals_clean):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    output_path = output_dir / "phase5_mass_bias.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def plot_scatter_structure(delta_true: np.ndarray, delta_pred: np.ndarray, output_dir: Path):
    """
    Plot B: The "Scatter Structure" (The Fingerprint)
    
    X-axis: Predicted Scatter (Δ_pred)
    Y-axis: True Scatter (Δ_true)
    """
    print("\n" + "=" * 80)
    print("Creating Plot B: Scatter Structure...")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Clean data
    mask = np.isfinite(delta_true) & np.isfinite(delta_pred)
    delta_true_clean = delta_true[mask]
    delta_pred_clean = delta_pred[mask]
    
    # Subsample for visualization if too many points
    if len(delta_true_clean) > 10000:
        idx = np.random.choice(len(delta_true_clean), 10000, replace=False)
        delta_true_clean = delta_true_clean[idx]
        delta_pred_clean = delta_pred_clean[idx]
    
    # Scatter plot
    ax.scatter(delta_pred_clean, delta_true_clean, alpha=0.3, s=5, rasterized=True, color='#2E86AB')
    
    # y=x line (perfect prediction)
    min_val = min(delta_true_clean.min(), delta_pred_clean.min())
    max_val = max(delta_true_clean.max(), delta_pred_clean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y=x (perfect prediction)')
    
    # Compute Pearson correlation coefficient
    pcc, p_value = pearsonr(delta_pred_clean, delta_true_clean)
    
    # Add text box with metrics
    textstr = f'Pearson Correlation (PCC) = {pcc:.4f}\n'
    textstr += f'p-value = {p_value:.2e}\n'
    textstr += f'N = {len(delta_true_clean):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Predicted Scatter (Δ_pred)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Scatter (Δ_true)', fontsize=13, fontweight='bold')
    ax.set_title('Plot B: The "Scatter Structure" (The Fingerprint)\n' +
                'Correlation indicates universality of DM halo assembly effects',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = output_dir / "phase5_scatter_structure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    print(f"  Pearson Correlation Coefficient: {pcc:.4f}")
    plt.close()
    
    return pcc


def plot_environment_test(df: pd.DataFrame, residuals: np.ndarray, concentration: np.ndarray, output_dir: Path):
    """
    Plot C: The "Environment Test" (The Detective)
    
    X-axis: Halo Concentration (V_max/V_vir)
    Y-axis: Residual ε
    """
    print("\n" + "=" * 80)
    print("Creating Plot C: Environment Test...")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Clean data
    mask = np.isfinite(concentration) & np.isfinite(residuals)
    concentration_clean = concentration[mask]
    residuals_clean = residuals[mask]
    
    # Subsample for visualization if too many points
    if len(concentration_clean) > 10000:
        idx = np.random.choice(len(concentration_clean), 10000, replace=False)
        concentration_clean = concentration_clean[idx]
        residuals_clean = residuals_clean[idx]
    
    # Scatter plot
    scatter = ax.scatter(
        concentration_clean,
        residuals_clean,
        c=residuals_clean,
        cmap='coolwarm',
        s=5,
        alpha=0.4,
        rasterized=True,
        vmin=-0.5,
        vmax=0.5
    )
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero residual')
    
    # Add ±1σ lines
    sigma = residuals_clean.std()
    ax.axhline(y=sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'±1σ = ±{sigma:.3f}')
    ax.axhline(y=-sigma, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Residual ε', fontsize=12)
    
    ax.set_xlabel('Halo Concentration log₁₀(V_max/V_vir)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Residual ε = Δ_true - Δ_pred', fontsize=13, fontweight='bold')
    ax.set_title('Plot C: The "Environment Test" (The Detective)\n' +
                'Environmental dependencies in failure pattern',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Compute correlation
    pcc_env, p_value_env = pearsonr(concentration_clean, residuals_clean)
    
    # Add statistics text box
    textstr = f'Mean(ε) = {residuals_clean.mean():.4f}\n'
    textstr += f'Std(ε) = {residuals_clean.std():.4f}\n'
    textstr += f'PCC(conc, ε) = {pcc_env:.4f}\n'
    textstr += f'N = {len(residuals_clean):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    output_path = output_dir / "phase5_environment_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    print(f"  Correlation (concentration, residual): {pcc_env:.4f}")
    plt.close()
    
    return pcc_env


def main():
    """Main execution function."""
    print("=" * 80)
    print("Phase 5: The 'Outlier Detective' - SIMBA LH_1 Analysis")
    print("=" * 80)
    print("\nGoal: Use pattern of failure to fingerprint physics differences")
    print("      between TNG and SIMBA.")
    print("\nDO NOT calibrate the mean - we want raw failure patterns!")
    print()
    
    # Create output directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load SIMBA LH_1 data
    df = load_simba_lh1_data()
    
    # Predict delta using TNG-trained model
    delta_pred = predict_delta_simba(df, model_run_name="robustness_check/LOO_3", device="cuda", n_samples=1000)
    
    # Compute true scatter using TNG SHMR baseline
    delta_true = compute_true_scatter(df)
    
    # Compute residuals
    residuals = compute_residuals(delta_true, delta_pred)
    
    # Compute concentration
    concentration = compute_concentration(df)
    
    # Add computed values to dataframe
    df["Delta_pred"] = delta_pred
    df["Delta_true"] = delta_true
    df["Residual"] = residuals
    df["Concentration"] = concentration
    
    # Create the 3 diagnostic plots
    plot_mass_bias(df, residuals, OUTPUTS_DIR)
    pcc_scatter = plot_scatter_structure(delta_true, delta_pred, OUTPUTS_DIR)
    pcc_env = plot_environment_test(df, residuals, concentration, OUTPUTS_DIR)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    # Save dataframe
    results_path = OUTPUTS_DIR / "phase5_results.parquet"
    df.to_parquet(results_path, index=False)
    print(f"  Saved results to: {results_path}")
    
    # Save summary
    summary_path = OUTPUTS_DIR / "phase5_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Phase 5: The 'Outlier Detective' - SIMBA LH_1 Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write("Summary Statistics:\n")
        f.write(f"  Number of halos: {len(df):,}\n")
        f.write(f"  Mean residual ε: {residuals.mean():.6f}\n")
        f.write(f"  Std residual ε: {residuals.std():.6f}\n")
        f.write(f"  Median |ε|: {np.median(np.abs(residuals)):.6f}\n")
        f.write(f"\nCorrelations:\n")
        f.write(f"  PCC(Δ_pred, Δ_true): {pcc_scatter:.6f}\n")
        f.write(f"  PCC(Concentration, ε): {pcc_env:.6f}\n")
        f.write(f"\nInterpretation:\n")
        f.write(f"  - If PCC(Δ_pred, Δ_true) > 0.5: Dark matter halo assembly\n")
        f.write(f"    drives scatter in both universes (Universality result)\n")
        f.write(f"  - Mass bias (level shift): Indicates feedback strength difference\n")
        f.write(f"  - Tilt/knee at M_h ≈ 10^12 M☉: SIMBA AGN jet mode (TNG lacks)\n")
    
    print(f"  Saved summary to: {summary_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE! ✓")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUTS_DIR}")
    print("  - phase5_mass_bias.png")
    print("  - phase5_scatter_structure.png")
    print("  - phase5_environment_test.png")
    print("  - phase5_results.parquet")
    print("  - phase5_summary.txt")
    print()
    print("Key Findings:")
    print(f"  Mean residual: {residuals.mean():.4f} (level shift)")
    print(f"  Scatter correlation: {pcc_scatter:.4f}")
    print(f"  Environment correlation: {pcc_env:.4f}")
    print()


if __name__ == "__main__":
    main()

