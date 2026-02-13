#!/usr/bin/env python3
"""
Phase 1: Generate Result Plots

Creates comprehensive plots for Phase 1 results across all test simulations.

FIGURE 1: The Extrapolation Landscape ⭐ (Main Result)
Panel A: Performance Heatmap by Conditioning Variable
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.path import Path as MPLPath
from pathlib import Path
import sys
from typing import Dict, List, Tuple

from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plots.nf_visualizer import load_nf_artifacts, sample_nf_posterior_for_halos
from src.utils.common import resolve_device
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs" / "phase1_plots"
COSMO_PARAMS_FILE = RAW_DATA_DIR / "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
RESULTS_CSV = OUTPUT_DIR / "phase1_results.csv"

# Training sets (excluded from test)
TRAINING_SETS = ["LH_135", "LH_473", "LH_798"]

# Model configurations
MODELS = [
    {
        "run_name": "nf_halo_omega",
        "label": "Halo + Omega_m",
        "param_name": "Omega_m",
        "param_col": "Omega_m"
    },
    {
        "run_name": "nf_halo_sigma8",
        "label": "Halo + sigma_8",
        "param_name": "sigma_8",
        "param_col": "sigma_8"
    },
    {
        "run_name": "nf_halo_asn1",
        "label": "Halo + A_SN1",
        "param_name": "A_SN1",
        "param_col": "A_SN1"
    },
    {
        "run_name": "nf_halo_aagn1",
        "label": "Halo + A_AGN1",
        "param_name": "A_AGN1",
        "param_col": "A_AGN1"
    },
]


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_cosmo_params() -> pd.DataFrame:
    """Load cosmological parameters from the seed file."""
    print(f"Loading cosmological parameters from {COSMO_PARAMS_FILE}...")
    
    # Read the file, skipping the header line
    df = pd.read_csv(
        COSMO_PARAMS_FILE,
        sep=r'\s+',
        skiprows=1,
        names=["Name", "Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2", "seed"]
    )
    
    # Clean up the Name column (remove leading spaces)
    df["Name"] = df["Name"].str.strip()
    
    print(f"  Loaded {len(df)} simulations")
    return df


def get_available_processed_datasets() -> List[str]:
    """Get list of all processed dataset directories (lowercase)."""
    available = []
    for item in PROCESSED_DATA_DIR.iterdir():
        if item.is_dir() and (item / "halo_galaxy.parquet").exists():
            available.append(item.name)
    return sorted(available)


def get_test_simulations(cosmo_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Get list of test simulation names (excluding training sets).
    
    Returns:
        List of tuples: (uppercase_name, lowercase_dir_name)
    """
    # Get all processed datasets
    processed_dirs = get_available_processed_datasets()
    
    print(f"\n  Available processed directories: {processed_dirs}")
    print(f"  Training sets to exclude: {TRAINING_SETS}")
    
    # Map processed dirs to cosmo_df names
    test_sims = []
    for proc_dir in processed_dirs:
        # Convert dir name to match cosmological param format
        # lh135 -> LH_135
        if proc_dir.startswith("lh"):
            # Extract the number part
            num_part = proc_dir[2:]  # Remove "lh"
            cosmo_name = f"LH_{num_part}"
        else:
            cosmo_name = proc_dir.upper()
        
        print(f"  Checking {proc_dir} -> {cosmo_name}")
        
        # Check if this is in our cosmological parameters
        matching_rows = cosmo_df[cosmo_df["Name"] == cosmo_name]
        
        if len(matching_rows) > 0:
            # Check if it's NOT in training sets
            if cosmo_name not in TRAINING_SETS:
                test_sims.append((cosmo_name, proc_dir))
                print(f"    [OK] Added {cosmo_name}")
            else:
                print(f"    [SKIP] Skipped {cosmo_name} (training set)")
        else:
            print(f"    [NOT_FOUND] Not found in cosmo_df")
    
    print(f"\n  Found {len(test_sims)} test simulations with data")
    for upper, lower in test_sims:
        print(f"    {upper} (dir: {lower})")
    
    return test_sims


def compute_distance_from_training(
    test_value: float,
    training_values: np.ndarray
) -> float:
    """
    Compute distance from training sets as standard deviation.
    
    Args:
        test_value: Parameter value for test simulation
        training_values: Array of parameter values for training simulations
    
    Returns:
        Distance in units of standard deviation (σ)
    """
    mean_train = np.mean(training_values)
    std_train = np.std(training_values)
    
    if std_train == 0:
        # If training sets have no variance, return absolute difference
        return abs(test_value - mean_train)
    
    # Distance in units of standard deviation
    distance_sigma = abs(test_value - mean_train) / std_train
    return distance_sigma


# ============================================================================
# Model Evaluation Functions
# ============================================================================

def load_test_data(sim_dir: str, features: List[str], targets: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load test dataset for a specific simulation directory."""
    data_path = PROCESSED_DATA_DIR / sim_dir / "halo_galaxy.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Test dataset '{sim_dir}' not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    missing_features = [f for f in features if f not in df.columns]
    missing_targets = [t for t in targets if t not in df.columns]
    
    if missing_features or missing_targets:
        raise ValueError(f"Missing columns in {sim_dir}: features={missing_features}, targets={missing_targets}")
    
    halos = df[features].values
    galaxies = df[targets].values
    
    return df, halos, galaxies


def compute_rmse(true_sm: np.ndarray, pred_sm: np.ndarray) -> float:
    """Compute RMSE for stellar mass predictions."""
    mask = np.isfinite(true_sm) & np.isfinite(pred_sm)
    true_clean = true_sm[mask]
    pred_clean = pred_sm[mask]
    
    if len(true_clean) == 0:
        return np.nan
    
    rmse = float(np.sqrt(np.mean((pred_clean - true_clean) ** 2)))
    return rmse


def compute_bias(true_sm: np.ndarray, pred_sm: np.ndarray) -> float:
    """Compute mean bias (pred - true)."""
    mask = np.isfinite(true_sm) & np.isfinite(pred_sm)
    true_clean = true_sm[mask]
    pred_clean = pred_sm[mask]
    if len(true_clean) == 0:
        return np.nan
    return float(np.mean(pred_clean - true_clean))


def compute_pearson_r(true_sm: np.ndarray, pred_sm: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    mask = np.isfinite(true_sm) & np.isfinite(pred_sm)
    true_clean = true_sm[mask]
    pred_clean = pred_sm[mask]
    if len(true_clean) < 2:
        return np.nan
    try:
        r = float(np.corrcoef(true_clean, pred_clean)[0, 1])
    except Exception:
        r = np.nan
    return r


def evaluate_model_on_simulation(
    model_info: Dict,
    sim_name_upper: str,
    sim_dir_lower: str,
    cosmo_df: pd.DataFrame,
    device: torch.device,
    n_samples: int = 1000
) -> Dict:
    """
    Evaluate a model on a test simulation.
    
    Args:
        model_info: Model configuration dictionary
        sim_name_upper: Uppercase simulation name (e.g., "LH_135")
        sim_dir_lower: Lowercase directory name (e.g., "lh135")
        cosmo_df: Cosmological parameters dataframe
        device: Torch device
        n_samples: Number of posterior samples
    
    Returns:
        Dictionary with:
        - sim_name: Simulation name
        - distance_sigma: Distance from training (σ)
        - param_value: Conditioning parameter value
        - rmse: RMSE (dex)
        - bias: Bias (dex)
        - pearson_r: Pearson correlation
    """
    print(f"  Evaluating {model_info['label']} on {sim_name_upper}...")
    
    try:
        # Load model
        artifacts = load_nf_artifacts(model_info["run_name"], device)
        artifacts["device"] = device
        features = artifacts["features"]
        targets = artifacts["targets"]
        
        # Load test data using lowercase directory name
        df_test, halos, galaxies = load_test_data(sim_dir_lower, features, targets)
        
        # Get parameter value for this simulation
        sim_row = cosmo_df[cosmo_df["Name"] == sim_name_upper].iloc[0]
        param_value = sim_row[model_info["param_col"]]
        
        # Get training parameter values
        training_rows = cosmo_df[cosmo_df["Name"].isin(TRAINING_SETS)]
        training_values = training_rows[model_info["param_col"]].values
        
        # Compute distance from training
        distance_sigma = compute_distance_from_training(param_value, training_values)
        
        # Generate predictions
        sm_index = targets.index("SM")
        true_sm = galaxies[:, sm_index]
        
        posteriors = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=n_samples,
            device=device,
            batch_size=256,
        )
        
        sm_samples = posteriors[:, :, sm_index]
        pred_sm = np.median(sm_samples, axis=1)
        
        # Compute metrics
        rmse = compute_rmse(true_sm, pred_sm)
        bias = compute_bias(true_sm, pred_sm)
        pearson_r = compute_pearson_r(true_sm, pred_sm)
        
        return {
            "sim_name": sim_name_upper,
            "distance_sigma": distance_sigma,
            "param_value": param_value,
            "rmse": rmse,
            "bias": bias,
            "pearson_r": pearson_r,
        }
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_figure1_panel_a(results: Dict[str, List[Dict]], output_path: Path):
    """
    Create Figure 1, Panel A: Performance Heatmap by Conditioning Variable
    
    4 subplots (2×2 grid), one per model:
    - Top-left: Halo + Ωm
    - Top-right: Halo + σ8
    - Bottom-left: Halo + A_SN1
    - Bottom-right: Halo + A_AGN1
    
    For each subplot:
    - X-axis: Distance from training (σ)
    - Y-axis: RMSE (dex)
    - Scatter plot of test sims
    - Color points by conditioning variable value
    - Horizontal line at RMSE = 1.0 (failure threshold)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, model_info in enumerate(MODELS):
        ax = axes[idx]
        model_results = results[model_info["run_name"]]
        
        # Filter out None results
        valid_results = [r for r in model_results if r is not None]
        
        if len(valid_results) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_info["label"], fontsize=14, fontweight='bold')
            continue
        
        # Extract data
        distances = [r["distance_sigma"] for r in valid_results]
        rmses = [r["rmse"] for r in valid_results]
        param_values = [r["param_value"] for r in valid_results]
        
        # Create scatter plot colored by parameter value
        scatter = ax.scatter(
            distances,
            rmses,
            c=param_values,
            cmap='viridis',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            zorder=3
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(model_info["param_name"], fontsize=11, fontweight='bold')
        
        # Add horizontal line at RMSE = 1.0 (failure threshold)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Failure threshold', zorder=2)
        
        # Labels and title
        ax.set_xlabel('Distance from training (σ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (dex)', fontsize=12, fontweight='bold')
        ax.set_title(model_info["label"], fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, zorder=1)
        
        # Set reasonable limits
        if len(distances) > 0:
            ax.set_xlim(left=0)
            if max(rmses) > 0:
                ax.set_ylim(bottom=0, top=max(1.2, max(rmses) * 1.1))
        
        # Legend
        ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved Figure 1 Panel A: {output_path}")


def _pick_representative_indices(a_sn1_vals: np.ndarray) -> List[int]:
    """Pick indices closest to target A_SN1 values for crash plots."""
    targets = np.array([0.3, 1.2, 2.5, 4.0])
    idxs = []
    for t in targets:
        diffs = np.abs(a_sn1_vals - t)
        idx = int(np.argmin(diffs))
        if idx not in idxs:
            idxs.append(idx)
    return idxs


def plot_figure2_mode_collapse(
    results: Dict[str, List[Dict]],
    output_path: Path,
    num_draws: int = 1000,
) -> None:
    """
    Figure 2: Mode Collapse Discovery
    Panel A: RMSE vs A_SN1
    Panel B: Bias vs A_SN1
    """
    model_key = "nf_halo_asn1"
    model_results = results.get(model_key, [])
    valid_results = [r for r in model_results if r is not None]

    if len(valid_results) == 0:
        print("No results for A_SN1 model; skipping Figure 2")
        return

    a_sn1_vals = np.array([r["param_value"] for r in valid_results])
    rmses = np.array([r["rmse"] for r in valid_results])
    biases = np.array([r["bias"] for r in valid_results])
    distances = np.array([r["distance_sigma"] for r in valid_results])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: RMSE vs A_SN1
    ax = axes[0, 0]
    sc1 = ax.scatter(
        a_sn1_vals, rmses, c=distances, cmap="viridis", s=100,
        edgecolors="black", linewidth=1.2
    )
    ax.axvline(x=1.16, color="red", linestyle="--", linewidth=2, label="Training mean A_SN1 = 1.16")
    ax.set_xlabel("True A_SN1", fontsize=12, fontweight="bold")
    ax.set_ylabel("RMSE (dex)", fontsize=12, fontweight="bold")
    ax.set_title("RMSE vs A_SN1 (Feedback Asymmetry)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=ax)
    cbar1.set_label("Distance from training (sigma)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    # Panel B: Bias vs A_SN1
    ax = axes[0, 1]
    sc2 = ax.scatter(
        a_sn1_vals, biases, c=distances, cmap="viridis", s=100,
        edgecolors="black", linewidth=1.2
    )
    ax.axhline(y=0.0, color="red", linestyle="--", linewidth=2, label="Bias = 0")
    ax.set_xlabel("True A_SN1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bias (dex)", fontsize=12, fontweight="bold")
    ax.set_title("Bias vs A_SN1 (Mode Collapse Signature)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(sc2, ax=ax)
    cbar2.set_label("Distance from training (sigma)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    # Panel C: Pearson R vs A_SN1
    pearsons = np.array([r.get("pearson_r", np.nan) for r in valid_results])
    ax = axes[1, 0]
    sc3 = ax.scatter(
        a_sn1_vals, pearsons, c=distances, cmap="viridis", s=100,
        edgecolors="black", linewidth=1.2
    )
    ax.axhline(y=0.9, color="red", linestyle="--", linewidth=2, label="R = 0.9")
    ax.set_xlabel("True A_SN1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pearson R", fontsize=12, fontweight="bold")
    ax.set_title("Pearson R vs A_SN1 (Correlation Breakdown)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(sc3, ax=ax)
    cbar3.set_label("Distance from training (sigma)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)

    # Panel D: Example Crash Plots (proper 2×2 grid)
    ax_main = axes[1, 1]
    # Remove the placeholder axis and replace with a 2x2 sub-grid
    ax_spec = ax_main.get_subplotspec()
    ax_main.remove()
    sub_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax_spec, wspace=0.25, hspace=0.25)
    sub_axes = []
    for i in range(2):
        for j in range(2):
            sub_axes.append(fig.add_subplot(sub_gs[i, j]))

    device = resolve_device("cpu")
    artifacts = load_nf_artifacts("nf_halo_asn1", device)
    artifacts["device"] = device
    features = artifacts["features"]
    targets = artifacts["targets"]
    sm_idx = targets.index("SM")

    idxs = _pick_representative_indices(a_sn1_vals)
    titles = ["Low A_SN1 (~0.3)", "Medium A_SN1 (~1.2)", "High A_SN1 (~2.5)", "Very high A_SN1 (~4.0)"]

    for ax_in, idx, title in zip(sub_axes, idxs, titles):
        sim_name = valid_results[idx]["sim_name"]
        # Map 'LH_122' -> 'lh122'
        if sim_name.startswith("LH_"):
            num_part = sim_name.split("_", 1)[1]
            sim_dir = f"lh{num_part}"
        else:
            sim_dir = sim_name.lower()

        df_test, halos, galaxies = load_test_data(sim_dir, features, targets)
        true_sm = galaxies[:, sm_idx]
        posteriors = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=num_draws,
            device=device,
            batch_size=256,
        )
        sm_samples = posteriors[:, :, sm_idx]
        pred_sm = np.median(sm_samples, axis=1)

        ax_in.scatter(true_sm, pred_sm, s=6, alpha=0.35, color="#1f77b4", rasterized=True)
        lims = [
            min(true_sm.min(), pred_sm.min()) - 0.2,
            max(true_sm.max(), pred_sm.max()) + 0.2,
        ]
        ax_in.plot(lims, lims, "r--", linewidth=1.0, alpha=0.8)
        ax_in.set_xlim(lims)
        ax_in.set_ylim(lims)
        ax_in.set_title(title, fontsize=10)
        ax_in.tick_params(labelsize=8)
        ax_in.grid(alpha=0.2)

    # Add an overall title for the crash plot panel
    fig.text(0.78, 0.52, "Example Crash Plots (A_SN1 model)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Figure 2: {output_path}")


# ============================================================================
# Figure 3: Physics Axis Decomposition
# ============================================================================

def _build_sim_result_table(results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Dict]]:
    """Re-index results as sim_name -> model_name -> result dict."""
    table: Dict[str, Dict[str, Dict]] = {}
    for model_name, model_results in results.items():
        for res in model_results:
            if res is None:
                continue
            sim = res["sim_name"]
            table.setdefault(sim, {})[model_name] = res
    return table


def _collect_joint_grid(
    sim_table: Dict[str, Dict[str, Dict]],
    model_x: str,
    model_y: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect paired distances and mean RMSE for two models."""
    xs, ys, rmses = [], [], []
    for res in sim_table.values():
        if model_x not in res or model_y not in res:
            continue
        rx, ry = res[model_x], res[model_y]
        xs.append(rx["distance_sigma"])
        ys.append(ry["distance_sigma"])
        rmses.append(np.mean([rx["rmse"], ry["rmse"]]))
    return np.array(xs), np.array(ys), np.array(rmses)


def _add_contours(ax, xs, ys, vals):
    """Deprecated helper kept for backward compatibility (unused)."""
    if len(xs) == 0:
        return None
    return None


def _plot_hull_interpolated_heatmap(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    vals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
):
    """
    GridData + ConvexHull masking heatmap with contours and data points.
    Falls back to scatter if fewer than 3 points.
    """
    if len(xs) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        return None

    if len(xs) < 3:
        scatter = ax.scatter(
            xs,
            ys,
            c=vals,
            cmap="viridis",
            s=80,
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Mean RMSE (dex)", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        return scatter

    pts = np.column_stack([xs, ys])
    hull = ConvexHull(pts)
    hull_path = MPLPath(pts[hull.vertices])

    x_pad = max(0.05 * (xs.max() - xs.min()), 1e-3)
    y_pad = max(0.05 * (ys.max() - ys.min()), 1e-3)
    xi = np.linspace(xs.min() - x_pad, xs.max() + x_pad, 220)
    yi = np.linspace(ys.min() - y_pad, ys.max() + y_pad, 220)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((xs, ys), vals, (Xi, Yi), method="cubic")
    mask_points = np.column_stack([Xi.ravel(), Yi.ravel()])
    inside = hull_path.contains_points(mask_points).reshape(Xi.shape)
    Zi_masked = np.where(inside, Zi, np.nan)

    contourf = ax.contourf(Xi, Yi, Zi_masked, levels=20, cmap="viridis")
    ax.contour(Xi, Yi, Zi_masked, levels=[0.5, 1.0, 1.5], colors="white", linewidths=2)
    ax.scatter(
        xs,
        ys,
        c=vals,
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidth=1.2,
        zorder=3,
    )

    # Draw convex hull boundary (light dashed)
    for simplex in hull.simplices:
        ax.plot(pts[simplex, 0], pts[simplex, 1], "r--", linewidth=1, alpha=0.25)

    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label("Mean RMSE (dex)", fontsize=11, fontweight="bold")

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    return contourf


def plot_figure3_physics_decomposition(
    results: Dict[str, List[Dict]],
    output_path: Path,
) -> None:
    """
    Figure 3: Physics Axis Decomposition
    Panels:
    A) Cosmology models performance matrix (Ωm vs σ8 distances)
    B) Feedback models performance matrix (A_SN1 vs A_AGN1 distances)
    C) Best model per simulation (grouped bars)
    D) Model failure types (pie)
    """
    sim_table = _build_sim_result_table(results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Cosmology matrix
    ax = axes[0, 0]
    xs, ys, vals = _collect_joint_grid(sim_table, "nf_halo_omega", "nf_halo_sigma8")
    _plot_hull_interpolated_heatmap(
        ax,
        xs,
        ys,
        vals,
        xlabel="Ωm distance from training (σ)",
        ylabel="σ8 distance from training (σ)",
        title="Cosmology Models Performance Matrix",
    )

    # Panel B: Feedback matrix
    ax = axes[0, 1]
    xs, ys, vals = _collect_joint_grid(sim_table, "nf_halo_asn1", "nf_halo_aagn1")
    _plot_hull_interpolated_heatmap(
        ax,
        xs,
        ys,
        vals,
        xlabel="A_SN1 distance from training (σ)",
        ylabel="A_AGN1 distance from training (σ)",
        title="Feedback Models Performance Matrix",
    )

    # Panel C: Best model per simulation
    ax = axes[1, 0]
    sims = sorted(
        sim_table.keys(),
        key=lambda s: np.mean([res["distance_sigma"] for res in sim_table[s].values()]) if len(sim_table[s]) else np.inf,
    )
    model_order = ["nf_halo_omega", "nf_halo_sigma8", "nf_halo_asn1", "nf_halo_aagn1"]
    model_labels = ["Ωm", "σ8", "A_SN1", "A_AGN1"]
    x = np.arange(len(sims))
    width = 0.18

    # Colors: base + highlight for best per sim
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, model_name in enumerate(model_order):
        offsets = x + (idx - 1.5) * width
        rmses = []
        colors = []
        for sim in sims:
            sim_res = sim_table.get(sim, {})
            rmse_val = sim_res.get(model_name, {}).get("rmse", np.nan)
            rmses.append(rmse_val)
            # Determine best model for this sim
            sim_rmses = {m: r["rmse"] for m, r in sim_res.items()}
            best_model = min(sim_rmses, key=sim_rmses.get) if sim_rmses else None
            if best_model == model_name:
                colors.append("#2ca02c")  # highlight best
            else:
                colors.append(base_colors[idx])
        ax.bar(offsets, rmses, width=width, color=colors, label=model_labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(sims, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (dex)", fontsize=12, fontweight="bold")
    ax.set_title("Best Model per Simulation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel D: Model failure types
    ax = axes[1, 1]
    categories = {
        "All models work (<0.5)": 0,
        "Cosmology works, Feedback fails": 0,
        "Feedback works, Cosmology fails": 0,
        "All models fail (>1.0)": 0,
    }
    for sim_res in sim_table.values():
        cosmo_rmses = [
            sim_res.get("nf_halo_omega", {}).get("rmse", np.inf),
            sim_res.get("nf_halo_sigma8", {}).get("rmse", np.inf),
        ]
        fb_rmses = [
            sim_res.get("nf_halo_asn1", {}).get("rmse", np.inf),
            sim_res.get("nf_halo_aagn1", {}).get("rmse", np.inf),
        ]
        cosmo_ok = all(r < 0.5 for r in cosmo_rmses)
        fb_ok = all(r < 0.5 for r in fb_rmses)
        cosmo_fail = all(r > 1.0 for r in cosmo_rmses)
        fb_fail = all(r > 1.0 for r in fb_rmses)
        if cosmo_ok and fb_ok:
            categories["All models work (<0.5)"] += 1
        elif cosmo_ok and fb_fail:
            categories["Cosmology works, Feedback fails"] += 1
        elif fb_ok and cosmo_fail:
            categories["Feedback works, Cosmology fails"] += 1
        elif cosmo_fail and fb_fail:
            categories["All models fail (>1.0)"] += 1
        else:
            # Mixed/partial performance falls into dominant failure
            if np.nanmean(cosmo_rmses) <= np.nanmean(fb_rmses):
                categories["Cosmology works, Feedback fails"] += 1
            else:
                categories["Feedback works, Cosmology fails"] += 1

    labels = list(categories.keys())
    sizes = list(categories.values())
    ax.pie(sizes, labels=labels, autopct="%d", startangle=140)
    ax.set_title("Model Failure Types", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Figure 3: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 plotting utility")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["figure1", "figure2", "figure3", "all"],
        default=["all"],
        help="Which figures to generate (default: all)"
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Reuse existing phase1_results.csv instead of recomputing."
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=1000,
        help="Number of posterior draws per halo (default: 1000)."
    )
    return parser.parse_args()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    args = parse_args()

    requested_figs = set()
    if "all" in args.figures:
        requested_figs = {"figure1", "figure2", "figure3"}
    else:
        requested_figs = set(args.figures)

    print("="*80)
    print("Phase 1: Generate Result Plots")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load cosmological parameters
    cosmo_df = load_cosmo_params()
    
    # Get test simulations
    test_sims = get_test_simulations(cosmo_df)
    
    if len(test_sims) == 0:
        print("ERROR: No test simulations found!")
        return
    
    print(f"\nTest simulations: {test_sims}")
    
    # Initialize device
    device = resolve_device("cpu")
    print(f"\nUsing device: {device}")
    
    # Load or compute results
    results: Dict[str, List[Dict]] = {model["run_name"]: [] for model in MODELS}

    reuse = args.reuse_results and RESULTS_CSV.exists()
    if reuse:
        df_res = pd.read_csv(RESULTS_CSV)
        if "bias" in df_res.columns and "pearson_r" in df_res.columns:
            print(f"\nReusing existing results from {RESULTS_CSV}")
            for _, row in df_res.iterrows():
                results[row["model"]].append({
                    "sim_name": row["simulation"],
                    "distance_sigma": row["distance_sigma"],
                    "param_value": row["param_value"],
                    "rmse": row["rmse"],
                    "bias": row["bias"],
                    "pearson_r": row["pearson_r"],
                })
        else:
            print(f"\nExisting results CSV at {RESULTS_CSV} is missing bias/pearson_r; recomputing metrics.")
            reuse = False

    if not reuse:
        # Evaluate all models on all test simulations
        print("\n" + "="*80)
        print("Evaluating Models on Test Simulations")
        print("="*80)

        for model_info in MODELS:
            print(f"\n{'-'*70}")
            print(f"Model: {model_info['label']}")
            print(f"{'-'*70}")
            
            for sim_name_upper, sim_dir_lower in test_sims:
                result = evaluate_model_on_simulation(
                    model_info,
                    sim_name_upper,
                    sim_dir_lower,
                    cosmo_df,
                    device,
                    n_samples=args.num_draws
                )
                
                if result is not None:
                    results[model_info["run_name"]].append(result)
                    print(f"    {sim_name_upper}: RMSE={result['rmse']:.4f} dex, "
                          f"distance={result['distance_sigma']:.2f} sigma, "
                          f"bias={result['bias']:.4f} dex, "
                          f"{model_info['param_name']}={result['param_value']:.4f}")

        # Save results to CSV for reference
        rows = []
        for model_name, model_results in results.items():
            for result in model_results:
                if result is not None:
                    rows.append({
                        "model": model_name,
                        "simulation": result["sim_name"],
                        "distance_sigma": result["distance_sigma"],
                        "param_value": result["param_value"],
                        "rmse": result["rmse"],
                        "bias": result["bias"],
                        "pearson_r": result["pearson_r"],
                    })

        results_df = pd.DataFrame(rows)
        results_df.to_csv(RESULTS_CSV, index=False)
        print(f"[OK] Saved results CSV: {RESULTS_CSV}")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating Plots")
    print("="*80)
    
    # Figure 1, Panel A
    if "figure1" in requested_figs:
        figure1a_path = OUTPUT_DIR / "figure1_panel_a_extrapolation_landscape.png"
        plot_figure1_panel_a(results, figure1a_path)

    # Figure 2: Mode Collapse Discovery
    if "figure2" in requested_figs:
        figure2_path = OUTPUT_DIR / "figure2_mode_collapse.png"
        plot_figure2_mode_collapse(results, figure2_path, num_draws=args.num_draws)

    # Figure 3: Physics Axis Decomposition
    if "figure3" in requested_figs:
        figure3_path = OUTPUT_DIR / "figure3_physics_axis.png"
        plot_figure3_physics_decomposition(results, figure3_path)
    
    print("\n" + "="*80)
    print("PHASE 1 PLOTS COMPLETE!")
    print("="*80)
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

