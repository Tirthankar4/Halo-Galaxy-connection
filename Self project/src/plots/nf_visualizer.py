"""
NF-specific visualization utilities that reproduce the plots previously
generated in the `Plots/` directory from the NF.ipynb notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
import tarp

from src import config
from src.utils.io import ensure_dir
from src.utils.common import compute_regression_metrics, TARGET_COLUMNS, FEATURE_COLUMNS
from src.utils.data import SimpleScaler


sns.set_style("whitegrid")

PROPERTIES = TARGET_COLUMNS
PROPERTY_LABELS = {
    "SM": r"$\log_{10}(M_\star/M_\odot)$",
    "SFR": r"$\log_{10}(\mathrm{SFR})$",
    "SR": r"$\log_{10}(\mathrm{SR})$",
    "Colour": "Colour (g-r)",
}


def build_flow(halo_dim: int, gal_dim: int, device: torch.device):
    """Build flow with configurable dimensions."""
    base = dist.Normal(torch.zeros(halo_dim, device=device), torch.ones(halo_dim, device=device) * 0.2).to_event(1)
    x1_transform = T.spline(halo_dim)
    x3_transform = T.affine_coupling(halo_dim)
    dist_x1 = dist.TransformedDistribution(base, [x1_transform, x3_transform])

    cond_base = dist.Normal(torch.zeros(gal_dim, device=device), torch.ones(gal_dim, device=device) * 0.2).to_event(1)
    x2_transform = T.conditional_spline(gal_dim, context_dim=halo_dim)
    dist_x2_given_x1 = dist.ConditionalTransformedDistribution(cond_base, [x2_transform])

    modules = torch.nn.ModuleList([x1_transform, x3_transform, x2_transform]).to(device)
    return dist_x1, dist_x2_given_x1, modules


def load_nf_artifacts(run_name: str, device: torch.device) -> Dict[str, any]:
    run_dir = Path(config.NF_MODEL_DIR) / run_name
    summary_path = run_dir / "training_summary.json"
    preds_path = run_dir / "predictions.npz"
    weights_path = run_dir / "flow_state.pt"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")

    with summary_path.open("r") as f:
        summary = json.load(f)

    preds = np.load(preds_path)

    # Get features/targets from training summary, fallback to defaults
    features = summary.get("features", FEATURE_COLUMNS)
    targets = summary.get("targets", TARGET_COLUMNS)
    
    halo_dim = len(features)
    gal_dim = len(targets)
    
    logging.info(f"Loading NF trained on {halo_dim} features {features} → {gal_dim} targets {targets}")

    dist_x1, dist_x2_given_x1, modules = build_flow(halo_dim, gal_dim, device)
    state_dict = torch.load(weights_path, map_location=device)
    modules.load_state_dict(state_dict)

    halo_scaler = SimpleScaler(summary["halo_scaler"]["mean"], summary["halo_scaler"]["scale"])
    gal_scaler = SimpleScaler(summary["gal_scaler"]["mean"], summary["gal_scaler"]["scale"])

    return {
        "run_dir": run_dir,
        "dist_x1": dist_x1,
        "dist_x2_given_x1": dist_x2_given_x1,
        "halo_scaler": halo_scaler,
        "gal_scaler": gal_scaler,
        "predictions": preds,
        "features": features,
        "targets": targets,
    }


def load_processed_data(data_path: Path, features: List[str], targets: List[str]):
    """Load data with specific features and targets."""
    df = pd.read_parquet(data_path)
    halos = df[features].values
    gals = df[targets].values
    halos_train, halos_test, gals_train, gals_test = train_test_split(
        halos, gals, test_size=0.2, random_state=42
    )
    return halos_train, halos_test, gals_train, gals_test


def scatter_with_marginals(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_flow: np.ndarray,
    y_flow: np.ndarray,
    output_path: Path,
    title: str,
):
    gs_kw = dict(width_ratios=[2, 1], height_ratios=[1, 2])
    fig, axd = plt.subplot_mosaic(
        [["upper left", "upper right"], ["lower left", "lower right"]],
        gridspec_kw=gs_kw,
        figsize=(8, 8),
        constrained_layout=True,
    )

    axd["upper right"].remove()

    ax = axd["lower left"]
    ax.scatter(x_data, y_data, s=3, alpha=0.6, label="Data", color="#B22222")
    ax.scatter(x_flow, y_flow, s=3, alpha=0.5, label="Flow", color="#1f77b4")
    ax.set_xlabel(r"$\log_{10}(M_{\mathrm{halo}} / M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax_top = axd["upper left"]
    sns.kdeplot(x_data, ax=ax_top, color="#B22222", lw=2)
    sns.kdeplot(x_flow, ax=ax_top, color="#1f77b4", lw=2)
    ax_top.set_xticks([])
    ax_top.set_ylabel("Density")
    ax_top.grid(True, alpha=0.3)

    ax_right = axd["lower right"]
    sns.kdeplot(y=y_data, ax=ax_right, color="#B22222", lw=2)
    sns.kdeplot(y=y_flow, ax=ax_right, color="#1f77b4", lw=2)
    ax_right.set_yticks([])
    ax_right.set_xlabel("Density")
    ax_right.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_conditioned_halo_property(
    artifacts: Dict[str, any],
    output_path: Path,
    property_idx: int,
    property_label: str,
    halo_masses: List[float] | None = None,
    num_samples: int = 1000,
    device: torch.device | None = None,
    targets: List[str] = None,
):
    """
    Reproduce NF.ipynb scatter + KDE plot for fixed halo masses and an arbitrary galaxy property.
    """
    if halo_masses is None:
        halo_masses = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]

    halo_scaler: SimpleScaler = artifacts["halo_scaler"]
    gal_scaler: SimpleScaler = artifacts["gal_scaler"]
    dist_x2_given_x1 = artifacts["dist_x2_given_x1"]

    means_raw = halo_scaler.mean.copy()
    cmap = plt.colormaps["Spectral"]
    xlims = [min(halo_masses) - 0.5, max(halo_masses) + 0.5]

    gs_kw = dict(width_ratios=[2, 1], height_ratios=[1, 2])
    fig, axd = plt.subplot_mosaic(
        [["upper left", "upper right"], ["lower left", "lower right"]],
        gridspec_kw=gs_kw,
        figsize=(8, 8),
        constrained_layout=True,
    )
    axd["upper right"].remove()
    axd["upper left"].remove()

    scatter_ax = axd["lower left"]
    kde_ax = axd["lower right"]

    scatter_ax.set_xlabel(r"$\log_{10}(M_{\mathrm{halo}} / M_\odot)$")
    scatter_ax.set_ylabel(property_label)
    scatter_ax.set_xlim(xlims)

    kde_ax.set_xlabel("Density")

    gathered_values: List[np.ndarray] = []

    if device is None:
        device = torch.device("cpu")

    with torch.no_grad():
        for idx, hmass in enumerate(halo_masses):
            color = cmap(idx / len(halo_masses))
            template = np.tile(means_raw, (num_samples, 1))
            template[:, 0] = hmass
            template_scaled = halo_scaler.transform(template)
            halos_cond_t = torch.tensor(template_scaled, dtype=torch.float32, device=device)

            gals_samples = []
            for halo_vec in halos_cond_t:
                gal_sample = dist_x2_given_x1.condition(halo_vec).sample()
                gals_samples.append(gal_sample.cpu())
            gals_samples = torch.stack(gals_samples).numpy()

            halos_mass = halo_scaler.inverse_transform(halos_cond_t.cpu().numpy())[:, 0]
            gals_property = gal_scaler.inverse_transform(gals_samples)[:, property_idx]

            gathered_values.append(gals_property)

            scatter_ax.scatter(halos_mass, gals_property, s=2, alpha=0.6, color=color, label=str(hmass))
            sns.kdeplot(y=gals_property, ax=kde_ax, color=color, fill=False, linewidth=2)

    all_values = np.concatenate(gathered_values)
    vmin, vmax = np.percentile(all_values, [0.5, 99.5])
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    ylims = [vmin - pad, vmax + pad]
    scatter_ax.set_ylim(ylims)
    kde_ax.set_ylim(ylims)
    kde_ax.set_ylabel(property_label)

    scatter_ax.legend(title=r"$\log_{10}(M_{\mathrm{halo}})$", fontsize=9)
    kde_ax.grid(True, alpha=0.3)
    scatter_ax.grid(True, alpha=0.3)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sample_nf_posterior_for_halos(
    halos: np.ndarray,
    artifacts: Dict[str, any],
    num_draws: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Generate NF posterior samples for an arbitrary set of halos.

    Args:
        halos: Array of halo features (unscaled) with shape (N, halo_dim).
        artifacts: Dictionary returned by load_nf_artifacts.
        num_draws: Number of posterior samples per halo.
        device: Torch device to run sampling on.
        batch_size: Number of halos to process per batch.

    Returns:
        Array of shape (N, num_draws, gal_dim) with unscaled galaxy samples.
    """

    halo_scaler: SimpleScaler = artifacts["halo_scaler"]
    gal_scaler: SimpleScaler = artifacts["gal_scaler"]
    dist_x2_given_x1 = artifacts["dist_x2_given_x1"]

    halos_scaled = halo_scaler.transform(halos)
    halos_t = torch.tensor(halos_scaled, dtype=torch.float32, device=device)

    gal_dim = len(artifacts["targets"])
    sample_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, halos_t.shape[0], batch_size):
            end = start + batch_size
            halos_chunk = halos_t[start:end]
            chunk_samples = []
            for halo_vec in halos_chunk:
                gal_samples = (
                    dist_x2_given_x1.condition(halo_vec).sample(torch.Size([num_draws]))
                )  # (num_draws, gal_dim)
                chunk_samples.append(gal_samples.cpu())
            if chunk_samples:
                chunk_tensor = torch.stack(chunk_samples)  # (chunk, num_draws, gal_dim)
                sample_chunks.append(chunk_tensor.numpy())

    if not sample_chunks:
        return np.empty((0, num_draws, gal_dim))

    samples = np.concatenate(sample_chunks, axis=0)  # (N, num_draws, gal_dim)
    samples_reshaped = samples.reshape(-1, gal_dim)
    samples_unscaled = gal_scaler.inverse_transform(samples_reshaped).reshape(
        samples.shape[0], num_draws, gal_dim
    )
    return samples_unscaled


def generate_train_test_plots(
    artifacts: Dict[str, any],
    halos_train: np.ndarray,
    halos_test: np.ndarray,
    gals_train: np.ndarray,
    gals_test: np.ndarray,
    output_dir: Path,
    num_samples: int,
    device: torch.device,
    targets: List[str] = None,
):
    halo_scaler: SimpleScaler = artifacts["halo_scaler"]
    gal_scaler: SimpleScaler = artifacts["gal_scaler"]
    dist_x1 = artifacts["dist_x1"]
    dist_x2_given_x1 = artifacts["dist_x2_given_x1"]

    with torch.no_grad():
        halos_flow = dist_x1.sample(torch.Size([num_samples]))
        gals_flow_list = []
        for halo in halos_flow:
            # Pyro's spline transform expects the sample input to match the mask dims.
            gal_sample = dist_x2_given_x1.condition(halo).sample()
            gals_flow_list.append(gal_sample)
        gals_flow = torch.stack(gals_flow_list)

    halos_flow_unscaled = halo_scaler.inverse_transform(halos_flow.cpu().numpy())
    gals_flow_unscaled = gal_scaler.inverse_transform(gals_flow.cpu().numpy())

    scatter_with_marginals(
        halos_train[:, 0],
        gals_train[:, 0],
        halos_flow_unscaled[:, 0],
        gals_flow_unscaled[:, 0],
        output_dir / "train set.png",
        "NF vs Training Distribution",
    )

    halos_test_scaled = halo_scaler.transform(halos_test)
    halos_test_t = torch.tensor(halos_test_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        gals_test_flow_list = []
        for halo in halos_test_t:
            gal_sample = dist_x2_given_x1.condition(halo).sample()
            gals_test_flow_list.append(gal_sample.cpu())
        gals_test_flow = torch.stack(gals_test_flow_list).numpy()
    gals_test_flow_unscaled = gal_scaler.inverse_transform(gals_test_flow)

    scatter_with_marginals(
        halos_test[:, 0],
        gals_test[:, 0],
        halos_test[:, 0],
        gals_test_flow_unscaled[:, 0],
        output_dir / "test set.png",
        "NF Predictions on Test Halos",
    )


def plot_six_galaxies(
    halos_test: np.ndarray,
    gals_test: np.ndarray,
    posterior: np.ndarray,
    output_path: Path,
    property_idx: int = None,
    targets: List[str] = None,
):
    """
    Plot posterior distributions for 6 galaxies selected based on a property.
    
    Args:
        halos_test: Test halo data
        gals_test: Test galaxy true values
        posterior: Posterior samples from NF model
        output_path: Output file path
        property_idx: Index of property to use for selection.
                     If None, uses halo mass (original behavior).
        targets: List of target property names
    """
    if targets is None:
        targets = TARGET_COLUMNS
    
    percentiles = [5, 25, 40, 60, 80, 95]

    # Guard against mismatched lengths between posterior and provided test arrays.
    max_points = min(posterior.shape[0], gals_test.shape[0], halos_test.shape[0])
    if max_points == 0:
        logging.warning("No posterior samples available to plot six galaxies.")
        return
    
    if property_idx is None:
        # Original behavior: select based on halo mass
        selection_values = halos_test[:max_points, 0]
        selection_name = "Halo Mass"
    else:
        # Select based on the specified galaxy property
        selection_values = gals_test[:max_points, property_idx]
        selection_name = targets[property_idx]
    
    indices = []
    for p in percentiles:
        target = np.percentile(selection_values, p)
        idx = np.argmin(np.abs(selection_values - target))
        indices.append(idx)

    if property_idx is None:
        # Original plot: all properties for galaxies selected by halo mass
        fig, axes = plt.subplots(len(indices), len(targets), figsize=(16, 12), sharex=False)
        if axes.ndim == 1:
            axes = axes[None, :]

        for row, idx in enumerate(indices):
            samples = posterior[idx]
            true_vals = gals_test[idx]
            for col, prop in enumerate(targets):
                label = PROPERTY_LABELS.get(prop, prop)
                ax = axes[row, col]
                sns.kdeplot(samples[:, col], ax=ax, fill=True, color="#CC78BC", alpha=0.6)
                ax.axvline(true_vals[col], color="black", linestyle="--", linewidth=2)
                ax.set_ylabel(f"Galaxy {row+1}" if col == 0 else "")
                ax.set_xlabel(label)
                if row == 0:
                    ax.set_title(prop)
                ax.grid(True, alpha=0.3)
    else:
        # New plot: single property for galaxies selected by that property
        prop = targets[property_idx]
        prop_label = PROPERTY_LABELS.get(prop, prop)
        fig, axes = plt.subplots(len(indices), 1, figsize=(8, 12), sharex=True)
        if axes.ndim == 0:
            axes = np.array([axes])
        
        for row, idx in enumerate(indices):
            samples = posterior[idx]
            true_vals = gals_test[idx]
            ax = axes[row]
            sns.kdeplot(samples[:, property_idx], ax=ax, fill=True, color="#CC78BC", alpha=0.6)
            ax.axvline(true_vals[property_idx], color="black", linestyle="--", linewidth=2, label="True value")
            ax.set_ylabel(f"Galaxy {row+1}\n(Percentile: {percentiles[row]}%)", fontsize=10)
            ax.set_xlabel(prop_label, fontsize=11)
            if row == 0:
                ax.set_title(f"{prop} - Selected by {selection_name}", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)




def compute_tarp_coverage(
    samples: np.ndarray,
    truth: np.ndarray,
    num_refs: int = 100,
):
    """
    TARP coverage using the official tarp package.

    Args:
        samples: Posterior samples with shape (n_points, n_draws, n_dims).
        truth: True parameter values with shape (n_points, n_dims).
        num_refs: Unused - kept for API compatibility.

    Returns:
        alphas: Credibility levels
        coverage_mean: Mean expected coverage probabilities
        coverage_lower: Lower bound of 95% confidence interval (2.5th percentile)
        coverage_upper: Upper bound of 95% confidence interval (97.5th percentile)
    """
    # The official function expects (n_samples, n_sims, n_dims)
    # where n_samples = number of posterior draws per simulation
    # and n_sims = number of simulations/test points.
    # Our `samples` has shape (n_points, n_draws, n_dims) = (n_sims, n_samples, n_dims),
    # so we need to transpose to (n_samples, n_sims, n_dims).
    samples_transposed = samples.transpose(1, 0, 2)  # (n_draws, n_points, n_dims)

    ecp, alphas = tarp.get_tarp_coverage(
        samples=samples_transposed,  # Shape: (n_samples, n_sims, n_dims)
        theta=truth,                 # Shape: (n_sims, n_dims)
        references="random",          # Random reference points from unit hypercube
        metric="euclidean",           # Distance metric
        num_alpha_bins=100,           # Number of credibility bins
        bootstrap=True,               # Enable bootstrap for uncertainty estimates
        num_bootstrap=100,            # Number of bootstrap iterations
        norm=False,                   # No normalization
        seed=None,                    # Don't fix seed to allow bootstrap variance
    )

    if ecp.ndim == 1:
        # If bootstrap didn't work as expected, fall back to single estimate
        logging.warning("Bootstrap returned 1D array! Bootstrap may not be working. Shape: %s", ecp.shape)
        coverage_mean = ecp
        coverage_lower = ecp
        coverage_upper = ecp
    else:
        # ecp has shape (num_bootstrap, num_alpha_bins)
        # Check if bootstrap actually produced variance
        if ecp.shape[0] == 1:
            logging.warning("Bootstrap returned only 1 iteration! Shape: %s", ecp.shape)
            coverage_mean = ecp[0]
            coverage_lower = ecp[0]
            coverage_upper = ecp[0]
        else:
            # Compute mean and percentiles for plotting uncertainty bands
            coverage_mean = ecp.mean(axis=0)
            coverage_lower = np.percentile(ecp, 2.5, axis=0)  # 2.5th percentile
            coverage_upper = np.percentile(ecp, 97.5, axis=0)  # 97.5th percentile
            
            # Check variance across bootstrap samples
            ci_width = coverage_upper - coverage_lower
            max_variance = np.max([np.std(ecp[:, i]) for i in range(min(10, ecp.shape[1]))])
            
            # Verify bootstrap is producing different values
            if max_variance < 1e-6:
                logging.warning("Bootstrap appears to produce no variance! All bootstrap samples may be identical.")

    return alphas, coverage_mean, coverage_lower, coverage_upper


def plot_tarp_curves(
    posterior: np.ndarray,
    truth: np.ndarray,
    output_dir: Path,
    suffix: str = "test",
    targets: List[str] = None,
):
    """Plot TARP curves for joint and per-property calibration.
    
    Args:
        posterior: Posterior samples
        truth: True values
        output_dir: Output directory
        suffix: Suffix for output files
        targets: List of target property names
    """
    if targets is None:
        targets = TARGET_COLUMNS
        
    ensure_dir(output_dir)
    alphas, coverage_mean, coverage_lower, coverage_upper = compute_tarp_coverage(posterior, truth)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Ideal calibration")
    ax.plot(alphas, coverage_mean, color="#029E73", linewidth=2.5, label="NF posterior (mean)")
    # Plot uncertainty bands (95% confidence interval) - make more visible
    ax.fill_between(
        alphas,
        coverage_lower,
        coverage_upper,
        alpha=0.4,
        color="#029E73",
        label="95% CI (bootstrap)",
        edgecolor="#029E73",
        linewidth=0.5,
    )
    # Also plot the bounds as lines for better visibility
    ax.plot(alphas, coverage_lower, color="#029E73", linewidth=1, linestyle=":", alpha=0.6)
    ax.plot(alphas, coverage_upper, color="#029E73", linewidth=1, linestyle=":", alpha=0.6)
    ax.fill_between(alphas, alphas, coverage_mean, alpha=0.1, color="#029E73")
    ax.set_xlabel("Credibility level (α)")
    ax.set_ylabel("Coverage")
    # Use the suffix to clarify which dataset this TARP curve corresponds to
    pretty_suffix = suffix.capitalize()
    ax.set_title(f"TARP {pretty_suffix} (Joint) - Bootstrap Enabled", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.savefig(output_dir / f"tarp {suffix} set.png", dpi=200)
    plt.close(fig)

    # Create subplots with appropriate grid size
    n_props = len(targets)
    n_rows = (n_props + 1) // 2
    n_cols = min(2, n_props)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_props == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_props > 1 else axes
    
    for idx, prop in enumerate(targets):
        prop_samples = posterior[:, :, idx][:, :, None]
        prop_truth = truth[:, idx][:, None]
        alphas, coverage_mean_prop, coverage_lower_prop, coverage_upper_prop = compute_tarp_coverage(prop_samples, prop_truth)
        ax = axes[idx] if n_props > 1 else axes[0]
        ax.plot([0, 1], [0, 1], "k--", linewidth=2)
        ax.plot(alphas, coverage_mean_prop, color="#CC78BC", linewidth=2.5)
        # Plot uncertainty bands (95% confidence interval) - make more visible
        ax.fill_between(
            alphas,
            coverage_lower_prop,
            coverage_upper_prop,
            alpha=0.4,
            color="#CC78BC",
            edgecolor="#CC78BC",
            linewidth=0.5,
        )
        # Also plot the bounds as lines for better visibility
        ax.plot(alphas, coverage_lower_prop, color="#CC78BC", linewidth=1, linestyle=":", alpha=0.6)
        ax.plot(alphas, coverage_upper_prop, color="#CC78BC", linewidth=1, linestyle=":", alpha=0.6)
        ax.set_title(f"{prop} (Bootstrap)", fontweight="bold")
        ax.set_xlabel("α")
        ax.set_ylabel("Coverage")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"tarp {suffix} set individual.png", dpi=200)
    plt.close(fig)


def _coverage_mad(alphas: np.ndarray, coverage: np.ndarray) -> float:
    """Mean absolute deviation from the ideal calibration diagonal."""
    return float(np.mean(np.abs(coverage - alphas)))


def _nearest_coverage(alphas: np.ndarray, coverage: np.ndarray, target_alpha: float) -> Tuple[float, float]:
    """Return (alpha_used, coverage_value) for the alpha bin closest to target_alpha."""
    idx = int(np.argmin(np.abs(alphas - target_alpha)))
    return float(alphas[idx]), float(coverage[idx])


def summarize_tarp_metrics(
    posterior: np.ndarray,
    truth: np.ndarray,
    targets: List[str] = None,
    alpha_targets: Tuple[float, ...] = (0.5, 0.68, 0.95),
) -> Dict[str, any]:
    """
    Compute text-friendly TARP calibration metrics.

    Args:
        posterior: Posterior samples
        truth: True values
        targets: List of target property names
        alpha_targets: Confidence levels to evaluate

    Returns:
        Dictionary with joint MAD, coverages at key alphas, and per-property MAD.
    """
    if targets is None:
        targets = TARGET_COLUMNS
    
    alphas, coverage_mean, _, _ = compute_tarp_coverage(posterior, truth)
    joint_mad = _coverage_mad(alphas, coverage_mean)

    coverage_targets = []
    for target in alpha_targets:
        alpha_used, cov_val = _nearest_coverage(alphas, coverage_mean, target)
        coverage_targets.append({"target": target, "alpha_used": alpha_used, "coverage": cov_val})

    per_property = []
    for idx, prop in enumerate(targets):
        prop_samples = posterior[:, :, idx][:, :, None]
        prop_truth = truth[:, idx][:, None]
        alphas_prop, cov_prop, _, _ = compute_tarp_coverage(prop_samples, prop_truth)
        per_property.append({"property": prop, "mad": _coverage_mad(alphas_prop, cov_prop)})

    return {"joint_mad": joint_mad, "coverage_targets": coverage_targets, "per_property": per_property}


def format_tarp_report(title: str, metrics: Dict[str, any]) -> str:
    """Render a human-readable TARP metrics report."""
    lines = [f"=== TARP Results: {title} ==="]
    lines.append(f"Joint TARP MAD: {metrics['joint_mad']:.3f}")
    for item in metrics["coverage_targets"]:
        lines.append(f"Coverage @ α={item['target']:.2f}: {item['coverage']:.3f}")
    lines.append("")
    lines.append("Individual Property MAD:")
    for item in metrics["per_property"]:
        lines.append(f"  {item['property']}: {item['mad']:.3f}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="NF-specific visualization suite.")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_PROCESSED_PARQUET,
        help="Processed dataset path.",
    )
    parser.add_argument(
        "--nf-run",
        default="baseline",
        help="NF run name under models/nf/.",
    )
    parser.add_argument(
        "--nn-runs",
        nargs=2,
        default=["raw", "smogn"],
        help="NN runs (raw, smogn) used for metrics plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Plots"),
        help="Directory to store generated figures.",
    )
    parser.add_argument("--num-flow-samples", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--posterior-mode",
        choices=["artifacts", "resample"],
        default="artifacts",
        help="Source of posterior samples: load saved training predictions ('artifacts') or resample on the provided dataset ('resample').",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--tarp-output",
        choices=["plots", "text", "both"],
        default="text",
        help="Generate TARP plots, text metrics, or both (default: text).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    device = torch.device(args.device)

    make_plots = args.tarp_output in ("plots", "both")

    artifacts = load_nf_artifacts(args.nf_run, device)
    features = artifacts["features"]
    targets = artifacts["targets"]
    
    halos_train, halos_test, gals_train, gals_test = load_processed_data(args.data, features, targets)

    if make_plots:
        generate_train_test_plots(
            artifacts,
            halos_train,
            halos_test,
            gals_train,
            gals_test,
            args.output_dir,
            args.num_flow_samples,
            device,
            targets=targets,
        )

    preds = artifacts["predictions"]
    posterior_source = args.posterior_mode

    if posterior_source == "artifacts":
        posterior_test = preds["posterior"]
        y_true_test = preds["y_true"]
    else:
        posterior_test = sample_nf_posterior_for_halos(
            halos_test,
            artifacts,
            num_draws=args.num_flow_samples,
            device=device,
        )
        y_true_test = gals_test

    if make_plots:
        # Original plot: all properties for galaxies selected by halo mass
        plot_six_galaxies(halos_test, gals_test, posterior_test, args.output_dir / "six galaxies.png", 
                         targets=targets)
        
        # New plots: one property per plot, galaxies selected by that property
        for prop_idx, prop_name in enumerate(targets):
            output_path = args.output_dir / f"six galaxies - {prop_name}.png"
            plot_six_galaxies(halos_test, gals_test, posterior_test, output_path, 
                            property_idx=prop_idx, targets=targets)

        for prop_idx, prop_name in enumerate(targets):
            prop_label = PROPERTY_LABELS.get(prop_name, prop_name)
            plot_conditioned_halo_property(
                artifacts,
                args.output_dir / f"conditioned halo masses - {prop_name}.png",
                property_idx=prop_idx,
                property_label=prop_label,
                num_samples=min(1000, args.num_flow_samples),
                device=device,
                targets=targets,
            )

    # Complete set TARP: combine train + test halos and generate posterior samples on the fly
    full_halos = np.vstack([halos_train, halos_test])
    full_truth = np.vstack([gals_train, gals_test])
    if posterior_source == "artifacts":
        num_draws = posterior_test.shape[1] if posterior_test.ndim >= 2 else 128
        full_posterior = sample_nf_posterior_for_halos(
            full_halos,
            artifacts,
            num_draws=num_draws,
            device=device,
        )
    else:
        full_posterior = sample_nf_posterior_for_halos(
            full_halos,
            artifacts,
            num_draws=posterior_test.shape[1] if posterior_test.ndim >= 2 else args.num_flow_samples,
            device=device,
        )
    if make_plots:
        plot_tarp_curves(posterior_test, y_true_test, args.output_dir, suffix="test", targets=targets)
        plot_tarp_curves(full_posterior, full_truth, args.output_dir, suffix="complete", targets=targets)
    if args.tarp_output in ("text", "both"):
        dataset_label = Path(args.data).stem
        test_title = f"{args.nf_run} on {dataset_label} (test set)"
        full_title = f"{args.nf_run} on {dataset_label} (complete set)"
        test_metrics = summarize_tarp_metrics(posterior_test, y_true_test, targets=targets)
        full_metrics = summarize_tarp_metrics(full_posterior, full_truth, targets=targets)
        print(format_tarp_report(test_title, test_metrics))
        print()
        print(format_tarp_report(full_title, full_metrics))


if __name__ == "__main__":
    main()

