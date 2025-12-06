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
from scipy.stats import pearsonr, ks_2samp, wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tarp

from src import config
from src.plots.visualize import load_nn_run
from src.utils.io import ensure_dir


sns.set_style("whitegrid")

PROPERTIES = ["SM", "SFR", "Colour", "SR"]
PROPERTY_LABELS = [
    r"$\log_{10}(M_\star/M_\odot)$",
    r"$\log_{10}(\mathrm{SFR})$",
    "Colour (g-r)",
    r"$\log_{10}(\mathrm{SR})$",
]


class SimpleScaler:
    def __init__(self, mean: List[float], scale: List[float]):
        self.mean = np.array(mean)
        self.scale = np.array(scale)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.scale + self.mean


def build_flow(device: torch.device):
    halo_dim = 3
    gal_dim = 4

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

    dist_x1, dist_x2_given_x1, modules = build_flow(device)
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
    }


def load_processed_data(data_path: Path):
    df = pd.read_parquet(data_path)
    halos = df[["M_h", "R_h", "V_h"]].values
    gals = df[PROPERTIES].values
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
        Array of shape (N, num_draws, len(PROPERTIES)) with unscaled galaxy samples.
    """

    halo_scaler: SimpleScaler = artifacts["halo_scaler"]
    gal_scaler: SimpleScaler = artifacts["gal_scaler"]
    dist_x2_given_x1 = artifacts["dist_x2_given_x1"]

    halos_scaled = halo_scaler.transform(halos)
    halos_t = torch.tensor(halos_scaled, dtype=torch.float32, device=device)

    gal_dim = len(PROPERTIES)
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
):
    """
    Plot posterior distributions for 6 galaxies selected based on a property.
    
    Args:
        halos_test: Test halo data
        gals_test: Test galaxy true values
        posterior: Posterior samples from NF model
        output_path: Output file path
        property_idx: Index of property to use for selection (0=SM, 1=SFR, 2=Colour, 3=SR).
                     If None, uses halo mass (original behavior).
    """
    percentiles = [5, 25, 40, 60, 80, 95]
    
    if property_idx is None:
        # Original behavior: select based on halo mass
        selection_values = halos_test[:, 0]
        selection_name = "Halo Mass"
    else:
        # Select based on the specified galaxy property
        selection_values = gals_test[:, property_idx]
        selection_name = PROPERTIES[property_idx]
    
    indices = []
    for p in percentiles:
        target = np.percentile(selection_values, p)
        idx = np.argmin(np.abs(selection_values - target))
        indices.append(idx)

    if property_idx is None:
        # Original plot: all properties for galaxies selected by halo mass
        fig, axes = plt.subplots(len(indices), len(PROPERTIES), figsize=(16, 12), sharex=False)
        if axes.ndim == 1:
            axes = axes[None, :]

        for row, idx in enumerate(indices):
            samples = posterior[idx]
            true_vals = gals_test[idx]
            for col, (prop, label) in enumerate(zip(PROPERTIES, PROPERTY_LABELS)):
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
        prop = PROPERTIES[property_idx]
        prop_label = PROPERTY_LABELS[property_idx]
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


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    ks_stat, _ = ks_2samp(y_true, y_pred)
    wass = wasserstein_distance(y_true, y_pred)
    return {"RMSE": rmse, "PCC": pcc, "KS_stat": ks_stat, "Wasserstein": wass}


def plot_all_metrics(
    nn_runs: List[Dict],
    nf_preds: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
):
    properties = PROPERTIES
    metrics_to_plot = ["RMSE", "PCC", "KS_stat", "Wasserstein"]
    metric_labels = [
        "RMSE (↓ better)",
        "Pearson Correlation (↑ better)",
        "K-S Statistic (↓ better)",
        "Wasserstein Distance (↓ better)",
    ]
    colors = {
        "NN (Raw)": "#0173B2",
        "NN (SMOGN)": "#DE8F05",
        "NF (Mean)": "#029E73",
        "NF (Random)": "#CC78BC",
    }

    metrics_comparison: Dict[str, Dict[str, Dict[str, float]]] = {prop: {} for prop in properties}

    for run in nn_runs:
        label = run["label"]
        pretty_label = "NN (SMOGN)" if "SMOGN" in label else "NN (Raw)"
        for prop in properties:
            metrics_comparison[prop][pretty_label] = compute_all_metrics(
                run["targets"][prop]["y_true"],
                run["targets"][prop]["y_pred"],
            )

    nf_mean = nf_preds["mean"]
    nf_random = nf_preds["random"]
    for idx, prop in enumerate(properties):
        metrics_comparison[prop]["NF (Mean)"] = compute_all_metrics(
            y_true[:, idx], nf_mean[:, idx]
        )
        metrics_comparison[prop]["NF (Random)"] = compute_all_metrics(
            y_true[:, idx], nf_random[:, idx]
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    x = np.arange(len(properties))
    width = 0.2

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        for offset, key in zip([-1.5, -0.5, 0.5, 1.5], colors.keys()):
            vals = [metrics_comparison[prop][key][metric] for prop in properties]
            ax.bar(
                x + offset * width,
                vals,
                width,
                label=key,
                color=colors[key],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xlabel("Galaxy Property", fontsize=12, fontweight="bold")
        ax.set_ylabel(label, fontsize=12, fontweight="bold")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(properties, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        if idx == 0:
            ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pcc_rmse_lines(
    nf_preds: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    prop_labels = ["SM", "SFR", "Colour", "SR"]

    metrics = {"NF (Mean)": {}, "NF (Random)": {}}
    for key in metrics.keys():
        preds = nf_preds["mean"] if key == "NF (Mean)" else nf_preds["random"]
        metrics[key]["PCC"] = [
            pearsonr(y_true[:, i], preds[:, i])[0] for i in range(len(PROPERTIES))
        ]
        metrics[key]["RMSE"] = [
            np.sqrt(mean_squared_error(y_true[:, i], preds[:, i])) for i in range(len(PROPERTIES))
        ]

    for ax, metric_name in zip(axes, ["PCC", "RMSE"]):
        for label, color, marker in zip(
            metrics.keys(), ["#029E73", "#CC78BC"], ["o-", "s-"]
        ):
            ax.plot(
                prop_labels,
                metrics[label][metric_name],
                marker,
                label=label,
                linewidth=2,
                color=color,
                markersize=7,
            )
        ax.set_title(
            "Pearson Correlation (↑ better)"
            if metric_name == "PCC"
            else "RMSE (↓ better)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Galaxy Property", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

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

    # Verify shape: ecp should have shape (num_bootstrap, num_alpha_bins) when bootstrap=True
    logging.info(f"ECP shape after bootstrap: {ecp.shape}, Expected: (100, 100) for bootstrap=True")
    
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
            logging.info(f"Bootstrap variance check - Max CI width: {ci_width.max():.6f}, Mean CI width: {ci_width.mean():.6f}, Max std across first 10 bins: {max_variance:.6f}")
            
            # Verify bootstrap is producing different values
            if max_variance < 1e-6:
                logging.warning("Bootstrap appears to produce no variance! All bootstrap samples may be identical.")

    return alphas, coverage_mean, coverage_lower, coverage_upper


def plot_tarp_curves(
    posterior: np.ndarray,
    truth: np.ndarray,
    output_dir: Path,
    suffix: str = "test",
):
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, prop in enumerate(PROPERTIES):
        prop_samples = posterior[:, :, idx][:, :, None]
        prop_truth = truth[:, idx][:, None]
        alphas, coverage_mean_prop, coverage_lower_prop, coverage_upper_prop = compute_tarp_coverage(prop_samples, prop_truth)
        ax = axes[idx]
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    device = torch.device(args.device)

    halos_train, halos_test, gals_train, gals_test = load_processed_data(args.data)
    artifacts = load_nf_artifacts(args.nf_run, device)

    generate_train_test_plots(
        artifacts,
        halos_train,
        halos_test,
        gals_train,
        gals_test,
        args.output_dir,
        args.num_flow_samples,
        device,
    )

    preds = artifacts["predictions"]
    posterior = preds["posterior"]
    y_true = preds["y_true"]

    # Original plot: all properties for galaxies selected by halo mass
    plot_six_galaxies(halos_test, gals_test, posterior, args.output_dir / "six galaxies.png")
    
    # New plots: one property per plot, galaxies selected by that property
    for prop_idx, prop_name in enumerate(PROPERTIES):
        output_path = args.output_dir / f"six galaxies - {prop_name}.png"
        plot_six_galaxies(halos_test, gals_test, posterior, output_path, property_idx=prop_idx)

    for prop_idx, prop_label in enumerate(PROPERTY_LABELS):
        plot_conditioned_halo_property(
            artifacts,
            args.output_dir / f"conditioned halo masses - {PROPERTIES[prop_idx]}.png",
            property_idx=prop_idx,
            property_label=prop_label,
            num_samples=min(1000, args.num_flow_samples),
            device=device,
        )

    nn_runs = []
    for run_name, label in zip(args.nn_runs, ["NN Raw", "NN SMOGN"]):
        nn_runs.append(load_nn_run(Path(config.NN_MODEL_DIR), run_name, label))

    plot_all_metrics(nn_runs, preds, y_true, args.output_dir / "all metrics.png")
    plot_pcc_rmse_lines(preds, y_true, args.output_dir / "pcc rmse.png")
    plot_tarp_curves(posterior, y_true, args.output_dir, suffix="test")

    # Complete set TARP: combine train + test halos and generate posterior samples on the fly
    full_halos = np.vstack([halos_train, halos_test])
    full_truth = np.vstack([gals_train, gals_test])
    num_draws = posterior.shape[1] if posterior.ndim >= 2 else 128
    full_posterior = sample_nf_posterior_for_halos(
        full_halos,
        artifacts,
        num_draws=num_draws,
        device=device,
    )
    plot_tarp_curves(full_posterior, full_truth, args.output_dir, suffix="complete")


if __name__ == "__main__":
    main()

