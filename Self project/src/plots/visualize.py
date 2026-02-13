"""
Visualization CLI for NN vs NF comparison plots.
Supports:
  - metrics: grouped bar charts comparing RMSE/PCC/KS/Wasserstein
  - contours: 2x2 contour + marginal plots for property pairs
  - tarp: TARP calibration curves for NN and NF predictions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from src import config
from src.utils.io import ensure_dir
from src.utils.common import (
    compute_regression_metrics,
    compute_single_metric,
    get_model_color,
    MODEL_COLORS,
    TARGET_COLUMNS,
)
# Property pairs for contour plots: (prop1, prop2, xlabel, ylabel)
CONTOUR_PROPERTY_PAIRS = [
    ("SM", "Colour", r"$\log_{10}(M_{\star} [h^{-1} M_{\odot}])$", r"$colour$"),
    ("SM", "SFR", r"$\log_{10}(M_{\star} [h^{-1} M_{\odot}])$", r"$\log_{10}(\mathrm{SFR})$"),
    ("SM", "SR", r"$\log_{10}(M_{\star} [h^{-1} M_{\odot}])$", r"$\log_{10}(\mathrm{SR})$"),
    ("Colour", "SFR", r"$colour$", r"$\log_{10}(\mathrm{SFR})$"),
]

# Manual axis limits to keep key regions centered: (xlim, ylim)
PAIR_AXIS_LIMITS = {
    ("SM", "Colour"): ((6.5, 10.5), (0.05, 0.8)),
    ("SM", "SFR"): ((6.5, 10.5), (7.2, 10.4)),
    ("Colour", "SFR"): ((0.05, 0.8), (7.2, 10.4)),
}


def load_nn_run(base_dir: Path, run_name: str, label: str) -> Dict:
    """Load NN run predictions. Tries all TARGET_COLUMNS and loads what exists."""
    run_path = base_dir / run_name
    results = {"label": label, "type": "nn", "predictions": {}}
    
    # Check which targets actually have predictions
    available_targets = []
    for target in TARGET_COLUMNS:
        pred_path = run_path / target / "predictions.npz"
        if pred_path.exists():
            data = np.load(pred_path)
            results["predictions"][target] = {
                "y_true": data["y_true"].flatten(),
                "y_pred": data["y_pred"].flatten(),
            }
            available_targets.append(target)
        else:
            logging.debug(f"No predictions for {target} in run {run_name}")
    
    if not available_targets:
        logging.warning(f"No valid predictions found in NN run {run_name}")
    else:
        logging.info(f"NN run {run_name} has predictions for: {available_targets}")
    
    results["targets"] = available_targets
    return results


def load_nf_run(base_dir: Path, run_name: str, label: str) -> Dict:
    run_path = base_dir / run_name
    pred_path = run_path / "predictions.npz"
    summary_path = run_path / "training_summary.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"NF predictions not found at {pred_path}")
    data = np.load(pred_path)
    
    # Get targets from training summary
    targets = TARGET_COLUMNS  # Default
    if summary_path.exists():
        import json
        with summary_path.open("r") as f:
            summary = json.load(f)
        # Prefer explicit targets field, fall back to metrics keys for backward compatibility
        if "targets" in summary:
            targets = summary["targets"]
            logging.info(f"NF run {run_name} was trained on targets: {targets}")
        elif "metrics" in summary and isinstance(summary["metrics"], dict):
            targets = list(summary["metrics"].keys())
            logging.info(f"NF run {run_name} inferred targets from metrics: {targets}")
    
    return {
        "label": label,
        "type": "nf",
        "y_true": data["y_true"],
        "mean": data["mean"],
        "random": data["random"],
        "posterior": data["posterior"],
        "targets": targets,  # Actual targets used in training
    }




def compute_mode_from_posterior(posterior_samples: np.ndarray) -> np.ndarray:
    """
    Compute the mode (MAP) from posterior samples using kernel density estimation.
    
    Args:
        posterior_samples: Array of shape (n_samples, n_features) containing posterior samples
        
    Returns:
        Array of shape (n_features,) containing the mode prediction
    """
    if posterior_samples.shape[0] == 0:
        raise ValueError("Posterior samples cannot be empty")
    
    # If only one sample, return it
    if posterior_samples.shape[0] == 1:
        return posterior_samples[0]
    
    try:
        # Compute KDE on the samples
        kde = gaussian_kde(posterior_samples.T)
        # Evaluate density at each sample point
        densities = kde(posterior_samples.T)
        # Find the sample with highest density (mode)
        mode_idx = np.argmax(densities)
        return posterior_samples[mode_idx]
    except Exception as e:
        # Fallback to mean if KDE fails (e.g., degenerate samples)
        logging.warning(f"KDE failed for mode computation, using mean instead: {e}")
        return posterior_samples.mean(axis=0)


def metrics_plot(args, nn_runs: List[Dict], nf_run: Dict) -> None:
    properties = TARGET_COLUMNS
    metrics_to_plot = ["RMSE", "PCC", "KS_stat", "Wasserstein"]
    metric_labels = [
        "RMSE (\u2193 better)",
        "Pearson Correlation (\u2191 better)",
        "K-S Statistic (\u2193 better)",
        "Wasserstein Distance (\u2193 better)",
    ]

    labels: List[str] = []
    model_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for run in nn_runs:
        labels.append(run["label"])
        model_metrics[run["label"]] = {}
        for prop in properties:
            target_data = run["predictions"].get(prop)
            if target_data is None:
                continue
            model_metrics[run["label"]][prop] = compute_regression_metrics(
                target_data["y_true"], target_data["y_pred"]
            )

    # Get targets from NF run
    nf_targets = nf_run.get("targets", TARGET_COLUMNS)
    # Create mapping from property name to index in saved predictions
    nf_prop_to_idx = {prop: idx for idx, prop in enumerate(nf_targets)}
    
    # Determine which NF mode(s) to plot based on command line argument
    nf_modes_to_plot = args.nf_mode if isinstance(args.nf_mode, list) else [args.nf_mode]
    
    # If mode is requested, compute it from posterior samples
    if "mode" in nf_modes_to_plot:
        if "posterior" not in nf_run:
            raise ValueError("Posterior samples not found in NF run. Cannot compute mode.")
        logging.info("Computing mode predictions from posterior samples...")
        n_test = nf_run["posterior"].shape[0]
        mode_predictions = np.zeros_like(nf_run["mean"])
        for i in range(n_test):
            mode_predictions[i] = compute_mode_from_posterior(nf_run["posterior"][i])
        nf_run["mode"] = mode_predictions
    
    for mode in nf_modes_to_plot:
        name = f"NF ({mode})"
        labels.append(name)
        model_metrics[name] = {}
        for prop in properties:
            # Use the correct index from when the model was saved
            idx = nf_prop_to_idx.get(prop)
            if idx is None:
                logging.warning("Property %s not found in NF property order, skipping", prop)
                continue
            model_metrics[name][prop] = compute_regression_metrics(
                nf_run["y_true"][:, idx], nf_run[mode][:, idx]
            )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    x = np.arange(len(properties))
    width = 0.2

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        series = []
        for label_name in labels:
            vals = [model_metrics[label_name][prop][metric] for prop in properties]
            series.append((label_name, vals))

        if len(series) == 4:
            offsets = [-1.5, -0.5, 0.5, 1.5]
        elif len(series) == 3:
            offsets = [-1.0, 0.0, 1.0]
        elif len(series) == 2:
            offsets = [-0.5, 0.5]
        else:
            offsets = np.linspace(-0.5 * (len(series) - 1), 0.5 * (len(series) - 1), len(series))

        for offset, (series_label, vals) in zip(offsets, series):
            color = get_model_color(series_label)
            
            ax.bar(
                x + offset * width,
                vals,
                width,
                label=series_label,
                color=color,
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
        ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    ensure_dir(args.output.parent)
    plt.savefig(args.output, dpi=200)
    logging.info("Metrics plot saved to %s", args.output)
    
    # Also generate PCC-only line plot
    pcc_output = args.output.parent / f"{args.output.stem}_pcc.png"
    plot_pcc_line(model_metrics, labels, properties, pcc_output)


def plot_pcc_line(
    model_metrics: Dict[str, Dict[str, Dict[str, float]]],
    labels: List[str],
    properties: List[str],
    output_path: Path,
) -> None:
    """Plot PCC as a line plot for all models."""
    markers = ["o-", "s-", "^-", "D-"]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    for idx, label in enumerate(labels):
        color = get_model_color(label)
        marker = markers[idx % len(markers)]
        pcc_vals = [model_metrics[label][prop]["PCC"] for prop in properties]
        
        ax.plot(
            properties,
            pcc_vals,
            marker,
            label=label,
            linewidth=2,
            color=color,
            markersize=7,
        )
    
    ax.set_title("Pearson Correlation (\u2191 better)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Galaxy Property", fontsize=12)
    ax.set_ylabel("PCC", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info("PCC line plot saved to %s", output_path)


def plot_pcc_rmse_lines(args, nn_runs: List[Dict], nf_run: Dict) -> None:
    """Plot PCC as a line plot for all models."""
    properties = TARGET_COLUMNS
    
    # Collect all model labels and their predictions
    labels: List[str] = []
    model_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    
    # Add NN runs
    for run in nn_runs:
        label = run["label"]
        labels.append(label)
        model_predictions[label] = {}
        for prop in properties:
            target_data = run["predictions"].get(prop)
            if target_data is None:
                continue
            model_predictions[label][prop] = {
                "y_true": target_data["y_true"],
                "y_pred": target_data["y_pred"],
            }
    
    # Get targets from NF run
    nf_targets = nf_run.get("targets", TARGET_COLUMNS)
    nf_prop_to_idx = {prop: idx for idx, prop in enumerate(nf_targets)}
    
    # Determine which NF mode(s) to plot based on command line argument
    nf_modes_to_plot = getattr(args, "nf_mode", ["mean", "random"])
    if isinstance(nf_modes_to_plot, str):
        if "," in nf_modes_to_plot:
            nf_modes_to_plot = [m.strip() for m in nf_modes_to_plot.split(",")]
        else:
            nf_modes_to_plot = [nf_modes_to_plot]
    elif not isinstance(nf_modes_to_plot, list):
        nf_modes_to_plot = ["mean", "random"]  # Default fallback
    
    # If mode is requested, compute it from posterior samples
    if "mode" in nf_modes_to_plot:
        if "posterior" not in nf_run:
            raise ValueError("Posterior samples not found in NF run. Cannot compute mode.")
        logging.info("Computing mode predictions from posterior samples...")
        n_test = nf_run["posterior"].shape[0]
        mode_predictions = np.zeros_like(nf_run["mean"])
        for i in range(n_test):
            mode_predictions[i] = compute_mode_from_posterior(nf_run["posterior"][i])
        nf_run["mode"] = mode_predictions
    
    # Add NF runs
    for mode in nf_modes_to_plot:
        name = f"NF ({mode})"
        labels.append(name)
        model_predictions[name] = {}
        for prop in properties:
            idx = nf_prop_to_idx.get(prop)
            if idx is None:
                logging.warning("Property %s not found in NF property order, skipping", prop)
                continue
            model_predictions[name][prop] = {
                "y_true": nf_run["y_true"][:, idx],
                "y_pred": nf_run[mode][:, idx],
            }
    
    # Compute PCC for each model
    metrics = {}
    for label in labels:
        metrics[label] = {"PCC": []}
        for prop in properties:
            data = model_predictions[label][prop]
            pcc, _ = pearsonr(data["y_true"], data["y_pred"])
            metrics[label]["PCC"].append(pcc)
    
    markers = ["o-", "s-", "^-", "D-"]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    for idx, label in enumerate(labels):
        color = get_model_color(label)
        marker = markers[idx % len(markers)]
        
        ax.plot(
            properties,
            metrics[label]["PCC"],
            marker,
            label=label,
            linewidth=2,
            color=color,
            markersize=7,
        )
    
    ax.set_title("Pearson Correlation (\u2191 better)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Galaxy Property", fontsize=12)
    ax.set_ylabel("PCC", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    ensure_dir(args.output.parent)
    plt.savefig(args.output, dpi=200)
    logging.info("PCC line plot saved to %s", args.output)


def get_model_predictions_for_contours(
    model_key: str,
    nn_runs: List[Dict],
    nf_run: Dict,
) -> Tuple[str, Dict[str, np.ndarray]]:
    if model_key.startswith("nn:"):
        idx = int(model_key.split(":")[1])
        if idx >= len(nn_runs):
            raise IndexError(f"Requested NN run index {idx} but only {len(nn_runs)} loaded.")
        run = nn_runs[idx]
        return run["label"], {prop: run["predictions"][prop]["y_pred"] for prop in run["targets"]}
    
    # Get targets from NF run
    nf_targets = nf_run.get("targets", TARGET_COLUMNS)
    nf_prop_to_idx = {prop: idx for idx, prop in enumerate(nf_targets)}
    
    if model_key == "nf:mean":
        return "NF (Mean)", {
            prop: nf_run["mean"][:, idx]
            for prop, idx in nf_prop_to_idx.items()
        }
    if model_key == "nf:random":
        return "NF (Random)", {
            prop: nf_run["random"][:, idx]
            for prop, idx in nf_prop_to_idx.items()
        }
    if model_key == "nf:mode":
        # Compute mode if not already computed
        if "mode" not in nf_run:
            if "posterior" not in nf_run:
                raise ValueError("Posterior samples not found in NF run. Cannot compute mode.")
            logging.info("Computing mode predictions from posterior samples...")
            n_test = nf_run["posterior"].shape[0]
            mode_predictions = np.zeros_like(nf_run["mean"])
            for i in range(n_test):
                mode_predictions[i] = compute_mode_from_posterior(nf_run["posterior"][i])
            nf_run["mode"] = mode_predictions
        return "NF (Mode)", {
            prop: nf_run["mode"][:, nf_prop_to_idx.get(prop, i)]
            for i, prop in enumerate(TARGET_COLUMNS)
            if prop in nf_prop_to_idx
        }
    raise ValueError(f"Unsupported model key {model_key}")


def plot_contour_panel_overlay(
    ax,
    x_true: np.ndarray,
    y_true_vals: np.ndarray,
    model_predictions: List[Tuple[str, np.ndarray, np.ndarray, str]],
    xlabel: str,
    ylabel: str,
    subsample: float,
    axis_limits: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
):
    """Plot truth and all models overlaid on a single subplot."""
    n_samples = len(x_true)
    subsample_size = max(1, int(n_samples * subsample))
    idx = np.random.choice(n_samples, subsample_size, replace=False)
    x_true_sub, y_true_sub = x_true[idx], y_true_vals[idx]

    if axis_limits is not None:
        xlim, ylim = axis_limits
    else:
        # Calculate limits from all data (truth + all models) to ensure nothing gets cut off
        all_x_values = [x_true]
        all_y_values = [y_true_vals]
        for _, x_pred, y_pred, _ in model_predictions:
            all_x_values.append(x_pred)
            all_y_values.append(y_pred)
        
        all_x = np.concatenate(all_x_values)
        all_y = np.concatenate(all_y_values)
        
        # Use wider percentiles and add padding
        x_range = np.percentile(all_x, [0.1, 99.9])
        y_range = np.percentile(all_y, [0.1, 99.9])
        
        # Add 5% padding on each side
        x_padding = (x_range[1] - x_range[0]) * 0.05
        y_padding = (y_range[1] - y_range[0]) * 0.05
        
        xlim = [x_range[0] - x_padding, x_range[1] + x_padding]
        ylim = [y_range[0] - y_padding, y_range[1] + y_padding]
    
    xx, yy = np.mgrid[xlim[0]:xlim[1]:150j, ylim[0]:ylim[1]:150j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Plot truth as gray filled contours
    try:
        kernel_true = gaussian_kde(np.vstack([x_true_sub, y_true_sub]), bw_method="scott")
        density_true = np.reshape(kernel_true(positions).T, xx.shape)
        density_true = gaussian_filter(density_true, sigma=1.0)
    except Exception:
        density_true, _, _ = np.histogram2d(x_true_sub, y_true_sub, bins=50)
        density_true = gaussian_filter(density_true.T, sigma=1.5)

    levels_true = np.linspace(density_true.min(), density_true.max(), 8)
    ax.contourf(xx, yy, density_true, levels=levels_true, cmap="Greys", alpha=0.6, zorder=1)

    # Plot each model as colored contour lines
    for label, x_pred, y_pred, color in model_predictions:
        x_pred_sub, y_pred_sub = x_pred[idx], y_pred[idx]
        try:
            kernel_pred = gaussian_kde(np.vstack([x_pred_sub, y_pred_sub]), bw_method="scott")
            density_pred = np.reshape(kernel_pred(positions).T, xx.shape)
            density_pred = gaussian_filter(density_pred, sigma=1.0)
        except Exception:
            density_pred, _, _ = np.histogram2d(x_pred_sub, y_pred_sub, bins=50)
            density_pred = gaussian_filter(density_pred.T, sigma=1.5)

        levels_pred = np.linspace(density_pred.min(), density_pred.max(), 8)
        ax.contour(xx, yy, density_pred, levels=levels_pred, colors=color, linewidths=2.0, zorder=2)

    # Add marginal histograms
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0.1)
    ax_right = divider.append_axes("right", size="20%", pad=0.1)

    # Top marginal: x-axis
    bins_x = np.linspace(xlim[0], xlim[1], 60)
    hist_true_x, _ = np.histogram(x_true_sub, bins=bins_x, density=True)
    hist_true_x = gaussian_filter(hist_true_x, sigma=1.0)
    centers_x = (bins_x[:-1] + bins_x[1:]) / 2
    ax_top.fill_between(centers_x, hist_true_x, color="gray", alpha=0.5, label="Truth")
    
    for label, x_pred, y_pred, color in model_predictions:
        x_pred_sub = x_pred[idx]
        hist_pred_x, _ = np.histogram(x_pred_sub, bins=bins_x, density=True)
        hist_pred_x = gaussian_filter(hist_pred_x, sigma=1.0)
        ax_top.plot(centers_x, hist_pred_x, color=color, linewidth=2.0, label=label)
    
    ax_top.set_xlim(xlim)
    ax_top.axis("off")

    # Right marginal: y-axis
    bins_y = np.linspace(ylim[0], ylim[1], 60)
    hist_true_y, _ = np.histogram(y_true_sub, bins=bins_y, density=True)
    hist_true_y = gaussian_filter(hist_true_y, sigma=1.0)
    centers_y = (bins_y[:-1] + bins_y[1:]) / 2
    ax_right.fill_betweenx(centers_y, hist_true_y, color="gray", alpha=0.5)
    
    for label, x_pred, y_pred, color in model_predictions:
        y_pred_sub = y_pred[idx]
        hist_pred_y, _ = np.histogram(y_pred_sub, bins=bins_y, density=True)
        hist_pred_y = gaussian_filter(hist_pred_y, sigma=1.0)
        ax_right.plot(hist_pred_y, centers_y, color=color, linewidth=2.0)
    
    ax_right.set_ylim(ylim)
    ax_right.axis("off")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def contour_plots(args, nn_runs: List[Dict], nf_run: Dict) -> None:
    """Generate individual contour plots per property pair with all models overlaid."""
    property_pairs = CONTOUR_PROPERTY_PAIRS
    # Get property order from NF run (may differ from current TARGET_COLUMNS for backward compatibility)
    nf_property_order = nf_run.get("property_order", TARGET_COLUMNS)
    nf_prop_to_idx = {prop: idx for idx, prop in enumerate(nf_property_order)}
    y_true = {
        prop: nf_run["y_true"][:, nf_prop_to_idx.get(prop, idx)]
        for idx, prop in enumerate(TARGET_COLUMNS)
        if prop in nf_prop_to_idx
    }


    # Determine which NF mode(s) to plot based on command line argument
    nf_modes_to_plot = getattr(args, "nf_mode", ["mean", "random"])
    if isinstance(nf_modes_to_plot, str):
        if "," in nf_modes_to_plot:
            nf_modes_to_plot = [m.strip() for m in nf_modes_to_plot.split(",")]
        else:
            nf_modes_to_plot = [nf_modes_to_plot]
    elif not isinstance(nf_modes_to_plot, list):
        nf_modes_to_plot = ["mean", "random"]  # Default fallback
    
    # If mode is requested, compute it from posterior samples
    if "mode" in nf_modes_to_plot:
        if "posterior" not in nf_run:
            raise ValueError("Posterior samples not found in NF run. Cannot compute mode.")
        logging.info("Computing mode predictions from posterior samples...")
        n_test = nf_run["posterior"].shape[0]
        mode_predictions = np.zeros_like(nf_run["mean"])
        for i in range(n_test):
            mode_predictions[i] = compute_mode_from_posterior(nf_run["posterior"][i])
        nf_run["mode"] = mode_predictions

    ensure_dir(args.output_prefix)

    for prop1, prop2, xlabel, ylabel in property_pairs:
        fig, ax = plt.subplots(figsize=(7, 7))

        model_predictions: List[Tuple[str, np.ndarray, np.ndarray, str]] = []
        added_models = set()

        for nn_run in nn_runs:
            if prop1 in nn_run["predictions"] and prop2 in nn_run["predictions"]:
                label = nn_run["label"]
                label_lower = label.lower()
                # Determine standardized label based on keywords
                if "optuna" in label_lower:
                    std_label = "NN (Optuna)"
                elif "smogn" in label_lower:
                    std_label = "NN (SMOGN)"
                else:
                    std_label = "NN (Raw)"
                
                if std_label not in added_models:
                    model_predictions.append(
                        (
                            std_label,
                            nn_run["predictions"][prop1]["y_pred"],
                            nn_run["predictions"][prop2]["y_pred"],
                            get_model_color(std_label),
                        )
                    )
                    added_models.add(std_label)

        # Use correct indices from NF property order
        prop1_idx = nf_prop_to_idx.get(prop1)
        prop2_idx = nf_prop_to_idx.get(prop2)
        if prop1_idx is None or prop2_idx is None:
            logging.warning("Property pair (%s, %s) not found in NF property order, skipping NF models", prop1, prop2)
        else:
            # Add NF models based on requested modes
            for mode in nf_modes_to_plot:
                mode_label = f"NF ({mode.capitalize()})"
                if mode == "mean":
                    model_predictions.append(
                        (
                            "NF (Mean)",
                            nf_run["mean"][:, prop1_idx],
                            nf_run["mean"][:, prop2_idx],
                            get_model_color("NF (Mean)"),
                        )
                    )
                elif mode == "random":
                    model_predictions.append(
                        (
                            "NF (Random)",
                            nf_run["random"][:, prop1_idx],
                            nf_run["random"][:, prop2_idx],
                            get_model_color("NF (Random)"),
                        )
                    )
                elif mode == "mode":
                    model_predictions.append(
                        (
                            "NF (Mode)",
                            nf_run["mode"][:, prop1_idx],
                            nf_run["mode"][:, prop2_idx],
                            get_model_color("NF (Mode)"),
                        )
                    )

        axis_limits = PAIR_AXIS_LIMITS.get((prop1, prop2))

        plot_contour_panel_overlay(
            ax=ax,
            x_true=y_true[prop1],
            y_true_vals=y_true[prop2],
            model_predictions=model_predictions,
            xlabel=xlabel,
            ylabel=ylabel,
            subsample=args.subsample,
            axis_limits=axis_limits,
        )

        legend_elements = [
            Rectangle((0, 0), 1, 1, fc="gray", alpha=0.5, label="Truth"),
        ]
        # Only add legend entries for models that are actually plotted
        plotted_labels = {label for label, _, _, _ in model_predictions}
        for label in plotted_labels:
            color = get_model_color(label)
            legend_elements.append(Rectangle((0, 0), 1, 1, fc="none", ec=color, linewidth=2, label=label))
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

        fig.tight_layout()
        out_path = args.output_prefix / f"{prop1}_vs_{prop2}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logging.info("Saved contour plot to %s", out_path)


def plot_contour_panel(
    ax,
    x_true: np.ndarray,
    y_true_vals: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray,
    xlabel: str,
    ylabel: str,
    color,
    title: str,
    subsample: float,
):
    n_samples = len(x_true)
    subsample_size = max(1, int(n_samples * subsample))
    idx = np.random.choice(n_samples, subsample_size, replace=False)
    x_true_sub, y_true_sub = x_true[idx], y_true_vals[idx]
    x_pred_sub, y_pred_sub = x_pred[idx], y_pred[idx]

    xlim = np.percentile(x_true, [0.5, 99.5])
    ylim = np.percentile(y_true_vals, [0.5, 99.5])
    xx, yy = np.mgrid[xlim[0]:xlim[1]:150j, ylim[0]:ylim[1]:150j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kernel_true = gaussian_kde(np.vstack([x_true_sub, y_true_sub]), bw_method="scott")
        density_true = np.reshape(kernel_true(positions).T, xx.shape)
        density_true = gaussian_filter(density_true, sigma=1.0)
    except Exception:
        density_true, _, _ = np.histogram2d(x_true_sub, y_true_sub, bins=50)
        density_true = gaussian_filter(density_true.T, sigma=1.5)

    try:
        kernel_pred = gaussian_kde(np.vstack([x_pred_sub, y_pred_sub]), bw_method="scott")
        density_pred = np.reshape(kernel_pred(positions).T, xx.shape)
        density_pred = gaussian_filter(density_pred, sigma=1.0)
    except Exception:
        density_pred, _, _ = np.histogram2d(x_pred_sub, y_pred_sub, bins=50)
        density_pred = gaussian_filter(density_pred.T, sigma=1.5)

    levels_true = np.linspace(density_true.min(), density_true.max(), 8)
    ax.contourf(xx, yy, density_true, levels=levels_true, cmap="Greys", alpha=0.6, zorder=1)
    levels_pred = np.linspace(density_pred.min(), density_pred.max(), 8)
    ax.contour(xx, yy, density_pred, levels=levels_pred, colors=color, linewidths=2.0, zorder=2)

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0.1)
    ax_right = divider.append_axes("right", size="20%", pad=0.1)

    bins_x = np.linspace(xlim[0], xlim[1], 60)
    hist_true_x, _ = np.histogram(x_true_sub, bins=bins_x, density=True)
    hist_pred_x, _ = np.histogram(x_pred_sub, bins=bins_x, density=True)
    hist_true_x = gaussian_filter(hist_true_x, sigma=1.0)
    hist_pred_x = gaussian_filter(hist_pred_x, sigma=1.0)
    centers_x = (bins_x[:-1] + bins_x[1:]) / 2
    ax_top.fill_between(centers_x, hist_true_x, color="gray", alpha=0.5)
    ax_top.plot(centers_x, hist_pred_x, color=color, linewidth=2.0)
    ax_top.set_xlim(xlim)
    ax_top.axis("off")

    bins_y = np.linspace(ylim[0], ylim[1], 60)
    hist_true_y, _ = np.histogram(y_true_sub, bins=bins_y, density=True)
    hist_pred_y, _ = np.histogram(y_pred_sub, bins=bins_y, density=True)
    hist_true_y = gaussian_filter(hist_true_y, sigma=1.0)
    hist_pred_y = gaussian_filter(hist_pred_y, sigma=1.0)
    centers_y = (bins_y[:-1] + bins_y[1:]) / 2
    ax_right.fill_betweenx(centers_y, hist_true_y, color="gray", alpha=0.5)
    ax_right.plot(hist_pred_y, centers_y, color=color, linewidth=2.0)
    ax_right.set_ylim(ylim)
    ax_right.axis("off")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=color)
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="gray", alpha=0.5, label="Truth"),
        Rectangle((0, 0), 1, 1, fc="none", ec=color, linewidth=2, label="Model"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization utilities for NN vs NF runs.")
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Shared arguments for all subcommands
    shared_args = argparse.ArgumentParser(add_help=False)
    shared_args.add_argument("--nn-dir", type=Path, default=config.NN_MODEL_DIR)
    shared_args.add_argument("--nf-dir", type=Path, default=config.NF_MODEL_DIR)
    shared_args.add_argument("--nn-runs", nargs="*", default=["default"])
    shared_args.add_argument("--nn-labels", nargs="*", default=None)
    shared_args.add_argument("--nf-run", default="default")

    metrics_parser = subparsers.add_parser("metrics", parents=[shared_args], help="Create grouped metric bar plots.")
    metrics_parser.add_argument(
        "--output",
        type=Path,
        default=config.PLOTS_DIR / "metrics_comparison.png",
        help="Output path for metrics figure.",
    )
    metrics_parser.add_argument(
        "--nf-mode",
        type=str,
        default="mean",
        help="NF prediction mode to plot: 'mean' (default), 'random', or 'mode' (MAP). Can specify multiple as comma-separated list (e.g., 'mean,mode').",
    )

    contour_parser = subparsers.add_parser(
        "contours",
        parents=[shared_args],
        help="Generate per-pair contour plots with all models overlaid.",
    )
    contour_parser.add_argument("--subsample", type=float, default=0.5)
    contour_parser.add_argument(
        "--output-prefix",
        type=Path,
        default=config.PLOTS_DIR / "contours",
        help="Directory to write individual contour figures.",
    )
    contour_parser.add_argument(
        "--nf-mode",
        type=str,
        default="mean,random",
        help="NF prediction mode(s) to plot: 'mean', 'random', or 'mode' (MAP). Can specify multiple as comma-separated list (e.g., 'mean,mode').",
    )

    pcc_rmse_parser = subparsers.add_parser(
        "pcc-rmse",
        parents=[shared_args],
        help="Create PCC line plot for all models.",
    )
    pcc_rmse_parser.add_argument(
        "--output",
        type=Path,
        default=config.PLOTS_DIR / "pcc_rmse.png",
        help="Output path for PCC line plot.",
    )
    pcc_rmse_parser.add_argument(
        "--nf-mode",
        type=str,
        default="mean,random",
        help="NF prediction mode(s) to plot: 'mean', 'random', or 'mode' (MAP). Can specify multiple as comma-separated list (e.g., 'mean,mode').",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    nn_labels = args.nn_labels or args.nn_runs
    if len(nn_labels) != len(args.nn_runs):
        raise ValueError("nn-labels count must match nn-runs count.")
    
    # Parse nf_mode: support comma-separated list or single value
    if hasattr(args, "nf_mode"):
        if isinstance(args.nf_mode, str):
            if "," in args.nf_mode:
                args.nf_mode = [m.strip() for m in args.nf_mode.split(",")]
            else:
                args.nf_mode = [args.nf_mode]
        # Validate all modes
        valid_modes = {"mean", "random", "mode"}
        for mode in args.nf_mode:
            if mode not in valid_modes:
                raise ValueError(f"Invalid NF mode: {mode}. Must be one of {valid_modes}")

    nn_runs = [
        load_nn_run(args.nn_dir, run_name, label)
        for run_name, label in zip(args.nn_runs, nn_labels)
    ]
    nf_run = load_nf_run(args.nf_dir, args.nf_run, "NF")

    if args.command == "metrics":
        metrics_plot(args, nn_runs, nf_run)
    elif args.command == "contours":
        if len(nn_runs) == 0:
            raise ValueError("At least one NN run required for contour plots.")
        contour_plots(args, nn_runs, nf_run)
    elif args.command == "pcc-rmse":
        plot_pcc_rmse_lines(args, nn_runs, nf_run)


if __name__ == "__main__":
    main()

