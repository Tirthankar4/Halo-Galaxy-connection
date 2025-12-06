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
from scipy.stats import gaussian_kde, ks_2samp, pearsonr, wasserstein_distance

from src import config
from src.utils.io import ensure_dir

TARGET_COLUMNS = ["SM", "SFR", "Colour", "SR"]
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
    run_path = base_dir / run_name
    results = {"label": label, "type": "nn", "targets": {}}
    for target in TARGET_COLUMNS:
        pred_path = run_path / target / "predictions.npz"
        if not pred_path.exists():
            logging.warning("Missing predictions for %s in run %s", target, run_name)
            continue
        data = np.load(pred_path)
        results["targets"][target] = {
            "y_true": data["y_true"].flatten(),
            "y_pred": data["y_pred"].flatten(),
        }
    return results


def load_nf_run(base_dir: Path, run_name: str, label: str) -> Dict:
    run_path = base_dir / run_name
    pred_path = run_path / "predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"NF predictions not found at {pred_path}")
    data = np.load(pred_path)
    return {
        "label": label,
        "type": "nf",
        "y_true": data["y_true"],
        "mean": data["mean"],
        "random": data["random"],
        "posterior": data["posterior"],
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pcc, _ = pearsonr(y_true, y_pred)
    ks_stat, _ = ks_2samp(y_true, y_pred)
    wass = float(wasserstein_distance(y_true, y_pred))
    return {"RMSE": rmse, "PCC": float(pcc), "KS_stat": float(ks_stat), "Wasserstein": wass}


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
            target_data = run["targets"].get(prop)
            if target_data is None:
                continue
            model_metrics[run["label"]][prop] = compute_metrics(
                target_data["y_true"], target_data["y_pred"]
            )

    for mode in ["mean", "random"]:
        name = f"NF ({mode})"
        labels.append(name)
        model_metrics[name] = {}
        for idx, prop in enumerate(properties):
            model_metrics[name][prop] = compute_metrics(
                nf_run["y_true"][:, idx], nf_run[mode][:, idx]
            )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    colors = {
        # Raw - Blue
        "NN_Raw": "#0173B2",
        "NN Raw": "#0173B2",
        "NN (Raw)": "#0173B2",
        "Raw": "#0173B2",
        # SMOGN - Yellow/Orange
        "NN_SMOGN": "#DE8F05",
        "NN SMOGN": "#DE8F05",
        "NN (SMOGN)": "#DE8F05",
        "SMOGN": "#DE8F05",
        # Optuna - Red
        "OPTUNA_Raw": "#DC143C",
        "NN_Optuna": "#DC143C",
        "NN Optuna": "#DC143C",
        "NN (Optuna)": "#DC143C",
        "optuna_best": "#DC143C",
        "Optuna": "#DC143C",
        # NF (mean) - Green
        "NF_Mean": "#029E73",
        "NF Mean": "#029E73",
        "NF (Mean)": "#029E73",
        "NF (mean)": "#029E73",
        # NF (random) - Pink
        "NF_Random": "#CC78BC",
        "NF Random": "#CC78BC",
        "NF (Random)": "#CC78BC",
        "NF (random)": "#CC78BC",
    }

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
            # Try exact match first
            color = colors.get(series_label, None)
            
            # If no exact match, try to match by keywords
            if color is None:
                label_lower = series_label.lower()
                if "optuna" in label_lower:
                    color = "#DC143C"  # Red for Optuna
                elif "smogn" in label_lower:
                    color = "#DE8F05"  # Yellow/Orange for SMOGN
                elif "raw" in label_lower:
                    color = "#0173B2"  # Blue for Raw
                elif "nf" in label_lower and "mean" in label_lower:
                    color = "#029E73"  # Green for NF (mean)
                elif "nf" in label_lower and "random" in label_lower:
                    color = "#CC78BC"  # Pink for NF (random)
                else:
                    color = "#888888"  # Default grey
            
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
        return run["label"], {prop: run["targets"][prop]["y_pred"] for prop in TARGET_COLUMNS}
    if model_key == "nf:mean":
        return "NF (Mean)", {prop: nf_run["mean"][:, i] for i, prop in enumerate(TARGET_COLUMNS)}
    if model_key == "nf:random":
        return "NF (Random)", {
            prop: nf_run["random"][:, i] for i, prop in enumerate(TARGET_COLUMNS)
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
    y_true = {prop: nf_run["y_true"][:, idx] for idx, prop in enumerate(TARGET_COLUMNS)}

    model_colors = {
        "NN (Raw)": "#0173B2",
        "NN (SMOGN)": "#DE8F05",
        "NN (Optuna)": "#DC143C",
        "NF (Mean)": "#029E73",
        "NF (Random)": "#CC78BC",
    }

    ensure_dir(args.output_prefix)

    for prop1, prop2, xlabel, ylabel in property_pairs:
        fig, ax = plt.subplots(figsize=(7, 7))

        model_predictions: List[Tuple[str, np.ndarray, np.ndarray, str]] = []
        added_models = set()

        for nn_run in nn_runs:
            if prop1 in nn_run["targets"] and prop2 in nn_run["targets"]:
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
                            nn_run["targets"][prop1]["y_pred"],
                            nn_run["targets"][prop2]["y_pred"],
                            model_colors.get(std_label, "#888888"),
                        )
                    )
                    added_models.add(std_label)

        prop1_idx = TARGET_COLUMNS.index(prop1)
        prop2_idx = TARGET_COLUMNS.index(prop2)
        model_predictions.append(
            (
                "NF (Mean)",
                nf_run["mean"][:, prop1_idx],
                nf_run["mean"][:, prop2_idx],
                model_colors["NF (Mean)"],
            )
        )
        model_predictions.append(
            (
                "NF (Random)",
                nf_run["random"][:, prop1_idx],
                nf_run["random"][:, prop2_idx],
                model_colors["NF (Random)"],
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
        for label, color in model_colors.items():
            if label in plotted_labels:
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

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    nn_labels = args.nn_labels or args.nn_runs
    if len(nn_labels) != len(args.nn_runs):
        raise ValueError("nn-labels count must match nn-runs count.")

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


if __name__ == "__main__":
    main()

