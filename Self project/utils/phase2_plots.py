#!/usr/bin/env python3
"""
Phase 2: Plotting utilities

Generates Phase 2 figures (starting with Figure 4) using precomputed
evaluation metrics for the 20 test simulations.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2_plots"
DEFAULT_RESULTS_CSV = (
    BASE_DIR / "outputs" / "phase2_universal_scatter_model" / "phase2_results.csv"
)
PHASE1_RESULTS_CSV = BASE_DIR / "outputs" / "phase1_plots" / "phase1_results.csv"

# Training sets (excluded from tests)
TRAINING_SETS = ["LH_135", "LH_473", "LH_798"]
COSMO_PARAMS_FILE = (
    BASE_DIR / "data" / "raw" / "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_cosmo_params() -> pd.DataFrame:
    """Load cosmological parameters table."""
    df = pd.read_csv(
        COSMO_PARAMS_FILE,
        sep=r"\s+",
        skiprows=1,
        names=[
            "Name",
            "Omega_m",
            "sigma_8",
            "A_SN1",
            "A_AGN1",
            "A_SN2",
            "A_AGN2",
            "seed",
        ],
    )
    df["Name"] = df["Name"].str.strip()
    return df


def _simulation_to_cosmo_name(sim_name: str) -> str:
    """Map lowercase dir or alias to LH_### format."""
    name = sim_name.strip()
    if name.lower().startswith("lh") and "_" not in name:
        num = name.lower().replace("lh", "")
        return f"LH_{num}"
    return name.upper()


def compute_multivariate_distance(test_row: pd.Series, training_rows: pd.DataFrame) -> float:
    """
    Mahalanobis-like distance in standardized units across four parameters.

    Parameters used: Omega_m, sigma_8, A_SN1, A_AGN1.
    If any std is zero, fall back to absolute difference (std=1 guard).
    """
    params = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1"]
    means = training_rows[params].mean()
    stds = training_rows[params].std().replace(0, 1.0)
    diffs = (test_row[params] - means) / stds
    # Euclidean norm in standardized space
    return float(np.sqrt(np.sum(diffs.values**2)))


def ensure_results_columns(df: pd.DataFrame, cosmo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and fill derived columns.

    Required columns after processing:
    simulation, distance_sigma, a_sn1, rmse, pearson_r, num_galaxies
    """
    rename_map: Dict[str, str] = {
        "sim": "simulation",
        "sim_name": "simulation",
        "simulation_name": "simulation",
        "distance": "distance_sigma",
        "distance_from_training": "distance_sigma",
        "A_SN1": "a_sn1",
        "asn1": "a_sn1",
        "rmse_phase2": "rmse",
        "pearson": "pearson_r",
        "pearsonR": "pearson_r",
        "pearson_rho": "pearson_r",
        "num_galaxies": "num_galaxies",
        "n_galaxies": "num_galaxies",
        "N": "num_galaxies",
        "count": "num_galaxies",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Core columns
    required = ["simulation", "rmse", "a_sn1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results: {missing}")

    # Distance: compute if absent (uses 4-parameter standardized distance)
    if "distance_sigma" not in df.columns:
        training_rows = cosmo_df[cosmo_df["Name"].isin(TRAINING_SETS)]
        distances: List[float] = []
        for sim in df["simulation"]:
            cosmo_name = _simulation_to_cosmo_name(sim)
            row = cosmo_df[cosmo_df["Name"] == cosmo_name]
            if len(row) == 0:
                distances.append(np.nan)
                continue
            test_row = row.iloc[0]
            distances.append(compute_multivariate_distance(test_row, training_rows))
        df["distance_sigma"] = distances

    # Pearson R optional but needed for Panel D
    if "pearson_r" not in df.columns:
        df["pearson_r"] = np.nan

    # Number of galaxies optional (used for point size)
    if "num_galaxies" not in df.columns:
        df["num_galaxies"] = 1

    # Type conversion
    for col in ["distance_sigma", "rmse", "a_sn1", "pearson_r", "num_galaxies"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows lacking key metrics
    df = df.dropna(subset=["distance_sigma", "rmse", "a_sn1"])
    return df


def load_phase1_best_results(path: Path) -> pd.DataFrame:
    """
    Load Phase 1 results and keep the best (lowest RMSE) model per simulation.

    Returns a dataframe with columns:
    - simulation
    - phase1_rmse
    - phase1_model
    """
    df = pd.read_csv(path)
    if not {"simulation", "rmse", "model"}.issubset(df.columns):
        raise ValueError("Phase 1 results CSV must contain 'simulation', 'rmse', 'model' columns.")

    idx = df.groupby("simulation")["rmse"].idxmin()
    best = df.loc[idx, ["simulation", "rmse", "model"]].copy()
    best = best.rename(columns={"rmse": "phase1_rmse", "model": "phase1_model"})
    return best


def _compute_bins(series: pd.Series, num_bins: int = 5) -> pd.IntervalIndex:
    """Helper to create quantile-ish bins (fallback to linear if insufficient unique)."""
    unique_vals = series.dropna().unique()
    if len(unique_vals) < num_bins:
        # fallback: min-max linear bins
        eps = 1e-6
        bins = np.linspace(series.min() - eps, series.max() + eps, num_bins + 1)
        return pd.IntervalIndex.from_breaks(bins)
    try:
        return pd.qcut(series, q=num_bins, retbins=False, duplicates="drop")
    except Exception:
        eps = 1e-6
        bins = np.linspace(series.min() - eps, series.max() + eps, num_bins + 1)
        return pd.IntervalIndex.from_breaks(bins)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _add_trend_line(ax, x: np.ndarray, y: np.ndarray, color: str = "#555"):
    """Add linear fit with R² annotation."""
    if len(x) < 2:
        return None
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    x_span = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(x_span, np.polyval(coeffs, x_span), color=color, linestyle="-", linewidth=2, label=f"Trend (R²={r2:.2f})")
    ax.legend(loc="upper left", fontsize=9)
    return r2


def plot_figure4(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 4: Phase 2 Standalone Performance (Panels A–D)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.28, hspace=0.32, bottom=0.08, top=0.94)
    panel_a, panel_b, panel_c, panel_d = axes.flatten()

    # Panel A: RMSE vs Distance
    sizes = 60 + 140 * (df["num_galaxies"] / df["num_galaxies"].max())
    sc = panel_a.scatter(
        df["distance_sigma"],
        df["rmse"],
        c=df["a_sn1"],
        s=sizes,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidth=1.0,
    )
    panel_a.axhline(0.40, color="#2ca02c", linestyle="--", linewidth=2, alpha=0.7, label="Success (0.40 dex)")
    panel_a.axhline(1.0, color="#d62728", linestyle="--", linewidth=2, alpha=0.7, label="Phase 1 failure (1.0 dex)")
    _add_trend_line(panel_a, df["distance_sigma"].values, df["rmse"].values)
    cbar = plt.colorbar(sc, ax=panel_a)
    cbar.set_label("A_SN1", fontsize=11, fontweight="bold")
    panel_a.set_xlabel("Distance from training (σ)", fontsize=12, fontweight="bold")
    panel_a.set_ylabel("Phase 2 RMSE (dex)", fontsize=12, fontweight="bold")
    panel_a.set_title("Panel A: RMSE vs Distance", fontsize=14, fontweight="bold")
    panel_a.grid(alpha=0.3)

    # Panel B: Performance by Feedback Regime
    sc_b = panel_b.scatter(
        df["a_sn1"],
        df["rmse"],
        c=df["distance_sigma"],
        cmap="plasma",
        s=90,
        edgecolors="black",
        linewidth=1.0,
    )
    panel_b.axvline(1.16, color="#ff7f0e", linestyle="--", linewidth=2, label="Training mean A_SN1 = 1.16")
    panel_b.axhline(0.40, color="#2ca02c", linestyle="--", linewidth=2, label="Success (0.40 dex)")
    panel_b.axvspan(1.5, df["a_sn1"].max() + 0.1, color="#d62728", alpha=0.08, label="Mode collapse zone (A_SN1 > 1.5)")
    cbar_b = plt.colorbar(sc_b, ax=panel_b)
    cbar_b.set_label("Distance (σ)", fontsize=11, fontweight="bold")
    panel_b.set_xlabel("True A_SN1", fontsize=12, fontweight="bold")
    panel_b.set_ylabel("Phase 2 RMSE (dex)", fontsize=12, fontweight="bold")
    panel_b.set_title("Panel B: Performance by Feedback Regime", fontsize=14, fontweight="bold")
    panel_b.legend(loc="upper left", fontsize=9)
    panel_b.grid(alpha=0.3)

    # Panel C: Success Rate by Distance Bin
    bins = [0, 5, 10, 15, 20, np.inf]
    labels = ["<5σ", "5–10σ", "10–15σ", "15–20σ", ">20σ"]
    df["distance_bin"] = pd.cut(df["distance_sigma"], bins=bins, labels=labels, include_lowest=True)
    success_rates = []
    for label in labels:
        subset = df[df["distance_bin"] == label]
        if len(subset) == 0:
            success_rates.append(0.0)
            continue
        success = np.mean(subset["rmse"] < 0.40) * 100
        success_rates.append(success)
    colors = []
    for rate in success_rates:
        if rate >= 80:
            colors.append("#2ca02c")
        elif rate >= 60:
            colors.append("#ffbf00")
        else:
            colors.append("#d62728")
    bars = panel_c.bar(labels, success_rates, color=colors, edgecolor="black")
    panel_c.set_ylim(0, max(100, max(success_rates) * 1.1 if success_rates else 100))
    panel_c.set_ylabel("Success rate (RMSE < 0.40) [%]", fontsize=12, fontweight="bold")
    panel_c.set_title("Panel C: Success Rate by Distance Bin", fontsize=14, fontweight="bold")
    panel_c.grid(alpha=0.3, axis="y")
    for bar, rate in zip(bars, success_rates):
        panel_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel D: Correlation Preservation
    sc_d = panel_d.scatter(
        df["distance_sigma"],
        df["pearson_r"],
        c=df["a_sn1"],
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidth=1.0,
    )
    panel_d.axhline(0.90, color="#2ca02c", linestyle="--", linewidth=2, label="Threshold R = 0.90")
    panel_d.axhspan(0.85, 1.0, color="#2ca02c", alpha=0.08, label="Acceptable (R > 0.85)")
    cbar_d = plt.colorbar(sc_d, ax=panel_d)
    cbar_d.set_label("A_SN1", fontsize=11, fontweight="bold")
    panel_d.set_xlabel("Distance from training (σ)", fontsize=12, fontweight="bold")
    panel_d.set_ylabel("Pearson R", fontsize=12, fontweight="bold")
    panel_d.set_title("Panel D: Correlation Preservation", fontsize=14, fontweight="bold")
    panel_d.legend(loc="lower left", fontsize=9)
    panel_d.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Figure 4 to {output_path}")


def _pick_representative_indices_by_asn1(a_sn1_vals: np.ndarray) -> List[int]:
    """Pick indices closest to target A_SN1 values for crash-style panels."""
    targets = np.array([0.3, 1.2, 2.5, 4.0])
    idxs: List[int] = []
    for t in targets:
        diffs = np.abs(a_sn1_vals - t)
        idx = int(np.argmin(diffs))
        if idx not in idxs:
            idxs.append(idx)
    return idxs


def plot_figure5(df_merged: pd.DataFrame, output_path: Path) -> None:
    """
    Generate Figure 5: Phase 1 vs Phase 2 Direct Comparison.

    Expects df_merged to contain:
    - simulation
    - distance_sigma
    - a_sn1
    - rmse (Phase 2)
    - phase1_rmse
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df_merged.copy()
    df["improvement"] = df["phase1_rmse"] / df["rmse"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # Panel A: Before-After Scatter
    sizes = 60 + 140 * (df["improvement"] / df["improvement"].max())
    sc_a = ax_a.scatter(
        df["phase1_rmse"],
        df["rmse"],
        c=df["distance_sigma"],
        s=sizes,
        cmap="viridis",
        edgecolors="black",
        linewidth=1.0,
        alpha=0.85,
    )
    lim_min = 0
    lim_max = max(df["phase1_rmse"].max(), df["rmse"].max()) * 1.05
    ax_a.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1.5, label="y = x")
    ax_a.set_xlim(lim_min, lim_max)
    ax_a.set_ylim(lim_min, lim_max)
    ax_a.set_xlabel("Phase 1 RMSE (best model)", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Phase 2 RMSE", fontsize=12, fontweight="bold")
    ax_a.set_title("Panel A: Before vs After", fontsize=14, fontweight="bold")
    below = int(np.sum(df["rmse"] < df["phase1_rmse"]))
    ax_a.text(
        0.05,
        0.95,
        f"{below}/{len(df)} below diagonal",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    cbar_a = plt.colorbar(sc_a, ax=ax_a)
    cbar_a.set_label("Distance from training (σ)", fontsize=11, fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=9)
    ax_a.grid(alpha=0.3)

    # Panel B: Improvement factor vs distance
    sc_b = ax_b.scatter(
        df["distance_sigma"],
        df["improvement"],
        c=df["a_sn1"],
        cmap="plasma",
        s=90,
        edgecolors="black",
        linewidth=1.0,
    )
    ax_b.axhline(1.0, color="#d62728", linestyle="--", linewidth=2, label="No improvement (×1)")
    ax_b.axhline(3.0, color="#2ca02c", linestyle="--", linewidth=2, label="Target (×3)")
    ax_b.set_xlabel("Distance from training (σ)", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Improvement factor (Phase1 / Phase2)", fontsize=12, fontweight="bold")
    ax_b.set_title("Panel B: Improvement vs Distance", fontsize=14, fontweight="bold")
    cbar_b = plt.colorbar(sc_b, ax=ax_b)
    cbar_b.set_label("A_SN1", fontsize=11, fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=9)
    ax_b.grid(alpha=0.3)

    # Panel C: Side-by-side bar chart
    df_sorted = df.sort_values("distance_sigma").reset_index(drop=True)
    x = np.arange(len(df_sorted))
    width = 0.38

    ax_c.bar(
        x - width / 2,
        df_sorted["phase1_rmse"],
        width=width,
        color="#1f77b4",
        label="Phase 1 (best model)",
    )
    ax_c.bar(
        x + width / 2,
        df_sorted["rmse"],
        width=width,
        color="#ff7f0e",
        label="Phase 2 (uncalibrated)",
    )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(df_sorted["simulation"], rotation=45, ha="right", fontsize=9)
    ax_c.set_ylabel("RMSE (dex)", fontsize=12, fontweight="bold")
    ax_c.set_title("Panel C: Simulation-wise RMSE Comparison", fontsize=14, fontweight="bold")
    ax_c.legend(fontsize=9)
    ax_c.grid(alpha=0.3, axis="y")

    # Panel D: Failure mode resolution (2×2 crash-style bars)
    # Reuse representative A_SN1 targets
    a_sn1_vals = df_sorted["a_sn1"].values
    idxs = _pick_representative_indices_by_asn1(a_sn1_vals)
    titles = ["Low A_SN1 (~0.3)", "Medium A_SN1 (~1.2)", "High A_SN1 (~2.5)", "Very high A_SN1 (~4.0)"]

    ax_main = ax_d
    ax_spec = ax_main.get_subplotspec()
    ax_main.remove()
    sub_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax_spec, wspace=0.25, hspace=0.3)
    sub_axes: List = []
    for i in range(2):
        for j in range(2):
            sub_axes.append(fig.add_subplot(sub_gs[i, j]))

    for ax_sub, idx, title in zip(sub_axes, idxs, titles):
        row = df_sorted.iloc[idx]
        vals = [row["phase1_rmse"], row["rmse"]]
        labels = ["Phase 1", "Phase 2"]
        colors = ["#1f77b4", "#ff7f0e"]
        xpos = np.arange(2)
        ax_sub.bar(xpos, vals, color=colors, width=0.6)
        ax_sub.axhline(0.40, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_sub.set_xticks(xpos)
        ax_sub.set_xticklabels(labels, rotation=0, fontsize=8)
        ax_sub.set_title(f"{title}\n{row['simulation']}", fontsize=9)
        ax_sub.tick_params(labelsize=8)
        ax_sub.grid(alpha=0.2, axis="y")

    # Position the Panel D title slightly lower/right to avoid overlap with Panel B x-label
    fig.text(0.80, 0.50, "Panel D: Failure Mode Resolution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Figure 5 to {output_path}")


def plot_figure6_bias(df: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 6: Bias Analysis.
    Requires df with columns: a_sn1, bias (Phase 2), distance_sigma, num_galaxies.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.28, hspace=0.32, bottom=0.08, top=0.94)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # Panel A: Bias vs A_SN1 (scatter, color=distance, size=galaxies)
    sizes = 40 + 160 * (df["num_galaxies"] / df["num_galaxies"].max())
    sc_a = ax_a.scatter(
        df["a_sn1"],
        df["bias"],
        c=df["distance_sigma"],
        s=sizes,
        cmap="viridis",
        edgecolors="black",
        linewidth=1.0,
        alpha=0.85,
    )
    ax_a.axhline(0.0, color="#d62728", linestyle="--", linewidth=2, alpha=0.8, label="Zero bias")
    ax_a.set_xlabel("True A_SN1", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Bias (dex)", fontsize=12, fontweight="bold")
    ax_a.set_title("Panel A: Bias vs Feedback Strength", fontsize=14, fontweight="bold")
    ax_a.grid(alpha=0.3)
    cbar_a = plt.colorbar(sc_a, ax=ax_a)
    cbar_a.set_label("Distance from training (σ)", fontsize=11, fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=9)

    # Panel B: Bias vs distance
    sc_b = ax_b.scatter(
        df["distance_sigma"],
        df["bias"],
        c=df["a_sn1"],
        cmap="plasma",
        s=90,
        edgecolors="black",
        linewidth=1.0,
        alpha=0.85,
    )
    ax_b.axhline(0.0, color="#d62728", linestyle="--", linewidth=2, alpha=0.8, label="Zero bias")
    ax_b.set_xlabel("Distance from training (σ)", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Bias (dex)", fontsize=12, fontweight="bold")
    ax_b.set_title("Panel B: Bias vs Extrapolation Distance", fontsize=14, fontweight="bold")
    cbar_b = plt.colorbar(sc_b, ax=ax_b)
    cbar_b.set_label("A_SN1", fontsize=11, fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=9)
    ax_b.grid(alpha=0.3)

    # Panel C: Binned bias by A_SN1
    bins_asn1 = _compute_bins(df["a_sn1"], num_bins=5)
    df["asn1_bin"] = pd.cut(df["a_sn1"], bins=bins_asn1) if not isinstance(bins_asn1, pd.Series) else bins_asn1
    grouped = df.groupby("asn1_bin")["bias"].agg(["mean", "count"])
    labels = [f"{idx.left:.2f}-{idx.right:.2f}" for idx in grouped.index]
    colors = ["#1f77b4"] * len(grouped)
    bars = ax_c.bar(labels, grouped["mean"], color=colors, edgecolor="black")
    ax_c.axhline(0.0, color="#d62728", linestyle="--", linewidth=2, alpha=0.8)
    ax_c.set_xlabel("A_SN1 bin", fontsize=12, fontweight="bold")
    ax_c.set_ylabel("Mean bias (dex)", fontsize=12, fontweight="bold")
    ax_c.set_title("Panel C: Binned Bias vs A_SN1", fontsize=14, fontweight="bold")
    ax_c.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    for bar, n in zip(bars, grouped["count"]):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}", ha="center", va="bottom", fontsize=9)
    ax_c.grid(alpha=0.3, axis="y")

    # Panel D: Binned bias by distance
    bins_dist = [0, 5, 10, 15, 20, np.inf]
    labels_dist = ["<5σ", "5–10σ", "10–15σ", "15–20σ", ">20σ"]
    df["distance_bin_bias"] = pd.cut(df["distance_sigma"], bins=bins_dist, labels=labels_dist, include_lowest=True)
    grouped_d = df.groupby("distance_bin_bias")["bias"].agg(["mean", "count"])
    colors_d = ["#2ca02c" if m > -0.05 and m < 0.05 else "#ff7f0e" for m in grouped_d["mean"]]
    bars_d = ax_d.bar(labels_dist, grouped_d["mean"], color=colors_d, edgecolor="black")
    ax_d.axhline(0.0, color="#d62728", linestyle="--", linewidth=2, alpha=0.8)
    ax_d.set_xlabel("Distance bin (σ)", fontsize=12, fontweight="bold")
    ax_d.set_ylabel("Mean bias (dex)", fontsize=12, fontweight="bold")
    ax_d.set_title("Panel D: Binned Bias vs Distance", fontsize=14, fontweight="bold")
    for bar, n in zip(bars_d, grouped_d["count"]):
        ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}", ha="center", va="bottom", fontsize=9)
    ax_d.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Figure 6 to {output_path}")

# Backward-compatible alias
def plot_figure6(df: pd.DataFrame, output_path: Path) -> None:
    """Alias for plot_figure6_bias to match CLI call."""
    plot_figure6_bias(df, output_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 plotting utility")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to Phase 2 results CSV (default: outputs/phase2_universal_scatter_model/phase2_results.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "figure4_phase2_standalone_performance.png",
        help="Output path for Figure 4 PNG.",
    )
    parser.add_argument(
        "--phase1-results-csv",
        type=Path,
        default=PHASE1_RESULTS_CSV,
        help="Path to Phase 1 results CSV (default: outputs/phase1_plots/phase1_results.csv).",
    )
    parser.add_argument(
        "--make-figure5",
        action="store_true",
        help="If set, also generate Figure 5 (Phase 1 vs Phase 2 comparison).",
    )
    parser.add_argument(
        "--output-fig5",
        type=Path,
        default=OUTPUT_DIR / "figure5_phase1_vs_phase2_comparison.png",
        help="Output path for Figure 5 PNG.",
    )
    parser.add_argument(
        "--make-figure6",
        action="store_true",
        help="If set, also generate Figure 6 (Bias analysis). Requires bias column in Phase 2 results.",
    )
    parser.add_argument(
        "--output-fig6",
        type=Path,
        default=OUTPUT_DIR / "figure6_phase2_bias_analysis.png",
        help="Output path for Figure 6 PNG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("Phase 2: Generate Result Plots")
    print("=" * 80)

    if not args.results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

    cosmo_df = load_cosmo_params()
    df = pd.read_csv(args.results_csv)
    df = ensure_results_columns(df, cosmo_df)

    if len(df) == 0:
        raise ValueError("No valid rows found after cleaning the results CSV.")

    print(f"Loaded {len(df)} simulations for plotting.")
    plot_figure4(df, args.output)

    if args.make_figure5:
        if not args.phase1_results_csv.exists():
            raise FileNotFoundError(f"Phase 1 results CSV not found: {args.phase1_results_csv}")
        phase1_best = load_phase1_best_results(args.phase1_results_csv)
        merged = df.merge(phase1_best, on="simulation", how="inner")
        if len(merged) == 0:
            raise ValueError("No overlapping simulations between Phase 1 and Phase 2 results for Figure 5.")
        print(f"Generating Figure 5 with {len(merged)} overlapping simulations.")
        plot_figure5(merged, args.output_fig5)

    if args.make_figure6:
        if "bias" not in df.columns:
            raise ValueError("Phase 2 results CSV must include 'bias' column to plot Figure 6.")
        print(f"Generating Figure 6 (bias analysis) with {len(df)} simulations.")
        plot_figure6(df, args.output_fig6)

    print("\nPhase 2 plotting complete.")


if __name__ == "__main__":
    main()

