#!/usr/bin/env python3
"""
Phase 3: Few‑Shot Calibration – Plotting utilities

Generates Phase 3 figures (starting with Figure 7) using pre‑computed
calibration learning‑curve metrics across all 20 test simulations.

FIGURE 7: The Calibration Learning Curves (money plot)
  Panel A: RMSE learning curves by A_SN1 regime
  Panel B: |bias| learning curves by A_SN1 regime
  Panel C: Calibration efficiency (minimum N to reach RMSE < 0.40)
  Panel D: Individual simulation trajectories (spaghetti plot)

Expected input (one row per (simulation, N_calib)):
  - simulation : simulation name (e.g., "LH_122" or "lh122")
  - N_calib   : number of calibration galaxies used (int)
  - rmse      : RMSE after calibration (dex)
  - bias      : bias after calibration (dex, pred - true)

Optional columns (will be inferred/renamed if present):
  - a_sn1     : feedback strength A_SN1 for that simulation

If a_sn1 is missing, it will be looked up from the cosmology table
`CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt` using the simulation name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import sys

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "outputs" / "phase3_plots"

# Default location for the consolidated few‑shot calibration results.
# You can override this on the CLI with --results-csv.
DEFAULT_RESULTS_CSV = (
    BASE_DIR / "outputs" / "phase3_calibration" / "phase3_learning_curves.csv"
)

TRAINING_SETS: List[str] = ["LH_135", "LH_473", "LH_798"]
COSMO_PARAMS_FILE = (
    BASE_DIR / "data" / "raw" / "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
)
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SHMR_JSON = (
    BASE_DIR
    / "outputs"
    / "phase2_universal_scatter_model"
    / "shmr_fit_parameters.json"
)
PHASE2_RESULTS_CSV = (
    BASE_DIR
    / "outputs"
    / "phase2_universal_scatter_model"
    / "phase2_results.csv"
)

from src.plots.nf_visualizer import (
    load_nf_artifacts,
    sample_nf_posterior_for_halos,
)
from src.utils.common import resolve_device


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def load_cosmo_params() -> pd.DataFrame:
    """Load cosmological / feedback parameters table."""
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
    """
    Map simulation identifier to the LH_### format used in the cosmology table.

    Examples:
      "lh122" -> "LH_122"
      "LH_122" -> "LH_122"
      "lh_122" (already LH-like) -> "LH_122"
    """
    name = sim_name.strip()
    # If already of the form LH_###
    if name.upper().startswith("LH_"):
        return name.upper()
    # If of the form lh### (no underscore)
    if name.lower().startswith("lh") and "_" not in name:
        num = name.lower().replace("lh", "")
        return f"LH_{num}"
    return name.upper()


def _assign_asn1_regime(a_sn1: float) -> str:
    """
    Assign A_SN1 regime used for calibration plots.

    Bin edges follow the science description:
      - Below Training (<1.0)
      - Within Training (1.0–1.3)
      - Moderate Extrap (1.3–2.5)
      - Extreme Extrap (>2.5)
    """
    if pd.isna(a_sn1):
        return "Unknown"
    if a_sn1 < 1.0:
        return "Below Training (<1.0)"
    if a_sn1 < 1.3:
        return "Within Training (1.0–1.3)"
    if a_sn1 < 2.5:
        return "Moderate Extrap (1.3–2.5)"
    return "Extreme Extrap (>2.5)"


def ensure_phase3_results_columns(df: pd.DataFrame, cosmo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Phase 3 learning‑curve results columns.

    Required output columns:
      simulation, N_calib, rmse, bias, a_sn1, regime

    The function is intentionally permissive about input column names
    (e.g., accepts 'N', 'n_calib', 'n_obs', 'rmse_calibrated', etc.).
    """
    rename_map: Dict[str, str] = {
        # simulation identifiers
        "sim": "simulation",
        "sim_name": "simulation",
        "simulation_name": "simulation",
        # calibration sample size
        "N": "N_calib",
        "n": "N_calib",
        "n_calib": "N_calib",
        "n_obs": "N_calib",
        "num_calib": "N_calib",
        "num_calibration_galaxies": "N_calib",
        "n_calibration_galaxies": "N_calib",
        # RMSE and bias
        "rmse_cal": "rmse",
        "rmse_calibrated": "rmse",
        "rmse_phase3": "rmse",
        "bias_cal": "bias",
        "bias_calibrated": "bias",
        # A_SN1
        "A_SN1": "a_sn1",
        "asn1": "a_sn1",
        "feedback_asn1": "a_sn1",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["simulation", "N_calib", "rmse", "bias"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Phase 3 results CSV is missing required columns: {missing}. "
            "Expected at least one row per (simulation, N_calib) with rmse and bias."
        )

    # Attach A_SN1 if missing
    if "a_sn1" not in df.columns:
        asn1_vals: List[float] = []
        for sim in df["simulation"]:
            cosmo_name = _simulation_to_cosmo_name(str(sim))
            row = cosmo_df[cosmo_df["Name"] == cosmo_name]
            if len(row) == 0:
                asn1_vals.append(np.nan)
            else:
                asn1_vals.append(float(row.iloc[0]["A_SN1"]))
        df["a_sn1"] = asn1_vals

    # Type conversions
    df["N_calib"] = pd.to_numeric(df["N_calib"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["bias"] = pd.to_numeric(df["bias"], errors="coerce")
    df["a_sn1"] = pd.to_numeric(df["a_sn1"], errors="coerce")

    df = df.dropna(subset=["simulation", "N_calib", "rmse", "bias", "a_sn1"])

    # Assign regimes
    df["regime"] = df["a_sn1"].apply(_assign_asn1_regime)

    # Filter to known regimes only
    df = df[df["regime"] != "Unknown"].copy()

    return df


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #


def _load_shmr_params() -> Tuple[float, float]:
    """Load SHMR linear fit parameters (alpha, beta) from Phase 2 outputs."""
    if not SHMR_JSON.exists():
        raise FileNotFoundError(
            f"SHMR parameters not found at {SHMR_JSON}. "
            "Run the Phase 2 SHMR / universal scatter script first."
        )
    import json

    with SHMR_JSON.open("r") as f:
        raw = json.load(f)
    params = raw["parameters"]
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return alpha, beta


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return np.nan
    return float(np.mean(y_p - y_t))


# --------------------------------------------------------------------------- #
# Baseline Phase 2 loader (for Figures 7–9)
# --------------------------------------------------------------------------- #


def _load_phase2_baseline() -> pd.DataFrame:
    """
    Load Phase 2 baseline metrics (uncalibrated model).

    Returns dataframe with columns:
      simulation, rmse_baseline, bias_baseline

    We deliberately omit A_SN1 and distance_sigma here to avoid duplicate
    columns when merging with Phase 3 results, which already carry those.
    """
    if not PHASE2_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Phase 2 baseline results CSV not found: {PHASE2_RESULTS_CSV}"
        )
    df = pd.read_csv(PHASE2_RESULTS_CSV)
    required = {"simulation", "rmse", "bias"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Phase 2 results CSV is missing required columns: {sorted(missing)}"
        )
    df = df[list(required)].rename(
        columns={"rmse": "rmse_baseline", "bias": "bias_baseline"}
    )
    return df


# --------------------------------------------------------------------------- #
# Plotting – Figure 7: Calibration Learning Curves
# --------------------------------------------------------------------------- #


def plot_figure7(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate Figure 7: Calibration learning curves (4‑panel layout).

    Panels:
      A: RMSE vs N_calib (mean ± std) per regime
      B: |bias| vs N_calib (mean ± std) per regime
      C: Minimum N_calib to reach RMSE < 0.40 per regime
      D: Individual simulation trajectories (spaghetti; RMSE vs N_calib)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort and get unique calibration sizes
    df = df.copy()
    df["N_calib"] = df["N_calib"].astype(int)
    unique_ns = np.sort(df["N_calib"].unique())

    regimes_order = [
        "Below Training (<1.0)",
        "Within Training (1.0–1.3)",
        "Moderate Extrap (1.3–2.5)",
        "Extreme Extrap (>2.5)",
    ]
    colors: Dict[str, str] = {
        "Below Training (<1.0)": "#d62728",        # red
        "Within Training (1.0–1.3)": "#1f77b4",    # blue
        "Moderate Extrap (1.3–2.5)": "#2ca02c",    # green
        "Extreme Extrap (>2.5)": "#9467bd",        # purple
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.28, hspace=0.30, bottom=0.08, top=0.93)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # ----------------------------- Panel A --------------------------------- #
    # RMSE learning curves by regime
    for regime in regimes_order:
        df_reg = df[df["regime"] == regime]
        if df_reg.empty:
            continue
        grouped = (
            df_reg.groupby("N_calib")
            .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
            .reset_index()
            .sort_values("N_calib")
        )
        ax_a.plot(
            grouped["N_calib"],
            grouped["rmse_mean"],
            label=regime,
            color=colors.get(regime, "#333333"),
            linewidth=2.0,
        )
        ax_a.fill_between(
            grouped["N_calib"],
            grouped["rmse_mean"] - grouped["rmse_std"],
            grouped["rmse_mean"] + grouped["rmse_std"],
            color=colors.get(regime, "#333333"),
            alpha=0.20,
        )

    ax_a.axhline(
        0.40,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Success threshold (RMSE = 0.40)",
    )
    ax_a.set_xscale("log")
    # Use the actual N values as the major ticks so the x‑axis is interpretable
    ax_a.set_xticks(unique_ns)
    ax_a.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_a.get_xaxis().set_minor_formatter(ScalarFormatter())
    ax_a.set_xlabel("N calibration galaxies", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Mean RMSE (dex)", fontsize=12, fontweight="bold")
    ax_a.set_title("Panel A: RMSE Learning Curves by Regime", fontsize=14, fontweight="bold")
    ax_a.grid(alpha=0.3, which="both")
    ax_a.legend(loc="upper right", fontsize=9)

    # ----------------------------- Panel B --------------------------------- #
    # |bias| learning curves by regime
    df["abs_bias"] = df["bias"].abs()
    for regime in regimes_order:
        df_reg = df[df["regime"] == regime]
        if df_reg.empty:
            continue
        grouped = (
            df_reg.groupby("N_calib")
            .agg(bias_mean=("abs_bias", "mean"), bias_std=("abs_bias", "std"))
            .reset_index()
            .sort_values("N_calib")
        )
        ax_b.plot(
            grouped["N_calib"],
            grouped["bias_mean"],
            label=regime,
            color=colors.get(regime, "#333333"),
            linewidth=2.0,
        )
        ax_b.fill_between(
            grouped["N_calib"],
            grouped["bias_mean"] - grouped["bias_std"],
            grouped["bias_mean"] + grouped["bias_std"],
            color=colors.get(regime, "#333333"),
            alpha=0.20,
        )

    ax_b.set_xscale("log")
    ax_b.set_xticks(unique_ns)
    ax_b.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_b.get_xaxis().set_minor_formatter(ScalarFormatter())
    ax_b.set_xlabel("N calibration galaxies", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Mean |bias| (dex)", fontsize=12, fontweight="bold")
    ax_b.set_title("Panel B: Bias Correction by Regime", fontsize=14, fontweight="bold")
    ax_b.grid(alpha=0.3, which="both")
    ax_b.legend(loc="upper right", fontsize=9)

    # ----------------------------- Panel C --------------------------------- #
    # Efficiency: minimum N needed for success (RMSE < 0.40)
    min_ns: List[float] = []
    bar_colors: List[str] = []
    for regime in regimes_order:
        df_reg = df[df["regime"] == regime]
        if df_reg.empty:
            min_ns.append(np.nan)
            bar_colors.append("#d62728")
            continue

        grouped = (
            df_reg.groupby("N_calib")
            .agg(rmse_mean=("rmse", "mean"))
            .reset_index()
            .sort_values("N_calib")
        )
        below = grouped[grouped["rmse_mean"] < 0.40]
        if below.empty:
            min_ns.append(np.nan)
            bar_colors.append("#d62728")  # red = not achievable
        else:
            n_min = float(below["N_calib"].min())
            min_ns.append(n_min)
            bar_colors.append("#2ca02c")  # green = achievable

    x_pos = np.arange(len(regimes_order))
    # Replace NaN with zero just for plotting; we'll annotate those as "N/A"
    plot_values = [v if np.isfinite(v) else 0.0 for v in min_ns]
    bars = ax_c.bar(
        x_pos,
        plot_values,
        color=bar_colors,
        edgecolor="black",
    )
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(regimes_order, rotation=20, ha="right", fontsize=10)
    ax_c.set_ylabel("Minimum N for RMSE < 0.40", fontsize=12, fontweight="bold")
    ax_c.set_title(
        "Panel C: Calibration Efficiency – N Needed for Success",
        fontsize=14,
        fontweight="bold",
    )
    ax_c.grid(alpha=0.3, axis="y")

    for bar, val in zip(bars, min_ns):
        if np.isfinite(val):
            ax_c.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax_c.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * (ax_c.get_ylim()[1] or 1.0),
                "N/A",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # ----------------------------- Panel D --------------------------------- #
    # Individual simulation trajectories ("spaghetti")
    for (sim, regime), df_sim in df.groupby(["simulation", "regime"]):
        df_sim_sorted = df_sim.sort_values("N_calib")
        ax_d.plot(
            df_sim_sorted["N_calib"],
            df_sim_sorted["rmse"],
            color=colors.get(regime, "#aaaaaa"),
            alpha=0.6,
            linewidth=1.2,
        )

    ax_d.set_xscale("log")
    ax_d.set_xticks(unique_ns)
    ax_d.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_d.get_xaxis().set_minor_formatter(ScalarFormatter())
    ax_d.set_xlabel("N calibration galaxies", fontsize=12, fontweight="bold")
    ax_d.set_ylabel("RMSE (dex)", fontsize=12, fontweight="bold")
    ax_d.set_title(
        "Panel D: Individual Simulation Trajectories", fontsize=14, fontweight="bold"
    )
    ax_d.grid(alpha=0.3, which="both")

    # Build a legend proxy for regimes (so spaghetti colors are interpretable)
    regime_handles = []
    for regime in regimes_order:
        if regime not in df["regime"].unique():
            continue
        handle = plt.Line2D(
            [], [], color=colors.get(regime, "#333333"), linewidth=2, label=regime
        )
        regime_handles.append(handle)
    if regime_handles:
        ax_d.legend(
            handles=regime_handles,
            title="Regime (color)",
            loc="upper right",
            fontsize=9,
            title_fontsize=10,
        )

    fig.suptitle(
        "Figure 7: Calibration Learning Curves (Few‑Shot Calibration Phase)",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Figure 7 to {output_path}")


# --------------------------------------------------------------------------- #
# Plotting – Figure 8: Calibration Case Studies
# --------------------------------------------------------------------------- #


def plot_figure8_case_studies(
    output_path: Path,
    model_run: str = "delta_sm",
    num_draws: int = 500,
    n_calib: int = 10,
    rng_seed: int = 123,
) -> None:
    """
    Figure 8: Calibration Case Studies (2×4 grid).

    For each of four representative simulations:
      - Left panel: Uncalibrated Phase 2 (N = 0)
      - Right panel: After N_calib‑shot calibration (default N_calib = 10)

    Scatter: SM_pred vs SM_true with diagonal line and annotated RMSE/bias.
    """
    sims = [
        ("LH_306", "Low A_SN1 (~0.3): LH_306"),
        ("LH_493", "Medium A_SN1 (~1.2): LH_493"),
        ("LH_817", "High A_SN1 (~2.5): LH_817"),
        ("LH_296", "Very High A_SN1 (~4.0): LH_296"),
    ]

    device = resolve_device("cpu")
    artifacts = load_nf_artifacts(model_run, device)
    features = artifacts["features"]
    targets = artifacts["targets"]

    # Determine which target we have
    if "Delta_SM" in targets:
        sm_idx = targets.index("Delta_SM")
        is_delta = True
    elif "SM_delta" in targets:
        sm_idx = targets.index("SM_delta")
        is_delta = True
    elif "SM" in targets:
        sm_idx = targets.index("SM")
        is_delta = False
    else:
        raise ValueError(
            f"Model must predict 'Delta_SM'/'SM_delta' or 'SM'; got targets={targets}"
        )

    alpha, beta = _load_shmr_params()
    rng = np.random.default_rng(rng_seed)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    plt.subplots_adjust(wspace=0.30, hspace=0.35, bottom=0.08, top=0.90)

    for i, (sim_upper, label) in enumerate(sims):
        # Map 'LH_306' -> 'lh306'
        num = sim_upper.split("_", 1)[1]
        sim_dir = f"lh{num}"
        parquet_path = PROCESSED_DATA_DIR / sim_dir / "halo_galaxy.parquet"
        if not parquet_path.exists():
            print(f"[Figure 8] WARNING: Missing parquet for {sim_upper} at {parquet_path}; skipping.")
            continue

        df = pd.read_parquet(parquet_path)
        if not set(features).issubset(df.columns) or "SM" not in df.columns:
            print(f"[Figure 8] WARNING: Missing required columns for {sim_upper}; skipping.")
            continue

        halos = df[features].values
        sm_true = df["SM"].values

        post = sample_nf_posterior_for_halos(
            halos=halos,
            artifacts=artifacts,
            num_draws=num_draws,
            device=device,
            batch_size=256,
        )
        target_samples = post[:, :, sm_idx]
        target_median = np.median(target_samples, axis=1)

        if is_delta:
            sm_shmr = alpha * df["M_h"].values + beta
            sm_pred_nf = sm_shmr + target_median
        else:
            sm_pred_nf = target_median

        # Uncalibrated metrics
        rmse_uncal = _rmse(sm_true, sm_pred_nf)
        bias_uncal = _bias(sm_true, sm_pred_nf)

        # Calibrated metrics (N = n_calib)
        n_gal = len(df)
        if n_gal < n_calib:
            print(
                f"[Figure 8] WARNING: {sim_upper} has only {n_gal} galaxies; "
                f"cannot perform {n_calib}-shot calibration. Using all galaxies instead."
            )
            n_use = n_gal
        else:
            n_use = n_calib

        idx = rng.choice(n_gal, size=n_use, replace=False)
        sm_true_obs = sm_true[idx]
        sm_pred_obs = sm_pred_nf[idx]
        delta = float(np.mean(sm_true_obs - sm_pred_obs))
        sm_cal = sm_pred_nf + delta

        rmse_cal = _rmse(sm_true, sm_cal)
        bias_cal = _bias(sm_true, sm_cal)

        # Common plot limits
        all_vals = np.concatenate([sm_true, sm_pred_nf, sm_cal])
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)
        pad = 0.1 * (vmax - vmin if vmax > vmin else 1.0)
        lims = [vmin - pad, vmax + pad]

        # Choose axes: row 0 for first two sims, row 1 for last two
        row = 0 if i < 2 else 1
        col_uncal = (i % 2) * 2
        col_cal = col_uncal + 1

        ax_uncal = axes[row, col_uncal]
        ax_cal = axes[row, col_cal]

        # Uncalibrated scatter
        ax_uncal.scatter(
            sm_true,
            sm_pred_nf,
            s=6,
            alpha=0.35,
            color="#1f77b4",
            rasterized=True,
        )
        ax_uncal.plot(lims, lims, "r--", linewidth=1.0, alpha=0.8)
        ax_uncal.set_xlim(lims)
        ax_uncal.set_ylim(lims)
        ax_uncal.set_title(f"{label}\nUncalibrated (Phase 2)", fontsize=10)
        ax_uncal.tick_params(labelsize=8)
        ax_uncal.grid(alpha=0.2)
        ax_uncal.text(
            0.04,
            0.96,
            f"RMSE = {rmse_uncal:.3f}\nBias = {bias_uncal:+.3f} dex",
            transform=ax_uncal.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )

        # Calibrated scatter
        ax_cal.scatter(
            sm_true,
            sm_cal,
            s=6,
            alpha=0.35,
            color="#2ca02c",
            rasterized=True,
        )
        ax_cal.plot(lims, lims, "r--", linewidth=1.0, alpha=0.8)
        ax_cal.set_xlim(lims)
        ax_cal.set_ylim(lims)
        ax_cal.set_title(
            f"{label}\nAfter {n_use}-shot calibration", fontsize=10
        )
        ax_cal.tick_params(labelsize=8)
        ax_cal.grid(alpha=0.2)
        ax_cal.text(
            0.04,
            0.96,
            f"RMSE = {rmse_cal:.3f}\nBias = {bias_cal:+.3f} dex",
            transform=ax_cal.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )

        if row == 1:
            ax_uncal.set_xlabel("True SM (dex)", fontsize=9, fontweight="bold")
            ax_cal.set_xlabel("True SM (dex)", fontsize=9, fontweight="bold")
        if col_uncal == 0:
            ax_uncal.set_ylabel("Predicted SM (dex)", fontsize=9, fontweight="bold")
        if col_cal == 1:
            ax_cal.set_ylabel("Predicted SM (dex)", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Figure 8: Calibration Case Studies (Uncalibrated vs 10‑Shot Calibration)",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Figure 8 to {output_path}")


# --------------------------------------------------------------------------- #
# Plotting – Figure 9: Calibration Strategy Analysis
# --------------------------------------------------------------------------- #


def plot_figure9_strategy(
    df_phase3: pd.DataFrame,
    output_path: Path,
    n_target: int = 10,
) -> None:
    """
    Figure 9: Calibration Strategy Analysis (4‑panel layout).

    Uses:
      - Phase 2 baseline results (uncalibrated, N=0)
      - Phase 3 learning‑curve results (N > 0)
    """
    baseline = _load_phase2_baseline()

    # Merge baseline into Phase 3 dataframe
    df = df_phase3.merge(baseline, on="simulation", how="inner")
    if len(df) == 0:
        raise ValueError(
            "No overlapping simulations between Phase 2 baseline and Phase 3 results."
        )

    df = df.copy()
    df["N_calib"] = df["N_calib"].astype(int)
    df["rmse_improvement"] = df["rmse_baseline"] - df["rmse"]

    # For panels that specifically use N = n_target (e.g., 10)
    df_n_target = df[df["N_calib"] == n_target].copy()
    if df_n_target.empty:
        raise ValueError(
            f"No Phase 3 rows found with N_calib == {n_target}. "
            "Regenerate the learning‑curve CSV with this N value included."
        )

    regimes_order = [
        "Below Training (<1.0)",
        "Within Training (1.0–1.3)",
        "Moderate Extrap (1.3–2.5)",
        "Extreme Extrap (>2.5)",
    ]
    colors: Dict[str, str] = {
        "Below Training (<1.0)": "#d62728",        # red
        "Within Training (1.0–1.3)": "#1f77b4",    # blue
        "Moderate Extrap (1.3–2.5)": "#2ca02c",    # green
        "Extreme Extrap (>2.5)": "#9467bd",        # purple
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.30, hspace=0.32, bottom=0.08, top=0.92)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # ----------------------------- Panel A --------------------------------- #
    # Cost‑Benefit: RMSE improvement vs N_calib
    grouped_all = (
        df.groupby("N_calib")["rmse_improvement"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("N_calib")
    )

    ax_a.plot(
        grouped_all["N_calib"],
        grouped_all["mean"],
        color="black",
        linewidth=2.5,
        label="All simulations (mean)",
    )
    ax_a.fill_between(
        grouped_all["N_calib"],
        grouped_all["mean"] - grouped_all["std"],
        grouped_all["mean"] + grouped_all["std"],
        color="gray",
        alpha=0.20,
    )

    # Low‑A_SN1 and High‑A_SN1 subsets
    low_mask = df["regime"] == "Below Training (<1.0)"
    high_mask = df["regime"] == "Extreme Extrap (>2.5)"
    for mask, label, color in [
        (low_mask, "Low A_SN1 regime", "#d62728"),
        (high_mask, "High A_SN1 regime", "#9467bd"),
    ]:
        df_reg = df[mask]
        if df_reg.empty:
            continue
        grouped = (
            df_reg.groupby("N_calib")["rmse_improvement"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("N_calib")
        )
        ax_a.plot(
            grouped["N_calib"],
            grouped["mean"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        ax_a.fill_between(
            grouped["N_calib"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            color=color,
            alpha=0.15,
        )

    ax_a.axvline(
        n_target,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
        label=f"N = {n_target}",
    )
    ax_a.set_xscale("log")
    unique_ns = np.sort(df["N_calib"].unique())
    ax_a.set_xticks(unique_ns)
    ax_a.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_a.set_xlabel("N calibration galaxies", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("RMSE improvement vs baseline (dex)", fontsize=12, fontweight="bold")
    ax_a.set_title("Panel A: Cost–Benefit of Calibration", fontsize=14, fontweight="bold")
    ax_a.grid(alpha=0.3, which="both")
    ax_a.legend(loc="upper left", fontsize=9)

    # ----------------------------- Panel B --------------------------------- #
    # "Calibration stability" ~ variability of improvement across simulations
    # within each regime as a function of N_calib.
    for regime in regimes_order:
        df_reg = df[df["regime"] == regime]
        if df_reg.empty:
            continue
        grouped = (
            df_reg.groupby("N_calib")["rmse_improvement"]
            .agg(["std", "count"])
            .reset_index()
            .sort_values("N_calib")
        )
        ax_b.plot(
            grouped["N_calib"],
            grouped["std"],
            color=colors.get(regime, "#333333"),
            linewidth=2.0,
            label=regime,
        )

    ax_b.set_xscale("log")
    ax_b.set_xticks(unique_ns)
    ax_b.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_b.set_xlabel("N calibration galaxies", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Std of RMSE improvement (dex)", fontsize=12, fontweight="bold")
    ax_b.set_title(
        "Panel B: Calibration Stability Across Simulations", fontsize=14, fontweight="bold"
    )
    ax_b.grid(alpha=0.3, which="both")
    ax_b.legend(loc="upper right", fontsize=9)

    # ----------------------------- Panel C --------------------------------- #
    # Residual structure before/after: use mean bias per simulation as proxy.
    # Left: baseline (Phase 2, N=0); Right: Phase 3 with N = n_target.
    bias_data_before = []
    bias_labels_before = []
    bias_data_after = []
    bias_labels_after = []

    for regime in regimes_order:
        sims_reg = df_n_target[df_n_target["regime"] == regime]["simulation"].unique()
        if len(sims_reg) == 0:
            continue
        # baseline biases
        b_before = baseline[baseline["simulation"].isin(sims_reg)]["bias_baseline"]
        b_after = df_n_target[df_n_target["regime"] == regime]["bias"]

        if len(b_before) == 0 or len(b_after) == 0:
            continue

        bias_data_before.append(b_before.values)
        bias_labels_before.append(regime)
        bias_data_after.append(b_after.values)
        bias_labels_after.append(regime)

    parts_before = ax_c.violinplot(
        bias_data_before,
        positions=np.arange(1, len(bias_data_before) + 1) - 0.15,
        widths=0.25,
        showmeans=True,
        showextrema=False,
    )
    parts_after = ax_c.violinplot(
        bias_data_after,
        positions=np.arange(1, len(bias_data_after) + 1) + 0.15,
        widths=0.25,
        showmeans=True,
        showextrema=False,
    )

    for pc in parts_before["bodies"]:
        pc.set_facecolor("#1f77b4")
        pc.set_alpha(0.4)
    for pc in parts_after["bodies"]:
        pc.set_facecolor("#2ca02c")
        pc.set_alpha(0.4)

    for partname in ("cmeans",):
        if partname in parts_before:
            parts_before[partname].set_color("#1f77b4")
        if partname in parts_after:
            parts_after[partname].set_color("#2ca02c")

    ax_c.axhline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_c.set_xticks(np.arange(1, len(bias_labels_before) + 1))
    ax_c.set_xticklabels(bias_labels_before, rotation=20, ha="right", fontsize=9)
    ax_c.set_ylabel("Mean residual (M_true − M_pred) [dex]", fontsize=12, fontweight="bold")
    ax_c.set_title(
        f"Panel C: Residual Structure Before (N=0) vs After (N={n_target})",
        fontsize=14,
        fontweight="bold",
    )
    ax_c.grid(alpha=0.3, axis="y")
    ax_c.legend(
        handles=[
            plt.Line2D([], [], color="#1f77b4", linewidth=6, label="Baseline (N=0)"),
            plt.Line2D([], [], color="#2ca02c", linewidth=6, label=f"Calibrated (N={n_target})"),
        ],
        loc="upper right",
        fontsize=9,
    )

    # ----------------------------- Panel D --------------------------------- #
    # Extrapolation distance vs benefit at N = n_target
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(
        vmin=np.nanmin(df_n_target["a_sn1"]), vmax=np.nanmax(df_n_target["a_sn1"])
    )

    improvements_n = df_n_target["rmse_baseline"] - df_n_target["rmse"]

    sc = ax_d.scatter(
        df_n_target["distance_sigma"],
        improvements_n,
        c=df_n_target["a_sn1"],
        cmap=cmap,
        norm=norm,
        s=90,
        edgecolors="black",
        linewidth=1.0,
        alpha=0.9,
    )
    ax_d.axhline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_d.set_xlabel("Distance from training (σ)", fontsize=12, fontweight="bold")
    ax_d.set_ylabel(
        f"RMSE improvement (N=0 → N={n_target}) [dex]",
        fontsize=12,
        fontweight="bold",
    )
    ax_d.set_title(
        "Panel D: Extrapolation Distance vs Calibration Benefit",
        fontsize=14,
        fontweight="bold",
    )
    ax_d.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax_d)
    cbar.set_label("A_SN1", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Figure 9: Calibration Strategy Analysis",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Figure 9 to {output_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 plotting utility – calibration learning curves (Figure 7) "
            "and calibration case studies (Figure 8)."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=(
            "Path to Phase 3 learning‑curve results CSV "
            "(default: outputs/phase3_calibration/phase3_learning_curves.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "figure7_calibration_learning_curves.png",
        help="Output path for Figure 7 PNG.",
    )
    parser.add_argument(
        "--make-figure8",
        action="store_true",
        help=(
            "If set, also generate Figure 8 (Calibration Case Studies). "
            "Requires access to the NF model and SHMR parameters."
        ),
    )
    parser.add_argument(
        "--output-fig8",
        type=Path,
        default=OUTPUT_DIR / "figure8_calibration_case_studies.png",
        help="Output path for Figure 8 PNG.",
    )
    parser.add_argument(
        "--model-run",
        type=str,
        default="delta_sm",
        help="NF run name to use for Figures 7–9 calibration visualizations (default: delta_sm).",
    )
    parser.add_argument(
        "--num-draws-fig8",
        type=int,
        default=500,
        help="Number of posterior draws per halo for Figure 8 (default: 500).",
    )
    parser.add_argument(
        "--rng-seed-fig8",
        type=int,
        default=123,
        help="Base RNG seed for Figure 8 calibration sampling (default: 123).",
    )
    parser.add_argument(
        "--make-figure9",
        action="store_true",
        help="If set, also generate Figure 9 (Calibration Strategy Analysis).",
    )
    parser.add_argument(
        "--output-fig9",
        type=Path,
        default=OUTPUT_DIR / "figure9_calibration_strategy_analysis.png",
        help="Output path for Figure 9 PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("Phase 3: Generate Calibration Learning‑Curve Plots")
    print("=" * 80)
    print(f"Using NF model run: {args.model_run}")

    if not args.results_csv.exists():
        raise FileNotFoundError(f"Phase 3 results CSV not found: {args.results_csv}")

    cosmo_df = load_cosmo_params()
    df_raw = pd.read_csv(args.results_csv)
    df = ensure_phase3_results_columns(df_raw, cosmo_df)

    if len(df) == 0:
        raise ValueError("No valid rows found after cleaning the Phase 3 results CSV.")

    n_sims = df["simulation"].nunique()
    n_ns = df["N_calib"].nunique()
    print(f"Loaded Phase 3 calibration results for {n_sims} simulations and {n_ns} N values.")

    plot_figure7(df, args.output)

    if args.make_figure8:
        print("\nGenerating Figure 8 (Calibration Case Studies)...")
        plot_figure8_case_studies(
            output_path=args.output_fig8,
            model_run=args.model_run,
            num_draws=args.num_draws_fig8,
            n_calib=10,
            rng_seed=args.rng_seed_fig8,
        )

    if args.make_figure9:
        print("\nGenerating Figure 9 (Calibration Strategy Analysis)...")
        plot_figure9_strategy(
            df_phase3=df,
            output_path=args.output_fig9,
            n_target=10,
        )

    print("\nPhase 3 plotting complete.")


if __name__ == "__main__":
    main()


