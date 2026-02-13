#!/usr/bin/env python3
"""
Distribution plots for CAMELS ASTRID dataset.

Subcommands:
1) transforms  : Histograms for shifted-log transformed galaxy properties
2) splits      : Histograms of cosmological/astrophysical parameters across train/test splits

Both subcommands accept --n-galaxies to randomly sample a fixed number of galaxies
per simulation before plotting (useful for quick inspection).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from transformation import prepare_features_and_params, load_data, transform_features

# ── Styling (matches plot_camels.py) ──────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10

# ── Constants (mirrors NF_train.py) ──────────────────────────────────────────
GALAXY_PROPERTIES = [
    "Mg", "MBH", "Mstar", "Mt", "Vmax", "sigma_v",
    "Zg", "Zstar", "SFR", "J", "Rstar", "Rt", "Rmax",
]
PARAM_COLUMNS = [
    "Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2", "Omega_b",
]
SHIFTED_LOG_PROPERTIES = ["SFR", "Mg", "MBH", "Zg"]
TEST_SIZE = 0.2
SEED = 42

# Pretty LaTeX-style labels for parameters
PARAM_LABELS = {
    "Omega_m": r"$\Omega_m$",
    "sigma_8": r"$\sigma_8$",
    "A_SN1":   r"$A_\mathrm{SN1}$",
    "A_AGN1":  r"$A_\mathrm{AGN1}$",
    "A_SN2":   r"$A_\mathrm{SN2}$",
    "A_AGN2":  r"$A_\mathrm{AGN2}$",
    "Omega_b": r"$\Omega_b$",
}


# ══════════════════════════════════════════════════════════════════════════════
# Subsampling helper
# ══════════════════════════════════════════════════════════════════════════════

def subsample_per_simulation(df, n_galaxies):
    """
    Randomly sample up to n_galaxies from each simulation.

    Simulations with fewer than n_galaxies galaxies are kept in full.
    Sampling uses a fixed seed for reproducibility.

    Args:
        df:          DataFrame with a 'simulation_id' column (central galaxies only)
        n_galaxies:  Maximum number of galaxies to draw from each simulation

    Returns:
        Subsampled DataFrame with reset index.
    """
    rng = np.random.default_rng(SEED)

    kept = []
    n_sims = df["simulation_id"].nunique()

    for sim_id, group in df.groupby("simulation_id"):
        if len(group) <= n_galaxies:
            kept.append(group)
        else:
            idx = rng.choice(group.index, size=n_galaxies, replace=False)
            kept.append(group.loc[idx])

    result = pd.concat(kept).reset_index(drop=True)

    n_before = len(df)
    n_after  = len(result)
    print(
        f"  Subsampled: {n_galaxies} galaxies/sim x {n_sims} sims "
        f"-> {n_after:,} galaxies  (was {n_before:,})"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Task 1 - Before / After transformation histograms
# ══════════════════════════════════════════════════════════════════════════════

def plot_transforms(data_path, output_dir, n_galaxies):
    """
    For each shifted-log property, create a separate figure with side-by-side
    histograms: raw values and shifted-log transformed values.
    One PNG per property.
    """
    print("Loading data for transform distributions...")
    df_raw = load_data(data_path)   # central galaxies only

    if n_galaxies is not None:
        print(f"\nSubsampling to {n_galaxies} galaxies per simulation...")
        df_raw = subsample_per_simulation(df_raw, n_galaxies)

    feature_cols = SHIFTED_LOG_PROPERTIES
    df_transformed, transformed_cols = transform_features(df_raw.copy(), feature_cols)

    raw_to_transformed = dict(zip(feature_cols, transformed_cols))

    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(df_raw)
    print(f"\nPlotting {len(feature_cols)} properties as separate figures...")

    for col in feature_cols:
        transformed_col = raw_to_transformed[col]
        raw_vals   = df_raw[col].values
        trans_vals = df_transformed[transformed_col].values

        n_zeros   = int((raw_vals == 0).sum())
        n_nonzero = n_total - n_zeros

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

        subsample_note = f", {n_galaxies} gal/sim" if n_galaxies is not None else ""
        fig.suptitle(
            f"{col} - Shifted-Log Transform\n"
            f"(Central Galaxies, N = {n_total:,}{subsample_note})",
            fontsize=13, fontweight="bold",
        )

        # Before (left)
        ax_before = axes[0]
        ax_before.hist(raw_vals, bins=80, color="#4C72B0", edgecolor="none", alpha=0.85)
        ax_before.set_yscale("log")
        ax_before.set_xlabel(f"{col}  (raw)", fontsize=10)
        ax_before.set_ylabel("Count (log scale)", fontsize=10)
        ax_before.set_title(f"{col} - Before (raw values)", fontsize=11)
        ax_before.text(
            0.97, 0.95,
            f"non-zero: {n_nonzero:,} ({100*n_nonzero/n_total:.1f}%)\n"
            f"zeros: {n_zeros:,} ({100*n_zeros/n_total:.1f}%)",
            transform=ax_before.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        # After (right)
        ax_after = axes[1]
        ax_after.hist(trans_vals, bins=80, color="#DD8452", edgecolor="none", alpha=0.85)
        ax_after.set_yscale("log")
        ax_after.set_xlabel(f"{transformed_col}", fontsize=10)
        ax_after.set_ylabel("Count (log scale)", fontsize=10)
        ax_after.set_title(f"{transformed_col} - After (shifted log)", fontsize=11)
        ax_after.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

        if n_zeros > 0:
            zero_mapped = trans_vals.min()
            ax_after.axvline(
                zero_mapped, color="red", linewidth=1.2,
                linestyle="--", label=f"zero-mapped = {zero_mapped:.2f}",
            )
            ax_after.legend(fontsize=8, loc="upper right")

        out_file = output_dir / f"transform_{col}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_file.name}")

    print(f"\nDone. {len(feature_cols)} figure(s) saved to {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Task 2 - Parameter distributions across train / test splits
# ══════════════════════════════════════════════════════════════════════════════

def plot_splits(data_path, output_dir, n_galaxies):
    """
    For each cosmological/astrophysical parameter, plot overlapping step
    histograms for the total dataset, train split, and test split.
    One PNG per parameter.
    """
    print("Loading data for split distributions...")

    # Load raw DataFrame so we can subsample before the transform pipeline
    df_raw = load_data(data_path)

    if n_galaxies is not None:
        print(f"\nSubsampling to {n_galaxies} galaxies per simulation...")
        df_raw = subsample_per_simulation(df_raw, n_galaxies)

    # Re-run transform pipeline on the (possibly subsampled) df
    feature_cols = GALAXY_PROPERTIES + ["V"]
    df_transformed, transformed_feature_cols = transform_features(df_raw.copy(), feature_cols)

    features = df_transformed[transformed_feature_cols].values
    params   = df_raw[PARAM_COLUMNS].values

    # Replicate exact split logic from NF_train.py
    _, _, params_train, params_test = train_test_split(
        features, params, test_size=TEST_SIZE, random_state=SEED
    )

    n_total = len(params)
    n_train = len(params_train)
    n_test  = len(params_test)

    print(f"  Total galaxies : {n_total:,}")
    print(f"  Train          : {n_train:,}  ({100*n_train/n_total:.1f}%)")
    print(f"  Test           : {n_test:,}   ({100*n_test/n_total:.1f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)

    colors     = {"Total": "#333333", "Train": "#4C72B0", "Test": "#DD8452"}
    linestyles = {"Total": "-",       "Train": "--",      "Test": ":"}

    subsample_note = f", {n_galaxies} gal/sim" if n_galaxies is not None else ""
    print(f"\nPlotting {len(PARAM_COLUMNS)} parameters as separate figures...")

    for idx, param in enumerate(PARAM_COLUMNS):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        param_label = PARAM_LABELS.get(param, param)
        fig.suptitle(
            f"{param_label} - Train & Test Splits\n"
            f"(Central Galaxies, N = {n_total:,}{subsample_note})",
            fontsize=13, fontweight="bold",
        )

        all_vals   = params[:, idx]
        train_vals = params_train[:, idx]
        test_vals  = params_test[:, idx]

        bins = np.linspace(all_vals.min(), all_vals.max(), 50)

        for label, vals in [("Total", all_vals), ("Train", train_vals), ("Test", test_vals)]:
            ax.hist(
                vals, bins=bins, histtype="step", linewidth=1.8,
                color=colors[label], linestyle=linestyles[label],
                label=f"{label} (N = {len(vals):,})",
            )

        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(True, alpha=0.3, linestyle="--")

        out_file = output_dir / f"split_{param}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_file.name}")

    print(f"\nDone. {len(PARAM_COLUMNS)} figure(s) saved to {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def add_common_args(parser):
    """Add arguments shared by both subcommands."""
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to the combined parquet file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--n-galaxies", type=int, default=None, metavar="N",
        help=(
            "If set, randomly sample N galaxies from each simulation before "
            "plotting. Simulations with fewer than N galaxies are kept in full. "
            "Omit to use all galaxies (default)."
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Distribution plots for CAMELS ASTRID dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    t_parser = subparsers.add_parser(
        "transforms",
        help="Plot shifted-log property distributions before and after transform",
    )
    add_common_args(t_parser)

    s_parser = subparsers.add_parser(
        "splits",
        help="Plot parameter distributions across train/test splits",
    )
    add_common_args(s_parser)

    args = parser.parse_args()

    if args.command == "transforms":
        print("=" * 60)
        print("Transform Distribution Plots")
        print("=" * 60)
        plot_transforms(Path(args.data_path), Path(args.output_dir), args.n_galaxies)

    elif args.command == "splits":
        print("=" * 60)
        print("Train/Test Split Distribution Plots")
        print("=" * 60)
        plot_splits(Path(args.data_path), Path(args.output_dir), args.n_galaxies)


if __name__ == "__main__":
    main()
