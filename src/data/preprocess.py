"""

CLI to ingest the TNG halo/galaxy catalogs, merge them, and persist a

processed table with log-scaled features ready for modeling.

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils.io import ensure_dir, save_json


C_LIGHT = 3.0 * 10**8  # Speed of light in m/s; reused from notebook constant


def load_catalogs(h5_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading halo/galaxy catalogs from %s", h5_path)
    with h5py.File(h5_path, "r") as f:
        m_h = f["Group/Group_M_Crit200"][:] * 1e10  # Msun/h
        r_h = f["Group/Group_R_Crit200"][:] / C_LIGHT  # kpc/h
        v_h = f["Group/GroupVel"][:]
        v_h = np.linalg.norm(v_h, axis=1)
        id_r = f["Group/GroupFirstSub"][:]
        id_h = np.arange(0, m_h.shape[0], 1, dtype=float)

        sm = f["Subhalo/SubhaloMassType"][:, 4] * 1e10
        sfr = f["Subhalo/SubhaloSFR"][:] * 1e10
        sr = f["Subhalo/SubhaloHalfmassRadType"][:, 4]
        colour = (
            f["Subhalo/SubhaloStellarPhotometrics"][:, 4]
            - f["Subhalo/SubhaloStellarPhotometrics"][:, 5]
        )
        id_g = np.array(f["Subhalo/SubhaloGrNr"])

    # Filter halos that host galaxies (GroupFirstSub != -1)
    keep = np.where(id_r != -1)[0]
    data = np.array([m_h[keep], r_h[keep], v_h[keep], id_h[keep]]).T
    halos = pd.DataFrame(data, columns=["M_h", "R_h", "V_h", "ID"])

    # Filter galaxies with positive stellar mass
    gal_keep = np.where(sm > 0)[0]
    gal_data = np.array(
        [sm[gal_keep], sfr[gal_keep], colour[gal_keep], sr[gal_keep], id_g[gal_keep]]
    ).T
    gals = pd.DataFrame(gal_data, columns=["SM", "SFR", "Colour", "SR", "ID"])
    gals = gals.drop_duplicates(subset=["ID"], keep="first")

    return halos, gals


def preprocess_dataframe(halos: pd.DataFrame, gals: pd.DataFrame) -> pd.DataFrame:
    logging.info("Merging %d halos with %d galaxies", len(halos), len(gals))
    df = pd.merge(halos, gals, on="ID")

    # Log transforms (matching the notebook)
    for col in ["M_h", "R_h", "V_h", "SM"]:
        df[col] = np.log10(df[col])

    # Handle SFR zeros before log scaling
    df["SFR"] = df["SFR"].replace(0, 1)
    df["SFR"] = np.log10(df["SFR"])
    mask_zero = df["SFR"] == 0
    if mask_zero.any():
        df.loc[mask_zero, "SFR"] = np.random.normal(8.0, 0.5, mask_zero.sum())

    logging.info("Final dataframe shape: %s", df.shape)
    return df


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary = {}
    describe_df = df.describe(include="all")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col] = {
                "mean": float(describe_df.loc["mean", col]),
                "std": float(describe_df.loc["std", col]),
                "min": float(describe_df.loc["min", col]),
                "max": float(describe_df.loc["max", col]),
            }
    return summary


def plot_halo_properties(df: pd.DataFrame, output_path: Path) -> None:
    """Plot histograms of halo properties (M_h, R_h, V_h)."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=100)
    fig.suptitle("Halo properties", fontsize=14, fontweight="bold")

    properties = ["M_h", "R_h", "V_h"]
    labels = ["M_h", "R_h", "V_h"]

    for idx, (prop, label) in enumerate(zip(properties, labels)):
        axs[idx].hist(df[prop], bins=40, edgecolor="black", alpha=0.7)
        axs[idx].set_yscale("log")
        axs[idx].set_xlabel(label, fontsize=11)
        if idx == 0:
            axs[idx].set_ylabel("# halos", fontsize=11)
        axs[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("Saved halo properties plot to %s", output_path)


def plot_galaxy_properties(df: pd.DataFrame, output_path: Path) -> None:
    """Plot histograms of galaxy properties (SM, SFR, Colour, SR)."""
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True, dpi=100)
    fig.suptitle("Galaxy properties", fontsize=14, fontweight="bold")

    properties = ["SM", "SFR", "Colour", "SR"]
    labels = ["SM", "SFR", "Colour", "SR"]

    for idx, (prop, label) in enumerate(zip(properties, labels)):
        axs[idx].hist(df[prop], bins=40, edgecolor="black", alpha=0.7)
        axs[idx].set_yscale("log")
        axs[idx].set_xlabel(label, fontsize=11)
        if idx == 0:
            axs[idx].set_ylabel("# halos", fontsize=11)
        axs[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("Saved galaxy properties plot to %s", output_path)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot Pearson correlation heatmap for all numeric columns."""
    correlations = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlations,
        vmax=1.0,
        center=0,
        fmt=".4f",
        square=True,
        linewidths=0.5,
        annot=True,
        cbar_kws={"shrink": 0.82},
        cmap="coolwarm",
        ax=ax,
    )
    plt.title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("Saved correlation heatmap to %s", output_path)


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all preprocessing diagnostic plots."""
    ensure_dir(output_dir)
    plot_halo_properties(df, output_dir / "halo_properties.png")
    plot_galaxy_properties(df, output_dir / "galaxy_properties.png")
    plot_correlation_heatmap(df, output_dir / "correlation_heatmap.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess halo/galaxy catalog.")
    parser.add_argument(
        "--halo-hdf5",
        type=Path,
        default=config.DEFAULT_HALO_HDF5,
        help="Path to groups_XXX.hdf5 file with halo + subhalo data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEFAULT_PROCESSED_PARQUET,
        help="Path to save processed parquet dataset.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=config.PROCESSED_DATA_DIR / "halo_galaxy_stats.json",
        help="Where to store summary statistics for downstream scripts.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV copy of the processed table.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_RANDOM_STATE,
        help="Random seed for stochastic steps (e.g., SFR noise).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    np.random.seed(args.seed)

    halos, gals = load_catalogs(args.halo_hdf5)
    df = preprocess_dataframe(halos, gals)

    ensure_dir(args.output.parent)
    df.to_parquet(args.output, index=False)
    logging.info("Saved processed parquet to %s", args.output)

    if args.csv:
        ensure_dir(args.csv.parent)
        df.to_csv(args.csv, index=False)
        logging.info("Saved CSV copy to %s", args.csv)

    stats = compute_summary_stats(df)
    save_json(stats, args.stats_json)
    logging.info("Wrote summary stats to %s", args.stats_json)

    # Generate diagnostic plots in the same directory as the output parquet
    if not args.no_plots:
        generate_plots(df, args.output.parent)
        logging.info("Generated diagnostic plots in %s", args.output.parent)


if __name__ == "__main__":
    main()

