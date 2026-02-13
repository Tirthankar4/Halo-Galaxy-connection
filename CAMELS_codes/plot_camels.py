#!/usr/bin/env python3
"""
CAMELS plotting utilities.

Subcommands:
1) heatmaps: galaxy-level correlation heatmaps from combined parquet
2) correlations: simulation-level binned correlation plots from CSVs
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from scipy import stats


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)
plt.rcParams["font.size"] = 10


def load_parameter_file(param_file: str) -> pd.DataFrame:
    """
    Load and parse the cosmological and astrophysical parameter file.
    """
    with open(param_file, "r") as f:
        first_line = f.readline().strip()
        if first_line.startswith("#"):
            first_line = first_line[1:]

    df = pd.read_csv(param_file, sep=r"\s+", comment=None, skiprows=0)

    if df.columns[0].startswith("#"):
        df.rename(columns={df.columns[0]: df.columns[0][1:]}, inplace=True)

    df.set_index("Name", inplace=True)

    print(f"Loaded parameters for {len(df)} simulations")
    print(f"Parameters: {list(df.columns)}")

    return df


def load_combined_parquet(parquet_file: str) -> pd.DataFrame:
    """
    Load the combined parquet file and compute derived columns if missing.
    """
    print(f"\nLoading combined parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    print(f"Loaded {len(df)} galaxies from {df['simulation_name'].nunique()} simulations")

    if "pos_modulus" not in df.columns:
        if all(col in df.columns for col in ["pos_x", "pos_y", "pos_z"]):
            df["pos_modulus"] = np.sqrt(
                df["pos_x"] ** 2 + df["pos_y"] ** 2 + df["pos_z"] ** 2
            )
            print("  ✓ Calculated pos_modulus")

    if "vel_modulus" not in df.columns:
        if all(col in df.columns for col in ["vel_x", "vel_y", "vel_z"]):
            df["vel_modulus"] = np.sqrt(
                df["vel_x"] ** 2 + df["vel_y"] ** 2 + df["vel_z"] ** 2
            )
            print("  ✓ Calculated vel_modulus")

    print(f"\nFinal dataset: {len(df)} galaxies from {df['simulation_name'].nunique()} simulations")

    return df


def create_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Create correlation heatmaps for galaxy-level data.
    """
    cosmo_params = ["Omega_m", "sigma_8", "Omega_b"]
    astro_params = ["A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
    all_params = [p for p in cosmo_params + astro_params if p in df.columns]

    property_groups = {
        "mass": ["Mg", "MBH", "Mstar", "Mt"],
        "kinematics": ["Vmax", "sigma_v", "vel_modulus", "J"],
        "chemistry": ["Zg", "Zstar", "SFR"],
        "structure": ["Rstar", "Rt", "Rmax", "pos_modulus"],
    }

    all_properties: List[str] = []
    for props in property_groups.values():
        all_properties.extend(props)

    all_properties = [p for p in all_properties if p in df.columns]

    categories = ["central", "satellite", "all"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating correlation heatmaps...")

    for category in categories:
        if category == "central":
            cat_df = df[df["is_central"] == 1].copy()
        elif category == "satellite":
            cat_df = df[df["is_central"] == 0].copy()
        else:
            cat_df = df.copy()

        if len(cat_df) == 0:
            print(f"  Skipping {category} galaxies (no galaxies found)")
            continue

        print(f"  Creating heatmap for {category} galaxies ({len(cat_df)} galaxies)...")

        log_properties = ["Mg", "MBH", "Mstar", "Mt", "Rstar", "Rt", "Rmax", "SFR"]

        corr_data = pd.DataFrame()
        property_labels = []

        for prop in all_properties:
            if prop not in cat_df.columns:
                continue

            if prop in log_properties:
                prop_data = cat_df[prop].replace([np.inf, -np.inf], np.nan).copy()
                prop_data[prop_data <= 0] = np.nan
                valid_count = prop_data.notna().sum()

                if valid_count >= 10:
                    corr_data[f"log_{prop}"] = np.log10(prop_data)
                    property_labels.append(f"log({prop})")
            else:
                valid_data = cat_df[prop].replace([np.inf, -np.inf], np.nan)
                if valid_data.notna().sum() >= 10:
                    corr_data[prop] = valid_data
                    property_labels.append(prop)

        for param in all_params:
            corr_data[param] = cat_df[param]

        if len(property_labels) == 0:
            print(f"  Skipping {category} galaxies (no valid properties)")
            continue

        n_props = len(property_labels)
        n_params = len(all_params)
        corr_matrix = np.zeros((n_props, n_params))
        pval_matrix = np.zeros((n_props, n_params))

        for i, prop_label in enumerate(property_labels):
            if prop_label.startswith("log("):
                prop_col = "log_" + prop_label[4:-1]
            else:
                prop_col = prop_label

            for j, param in enumerate(all_params):
                valid_mask = corr_data[prop_col].notna() & corr_data[param].notna()

                if valid_mask.sum() >= 10:
                    x = corr_data.loc[valid_mask, param]
                    y = corr_data.loc[valid_mask, prop_col]

                    r, p = stats.pearsonr(x, y)
                    corr_matrix[i, j] = r
                    pval_matrix[i, j] = p
                else:
                    corr_matrix[i, j] = np.nan
                    pval_matrix[i, j] = np.nan

        fig, ax = plt.subplots(figsize=(10, max(8, n_props * 0.4)))

        mask = np.isnan(corr_matrix)

        valid_corrs = corr_matrix[~mask]
        if len(valid_corrs) > 0:
            max_abs_corr = np.max(np.abs(valid_corrs))
            vmax = np.ceil(max_abs_corr * 10) / 10
        else:
            vmax = 0.5
        vmin = -vmax

        sns.heatmap(
            corr_matrix,
            xticklabels=all_params,
            yticklabels=property_labels,
            cmap="RdBu_r",
            center=0,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".4f",
            cbar_kws={"label": "Pearson Correlation Coefficient"},
            mask=mask,
            linewidths=0.5,
            linecolor="gray",
            ax=ax,
        )

        for i in range(n_props):
            for j in range(n_params):
                if not np.isnan(pval_matrix[i, j]):
                    if pval_matrix[i, j] < 0.001:
                        marker = "***"
                    elif pval_matrix[i, j] < 0.01:
                        marker = "**"
                    elif pval_matrix[i, j] < 0.05:
                        marker = "*"
                    else:
                        marker = ""

                    if marker:
                        ax.text(
                            j + 0.5,
                            i + 0.85,
                            marker,
                            ha="center",
                            va="top",
                            fontsize=8,
                            fontweight="bold",
                            color="black",
                        )

        ax.set_title(
            f"{category.capitalize()} Galaxies: Property-Parameter Correlations\n"
            f"(N={len(cat_df)} galaxies, ***p<0.001, **p<0.01, *p<0.05)",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )

        ax.set_xlabel(
            "Cosmological & Astrophysical Parameters", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel(
            "Galaxy Properties (all individual galaxies)", fontsize=11, fontweight="bold"
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        output_file = output_path / f"heatmap_{category}_all_params.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"    ✓ Saved {output_file.name}")

        if len(cosmo_params) > 0 and len(astro_params) > 0:
            cosmo_present = [p for p in cosmo_params if p in all_params]
            astro_present = [p for p in astro_params if p in all_params]

            if len(cosmo_present) > 0 and len(astro_present) > 0:
                fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(16, max(8, n_props * 0.4))
                )

                cosmo_idx = [all_params.index(p) for p in cosmo_present]
                corr_cosmo = corr_matrix[:, cosmo_idx]
                pval_cosmo = pval_matrix[:, cosmo_idx]

                sns.heatmap(
                    corr_cosmo,
                    xticklabels=cosmo_present,
                    yticklabels=property_labels,
                    cmap="RdBu_r",
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    annot=True,
                    fmt=".4f",
                    cbar_kws={"label": "Pearson r"},
                    linewidths=0.5,
                    linecolor="gray",
                    ax=ax1,
                )

                for i in range(n_props):
                    for j in range(len(cosmo_idx)):
                        if not np.isnan(pval_cosmo[i, j]):
                            if pval_cosmo[i, j] < 0.001:
                                marker = "***"
                            elif pval_cosmo[i, j] < 0.01:
                                marker = "**"
                            elif pval_cosmo[i, j] < 0.05:
                                marker = "*"
                            else:
                                marker = ""

                            if marker:
                                ax1.text(
                                    j + 0.5,
                                    i + 0.85,
                                    marker,
                                    ha="center",
                                    va="top",
                                    fontsize=8,
                                    fontweight="bold",
                                    color="black",
                                )

                ax1.set_title("Cosmological Parameters", fontweight="bold", fontsize=12)
                ax1.set_xlabel("", fontsize=10)
                ax1.set_ylabel("Galaxy Properties", fontweight="bold", fontsize=11)
                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

                astro_idx = [all_params.index(p) for p in astro_present]
                corr_astro = corr_matrix[:, astro_idx]
                pval_astro = pval_matrix[:, astro_idx]

                sns.heatmap(
                    corr_astro,
                    xticklabels=astro_present,
                    yticklabels=property_labels,
                    cmap="RdBu_r",
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    annot=True,
                    fmt=".4f",
                    cbar_kws={"label": "Pearson r"},
                    linewidths=0.5,
                    linecolor="gray",
                    ax=ax2,
                )

                for i in range(n_props):
                    for j in range(len(astro_idx)):
                        if not np.isnan(pval_astro[i, j]):
                            if pval_astro[i, j] < 0.001:
                                marker = "***"
                            elif pval_astro[i, j] < 0.01:
                                marker = "**"
                            elif pval_astro[i, j] < 0.05:
                                marker = "*"
                            else:
                                marker = ""

                            if marker:
                                ax2.text(
                                    j + 0.5,
                                    i + 0.85,
                                    marker,
                                    ha="center",
                                    va="top",
                                    fontsize=8,
                                    fontweight="bold",
                                    color="black",
                                )

                ax2.set_title("Astrophysical Parameters", fontweight="bold", fontsize=12)
                ax2.set_xlabel("", fontsize=10)
                ax2.set_ylabel("", fontsize=10)
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
                plt.setp(ax2.get_yticklabels(), visible=False)

                fig.suptitle(
                    f"{category.capitalize()} Galaxies: Correlation Analysis\n"
                    f"(N={len(cat_df)} galaxies)",
                    fontsize=14,
                    fontweight="bold",
                    y=0.98,
                )

                plt.tight_layout()

                output_file_split = output_path / f"heatmap_{category}_split.png"
                plt.savefig(output_file_split, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"    ✓ Saved {output_file_split.name}")


def extract_sim_name(filename: str) -> str:
    """
    Extract simulation name from CSV filename.
    """
    pattern = r"Astrid_(.+?)_groups"
    match = re.search(pattern, filename)

    if match:
        return match.group(1)

    raise ValueError(f"Could not extract simulation name from: {filename}")


def calculate_modulus(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Calculate modulus (magnitude) of 3D vectors.
    """
    x_col = f"{prefix}_x"
    y_col = f"{prefix}_y"
    z_col = f"{prefix}_z"

    if all(col in df.columns for col in [x_col, y_col, z_col]):
        return np.sqrt(df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2)

    return pd.Series([np.nan] * len(df), index=df.index)


def compute_simulation_level_data(input_folder: str, param_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median galaxy properties for each simulation.
    """
    csv_files = list(Path(input_folder).glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {input_folder}")

    print(f"\nProcessing {len(csv_files)} CSV files...")

    property_groups = {
        "mass": ["Mg", "MBH", "Mstar", "Mt"],
        "kinematics": ["Vmax", "sigma_v", "vel_modulus", "J"],
        "chemistry": ["Zg", "Zstar", "SFR"],
        "structure": ["Rstar", "Rt", "Rmax", "pos_modulus"],
    }

    all_properties: List[str] = []
    for props in property_groups.values():
        all_properties.extend(props)

    results = []

    for csv_file in csv_files:
        try:
            sim_name = extract_sim_name(csv_file.name)

            if sim_name not in param_df.index:
                continue

            df = pd.read_csv(csv_file)

            if "pos_modulus" not in df.columns:
                df["pos_modulus"] = calculate_modulus(df, "pos")
            if "vel_modulus" not in df.columns:
                df["vel_modulus"] = calculate_modulus(df, "vel")

            sim_params = param_df.loc[sim_name].to_dict()

            row = {"sim_name": sim_name}
            row.update(sim_params)

            centrals = df[df["is_central"] == 1]
            satellites = df[df["is_central"] == 0]
            all_galaxies = df

            for prop in all_properties:
                if prop not in df.columns:
                    continue

                use_log = prop not in ["Zg", "Zstar", "J", "Vmax", "sigma_v", "vel_modulus"]

                for category_name, category_df in [
                    ("central", centrals),
                    ("satellite", satellites),
                    ("all", all_galaxies),
                ]:
                    if len(category_df) == 0:
                        continue

                    valid_data = category_df[prop].replace([np.inf, -np.inf], np.nan).dropna()
                    valid_data = valid_data[valid_data > 0]

                    if len(valid_data) >= 10:
                        if use_log:
                            median_value = np.median(np.log10(valid_data))
                            col_name = f"log_{prop}_{category_name}"
                        else:
                            median_value = np.median(valid_data)
                            col_name = f"{prop}_{category_name}"

                        row[col_name] = median_value

            results.append(row)
            print(f"✓ Processed {csv_file.name} ({len(df)} galaxies)")

        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")

    df_summary = pd.DataFrame(results)
    print(f"\nComputed simulation-level data for {len(df_summary)} simulations")

    return df_summary


def create_plots(df_summary: pd.DataFrame, output_dir: str):
    """
    Create correlation plots using simulation-level statistics.
    """
    property_groups: Dict[str, Dict[str, List[str]]] = {
        "mass": {
            "properties": ["Mg", "MBH", "Mstar", "Mt"],
            "title": "Mass Properties",
        },
        "kinematics": {
            "properties": ["Vmax", "sigma_v", "vel_modulus", "J"],
            "title": "Kinematic Properties",
        },
        "chemistry": {
            "properties": ["Zg", "Zstar", "SFR"],
            "title": "Chemical Properties & Star Formation",
        },
        "structure": {
            "properties": ["Rstar", "Rt", "Rmax", "pos_modulus"],
            "title": "Structural Properties & Position",
        },
    }

    def uses_log_scale(prop):
        return prop not in ["Zg", "Zstar", "J", "Vmax", "sigma_v", "vel_modulus"]

    cosmo_astro_params = [
        "Omega_m",
        "sigma_8",
        "Omega_b",
        "A_SN1",
        "A_AGN1",
        "A_SN2",
        "A_AGN2",
    ]

    cosmo_astro_params = [p for p in cosmo_astro_params if p in df_summary.columns]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating simulation-level binned correlation plots...")
    print(f"Using {len(df_summary)} simulations")

    categories = ["central", "satellite", "all"]

    for gal_category in categories:
        print(f"\nProcessing {gal_category} galaxies...")

        for param in cosmo_astro_params:
            for group_name, group_info in property_groups.items():
                properties = group_info["properties"]
                group_title = group_info["title"]

                n_props = len(properties)
                n_cols = min(2, n_props)
                n_rows = (n_props + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)

                fig.suptitle(
                    f"{gal_category.capitalize()} Galaxies - {group_title}\n"
                    f"vs {param}\n"
                    f"(Simulation-Level Binning: {len(df_summary)} simulations)",
                    fontsize=14,
                    fontweight="bold",
                )

                for idx, prop in enumerate(properties):
                    row = idx // n_cols
                    col = idx % n_cols
                    ax = axes[row, col]

                    use_log = uses_log_scale(prop)
                    if use_log:
                        prop_col = f"log_{prop}_{gal_category}"
                    else:
                        prop_col = f"{prop}_{gal_category}"

                    if prop_col not in df_summary.columns:
                        ax.text(
                            0.5,
                            0.5,
                            f"{prop}\nNot Available",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    valid_mask = df_summary[prop_col].notna() & df_summary[param].notna()

                    if valid_mask.sum() < 5:
                        ax.text(
                            0.5,
                            0.5,
                            f"Insufficient Data\n(N={valid_mask.sum()} sims)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=12,
                        )
                        ax.set_xlabel(param, fontsize=11)
                        ylabel_text = f"log$_{{10}}$({prop})" if use_log else prop
                        ax.set_ylabel(ylabel_text, fontsize=11)
                        continue

                    x_data = df_summary.loc[valid_mask, param].values
                    y_data = df_summary.loc[valid_mask, prop_col].values

                    nbins = min(10, valid_mask.sum() // 3)
                    nbins = max(5, nbins)

                    bin_median, bin_edges, _ = stats.binned_statistic(
                        x_data, y_data, statistic="median", bins=nbins
                    )
                    bin_16, _, _ = stats.binned_statistic(
                        x_data,
                        y_data,
                        statistic=lambda x: np.percentile(x, 16),
                        bins=nbins,
                    )
                    bin_84, _, _ = stats.binned_statistic(
                        x_data,
                        y_data,
                        statistic=lambda x: np.percentile(x, 84),
                        bins=nbins,
                    )
                    bin_count, _, _ = stats.binned_statistic(
                        x_data, y_data, statistic="count", bins=nbins
                    )

                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    valid_bins = (~np.isnan(bin_median)) & (bin_count >= 3)

                    if valid_bins.sum() < 2:
                        ax.text(
                            0.5,
                            0.5,
                            f"Insufficient bins\n(N={valid_bins.sum()})",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=12,
                        )
                        ax.set_xlabel(param, fontsize=11)
                        ylabel_text = f"log$_{{10}}$({prop})" if use_log else prop
                        ax.set_ylabel(ylabel_text, fontsize=11)
                        continue

                    ax.fill_between(
                        bin_centers[valid_bins],
                        bin_16[valid_bins],
                        bin_84[valid_bins],
                        alpha=0.3,
                        color="lightblue",
                        label="16-84th percentile\n(sim medians)",
                        zorder=8,
                    )

                    norm = Normalize(
                        vmin=bin_count[valid_bins].min(),
                        vmax=bin_count[valid_bins].max(),
                    )
                    cmap = cm.viridis

                    valid_centers = bin_centers[valid_bins]
                    valid_medians = bin_median[valid_bins]
                    valid_counts = bin_count[valid_bins]

                    for i in range(len(valid_centers) - 1):
                        ax.plot(
                            valid_centers[i : i + 2],
                            valid_medians[i : i + 2],
                            color=cmap(norm(valid_counts[i])),
                            linewidth=3.0,
                            zorder=10,
                        )

                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, label="Simulations per Bin")
                    cbar.ax.tick_params(labelsize=9)

                    r, p = stats.pearsonr(x_data, y_data)
                    ax.text(
                        0.05,
                        0.95,
                        f"PCC = {r:+.3f}\np = {p:.1e}\nN = {valid_mask.sum()} sims",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                    ax.set_xlabel(param, fontsize=11)
                    ylabel_text = f"log$_{{10}}$({prop})" if use_log else prop
                    ax.set_ylabel(ylabel_text, fontsize=11)
                    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
                    ax.grid(True, alpha=0.3, linestyle="--")

                for idx in range(n_props, n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].set_visible(False)

                plt.tight_layout(rect=[0, 0, 1, 0.95])

                gal_output_path = output_path / f"{gal_category}_correlations"
                gal_output_path.mkdir(parents=True, exist_ok=True)

                output_file = gal_output_path / f"{param}_{group_name}.png"
                plt.savefig(output_file, dpi=150, bbox_inches="tight")
                plt.close()

                print(f"  ✓ Saved {gal_category}_correlations/{param}_{group_name}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot CAMELS correlations and heatmaps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    heatmap_parser = subparsers.add_parser(
        "heatmaps", help="Generate galaxy-level correlation heatmaps from parquet"
    )
    heatmap_parser.add_argument(
        "--combined-file",
        type=str,
        required=True,
        help="Path to combined parquet file containing all galaxy data",
    )
    heatmap_parser.add_argument(
        "--param-file",
        type=str,
        default="CAMELS_datas/CosmoAstroSeed_Astrid_L25n256_SB7.txt",
        help="Path to parameter file (default: CAMELS_datas/CosmoAstroSeed_Astrid_L25n256_SB7.txt)",
    )
    heatmap_parser.add_argument(
        "--output-dir",
        type=str,
        default="CAMELS_heatmaps",
        help="Output directory for heatmaps (default: CAMELS_heatmaps)",
    )

    corr_parser = subparsers.add_parser(
        "correlations",
        help="Generate simulation-level binned correlation plots from CSVs",
    )
    corr_parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing CSV files",
    )
    corr_parser.add_argument(
        "--param-file",
        type=str,
        default="CAMELS_datas/CosmoAstroSeed_Astrid_L25n256_SB7.txt",
        help="Path to parameter file (default: CAMELS_datas/CosmoAstroSeed_Astrid_L25n256_SB7.txt)",
    )
    corr_parser.add_argument(
        "--output-dir",
        type=str,
        default="CAMELS_plots",
        help="Output directory for plots (default: CAMELS_plots)",
    )

    args = parser.parse_args()

    if args.command == "heatmaps":
        print("=" * 70)
        print("CAMELS Correlation Heatmap Generator")
        print("=" * 70)

        print(f"\nLoading parameter file: {args.param_file}")
        _ = load_parameter_file(args.param_file)

        print(f"\nLoading combined galaxy data from: {args.combined_file}")
        df = load_combined_parquet(args.combined_file)

        if len(df) == 0:
            print("\nNo data to process. Exiting.")
            return

        print(f"\nCreating correlation heatmaps...")
        create_correlation_heatmap(df, args.output_dir)

        print("\n" + "=" * 70)
        print(f"✓ Complete! Heatmaps saved to {args.output_dir}/")
        print(f"  Analyzed {len(df)} galaxies from {df['simulation_name'].nunique()} simulations")
        print("=" * 70)
        return

    if args.command == "correlations":
        print("=" * 70)
        print("CAMELS Correlation Plotting Script")
        print("=" * 70)

        print(f"\nLoading parameter file: {args.param_file}")
        param_df = load_parameter_file(args.param_file)

        print(f"\nProcessing CSV files from: {args.input_folder}")
        df_summary = compute_simulation_level_data(args.input_folder, param_df)

        if len(df_summary) == 0:
            print("\nNo data to process. Exiting.")
            return

        print(f"\nCreating plots in: {args.output_dir}")
        create_plots(df_summary, args.output_dir)

        print("\n" + "=" * 70)
        print(f"✓ Complete! Generated simulation-level binned plots for {len(df_summary)} simulations")
        print(f"  Output directory: {args.output_dir}")
        print("=" * 70)
        return


if __name__ == "__main__":
    main()
