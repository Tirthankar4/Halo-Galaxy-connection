"""Common plotting utilities for data visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


def plot_histograms(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """
    Plot histograms for multiple columns in a single figure.
    
    Args:
        df: DataFrame containing the data
        columns: Column names to plot
        labels: Labels for x-axis
        title: Figure title
        output_path: Where to save the figure
        log_scale: Whether to use log scale for y-axis
    """
    n_cols = len(columns)
    fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True, dpi=100)
    if n_cols == 1:
        axs = [axs]
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    for idx, (col, label) in enumerate(zip(columns, labels)):
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) > 0:
            axs[idx].hist(data, bins=40, edgecolor="black", alpha=0.7)
        if log_scale:
            axs[idx].set_yscale("log")
        axs[idx].set_xlabel(label, fontsize=11)
        if idx == 0:
            axs[idx].set_ylabel("Count", fontsize=11)
        axs[idx].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot Pearson correlation heatmap for numeric columns."""
    # Select only numeric columns for correlation computation
    numeric_df = df.select_dtypes(include=['number'])
    correlations = numeric_df.corr(method="pearson")
    
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
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

