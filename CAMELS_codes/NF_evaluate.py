"""
Evaluate trained normalizing flow model on test data.
Loads a trained model and computes test set NLL.

Training configuration (including n_galaxies subsampling) is read automatically
from train_config.json saved alongside the model by NF_train.py.
"""

import json
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tarp import get_tarp_coverage
from transformation import (
    ASTROPHYSICAL_LOG_PARAMS,
    prepare_features_and_params,
    load_data,
    transform_features,
    transform_params,
    inverse_transform_params,
)

# ============ Configuration ============
DATA_PATH = Path(__file__).parent.parent / "CAMELS_datas" / "camels_astrid_sb7_090.parquet"
MODEL_DIR = Path(__file__).parent.parent / "CAMELS_outputs" / "model_data"
PLOTS_DIR = Path(__file__).parent.parent / "CAMELS_outputs" / "plots"

# Galaxy properties (features/conditions)
GALAXY_PROPERTIES = [
    'Mg', 'MBH', 'Mstar', 'Mt', 'Vmax', 'sigma_v', 'Zg', 'Zstar',
    'SFR', 'J', 'Rstar', 'Rt', 'Rmax'
]

# Parameters to learn (targets)
PARAM_COLUMNS = ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2', 'Omega_b']
TARGET_LOG_PARAMS = ASTROPHYSICAL_LOG_PARAMS

# Evaluation settings
TEST_SIZE = 0.2
SEED = 42

# Plot generation settings
N_POSTERIOR_SAMPLES = 1000
MAX_TEST_SAMPLES    = 1000
PROGRESS_INTERVAL   = 1000


# ── Training config helpers ───────────────────────────────────────────────────

def load_train_config(model_dir: Path) -> dict:
    """
    Load train_config.json from model_dir.
    Raises a clear error if the file is missing (e.g. model trained with old script).
    """
    config_path = model_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"train_config.json not found in {model_dir}.\n"
            "This file is written by NF_train.py at the end of training.\n"
            "If your model was trained with an older version of NF_train.py, "
            "please retrain with the updated script."
        )
    with open(config_path) as f:
        config = json.load(f)

    n_galaxies   = config.get("n_galaxies")   # None = full dataset
    n_transforms = config.get("n_transforms", 2)
    seed         = config.get("seed", SEED)
    test_size    = config.get("test_size", TEST_SIZE)
    target_log_params = config.get("target_log_params", [])

    print("=" * 60)
    print("Training configuration (from train_config.json)")
    print("=" * 60)
    print(f"  n_galaxies   : {n_galaxies if n_galaxies is not None else 'all (no subsampling)'}")
    print(f"  n_transforms : {n_transforms}")
    print(f"  test_size    : {test_size}")
    print(f"  seed         : {seed}")
    print(f"  epochs       : {config.get('epochs')}")
    print(f"  batch_size   : {config.get('batch_size')}")
    print(f"  lr           : {config.get('lr')}")
    print(f"  target_log_params: {target_log_params}")
    print("=" * 60)

    return config


def subsample_per_simulation(df, n_galaxies, seed):
    """Randomly sample up to n_galaxies from each simulation (same logic as NF_train.py)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    kept = []
    for sim_id, group in df.groupby("simulation_id"):
        if len(group) <= n_galaxies:
            kept.append(group)
        else:
            idx = rng.choice(group.index, size=n_galaxies, replace=False)
            kept.append(group.loc[idx])
    result = pd.concat(kept).reset_index(drop=True)
    print(
        f"  Subsampled: {n_galaxies} galaxies/sim -> "
        f"{len(result):,} galaxies  (was {len(df):,})"
    )
    return result


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_flow(cond_dim: int, param_dim: int, n_transforms: int = 4):
    base = dist.Normal(torch.zeros(param_dim), torch.ones(param_dim)).to_event(1)
    transforms = [T.conditional_spline(param_dim, context_dim=cond_dim)
                  for _ in range(n_transforms)]
    cond_dist = dist.ConditionalTransformedDistribution(base, transforms)
    modules = torch.nn.ModuleList(transforms)
    return cond_dist, modules


def load_model(model_path, scaler_path, cond_dim, param_dim, n_transforms, device):
    """Load trained model and scalers from disk."""
    cond_dist, modules = build_flow(cond_dim, param_dim, n_transforms)

    modules.load_state_dict(torch.load(model_path, map_location=device))
    modules = modules.to(device)

    base_dist = cond_dist.base_dist
    while hasattr(base_dist, 'base_dist'):
        base_dist = base_dist.base_dist
    base_dist.loc   = base_dist.loc.to(device)
    base_dist.scale = base_dist.scale.to(device)

    scaler_data = np.load(scaler_path)
    feat_scaler = StandardScaler()
    feat_scaler.mean_  = scaler_data['feat_mean']
    feat_scaler.scale_ = scaler_data['feat_scale']

    param_scaler = StandardScaler()
    param_scaler.mean_  = scaler_data['param_mean']
    param_scaler.scale_ = scaler_data['param_scale']

    return cond_dist, modules, feat_scaler, param_scaler


# ── Prediction / evaluation ───────────────────────────────────────────────────

def compute_predictions(
    cond_dist,
    features_test,
    device,
    param_scaler,
    target_log_params,
    param_names,
    n_samples=1000,
    max_test_samples=None,
):
    """
    Draw posterior samples for each test galaxy and return means and stds
    in physical units.

    Samples are drawn in the model's scaled space, then transformed to physical
    units per-sample (StandardScaler inverse → log10 inverse for astrophysical
    params) before computing mean and std. This is exact and avoids the delta
    method approximation that would be needed if converting stats after the fact.
    """
    if max_test_samples is not None and len(features_test) > max_test_samples:
        indices = np.random.choice(len(features_test), max_test_samples, replace=False)
        features_test = features_test[indices]
    else:
        indices = None

    all_means, all_stds = [], []
    n_test = len(features_test)

    with torch.no_grad():
        for batch_start in range(0, n_test, PROGRESS_INTERVAL):
            batch_end = min(batch_start + PROGRESS_INTERVAL, n_test)
            if batch_start % 1000 == 0:
                print(f"  Processing samples {batch_start}/{n_test}...")
            for i in range(batch_start, batch_end):
                single_feature   = features_test[i:i+1].to(device)
                conditioned_dist = cond_dist.condition(single_feature)

                # samples shape: (n_samples, n_params) — in StandardScaler space
                samples = conditioned_dist.sample((n_samples,)).squeeze(1).cpu().numpy()

                # Step 1: invert StandardScaler  →  log-transformed param space
                samples_log = param_scaler.inverse_transform(samples)

                # Step 2: invert log10 for astrophysical params  →  physical space
                # This is exact per-sample, no approximation needed
                samples_physical = inverse_transform_params(
                    samples_log, param_names, target_log_params
                )

                all_means.append(samples_physical.mean(axis=0))
                all_stds.append(samples_physical.std(axis=0))

    return np.array(all_means), np.array(all_stds), indices


def plot_pred_vs_true(pred_means, pred_stds, true_values, param_names, output_dir):
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, r2_score

    for i, param_name in enumerate(param_names):
        fig = plt.figure(figsize=(8, 6))
        ax  = plt.subplot2grid((1, 10), (0, 0), colspan=7)

        true_vals = true_values[:, i]
        pred_vals = pred_means[:, i]
        pred_err  = pred_stds[:, i]

        ax.errorbar(true_vals, pred_vals, yerr=pred_err,
                    fmt='o', alpha=0.4, markersize=4,
                    elinewidth=0.8, capsize=0, color='steelblue')

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.6, linewidth=2, label='1:1 line')

        ax.set_xlabel(f'True {param_name}',      fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted {param_name}', fontsize=12, fontweight='bold')
        ax.set_title(f'{param_name} - Predicted vs True', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        pcc, _     = pearsonr(true_vals, pred_vals)
        r2         = r2_score(true_vals, pred_vals)
        rmse       = np.sqrt(mean_squared_error(true_vals, pred_vals))
        mask       = np.abs(true_vals) > 1e-10
        rel_error  = np.mean(np.abs(pred_vals[mask] - true_vals[mask]) / np.abs(true_vals[mask])) * 100
        residuals  = pred_vals - true_vals
        variances  = pred_err ** 2
        valid_mask = variances > 1e-10
        chi2       = np.mean((residuals[valid_mask] ** 2) / variances[valid_mask])

        stats_text = (
            f"Metrics\n"
            f"{'─' * 20}\n"
            f"PCC (r):  {pcc:>7.4f}\n"
            f"R²:       {r2:>7.4f}\n"
            f"RMSE:     {rmse:>7.4f}\n"
            f"Rel Err(%): {rel_error:>5.2f}\n"
            f"{'─' * 20}\n"
            f"χ²:       {chi2:>7.4f}\n"
            f"(χ²≈1 = calibrated)"
        )

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.15, edgecolor='gray', linewidth=1.5)
        fig.text(0.75, 0.5, stats_text, fontsize=11, verticalalignment='center',
                 bbox=props, family='monospace')

        plt.tight_layout()
        output_path = output_dir / f"pred_vs_true_{param_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")

    print(f"\nAll plots saved to: {output_dir}")


def compute_chi2_metrics(pred_means, pred_stds, true_values, param_names, verbose=False):
    chi2_values = {}
    if verbose:
        print(f"\n{'='*60}")
        print(f"Chi-Squared Analysis (per parameter)")
        print(f"{'='*60}")
        print(f"Well-calibrated uncertainties should have χ² ≈ 1.0")
        print(f"χ² < 1: Uncertainties overestimated")
        print(f"χ² > 1: Uncertainties underestimated")
        print(f"{'-'*60}")

    for i, param_name in enumerate(param_names):
        residuals      = pred_means[:, i] - true_values[:, i]
        variances      = pred_stds[:, i] ** 2
        valid_mask     = variances > 1e-10
        chi2           = np.mean((residuals[valid_mask] ** 2) / variances[valid_mask])
        chi2_values[param_name] = chi2
        if verbose:
            print(f"{param_name:12s}: χ² = {chi2:.4f}")

    if verbose:
        print(f"{'='*60}\n")
    return chi2_values


def tarp_test(cond_dist, features_test, params_test, param_scaler, param_names,
              device, n_samples=1000, max_test_samples=1000, output_dir=None):
    print(f"\n{'='*60}")
    print(f"TARP Test (Test of Accurate Rank Probability)")
    print(f"{'='*60}")
    print(f"Testing posterior calibration using {max_test_samples} test samples")
    print(f"Drawing {n_samples} samples per posterior")
    print(f"{'-'*60}\n")

    if max_test_samples is not None and len(features_test) > max_test_samples:
        indices      = np.random.choice(len(features_test), max_test_samples, replace=False)
        features_test = features_test[indices]
        params_test  = params_test[indices]

    n_test             = len(features_test)
    params_test_scaled = param_scaler.transform(params_test)

    print("  Generating posterior samples...")
    all_samples = []

    with torch.no_grad():
        for i in range(n_test):
            if i % 200 == 0:
                print(f"    Processing sample {i}/{n_test}...")
            single_feature   = features_test[i:i+1].to(device)
            conditioned_dist = cond_dist.condition(single_feature)
            samples          = conditioned_dist.sample((n_samples,)).squeeze(1).cpu().numpy()
            all_samples.append(samples)

    all_samples = np.array(all_samples)   # (n_test, n_samples, n_params)

    print(f"\n{'='*60}")
    print(f"TARP Coverage Analysis")
    print(f"{'='*60}\n")

    tarp_results = {}
    for i, param_name in enumerate(param_names):
        samples_param         = all_samples[:, :, i]
        true_param            = params_test_scaled[:, i]
        samples_param_reshaped = samples_param.T[:, :, np.newaxis]
        true_param_reshaped   = true_param[:, np.newaxis]

        ecp, alpha = get_tarp_coverage(
            samples_param_reshaped,
            true_param_reshaped,
            bootstrap=True,
            num_bootstrap=100,
        )
        tarp_results[param_name] = {
            'ecp':    ecp,    # shape (n_bootstrap, n_alpha_bins) when bootstrap=True
            'alpha':  alpha,  # credibility levels
            'n_test': n_test, # actual number of test observations — needed for correct target zone width
        }
        print(f"  Computed TARP for {param_name}")

    print(f"\n{'='*60}\n")

    if output_dir is not None:
        plot_tarp_library(tarp_results, param_names, output_dir)

    return tarp_results


def plot_tarp_library(tarp_results, param_names, output_dir):
    for param_name in param_names:
        ecp    = tarp_results[param_name]['ecp']
        alpha  = tarp_results[param_name]['alpha']
        n_test = tarp_results[param_name]['n_test']

        fig, ax = plt.subplots(figsize=(8, 8))

        if ecp.ndim == 2:
            ecp_mean  = np.mean(ecp, axis=0)
            ecp_lower = np.percentile(ecp, 2.5,  axis=0)
            ecp_upper = np.percentile(ecp, 97.5, axis=0)
            ax.plot(alpha, ecp_mean, linewidth=2.5, label='Actual Coverage', color='steelblue')
            ax.fill_between(alpha, ecp_lower, ecp_upper,
                            alpha=0.4, color='steelblue', label='Bootstrap Uncertainty (95% CI)')
        else:
            ax.plot(alpha, ecp, linewidth=2.5, label='Actual Coverage', color='steelblue')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect calibration')

        ax.set_xlabel('Expected Coverage Probability', fontsize=13, fontweight='bold')
        ax.set_ylabel('Empirical Coverage Probability', fontsize=13, fontweight='bold')
        ax.set_title(
            f'TARP Test - {param_name}  (N = {n_test} test galaxies)',
            fontsize=14, fontweight='bold', pad=15,
        )
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        props    = dict(boxstyle='round', facecolor='wheat', alpha=0.15, edgecolor='gray', linewidth=1.5)
        text_str = (
            "Interpretation:\n"
            "Line on diagonal = well calibrated\n"
            "Line above diagonal = underconfident\n"
            "Line below diagonal = overconfident"
        )
        ax.text(0.98, 0.02, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        output_path = output_dir / f"tarp_{param_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved TARP plot: {output_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Normalizing Flow model on test data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PATH),
        help=f"Path to the parquet file for evaluation (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help=f"Directory containing trained model files (default: {MODEL_DIR})"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(PLOTS_DIR),
        help=f"Directory to save evaluation plots (default: {PLOTS_DIR})"
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    plots_dir = Path(args.plots_dir)

    model_path  = model_dir / "flow_state.pt"
    scaler_path = model_dir / "scalers.npz"

    if not model_path.exists() or not scaler_path.exists():
        print("ERROR: No trained model found!")
        print(f"Expected model at:   {model_path}")
        print(f"Expected scalers at: {scaler_path}")
        print("\nPlease run NF_train.py first to train a model.")
        return

    # ── Read training config ──────────────────────────────────────────────────
    config       = load_train_config(model_dir)
    n_galaxies   = config.get("n_galaxies")   # None = full dataset
    n_transforms = config.get("n_transforms", 2)
    seed         = config.get("seed", SEED)
    test_size    = config.get("test_size", TEST_SIZE)
    target_log_params = config.get("target_log_params", [])

    # Set seeds for reproducibility
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Loading data from: {data_path}")

    # ── Reproduce training data pipeline exactly ──────────────────────────────
    if n_galaxies is not None:
        print(f"\nReproducing subsampling: {n_galaxies} galaxies/simulation...")
        df_raw = load_data(data_path)
        df_raw = subsample_per_simulation(df_raw, n_galaxies, seed)
        feature_cols_raw = GALAXY_PROPERTIES + ["V"]
        df_transformed, transformed_feature_cols = transform_features(df_raw.copy(), feature_cols_raw)
        df_transformed, transformed_param_cols = transform_params(
            df_transformed, PARAM_COLUMNS, target_log_params
        )
        features = df_transformed[transformed_feature_cols].values
        params   = df_transformed[transformed_param_cols].values
    else:
        print("\nUsing full dataset (no subsampling)...")
        features, params, _, transformed_feature_cols = prepare_features_and_params(
            data_path, GALAXY_PROPERTIES, PARAM_COLUMNS
        )
        if set(target_log_params) != set(TARGET_LOG_PARAMS):
            df_raw = load_data(data_path)
            df_raw, _ = transform_features(df_raw, GALAXY_PROPERTIES + ["V"])
            df_raw, transformed_param_cols = transform_params(
                df_raw, PARAM_COLUMNS, target_log_params
            )
            params = df_raw[transformed_param_cols].values

    feature_cols = transformed_feature_cols

    # ── Reproduce exact train/test split ──────────────────────────────────────
    feat_train, feat_test, params_train, params_test = train_test_split(
        features, params, test_size=test_size, random_state=seed
    )

    print(f"\nLoading trained model from {model_dir}...")
    cond_dim  = len(feature_cols)
    param_dim = len(PARAM_COLUMNS)
    print(f"  n_transforms = {n_transforms}")

    cond_dist, modules, feat_scaler, param_scaler = load_model(
        model_path, scaler_path, cond_dim, param_dim, n_transforms, device
    )
    print("Model loaded successfully!")

    # Standardize test features using training scaler
    feat_test_scaled    = feat_scaler.transform(feat_test)
    feat_test_tensor    = torch.tensor(feat_test_scaled, dtype=torch.float32)

    # ── Compute predictions ───────────────────────────────────────────────────
    print(f"\nComputing predicted means and stds for {len(feat_test)} test samples...")
    pred_means_physical, pred_stds_physical, subsample_indices = compute_predictions(
        cond_dist, feat_test_tensor, device,
        param_scaler=param_scaler,
        target_log_params=target_log_params,
        param_names=PARAM_COLUMNS,
        n_samples=N_POSTERIOR_SAMPLES,
        max_test_samples=MAX_TEST_SAMPLES,
    )

    params_test_subset = (params_test[subsample_indices]
                          if subsample_indices is not None else params_test)

    # Convert true params from (log-)transformed space to physical units
    # for comparison against the physical-space predictions
    params_test_subset_physical = inverse_transform_params(
        params_test_subset, PARAM_COLUMNS, target_log_params
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nCreating predicted vs true plots...")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_pred_vs_true(
        pred_means_physical, pred_stds_physical,
        params_test_subset_physical, PARAM_COLUMNS, plots_dir,
    )

    feat_test_subset_tensor = (feat_test_tensor[subsample_indices]
                               if subsample_indices is not None else feat_test_tensor)

    tarp_results = tarp_test(
        cond_dist, feat_test_subset_tensor, params_test_subset,
        param_scaler, PARAM_COLUMNS, device,
        n_samples=N_POSTERIOR_SAMPLES,
        max_test_samples=MAX_TEST_SAMPLES,
        output_dir=plots_dir,
    )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"All results saved to: {plots_dir}")
    print(f"  - Predicted vs True plots (with metrics + χ²)")
    print(f"  - TARP calibration plots (with bootstrap CI)")
    if n_galaxies is not None:
        print(f"  - Data pipeline reproduced with {n_galaxies} galaxies/simulation")
    print("="*60)


if __name__ == "__main__":
    main()
