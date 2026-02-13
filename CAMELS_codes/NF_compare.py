"""
Compare multiple trained Normalizing Flow models on the same test data.

Loads all models from a directory, evaluates each on its corresponding test set,
and produces a leaderboard ranked by Test NLL. Also generates a training loss
convergence plot and saves results to CSV.

Usage:
    python NF_compare.py --models-dir ./CAMELS_outputs/
"""

import argparse
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import pyro

from NF_evaluate import (
    load_model,
    subsample_per_simulation,
    build_flow,
    get_tarp_coverage,
)
from transformation import prepare_features_and_params, load_data, transform_features


# ============ Configuration ============
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "CAMELS_datas" / "camels_astrid_sb7_090.parquet"
DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "CAMELS_outputs"

# Evaluation settings (reduced for fast comparison; use NF_evaluate.py for detailed analysis)
MAX_EVAL_SAMPLES = 500   # galaxies for PCC/RMSE
MAX_TARP_SAMPLES = 200   # galaxies for TARP calibration score
N_POSTERIOR_DRAWS = 50   # posterior samples per galaxy


# ── Data loading helpers ──────────────────────────────────────────────────────

def _strip_log_prefix(cols):
    """Convert ['log_Mg', 'log_MBH', ...] -> ['Mg', 'MBH', ...]"""
    return [c[4:] if c.startswith("log_") else c for c in cols]


def load_test_data(config, data_path):
    """
    Reproduce the exact train/test split used during training.
    
    Returns:
        features: np.ndarray of shape (n_test, n_features)
        params: np.ndarray of shape (n_test, n_params)
        feature_cols: list of feature column names
        param_cols: list of parameter column names
    """
    n_galaxies = config.get("n_galaxies")
    seed = config.get("seed", 42)
    test_size = config.get("test_size", 0.2)
    feature_cols = config.get("feature_cols")
    param_cols = config.get("param_cols")
    
    # Set seeds for reproducibility
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Derive raw column names (before log transform)
    feature_cols_raw = _strip_log_prefix(feature_cols)
    
    # Load and transform data
    df_raw = load_data(data_path)
    if n_galaxies is not None:
        df_raw = subsample_per_simulation(df_raw, n_galaxies, seed)
    
    df_transformed, transformed_cols = transform_features(df_raw.copy(), feature_cols_raw)
    features = df_transformed[transformed_cols].values
    params = df_raw[param_cols].values
    
    # Reproduce train/test split
    _, feat_test, _, params_test = train_test_split(
        features, params, test_size=test_size, random_state=seed
    )
    
    return feat_test, params_test, feature_cols, param_cols


# ── Metric computation ────────────────────────────────────────────────────────

def compute_test_nll(cond_dist, features, params, device, batch_size=1024):
    """
    Compute negative log-likelihood on test set.
    This is the primary metric for normalizing flow quality.
    """
    n_test = len(features)
    nll_sum = 0.0
    
    cond_dist.clear_cache()
    
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            batch_feat = features[i:i+batch_size].to(device)
            batch_param = params[i:i+batch_size].to(device)
            
            dist = cond_dist.condition(batch_feat)
            log_prob = dist.log_prob(batch_param).sum()
            nll_sum -= log_prob.item()
    
    return nll_sum / n_test


def compute_point_metrics(cond_dist, features, params_true, param_scaler, param_cols, device):
    """
    Compute point estimate metrics (PCC, RMSE) using posterior means.
    
    Returns:
        avg_pcc: average Pearson correlation across all parameters
        rmse: root mean squared error in physical units
        pccs: dict mapping parameter name -> PCC value
    """
    # Subsample for speed
    if len(features) > MAX_EVAL_SAMPLES:
        idx = np.random.choice(len(features), MAX_EVAL_SAMPLES, replace=False)
        features = features[idx]
        params_true = params_true[idx]
    
    # Generate posterior means (one galaxy at a time for Pyro compatibility)
    pred_scaled = []
    with torch.no_grad():
        for i in range(len(features)):
            conditioned = cond_dist.condition(features[i:i+1].to(device))
            samples = conditioned.sample((N_POSTERIOR_DRAWS,)).squeeze(1)
            pred_scaled.append(samples.mean(dim=0).cpu().numpy())
    
    pred_scaled = np.array(pred_scaled)
    preds_phys = param_scaler.inverse_transform(pred_scaled)
    
    # Compute per-parameter PCC
    pccs = {}
    for i, name in enumerate(param_cols):
        pcc, _ = pearsonr(params_true[:, i], preds_phys[:, i])
        pccs[name] = pcc
    
    avg_pcc = np.mean(list(pccs.values()))
    rmse = np.sqrt(mean_squared_error(params_true, preds_phys))
    
    return avg_pcc, rmse, pccs


def compute_tarp_score(cond_dist, features, params, device):
    """
    Compute TARP calibration score (mean absolute deviation from diagonal).
    
    A score of 0.0 means perfect calibration. Lower is better.
    """
    # Subsample for speed
    if len(features) > MAX_TARP_SAMPLES:
        idx = np.random.choice(len(features), MAX_TARP_SAMPLES, replace=False)
        features = features[idx]
        params = params[idx]
    
    # Generate posterior samples
    all_samples = []
    with torch.no_grad():
        for i in range(len(features)):
            conditioned = cond_dist.condition(features[i:i+1].to(device))
            sample_i = conditioned.sample((N_POSTERIOR_DRAWS,)).squeeze(1).cpu().numpy()
            all_samples.append(sample_i)
    
    # Shape: (n_test, n_samples, n_params) -> (n_samples, n_test, n_params)
    samples = np.array(all_samples).transpose(1, 0, 2)
    
    ecp, alpha = get_tarp_coverage(samples, params.numpy(), bootstrap=False)
    
    return np.mean(np.abs(ecp - alpha))


# ── Model evaluation ──────────────────────────────────────────────────────────

def evaluate_single_model(model_dir, data_path, device):
    """
    Load a model and compute all evaluation metrics.
    
    Returns:
        dict with model name, hyperparameters, and all metrics
        None if model cannot be loaded
    """
    config_path = model_dir / "train_config.json"
    if not config_path.exists():
        print(f"Skipping {model_dir.name}: No train_config.json")
        return None
    
    print(f"--- Evaluating {model_dir.name} ---")
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Load test data
    feat_test, params_test, feature_cols, param_cols = load_test_data(config, data_path)
    
    # Load model
    n_transforms = config.get("n_transforms", 2)
    cond_dist, modules, feat_scaler, param_scaler = load_model(
        model_dir / "flow_state.pt",
        model_dir / "scalers.npz",
        len(feature_cols),
        len(param_cols),
        n_transforms,
        device,
    )
    
    # Prepare scaled tensors
    feat_scaled = torch.tensor(feat_scaler.transform(feat_test), dtype=torch.float32)
    params_scaled = torch.tensor(param_scaler.transform(params_test), dtype=torch.float32)
    
    # Compute metrics
    nll = compute_test_nll(cond_dist, feat_scaled, params_scaled, device)
    avg_pcc, rmse, pccs = compute_point_metrics(
        cond_dist, feat_scaled, params_test, param_scaler, param_cols, device
    )
    tarp_score = compute_tarp_score(cond_dist, feat_scaled, params_scaled, device)
    
    return {
        "Model Name": model_dir.name,
        "n_trans": n_transforms,
        "n_gal": config.get("n_galaxies") or "All",
        "lr": config.get("lr"),
        "epochs": config.get("epochs"),
        "Test NLL": nll,
        "Avg PCC": avg_pcc,
        "Avg RMSE": rmse,
        "TARP Score": tarp_score,
        "Omega_m PCC": pccs.get("Omega_m", 0.0),
        "sigma_8 PCC": pccs.get("sigma_8", 0.0),
        "loss_history_path": model_dir / "loss_history.npy",
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_loss_history(results_df):
    """Overlay training loss curves for all models."""
    plt.figure(figsize=(10, 6))
    
    for _, row in results_df.iterrows():
        path = row["loss_history_path"]
        if path.exists():
            loss = np.load(path)
            smoothed = pd.Series(loss).rolling(10).mean()
            plt.plot(smoothed, label=f"{row['Model Name']} (NLL={row['Test NLL']:.2f})")
    
    plt.xlabel("Step")
    plt.ylabel("Training Loss (NLL)")
    plt.title("Training Convergence Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple NF models and produce a leaderboard."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Root directory containing model subfolders",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to data file",
    )
    args = parser.parse_args()

    models_root = Path(args.models_dir)
    data_path = Path(args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate all models
    results = []
    for model_dir in sorted(models_root.iterdir()):
        if model_dir.is_dir() and (model_dir / "train_config.json").exists():
            res = evaluate_single_model(model_dir, data_path, device)
            if res:
                results.append(res)

    if not results:
        print("No models found to evaluate.")
        return

    df = pd.DataFrame(results).sort_values("Test NLL")

    # Print leaderboard
    print("\n" + "=" * 80)
    print("MODEL COMPARISON LEADERBOARD")
    print("Sorted by Test NLL (Lower is Better)")
    print("=" * 80)

    display_cols = ["Model Name", "n_trans", "n_gal", "Test NLL", "Avg PCC", "TARP Score", "Omega_m PCC"]
    print(df[display_cols].to_string(index=False, float_format="%.4f"))

    print("-" * 80)
    print("NOTES:")
    print("  * Test NLL: Global fit quality. Lower is better.")
    print("  * Avg PCC:  Linear correlation (structural capture). Higher is better.")
    print("  * TARP Score: Calibration error. 0.0 is perfect. Lower is better.")
    print("=" * 80)

    # Plot and save
    plot_loss_history(df)
    
    output_path = models_root / "comparison_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
