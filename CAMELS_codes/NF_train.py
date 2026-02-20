"""
Normalizing Flow to predict cosmological & astrophysical parameters from galaxy properties.
Uses CAMELS ASTRID data, training only on central galaxies.
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
from transformation import (
    ASTROPHYSICAL_LOG_PARAMS,
    prepare_features_and_params,
    load_data,
    transform_features,
    transform_params,
)

# ============ Configuration ============
DATA_PATH = Path(__file__).parent.parent / "CAMELS_datas" / "camels_astrid_sb7_090.parquet"
OUTPUT_DIR = Path(__file__).parent.parent / "CAMELS_outputs" / "model_data"

# Galaxy properties (features/conditions)
GALAXY_PROPERTIES = [
    'Mg', 'MBH', 'Mstar', 'Mt', 'Vmax', 'sigma_v', 'Zg', 'Zstar',
    'SFR', 'J', 'Rstar', 'Rt', 'Rmax'
]
# Note: 'V' (velocity modulus) is computed from vel_x, vel_y, vel_z

# Parameters to learn (targets)
PARAM_COLUMNS = ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2', 'Omega_b']
TARGET_LOG_PARAMS = ASTROPHYSICAL_LOG_PARAMS

# Training hyperparameters
EPOCHS = 200
BATCH_SIZE = 641860
LR = 1e-3
N_TRANSFORMS = 4
TEST_SIZE = 0.2
SEED = 42


def subsample_per_simulation(df, n_galaxies):
    """
    Randomly sample up to n_galaxies from each simulation.
    Simulations with fewer galaxies than n_galaxies are kept in full.
    """
    import pandas as pd
    rng = np.random.default_rng(SEED)
    kept = []
    for sim_id, group in df.groupby("simulation_id"):
        if len(group) <= n_galaxies:
            kept.append(group)
        else:
            idx = rng.choice(group.index, size=n_galaxies, replace=False)
            kept.append(group.loc[idx])
    result = pd.concat(kept).reset_index(drop=True)
    n_sims = df["simulation_id"].nunique()
    print(
        f"  Subsampled: {n_galaxies} galaxies/sim x {n_sims} sims "
        f"-> {len(result):,} galaxies  (was {len(df):,})"
    )
    return result


def build_flow(cond_dim: int, param_dim: int, n_transforms: int = 4):
    """
    Build conditional normalizing flow: P(params | galaxy_properties)

    Args:
        cond_dim: Dimension of conditioning variables (galaxy properties)
        param_dim: Dimension of target parameters
        n_transforms: Number of stacked spline transforms

    Returns:
        Conditional distribution and trainable modules
    """
    base = dist.Normal(torch.zeros(param_dim), torch.ones(param_dim)).to_event(1)

    transforms = []
    for i in range(n_transforms):
        transforms.append(T.conditional_spline(param_dim, context_dim=cond_dim))

    cond_dist = dist.ConditionalTransformedDistribution(base, transforms)
    modules = torch.nn.ModuleList(transforms)
    return cond_dist, modules


def train_flow(
    features_train: torch.Tensor,
    params_train: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    n_transforms: int,
    device: torch.device,
):
    """Train the conditional normalizing flow with mini-batch training."""
    cond_dim = features_train.shape[1]
    param_dim = params_train.shape[1]

    cond_dist, modules = build_flow(cond_dim, param_dim, n_transforms)
    modules = modules.to(device)

    base_dist = cond_dist.base_dist
    while hasattr(base_dist, 'base_dist'):
        base_dist = base_dist.base_dist
    base_dist.loc = base_dist.loc.to(device)
    base_dist.scale = base_dist.scale.to(device)

    optimizer = torch.optim.Adam(modules.parameters(), lr=lr)
    loss_history = []

    dataset = torch.utils.data.TensorDataset(features_train, params_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        epoch_losses = []

        for batch_features, batch_params in dataloader:
            batch_features = batch_features.to(device)
            batch_params = batch_params.to(device)

            optimizer.zero_grad()

            log_prob = cond_dist.condition(batch_features).log_prob(batch_params)
            loss = -log_prob.mean()

            loss.backward()
            optimizer.step()
            cond_dist.clear_cache()

            loss_val = float(loss.item())
            loss_history.append(loss_val)
            epoch_losses.append(loss_val)

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

    return cond_dist, modules, loss_history


def main():
    parser = argparse.ArgumentParser(
        description="Train Normalizing Flow to predict cosmological & astrophysical parameters"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PATH),
        help=f"Path to the parquet file for training (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--n-galaxies",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, randomly sample N galaxies from each simulation before training. "
            "Simulations with fewer than N galaxies are kept in full. "
            "Omit to use all galaxies (default)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Directory to save trained model outputs (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    n_galaxies = args.n_galaxies
    output_dir = Path(args.output_dir)

    # Set seeds for reproducibility
    pyro.set_rng_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from: {data_path}")

    # ── Load and optionally subsample ────────────────────────────────────────
    if n_galaxies is not None:
        print(f"\nSubsampling to {n_galaxies} galaxies per simulation...")
        df_raw = load_data(data_path)
        df_raw = subsample_per_simulation(df_raw, n_galaxies)
        feature_cols = GALAXY_PROPERTIES + ["V"]
        df_transformed, transformed_feature_cols = transform_features(df_raw.copy(), feature_cols)
        df_transformed, transformed_param_cols = transform_params(
            df_transformed, PARAM_COLUMNS, TARGET_LOG_PARAMS
        )
        features = df_transformed[transformed_feature_cols].values
        params   = df_transformed[transformed_param_cols].values
    else:
        features, params, _, transformed_feature_cols = prepare_features_and_params(
            data_path, GALAXY_PROPERTIES, PARAM_COLUMNS
        )

    feature_cols = transformed_feature_cols

    # ── Train/test split ──────────────────────────────────────────────────────
    feat_train, feat_test, params_train, params_test = train_test_split(
        features, params, test_size=TEST_SIZE, random_state=SEED
    )

    # ── Standardize ──────────────────────────────────────────────────────────
    feat_scaler = StandardScaler()
    param_scaler = StandardScaler()

    feat_train_scaled   = feat_scaler.fit_transform(feat_train)
    feat_test_scaled    = feat_scaler.transform(feat_test)
    params_train_scaled = param_scaler.fit_transform(params_train)
    params_test_scaled  = param_scaler.transform(params_test)

    # ── Convert to tensors ────────────────────────────────────────────────────
    feat_train_tensor   = torch.tensor(feat_train_scaled,   dtype=torch.float32)
    params_train_tensor = torch.tensor(params_train_scaled, dtype=torch.float32)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining NF with {len(feat_train)} samples...")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, N_Transforms: {N_TRANSFORMS}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Targets ({len(PARAM_COLUMNS)}): {PARAM_COLUMNS}")
    print(f"Target log-transform cols: {TARGET_LOG_PARAMS}")
    if n_galaxies is not None:
        print(f"Subsampling: {n_galaxies} galaxies/simulation")

    cond_dist, modules, losses = train_flow(
        feat_train_tensor, params_train_tensor, EPOCHS, BATCH_SIZE, LR, N_TRANSFORMS, device
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(modules.state_dict(), output_dir / "flow_state.pt")
    np.savez(
        output_dir / "scalers.npz",
        feat_mean=feat_scaler.mean_,
        feat_scale=feat_scaler.scale_,
        param_mean=param_scaler.mean_,
        param_scale=param_scaler.scale_,
    )
    np.save(output_dir / "loss_history.npy", np.array(losses))

    # ── Save training config (read by NF_evaluate.py) ─────────────────────────
    train_config = {
        "n_galaxies":   n_galaxies,      # None means all galaxies were used
        "n_transforms": N_TRANSFORMS,
        "test_size":    TEST_SIZE,
        "seed":         SEED,
        "epochs":       EPOCHS,
        "batch_size":   BATCH_SIZE,
        "lr":           LR,
        "data_path":    str(data_path),
        "feature_cols": feature_cols,
        "param_cols":   PARAM_COLUMNS,
        "target_log_params": TARGET_LOG_PARAMS,
    }
    config_path = output_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print(f"  flow_state.pt, scalers.npz, loss_history.npy, train_config.json")
    print(f"Final loss: {losses[-1]:.4f}")
    if n_galaxies is not None:
        print(f"Trained with {n_galaxies} galaxies/simulation subsampling.")


if __name__ == "__main__":
    main()
