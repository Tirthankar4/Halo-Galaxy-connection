"""
Train the conditional normalizing flow used for galaxy property inference.
Produces saved transform weights, scalers, metrics, and cached posterior
samples for downstream plotting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.utils.io import ensure_dir, save_json
from src.utils.common import (
    resolve_device,
    compute_regression_metrics,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    ALL_HALO_PROPERTIES,
    ALL_GALAXY_PROPERTIES,
    validate_features,
    validate_targets,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def build_flow(halo_dim: int, gal_dim: int) -> Tuple[dist.Distribution, dist.ConditionalDistribution, torch.nn.ModuleList]:
    base = dist.Normal(torch.zeros(halo_dim), torch.ones(halo_dim) * 0.2).to_event(1)
    x1_transform = T.spline(halo_dim)
    x3_transform = T.affine_coupling(halo_dim)
    dist_x1 = dist.TransformedDistribution(base, [x1_transform, x3_transform])

    x2_transform = T.conditional_spline(gal_dim, context_dim=halo_dim)
    cond_base = dist.Normal(torch.zeros(gal_dim), torch.ones(gal_dim) * 0.2).to_event(1)
    dist_x2_given_x1 = dist.ConditionalTransformedDistribution(cond_base, [x2_transform])

    modules = torch.nn.ModuleList([x1_transform, x3_transform, x2_transform])
    return dist_x1, dist_x2_given_x1, modules


def train_flow(
    halos_train: torch.Tensor,
    gals_train: torch.Tensor,
    steps: int,
    lr: float,
    device: torch.device,
    use_wandb: bool = False,
):
    halo_dim = halos_train.shape[1]
    gal_dim = gals_train.shape[1]
    dist_x1, dist_x2_given_x1, modules = build_flow(halo_dim, gal_dim)
    modules = modules.to(device)
    
    # Move base distributions to device (handles Independent wrapper from to_event)
    # Find the actual Normal distribution by traversing through Independent wrappers
    base_dist_x1 = dist_x1.base_dist
    while hasattr(base_dist_x1, 'base_dist'):
        base_dist_x1 = base_dist_x1.base_dist
    base_dist_x1.loc = base_dist_x1.loc.to(device)
    base_dist_x1.scale = base_dist_x1.scale.to(device)
    
    base_dist_x2 = dist_x2_given_x1.base_dist
    while hasattr(base_dist_x2, 'base_dist'):
        base_dist_x2 = base_dist_x2.base_dist
    base_dist_x2.loc = base_dist_x2.loc.to(device)
    base_dist_x2.scale = base_dist_x2.scale.to(device)

    optimizer = torch.optim.Adam(modules.parameters(), lr=lr)
    loss_history = []

    halos_train = halos_train.to(device)
    gals_train = gals_train.to(device)

    for step in range(steps + 1):
        optimizer.zero_grad()
        ln_p_x1 = dist_x1.log_prob(halos_train)
        ln_p_x2_given_x1 = dist_x2_given_x1.condition(halos_train.detach()).log_prob(
            gals_train.detach()
        )
        loss = -(ln_p_x1 + ln_p_x2_given_x1).mean()
        loss.backward()
        optimizer.step()
        dist_x1.clear_cache()
        dist_x2_given_x1.clear_cache()
        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if step % 100 == 0:
            logging.info("Step %d / %d, loss %.4f", step, steps, loss.item())
            # Log to WandB every 100 steps
            if use_wandb and wandb is not None:
                wandb.log({"loss": loss_val, "step": step})

    return dist_x1, dist_x2_given_x1, modules, loss_history




def generate_predictions(
    dist_x2_given_x1,
    halos_test_tensor: torch.Tensor,
    gals_scaler: StandardScaler,
    n_samples: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    halos_test_tensor = halos_test_tensor.to(device)
    mean_preds = []
    random_preds = []
    posterior_samples = []

    with torch.no_grad():
        for halo in halos_test_tensor:
            halo = halo.unsqueeze(0)
            samples = dist_x2_given_x1.condition(halo).sample(torch.Size([n_samples]))
            samples_np = samples.squeeze(1).cpu().numpy()
            posterior_samples.append(samples_np)
            mean_preds.append(samples_np.mean(axis=0))
            rand_idx = np.random.randint(0, n_samples)
            random_preds.append(samples_np[rand_idx])

    posterior_samples = np.stack(posterior_samples)  # (n_test, n_samples, gal_dim)
    mean_preds = np.array(mean_preds)
    random_preds = np.array(random_preds)

    # Ensure 2D arrays for inverse_transform (handles single target case)
    mean_was_1d = mean_preds.ndim == 1
    random_was_1d = random_preds.ndim == 1
    
    if mean_was_1d:
        mean_preds = mean_preds.reshape(-1, 1)
    if random_was_1d:
        random_preds = random_preds.reshape(-1, 1)

    # inverse scale
    mean_unscaled = gals_scaler.inverse_transform(mean_preds)
    rand_unscaled = gals_scaler.inverse_transform(random_preds)
    
    # Squeeze back to original shape if needed (for single target case)
    if mean_was_1d:
        mean_unscaled = mean_unscaled.squeeze(1)
    if random_was_1d:
        rand_unscaled = rand_unscaled.squeeze(1)
    
    posterior_unscaled = np.zeros_like(posterior_samples)
    for i in range(posterior_samples.shape[0]):
        posterior_sample = posterior_samples[i]
        sample_was_1d = posterior_sample.ndim == 1
        if sample_was_1d:
            posterior_sample = posterior_sample.reshape(-1, 1)
        transformed = gals_scaler.inverse_transform(posterior_sample)
        # Squeeze back to match original shape
        if sample_was_1d:
            transformed = transformed.squeeze(1)
        posterior_unscaled[i] = transformed

    return {
        "posterior": posterior_unscaled,
        "mean": mean_unscaled,
        "random": rand_unscaled,
    }


def run_nf_training(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    test_size: float,
    steps: int,
    lr: float,
    n_samples: int,
    seed: int,
    output_dir: Path,
    run_name: str,
    device: torch.device,
    use_wandb: bool = False,
    data_fraction: float = 1.0,
):
    ensure_dir(output_dir / run_name)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize WandB if requested
    if use_wandb and wandb is not None:
        wandb.init(
            project="halo-galaxy-connection",
            name=run_name,
            config={
                "steps": steps,
                "lr": lr,
                "test_size": test_size,
                "seed": seed,
                "n_samples": n_samples,
                "device": str(device),
                "data_fraction": data_fraction,
                "features": features,
                "targets": targets,
            },
        )

    logging.info("Training with features: %s", features)
    logging.info("Training targets: %s", targets)

    halos = df[features].values
    gals = df[targets].values

    halos_train, halos_test, gals_train, gals_test = train_test_split(
        halos, gals, test_size=test_size, random_state=seed
    )

    # Subsample training data if requested
    if data_fraction < 1.0:
        original_train_size = len(halos_train)
        n_train_subset = int(len(halos_train) * data_fraction)
        indices = np.random.choice(len(halos_train), n_train_subset, replace=False)
        halos_train = halos_train[indices]
        gals_train = gals_train[indices]
        logging.info("Subsampled training data: %d / %d samples (%.1f%%)", 
                     n_train_subset, original_train_size, 
                     data_fraction * 100)

    halo_scaler = StandardScaler()
    gal_scaler = StandardScaler()
    halos_train_scaled = halo_scaler.fit_transform(halos_train)
    halos_test_scaled = halo_scaler.transform(halos_test)
    gals_train_scaled = gal_scaler.fit_transform(gals_train)
    gals_test_scaled = gal_scaler.transform(gals_test)

    halos_train_tensor = torch.tensor(halos_train_scaled, dtype=torch.float32)
    gals_train_tensor = torch.tensor(gals_train_scaled, dtype=torch.float32)
    halos_test_tensor = torch.tensor(halos_test_scaled, dtype=torch.float32)
    gals_test_tensor = torch.tensor(gals_test_scaled, dtype=torch.float32)

    dist_x1, dist_x2_given_x1, modules, losses = train_flow(
        halos_train_tensor, gals_train_tensor, steps, lr, device, use_wandb=use_wandb
    )

    predictions = generate_predictions(
        dist_x2_given_x1, halos_test_tensor, gal_scaler, n_samples, device
    )
    y_true = gal_scaler.inverse_transform(gals_test_scaled)

    metrics = {}
    for idx, prop in enumerate(targets):
        # Handle single target case (1D arrays)
        if len(targets) == 1:
            y_true_prop = y_true
            y_pred_prop = predictions["mean"]
        else:
            y_true_prop = y_true[:, idx]
            y_pred_prop = predictions["mean"][:, idx]
        
        metrics[prop] = compute_regression_metrics(y_true_prop, y_pred_prop)
        # Log metrics to WandB
        if use_wandb and wandb is not None:
            wandb.log({
                f"{prop}/RMSE": metrics[prop]["RMSE"],
                f"{prop}/Pearson_r": metrics[prop]["Pearson_r"],
                f"{prop}/KS_stat": metrics[prop]["KS_stat"],
                f"{prop}/Wasserstein": metrics[prop]["Wasserstein"],
            })

    run_dir = output_dir / run_name
    torch.save(modules.state_dict(), run_dir / "flow_state.pt")
    save_json(
        {
            "halo_scaler": {"mean": halo_scaler.mean_.tolist(), "scale": halo_scaler.scale_.tolist()},
            "gal_scaler": {"mean": gal_scaler.mean_.tolist(), "scale": gal_scaler.scale_.tolist()},
            "loss_history": losses,
            "metrics": metrics,
            "train_size": int(len(halos_train)),
            "test_size": int(len(halos_test)),
            "features": features,
            "targets": targets,
        },
        run_dir / "training_summary.json",
    )
    np.savez(
        run_dir / "predictions.npz",
        y_true=y_true,
        mean=predictions["mean"],
        random=predictions["random"],
        posterior=predictions["posterior"],
    )
    logging.info("Saved NF artifacts to %s", run_dir)

    # Finish WandB run
    if use_wandb and wandb is not None:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional normalizing flow.")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_PROCESSED_PARQUET,
        help="Processed parquet path.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=FEATURE_COLUMNS,
        choices=ALL_HALO_PROPERTIES,
        help=f"Halo property features to use for training (default: {FEATURE_COLUMNS}). Available: {ALL_HALO_PROPERTIES}",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TARGET_COLUMNS,
        choices=ALL_GALAXY_PROPERTIES,
        help=f"Galaxy property targets to predict (default: {TARGET_COLUMNS}). Available: {ALL_GALAXY_PROPERTIES}",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.DEFAULT_TEST_SIZE,
        help="Fraction reserved for testing.",
    )
    parser.add_argument("--steps", type=int, default=1300, help="Training steps.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples per halo for cached predictions.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.DEFAULT_RANDOM_STATE, help="Random seed."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (defaults to CUDA, falls back to CPU if confirmed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.NF_MODEL_DIR,
        help="Directory to store NF weights and metrics.",
    )
    parser.add_argument(
        "--run-name",
        default="default",
        help="Name for this NF experiment run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging for experiment tracking.",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.01 = 1%%, 1.0 = 100%%)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    df = pd.read_parquet(args.data)
    device = resolve_device(args.device)
    ensure_dir(args.output_dir / args.run_name)

    # Validate feature and target selections
    features = validate_features(args.features)
    targets = validate_targets(args.targets)

    run_nf_training(
        df=df,
        features=features,
        targets=targets,
        test_size=args.test_size,
        steps=args.steps,
        lr=args.lr,
        n_samples=args.posterior_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=device,
        use_wandb=args.use_wandb,
        data_fraction=args.data_fraction,
    )


if __name__ == "__main__":
    main()

