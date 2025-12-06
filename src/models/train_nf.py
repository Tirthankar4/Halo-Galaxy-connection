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
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.utils.io import ensure_dir, save_json

FEATURE_COLUMNS = ["M_h", "R_h", "V_h"]
TARGET_COLUMNS = ["SM", "SFR", "Colour", "SR"]


def resolve_device(device_arg: str) -> torch.device:
    """Prefer CUDA and prompt before falling back to CPU."""
    device_arg = device_arg or "cuda"
    wants_cuda = device_arg.lower().startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        message = (
            "CUDA device requested but no GPU is available. "
            "Type 'Y' to continue on CPU (anything else aborts): "
        )
        if input(message).strip().lower() != "y":
            raise SystemExit("Aborted: GPU unavailable and CPU fallback declined.")
        logging.warning("Falling back to CPU because CUDA was requested but unavailable.")
        return torch.device("cpu")
    return torch.device(device_arg)


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
        loss_history.append(float(loss.item()))

        if step % 100 == 0:
            logging.info("Step %d / %d, loss %.4f", step, steps, loss.item())

    return dist_x1, dist_x2_given_x1, modules, loss_history


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    ks_stat, ks_p = ks_2samp(y_true, y_pred)
    wass = float(wasserstein_distance(y_true, y_pred))
    return {
        "RMSE": rmse,
        "Pearson_r": float(pearson_r),
        "Pearson_p": float(pearson_p),
        "KS_stat": float(ks_stat),
        "KS_p": float(ks_p),
        "Wasserstein": wass,
    }


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

    # inverse scale
    mean_unscaled = gals_scaler.inverse_transform(mean_preds)
    rand_unscaled = gals_scaler.inverse_transform(random_preds)
    posterior_unscaled = np.zeros_like(posterior_samples)
    for i in range(posterior_samples.shape[0]):
        posterior_unscaled[i] = gals_scaler.inverse_transform(posterior_samples[i])

    return {
        "posterior": posterior_unscaled,
        "mean": mean_unscaled,
        "random": rand_unscaled,
    }


def run_nf_training(
    df: pd.DataFrame,
    test_size: float,
    steps: int,
    lr: float,
    n_samples: int,
    seed: int,
    output_dir: Path,
    run_name: str,
    device: torch.device,
):
    ensure_dir(output_dir / run_name)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    halos = df[FEATURE_COLUMNS].values
    gals = df[TARGET_COLUMNS].values

    halos_train, halos_test, gals_train, gals_test = train_test_split(
        halos, gals, test_size=test_size, random_state=seed
    )

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
        halos_train_tensor, gals_train_tensor, steps, lr, device
    )

    predictions = generate_predictions(
        dist_x2_given_x1, halos_test_tensor, gal_scaler, n_samples, device
    )
    y_true = gal_scaler.inverse_transform(gals_test_scaled)

    metrics = {}
    for idx, prop in enumerate(TARGET_COLUMNS):
        metrics[prop] = compute_metrics(y_true[:, idx], predictions["mean"][:, idx])

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional normalizing flow.")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_PROCESSED_PARQUET,
        help="Processed parquet path.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    df = pd.read_parquet(args.data)
    device = resolve_device(args.device)
    ensure_dir(args.output_dir / args.run_name)

    run_nf_training(
        df=df,
        test_size=args.test_size,
        steps=args.steps,
        lr=args.lr,
        n_samples=args.posterior_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=device,
    )


if __name__ == "__main__":
    main()

