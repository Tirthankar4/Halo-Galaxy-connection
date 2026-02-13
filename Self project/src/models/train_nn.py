"""
Train feed-forward neural networks (one per galaxy property) against the
processed halo dataset. Supports optional SMOGN resampling to emphasize
tail regions. Saves model weights, scalers, and evaluation metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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
from src.utils.data import normalize_features, normalize_targets

try:
    import smogn
except ImportError:  # pragma: no cover
    smogn = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

# SMOGN configuration for each target property
SMOGN_CONFIG = {
    "SM": dict(
        k=3,
        pert=0.01,
        samp_method="extreme",
        rel_method="manual",
        rel_ctrl_pts_rg=[
            [6.5, 0, 0],
            [10.8, 0, 0],
            [11.2, 0.8, 0],
            [11.4, 1, 0],
            [11.63, 1, 0],
        ],
        under_samp=True,
    ),
    "SFR": dict(
        k=5,
        pert=0.025,
        samp_method="extreme",
        rel_method="manual",
        rel_ctrl_pts_rg=[
            [-2.5, 1, 0],
            [-0.5, 1, 0],
            [2.0, 0, 0],
            [8.5, 0.8, 0],
            [9.2, 1, 0],
            [9.57, 1, 0],
        ],
        under_samp=True,
    ),
    "Colour": dict(
        k=5,
        pert=0.025,
        samp_method="extreme",
        rel_method="manual",
        rel_ctrl_pts_rg=[
            [-0.13, 0.9, 0],
            [0.3, 0, 0],
            [1.05, 0.7, 0],
            [1.25, 1, 0],
            [1.32, 1, 0],
        ],
        under_samp=True,
    ),
    "SR": dict(
        k=5,
        pert=0.025,
        samp_method="extreme",
        rel_method="manual",
        rel_ctrl_pts_rg=[
            [-2.0, 0, 0],
            [0.5, 0, 0],
            [1.0, 0.6, 0],
            [1.3, 0.9, 0],
            [1.48, 1, 0],
        ],
        under_samp=True,
    ),
}




class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)




def apply_smogn_resampling(
    X_train: pd.DataFrame, y_train: pd.Series, target: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    if smogn is None:
        raise ImportError("smogn package not found. Install it or disable --use-smogn.")
    dataset = X_train.reset_index(drop=True).copy()
    target_series = (
        y_train.reset_index(drop=True)
        if isinstance(y_train, pd.Series)
        else pd.Series(y_train).reset_index(drop=True)
    )
    dataset[target] = target_series.values
    cfg = SMOGN_CONFIG[target]
    logging.info("Applying SMOGN to %s with params %s", target, cfg)
    resampled = smogn.smoter(data=dataset, y=target, **cfg)
    y_res = resampled[target].values
    X_res = resampled.drop(columns=[target])
    return X_res, y_res


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
    use_wandb: bool = False,
    target: str = None,
) -> Tuple[SimpleMLP, List[float], List[float], np.ndarray]:
    input_size = X_train.shape[1]
    model = SimpleMLP(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_loss_val = float(loss.item())
        train_losses.append(train_loss_val)

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = criterion(test_pred, y_test_tensor).item()
            test_losses.append(float(test_loss))

        # Log to WandB every epoch
        if use_wandb and wandb is not None and target is not None:
            wandb.log({
                f"{target}/train_loss": train_loss_val,
                f"{target}/test_loss": float(test_loss),
                "epoch": epoch,
            })

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()
    return model, train_losses, test_losses, preds




def run_training(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    use_smogn: bool,
    test_size: float,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    run_name: str,
    seed: int,
    use_wandb: bool = False,
    data_fraction: float = 1.0,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize WandB if requested
    if use_wandb and wandb is not None:
        wandb.init(
            project="halo-galaxy-connection",
            name=run_name,
            config={
                "epochs": epochs,
                "lr": lr,
                "test_size": test_size,
                "seed": seed,
                "use_smogn": use_smogn,
                "features": features,
                "targets": targets,
                "device": str(device),
                "data_fraction": data_fraction,
            },
        )

    logging.info("Training with features: %s", features)
    logging.info("Training targets: %s", targets)

    for target in targets:
        logging.info("==== Training NN for %s ====", target)
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        # Subsample training data if requested
        if data_fraction < 1.0:
            original_train_size = len(X_train)
            n_train_subset = int(len(X_train) * data_fraction)
            indices = np.random.choice(len(X_train), n_train_subset, replace=False)
            X_train = X_train.iloc[indices] if isinstance(X_train, pd.DataFrame) else X_train[indices]
            y_train = y_train.iloc[indices] if isinstance(y_train, pd.Series) else y_train[indices]
            logging.info("Subsampled training data for %s: %d / %d samples (%.1f%%)", 
                         target, n_train_subset, original_train_size, 
                         data_fraction * 100)

        if use_smogn:
            X_train, y_train = apply_smogn_resampling(X_train, y_train, target)

        X_train_norm, X_test_norm, feature_scaler = normalize_features(X_train, X_test)
        y_train_np = np.asarray(y_train).reshape(-1, 1)
        y_test_np = np.asarray(y_test).reshape(-1, 1)
        y_train_norm, y_test_norm, target_scaler = normalize_targets(y_train_np, y_test_np)

        model, train_losses, test_losses, preds_norm = train_model(
            X_train_norm, y_train_norm, X_test_norm, y_test_norm, epochs, lr, device,
            use_wandb=use_wandb, target=target
        )

        target_std = np.array(target_scaler["std"])
        target_mean = np.array(target_scaler["mean"])
        preds = preds_norm * target_std + target_mean
        metrics = compute_regression_metrics(y_test_np, preds)

        target_dir = output_dir / run_name / target
        ensure_dir(target_dir)
        torch.save(model.state_dict(), target_dir / "model.pt")
        save_json(
            {
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "metrics": metrics,
            },
            target_dir / "training_summary.json",
        )
        np.savez(
            target_dir / "predictions.npz",
            y_true=y_test_np,
            y_pred=preds,
            y_pred_norm=preds_norm,
        )
        logging.info("Saved artifacts to %s", target_dir)

        # Log metrics to WandB
        if use_wandb and wandb is not None:
            wandb.log({
                f"{target}/MSE": metrics["MSE"],
                f"{target}/RMSE": metrics["RMSE"],
                f"{target}/Pearson": metrics["Pearson"],
            })

    # Finish WandB run after all targets are trained
    if use_wandb and wandb is not None:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NN regressors for galaxy properties.")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_PROCESSED_PARQUET,
        help="Path to processed parquet produced by preprocess.py",
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
        "--use-smogn",
        action="store_true",
        help="Apply SMOGN resampling on the training split.",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.DEFAULT_TEST_SIZE,
        help="Fraction reserved for evaluation.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.DEFAULT_RANDOM_STATE, help="Random seed."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device identifier (defaults to CUDA, confirm before CPU fallback).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.NN_MODEL_DIR,
        help="Directory for saved weights, scalers, and metrics.",
    )
    parser.add_argument(
        "--run-name",
        default="default",
        help="Name of the NN run (e.g., raw, smogn) to organize artifacts.",
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

    run_training(
        df=df,
        features=features,
        targets=targets,
        use_smogn=args.use_smogn,
        test_size=args.test_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        use_wandb=args.use_wandb,
        data_fraction=args.data_fraction,
    )


if __name__ == "__main__":
    main()
