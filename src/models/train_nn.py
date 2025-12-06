"""
Train feed-forward neural networks (one per galaxy property) against the
processed halo dataset. Supports optional SMOGN resampling to emphasize
tail regions. Saves model weights, scalers, and evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src import config
from src.utils.io import ensure_dir, save_json

try:
    import smogn
except ImportError:  # pragma: no cover
    smogn = None


TARGET_COLUMNS = ["SM", "SFR", "Colour", "SR"]
FEATURE_COLUMNS = ["M_h", "R_h", "V_h"]

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


def resolve_device(device_arg: str) -> torch.device:
    """Prefer CUDA and fall back to CPU only with user confirmation."""
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


def normalize_data(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0, 1)
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    return train_norm.values, test_norm.values, {"mean": mean.tolist(), "std": std.tolist()}


def prepare_targets(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    mean = np.atleast_1d(train.mean(axis=0))
    std = np.atleast_1d(train.std(axis=0))
    std = np.where(std == 0, 1, std)
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    return train_norm, test_norm, {"mean": mean.tolist(), "std": std.tolist()}


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
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = criterion(test_pred, y_test_tensor).item()
            test_losses.append(float(test_loss))

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()
    return model, train_losses, test_losses, preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    pearson = np.corrcoef(y_true.squeeze(), y_pred.squeeze())[0, 1]
    return {"MSE": float(mse), "RMSE": rmse, "Pearson": float(pearson)}


def run_training(
    df: pd.DataFrame,
    targets: List[str],
    use_smogn: bool,
    test_size: float,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    run_name: str,
    seed: int,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    for target in targets:
        logging.info("==== Training NN for %s ====", target)
        X = df[FEATURE_COLUMNS]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        if use_smogn:
            X_train, y_train = apply_smogn_resampling(X_train, y_train, target)

        X_train_norm, X_test_norm, feature_scaler = normalize_data(X_train, X_test)
        y_train_np = np.asarray(y_train).reshape(-1, 1)
        y_test_np = np.asarray(y_test).reshape(-1, 1)
        y_train_norm, y_test_norm, target_scaler = prepare_targets(y_train_np, y_test_np)

        model, train_losses, test_losses, preds_norm = train_model(
            X_train_norm, y_train_norm, X_test_norm, y_test_norm, epochs, lr, device
        )

        target_std = np.array(target_scaler["std"])
        target_mean = np.array(target_scaler["mean"])
        preds = preds_norm * target_std + target_mean
        metrics = compute_metrics(y_test_np, preds)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NN regressors for galaxy properties.")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_PROCESSED_PARQUET,
        help="Path to processed parquet produced by preprocess.py",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TARGET_COLUMNS,
        choices=TARGET_COLUMNS,
        help="Target columns to train on.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    df = pd.read_parquet(args.data)
    device = resolve_device(args.device)
    ensure_dir(args.output_dir / args.run_name)

    run_training(
        df=df,
        targets=args.targets,
        use_smogn=args.use_smogn,
        test_size=args.test_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
