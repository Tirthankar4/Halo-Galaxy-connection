"""
Hyperparameter optimization for neural networks using Optuna.
Optimizes hidden_size, learning_rate, and epochs for each galaxy property separately.
Supports multiple optimization metrics: RMSE, PCC, K-S statistic, and Wasserstein distance.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from src import config
from src.models.train_nn import (
    TARGET_COLUMNS,
    FEATURE_COLUMNS,
    SimpleMLP,
    apply_smogn_resampling,
    compute_metrics,
    normalize_data,
    prepare_targets,
    resolve_device,
)
from src.utils.io import ensure_dir, save_json


def train_model_with_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[SimpleMLP, float]:
    """
    Train a model with specified hyperparameters and return the model and train RMSE.
    Used during optimization (lightweight, no loss history tracking).
    
    Args:
        X_train: Training features
        y_train: Training targets
        hidden_size: Hidden layer size
        epochs: Number of training epochs
        lr: Learning rate
        device: Torch device
        
    Returns:
        Tuple of (trained model, train RMSE)
    """
    input_size = X_train.shape[1]
    model = SimpleMLP(input_size, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Compute RMSE on training set
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).cpu().numpy()
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    return model, train_rmse


def train_model_with_loss_history(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[SimpleMLP, List[float], List[float], np.ndarray]:
    """
    Train a model with specified hyperparameters and return model, loss history, and predictions.
    Used for final best model training (tracks full loss history).
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        hidden_size: Hidden layer size
        epochs: Number of training epochs
        lr: Learning rate
        device: Torch device
        
    Returns:
        Tuple of (trained model, train_losses, test_losses, test_predictions)
    """
    input_size = X_train.shape[1]
    model = SimpleMLP(input_size, hidden_size=hidden_size).to(device)
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


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """
    Compute a specified metric between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric: Metric name ('RMSE', 'PCC', 'K-S', or 'Wasserstein')
        
    Returns:
        Metric value
    """
    if metric == "RMSE":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric == "PCC":
        pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        return float(pcc)
    elif metric == "K-S":
        ks_stat, _ = ks_2samp(y_true.flatten(), y_pred.flatten())
        return float(ks_stat)
    elif metric == "Wasserstein":
        return float(wasserstein_distance(y_true.flatten(), y_pred.flatten()))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    metric: str,
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective function for hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training targets
        device: Torch device
        metric: Optimization metric ('RMSE', 'PCC', 'K-S', or 'Wasserstein')
        
    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 200, 1000)

        # 5-fold cross-validation on the (normalized) training data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_metrics: List[float] = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model, _ = train_model_with_hyperparams(
                X_tr, y_tr, hidden_size, epochs, lr, device
            )

            model.eval()
            with torch.no_grad():
                val_pred = (
                    model(torch.tensor(X_val, dtype=torch.float32).to(device))
                    .cpu()
                    .numpy()
                )
            val_metric = compute_metric(y_val, val_pred, metric)
            val_metrics.append(val_metric)

        # Return average validation metric across folds
        return float(np.mean(val_metrics))
    
    return objective


def optimize_and_save_best_model(
    df: pd.DataFrame,
    target: str,
    use_smogn: bool,
    test_size: float,
    n_trials: int,
    device: torch.device,
    output_dir: Path,
    run_name: str,
    seed: int,
    metric: str,
) -> None:
    """
    Optimize hyperparameters for a single target and save the best model.
    
    Args:
        df: DataFrame with features and targets
        target: Target column name
        use_smogn: Whether to use SMOGN resampling
        test_size: Test split size
        n_trials: Number of Optuna trials
        device: Torch device
        output_dir: Output directory
        run_name: Run name for saved models
        seed: Random seed
        metric: Optimization metric ('RMSE', 'PCC', 'K-S', or 'Wasserstein')
    """
    logging.info("==== Optimizing hyperparameters for %s ====", target)
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Prepare data
    X = df[FEATURE_COLUMNS]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    if use_smogn:
        X_train, y_train = apply_smogn_resampling(X_train, y_train, target)
    
    # Normalize data (fit scalers on training split only)
    X_train_norm, X_test_norm, feature_scaler = normalize_data(X_train, X_test)
    y_train_np = np.asarray(y_train).reshape(-1, 1)
    y_test_np = np.asarray(y_test).reshape(-1, 1)
    y_train_norm, y_test_norm, target_scaler = prepare_targets(y_train_np, y_test_np)
    
    # Determine optimization direction based on metric
    # RMSE, K-S, and Wasserstein should be minimized; PCC should be maximized
    direction = "maximize" if metric == "PCC" else "minimize"
    
    # Create Optuna study
    study = optuna.create_study(
        direction=direction,
        study_name=f"nn_optimization_{target}_{metric}",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    
    # Create objective function
    objective = create_objective(X_train_norm, y_train_norm, device, metric)
    
    # Optimize
    logging.info("Running %d trials for %s...", n_trials, target)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best hyperparameters
    best_params = study.best_params
    best_hidden_size = best_params["hidden_size"]
    best_lr = best_params["lr"]
    best_epochs = best_params["epochs"]
    best_metric_value = study.best_value
    
    logging.info(
        "Best hyperparameters for %s: hidden_size=%d, lr=%.6f, epochs=%d, train_%s=%.6f",
        target, best_hidden_size, best_lr, best_epochs, metric, best_metric_value
    )
    
    # Train best model using only the training split.
    # We further split the training data into an inner train/validation pair
    # for tracking losses, but the held-out test split remains unused here.
    logging.info("Training best model for %s with optimal hyperparameters...", target)
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_train_norm,
        y_train_norm,
        test_size=0.2,
        random_state=seed,
    )
    best_model, train_losses, val_losses, _ = train_model_with_loss_history(
        X_tr_final,
        y_tr_final,
        X_val_final,
        y_val_final,
        best_hidden_size,
        best_epochs,
        best_lr,
        device,
    )

    # After training is complete, evaluate once on the held-out test split.
    best_model.eval()
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_norm = best_model(X_test_tensor).cpu().numpy()

    # Denormalize predictions to original target space
    target_std = np.array(target_scaler["std"])
    target_mean = np.array(target_scaler["mean"])
    preds = preds_norm * target_std + target_mean
    
    # Compute metrics on the held-out test split (never seen during
    # optimization or training, only used here for final evaluation).
    metrics = compute_metrics(y_test_np, preds)
    
    # Save model and artifacts
    target_dir = output_dir / run_name / target
    ensure_dir(target_dir)
    
    torch.save(best_model.state_dict(), target_dir / "model.pt")
    
    # Save training summary with best hyperparameters
    save_json(
        {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "train_losses": train_losses,
            "test_losses": val_losses,
            "metrics": metrics,
            "best_hyperparameters": {
                "hidden_size": best_hidden_size,
                "lr": best_lr,
                "epochs": best_epochs,
            },
            "optimization_metric": metric,
            f"best_train_{metric.lower().replace('-', '_')}": float(best_metric_value),
            "n_trials": n_trials,
        },
        target_dir / "training_summary.json",
    )
    
    # Save predictions
    np.savez(
        target_dir / "predictions.npz",
        y_true=y_test_np,
        y_pred=preds,
        y_pred_norm=preds_norm,
    )
    
    # Save Optuna study
    study_path = target_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    logging.info("Saved best model and artifacts to %s", target_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize NN hyperparameters using Optuna."
    )
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
        help="Target columns to optimize (default: all).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per target.",
    )
    parser.add_argument(
        "--use-smogn",
        action="store_true",
        help="Apply SMOGN resampling on the training split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.DEFAULT_TEST_SIZE,
        help="Fraction reserved for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_RANDOM_STATE,
        help="Random seed.",
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
        default="optuna_best",
        help="Name of the NN run (e.g., optuna_best, optuna_best_smogn) to organize artifacts.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="RMSE",
        choices=["RMSE", "PCC", "K-S", "Wasserstein"],
        help="Optimization metric to use: RMSE (minimize), PCC (maximize), K-S (minimize), or Wasserstein (minimize).",
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
    
    logging.info("Starting hyperparameter optimization for targets: %s", args.targets)
    logging.info("Using SMOGN: %s", args.use_smogn)
    logging.info("Run name: %s", args.run_name)
    logging.info("Optimization metric: %s", args.metric)
    logging.info("Number of trials per target: %d", args.n_trials)
    
    # Optimize each target separately
    for target in args.targets:
        optimize_and_save_best_model(
            df=df,
            target=target,
            use_smogn=args.use_smogn,
            test_size=args.test_size,
            n_trials=args.n_trials,
            device=device,
            output_dir=args.output_dir,
            run_name=args.run_name,
            seed=args.seed,
            metric=args.metric,
        )
    
    logging.info("Hyperparameter optimization completed for all targets.")


if __name__ == "__main__":
    main()

