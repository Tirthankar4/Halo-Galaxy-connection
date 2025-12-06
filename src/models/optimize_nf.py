"""
Hyperparameter optimization for normalizing flow using Optuna.
Optimizes learning_rate, number of flow layers, hidden dimensions, and weight decay.
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
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.models.train_nf import (
    TARGET_COLUMNS,
    FEATURE_COLUMNS,
    resolve_device,
    compute_metrics,
    generate_predictions,
)
from src.utils.io import ensure_dir, save_json


def build_flow(
    halo_dim: int,
    gal_dim: int,
    n_halo_layers: int = 2,
    n_gal_layers: int = 1,
    hidden_dim: int = 64,
    device: torch.device = None,
) -> Tuple[dist.Distribution, dist.ConditionalDistribution, torch.nn.ModuleList]:
    """
    Build a normalizing flow with configurable architecture.
    
    Args:
        halo_dim: Dimension of halo features
        gal_dim: Dimension of galaxy properties
        n_halo_layers: Number of transform layers for halo distribution
        n_gal_layers: Number of transform layers for conditional galaxy distribution
        hidden_dim: Hidden dimension for internal neural networks in transforms
        device: Device to create tensors on (if None, uses CPU)
        
    Returns:
        Tuple of (halo distribution, conditional galaxy distribution, module list)
    """
    if device is None:
        device = torch.device("cpu")
    base = dist.Normal(torch.zeros(halo_dim, device=device), torch.ones(halo_dim, device=device) * 0.2).to_event(1)
    
    # Build halo transforms: alternate spline and affine_coupling
    halo_transforms = []
    for i in range(n_halo_layers):
        if i % 2 == 0:
            # Use spline transform with hidden_dims if supported
            try:
                halo_transforms.append(T.spline(halo_dim, hidden_dims=[hidden_dim, hidden_dim]))
            except TypeError:
                # Fallback if hidden_dims not supported in this Pyro version
                halo_transforms.append(T.spline(halo_dim))
        else:
            try:
                halo_transforms.append(T.affine_coupling(halo_dim, hidden_dims=[hidden_dim, hidden_dim]))
            except TypeError:
                halo_transforms.append(T.affine_coupling(halo_dim))
    
    dist_x1 = dist.TransformedDistribution(base, halo_transforms)
    
    # Build conditional galaxy transforms: stack conditional_spline
    gal_transforms = []
    for _ in range(n_gal_layers):
        try:
            gal_transforms.append(
                T.conditional_spline(
                    gal_dim,
                    context_dim=halo_dim,
                    hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
                )
            )
        except TypeError:
            # Fallback if hidden_dims not supported
            gal_transforms.append(T.conditional_spline(gal_dim, context_dim=halo_dim))
    
    cond_base = dist.Normal(torch.zeros(gal_dim, device=device), torch.ones(gal_dim, device=device) * 0.2).to_event(1)
    dist_x2_given_x1 = dist.ConditionalTransformedDistribution(cond_base, gal_transforms)
    
    # Collect all modules
    all_modules = halo_transforms + gal_transforms
    modules = torch.nn.ModuleList(all_modules)
    
    return dist_x1, dist_x2_given_x1, modules


def train_flow_with_hyperparams(
    halos_train: torch.Tensor,
    gals_train: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float,
    n_halo_layers: int,
    n_gal_layers: int,
    hidden_dim: int,
    device: torch.device,
) -> Tuple[dist.Distribution, dist.ConditionalDistribution, torch.nn.ModuleList, float]:
    """
    Train a flow model with specified hyperparameters and return the model and train loss.
    Used during optimization (lightweight, no loss history tracking).
    
    Args:
        halos_train: Training halo features
        gals_train: Training galaxy properties
        steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        n_halo_layers: Number of halo transform layers
        n_gal_layers: Number of galaxy transform layers
        hidden_dim: Hidden dimension for transforms
        device: Torch device
        
    Returns:
        Tuple of (halo distribution, conditional galaxy distribution, modules, final train loss)
    """
    halo_dim = halos_train.shape[1]
    gal_dim = gals_train.shape[1]
    
    dist_x1, dist_x2_given_x1, modules = build_flow(
        halo_dim, gal_dim, n_halo_layers, n_gal_layers, hidden_dim, device
    )
    modules = modules.to(device)
    
    optimizer = torch.optim.Adam(modules.parameters(), lr=lr, weight_decay=weight_decay)
    
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
    
    # Compute final loss on training set
    modules.eval()
    with torch.no_grad():
        final_ln_p_x1 = dist_x1.log_prob(halos_train)
        final_ln_p_x2_given_x1 = dist_x2_given_x1.condition(halos_train.detach()).log_prob(
            gals_train.detach()
        )
        final_loss = -(final_ln_p_x1 + final_ln_p_x2_given_x1).mean().item()
    
    return dist_x1, dist_x2_given_x1, modules, final_loss


def train_flow_with_loss_history(
    halos_train: torch.Tensor,
    gals_train: torch.Tensor,
    halos_val: torch.Tensor,
    gals_val: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float,
    n_halo_layers: int,
    n_gal_layers: int,
    hidden_dim: int,
    device: torch.device,
) -> Tuple[dist.Distribution, dist.ConditionalDistribution, torch.nn.ModuleList, List[float], List[float]]:
    """
    Train a flow model with specified hyperparameters and return model, loss history, and predictions.
    Used for final best model training (tracks full loss history).
    
    Args:
        halos_train: Training halo features
        gals_train: Training galaxy properties
        halos_val: Validation halo features
        gals_val: Validation galaxy properties
        steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        n_halo_layers: Number of halo transform layers
        n_gal_layers: Number of galaxy transform layers
        hidden_dim: Hidden dimension for transforms
        device: Torch device
        
    Returns:
        Tuple of (halo distribution, conditional galaxy distribution, modules, train_losses, val_losses)
    """
    halo_dim = halos_train.shape[1]
    gal_dim = gals_train.shape[1]
    
    dist_x1, dist_x2_given_x1, modules = build_flow(
        halo_dim, gal_dim, n_halo_layers, n_gal_layers, hidden_dim, device
    )
    modules = modules.to(device)
    
    optimizer = torch.optim.Adam(modules.parameters(), lr=lr, weight_decay=weight_decay)
    
    halos_train = halos_train.to(device)
    gals_train = gals_train.to(device)
    halos_val = halos_val.to(device)
    gals_val = gals_val.to(device)
    
    train_losses, val_losses = [], []
    
    for step in range(steps + 1):
        # Training step
        modules.train()
        optimizer.zero_grad()
        ln_p_x1 = dist_x1.log_prob(halos_train)
        ln_p_x2_given_x1 = dist_x2_given_x1.condition(halos_train.detach()).log_prob(
            gals_train.detach()
        )
        loss = -(ln_p_x1 + ln_p_x2_given_x1).mean()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))
        dist_x1.clear_cache()
        dist_x2_given_x1.clear_cache()
        
        # Validation step
        modules.eval()
        with torch.no_grad():
            val_ln_p_x1 = dist_x1.log_prob(halos_val)
            val_ln_p_x2_given_x1 = dist_x2_given_x1.condition(halos_val.detach()).log_prob(
                gals_val.detach()
            )
            val_loss = -(val_ln_p_x1 + val_ln_p_x2_given_x1).mean()
            val_losses.append(float(val_loss.item()))
        
        if step % 100 == 0:
            logging.info("Step %d / %d, train_loss %.4f, val_loss %.4f", step, steps, loss.item(), val_loss.item())
    
    return dist_x1, dist_x2_given_x1, modules, train_losses, val_losses


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


def compute_metric_per_property(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> float:
    """
    Compute metric per property and return mean across properties.
    
    Args:
        y_true: True values (n_samples, n_properties)
        y_pred: Predicted values (n_samples, n_properties)
        metric: Metric name
        
    Returns:
        Mean metric value across all properties
    """
    metrics_per_prop = []
    for prop_idx in range(y_true.shape[1]):
        prop_metric = compute_metric(y_true[:, prop_idx], y_pred[:, prop_idx], metric)
        metrics_per_prop.append(prop_metric)
    return float(np.mean(metrics_per_prop))


def create_objective(
    halos_train: np.ndarray,
    gals_train: np.ndarray,
    device: torch.device,
    metric: str,
    steps: int,
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective function for hyperparameter optimization.
    
    Args:
        halos_train: Training halo features
        gals_train: Training galaxy properties
        device: Torch device
        metric: Optimization metric ('RMSE', 'PCC', 'K-S', or 'Wasserstein')
        steps: Number of training steps per trial
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        n_halo_layers = trial.suggest_int("n_halo_layers", 2, 8)
        n_gal_layers = trial.suggest_int("n_gal_layers", 1, 6)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        # 5-fold cross-validation on the training data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_metrics: List[float] = []
        
        for train_idx, val_idx in kf.split(halos_train):
            halos_tr = halos_train[train_idx]
            halos_val = halos_train[val_idx]
            gals_tr = gals_train[train_idx]
            gals_val = gals_train[val_idx]
            
            # Convert to tensors
            halos_tr_tensor = torch.tensor(halos_tr, dtype=torch.float32)
            gals_tr_tensor = torch.tensor(gals_tr, dtype=torch.float32)
            halos_val_tensor = torch.tensor(halos_val, dtype=torch.float32)
            gals_val_tensor = torch.tensor(gals_val, dtype=torch.float32)
            
            # Train model
            dist_x1, dist_x2_given_x1, modules, _ = train_flow_with_hyperparams(
                halos_tr_tensor,
                gals_tr_tensor,
                steps,
                lr,
                weight_decay,
                n_halo_layers,
                n_gal_layers,
                hidden_dim,
                device,
            )
            
            # Evaluate on validation set
            modules.eval()
            with torch.no_grad():
                # Generate predictions (mean of samples)
                mean_preds = []
                halos_val_tensor = halos_val_tensor.to(device)
                for halo in halos_val_tensor:
                    halo = halo.unsqueeze(0)
                    samples = dist_x2_given_x1.condition(halo).sample(torch.Size([100]))
                    mean_pred = samples.squeeze(1).mean(dim=0).cpu().numpy()
                    mean_preds.append(mean_pred)
                
                val_pred = np.array(mean_preds)
                val_metric = compute_metric_per_property(gals_val, val_pred, metric)
                val_metrics.append(val_metric)
            
            # Clear cache and cleanup
            dist_x1.clear_cache()
            dist_x2_given_x1.clear_cache()
            del dist_x1, dist_x2_given_x1, modules
            torch.cuda.empty_cache() if device.type == "cuda" else None
        
        # Return average validation metric across folds
        return float(np.mean(val_metrics))
    
    return objective


def optimize_and_save_best_model(
    df: pd.DataFrame,
    test_size: float,
    n_trials: int,
    steps: int,
    device: torch.device,
    output_dir: Path,
    run_name: str,
    seed: int,
    metric: str,
    n_samples: int,
) -> None:
    """
    Optimize hyperparameters for NF model and save the best model.
    
    Args:
        df: DataFrame with features and targets
        test_size: Test split size
        n_trials: Number of Optuna trials
        steps: Number of training steps
        device: Torch device
        output_dir: Output directory
        run_name: Run name for saved models
        seed: Random seed
        metric: Optimization metric ('RMSE', 'PCC', 'K-S', or 'Wasserstein')
        n_samples: Number of posterior samples for predictions
    """
    logging.info("==== Optimizing hyperparameters for NF model ====")
    
    # Set random seeds
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Prepare data
    halos = df[FEATURE_COLUMNS].values
    gals = df[TARGET_COLUMNS].values
    
    halos_train, halos_test, gals_train, gals_test = train_test_split(
        halos, gals, test_size=test_size, random_state=seed
    )
    
    # Normalize data
    halo_scaler = StandardScaler()
    gal_scaler = StandardScaler()
    halos_train_scaled = halo_scaler.fit_transform(halos_train)
    halos_test_scaled = halo_scaler.transform(halos_test)
    gals_train_scaled = gal_scaler.fit_transform(gals_train)
    gals_test_scaled = gal_scaler.transform(gals_test)
    
    # Determine optimization direction based on metric
    # RMSE, K-S, and Wasserstein should be minimized; PCC should be maximized
    direction = "maximize" if metric == "PCC" else "minimize"
    
    # Create Optuna study
    study = optuna.create_study(
        direction=direction,
        study_name=f"nf_optimization_{metric}",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    
    # Create objective function
    objective = create_objective(
        halos_train_scaled,
        gals_train_scaled,
        device,
        metric,
        steps,
    )
    
    # Optimize
    logging.info("Running %d trials for NF model...", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best hyperparameters
    best_params = study.best_params
    best_lr = best_params["lr"]
    best_n_halo_layers = best_params["n_halo_layers"]
    best_n_gal_layers = best_params["n_gal_layers"]
    best_hidden_dim = best_params["hidden_dim"]
    best_weight_decay = best_params["weight_decay"]
    best_metric_value = study.best_value
    
    logging.info(
        "Best hyperparameters: lr=%.6f, n_halo_layers=%d, n_gal_layers=%d, "
        "hidden_dim=%d, weight_decay=%.6f, train_%s=%.6f",
        best_lr,
        best_n_halo_layers,
        best_n_gal_layers,
        best_hidden_dim,
        best_weight_decay,
        metric,
        best_metric_value,
    )
    
    # Train best model using only the training split.
    # We further split the training data into an inner train/validation pair
    # for tracking losses, but the held-out test split remains unused here.
    logging.info("Training best model with optimal hyperparameters...")
    halos_tr_final, halos_val_final, gals_tr_final, gals_val_final = train_test_split(
        halos_train_scaled,
        gals_train_scaled,
        test_size=0.2,
        random_state=seed,
    )
    
    halos_tr_final_tensor = torch.tensor(halos_tr_final, dtype=torch.float32)
    gals_tr_final_tensor = torch.tensor(gals_tr_final, dtype=torch.float32)
    halos_val_final_tensor = torch.tensor(halos_val_final, dtype=torch.float32)
    gals_val_final_tensor = torch.tensor(gals_val_final, dtype=torch.float32)
    
    best_dist_x1, best_dist_x2_given_x1, best_modules, train_losses, val_losses = (
        train_flow_with_loss_history(
            halos_tr_final_tensor,
            gals_tr_final_tensor,
            halos_val_final_tensor,
            gals_val_final_tensor,
            steps,
            best_lr,
            best_weight_decay,
            best_n_halo_layers,
            best_n_gal_layers,
            best_hidden_dim,
            device,
        )
    )
    
    # After training is complete, evaluate once on the held-out test split.
    best_modules.eval()
    halos_test_tensor = torch.tensor(halos_test_scaled, dtype=torch.float32).to(device)
    predictions = generate_predictions(
        best_dist_x2_given_x1, halos_test_tensor, gal_scaler, n_samples, device
    )
    
    y_true = gal_scaler.inverse_transform(gals_test_scaled)
    
    # Compute metrics on the held-out test split per property
    metrics = {}
    for idx, prop in enumerate(TARGET_COLUMNS):
        metrics[prop] = compute_metrics(y_true[:, idx], predictions["mean"][:, idx])
    
    # Save model and artifacts
    run_dir = output_dir / run_name
    ensure_dir(run_dir)
    
    torch.save(best_modules.state_dict(), run_dir / "flow_state.pt")
    
    # Save training summary with best hyperparameters
    save_json(
        {
            "halo_scaler": {"mean": halo_scaler.mean_.tolist(), "scale": halo_scaler.scale_.tolist()},
            "gal_scaler": {"mean": gal_scaler.mean_.tolist(), "scale": gal_scaler.scale_.tolist()},
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics,
            "best_hyperparameters": {
                "lr": best_lr,
                "n_halo_layers": best_n_halo_layers,
                "n_gal_layers": best_n_gal_layers,
                "hidden_dim": best_hidden_dim,
                "weight_decay": best_weight_decay,
            },
            "optimization_metric": metric,
            f"best_train_{metric.lower().replace('-', '_')}": float(best_metric_value),
            "n_trials": n_trials,
            "steps": steps,
            "train_size": int(len(halos_train)),
            "test_size": int(len(halos_test)),
        },
        run_dir / "training_summary.json",
    )
    
    # Save predictions
    np.savez(
        run_dir / "predictions.npz",
        y_true=y_true,
        mean=predictions["mean"],
        random=predictions["random"],
        posterior=predictions["posterior"],
    )
    
    # Save Optuna study
    study_path = run_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    logging.info("Saved best model and artifacts to %s", run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize NF hyperparameters using Optuna."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_PROCESSED_PARQUET,
        help="Path to processed parquet produced by preprocess.py",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.DEFAULT_TEST_SIZE,
        help="Fraction reserved for evaluation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1300,
        help="Number of training steps per trial.",
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
        default=config.NF_MODEL_DIR,
        help="Directory for saved weights, scalers, and metrics.",
    )
    parser.add_argument(
        "--run-name",
        default="optuna_best",
        help="Name of the NF run (e.g., optuna_best) to organize artifacts.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="RMSE",
        choices=["RMSE", "PCC", "K-S", "Wasserstein"],
        help="Optimization metric to use: RMSE (minimize), PCC (maximize), K-S (minimize), or Wasserstein (minimize).",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples per halo for cached predictions.",
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
    
    logging.info("Starting hyperparameter optimization for NF model")
    logging.info("Optimization metric: %s", args.metric)
    logging.info("Number of trials: %d", args.n_trials)
    logging.info("Training steps per trial: %d", args.steps)
    
    optimize_and_save_best_model(
        df=df,
        test_size=args.test_size,
        n_trials=args.n_trials,
        steps=args.steps,
        device=device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        metric=args.metric,
        n_samples=args.posterior_samples,
    )
    
    logging.info("Hyperparameter optimization completed.")


if __name__ == "__main__":
    main()

