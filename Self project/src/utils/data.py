"""Data loading and preprocessing utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SimpleScaler:
    """Simple scaler for inverse transform compatibility."""
    
    def __init__(self, mean: list | np.ndarray, scale: list | np.ndarray):
        self.mean = np.array(mean)
        self.scale = np.array(scale)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.scale + self.mean


def normalize_features(
    train: pd.DataFrame | np.ndarray,
    test: pd.DataFrame | np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Normalize features using training set statistics."""
    if isinstance(train, pd.DataFrame):
        train = train.values
    if isinstance(test, pd.DataFrame):
        test = test.values
        
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    
    return train_norm, test_norm, {"mean": mean.tolist(), "std": std.tolist()}


def normalize_targets(
    train: np.ndarray,
    test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Normalize targets using training set statistics."""
    mean = np.atleast_1d(train.mean(axis=0))
    std = np.atleast_1d(train.std(axis=0))
    std = np.where(std == 0, 1, std)
    
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    
    return train_norm, test_norm, {"mean": mean.tolist(), "std": std.tolist()}

