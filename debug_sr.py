"""
Debug script for SR predictions - Priority 0
Runs three diagnostic tests to identify issues with SR predictions.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from src import config

# Constants
C_LIGHT = 3.0 * 10**8  # Speed of light in m/s

print("=" * 80)
print("SR DEBUGGING TESTS - Priority 0")
print("=" * 80)
print()

# ============================================================================
# Test 1: Verify data ranges
# ============================================================================
print("1. VERIFY DATA RANGES")
print("-" * 80)

# Load raw SR from HDF5
h5_path = config.DEFAULT_HALO_HDF5
print(f"Loading raw data from: {h5_path}")
with h5py.File(h5_path, "r") as f:
    id_r = f["Group/GroupFirstSub"][:]
    sm = f["Subhalo/SubhaloMassType"][:, 4] * 1e10
    sr_raw = f["Subhalo/SubhaloHalfmassRadType"][:, 4]
    id_g = np.array(f["Subhalo/SubhaloGrNr"])

# Filter to match preprocessing logic
keep_halos = np.where(id_r != -1)[0]
keep_gals = np.where(sm > 0)[0]
sr_raw_filtered = sr_raw[keep_gals]
id_g_filtered = id_g[keep_gals]

# Create dataframe and drop duplicates (matching preprocessing)
gals_raw = pd.DataFrame({"SR": sr_raw_filtered, "ID": id_g_filtered})
gals_raw = gals_raw.drop_duplicates(subset=["ID"], keep="first")

print(f"Raw SR from catalog: min = {gals_raw['SR'].min():.6f}, max = {gals_raw['SR'].max():.6f}")

# Load processed SR from parquet
parquet_path = config.DEFAULT_PROCESSED_PARQUET
print(f"Loading processed data from: {parquet_path}")
df_processed = pd.read_parquet(parquet_path)
print(f"After preprocessing: min = {df_processed['SR'].min():.6f}, max = {df_processed['SR'].max():.6f}")

# Load test set true SR and predicted SR
# Using optuna_best model for SR
model_path = config.NN_MODEL_DIR / "optuna_best" / "SR" / "predictions.npz"
print(f"Loading predictions from: {model_path}")
pred_data = np.load(model_path)
y_true_SR = pred_data["y_true"].flatten()
y_pred_SR = pred_data["y_pred"].flatten()

print(f"True SR in test set: min = {y_true_SR.min():.6f}, max = {y_true_SR.max():.6f}")
print(f"Predicted SR: min = {y_pred_SR.min():.6f}, max = {y_pred_SR.max():.6f}")
print()

# ============================================================================
# Test 2: Check if you're inverse-transforming
# ============================================================================
print("2. CHECK IF YOU'RE INVERSE-TRANSFORMING")
print("-" * 80)

# Check if predictions are in log space
# If SR was log-transformed, predictions might need: 10 ** y_pred_SR_log
# If SR was arcsinh-transformed, predictions might need: np.sinh(y_pred_SR_asinh) * 0.1

# Check if predictions look like they're in log space (negative values, small range)
if y_pred_SR.min() < 0 or (y_pred_SR.max() - y_pred_SR.min()) < 2:
    print("WARNING: Predictions might be in log space!")
    print("  - Predictions have negative values or very small range")
    print("  - Try inverse transform: y_pred_SR_linear = 10 ** y_pred_SR_log")
else:
    print("Predictions appear to be in linear space (no obvious log transform needed)")

# Check if predictions look like they're in arcsinh space
# arcsinh typically produces values in a compressed range
if abs(y_pred_SR.max()) < 5 and abs(y_pred_SR.min()) < 5:
    print("  - Predictions might be in arcsinh space!")
    print("  - Try inverse transform: y_pred_SR_linear = np.sinh(y_pred_SR_asinh) * 0.1")

# Compare ranges
print(f"\nRange comparison:")
print(f"  Raw SR range: [{gals_raw['SR'].min():.6f}, {gals_raw['SR'].max():.6f}]")
print(f"  Processed SR range: [{df_processed['SR'].min():.6f}, {df_processed['SR'].max():.6f}]")
print(f"  True SR (test) range: [{y_true_SR.min():.6f}, {y_true_SR.max():.6f}]")
print(f"  Predicted SR range: [{y_pred_SR.min():.6f}, {y_pred_SR.max():.6f}]")

# Check if processed SR matches raw SR (SR should NOT be transformed in preprocessing)
if np.allclose(df_processed['SR'].min(), gals_raw['SR'].min(), rtol=1e-3) and \
   np.allclose(df_processed['SR'].max(), gals_raw['SR'].max(), rtol=1e-3):
    print("\n✓ Processed SR matches raw SR (no transformation applied, as expected)")
else:
    print("\n⚠ WARNING: Processed SR does NOT match raw SR!")
    print("  This suggests SR was transformed during preprocessing (unexpected)")

print()

# ============================================================================
# Test 3: Inspect individual predictions
# ============================================================================
print("3. INSPECT INDIVIDUAL PREDICTIONS")
print("-" * 80)

# Load test set halos to get M_h
# We need to match the test split used during training
# Since we can't easily reconstruct the exact test split, we'll load the full dataset
# and note that predictions correspond to a subset

# Load full dataset to get halo masses
df_full = pd.read_parquet(parquet_path)
halos_full = df_full[["M_h", "R_h", "V_h"]].values

# For the test set, we need to match indices
# The predictions were saved with y_test, so we need to figure out which halos they correspond to
# Since train_test_split uses random_state=42, we can reconstruct it
from sklearn.model_selection import train_test_split

# Reconstruct the test split
_, df_test, _, _ = train_test_split(
    df_full, df_full["SR"], test_size=0.2, random_state=42
)

# Get M_h for test set
M_h_test = df_test["M_h"].values

# Make sure we have the same number of predictions as test samples
n_samples = min(10, len(y_true_SR), len(y_pred_SR), len(M_h_test))
print(f"Showing first {n_samples} predictions:\n")

for i in range(n_samples):
    print(f"Halo {i}: True SR = {y_true_SR[i]:.6f}, "
          f"Pred SR = {y_pred_SR[i]:.6f}, "
          f"M_h = {M_h_test[i]:.6f}")

print()
print("=" * 80)
print("DEBUGGING TESTS COMPLETE")
print("=" * 80)

