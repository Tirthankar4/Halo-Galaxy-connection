# Halo – Galaxy Connection Pipeline

This repository started as a large exploratory notebook (`NN vs NF comparison.ipynb`) that compares neural networks (NN) and normalizing flows (NF) for predicting galaxy properties from halo catalogs. All heavy lifting has now been moved into standalone Python modules so you can rerun individual steps without touching the notebook.

## Project layout

```
src/
  data/preprocess.py     # build processed parquet from the raw TNG HDF5
  models/train_nn.py     # train feed-forward regressors (raw or SMOGN resampled)
  models/train_nf.py     # train the conditional normalizing flow
  models/optimize_nn.py  # hyperparameter optimization with Optuna (NN models)
  models/optimize_nf.py  # hyperparameter optimization with Optuna (NF model)
  plots/visualize.py     # metrics, contour grids, and TARP plots
  plots/nf_visualizer.py # NF-specific visualization suite
data/
  raw/                   # place raw halo/galaxy HDF5 here
  processed/             # cached parquet + stats JSON
models/
  nn/<run>/<target>/     # NN checkpoints, scalers, predictions
  nf/<run>/              # NF weights, scalers, posterior samples
outputs/plots/           # generated figures
docs/notebook_map.md     # map of the legacy notebook cells
```

## Typical workflow

1. **Preprocess once**

   ```bash
   python -m src.data.preprocess \
     --halo-hdf5 data/groups_090.hdf5 \
     --output data/processed/halo_galaxy.parquet \
     --stats-json data/processed/halo_galaxy_stats.json
   ```

2. **Train neural networks**

   ```bash
   # Baseline (raw)
   python -m src.models.train_nn \
     --data data/processed/halo_galaxy.parquet \
     --run-name raw

   # SMOGN-resampled tails
   python -m src.models.train_nn \
     --data data/processed/halo_galaxy.parquet \
     --run-name smogn \
     --use-smogn
   ```

3. **Train the normalizing flow**

   ```bash
   python -m src.models.train_nf \
     --data data/processed/halo_galaxy.parquet \
     --run-name baseline
   ```

4. **Generate plots/metrics** (runs entirely from saved artifacts)

   ```bash
   # RMSE/PCC/KS/Wasserstein comparison
   python -m src.plots.visualize metrics \
     --nn-runs raw smogn \
     --nn-labels "NN Raw" "NN SMOGN" \
     --nf-run baseline

   # 2x2 contour + marginal grids for multiple property pairs
   python -m src.plots.visualize contours \
     --nn-runs raw smogn \
     --nf-run baseline \
     --output-prefix outputs/plots/contours

   # TARP calibration curves (via NF visualizer)
   python -m src.plots.nf_visualizer \
     --data data/processed/halo_galaxy.parquet \
     --nf-run baseline \
     --nn-runs raw smogn \
     --output-dir outputs/plots/nf
   ```

## Script Reference

### Quick Reference

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `src/data/preprocess.py` | Load and preprocess TNG HDF5 catalogs | Parquet file, stats JSON, diagnostic plots |
| `src/models/train_nn.py` | Train feed-forward MLPs per galaxy property | Model weights, scalers, predictions per target |
| `src/models/train_nf.py` | Train conditional normalizing flow | Flow weights, scalers, posterior samples |
| `src/models/optimize_nn.py` | Hyperparameter optimization with Optuna (supports RMSE/PCC/K-S/Wasserstein metrics) | Best models, Optuna study, hyperparameter reports |
| `src/models/optimize_nf.py` | Hyperparameter optimization for NF model using Optuna (optimizes layers, hidden_dim, lr, weight_decay) | Best NF model, Optuna study, hyperparameter reports |
| `src/plots/visualize.py` | NN vs NF comparison plots | Metrics bar charts, contour plots |
| `src/plots/nf_visualizer.py` | NF-specific visualizations | Scatter plots, TARP curves, posterior distributions |

### Common CLI Options

Most scripts support these shared options:

- `--log-level`: Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default: `INFO`
- `--device`: PyTorch device (`cuda` or `cpu`). Default: `cuda` (with auto-fallback prompt if GPU unavailable)
- `--seed`: Random seed for reproducibility. Default: `42`
- `--test-size`: Fraction of data reserved for testing. Default: `0.2`

### Data Processing

#### `src/data/preprocess.py`

**Purpose**: Loads TNG HDF5 halo/galaxy catalogs, merges them, applies log transforms, and generates diagnostic plots.

**Key Arguments**:
- `--halo-hdf5`: Path to input HDF5 file (default: `data/groups_090.hdf5`)
- `--output`: Path to save processed parquet (default: `data/processed/halo_galaxy.parquet`)
- `--stats-json`: Path to save summary statistics JSON (default: `data/processed/halo_galaxy_stats.json`)
- `--no-plots`: Skip generating diagnostic plots
- `--seed`: Random seed for stochastic steps (e.g., SFR zero handling)

**Inputs**: HDF5 file with `Group/` and `Subhalo/` datasets containing halo and galaxy properties.

**Outputs**:
- Processed parquet file with log-scaled features
- Summary statistics JSON
- Diagnostic plots: halo property histograms, galaxy property histograms, correlation heatmap

**Example**:
```bash
python -m src.data.preprocess \
  --halo-hdf5 data/groups_090.hdf5 \
  --output data/processed/halo_galaxy.parquet \
  --stats-json data/processed/halo_galaxy_stats.json
```

### Training Scripts

#### `src/models/train_nn.py`

**Purpose**: Trains feed-forward MLPs (one per galaxy property: SM, SFR, Colour, SR). Supports optional SMOGN resampling to emphasize tail regions.

**Key Arguments**:
- `--data`: Path to processed parquet (default: `data/processed/halo_galaxy.parquet`)
- `--targets`: Target columns to train (default: all `["SM", "SFR", "Colour", "SR"]`)
- `--use-smogn`: Apply SMOGN resampling on training split
- `--epochs`: Number of training epochs (default: `500`)
- `--lr`: Learning rate (default: `1e-3`)
- `--run-name`: Name for this experiment run (default: `default`)
- `--output-dir`: Directory for saved artifacts (default: `models/nn/`)

**Inputs**: Processed parquet file from `preprocess.py`.

**Outputs** (per target property):
- `model.pt`: PyTorch model weights
- `training_summary.json`: Feature/target scalers, loss history, metrics
- `predictions.npz`: Test set predictions (normalized and denormalized)

**Example**:
```bash
# Baseline training
python -m src.models.train_nn \
  --data data/processed/halo_galaxy.parquet \
  --run-name raw \
  --epochs 500

# With SMOGN resampling
python -m src.models.train_nn \
  --data data/processed/halo_galaxy.parquet \
  --run-name smogn \
  --use-smogn \
  --epochs 500
```

#### `src/models/train_nf.py`

**Purpose**: Trains a conditional normalizing flow for joint prediction of all galaxy properties simultaneously.

**Key Arguments**:
- `--data`: Path to processed parquet (default: `data/processed/halo_galaxy.parquet`)
- `--steps`: Number of training steps (default: `1300`)
- `--lr`: Learning rate (default: `1e-2`)
- `--posterior-samples`: Number of posterior samples per halo for cached predictions (default: `1000`)
- `--run-name`: Name for this experiment run (default: `default`)
- `--output-dir`: Directory for saved artifacts (default: `models/nf/`)

**Inputs**: Processed parquet file from `preprocess.py`.

**Outputs**:
- `flow_state.pt`: PyTorch flow transform weights
- `training_summary.json`: Halo/galaxy scalers, loss history, metrics per property
- `predictions.npz`: Test set predictions (mean, random sample, full posterior)

**Example**:
```bash
python -m src.models.train_nf \
  --data data/processed/halo_galaxy.parquet \
  --run-name baseline \
  --steps 1300 \
  --posterior-samples 1000
```

#### `src/models/optimize_nn.py`

**Purpose**: Hyperparameter optimization using Optuna. Optimizes `hidden_size`, `learning_rate`, and `epochs` for each galaxy property separately. Supports multiple optimization metrics: RMSE, PCC (Pearson Correlation Coefficient), K-S statistic (Kolmogorov-Smirnov), and Wasserstein distance.

**Key Arguments**:
- `--data`: Path to processed parquet (default: `data/processed/halo_galaxy.parquet`)
- `--targets`: Target columns to optimize (default: all)
- `--n-trials`: Number of Optuna trials per target (default: `50`)
- `--metric`: Optimization metric to use (default: `RMSE`)
  - `RMSE`: Root Mean Squared Error (minimize)
  - `PCC`: Pearson Correlation Coefficient (maximize)
  - `K-S`: Kolmogorov-Smirnov statistic (minimize)
  - `Wasserstein`: Wasserstein distance (minimize)
- `--use-smogn`: Apply SMOGN resampling on training split
- `--run-name`: Name for this experiment run (default: `optuna_best`)
- `--output-dir`: Directory for saved artifacts (default: `models/nn/`)

**Inputs**: Processed parquet file from `preprocess.py`.

**Outputs** (per target property):
- `model.pt`: Best model weights
- `optuna_study.pkl`: Optuna study object
- `training_summary.json`: Scalers, loss history, metrics, best hyperparameters, optimization metric used
- `predictions.npz`: Test set predictions

**Example**:
```bash
# Optimize using RMSE (default)
python -m src.models.optimize_nn \
  --data data/processed/halo_galaxy.parquet \
  --n-trials 50 \
  --run-name optuna_best \
  --use-smogn

# Optimize using Pearson Correlation Coefficient
python -m src.models.optimize_nn \
  --data data/processed/halo_galaxy.parquet \
  --metric PCC \
  --n-trials 50 \
  --run-name optuna_best_pcc

# Optimize using K-S statistic
python -m src.models.optimize_nn \
  --data data/processed/halo_galaxy.parquet \
  --metric K-S \
  --n-trials 50 \
  --run-name optuna_best_ks

# Optimize using Wasserstein distance
python -m src.models.optimize_nn \
  --data data/processed/halo_galaxy.parquet \
  --metric Wasserstein \
  --n-trials 50 \
  --run-name optuna_best_wasserstein
```

#### `src/models/optimize_nf.py`

**Purpose**: Hyperparameter optimization for Normalizing Flow model using Optuna. Optimizes `learning_rate`, `n_halo_layers`, `n_gal_layers`, `hidden_dim`, and `weight_decay` for the joint prediction of all galaxy properties. Supports multiple optimization metrics: RMSE, PCC (Pearson Correlation Coefficient), K-S statistic (Kolmogorov-Smirnov), and Wasserstein distance.

**Key Arguments**:
- `--data`: Path to processed parquet (default: `data/processed/halo_galaxy.parquet`)
- `--n-trials`: Number of Optuna trials (default: `50`)
- `--steps`: Number of training steps per trial (default: `1300`)
- `--metric`: Optimization metric to use (default: `RMSE`)
  - `RMSE`: Root Mean Squared Error (minimize)
  - `PCC`: Pearson Correlation Coefficient (maximize)
  - `K-S`: Kolmogorov-Smirnov statistic (minimize)
  - `Wasserstein`: Wasserstein distance (minimize)
- `--run-name`: Name for this experiment run (default: `optuna_best`)
- `--output-dir`: Directory for saved artifacts (default: `models/nf/`)
- `--posterior-samples`: Number of posterior samples per halo for cached predictions (default: `1000`)

**Inputs**: Processed parquet file from `preprocess.py`.

**Outputs**:
- `flow_state.pt`: Best model weights
- `optuna_study.pkl`: Optuna study object
- `training_summary.json`: Scalers, loss history, metrics per property, best hyperparameters, optimization metric used
- `predictions.npz`: Test set predictions (mean, random, full posterior)

**Example**:
```bash
# Optimize using RMSE (default)
python -m src.models.optimize_nf \
  --data data/processed/halo_galaxy.parquet \
  --n-trials 50 \
  --run-name optuna_best \
  --steps 1300

# Optimize using Pearson Correlation Coefficient
python -m src.models.optimize_nf \
  --data data/processed/halo_galaxy.parquet \
  --metric PCC \
  --n-trials 50 \
  --run-name optuna_best_pcc

# Optimize using K-S statistic
python -m src.models.optimize_nf \
  --data data/processed/halo_galaxy.parquet \
  --metric K-S \
  --n-trials 50 \
  --run-name optuna_best_ks

# Optimize using Wasserstein distance
python -m src.models.optimize_nf \
  --data data/processed/halo_galaxy.parquet \
  --metric Wasserstein \
  --n-trials 50 \
  --run-name optuna_best_wasserstein
```

### Visualization Scripts

#### `src/plots/visualize.py`

**Purpose**: Generate comparison plots between NN and NF models. Supports metrics bar charts and contour plots with marginals.

**Subcommands**:
- `metrics`: Grouped bar charts comparing RMSE, PCC, KS statistic, and Wasserstein distance
- `contours`: 2D contour plots with marginal histograms for property pairs

**Note**: TARP calibration curves are generated by `nf_visualizer.py`, not this script.

**Key Arguments**:
- `--nn-dir`: Directory containing NN runs (default: `models/nn/`)
- `--nf-dir`: Directory containing NF runs (default: `models/nf/`)
- `--nn-runs`: List of NN run names to compare
- `--nn-labels`: Optional labels for NN runs (defaults to run names)
- `--nf-run`: NF run name to compare
- `--subsample`: Fraction of data to use for contour plots (default: `0.5`)
- `--output` / `--output-prefix`: Output path(s) for generated plots

**Inputs**: Saved model artifacts from `train_nn.py` and `train_nf.py` (predictions, scalers).

**Outputs**:
- `metrics`: Single PNG with 2x2 subplot grid of metric comparisons
- `contours`: Individual PNG files per property pair (e.g., `SM_vs_Colour.png`)

**Example**:
```bash
# Metrics comparison
python -m src.plots.visualize metrics \
  --nn-runs raw smogn \
  --nn-labels "NN Raw" "NN SMOGN" \
  --nf-run baseline \
  --output outputs/plots/metrics.png

# Contour plots
python -m src.plots.visualize contours \
  --nn-runs raw smogn \
  --nf-run baseline \
  --output-prefix outputs/plots/contours \
  --subsample 0.5
```

#### `src/plots/nf_visualizer.py`

**Purpose**: NF-specific visualization suite. Generates scatter plots, conditioned property distributions, TARP calibration curves, and posterior distribution plots.

**Key Arguments**:
- `--data`: Path to processed parquet (default: `data/processed/halo_galaxy.parquet`)
- `--nf-run`: NF run name to visualize (default: `baseline`)
- `--nn-runs`: NN runs for metrics comparison (default: `["raw", "smogn"]`)
- `--output-dir`: Directory to store generated figures (default: `Plots/`)
- `--num-flow-samples`: Number of flow samples for distribution plots (default: `1000`)
- `--device`: Device for flow sampling (default: `cpu`)

**Inputs**: 
- Processed parquet file
- NF model artifacts (flow weights, scalers, predictions)
- Optional NN artifacts for metrics comparison

**Outputs**:
- `train set.png`: Training data vs flow samples (scatter + marginals)
- `test set.png`: Test data vs flow predictions (scatter + marginals)
- `six galaxies.png`: Posterior distributions for 6 galaxies selected by halo mass
- `six galaxies - <property>.png`: Posterior distributions for 6 galaxies selected by property
- `conditioned halo masses - <property>.png`: Property distributions conditioned on fixed halo masses
- `all metrics.png`: Comprehensive metrics comparison (NN vs NF)
- `pcc rmse.png`: Line plots of PCC and RMSE across properties
- `tarp test set.png`: TARP calibration curve (joint)
- `tarp test set individual.png`: TARP curves per property
- `tarp complete set.png`: TARP curve for full dataset

**Example**:
```bash
python -m src.plots.nf_visualizer \
  --data data/processed/halo_galaxy.parquet \
  --nf-run baseline \
  --nn-runs raw smogn \
  --output-dir outputs/plots/nf \
  --num-flow-samples 1000
```

## Notes

- Every script exposes `--help` for optional overrides (test split size, learning rate, posterior samples, etc.).
- The notebook is left untouched per request; refer to `docs/notebook_map.md` and this README when wiring new experiments.
- Model directories (`models/nn/<run>/...`, `models/nf/<run>/...`) are structured so you can keep multiple experiments side-by-side. Clean them up or add new `--run-name`s as needed.