from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
NN_MODEL_DIR = MODELS_DIR / "nn"
NF_MODEL_DIR = MODELS_DIR / "nf"

OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Default filenames / paths
# Use the local raw snapshot by default (relative to repo root)
DEFAULT_HALO_HDF5 = RAW_DATA_DIR / "groups_090_Lh_119.hdf5"
# Alternative: Use directory with multiple HDF5 files (e.g., TNG300 snapshot 99)
# DEFAULT_HALO_HDF5 = DATA_DIR / "raw"
DEFAULT_PROCESSED_PARQUET = PROCESSED_DATA_DIR / "halo_galaxy.parquet"
# Directory for preprocessing outputs (parquet, stats, plots, etc.)
DEFAULT_PREPROCESS_OUTPUT_DIR = PROCESSED_DATA_DIR / "preprocess"

# Training defaults
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

