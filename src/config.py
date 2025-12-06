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
DEFAULT_HALO_HDF5 = DATA_DIR / "groups_090.hdf5"
DEFAULT_PROCESSED_PARQUET = PROCESSED_DATA_DIR / "halo_galaxy.parquet"

# Training defaults
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

