"""
CLI to ingest the TNG halo/galaxy catalogs, merge them, and persist a
processed table with log-scaled features ready for modeling.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CLI
import numpy as np
import pandas as pd

from src import config
from src.utils.io import ensure_dir, save_json
from src.utils.plotting import plot_histograms, plot_correlation_heatmap


C_LIGHT = 3.0 * 10**8  # Speed of light in m/s


def detect_simulation_type(filepath: Path) -> str:
    """Detect whether HDF5 or parquet file is from TNG or SIMBA simulation.
    
    For HDF5 files: checks file contents (presence of V_max field) and filename patterns.
    For parquet files: checks for V_max column (SIMBA has it) or filename patterns.
    
    Note: HDF5 files themselves don't contain simulation type metadata, so detection
    relies on structural differences (e.g., SIMBA has Subhalo/SubhaloVmax field).
    
    Args:
        filepath: Path to HDF5 or parquet file
        
    Returns:
        "TNG" or "SIMBA"
    """
    try:
        filepath_str = str(filepath).lower()
        
        # Check filename patterns first (helpful but not reliable)
        if "simba" in filepath_str or "groups_090" in filepath_str:
            return "SIMBA"
        if "tng" in filepath_str or "illustris" in filepath_str:
            return "TNG"
        
        # For parquet files, check contents
        if filepath.suffix.lower() == ".parquet":
            return detect_simulation_type_from_parquet(filepath)
        
        # For HDF5 files, check contents
        if filepath.suffix.lower() in [".hdf5", ".h5"]:
            return detect_simulation_type_from_hdf5(filepath)
        
        # Default to TNG (most common)
        return "TNG"
    except Exception:
        # If we can't detect, default to TNG
        return "TNG"


def detect_simulation_type_from_hdf5(hdf5_path: Path) -> str:
    """Detect simulation type from an HDF5 file by checking its structure.
    
    This function checks for structural differences:
    1. SIMBA typically has Subhalo/SubhaloVmax field
    2. TNG may not have this field
    
    Note: This is not 100% reliable as HDF5 files don't contain explicit
    simulation type metadata. Users should specify --sim-type if unsure.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        "TNG" or "SIMBA"
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            # Check for V_max field (SIMBA typically has it)
            if "Subhalo/SubhaloVmax" in f:
                return "SIMBA"
            
            # If we can't determine from structure, default to TNG
            # User should specify --sim-type manually if this is wrong
            return "TNG"
    except Exception as e:
        logging.warning(f"Could not detect simulation type from HDF5 file {hdf5_path}: {e}")
        # Fallback to filename patterns
        filepath_str = str(hdf5_path).lower()
        if "simba" in filepath_str:
            return "SIMBA"
        return "TNG"


def detect_simulation_type_from_parquet(parquet_path: Path) -> str:
    """Detect simulation type from a parquet file by checking its contents.
    
    This function checks:
    1. If 'simulation_type' column exists (most reliable)
    2. If 'V_max' column exists (SIMBA has it, TNG typically doesn't)
    3. Filename patterns as fallback
    
    Args:
        parquet_path: Path to parquet file
        
    Returns:
        "TNG" or "SIMBA"
    """
    try:
        # First check if simulation_type column exists (most reliable)
        df = pd.read_parquet(parquet_path)
        if "simulation_type" in df.columns:
            sim_type = df["simulation_type"].iloc[0]
            if pd.notna(sim_type):
                return str(sim_type).upper()
        
        # Check for V_max column (SIMBA typically has it)
        if "V_max" in df.columns:
            return "SIMBA"
        
        # Fallback to filename patterns
        filepath_str = str(parquet_path).lower()
        if "simba" in filepath_str:
            return "SIMBA"
        if "tng" in filepath_str or "illustris" in filepath_str:
            return "TNG"
        
        # Default to TNG
        return "TNG"
    except Exception as e:
        logging.warning(f"Could not detect simulation type from parquet file {parquet_path}: {e}")
        # Fallback to filename patterns
        filepath_str = str(parquet_path).lower()
        if "simba" in filepath_str:
            return "SIMBA"
        return "TNG"


def load_catalogs(h5_path: Path, sim_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load halo/galaxy catalogs from single file or directory of HDF5 files.
    
    Args:
        h5_path: Path to HDF5 file or directory containing HDF5 files
        sim_type: Simulation type ("TNG" or "SIMBA") - must be explicitly provided
    
    Returns:
        Tuple of (halos DataFrame, galaxies DataFrame, simulation_type)
    """
    
    # Determine if we have a single file or directory
    if h5_path.is_file():
        files = [h5_path]
    elif h5_path.is_dir():
        files = sorted(glob.glob(str(h5_path / "TNG300-1_fof_subhalo_tab_099.*.hdf5")))
        if not files:
            # Fallback to old naming pattern
            files = sorted(glob.glob(str(h5_path / "fof_subhalo_tab_099.*.hdf5")))
        if not files:
            # Try any .hdf5 files (including SIMBA)
            files = sorted(glob.glob(str(h5_path / "*.hdf5")))
    else:
        raise ValueError(f"Path {h5_path} is neither file nor directory")
    
    if not files:
        raise ValueError(
            f"No HDF5 files found in {h5_path}. "
            "Expected files matching patterns: 'TNG300-1_fof_subhalo_tab_099.*.hdf5' or 'fof_subhalo_tab_099.*.hdf5' or 'groups_090*.hdf5'"
        )
    
    # Simulation type should be provided by caller via --sim-type argument
    # We don't auto-detect here since HDF5 files don't contain this metadata
    logging.info(f"Loading halo/galaxy catalogs from {len(files)} files in directory: {h5_path}")
    
    all_halos = []
    all_galaxies = []
    
    for file_idx, filepath in enumerate(files):
        if (file_idx + 1) % 10 == 0:
            logging.info(f"Processing file {file_idx+1}/{len(files)}: {Path(filepath).name}")
        
        try:
            with h5py.File(filepath, "r") as f:
                # Check if essential fields exist
                if "Group/Group_M_Crit200" not in f:
                    logging.warning(f"Skipping file {filepath} - missing Group_M_Crit200")
                    continue
                
                # === LOAD GROUP (HALO) DATA ===
                m_h = f["Group/Group_M_Crit200"][:] * 1e10  # Msun/h
                
                # Radius: both TNG and SIMBA are stored in kpc/h, no conversion needed
                r_h = f["Group/Group_R_Crit200"][:]  # Already in kpc/h
                
                v_h = f["Group/GroupVel"][:]
                v_h_mag = np.linalg.norm(v_h, axis=1)
                group_firstsub = f["Group/GroupFirstSub"][:]  # Index of first subhalo in group
                
                # === LOAD SUBHALO (GALAXY) DATA ===
                sm = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Stellar mass
                sfr = f["Subhalo/SubhaloSFR"][:] * 1e10  # Star formation rate
                sr = f["Subhalo/SubhaloHalfmassRadType"][:, 4]  # Half-mass radius
                
                # Try to load colors (may not exist in all catalogs, especially SIMBA)
                try:
                    colour = (
                        f["Subhalo/SubhaloStellarPhotometrics"][:, 4]
                        - f["Subhalo/SubhaloStellarPhotometrics"][:, 5]
                    )
                except (KeyError, ValueError):
                    colour = np.zeros(len(sm))  # Fallback if photometry unavailable
                
                # Try to load V_max (available in SIMBA, may not be in TNG)
                try:
                    v_max = f["Subhalo/SubhaloVmax"][:]  # km/s
                    has_vmax = True
                except KeyError:
                    v_max = None
                    has_vmax = False
                
                subhalo_grnr = f["Subhalo/SubhaloGrNr"][:]  # Parent group global ID
                
                # === MATCH CENTRAL GALAXIES TO HALOS ===
                # GroupFirstSub contains global subhalo indices that may point to subhalos
                # in other files. Instead, we match by SubhaloGrNr.
                # 
                # Strategy: Group subhalos by their parent group ID (SubhaloGrNr),
                # find the first subhalo (lowest index) for each group - this is the central.
                # Then match these to local group indices.
                
                file_halos = []
                file_galaxies = []
                
                # Create mapping from global group ID to list of (subhalo_index, stellar_mass)
                grnr_to_subhalos = {}
                for sub_idx in range(len(subhalo_grnr)):
                    global_grnr = subhalo_grnr[sub_idx]
                    if global_grnr not in grnr_to_subhalos:
                        grnr_to_subhalos[global_grnr] = []
                    grnr_to_subhalos[global_grnr].append(sub_idx)
                
                # For each unique group that has subhalos, find the central (first) subhalo
                # and try to match it to a local group index
                unique_global_grnrs = sorted(grnr_to_subhalos.keys())
                min_grnr = unique_global_grnrs[0] if unique_global_grnrs else 0
                
                # Try to map global group IDs to local indices using offset
                # This assumes groups in the file are a contiguous subset starting from some offset
                matched_groups = set()
                
                for global_grnr in unique_global_grnrs:
                    # Try to find corresponding local group index
                    # Strategy: try offset mapping (local_idx = global_grnr - min_grnr + first_valid_local_idx)
                    # We'll try mapping to group indices in order
                    subhalo_indices = sorted(grnr_to_subhalos[global_grnr])
                    central_sub_idx = subhalo_indices[0]  # First subhalo is the central
                    
                    # FILTER 1: Skip if no stellar mass or below minimum threshold
                    MIN_STELLAR_MASS = 1e6  # Msun/h - minimum stellar mass
                    if sm[central_sub_idx] <= 0 or sm[central_sub_idx] < MIN_STELLAR_MASS:
                        continue
                    
                    # FILTER 2: Skip unresolved galaxies (resolution limit)
                    # Stellar radius in kpc (not log-scaled yet)
                    sr_kpc = sr[central_sub_idx]  
                    # Minimum resolved size: 1 kpc 
                    # (Below this, sizes are dominated by numerical resolution)
                    MIN_RESOLVED_SR = 1.0  # kpc
                    if sr_kpc < MIN_RESOLVED_SR:
                        continue  # Skip unresolved galaxies
                    
                    # Try to find matching local group index
                    # Try offset: assume local_idx = global_grnr - min_grnr
                    local_grp_idx = global_grnr - min_grnr
                    
                    # Check if this local index is valid and not already matched
                    if 0 <= local_grp_idx < len(m_h) and local_grp_idx not in matched_groups:
                        # FILTER 3: Check halo properties BEFORE adding to dataset
                        # Physical thresholds (in linear units, BEFORE log transform)
                        MIN_HALO_MASS = 1e10  # Msun/h - minimum resolved halo mass
                        MIN_RADIUS = 0.1  # kpc/h - minimum halo radius
                        MIN_VELOCITY = 1.0  # km/s - minimum velocity
                        
                        m_h_val = m_h[local_grp_idx]
                        r_h_val = r_h[local_grp_idx]
                        v_h_val = v_h_mag[local_grp_idx]
                        
                        # Skip halos below physical thresholds
                        if (m_h_val < MIN_HALO_MASS or 
                            r_h_val < MIN_RADIUS or 
                            v_h_val < MIN_VELOCITY):
                            continue
                        
                        matched_groups.add(local_grp_idx)
                        
                        # Create unique ID across all files
                        unique_id = file_idx * 1_000_000 + local_grp_idx
                        
                        # Store halo properties
                        file_halos.append({
                            "M_h": m_h_val,
                            "R_h": r_h_val,
                            "V_h": v_h_val,
                            "ID": unique_id
                        })
                        
                        # Store central galaxy properties
                        gal_dict = {
                            "SM": sm[central_sub_idx],
                            "SFR": sfr[central_sub_idx],
                            "Colour": colour[central_sub_idx],
                            "SR": sr[central_sub_idx],
                            "ID": unique_id
                        }
                        
                        # Add V_max if available (SIMBA)
                        if has_vmax:
                            gal_dict["V_max"] = v_max[central_sub_idx]
                        
                        file_galaxies.append(gal_dict)
                
                all_halos.extend(file_halos)
                all_galaxies.extend(file_galaxies)
                
        except Exception as e:
            logging.warning(f"Skipping file {filepath} due to error: {e}")
            continue
    
    # Convert to DataFrames
    halos = pd.DataFrame(all_halos)
    gals = pd.DataFrame(all_galaxies)
    
    logging.info(f"Combined data: {len(halos)} unique groups, {len(gals)} central galaxies")
    
    # Check if we loaded any data
    if len(halos) == 0 or len(gals) == 0:
        raise ValueError(
            f"No valid data loaded from {len(files)} file(s) in {h5_path}. "
            "This could mean:\n"
            "  1. The HDF5 files don't contain the expected data structure (Group/Group_M_Crit200, etc.)\n"
            "  2. The files don't have any groups with central galaxies\n"
            "  3. The central galaxies don't have stellar mass > 0\n"
            "  4. All halos/galaxies were filtered out due to quality thresholds\n"
            f"Please verify the files are valid {sim_type} halo/galaxy catalogs."
        )
    
    return halos, gals, sim_type


def load_camels_params(master_file_path: Path) -> Dict[str, np.ndarray]:
    """Load CAMELS parameters from GitHub-hosted parameter file."""
    # Skip the header line (line 0) which starts with '#'
    params = np.loadtxt(master_file_path, dtype=str, skiprows=1)
    
    param_dict = {}
    for row in params:
        sim_name = row[0]  # e.g., "LH_119"
        # Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2
        # File order: Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2
        # Need to reorder: take columns 1,2,3,5,4,6
        values = np.array([
            float(row[1]),  # Omega_m
            float(row[2]),  # sigma_8
            float(row[3]),  # A_SN1
            float(row[5]),  # A_SN2
            float(row[4]),  # A_AGN1
            float(row[6]),  # A_AGN2
        ])
        param_dict[sim_name] = values
    
    return param_dict


def preprocess_dataframe(halos: pd.DataFrame, gals: pd.DataFrame, camels_params: Dict[str, np.ndarray] = None, sim_name: str = None, sim_type: str = "TNG") -> pd.DataFrame:
    logging.info("Merging %d halos with %d galaxies", len(halos), len(gals))
    
    # Check if DataFrames are empty
    if len(halos) == 0 or len(gals) == 0:
        raise ValueError(
            f"Cannot merge empty DataFrames. Halos: {len(halos)}, Galaxies: {len(gals)}. "
            "Please check that the input HDF5 files contain valid data and match the expected format."
        )
    
    # Check if "ID" column exists in both DataFrames
    if "ID" not in halos.columns:
        raise ValueError(f"Halos DataFrame missing 'ID' column. Available columns: {list(halos.columns)}")
    if "ID" not in gals.columns:
        raise ValueError(f"Galaxies DataFrame missing 'ID' column. Available columns: {list(gals.columns)}")
    
    df = pd.merge(halos, gals, on="ID")

    # Store simulation type as a column for later identification
    df['simulation_type'] = sim_type
    logging.info(f"Stored simulation type: {sim_type}")
    
    # Add CAMELS parameters if provided
    if camels_params is not None and sim_name is not None:
        if sim_name not in camels_params:
            raise ValueError(f"Simulation name '{sim_name}' not found in CAMELS parameter file. Available simulations: {list(camels_params.keys())[:10]}...")
        current_sim_params = camels_params[sim_name]
        df['Omega_m'] = current_sim_params[0]
        df['sigma_8'] = current_sim_params[1]
        df['A_SN1'] = current_sim_params[2]
        df['A_SN2'] = current_sim_params[3]
        df['A_AGN1'] = current_sim_params[4]
        df['A_AGN2'] = current_sim_params[5]
        logging.info(f"Added CAMELS parameters for simulation {sim_name}")

    # Log transforms - NO CLIPPING! We've already filtered bad values
    # Filter out any remaining invalid values BEFORE log transform
    initial_len = len(df)
    
    for col in ["M_h", "R_h", "V_h", "SM"]:
        # Check for invalid values (should be rare after filtering)
        invalid = (df[col] <= 0) | (~np.isfinite(df[col]))
        if invalid.any():
            n_invalid = invalid.sum()
            logging.warning(f"Found {n_invalid} invalid {col} values after filtering - removing them")
            df = df[~invalid].copy()
        
        # Now safe to log transform
        df[col] = np.log10(df[col])
    
    if len(df) < initial_len:
        logging.info(f"Removed {initial_len - len(df)} rows with invalid mass/size values")

    # Handle SFR zeros before log scaling
    # For TNG: Follow the notebook preprocessing logic:
    # 1. Replace 0 with 1
    # 2. Take log10
    # 3. Replace log10(1)=0 with Gaussian noise around 6.4 (representing quiescent galaxies)
    # For SIMBA: Same approach but may have different characteristics
    
    if sim_type == "TNG":
        # TNG-specific preprocessing (from notebook)
        sfr_zero_mask = df["SFR"] <= 0
        if sfr_zero_mask.any():
            n_zero_sfr = sfr_zero_mask.sum()
            logging.info(f"Found {n_zero_sfr} galaxies with SFR <= 0 in TNG")
            # Step 1: Replace 0 with 1
            df.loc[sfr_zero_mask, "SFR"] = 1
        
        # Step 2: Take log10 of all SFR values
        df["SFR"] = np.log10(df["SFR"])
        
        # Step 3: Replace log10(1)=0 with Gaussian noise
        # This represents non-forming galaxies with a realistic distribution
        sfr_zero_after_log = df["SFR"] == 0
        if sfr_zero_after_log.any():
            n_zero_after_log = sfr_zero_after_log.sum()
            logging.info(f"Replacing {n_zero_after_log} log(SFR)=0 values with Gaussian(8.0, 0.5)")
            # Cast to match the column's dtype to avoid FutureWarning
            sfr_dtype = df["SFR"].dtype
            df.loc[sfr_zero_after_log, "SFR"] = np.random.normal(8.0, 0.5, n_zero_after_log).astype(sfr_dtype)
    else:
        # SIMBA: Use small positive value for quiescent galaxies
        sfr_zero_mask = df["SFR"] <= 0
        if sfr_zero_mask.any():
            n_zero_sfr = sfr_zero_mask.sum()
            logging.info(f"Found {n_zero_sfr} galaxies with SFR <= 0 in SIMBA")
            df.loc[sfr_zero_mask, "SFR"] = 1e-10
        df["SFR"] = np.log10(df["SFR"])

    # Log transform SR (half-mass radius)
    df["SR"] = np.log10(df["SR"] + 0.001)
    
    # Log transform V_max if available (SIMBA)
    if "V_max" in df.columns:
        invalid_vmax = (df["V_max"] <= 0) | (~np.isfinite(df["V_max"]))
        if invalid_vmax.any():
            n_invalid = invalid_vmax.sum()
            logging.warning(f"Found {n_invalid} invalid V_max values")
            # Option: compute from M_h and R_h, or use V_h as proxy
            # For now, use V_h as proxy
            df.loc[invalid_vmax, "V_max"] = df.loc[invalid_vmax, "V_h"]
        df["V_max"] = np.log10(df["V_max"])
    
    # Final check for any remaining invalid values - DON'T FILL, DROP!
    for col in df.select_dtypes(include=[np.number]).columns:
        invalid = ~np.isfinite(df[col])
        if invalid.any():
            n_invalid = invalid.sum()
            logging.error(f"Found {n_invalid} invalid values in {col} after all processing - removing rows")
            df = df[~invalid].copy()
    
    # Verify no bad values remain
    if len(df) > 0:
        assert df['M_h'].min() > 9.0, f"Found bad M_h values: min={df['M_h'].min()}"
        assert df['SM'].min() > 6.0, f"Found bad SM values: min={df['SM'].min()}"

    logging.info("Final dataframe shape: %s", df.shape)
    return df


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary = {}
    describe_df = df.describe(include="all")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col] = {
                "mean": float(describe_df.loc["mean", col]),
                "std": float(describe_df.loc["std", col]),
                "min": float(describe_df.loc["min", col]),
                "max": float(describe_df.loc["max", col]),
            }
    return summary


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all preprocessing diagnostic plots."""
    ensure_dir(output_dir)
    
    # Plot halo properties
    plot_histograms(
        df, 
        columns=["M_h", "R_h", "V_h"],
        labels=["M_h", "R_h", "V_h"],
        title="Halo properties",
        output_path=output_dir / "halo_properties.png",
    )
    logging.info("Saved halo properties plot")
    
    # Plot galaxy properties
    plot_histograms(
        df,
        columns=["SM", "SFR", "Colour", "SR"],
        labels=["SM", "SFR", "Colour", "SR"],
        title="Galaxy properties",
        output_path=output_dir / "galaxy_properties.png",
    )
    logging.info("Saved galaxy properties plot")
    
    # Plot correlation heatmap
    plot_correlation_heatmap(df, output_dir / "correlation_heatmap.png")
    logging.info("Saved correlation heatmap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess halo/galaxy catalog.")
    parser.add_argument(
        "--halo-hdf5",
        type=Path,
        default=config.DEFAULT_HALO_HDF5,
        help="Path to groups_XXX.hdf5 file or directory containing multiple HDF5 files (e.g., fof_subhalo_tab_099.*.hdf5) with halo + subhalo data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEFAULT_PREPROCESS_OUTPUT_DIR,
        help="Directory to store processed outputs (parquet, plots, stats).",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional custom path for summary statistics JSON (defaults inside output dir).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV copy of the processed table. If relative, saved inside output dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_RANDOM_STATE,
        help="Random seed for stochastic steps (e.g., SFR noise).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots.",
    )
    parser.add_argument(
        "--sim-name",
        type=str,
        default=None,
        help="Simulation name (e.g., LH_119, LH_0) to look up parameters from CAMELS parameter file.",
    )
    parser.add_argument(
        "--param-file",
        type=Path,
        default=Path("data/raw/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"),
        help="Path to CAMELS parameter file.",
    )
    parser.add_argument(
        "--sim-type",
        type=str,
        required=True,
        choices=["TNG", "SIMBA"],
        help="REQUIRED: Specify simulation type (TNG or SIMBA). HDF5 files don't contain simulation type metadata, "
             "so you must explicitly specify which simulation the data comes from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    np.random.seed(args.seed)

    # Simulation type is required via --sim-type argument
    sim_type = args.sim_type.upper()
    logging.info(f"Using simulation type: {sim_type}")
    
    halos, gals, sim_type = load_catalogs(args.halo_hdf5, sim_type)
    
    # Load CAMELS parameters if sim-name is provided
    camels_params = None
    if args.sim_name:
        if not args.param_file.exists():
            raise FileNotFoundError(f"CAMELS parameter file not found: {args.param_file}")
        camels_params = load_camels_params(args.param_file)
        logging.info(f"Loaded CAMELS parameters for {len(camels_params)} simulations")
    
    df = preprocess_dataframe(halos, gals, camels_params=camels_params, sim_name=args.sim_name, sim_type=sim_type)

    output_dir = args.output
    ensure_dir(output_dir)

    parquet_filename = config.DEFAULT_PROCESSED_PARQUET.name
    parquet_path = output_dir / parquet_filename
    df.to_parquet(parquet_path, index=False)
    logging.info("Saved processed parquet to %s", parquet_path)

    if args.csv:
        csv_path = args.csv
        # If a relative path or bare filename is provided, place it under the output directory
        if not csv_path.is_absolute():
            csv_path = output_dir / csv_path
        # If a directory is provided, use the parquet filename with .csv extension
        if csv_path.is_dir():
            csv_path = csv_path / parquet_filename.replace(".parquet", ".csv")
        elif csv_path.suffix == "":
            csv_path = csv_path.with_suffix(".csv")
        ensure_dir(csv_path.parent)
        df.to_csv(csv_path, index=False)
        logging.info("Saved CSV copy to %s", csv_path)

    stats = compute_summary_stats(df)
    stats_path = args.stats_json
    if stats_path is None:
        stats_path = output_dir / "halo_galaxy_stats.json"
    elif not stats_path.is_absolute():
        stats_path = output_dir / stats_path
    ensure_dir(stats_path.parent)
    save_json(stats, stats_path)
    logging.info("Wrote summary stats to %s", stats_path)

    # Generate diagnostic plots inside the chosen output directory
    if not args.no_plots:
        generate_plots(df, output_dir)
        logging.info("Generated diagnostic plots in %s", output_dir)


if __name__ == "__main__":
    main()

