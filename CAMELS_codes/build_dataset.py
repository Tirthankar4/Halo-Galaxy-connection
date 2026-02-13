#!/usr/bin/env python3
"""
CAMELS dataset builder.

Combines:
1) HDF5 -> processed CSVs
2) Processed CSVs -> combined Parquet with simulation parameters
"""

import argparse
import glob
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_hdf5_to_csv(hdf5_path, output_csv_path=None, output_dir=None):
    """
    Convert CAMELS HDF5 galaxy catalog to CSV.
    """
    print(f"Processing: {hdf5_path}")

    # Open the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        # ===== Extract Group (Halo) Data =====
        GroupFirstSub = f["Group/GroupFirstSub"][:]  # Index of central subhalo

        # ===== Extract Subhalo (Galaxy) Data =====
        # The 14 properties
        Mg = f["Subhalo/SubhaloMassType"][:, 0] * 1e10  # Gas mass (Msun/h)
        MBH = f["Subhalo/SubhaloBHMass"][:] * 1e10      # Black hole mass (Msun/h)
        Mstar = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Stellar mass (Msun/h)
        Mt = f["Subhalo/SubhaloMass"][:] * 1e10         # Total mass (Msun/h)
        Vmax = f["Subhalo/SubhaloVmax"][:]              # Max circular velocity (km/s)
        sigma_v = f["Subhalo/SubhaloVelDisp"][:]        # Velocity dispersion (km/s)
        Zg = f["Subhalo/SubhaloGasMetallicity"][:]      # Gas metallicity
        Zstar = f["Subhalo/SubhaloStarMetallicity"][:]  # Stellar metallicity
        SFR = f["Subhalo/SubhaloSFR"][:]                # Star formation rate (Msun/year)

        # Vector quantities - compute modulus
        spin_vector = f["Subhalo/SubhaloSpin"][:]       # 3D vector
        J = np.linalg.norm(spin_vector, axis=1)         # Spin modulus

        # Radii
        Rstar = f["Subhalo/SubhaloHalfmassRadType"][:, 4]  # Half stellar mass radius (kpc/h)
        Rt = f["Subhalo/SubhaloHalfmassRad"][:]            # Half total mass radius (kpc/h)
        Rmax = f["Subhalo/SubhaloVmaxRad"][:]              # Radius at Vmax (kpc/h)

        # Position and velocity vectors (keep all 3 components)
        SubhaloPos = f["Subhalo/SubhaloPos"][:]         # N × 3 (kpc/h)
        SubhaloVel = f["Subhalo/SubhaloVel"][:]         # N × 3 (km/s)

        # Group membership
        SubhaloGrNr = f["Subhalo/SubhaloGrNr"][:]       # Which halo each galaxy belongs to

    # ===== Apply Stellar Mass Filter =====
    # Filter galaxies with Mstar > 1.3e8 (matching notebook logic)
    mass_threshold = 1.3e8
    mass_mask = Mstar > mass_threshold
    indexes_initial = np.where(mass_mask)[0]

    print(f"Total subhalos: {len(Mstar)}")
    print(f"Subhalos above mass threshold (Mstar > {mass_threshold:.1e}): {len(indexes_initial)}")

    # ===== Identify Centrals and Satellites (before filtering) =====
    is_central_initial = np.isin(indexes_initial, GroupFirstSub)

    # ===== Remove satellites whose central is below mass threshold =====
    central_indexes_above_threshold = set(indexes_initial[is_central_initial])
    keep_satellite_mask = np.ones(len(indexes_initial), dtype=bool)
    for i, idx in enumerate(indexes_initial):
        if not is_central_initial[i]:  # If it's a satellite
            central_idx = GroupFirstSub[SubhaloGrNr[idx]]
            if central_idx not in central_indexes_above_threshold:
                keep_satellite_mask[i] = False

    # Apply the filtering mask
    indexes = indexes_initial[keep_satellite_mask]

    print(f"After removing orphan satellites: {len(indexes)}")
    print(f"  Centrals: {is_central_initial[keep_satellite_mask].sum()}")
    print(f"  Satellites: {len(indexes) - is_central_initial[keep_satellite_mask].sum()}")

    # Filter all arrays
    Mg = Mg[indexes]
    MBH = MBH[indexes]
    Mstar = Mstar[indexes]
    Mt = Mt[indexes]
    Vmax = Vmax[indexes]
    sigma_v = sigma_v[indexes]
    Zg = Zg[indexes]
    Zstar = Zstar[indexes]
    SFR = SFR[indexes]
    J = J[indexes]
    Rstar = Rstar[indexes]
    Rt = Rt[indexes]
    Rmax = Rmax[indexes]
    SubhaloPos = SubhaloPos[indexes]
    SubhaloVel = SubhaloVel[indexes]
    SubhaloGrNr = SubhaloGrNr[indexes]

    # ===== Identify Centrals vs Satellites =====
    is_central = np.isin(indexes, GroupFirstSub)

    num_centrals = is_central.sum()
    num_satellites = len(indexes) - num_centrals
    print(f"Central galaxies: {num_centrals}")
    print(f"Satellite galaxies: {num_satellites}")

    # ===== Create Linking Metadata =====
    central_subhalo_index = np.zeros(len(indexes), dtype=int)
    for i, halo_id in enumerate(SubhaloGrNr):
        if 0 <= halo_id < len(GroupFirstSub):
            central_subhalo_index[i] = GroupFirstSub[halo_id]
        else:
            central_subhalo_index[i] = -1  # Invalid halo

    # ===== Create DataFrame =====
    data = {
        # Identification & Metadata
        "subhalo_index": indexes,
        "halo_id": SubhaloGrNr,
        "central_subhalo_index": central_subhalo_index,
        "is_central": is_central.astype(int),

        # 14 Properties
        "Mg": Mg,
        "MBH": MBH,
        "Mstar": Mstar,
        "Mt": Mt,
        "Vmax": Vmax,
        "sigma_v": sigma_v,
        "Zg": Zg,
        "Zstar": Zstar,
        "SFR": SFR,
        "J": J,
        "Rstar": Rstar,
        "Rt": Rt,
        "Rmax": Rmax,

        # Position & Velocity (3 components each)
        "pos_x": SubhaloPos[:, 0],
        "pos_y": SubhaloPos[:, 1],
        "pos_z": SubhaloPos[:, 2],
        "vel_x": SubhaloVel[:, 0],
        "vel_y": SubhaloVel[:, 1],
        "vel_z": SubhaloVel[:, 2],
    }

    df = pd.DataFrame(data)

    # ===== Save to CSV =====
    if output_csv_path is None:
        base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
        output_csv_path = f"{base_name}_processed.csv"

    if output_dir is not None:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True)
        output_csv_path = str(output_dir_path / Path(output_csv_path).name)

    df.to_csv(output_csv_path, index=False)
    print(f"Saved CSV to: {output_csv_path}")
    print(f"CSV shape: {df.shape} (rows, columns)")

    return df


def load_parameters(params_file: Path) -> pd.DataFrame:
    """
    Load simulation parameters from the CosmoAstroSeed file.
    """
    print(f"Loading parameters from: {params_file}")

    params_df = pd.read_csv(
        params_file,
        sep=r"\s+",
        comment="#",
        names=[
            "simulation_name",
            "Omega_m",
            "sigma_8",
            "A_SN1",
            "A_AGN1",
            "A_SN2",
            "A_AGN2",
            "Omega_b",
            "seed",
        ],
        header=None,
    )

    params_df["simulation_id"] = params_df["simulation_name"].str.extract(
        r"SB7_(\d+)"
    ).astype(int)

    print(f"Loaded {len(params_df)} simulation parameters")
    print(f"Parameter columns: {list(params_df.columns)}")

    return params_df


def extract_sim_id_from_filename(filename: str) -> int:
    """
    Extract simulation ID from CSV filename.
    """
    match = re.search(r"Astrid_SB7_(\d+)_groups_090_processed\.csv", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract simulation ID from filename: {filename}")


def load_and_combine_csvs(processed_dir: Path, params_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load all processed CSV files, merge with parameters, and combine.
    """
    csv_files = sorted(processed_dir.glob("Astrid_SB7_*_groups_090_processed.csv"))
    print(f"Found {len(csv_files)} CSV files to process")

    if len(csv_files) != 1024:
        print(f"WARNING: Expected 1024 files, found {len(csv_files)}")

    all_dataframes = []
    total_galaxies = 0

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        sim_id = extract_sim_id_from_filename(csv_file.name)

        galaxy_df = pd.read_csv(csv_file)
        n_galaxies = len(galaxy_df)
        total_galaxies += n_galaxies

        galaxy_df["simulation_id"] = sim_id

        sim_params = params_df[params_df["simulation_id"] == sim_id]

        if len(sim_params) == 0:
            print(
                f"WARNING: No parameters found for simulation {sim_id} (file: {csv_file.name})"
            )
            continue

        for col in [
            "simulation_name",
            "Omega_m",
            "sigma_8",
            "A_SN1",
            "A_AGN1",
            "A_SN2",
            "A_AGN2",
            "Omega_b",
            "seed",
        ]:
            galaxy_df[col] = sim_params[col].values[0]

        all_dataframes.append(galaxy_df)

    print(
        f"\nCombining {len(all_dataframes)} dataframes with {total_galaxies:,} total galaxies..."
    )
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    return combined_df


def organize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Organize columns in a logical order.
    """
    sim_id_cols = ["simulation_id", "simulation_name"]
    galaxy_id_cols = ["subhalo_index", "halo_id", "central_subhalo_index", "is_central"]
    param_cols = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2", "Omega_b", "seed"]

    all_cols = df.columns.tolist()
    property_cols = [c for c in all_cols if c not in sim_id_cols + galaxy_id_cols + param_cols]

    ordered_cols = sim_id_cols + galaxy_id_cols + property_cols + param_cols
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    return df[ordered_cols]


def print_summary(df: pd.DataFrame):
    """Print summary statistics about the combined dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal galaxies: {len(df):,}")
    print(f"Total simulations: {df['simulation_id'].nunique()}")
    print(f"Total columns: {len(df.columns)}")

    print(f"\nGalaxies per simulation:")
    galaxies_per_sim = df.groupby("simulation_id").size()
    print(f"  Min: {galaxies_per_sim.min()}")
    print(f"  Max: {galaxies_per_sim.max()}")
    print(f"  Mean: {galaxies_per_sim.mean():.1f}")
    print(f"  Median: {galaxies_per_sim.median():.1f}")

    print(f"\nCentral vs Satellite galaxies:")
    central_counts = df["is_central"].value_counts()
    print(f"  Central (is_central=1): {central_counts.get(1, 0):,}")
    print(f"  Satellite (is_central=0): {central_counts.get(0, 0):,}")

    print(f"\nColumn groups:")
    sim_id_cols = ["simulation_id", "simulation_name"]
    galaxy_id_cols = ["subhalo_index", "halo_id", "central_subhalo_index", "is_central"]
    param_cols = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2", "Omega_b", "seed"]
    property_cols = [c for c in df.columns if c not in sim_id_cols + galaxy_id_cols + param_cols]

    print(f"  Simulation ID columns: {sim_id_cols}")
    print(f"  Galaxy ID columns: {galaxy_id_cols}")
    print(f"  Galaxy property columns ({len(property_cols)}): {property_cols}")
    print(f"  Parameter columns (ML targets): {param_cols}")

    print(f"\nParameter ranges:")
    for param in param_cols:
        if param in df.columns:
            print(f"  {param}: [{df[param].min():.6f}, {df[param].max():.6f}]")

    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nFirst 5 rows:")
    print(df.head())


def run_hdf5_to_csv(input_dir: Path, output_dir: Path) -> int:
    hdf5_files = sorted(glob.glob(str(input_dir / "*.hdf5")))

    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in '{input_dir}'.")
        return 1

    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CAMELS HDF5 to CSV Batch Processor")
    print("=" * 70)
    print(f"Input directory: {input_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Found {len(hdf5_files)} HDF5 file(s) to process")
    print("=" * 70)
    print()

    success_count = 0
    failure_count = 0

    for i, hdf5_path in enumerate(hdf5_files, 1):
        print(f"\n[{i}/{len(hdf5_files)}] Processing: {Path(hdf5_path).name}")
        print("-" * 70)

        try:
            process_hdf5_to_csv(hdf5_path, output_dir=str(output_dir))
            success_count += 1
            print(f"✓ Successfully processed {Path(hdf5_path).name}")
        except Exception as e:
            failure_count += 1
            print(f"✗ Failed to process {Path(hdf5_path).name}")
            print(f"  Error: {str(e)}")

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Successful: {success_count}/{len(hdf5_files)}")
    print(f"Failed: {failure_count}/{len(hdf5_files)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 70)

    return 0 if failure_count == 0 else 1


def run_csv_to_parquet(processed_dir: Path, params_file: Path, output_file: Path, verify: bool):
    print("=" * 60)
    print("CAMELS ML Dataset Creator")
    print("=" * 60)
    print(f"\nProcessed CSVs: {processed_dir}")
    print(f"Parameters file: {params_file}")
    print(f"Output file: {output_file}")

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file}")

    params_df = load_parameters(params_file)
    combined_df = load_and_combine_csvs(processed_dir, params_df)
    combined_df = organize_columns(combined_df)
    print_summary(combined_df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_file}")
    combined_df.to_parquet(output_file, index=False, engine="pyarrow")

    file_size_mb = output_file.stat().st_size / 1024**2
    print(f"Saved successfully! File size: {file_size_mb:.2f} MB")

    if verify:
        print("\nVerifying saved file...")
        verify_df = pd.read_parquet(output_file)
        print(f"Verification: {len(verify_df):,} rows, {len(verify_df.columns)} columns")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build CAMELS datasets from raw data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hdf5_parser = subparsers.add_parser(
        "hdf5-to-csv", help="Process CAMELS HDF5 galaxy catalogs to CSV files"
    )
    hdf5_parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing HDF5 files"
    )
    hdf5_parser.add_argument(
        "--output-dir",
        type=str,
        default="CAMELS_processed",
        help="Directory to save processed CSV files (default: CAMELS_processed)",
    )

    parquet_parser = subparsers.add_parser(
        "csv-to-parquet", help="Combine processed CSVs into one parquet file"
    )
    parquet_parser.add_argument(
        "--processed-dir",
        type=str,
        default="CAMELS_processed",
        help="Directory containing processed CSV files (default: CAMELS_processed)",
    )
    parquet_parser.add_argument(
        "--params-file",
        type=str,
        default="CAMELS_datas/CosmoAstroSeed_Astrid_L25n256_SB7.txt",
        help="Path to parameter file (default: CAMELS_datas/CosmoAstroSeed_Astrid_L25n256_SB7.txt)",
    )
    parquet_parser.add_argument(
        "--output-file",
        type=str,
        default="CAMELS_datas/camels_astrid_sb7_090.parquet",
        help="Output parquet file path (default: CAMELS_datas/camels_astrid_sb7_090.parquet)",
    )
    parquet_parser.add_argument(
        "--verify",
        action="store_true",
        help="Read the saved parquet to verify output",
    )

    args = parser.parse_args()

    if args.command == "hdf5-to-csv":
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"ERROR: Input directory '{args.input_dir}' does not exist or is not a directory.")
            return 1
        return run_hdf5_to_csv(input_dir, output_dir)

    if args.command == "csv-to-parquet":
        processed_dir = Path(args.processed_dir)
        params_file = Path(args.params_file)
        output_file = Path(args.output_file)
        run_csv_to_parquet(processed_dir, params_file, output_file, args.verify)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
