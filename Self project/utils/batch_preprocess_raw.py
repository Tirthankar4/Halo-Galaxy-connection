"""
Batch preprocessing for all CAMELS HDF5 snapshots in ``data/raw``.

For each ``*.hdf5`` file in ``data/raw`` this script:
  - Infers the CAMELS simulation name (e.g. ``LH_135``) from the filename
  - Chooses an output directory ``data/processed/lh{N}`` (e.g. ``lh135``)
  - Calls ``src.data.preprocess`` with appropriate arguments:
        --halo-hdf5   data/raw/<file>.hdf5
        --output      data/processed/lh{N}
        --sim-type    TNG
        --sim-name    LH_{N}
        --param-file  data/raw/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt

This matches the existing folder structure such as:
    data/processed/lh135/halo_galaxy.parquet
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def infer_lh_id_from_name(name: str) -> str | None:
    """
    Infer CAMELS LH id (digits only) from an HDF5 base filename.

    Handles patterns like:
      - groups_090_lh_135.hdf5
      - groups_090_Lh_473.hdf5
      - LH_122_groups_090.hdf5
      - groups_090_lh_1.hdf5
    """
    # Prefer explicit LH_XXX style first (case insensitive)
    m = re.search(r"(?:LH_|lh_|Lh_)(\d+)", name)
    if m:
        return m.group(1)

    # Fallback: generic "lh_<digits>" anywhere
    m = re.search(r"lh_(\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    param_file = raw_dir / "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"

    # Always use the current (global) Python interpreter, as requested
    python_exe = Path(sys.executable)

    if not raw_dir.exists():
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        sys.exit(1)

    if not param_file.exists():
        print(f"[ERROR] CAMELS parameter file not found: {param_file}")
        sys.exit(1)

    hdf5_files = sorted(raw_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"[WARN] No .hdf5 files found in {raw_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files in {raw_dir}")

    for h5 in hdf5_files:
        name = h5.stem  # filename without extension
        lh_id = infer_lh_id_from_name(name)
        if lh_id is None:
            print(f"[SKIP] Could not infer LH id from {name} (file: {h5.name})")
            continue

        sim_name = f"LH_{lh_id}"
        dataset = f"lh{lh_id}"
        out_dir = processed_dir / dataset

        cmd = [
            str(python_exe),
            "-m",
            "src.data.preprocess",
            "--halo-hdf5",
            str(h5),
            "--output",
            str(out_dir),
            "--sim-type",
            "TNG",
            "--sim-name",
            sim_name,
            "--param-file",
            str(param_file),
            "--log-level",
            "INFO",
        ]

        print("\n" + "=" * 80)
        print(f"Processing {h5.name}")
        print("  LH id      :", lh_id)
        print("  sim_name   :", sim_name)
        print("  output_dir :", out_dir)
        print("  command    :", " ".join(cmd))
        print("=" * 80 + "\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] Preprocessing failed for {h5} with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\nAll available HDF5 files in data/raw have been preprocessed into data/processed/lh*/")


if __name__ == "__main__":
    main()


