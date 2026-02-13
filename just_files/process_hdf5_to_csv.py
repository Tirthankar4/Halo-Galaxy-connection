#!/usr/bin/env python3
"""
process_hdf5_to_csv.py
Helper script for bash version - processes HDF5 to CSV

Called by download_and_process_camels.sh with arguments:
    python3 process_hdf5_to_csv.py <hdf5_path> <sim_name> <output_dir>
"""

import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def process_hdf5_to_csv(hdf5_path, sim_name, output_dir):
    """
    Convert CAMELS HDF5 galaxy catalog to CSV.
    
    Parameters:
    -----------
    hdf5_path : str or Path
        Path to the HDF5 file (e.g., 'groups_090.hdf5')
    sim_name : str
        Simulation name for generating output filename
    output_dir : str or Path
        Directory where CSV will be saved
    
    Returns:
    --------
    int
        0 on success, 1 on failure
    """
    
    try:
        hdf5_path = Path(hdf5_path)
        output_dir = Path(output_dir)
        
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            # ===== Extract Group (Halo) Data =====
            GroupFirstSub = f['Group/GroupFirstSub'][:]  # Index of central subhalo for each halo
            
            # ===== Extract Subhalo (Galaxy) Data =====
            # The 14 properties
            Mg = f['Subhalo/SubhaloMassType'][:, 0] * 1e10  # Gas mass (Msun/h)
            MBH = f['Subhalo/SubhaloBHMass'][:] * 1e10      # Black hole mass (Msun/h)
            Mstar = f['Subhalo/SubhaloMassType'][:, 4] * 1e10  # Stellar mass (Msun/h)
            Mt = f['Subhalo/SubhaloMass'][:] * 1e10         # Total mass (Msun/h)
            Vmax = f['Subhalo/SubhaloVmax'][:]              # Max circular velocity (km/s)
            sigma_v = f['Subhalo/SubhaloVelDisp'][:]        # Velocity dispersion (km/s)
            Zg = f['Subhalo/SubhaloGasMetallicity'][:]      # Gas metallicity
            Zstar = f['Subhalo/SubhaloStarMetallicity'][:]  # Stellar metallicity
            SFR = f['Subhalo/SubhaloSFR'][:]                # Star formation rate (Msun/year)
            
            # Vector quantities - compute modulus
            spin_vector = f['Subhalo/SubhaloSpin'][:]       # 3D vector
            J = np.linalg.norm(spin_vector, axis=1)         # Spin modulus
            
            vel_vector = f['Subhalo/SubhaloVel'][:]         # 3D velocity vector (km/s)
            
            # Radii
            Rstar = f['Subhalo/SubhaloHalfmassRadType'][:, 4]  # Half stellar mass radius (kpc/h)
            Rt = f['Subhalo/SubhaloHalfmassRad'][:]            # Half total mass radius (kpc/h)
            Rmax = f['Subhalo/SubhaloVmaxRad'][:]              # Radius at Vmax (kpc/h)
            
            # Position and velocity vectors (keep all 3 components)
            SubhaloPos = f['Subhalo/SubhaloPos'][:]         # N × 3 (kpc/h)
            SubhaloVel = f['Subhalo/SubhaloVel'][:]         # N × 3 (km/s)
            
            # Group membership
            SubhaloGrNr = f['Subhalo/SubhaloGrNr'][:]       # Which halo each galaxy belongs to
        
        # ===== Apply Stellar Mass Filter =====
        # Filter galaxies with Mstar > 1.3e8 (matching notebook logic)
        mass_threshold = 1.3e8
        mass_mask = Mstar > mass_threshold
        indexes_initial = np.where(mass_mask)[0]
        
        # ===== Identify Centrals and Satellites (before filtering) =====
        is_central_initial = np.isin(indexes_initial, GroupFirstSub)
        
        # ===== Remove satellites whose central is below mass threshold =====
        # Get indices of centrals that passed the mass cut
        central_indexes_above_threshold = set(indexes_initial[is_central_initial])
        
        # Keep satellites only if their central is in the set
        keep_satellite_mask = np.ones(len(indexes_initial), dtype=bool)
        for i, idx in enumerate(indexes_initial):
            if not is_central_initial[i]:  # If it's a satellite
                central_idx = GroupFirstSub[SubhaloGrNr[idx]]
                if central_idx not in central_indexes_above_threshold:
                    keep_satellite_mask[i] = False
        
        # Apply the filtering mask
        indexes = indexes_initial[keep_satellite_mask]
        
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
        
        # ===== Create Linking Metadata =====
        # For each galaxy, get the central subhalo index for its halo
        central_subhalo_index = np.zeros(len(indexes), dtype=int)
        for i, halo_id in enumerate(SubhaloGrNr):
            if 0 <= halo_id < len(GroupFirstSub):
                central_subhalo_index[i] = GroupFirstSub[halo_id]
            else:
                central_subhalo_index[i] = -1  # Invalid halo
        
        # ===== Create DataFrame =====
        data = {
            # Identification & Metadata
            'subhalo_index': indexes,
            'halo_id': SubhaloGrNr,
            'central_subhalo_index': central_subhalo_index,
            'is_central': is_central.astype(int),
            
            # 14 Properties
            'Mg': Mg,
            'MBH': MBH,
            'Mstar': Mstar,
            'Mt': Mt,
            'Vmax': Vmax,
            'sigma_v': sigma_v,
            'Zg': Zg,
            'Zstar': Zstar,
            'SFR': SFR,
            'J': J,
            'Rstar': Rstar,
            'Rt': Rt,
            'Rmax': Rmax,
            
            # Position & Velocity (3 components each)
            'pos_x': SubhaloPos[:, 0],
            'pos_y': SubhaloPos[:, 1],
            'pos_z': SubhaloPos[:, 2],
            'vel_x': SubhaloVel[:, 0],
            'vel_y': SubhaloVel[:, 1],
            'vel_z': SubhaloVel[:, 2],
        }
        
        df = pd.DataFrame(data)
        
        # ===== Save to CSV =====
        output_dir.mkdir(exist_ok=True)
        output_csv_path = output_dir / f"Astrid_{sim_name}_groups_090_processed.csv"
        
        df.to_csv(output_csv_path, index=False)
        
        print(f"Processed {len(indexes)} galaxies ({num_centrals} centrals, {num_satellites} satellites)")
        print(f"Output: {output_csv_path.name}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Processing failed: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 process_hdf5_to_csv.py <hdf5_path> <sim_name> <output_dir>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    sim_name = sys.argv[2]
    output_dir = sys.argv[3]
    
    exit_code = process_hdf5_to_csv(hdf5_path, sim_name, output_dir)
    sys.exit(exit_code)
