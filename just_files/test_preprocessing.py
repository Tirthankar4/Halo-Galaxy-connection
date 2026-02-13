"""
Test script to verify the preprocessing.py implementation.
This demonstrates how to use the process_hdf5_to_csv function.
"""

from preprocessing import process_hdf5_to_csv
import pandas as pd

# Example 1: Process a single HDF5 file
# Replace this path with your actual HDF5 file path
hdf5_file = r'C:\Users\tirth\Documents\Projects\Halo - galaxy connection\CAMELS_data\Astrid_SB7_719_groups_090.hdf5'

# Process the file (will auto-generate CSV name)
df = process_hdf5_to_csv(hdf5_file)

print("\n" + "="*60)
print("CSV Structure Overview")
print("="*60)
print(f"\nTotal columns: {len(df.columns)}")
print(f"Total rows: {len(df)}")

print("\n" + "-"*60)
print("Column names:")
print("-"*60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "-"*60)
print("Sample data (first 3 rows):")
print("-"*60)
print(df.head(3).to_string())

print("\n" + "-"*60)
print("Central vs Satellite breakdown:")
print("-"*60)
print(f"Central galaxies: {(df['is_central'] == 1).sum()}")
print(f"Satellite galaxies: {(df['is_central'] == 0).sum()}")

print("\n" + "-"*60)
print("Example: Finding a satellite's central galaxy")
print("-"*60)
# Get a satellite galaxy
satellites = df[df['is_central'] == 0]
if len(satellites) > 0:
    satellite = satellites.iloc[0]
    print(f"\nSatellite galaxy:")
    print(f"  - subhalo_index: {satellite['subhalo_index']}")
    print(f"  - halo_id: {satellite['halo_id']}")
    print(f"  - Mstar: {satellite['Mstar']:.2e} Msun/h")
    print(f"  - Position: ({satellite['pos_x']:.2f}, {satellite['pos_y']:.2f}, {satellite['pos_z']:.2f}) kpc/h")
    
    # Find its central galaxy
    central = df[df['subhalo_index'] == satellite['central_subhalo_index']].iloc[0]
    print(f"\nIts central galaxy:")
    print(f"  - subhalo_index: {central['subhalo_index']}")
    print(f"  - halo_id: {central['halo_id']}")
    print(f"  - Mstar: {central['Mstar']:.2e} Msun/h")
    print(f"  - Position: ({central['pos_x']:.2f}, {central['pos_y']:.2f}, {central['pos_z']:.2f}) kpc/h")
    
    # Calculate distance between satellite and central
    import numpy as np
    dx = satellite['pos_x'] - central['pos_x']
    dy = satellite['pos_y'] - central['pos_y']
    dz = satellite['pos_z'] - central['pos_z']
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    print(f"\nDistance from central: {distance:.2f} kpc/h")

print("\n" + "-"*60)
print("Example: Get all galaxies in a halo")
print("-"*60)
# Pick a halo with multiple galaxies
halo_counts = df['halo_id'].value_counts()
multi_galaxy_halo = halo_counts[halo_counts > 1].index[0]
halo_galaxies = df[df['halo_id'] == multi_galaxy_halo]
print(f"\nHalo {multi_galaxy_halo} has {len(halo_galaxies)} galaxies:")
print(f"  - Centrals: {(halo_galaxies['is_central'] == 1).sum()}")
print(f"  - Satellites: {(halo_galaxies['is_central'] == 0).sum()}")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)