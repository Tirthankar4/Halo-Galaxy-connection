"""
Diagnostic script to check for satellites with higher stellar mass than their centrals
in CAMELS Astrid simulation data.
"""

import os
import glob
import pandas as pd
import numpy as np
from preprocessing import process_hdf5_to_csv

# Find all HDF5 files in CAMELS_data folder
camels_data_dir = 'CAMELS_data'
hdf5_files = glob.glob(os.path.join(camels_data_dir, '*_groups_090.hdf5'))

print("="*80)
print("DIAGNOSTIC: Satellites with Higher Stellar Mass than Centrals")
print("="*80)
print(f"\nFound {len(hdf5_files)} HDF5 files to analyze\n")

all_problem_cases = []
file_summaries = []

for hdf5_file in sorted(hdf5_files):
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(hdf5_file)}")
    print(f"{'='*80}")
    
    try:
        # Process the file to get CSV
        df = process_hdf5_to_csv(hdf5_file)
        
        # Get satellites
        satellites = df[df['is_central'] == 0].copy()
        
        if len(satellites) == 0:
            print("No satellites found in this file.")
            file_summaries.append({
                'file': os.path.basename(hdf5_file),
                'total_galaxies': len(df),
                'centrals': (df['is_central'] == 1).sum(),
                'satellites': 0,
                'problem_cases': 0,
                'problem_percentage': 0.0
            })
            continue
        
        # Check each satellite against its central
        problem_cases = []
        for idx, sat in satellites.iterrows():
            # Find the central galaxy
            central = df[df['subhalo_index'] == sat['central_subhalo_index']].iloc[0]
            
            if sat['Mstar'] > central['Mstar']:
                problem_cases.append({
                    'file': os.path.basename(hdf5_file),
                    'halo_id': sat['halo_id'],
                    'satellite_subhalo_index': sat['subhalo_index'],
                    'central_subhalo_index': central['subhalo_index'],
                    'satellite_Mstar': sat['Mstar'],
                    'central_Mstar': central['Mstar'],
                    'Mstar_ratio': sat['Mstar'] / central['Mstar'],
                    'satellite_Mt': sat['Mt'],
                    'central_Mt': central['Mt'],
                    'Mt_ratio': central['Mt'] / sat['Mt'] if sat['Mt'] > 0 else np.inf,
                    'satellite_Vmax': sat['Vmax'],
                    'central_Vmax': central['Vmax'],
                })
        
        # Store results
        problem_df = pd.DataFrame(problem_cases)
        all_problem_cases.append(problem_df)
        
        # Print summary for this file
        num_problem = len(problem_cases)
        num_satellites = len(satellites)
        problem_pct = (num_problem / num_satellites * 100) if num_satellites > 0 else 0
        
        print(f"\nFile Summary:")
        print(f"  Total galaxies: {len(df)}")
        print(f"  Centrals: {(df['is_central'] == 1).sum()}")
        print(f"  Satellites: {num_satellites}")
        print(f"  Problem cases (satellite Mstar > central Mstar): {num_problem}")
        print(f"  Percentage of satellites: {problem_pct:.2f}%")
        
        if num_problem > 0:
            print(f"\n  Top 5 most extreme cases:")
            top5 = problem_df.nlargest(5, 'Mstar_ratio')
            for i, row in top5.iterrows():
                print(f"    {i+1}. Halo {int(row['halo_id'])}: "
                      f"Sat Mstar={row['satellite_Mstar']:.2e}, "
                      f"Cent Mstar={row['central_Mstar']:.2e}, "
                      f"Ratio={row['Mstar_ratio']:.2f}x")
            
            # Check if centrals have higher total mass
            central_higher_Mt = (problem_df['central_Mt'] > problem_df['satellite_Mt']).sum()
            print(f"\n  Centrals with higher total mass: {central_higher_Mt}/{num_problem} "
                  f"({central_higher_Mt/num_problem*100:.1f}%)")
            
            # Check if centrals have higher Vmax
            central_higher_Vmax = (problem_df['central_Vmax'] > problem_df['satellite_Vmax']).sum()
            print(f"  Centrals with higher Vmax: {central_higher_Vmax}/{num_problem} "
                  f"({central_higher_Vmax/num_problem*100:.1f}%)")
        
        file_summaries.append({
            'file': os.path.basename(hdf5_file),
            'total_galaxies': len(df),
            'centrals': (df['is_central'] == 1).sum(),
            'satellites': num_satellites,
            'problem_cases': num_problem,
            'problem_percentage': problem_pct
        })
        
    except Exception as e:
        print(f"ERROR processing {hdf5_file}: {e}")
        import traceback
        traceback.print_exc()

# Combine all problem cases
if all_problem_cases:
    combined_problems = pd.concat(all_problem_cases, ignore_index=True)
    
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL FILES")
    print(f"{'='*80}")
    
    total_satellites = sum(s['satellites'] for s in file_summaries)
    total_problems = len(combined_problems)
    
    print(f"\nTotal satellites across all files: {total_satellites}")
    print(f"Total problem cases: {total_problems}")
    print(f"Overall percentage: {total_problems/total_satellites*100:.2f}%")
    
    if total_problems > 0:
        print(f"\nMass ratio statistics:")
        print(f"  Mean ratio (satellite/central): {combined_problems['Mstar_ratio'].mean():.2f}x")
        print(f"  Median ratio: {combined_problems['Mstar_ratio'].median():.2f}x")
        print(f"  Max ratio: {combined_problems['Mstar_ratio'].max():.2f}x")
        print(f"  Min ratio: {combined_problems['Mstar_ratio'].min():.2f}x")
        
        print(f"\nTotal mass (central/satellite) ratio statistics:")
        print(f"  Mean ratio: {combined_problems['Mt_ratio'].mean():.2f}x")
        print(f"  Median ratio: {combined_problems['Mt_ratio'].median():.2f}x")
        print(f"  Centrals with higher Mt: {(combined_problems['central_Mt'] > combined_problems['satellite_Mt']).sum()}/{total_problems} "
              f"({(combined_problems['central_Mt'] > combined_problems['satellite_Mt']).sum()/total_problems*100:.1f}%)")
        
        print(f"\nVmax (central/satellite) statistics:")
        print(f"  Centrals with higher Vmax: {(combined_problems['central_Vmax'] > combined_problems['satellite_Vmax']).sum()}/{total_problems} "
              f"({(combined_problems['central_Vmax'] > combined_problems['satellite_Vmax']).sum()/total_problems*100:.1f}%)")
        
        print(f"\nTop 10 most extreme cases across all files:")
        top10 = combined_problems.nlargest(10, 'Mstar_ratio')
        for i, row in top10.iterrows():
            print(f"  {i+1}. {row['file']} - Halo {int(row['halo_id'])}: "
                  f"Sat Mstar={row['satellite_Mstar']:.2e}, "
                  f"Cent Mstar={row['central_Mstar']:.2e}, "
                  f"Ratio={row['Mstar_ratio']:.2f}x, "
                  f"Cent Mt/Sat Mt={row['Mt_ratio']:.2f}x")
        
        # Save detailed results to CSV
        output_file = 'diagnostic_satellite_mass_results.csv'
        combined_problems.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    # Save file summaries
    summary_df = pd.DataFrame(file_summaries)
    summary_file = 'diagnostic_file_summaries.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"File summaries saved to: {summary_file}")
    
else:
    print("\n\nNo problem cases found across all files!")
    print("All satellites have lower stellar mass than their centrals.")

print(f"\n{'='*80}")
print("Diagnostic complete!")
print(f"{'='*80}")
