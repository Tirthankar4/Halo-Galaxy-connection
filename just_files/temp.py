# Complete script to check halo mass correlation
import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy import stats

# Load parameter file
param_df = pd.read_csv(r'C:\Users\tirth\Documents\Projects\Halo - galaxy connection\min_max_5files\CAMELS_data\CosmoAstroSeed_Astrid_L25n256_SB7.txt', 
                       sep=r'\s+', comment=None, skiprows=0)
if param_df.columns[0].startswith('#'):
    param_df.rename(columns={param_df.columns[0]: param_df.columns[0][1:]}, inplace=True)
param_df.set_index('Name', inplace=True)

print("Parameter correlations in SB7 suite:")
print(param_df[['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1']].corr())
print("\n" + "="*70)

# Load and combine CSV files
csv_files = list(Path('CAMELS_processed').glob('*.csv'))[:50]  # First 50 for speed
all_data = []

for csv_file in csv_files:
    pattern = r'Astrid_(.+?)_groups'
    match = re.search(pattern, csv_file.name)
    if not match:
        continue
    
    sim_name = match.group(1)
    if sim_name not in param_df.index:
        continue
    
    df = pd.read_csv(csv_file)
    
    # Add parameters
    df['Omega_m'] = param_df.loc[sim_name, 'Omega_m']
    df['sigma_8'] = param_df.loc[sim_name, 'sigma_8']
    
    # Keep only centrals
    df_central = df[df['is_central'] == 1]
    all_data.append(df_central)

combined = pd.concat(all_data, ignore_index=True)

print(f"\nCombined {len(all_data)} files")
print(f"Total central galaxies: {len(combined)}")

# Check correlations for all mass properties
print("\n" + "="*70)
print("CORRELATION ANALYSIS: Omega_m vs Mass Properties")
print("="*70)

mass_properties = {
    'Mstar': 'Stellar Mass',
    'Mt': 'Total Halo Mass',
    'Mg': 'Gas Mass',
    'MBH': 'Black Hole Mass'
}

for prop, label in mass_properties.items():
    if prop in combined.columns:
        mask = (combined['Omega_m'].notna()) & (combined[prop].notna()) & (combined[prop] > 0)
        if mask.sum() > 100:
            r, p = stats.pearsonr(combined.loc[mask, 'Omega_m'], combined.loc[mask, prop])
            r_log, p_log = stats.pearsonr(combined.loc[mask, 'Omega_m'], 
                                          np.log10(combined.loc[mask, prop]))
            
            print(f"\n{label} ({prop}):")
            print(f"  PCC (linear):     {r:+.4f} (p={p:.2e})")
            print(f"  PCC (log10):      {r_log:+.4f} (p={p_log:.2e})")
            print(f"  Valid galaxies:   {mask.sum():,}")
            
            # Interpretation
            if abs(r) > 0.2 or abs(r_log) > 0.2:
                print(f"  ✅ STRONG correlation detected")
            elif abs(r) > 0.1 or abs(r_log) > 0.1:
                print(f"  ⚠️  MODERATE correlation")
            else:
                print(f"  ❌ WEAK correlation")

# Check kinematic properties too
print("\n" + "="*70)
print("CORRELATION ANALYSIS: Omega_m vs Kinematic Properties")
print("="*70)

kinematic_properties = {
    'Vmax': 'Maximum Circular Velocity',
    'sigma_v': 'Velocity Dispersion'
}

for prop, label in kinematic_properties.items():
    if prop in combined.columns:
        mask = (combined['Omega_m'].notna()) & (combined[prop].notna()) & (combined[prop] > 0)
        if mask.sum() > 100:
            r, p = stats.pearsonr(combined.loc[mask, 'Omega_m'], combined.loc[mask, prop])
            
            print(f"\n{label} ({prop}):")
            print(f"  PCC: {r:+.4f} (p={p:.2e})")
            print(f"  Valid galaxies: {mask.sum():,}")
            
            if abs(r) > 0.2:
                print(f"  ✅ STRONG correlation")
            elif abs(r) > 0.1:
                print(f"  ⚠️  MODERATE correlation")
            else:
                print(f"  ❌ WEAK correlation")

# Mass-dependent analysis
print("\n" + "="*70)
print("MASS-DEPENDENT ANALYSIS: Does Omega_m correlation depend on galaxy mass?")
print("="*70)

mass_bins = [
    (1.3e8, 1e9, "Low mass (Dwarfs)"),
    (1e9, 1e10, "Intermediate mass"),
    (1e10, 1e12, "High mass (MW-like and above)")
]

for m_min, m_max, label in mass_bins:
    subset = combined[(combined['Mstar'] >= m_min) & (combined['Mstar'] < m_max)]
    if len(subset) > 100:
        mask = (subset['Omega_m'].notna()) & (subset['Mstar'].notna())
        if mask.sum() > 100:
            r, p = stats.pearsonr(subset.loc[mask, 'Omega_m'], subset.loc[mask, 'Mstar'])
            print(f"\n{label}:")
            print(f"  N_galaxies: {len(subset):,}")
            print(f"  PCC: {r:+.4f} (p={p:.2e})")

print("\n" + "="*70)

# Add this to your analysis
import matplotlib.pyplot as plt

# Count galaxies per simulation
galaxy_counts = combined.groupby('Omega_m').size()
plt.figure(figsize=(8, 5))
plt.scatter(galaxy_counts.index, galaxy_counts.values, alpha=0.6)
plt.xlabel('Omega_m')
plt.ylabel('Number of Galaxies')
plt.title('Galaxy Count vs Omega_m')
#plt.savefig('galaxy_count_vs_omega_m.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

r_count = stats.pearsonr(galaxy_counts.index, galaxy_counts.values)[0]
print(f"\nGalaxy count vs Omega_m: PCC = {r_count:.3f}")