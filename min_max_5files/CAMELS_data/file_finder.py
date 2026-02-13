import pandas as pd
import numpy as np

# --- CONFIGURATION ---
params_file = 'CAMELS_data\CosmoAstroSeed_Astrid_L25n256_SB7.txt'
col_names = ['Name', 'Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1',
             'A_SN2', 'A_AGN2', 'Omega_b', 'seed']

# Parameters of interest
target_params = ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2', 'Omega_b']
# ---------------------


def find_extreme_simulations():
    try:
        df = pd.read_csv(params_file, sep='\\s+', comment='#', names=col_names)
    except FileNotFoundError:
        print(f"Error: File '{params_file}' not found.")
        return

    # 1. Normalize data (0–1 scaling)
    subset = df[target_params].values
    min_vals = np.min(subset, axis=0)
    max_vals = np.max(subset, axis=0)
    
    print('\n')
    print(f"Omega_m min: {min_vals[0]:.5f}, max: {max_vals[0]:.5f}")
    print(f"sigma_8 min: {min_vals[1]:.5f}, max: {max_vals[1]:.5f}")
    print(f"A_SN1 min: {min_vals[2]:.5f}, max: {max_vals[2]:.5f}")
    print(f"A_AGN1 min: {min_vals[3]:.5f}, max: {max_vals[3]:.5f}")
    print(f"A_SN2 min: {min_vals[4]:.5f}, max: {max_vals[4]:.5f}")
    print(f"A_AGN2 min: {min_vals[5]:.5f}, max: {max_vals[5]:.5f}")
    print(f"Omega_b min: {min_vals[6]:.5f}, max: {max_vals[6]:.5f}")
    print('\n')

    # Calculate 25% and 75% values for each parameter
    percentile_25 = min_vals + 0.25 * (max_vals - min_vals)
    percentile_75 = min_vals + 0.75 * (max_vals - min_vals)

    print(f"Omega_m 25%: {percentile_25[0]:.5f}, 75%: {percentile_75[0]:.5f}")
    print(f"sigma_8 25%: {percentile_25[1]:.5f}, 75%: {percentile_75[1]:.5f}")
    print(f"A_SN1 25%: {percentile_25[2]:.5f}, 75%: {percentile_75[2]:.5f}")
    print(f"A_AGN1 25%: {percentile_25[3]:.5f}, 75%: {percentile_75[3]:.5f}")
    print(f"A_SN2 25%: {percentile_25[4]:.5f}, 75%: {percentile_75[4]:.5f}")
    print(f"A_AGN2 25%: {percentile_25[5]:.5f}, 75%: {percentile_75[5]:.5f}")
    print(f"Omega_b 25%: {percentile_25[6]:.5f}, 75%: {percentile_75[6]:.5f}")
    print('\n')

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    norm_data = (subset - min_vals) / range_vals

    # 2. Define targets in normalized space
    targets = [
        ("ALL MINIMUMS (0%)", np.zeros(len(target_params))),
        ("LOW (25%)",        np.full(len(target_params), 0.25)),
        ("HIGH (75%)",       np.full(len(target_params), 0.75)),
        ("ALL MAXIMUMS (100%)", np.ones(len(target_params))),
    ]

    print(f"{'Target':<20} | {'Name':<10} | {'Actual Values'}")
    print("-" * 75)

    found_names = []

    for label, target in targets:
        # Euclidean distance in normalized space
        distances = np.sqrt(np.sum((norm_data - target) ** 2, axis=1))

        best_idx = np.argmin(distances)
        best_name = df.iloc[best_idx]['Name']
        found_names.append(best_name)

        vals = df.iloc[best_idx][target_params].values
        val_str = ", ".join(f"{x:.5f}" for x in vals)

        print(f"{label:<20} | {best_name:<10} | {val_str}")
    
    print("-" * 75)
    print('\n')
    print(f"Files to download: {found_names}")
    print('\n')

if __name__ == "__main__":
    find_extreme_simulations()
