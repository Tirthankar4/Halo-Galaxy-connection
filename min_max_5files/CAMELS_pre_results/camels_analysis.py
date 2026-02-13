import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
import glob
from pathlib import Path

def analyze_simulation(hdf5_file_path, output_base_dir):
    """
    Analyze a single CAMELS simulation file and save results
    """
    # Extract simulation name from filename
    sim_name = Path(hdf5_file_path).stem  # e.g., 'Astrid_CV_0_groups_090'

    # Create output directory for this simulation
    output_dir = os.path.join(output_base_dir, sim_name)
    os.makedirs(output_dir, exist_ok=True)

    # Open file to save text results
    results_file = os.path.join(output_dir, 'analysis_results.txt')

    print(f"\nProcessing: {sim_name}")

    with open(results_file, 'w') as f:
        f.write(f"Analysis Results for {sim_name}\n")
        f.write("="*50 + "\n\n")

        # Load data from HDF5 file
        try:
            with h5py.File(hdf5_file_path, 'r') as hf:
                # Halo data
                M_h = hf['Group/Group_M_Crit200'][:]*1e10  # Msun/h
                R_h = hf['Group/Group_R_Crit200'][:]  # kpc/h
                V_h = hf['Group/GroupVel'][:]  # km/s
                V_h = np.linalg.norm(V_h, axis=1)
                ID_r = hf['Group/GroupFirstSub'][:]
                ID_r_original = ID_r.copy()  # Save original before filtering
                ID_h = np.arange(0, M_h.shape[0], 1, dtype=float)

                # Galaxy data
                SM = hf['Subhalo/SubhaloMassType'][:, 4]*1e10  # Msun/h
                Colour = hf['Subhalo/SubhaloStellarPhotometrics'][:, 4] - \
                         hf['Subhalo/SubhaloStellarPhotometrics'][:, 5]  # g - r
                SFR = hf['Subhalo/SubhaloSFR'][:]*1e10
                SR = hf['Subhalo/SubhaloHalfmassRadType'][:, 4]  # kpc/h
                ID_g = np.array(hf['Subhalo/SubhaloGrNr'])

            # Remove halos without galaxies
            indexes = np.where(ID_r != -1)[0]
            M_h = M_h[indexes]
            R_h = R_h[indexes]
            V_h = V_h[indexes]
            ID_h = ID_h[indexes]

            # Create halo catalog
            halos = pd.DataFrame({
                'M_h': M_h,
                'R_h': R_h,
                'V_h': V_h,
                'ID': ID_h
            })

            # Filter galaxies by stellar mass
            gal_indexes = np.where(SM > 1.3e8)[0]
            gal_indexes_original = gal_indexes.copy()  # Save original indices
            SM = SM[gal_indexes]
            Colour = Colour[gal_indexes]
            SFR = SFR[gal_indexes]
            SR = SR[gal_indexes]
            ID_g = ID_g[gal_indexes]

            # Create galaxy catalog
            gals = pd.DataFrame({
                'SM': SM,
                'SR': SR,
                'SFR': SFR,
                'Colour': Colour,
                'ID': ID_g
            })

            # Intersection check
            intersection_count = np.intersect1d(ID_h, ID_g).shape[0]
            msg = f"Number of halos with galaxies: {intersection_count}\n"
            print(msg, end='')
            f.write(msg)

            # Basic galaxy checks
            zero_radius_count = (gals['SR'] == 0).sum()
            min_mass = gals['SM'].min()
            msg = f"Number of galaxies with zero radius: {zero_radius_count}\n"
            print(msg, end='')
            f.write(msg)
            msg = f"Minimum galaxy stellar mass: {min_mass:.4e}\n\n"
            print(msg, end='')
            f.write(msg)

            # Central vs Satellite classification - CORRECTED
            valid_halo_ids = (ID_g >= 0) & (ID_g < len(ID_r_original))
            is_central = np.zeros(len(gal_indexes_original), dtype=bool)
            is_central[valid_halo_ids] = (
                ID_r_original[ID_g[valid_halo_ids].astype(int)] == gal_indexes_original[valid_halo_ids]
            )

            num_centrals = is_central.sum()

            msg = f"Number of total galaxies: {gals.shape[0]}\n"
            print(msg, end='')
            f.write(msg)

            msg = f"Number of central galaxies: {num_centrals}\n"
            print(msg, end='')
            f.write(msg)

            msg = f"Number of satellite galaxies: {len(gals) - num_centrals}\n\n"
            print(msg, end='')
            f.write(msg)

            gals['is_central'] = is_central

            # Top 100 halos analysis
            top_100_ids = halos.nlargest(100, 'M_h')['ID']
            top_gals = gals[gals['ID'].isin(top_100_ids)]
            num_satellites = (~top_gals['is_central']).sum()

            msg = f"Number of satellites in top 100 halos: {num_satellites}\n"
            print(msg, end='')
            f.write(msg)

            msg = f"Average satellites per top halo: {num_satellites / len(top_100_ids):.4f}\n"
            print(msg, end='')
            f.write(msg)

            # Merge for plotting
            df_plot = pd.merge(gals, halos, on='ID')

            # ============== PLOT 1: Stellar Mass vs Halo Mass ==============
            plt.figure(figsize=(8, 6))
            plt.scatter(df_plot['M_h'], df_plot['SM'], s=10, alpha=0.3,
                       label='All Galaxies', color='gray')
            plt.scatter(df_plot[df_plot['is_central']]['M_h'],
                       df_plot[df_plot['is_central']]['SM'],
                       s=15, alpha=0.6, label='Centrals', color='blue')
            plt.scatter(df_plot[~df_plot['is_central']]['M_h'],
                       df_plot[~df_plot['is_central']]['SM'],
                       s=15, alpha=0.6, label='Satellites', color='red')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Halo Mass ($M_h$) [$M_\\odot/h$]')
            plt.ylabel('Stellar Mass ($SM$) [$M_\\odot/h$]')
            plt.title(f'{sim_name} - Stellar Mass vs Halo Mass')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)

            # Save scatter plot
            scatter_file = os.path.join(output_dir, 'stellar_vs_halo_mass.png')
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()

            # ============== PLOT 2: Distribution Histograms ==============
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            plot_configs = [
                {'data': 'SM', 'label': 'Stellar Mass', 'ax': axes[0]},
                {'data': 'SR', 'label': 'Radius', 'ax': axes[1]},
                {'data': 'M_h', 'label': 'Halo Mass', 'ax': axes[2]}
            ]

            for config in plot_configs:
                ax = config['ax']
                col = config['data']
                bins = np.logspace(
                    np.log10(df_plot[col].min()),
                    np.log10(df_plot[col].max()),
                    30
                )

                ax.hist(df_plot[col], bins=bins, alpha=0.3, label='All', color='gray')
                ax.hist(df_plot[df_plot['is_central']][col], bins=bins,
                       histtype='step', label='Centrals', color='blue', lw=2)
                ax.hist(df_plot[~df_plot['is_central']][col], bins=bins,
                       histtype='step', label='Satellites', color='red', lw=2)

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel(config['label'])
                ax.set_ylabel('Count (Log)')
                ax.legend()

            plt.suptitle(f'{sim_name} - Distribution Analysis')
            plt.tight_layout()

            # Save histogram plot
            hist_file = os.path.join(output_dir, 'distribution_histograms.png')
            plt.savefig(hist_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Results saved to: {output_dir}\n")

            f.write(f"\nPlots saved to:\n")
            f.write(f"  - stellar_vs_halo_mass.png\n")
            f.write(f"  - distribution_histograms.png\n")

        except Exception as e:
            error_msg = f"Error processing {sim_name}: {str(e)}\n"
            print(error_msg)
            f.write(error_msg)

def main():
    """
    Main function to process all HDF5 files in a directory
    """
    # === CONFIGURATION ===
    # Change this to your folder containing HDF5 files
    input_folder = r'C:\Users\tirth\Documents\Projects\Halo - galaxy connection\CAMELS_data'

    # Change this to where you want to save results
    output_folder = r'C:\Users\tirth\Documents\Projects\Halo - galaxy connection\CAMELS_results'
    # =====================

    # Find all HDF5 files
    hdf5_files = glob.glob(os.path.join(input_folder, '*.hdf5'))

    if not hdf5_files:
        print(f"No HDF5 files found in {input_folder}")
        return

    print(f"Found {len(hdf5_files)} HDF5 file(s)")
    print(f"Output directory: {output_folder}\n")

    # Create output base directory
    os.makedirs(output_folder, exist_ok=True)

    # Process each file
    for hdf5_file in hdf5_files:
        analyze_simulation(hdf5_file, output_folder)

    print(f"\nAll done! Results saved to {output_folder}")

if __name__ == "__main__":
    main()
