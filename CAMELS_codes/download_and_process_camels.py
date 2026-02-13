#!/usr/bin/env python3
"""
download_and_process_camels_optimized.py  
OPTIMIZED VERSION - Faster downloads with reduced overhead
"""

import subprocess
import os
import sys
import platform
from pathlib import Path
import time
from datetime import datetime
import re
import h5py
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

CAMELS_ENDPOINT = "58bdcd24-6590-11ec-9b60-f9dfb1abb183"  # Main CAMELS
LOCAL_ENDPOINT = "9625c23e-e721-11f0-94d5-0213754b0ca1"  # Your endpoint ID

CAMELS_BASE_PATH = "/Sims/Astrid/L25n256/SB7"

# Local directories (relative to current working directory)
LOCAL_TEMP_DIR = Path("./CAMELS_datas")
OUTPUT_DIR = Path("./CAMELS_processed")
LOG_FILE = Path("./processing_log.txt")

# Globus Connect Personal configuration:
# /~ maps to: C:\\Users\\tirth\\OneDrive\\Documents
# Files download there first, then script moves them to CAMELS_datas/
# Working from: C:\\Users\\tirth\\Documents\\Projects\\Halo - galaxy connection
GLOBUS_DOWNLOAD_DIR = "/~"

START_SIM = 0
END_SIM = 1023  # Change to 4 for 5 files, 1023 for all

MAX_CONCURRENT_DOWNLOADS = 50  # Number of files to download simultaneously

TRANSFER_TIMEOUT = 1800  # 30 minutes per file

# OPTIMIZATION SETTINGS
STATUS_CHECK_INTERVAL = 30  # Increased from 10 to reduce overhead
SYNC_LEVEL = "mtime"  # Changed from "checksum" - faster, still reliable

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def log_message(message, also_print=True):
    """Write message to log file and optionally print"""
    if also_print:
        print(message)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")

def check_prerequisites():
    """Check if Globus CLI is installed and user is logged in"""

    if platform.system() == 'Windows':
        check_cmd = "where globus"
    else:
        check_cmd = "which globus"

    success, _, _ = run_command(check_cmd, check=False)
    if not success:
        print("ERROR: Globus CLI not found.")
        return False

    success, stdout, stderr = run_command("globus whoami", check=False)
    if not success:
        print("ERROR: Not logged into Globus.")
        return False

    print(f"Logged in as: {stdout.strip()}")
    return True

def check_globus_connect_personal():
    """Check if Globus Connect Personal is running"""
    print("\nChecking Globus Connect Personal status...")

    success, stdout, stderr = run_command("globus endpoint local-id", check=False)
    if success:
        print(f"  [OK] Globus Connect Personal is running")
        print(f"  Local endpoint: {stdout.strip()}")
        return True
    else:
        print("  [ERROR] Globus Connect Personal may not be running!")
        return False

def get_already_processed_simulations():
    """
    Scan OUTPUT_DIR for existing CSV files and return set of processed simulation numbers.
    Parses filenames like: Astrid_SB7_14_groups_090_processed.csv
    """
    if not OUTPUT_DIR.exists():
        return set()
    
    processed_sims = set()
    pattern = re.compile(r'Astrid_SB7_(\d+)_groups_090_processed\.csv')
    
    for csv_file in OUTPUT_DIR.glob('*.csv'):
        match = pattern.match(csv_file.name)
        if match:
            sim_num = int(match.group(1))
            processed_sims.add(sim_num)
    
    return processed_sims

def download_file(sim_name, remote_path, local_path):
    """
    Download file to OneDrive Documents folder, then move to project
    OPTIMIZED: Reduced status check frequency and improved progress reporting
    """

    # Download to OneDrive Documents (where Globus /~ maps to)
    # local_path will be like: CAMELS_datas/groups_090_SB7_0.hdf5 (includes sim_name)
    # We download to OneDrive Documents first, then move it to CAMELS_datas/

    filename = local_path.name  # e.g., "groups_090_SB7_0.hdf5"
    globus_path = f"{GLOBUS_DOWNLOAD_DIR}/{filename}"

    # Submit transfer with optimized settings
    cmd = (f'globus transfer "{CAMELS_ENDPOINT}:{remote_path}" '
           f'"{LOCAL_ENDPOINT}:{globus_path}" '
           f'--label "CAMELS {sim_name}" '
           f'--sync-level {SYNC_LEVEL} '  # Using mtime instead of checksum
           f'--notify off')

    success, stdout, stderr = run_command(cmd, check=False)

    if not success:
        return False, f"Transfer submission failed: {stderr}"

    # Extract task ID
    task_id = None
    for line in stdout.split('\n'):
        if 'Task ID:' in line:
            task_id = line.split('Task ID:')[1].strip()
            break

    if not task_id:
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        matches = re.findall(uuid_pattern, stdout)
        if matches:
            task_id = matches[0]

    if not task_id:
        return False, f"Could not extract task ID"

    # Wait for transfer
    print(f" [task: {task_id[:8]}...]\n", end="", flush=True)

    start_time = time.time()
    last_bytes = 0
    last_check_time = start_time
    stall_count = 0
    check_count = 0

    while time.time() - start_time < TRANSFER_TIMEOUT:
        check_count += 1
        status_cmd = f'globus task show {task_id}'
        success, stdout, stderr = run_command(status_cmd, check=False)

        if not success:
            return False, f"Failed to check task status"

        # Parse status
        status = "UNKNOWN"
        bytes_transferred = 0
        total_bytes = 0

        for line in stdout.split('\n'):
            if 'Status:' in line:
                status = line.split('Status:')[1].strip()
            elif 'Bytes Transferred:' in line:
                try:
                    bytes_transferred = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Total Bytes:' in line:
                try:
                    total_bytes = int(line.split(':')[1].strip())
                except:
                    pass

        # Calculate transfer rate
        elapsed = time.time() - start_time
        check_elapsed = time.time() - last_check_time
        
        # Show progress with transfer rate
        if total_bytes > 0 and bytes_transferred > 0:
            progress_pct = (bytes_transferred / total_bytes) * 100
            mb_transferred = bytes_transferred / (1024**2)
            mb_total = total_bytes / (1024**2)
            
            # Calculate instantaneous and average rates
            if bytes_transferred > last_bytes and check_elapsed > 0:
                instant_rate = ((bytes_transferred - last_bytes) / check_elapsed) / (1024**2)
                avg_rate = (bytes_transferred / elapsed) / (1024**2)
                
                print(f"\r[1/4] Downloading {sim_name}... "
                      f"{progress_pct:.1f}% ({mb_transferred:.1f}/{mb_total:.1f} MB) "
                      f"@ {instant_rate:.2f} MB/s (avg: {avg_rate:.2f} MB/s) "
                      f"[{int(elapsed)}s]", end="", flush=True)
            else:
                print(f"\r[1/4] Downloading {sim_name}... "
                      f"{progress_pct:.1f}% ({mb_transferred:.1f}/{mb_total:.1f} MB) "
                      f"[{int(elapsed)}s]", end="", flush=True)
        else:
            print(".", end="", flush=True)

        last_check_time = time.time()

        # Check for stall
        if bytes_transferred > 0 and bytes_transferred == last_bytes:
            stall_count += 1
            if stall_count > 12:  # More lenient with longer check interval
                return False, f"Transfer stalled at {bytes_transferred} bytes"
        else:
            stall_count = 0
            last_bytes = bytes_transferred

        # Check completion
        if status == "SUCCEEDED":
            # File downloaded to OneDrive Documents folder, now move to CAMELS_datas folder
            # Globus /~ maps to C:\\Users\\<username>\\OneDrive\\Documents
            onedrive_docs = Path.home() / "OneDrive" / "Documents"
            downloaded_file = onedrive_docs / filename
            
            if not downloaded_file.exists():
                # Fallback: try regular Documents folder
                downloaded_file = Path.home() / "Documents" / filename
                if not downloaded_file.exists():
                    return False, f"Transfer succeeded but file not found at {onedrive_docs / filename}"

            # Move to proper location
            local_path.parent.mkdir(exist_ok=True)
            downloaded_file.rename(local_path)

            file_size_mb = local_path.stat().st_size / (1024**2)
            avg_rate = (local_path.stat().st_size / elapsed) / (1024**2)
            return True, f"Download successful ({file_size_mb:.1f} MB in {int(elapsed)}s @ {avg_rate:.2f} MB/s avg)"

        elif status in ["FAILED", "INACTIVE"]:
            return False, f"Transfer failed with status: {status}"

        elif status in ["ACTIVE", "PENDING"]:
            time.sleep(STATUS_CHECK_INTERVAL)  # Optimized: less frequent checks
        else:
            time.sleep(STATUS_CHECK_INTERVAL)

    return False, f"Transfer timed out after {TRANSFER_TIMEOUT/60:.0f} minutes"

def process_hdf5_to_csv(hdf5_path, sim_name):
    """
    Convert CAMELS HDF5 galaxy catalog to CSV.
    
    Parameters:
    -----------
    hdf5_path : Path
        Path to the HDF5 file (e.g., 'groups_090.hdf5')
    sim_name : str
        Simulation name for generating output filename
    
    Returns:
    --------
    tuple (bool, str)
        Success status and message
    """
    
    try:
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
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_csv_path = OUTPUT_DIR / f"Astrid_{sim_name}_groups_090_processed.csv"
        
        df.to_csv(output_csv_path, index=False)
        
        return True, f"Processed {len(indexes)} galaxies ({num_centrals} centrals, {num_satellites} satellites) -> {output_csv_path.name}"
        
    except Exception as e:
        return False, f"Processing failed: {str(e)}"

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("CAMELS Data Processing Pipeline (OPTIMIZED - BATCH MODE)")
    print("="*70)
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"\nOPTIMIZATIONS:")
    print(f"  - Status check interval: {STATUS_CHECK_INTERVAL}s (was 10s)")
    print(f"  - Sync level: {SYNC_LEVEL} (was checksum)")
    print(f"  - Transfer rate monitoring enabled")
    print(f"  - Concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")

    # Setup
    LOCAL_TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(LOG_FILE, 'w') as f:
        f.write(f"CAMELS Processing Started: {datetime.now()}\n")
        f.write("="*70 + "\n")

    if not check_prerequisites():
        sys.exit(1)

    if not check_globus_connect_personal():
        print("\nERROR: Globus Connect Personal not running.\n")
        sys.exit(1)

    print(f"\nProcessing simulations {START_SIM} to {END_SIM}...")
    
    # Check for already processed simulations
    already_processed = get_already_processed_simulations()
    if already_processed:
        print(f"\n[RESUME MODE] Found {len(already_processed)} already processed simulations")
        print(f"Already processed: {sorted(list(already_processed))[:10]}{'...' if len(already_processed) > 10 else ''}")
    
    print(f"Download path: {GLOBUS_DOWNLOAD_DIR}")
    print(f"Working directory: {Path.cwd()}")
    onedrive_docs = Path.home() / "OneDrive" / "Documents"
    print(f"\nNote: Files download to {onedrive_docs}, then moved to CAMELS_datas/")
    print(f"\nBATCH MODE: Downloading {MAX_CONCURRENT_DOWNLOADS} files at a time\n")

    success_count = 0
    failure_count = 0
    skipped_count = 0
    total_start = time.time()

    # Create list of simulations to process (excluding already processed ones)
    sims_to_process = [i for i in range(START_SIM, END_SIM + 1) if i not in already_processed]
    total_to_process = len(sims_to_process)
    
    # Count skipped
    skipped_count = (END_SIM - START_SIM + 1) - total_to_process
    if skipped_count > 0:
        print(f"[SKIPPING] {skipped_count} already processed simulations")
        # Show a sample of what's being skipped
        skipped_list = sorted(list(already_processed))
        if len(skipped_list) <= 10:
            print(f"Already processed: {', '.join([f'SB7_{i}' for i in skipped_list])}")
        else:
            print(f"Already processed (sample): {', '.join([f'SB7_{i}' for i in skipped_list[:10]])} ... and {len(skipped_list)-10} more")
        print()

    # Process in batches
    batch_num = 0
    for batch_start_idx in range(0, len(sims_to_process), MAX_CONCURRENT_DOWNLOADS):
        batch_num += 1
        batch_sims = sims_to_process[batch_start_idx:batch_start_idx + MAX_CONCURRENT_DOWNLOADS]
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_num}: Processing {len(batch_sims)} simulations")
        print(f"Simulations: {', '.join([f'SB7_{i}' for i in batch_sims])}")
        print('='*70)

        # === PHASE 1: DOWNLOAD ALL FILES IN BATCH ===
        print(f"\n[PHASE 1/2] Downloading {len(batch_sims)} files concurrently...")
        
        download_tasks = []  # List of (sim_num, sim_name, task_id, local_path)
        
        for sim_num in batch_sims:
            sim_name = f"SB7_{sim_num}"
            remote_path = f"{CAMELS_BASE_PATH}/{sim_name}/groups_090.hdf5"
            
            # Create unique local filename for this simulation
            local_hdf5 = LOCAL_TEMP_DIR / f"groups_090_{sim_name}.hdf5"
            
            # Submit download (non-blocking)
            filename = local_hdf5.name
            globus_path = f"{GLOBUS_DOWNLOAD_DIR}/{filename}"
            
            cmd = (f'globus transfer "{CAMELS_ENDPOINT}:{remote_path}" '
                   f'"{LOCAL_ENDPOINT}:{globus_path}" '
                   f'--label "CAMELS {sim_name}" '
                   f'--sync-level {SYNC_LEVEL} '
                   f'--notify off')
            
            print(f"  Submitting {sim_name}...", end=" ", flush=True)
            success, stdout, stderr = run_command(cmd, check=False)
            
            if not success:
                print(f"[FAILED] {stderr}")
                log_message(f"{sim_name}: Transfer submission failed - {stderr}", also_print=False)
                failure_count += 1
                continue
            
            # Extract task ID
            task_id = None
            for line in stdout.split('\n'):
                if 'Task ID:' in line:
                    task_id = line.split('Task ID:')[1].strip()
                    break
            
            if not task_id:
                uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                matches = re.findall(uuid_pattern, stdout)
                if matches:
                    task_id = matches[0]
            
            if not task_id:
                print(f"[FAILED] Could not extract task ID")
                failure_count += 1
                continue
            
            print(f"[OK] task: {task_id[:8]}...")
            download_tasks.append((sim_num, sim_name, task_id, local_hdf5))
        
        if not download_tasks:
            print("  No downloads submitted successfully in this batch. Skipping...")
            continue
        
        # Wait for all downloads to complete
        print(f"\n  Monitoring {len(download_tasks)} concurrent downloads...")
        print(f"  (Checking every {STATUS_CHECK_INTERVAL}s)")
        
        completed_downloads = []
        failed_downloads = []
        
        start_batch_time = time.time()
        while download_tasks and (time.time() - start_batch_time) < TRANSFER_TIMEOUT:
            time.sleep(STATUS_CHECK_INTERVAL)
            
            remaining_tasks = []
            for sim_num, sim_name, task_id, local_hdf5 in download_tasks:
                status_cmd = f'globus task show {task_id}'
                success, stdout, stderr = run_command(status_cmd, check=False)
                
                if not success:
                    print(f"\n  [{sim_name}] Failed to check status")
                    failed_downloads.append((sim_num, sim_name, local_hdf5, "Status check failed"))
                    continue
                
                # Parse status
                status = "UNKNOWN"
                bytes_transferred = 0
                total_bytes = 0
                
                for line in stdout.split('\n'):
                    if 'Status:' in line:
                        status = line.split('Status:')[1].strip()
                    elif 'Bytes Transferred:' in line:
                        try:
                            bytes_transferred = int(line.split(':')[1].strip())
                        except:
                            pass
                    elif 'Total Bytes:' in line:
                        try:
                            total_bytes = int(line.split(':')[1].strip())
                        except:
                            pass
                
                if status == "SUCCEEDED":
                    # Move file from OneDrive Documents to CAMELS_datas
                    onedrive_docs = Path.home() / "OneDrive" / "Documents"
                    downloaded_file = onedrive_docs / local_hdf5.name
                    
                    if not downloaded_file.exists():
                        downloaded_file = Path.home() / "Documents" / local_hdf5.name
                        if not downloaded_file.exists():
                            print(f"\n  [{sim_name}] Transfer succeeded but file not found")
                            failed_downloads.append((sim_num, sim_name, local_hdf5, "File not found after transfer"))
                            continue
                    
                    # Move to proper location
                    local_hdf5.parent.mkdir(exist_ok=True)
                    downloaded_file.rename(local_hdf5)
                    
                    file_size_mb = local_hdf5.stat().st_size / (1024**2)
                    print(f"\n  [{sim_name}] Download complete ({file_size_mb:.1f} MB)")
                    completed_downloads.append((sim_num, sim_name, local_hdf5))
                    
                elif status in ["FAILED", "INACTIVE"]:
                    print(f"\n  [{sim_name}] Download failed with status: {status}")
                    failed_downloads.append((sim_num, sim_name, local_hdf5, f"Transfer failed: {status}"))
                    
                elif status in ["ACTIVE", "PENDING"]:
                    # Still downloading, show progress
                    if total_bytes > 0 and bytes_transferred > 0:
                        progress_pct = (bytes_transferred / total_bytes) * 100
                        mb_transferred = bytes_transferred / (1024**2)
                        mb_total = total_bytes / (1024**2)
                        print(f"\r  [{sim_name}] {progress_pct:.1f}% ({mb_transferred:.1f}/{mb_total:.1f} MB)", end="", flush=True)
                    remaining_tasks.append((sim_num, sim_name, task_id, local_hdf5))
                else:
                    remaining_tasks.append((sim_num, sim_name, task_id, local_hdf5))
            
            download_tasks = remaining_tasks
            
            if not download_tasks:
                print(f"\n  All downloads in batch complete!")
                break
        
        # Handle timeouts
        for sim_num, sim_name, task_id, local_hdf5 in download_tasks:
            print(f"\n  [{sim_name}] Download timed out")
            failed_downloads.append((sim_num, sim_name, local_hdf5, "Timeout"))
        
        # Update failure count
        failure_count += len(failed_downloads)
        for sim_num, sim_name, local_hdf5, error in failed_downloads:
            log_message(f"{sim_name}: Download failed - {error}", also_print=False)
        
        # === PHASE 2: PROCESS ALL DOWNLOADED FILES ===
        print(f"\n[PHASE 2/2] Processing {len(completed_downloads)} downloaded files...")
        
        for sim_num, sim_name, local_hdf5 in completed_downloads:
            try:
                # Double-check if CSV already exists (safety check for mid-batch resume)
                output_csv_path = OUTPUT_DIR / f"Astrid_{sim_name}_groups_090_processed.csv"
                if output_csv_path.exists():
                    print(f"\n  [{sim_name}] CSV already exists, skipping processing")
                    local_hdf5.unlink(missing_ok=True)  # Clean up the HDF5
                    continue
                
                print(f"\n  Processing {sim_name}...", end=" ", flush=True)
                success, msg = process_hdf5_to_csv(local_hdf5, sim_name)
                
                if not success:
                    print(f"[FAILED] {msg}")
                    log_message(f"{sim_name}: Processing failed - {msg}", also_print=False)
                    failure_count += 1
                    success_count -= 1  # Don't count as success if processing fails
                    local_hdf5.unlink(missing_ok=True)
                    continue
                
                print(f"[OK] {msg}")
                
                # Clean up HDF5 file
                print(f"  Cleaning up {local_hdf5.name}...", end=" ", flush=True)
                local_hdf5.unlink()
                print("[OK]")
                
                log_message(f"{sim_name}: Success", also_print=False)
                success_count += 1
                
            except Exception as e:
                print(f"[ERROR] {e}")
                log_message(f"{sim_name}: Error - {e}", also_print=False)
                failure_count += 1
                local_hdf5.unlink(missing_ok=True)
        
        # Progress update
        elapsed_total = int(time.time() - total_start)
        remaining = total_to_process - (success_count + failure_count)
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_num} COMPLETE")
        print(f"Batch: {len(completed_downloads)} succeeded, {len(failed_downloads)} failed")
        print(f"Overall: {success_count} succeeded, {failure_count} failed, {remaining} remaining")
        
        if success_count > 0 and remaining > 0:
            avg_time = elapsed_total / (success_count + failure_count)
            eta = int(avg_time * remaining)
            print(f"ETA: {eta//60}m {eta%60}s")
        
        print('='*70)
        
        # Small pause between batches
        if remaining > 0:
            time.sleep(2)

    # Summary
    total_time = int(time.time() - total_start)
    print(f"\n\n{'='*70}")
    print("PROCESSING COMPLETE")
    print('='*70)
    print(f"Successful: {success_count}/{total_to_process}")
    print(f"Failed: {failure_count}/{total_to_process}")
    print(f"Skipped (already processed): {skipped_count}/{END_SIM - START_SIM + 1}")
    print(f"Total time: {total_time//60}m {total_time%60}s")
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"Average time per file: {int(avg_time)}s")
    print('='*70)

    log_message(f"\nCompleted: {datetime.now()}", also_print=False)
    log_message(f"Successful: {success_count}/{total_to_process}", also_print=False)
    log_message(f"Failed: {failure_count}/{total_to_process}", also_print=False)
    log_message(f"Skipped: {skipped_count}/{END_SIM - START_SIM + 1}", also_print=False)

if __name__ == "__main__":
    main()
