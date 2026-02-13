#!/usr/bin/env python3
"""
download_camels_batch.py
Downloads CAMELS simulation files via Globus CLI
"""

import subprocess
import sys
import platform
from pathlib import Path
import time
from datetime import datetime
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CAMELS_ENDPOINT = "58bdcd24-6590-11ec-9b60-f9dfb1abb183"
LOCAL_ENDPOINT = "9625c23e-e721-11f0-94d5-0213754b0ca1"
CAMELS_BASE_PATH = "/Sims/Astrid/L25n256/SB7"

# Local storage
DOWNLOAD_DIR = Path(r"C:\Users\tirth\Documents\Projects\Halo - galaxy connection\downloads")
LOG_FILE = Path("./download_log.txt")

# Globus path (maps to OneDrive\Documents or local Documents depending on GCP config)
GLOBUS_DOWNLOAD_PATH = "/~/"

# Simulation range
START_SIM = 0
END_SIM = 4  # Process SB7_0 through SB7_4 (5 files)
# Change to 1023 for all 1024 simulations

TRANSFER_TIMEOUT = 1800  # 30 minutes per file

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_command(cmd, check=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def log_message(message, also_print=True):
    """Write to log file and optionally print"""
    if also_print:
        print(message)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")

def check_prerequisites():
    """Check Globus CLI and login status"""
    success, stdout, _ = run_command("globus whoami", check=False)
    if not success:
        print("ERROR: Not logged into Globus. Run: globus login")
        return False
    print(f"Logged in as: {stdout.strip()}")

    success, stdout, _ = run_command("globus endpoint local-id", check=False)
    if not success:
        print("ERROR: Globus Connect Personal not running!")
        return False
    print(f"GCP endpoint: {stdout.strip()}")
    return True

def find_downloaded_file(filename):
    """Find downloaded file in possible locations"""
    # Check multiple possible download locations
    possible_paths = [
        Path(r"C:\Users\tirth\OneDrive\Documents") / filename,
        Path(r"C:\Users\tirth\Documents") / filename,
        Path.cwd() / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return path
    return None

def download_file(sim_name, remote_path, local_filename):
    """Download file using Globus CLI"""

    globus_dest = f"{GLOBUS_DOWNLOAD_PATH}{local_filename}"

    # Submit transfer
    cmd = f'globus transfer "{CAMELS_ENDPOINT}:{remote_path}" "{LOCAL_ENDPOINT}:{globus_dest}" --label "CAMELS {sim_name}" --sync-level checksum'

    success, stdout, stderr = run_command(cmd, check=False)
    if not success:
        return False, f"Transfer submission failed: {stderr}", None

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
        return False, "Could not extract task ID", None

    # Monitor transfer
    print(f" [task: {task_id[:8]}...]", end="", flush=True)

    start_time = time.time()
    last_bytes = 0
    stall_count = 0

    while time.time() - start_time < TRANSFER_TIMEOUT:
        success, stdout, _ = run_command(f'globus task show {task_id}', check=False)

        if not success:
            return False, "Failed to check task status", None

        # Parse status
        status = "UNKNOWN"
        bytes_transferred = 0

        for line in stdout.split('\n'):
            if 'Status:' in line:
                status = line.split('Status:')[1].strip()
            elif 'Bytes Transferred:' in line:
                try:
                    bytes_transferred = int(line.split(':')[1].strip())
                except:
                    pass

        # Show progress
        elapsed = int(time.time() - start_time)
        if bytes_transferred > 0:
            mb = bytes_transferred / (1024**2)
            print(f"\r  Downloading {sim_name}... {mb:.1f} MB [{elapsed}s]", 
                  end="", flush=True)
        else:
            print(".", end="", flush=True)

        # Check for stall
        if bytes_transferred > 0 and bytes_transferred == last_bytes:
            stall_count += 1
            if stall_count > 12:
                return False, f"Transfer stalled at {bytes_transferred} bytes", None
        else:
            stall_count = 0
            last_bytes = bytes_transferred

        # Check completion
        if status == "SUCCEEDED":
            # Find the downloaded file
            downloaded_file = find_downloaded_file(local_filename)

            if not downloaded_file:
                return False, "Transfer succeeded but file not found", None

            file_size_mb = downloaded_file.stat().st_size / (1024**2)
            return True, f"Downloaded {file_size_mb:.1f} MB in {elapsed}s", downloaded_file

        elif status in ["FAILED", "INACTIVE"]:
            return False, f"Transfer failed: {status}", None

        time.sleep(10)

    return False, f"Timeout after {TRANSFER_TIMEOUT/60:.0f} min", None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("CAMELS Batch Downloader")
    print("="*70)
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version.split()[0]}")

    # Setup
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    with open(LOG_FILE, 'w') as f:
        f.write(f"CAMELS Download Started: {datetime.now()}\n")
        f.write("="*70 + "\n")

    if not check_prerequisites():
        sys.exit(1)

    print(f"\nDownloading simulations {START_SIM} to {END_SIM}")
    print(f"Globus download path: {GLOBUS_DOWNLOAD_PATH}")
    print(f"Local storage: {DOWNLOAD_DIR}")
    print(f"\nNote: Each file ~400 MB, 5-15 min per download")
    print(f"Estimated total time: {(END_SIM - START_SIM + 1) * 10}-{(END_SIM - START_SIM + 1) * 20} minutes\n")

    success_count = 0
    failure_count = 0
    total_start = time.time()

    for i in range(START_SIM, END_SIM + 1):
        sim_name = f"SB7_{i}"
        print(f"\n{'='*70}")
        print(f"[{i - START_SIM + 1}/{END_SIM - START_SIM + 1}] {sim_name}")
        print('='*70)

        remote_path = f"{CAMELS_BASE_PATH}/{sim_name}/groups_090.hdf5"
        local_filename = f"{sim_name}_groups_090.hdf5"
        final_path = DOWNLOAD_DIR / local_filename

        # Skip if already downloaded
        if final_path.exists():
            file_size_mb = final_path.stat().st_size / (1024**2)
            print(f"  ✓ Already exists ({file_size_mb:.1f} MB) - Skipping")
            log_message(f"{sim_name}: Already downloaded", also_print=False)
            success_count += 1
            continue

        try:
            # Download
            print(f"  Downloading {sim_name}...", end="", flush=True)
            success, msg, downloaded_file = download_file(sim_name, remote_path, local_filename)

            if not success:
                print(f"\n  ✗ {msg}")
                log_message(f"{sim_name}: Download failed - {msg}", also_print=False)
                failure_count += 1
                continue

            print(f"\n  ✓ {msg}")

            # Move to final location
            print(f"  Moving to storage...", end=" ", flush=True)
            downloaded_file.rename(final_path)
            print("✓")

            log_message(f"{sim_name}: Success - {final_path}", also_print=False)
            success_count += 1

            # Show progress
            elapsed = int(time.time() - total_start)
            remaining = END_SIM - i
            if success_count > 0:
                avg_time = elapsed / (i - START_SIM + 1)
                eta = int(avg_time * remaining)
                print(f"\n  Progress: {success_count} done, {failure_count} failed, "
                      f"{remaining} remaining, ETA: {eta//60}m")

            time.sleep(2)

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Resume by setting START_SIM={i}")
            break
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            log_message(f"{sim_name}: Error - {e}", also_print=False)
            failure_count += 1

    # Summary
    total_time = int(time.time() - total_start)
    print(f"\n\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print('='*70)
    print(f"Successful: {success_count}/{END_SIM - START_SIM + 1}")
    print(f"Failed: {failure_count}/{END_SIM - START_SIM + 1}")
    print(f"Total time: {total_time//60}m {total_time%60}s")
    print(f"Files saved to: {DOWNLOAD_DIR}")
    print('='*70)

    if success_count > 0:
        print(f"\n📁 Downloaded files are in: {DOWNLOAD_DIR}")
        print(f"\n💡 Next steps:")
        print(f"   1. Create your preprocessing script: process_simulation.py")
        print(f"   2. Process each file:")
        print(f"      for file in downloads/*.hdf5:")
        print(f"          python process_simulation.py file output.csv")

    log_message(f"\nCompleted: {datetime.now()}", also_print=False)
    log_message(f"Success: {success_count}, Failed: {failure_count}", also_print=False)

if __name__ == "__main__":
    main()
