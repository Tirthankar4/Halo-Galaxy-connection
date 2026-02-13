#!/bin/bash
################################################################################
# download_and_process_camels.sh
# Bash equivalent of download_and_process_camels.py
# Downloads CAMELS simulation data via Globus and processes HDF5 files to CSV
################################################################################

################################################################################
# USER CONFIGURATION - SET THESE VALUES FOR YOUR SYSTEM
################################################################################
# 
# LOCAL_ENDPOINT: Your Globus Connect Personal endpoint ID
#   - Find this by running: globus endpoint local-id
#   - Example: "9625c23e-e721-11f0-94d5-0213754b0ca1"
LOCAL_ENDPOINT="9625c23e-e721-11f0-94d5-0213754b0ca1"

# GLOBUS_DOWNLOAD_DIR: The path where Globus downloads files on your endpoint
#   - For Globus Connect Personal, /~ typically maps to your home directory
#   - On Linux/Mac: /~ = /home/<username> or /Users/<username>
GLOBUS_DOWNLOAD_DIR="/~"

# LOCAL_TEMP_DIR: Local directory where downloaded HDF5 files are moved to
#   - Set this to where you want the temporary HDF5 files stored
#   - Example: "./CAMELS_datas" (relative to script directory)
LOCAL_TEMP_DIR="./CAMELS_datas"

# OUTPUT_DIR: Directory where processed CSV files will be saved
#   - Example: "./CAMELS_processed"
OUTPUT_DIR="./CAMELS_processed"

# LOG_FILE: Path to log file
LOG_FILE="./processing_log.txt"

# ACTUAL_DOWNLOAD_PATH: The actual filesystem path where Globus downloads files
#   - Linux/Mac: "$HOME"
# This is where the script looks for files after Globus downloads them
ACTUAL_DOWNLOAD_PATH="$HOME"

# Python Processing Script: Path to Python script for HDF5->CSV conversion
#   - This bash script handles downloads, then calls Python for HDF5 processing
#   - Set this to the location of your Python processing script
PYTHON_SCRIPT="./process_hdf5_to_csv.py"

################################################################################
# END USER CONFIGURATION
################################################################################

# ============================================================================
# CONFIGURATION (typically don't need to change these)
# ============================================================================

CAMELS_ENDPOINT="58bdcd24-6590-11ec-9b60-f9dfb1abb183"  # Main CAMELS endpoint
CAMELS_BASE_PATH="/Sims/Astrid/L25n256/SB7"

START_SIM=633
END_SIM=1023  # Change to 4 for 5 files, 1023 for all

MAX_CONCURRENT_DOWNLOADS=50
TRANSFER_TIMEOUT=1800  # 30 minutes in seconds
STATUS_CHECK_INTERVAL=30  # seconds
SYNC_LEVEL="mtime"  # Use mtime instead of checksum for faster transfers

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_message() {
    local msg="$1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

check_prerequisites() {
    # Check if Globus CLI is installed
    if ! command -v globus &> /dev/null; then
        echo "ERROR: Globus CLI not found. Please install it first."
        return 1
    fi
    
    # Check if logged into Globus
    if ! globus whoami &> /dev/null; then
        echo "ERROR: Not logged into Globus. Run 'globus login' first."
        return 1
    fi
    
    local user=$(globus whoami)
    echo "Logged in as: $user"
    
    # Check if Globus Connect Personal is running
    if ! globus endpoint local-id &> /dev/null; then
        echo "ERROR: Globus Connect Personal may not be running!"
        return 1
    fi
    
    local local_id=$(globus endpoint local-id)
    echo "Local endpoint: $local_id"
    
    return 0
}

get_already_processed_simulations() {
    # Scan OUTPUT_DIR for existing CSV files and return list of processed simulation numbers
    # Parses filenames like: Astrid_SB7_14_groups_090_processed.csv
    
    local processed_sims=()
    
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        echo "${processed_sims[@]}"
        return
    fi
    
    for csv_file in "$OUTPUT_DIR"/Astrid_SB7_*_groups_090_processed.csv; do
        if [[ -f "$csv_file" ]]; then
            # Extract simulation number from filename
            local basename=$(basename "$csv_file")
            if [[ "$basename" =~ Astrid_SB7_([0-9]+)_groups_090_processed\.csv ]]; then
                local sim_num="${BASH_REMATCH[1]}"
                processed_sims+=("$sim_num")
            fi
        fi
    done
    
    echo "${processed_sims[@]}"
}

submit_download() {
    local sim_num="$1"
    local sim_name="SB7_${sim_num}"
    local remote_path="${CAMELS_BASE_PATH}/${sim_name}/groups_090.hdf5"
    local filename="groups_090_${sim_name}.hdf5"
    local globus_path="${GLOBUS_DOWNLOAD_DIR}/${filename}"
    
    # Submit transfer
    local output=$(globus transfer \
        "${CAMELS_ENDPOINT}:${remote_path}" \
        "${LOCAL_ENDPOINT}:${globus_path}" \
        --label "CAMELS ${sim_name}" \
        --sync-level "$SYNC_LEVEL" \
        --notify off 2>&1)
    
    if [[ $? -ne 0 ]]; then
        echo "FAILED"
        return 1
    fi
    
    # Extract task ID
    local task_id=$(echo "$output" | grep -oP 'Task ID: \K[a-f0-9-]+' | head -1)
    
    if [[ -z "$task_id" ]]; then
        # Try alternative extraction
        task_id=$(echo "$output" | grep -oP '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' | head -1)
    fi
    
    if [[ -z "$task_id" ]]; then
        echo "FAILED:NO_TASK_ID"
        return 1
    fi
    
    echo "$task_id"
    return 0
}

check_task_status() {
    local task_id="$1"
    
    local output=$(globus task show "$task_id" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "ERROR"
        return 1
    fi
    
    # Parse status
    local status=$(echo "$output" | grep "Status:" | awk -F': ' '{print $2}' | tr -d ' ')
    local bytes_transferred=$(echo "$output" | grep "Bytes Transferred:" | awk -F': ' '{print $2}' | tr -d ' ')
    local total_bytes=$(echo "$output" | grep "Total Bytes:" | awk -F': ' '{print $2}' | tr -d ' ')
    
    echo "${status}:${bytes_transferred}:${total_bytes}"
    return 0
}

move_downloaded_file() {
    local filename="$1"
    local local_path="$2"
    
    # Check where file was downloaded
    local downloaded_file="${ACTUAL_DOWNLOAD_PATH}/${filename}"
    
    if [[ ! -f "$downloaded_file" ]]; then
        # Fallback: try regular Documents folder
        downloaded_file="${HOME}/Documents/${filename}"
        if [[ ! -f "$downloaded_file" ]]; then
            echo "ERROR: File not found at expected location: $downloaded_file"
            return 1
        fi
    fi
    
    # Ensure destination directory exists
    mkdir -p "$(dirname "$local_path")"
    
    # Move file
    mv "$downloaded_file" "$local_path"
    
    if [[ ! -f "$local_path" ]]; then
        echo "ERROR: Failed to move file to $local_path"
        return 1
    fi
    
    return 0
}

process_hdf5_to_csv() {
    local hdf5_path="$1"
    local sim_name="$2"
    
    # This function would call a Python script to process HDF5 to CSV
    # Since the HDF5 processing logic is complex, we'll use Python for that part
    
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo "ERROR: Python processing script not found: $PYTHON_SCRIPT"
        return 1
    fi
    
    python3 "$PYTHON_SCRIPT" "$hdf5_path" "$sim_name" "$OUTPUT_DIR"
    return $?
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo "======================================================================"
    echo "CAMELS Data Processing Pipeline (Bash Version - BATCH MODE)"
    echo "======================================================================"
    echo "Platform: $(uname -s)"
    echo ""
    echo "OPTIMIZATIONS:"
    echo "  - Status check interval: ${STATUS_CHECK_INTERVAL}s"
    echo "  - Sync level: ${SYNC_LEVEL}"
    echo "  - Concurrent downloads: ${MAX_CONCURRENT_DOWNLOADS}"
    echo ""
    
    # Setup
    mkdir -p "$LOCAL_TEMP_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # Initialize log file
    echo "CAMELS Processing Started: $(date)" > "$LOG_FILE"
    echo "======================================================================" >> "$LOG_FILE"
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    echo ""
    echo "Processing simulations ${START_SIM} to ${END_SIM}..."
    
    # Get already processed simulations
    local already_processed=($(get_already_processed_simulations))
    local already_processed_count=${#already_processed[@]}
    
    if [[ $already_processed_count -gt 0 ]]; then
        echo ""
        echo "[RESUME MODE] Found $already_processed_count already processed simulations"
        if [[ $already_processed_count -le 10 ]]; then
            echo "Already processed: ${already_processed[*]}"
        else
            echo "Already processed (sample): ${already_processed[@]:0:10} ... and $((already_processed_count - 10)) more"
        fi
    fi
    
    echo ""
    echo "Download path: ${GLOBUS_DOWNLOAD_DIR}"
    echo "Working directory: $(pwd)"
    echo "Note: Files download to ${ACTUAL_DOWNLOAD_PATH}, then moved to ${LOCAL_TEMP_DIR}"
    echo ""
    echo "BATCH MODE: Downloading ${MAX_CONCURRENT_DOWNLOADS} files at a time"
    echo ""
    
    # Create list of simulations to process
    local sims_to_process=()
    for ((i=START_SIM; i<=END_SIM; i++)); do
        # Check if already processed
        local skip=0
        for processed_sim in "${already_processed[@]}"; do
            if [[ "$i" -eq "$processed_sim" ]]; then
                skip=1
                break
            fi
        done
        
        if [[ $skip -eq 0 ]]; then
            sims_to_process+=("$i")
        fi
    done
    
    local total_to_process=${#sims_to_process[@]}
    local skipped_count=$((END_SIM - START_SIM + 1 - total_to_process))
    
    if [[ $skipped_count -gt 0 ]]; then
        echo "[SKIPPING] $skipped_count already processed simulations"
        echo ""
    fi
    
    local success_count=0
    local failure_count=0
    local total_start=$(date +%s)
    
    # Process in batches
    local batch_num=0
    for ((batch_start_idx=0; batch_start_idx<total_to_process; batch_start_idx+=MAX_CONCURRENT_DOWNLOADS)); do
        batch_num=$((batch_num + 1))
        
        # Get batch simulations
        local batch_sims=("${sims_to_process[@]:batch_start_idx:MAX_CONCURRENT_DOWNLOADS}")
        local batch_size=${#batch_sims[@]}
        
        echo ""
        echo "======================================================================"
        echo "BATCH ${batch_num}: Processing ${batch_size} simulations"
        echo "Simulations: ${batch_sims[*]}"
        echo "======================================================================"
        
        # === PHASE 1: DOWNLOAD ALL FILES IN BATCH ===
        echo ""
        echo "[PHASE 1/2] Downloading ${batch_size} files concurrently..."
        
        # Arrays to track downloads
        declare -A download_tasks  # task_id -> sim_num
        declare -A download_filenames  # sim_num -> filename
        declare -A download_local_paths  # sim_num -> local_path
        
        # Submit all downloads
        for sim_num in "${batch_sims[@]}"; do
            local sim_name="SB7_${sim_num}"
            local filename="groups_090_${sim_name}.hdf5"
            local local_hdf5="${LOCAL_TEMP_DIR}/${filename}"
            
            echo -n "  Submitting ${sim_name}... "
            
            local task_id=$(submit_download "$sim_num")
            local exit_code=$?
            
            if [[ $exit_code -ne 0 ]] || [[ "$task_id" == "FAILED"* ]]; then
                echo "[FAILED]"
                log_message "${sim_name}: Transfer submission failed" >> /dev/null
                failure_count=$((failure_count + 1))
                continue
            fi
            
            echo "[OK] task: ${task_id:0:8}..."
            download_tasks["$task_id"]="$sim_num"
            download_filenames["$sim_num"]="$filename"
            download_local_paths["$sim_num"]="$local_hdf5"
        done
        
        if [[ ${#download_tasks[@]} -eq 0 ]]; then
            echo "  No downloads submitted successfully in this batch. Skipping..."
            continue
        fi
        
        # Wait for all downloads to complete
        echo ""
        echo "  Monitoring ${#download_tasks[@]} concurrent downloads..."
        echo "  (Checking every ${STATUS_CHECK_INTERVAL}s)"
        
        local completed_downloads=()
        local failed_downloads=()
        
        local start_batch_time=$(date +%s)
        
        while [[ ${#download_tasks[@]} -gt 0 ]]; do
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_batch_time))
            
            if [[ $elapsed -gt $TRANSFER_TIMEOUT ]]; then
                echo ""
                echo "  Batch timeout reached!"
                for task_id in "${!download_tasks[@]}"; do
                    local sim_num="${download_tasks[$task_id]}"
                    local sim_name="SB7_${sim_num}"
                    echo "  [${sim_name}] Download timed out"
                    failed_downloads+=("$sim_num")
                    unset download_tasks["$task_id"]
                done
                break
            fi
            
            sleep "$STATUS_CHECK_INTERVAL"
            
            # Check all tasks
            for task_id in "${!download_tasks[@]}"; do
                local sim_num="${download_tasks[$task_id]}"
                local sim_name="SB7_${sim_num}"
                local filename="${download_filenames[$sim_num]}"
                local local_hdf5="${download_local_paths[$sim_num]}"
                
                local status_info=$(check_task_status "$task_id")
                
                if [[ "$status_info" == "ERROR" ]]; then
                    echo ""
                    echo "  [${sim_name}] Failed to check status"
                    failed_downloads+=("$sim_num")
                    unset download_tasks["$task_id"]
                    continue
                fi
                
                # Parse status
                IFS=':' read -r status bytes_transferred total_bytes <<< "$status_info"
                
                if [[ "$status" == "SUCCEEDED" ]]; then
                    # Move file from download location to temp directory
                    if move_downloaded_file "$filename" "$local_hdf5"; then
                        local file_size_mb=$(du -m "$local_hdf5" | awk '{print $1}')
                        echo ""
                        echo "  [${sim_name}] Download complete (${file_size_mb} MB)"
                        completed_downloads+=("$sim_num")
                    else
                        echo ""
                        echo "  [${sim_name}] Transfer succeeded but file move failed"
                        failed_downloads+=("$sim_num")
                    fi
                    unset download_tasks["$task_id"]
                    
                elif [[ "$status" == "FAILED" ]] || [[ "$status" == "INACTIVE" ]]; then
                    echo ""
                    echo "  [${sim_name}] Download failed with status: ${status}"
                    failed_downloads+=("$sim_num")
                    unset download_tasks["$task_id"]
                    
                elif [[ "$status" == "ACTIVE" ]] || [[ "$status" == "PENDING" ]]; then
                    # Still downloading, show progress
                    if [[ -n "$total_bytes" ]] && [[ "$total_bytes" -gt 0 ]] && [[ -n "$bytes_transferred" ]]; then
                        local progress_pct=$((bytes_transferred * 100 / total_bytes))
                        local mb_transferred=$((bytes_transferred / 1048576))
                        local mb_total=$((total_bytes / 1048576))
                        echo -ne "\r  [${sim_name}] ${progress_pct}% (${mb_transferred}/${mb_total} MB)"
                    fi
                fi
            done
            
            if [[ ${#download_tasks[@]} -eq 0 ]]; then
                echo ""
                echo "  All downloads in batch complete!"
                break
            fi
        done
        
        # Update failure count
        failure_count=$((failure_count + ${#failed_downloads[@]}))
        
        # === PHASE 2: PROCESS ALL DOWNLOADED FILES ===
        echo ""
        echo "[PHASE 2/2] Processing ${#completed_downloads[@]} downloaded files..."
        
        for sim_num in "${completed_downloads[@]}"; do
            local sim_name="SB7_${sim_num}"
            local local_hdf5="${download_local_paths[$sim_num]}"
            local output_csv="${OUTPUT_DIR}/Astrid_${sim_name}_groups_090_processed.csv"
            
            # Safety check: skip if CSV already exists
            if [[ -f "$output_csv" ]]; then
                echo ""
                echo "  [${sim_name}] CSV already exists, skipping processing"
                rm -f "$local_hdf5"
                continue
            fi
            
            echo ""
            echo -n "  Processing ${sim_name}... "
            
            if process_hdf5_to_csv "$local_hdf5" "$sim_name"; then
                echo "[OK]"
                echo -n "  Cleaning up ${local_hdf5##*/}... "
                rm -f "$local_hdf5"
                echo "[OK]"
                log_message "${sim_name}: Success" >> /dev/null
                success_count=$((success_count + 1))
            else
                echo "[FAILED]"
                log_message "${sim_name}: Processing failed" >> /dev/null
                rm -f "$local_hdf5"
                failure_count=$((failure_count + 1))
            fi
        done
        
        # Progress update
        local elapsed_total=$(( $(date +%s) - total_start ))
        local remaining=$((total_to_process - success_count - failure_count))
        
        echo ""
        echo "======================================================================"
        echo "BATCH ${batch_num} COMPLETE"
        echo "Batch: ${#completed_downloads[@]} succeeded, ${#failed_downloads[@]} failed"
        echo "Overall: ${success_count} succeeded, ${failure_count} failed, ${remaining} remaining"
        
        if [[ $success_count -gt 0 ]] && [[ $remaining -gt 0 ]]; then
            local avg_time=$((elapsed_total / (success_count + failure_count)))
            local eta=$((avg_time * remaining))
            echo "ETA: $((eta / 60))m $((eta % 60))s"
        fi
        
        echo "======================================================================"
        
        # Small pause between batches
        if [[ $remaining -gt 0 ]]; then
            sleep 2
        fi
    done
    
    # Summary
    local total_time=$(( $(date +%s) - total_start ))
    echo ""
    echo ""
    echo "======================================================================"
    echo "PROCESSING COMPLETE"
    echo "======================================================================"
    echo "Successful: ${success_count}/${total_to_process}"
    echo "Failed: ${failure_count}/${total_to_process}"
    echo "Skipped (already processed): ${skipped_count}/$((END_SIM - START_SIM + 1))"
    echo "Total time: $((total_time / 60))m $((total_time % 60))s"
    if [[ $success_count -gt 0 ]]; then
        local avg_time=$((total_time / success_count))
        echo "Average time per file: ${avg_time}s"
    fi
    echo "======================================================================"
    
    log_message ""
    log_message "Completed: $(date)"
    log_message "Successful: ${success_count}/${total_to_process}"
    log_message "Failed: ${failure_count}/${total_to_process}"
    log_message "Skipped: ${skipped_count}/$((END_SIM - START_SIM + 1))"
}

# Run main function
main
