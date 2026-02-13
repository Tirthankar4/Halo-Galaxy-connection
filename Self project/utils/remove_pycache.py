#!/usr/bin/env python3
"""
Script to remove all __pycache__ folders from the codebase.
"""

import os
import shutil
from pathlib import Path


def remove_pycache_folders(root_dir="."):
    """
    Recursively find and remove all __pycache__ folders in the codebase.
    
    Args:
        root_dir: Root directory to search from (default: current directory)
    """
    root_path = Path(root_dir).resolve()
    removed_count = 0
    removed_paths = []
    
    print(f"Searching for __pycache__ folders in: {root_path}")
    print("-" * 60)
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check if __pycache__ is in the current directory
        pycache_path = Path(dirpath) / "__pycache__"
        
        if pycache_path.exists() and pycache_path.is_dir():
            try:
                # Remove the entire __pycache__ directory
                shutil.rmtree(pycache_path)
                removed_count += 1
                removed_paths.append(pycache_path)
                print(f"✓ Removed: {pycache_path.relative_to(root_path)}")
            except Exception as e:
                print(f"✗ Error removing {pycache_path.relative_to(root_path)}: {e}")
    
    print("-" * 60)
    print(f"\nTotal __pycache__ folders removed: {removed_count}")
    
    if removed_count == 0:
        print("No __pycache__ folders found.")
    else:
        print("\nCleanup complete!")


if __name__ == "__main__":
    remove_pycache_folders()

