import pandas as pd
import argparse
from pathlib import Path
from src.config import PROCESSED_DATA_DIR

def load_and_tag(file_path: Path, sim_id: str = None):
    """
    Load a parquet file and tag it with Simulation and SimID.
    
    Args:
        file_path: Path to the parquet file (relative to PROCESSED_DATA_DIR)
        sim_id: Optional simulation ID. If not provided, extracted from file path.
    """
    full_path = PROCESSED_DATA_DIR / file_path
    df = pd.read_parquet(full_path)
    
    # Extract simulation ID from path if not provided
    if sim_id is None:
        # Try to extract from path (e.g., 'lh135/halo_galaxy.parquet' -> 'lh135')
        path_parts = Path(file_path).parts
        if len(path_parts) > 1:
            sim_id = path_parts[0].lower()
        else:
            # If just filename, use filename without extension
            sim_id = Path(file_path).stem.lower()
    
    df['Simulation'] = sim_id.upper()
    df['SimID'] = sim_id.lower()
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple parquet datasets from data/processed folder'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Parquet file paths relative to data/processed (e.g., lh135/halo_galaxy.parquet)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='master_training_set.parquet',
        help='Output filename (default: master_training_set.parquet)'
    )
    
    args = parser.parse_args()
    
    # Load all datasets
    dataframes = []
    for file_path in args.files:
        print(f"Loading: {file_path}")
        df = load_and_tag(file_path)
        dataframes.append(df)
        print(f"  Loaded {len(df)} rows")
    
    # Combine them
    master_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined file
    output_path = PROCESSED_DATA_DIR / args.output
    master_df.to_parquet(output_path, index=False)
    
    print(f"\nCreated master training set with {len(master_df)} total galaxies.")
    print(f"Saved to: {output_path}")
    print("Columns:", master_df.columns.tolist())

if __name__ == '__main__':
    main()

