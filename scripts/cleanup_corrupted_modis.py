#!/usr/bin/env python
"""Clean up corrupted MODIS files by removing _x/_y duplicate columns.

This script handles files that have duplicate AOD columns with _x and _y suffixes
from earlier failed append attempts. It removes these duplicates, keeping only the
original columns, so they can be re-processed.

Usage:
    python scripts/cleanup_corrupted_modis.py --splits train val test --workers 8
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict
import pandas as pd
import time


def clean_corrupted_file(file_path: Path) -> Tuple[str, bool, str]:
    """Remove _x/_y suffix columns from a file.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        Tuple of (filename, success, message)
    """
    filename = file_path.name
    
    try:
        # Load file
        df = pd.read_parquet(file_path)
        
        # Check for _x/_y suffixes
        x_y_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
        if not x_y_cols:
            return (filename, False, "No _x/_y suffixes found")
        
        # Remove corrupted columns, keep original
        clean_cols = [c for c in df.columns if not (c.endswith('_x') or c.endswith('_y'))]
        df_clean = df[clean_cols]
        
        # Verify we didn't lose data
        if len(df_clean) != len(df):
            return (filename, False, "Row count changed during cleanup")
        
        # Write to temp file
        temp_file = file_path.parent / f".temp_{filename}"
        df_clean.to_parquet(temp_file, index=False)
        
        # Verify temp file
        df_verify = pd.read_parquet(temp_file)
        if len(df_verify) != len(df_clean):
            temp_file.unlink()
            return (filename, False, "Temp file verification failed")
        
        # Atomic rename
        temp_file.replace(file_path)
        
        return (filename, True, f"Removed {len(x_y_cols)} corrupted columns")
        
    except Exception as e:
        return (filename, False, f"Error: {str(e)[:60]}")


def clean_split_directory(
    split_name: str,
    splits_dir: Path,
    max_workers: int = 8,
) -> Dict[str, int]:
    """Clean all files in a split directory.
    
    Args:
        split_name: Split name ('train', 'val', 'test')
        splits_dir: Base directory containing splits
        max_workers: Number of parallel workers
        
    Returns:
        Dict with counts: {'processed': int, 'success': int, 'failed': int}
    """
    split_dir = splits_dir / split_name
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist, skipping")
        return {'processed': 0, 'success': 0, 'failed': 0}
    
    # Find all training files
    all_files = sorted(split_dir.glob("*_training.parquet"))
    
    if not all_files:
        print(f"No training files found in {split_dir}")
        return {'processed': 0, 'success': 0, 'failed': 0}
    
    print(f"\n{'='*80}")
    print(f"Cleaning {split_name.upper()} split")
    print(f"{'='*80}")
    print(f"Files to check: {len(all_files)}")
    print(f"Workers: {max_workers}")
    print()
    
    # Process in parallel
    counts = {'processed': 0, 'success': 0, 'failed': 0}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(clean_corrupted_file, f): f for f in all_files}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            filename, success, message = future.result()
            
            counts['processed'] += 1
            if success:
                counts['success'] += 1
                print(f"  ✓ {filename}: {message}")
            else:
                counts['failed'] += 1
                if "No _x/_y" not in message:  # Only print errors, not "not corrupted"
                    print(f"  ✗ {filename}: {message}")
            
            # Progress
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (len(all_files) - completed) / rate
                print(f"[{completed}/{len(all_files)}] Success: {counts['success']}, Failed: {counts['failed']} | Rate: {rate:.1f} files/sec | ETA: {remaining:.0f}s")
    
    elapsed = time.time() - start_time
    print(f"\n{split_name.upper()} complete in {elapsed/60:.1f} minutes")
    print(f"  ✓ Success: {counts['success']}")
    print(f"  ✗ Failed: {counts['failed']}")
    
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Clean corrupted MODIS files by removing _x/_y suffix columns"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base directory containing train/val/test splits",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("CLEANUP CORRUPTED MODIS FILES")
    print(f"{'='*80}")
    print(f"Splits: {args.splits}")
    print(f"Workers: {args.workers}")
    print()
    
    total_counts = {'processed': 0, 'success': 0, 'failed': 0}
    
    for split in args.splits:
        counts = clean_split_directory(
            split,
            args.splits_dir,
            args.workers,
        )
        total_counts['processed'] += counts['processed']
        total_counts['success'] += counts['success']
        total_counts['failed'] += counts['failed']
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {total_counts['processed']}")
    print(f"  ✓ Success: {total_counts['success']}")
    print(f"  ✗ Failed: {total_counts['failed']}")
    print()
    
    if total_counts['success'] > 0:
        print("✓ Cleanup complete! Re-run append_modis_train.py to add flags:")
        print(f"  python scripts/append_modis_train.py --splits {' '.join(args.splits)} --workers {args.workers}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
