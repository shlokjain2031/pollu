#!/usr/bin/env python3
"""Append MODIS AOD features to training split files.

This script adds MODIS AOD features (aod_047, aod_055, aod_uncertainty, column_wv)
to existing {date}_training.parquet files in artifacts/train, artifacts/val, artifacts/test.

Key features:
- Deterministic O(1) mapping from row_idx → pixel_id using spatial formula
- Uses nearest MODIS date if exact date missing (±1 day tolerance)
- Atomic file updates (temp → verify → rename)
- Only processes _training.parquet files (skips _target.parquet)
- Parallel processing with configurable workers

Usage:
    python scripts/append_modis_train.py --splits train val test --workers 8
    python scripts/append_modis_train.py --splits train --start-date 2019-01-01 --end-date 2019-12-31
    python scripts/append_modis_train.py --splits test --dry-run  # Preview without writing
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from patliputra.modis_pixel_utils import get_modis_pixel_id_vectorized


# Default paths
DEFAULT_GRID_PATH = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"
DEFAULT_MODIS_DIR = PROJECT_ROOT / "artifacts" / "v1" / "modis_aod" / "mumbai"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "artifacts"

# MODIS feature columns to append
MODIS_FEATURES = ['aod_047', 'aod_055', 'aod_uncertainty', 'column_wv']


def load_grid_with_pixel_id(grid_path: Path) -> pd.DataFrame:
    """Load grid and add pixel_id column using O(1) formula.
    
    Args:
        grid_path: Path to grid parquet file
        
    Returns:
        DataFrame with columns: row_idx, centroid_lon, centroid_lat, pixel_id
    """
    grid = pd.read_parquet(grid_path)
    
    # Compute pixel_id for all grid cells (O(1) vectorized)
    pixel_ids = get_modis_pixel_id_vectorized(
        grid['centroid_lon'].values,
        grid['centroid_lat'].values
    )
    grid['pixel_id'] = pixel_ids
    
    # Keep only necessary columns for joining
    return grid[['row_idx', 'pixel_id']]


def find_nearest_modis_date(
    target_date: str,
    modis_dir: Path,
    max_days: int = 1
) -> Optional[str]:
    """Find nearest MODIS date within ±max_days of target.
    
    Args:
        target_date: Target date (YYYY-MM-DD)
        modis_dir: MODIS data directory
        max_days: Maximum days offset to search (default: 1)
        
    Returns:
        Nearest MODIS date or None if not found
    """
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Try exact match first
    exact_file = modis_dir / f"{target_date}.parquet"
    if exact_file.exists():
        return target_date
    
    # Search nearby dates
    for offset in range(1, max_days + 1):
        # Try future date
        future_dt = target_dt + timedelta(days=offset)
        future_date = future_dt.strftime('%Y-%m-%d')
        if (modis_dir / f"{future_date}.parquet").exists():
            return future_date
        
        # Try past date
        past_dt = target_dt - timedelta(days=offset)
        past_date = past_dt.strftime('%Y-%m-%d')
        if (modis_dir / f"{past_date}.parquet").exists():
            return past_date
    
    return None


def process_single_file(
    file_path: Path,
    grid_with_pixels: pd.DataFrame,
    modis_dir: Path,
    dry_run: bool = False,
) -> Tuple[str, bool, str]:
    """Process a single training file: append MODIS features.
    
    Args:
        file_path: Path to training file
        grid_with_pixels: Grid DataFrame with pixel_id column
        modis_dir: MODIS data directory
        dry_run: If True, don't write files (just validate)
        
    Returns:
        Tuple of (filename, success: bool, message: str)
    """
    filename = file_path.name
    
    try:
        # Parse date from filename (YYYY-MM-DD_training.parquet)
        date_str = filename.replace('_training.parquet', '')
        
        # Validate date format
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return (filename, False, "Invalid date format in filename")
        
        # Load training file
        df_train = pd.read_parquet(file_path)
        
        # Check if features already exist
        if 'aod_055' in df_train.columns:
            return (filename, False, "MODIS features already present")
        
        # Find nearest MODIS date
        modis_date = find_nearest_modis_date(date_str, modis_dir, max_days=1)
        if modis_date is None:
            return (filename, False, f"No MODIS data within ±1 day of {date_str}")
        
        # Load MODIS data
        modis_file = modis_dir / f"{modis_date}.parquet"
        df_modis = pd.read_parquet(modis_file)
        
        # Select only feature columns (exclude pixel_id from final output)
        modis_cols = ['pixel_id'] + [c for c in MODIS_FEATURES if c in df_modis.columns]
        df_modis = df_modis[modis_cols]
        
        # Map training row_idx to pixel_id via grid
        df_merged = df_train.merge(
            grid_with_pixels[['row_idx', 'pixel_id']],
            on='row_idx',
            how='left'
        )
        
        # Join MODIS features on pixel_id
        df_merged = df_merged.merge(
            df_modis,
            on='pixel_id',
            how='left'
        )
        
        # Drop pixel_id (we only wanted it for joining)
        df_merged = df_merged.drop(columns=['pixel_id'])
        
        # Verify integrity
        if len(df_merged) != len(df_train):
            return (filename, False, f"Row count mismatch: {len(df_train)} → {len(df_merged)}")
        
        # Note: row_idx can have duplicates (same cell, different hours)
        # So we don't check for duplicates - just verify row count unchanged
        
        # Check that some AOD values exist
        aod_valid = df_merged['aod_055'].notna().sum() if 'aod_055' in df_merged.columns else 0
        aod_pct = (aod_valid / len(df_merged)) * 100 if len(df_merged) > 0 else 0
        
        if dry_run:
            return (filename, True, f"DRY RUN: Would add {len(MODIS_FEATURES)} cols, {aod_pct:.1f}% valid AOD")
        
        # Write to temp file
        temp_file = file_path.parent / f".temp_{filename}"
        df_merged.to_parquet(temp_file, index=False)
        
        # Verify temp file
        df_verify = pd.read_parquet(temp_file)
        if len(df_verify) != len(df_train):
            temp_file.unlink()
            return (filename, False, "Temp file verification failed")
        
        # Atomic rename
        temp_file.replace(file_path)
        
        offset_msg = f" (used {modis_date})" if modis_date != date_str else ""
        return (filename, True, f"Added {len(MODIS_FEATURES)} cols, {aod_pct:.1f}% valid AOD{offset_msg}")
    
    except Exception as e:
        # Clean up temp file if exists
        temp_file = file_path.parent / f".temp_{filename}"
        if temp_file.exists():
            temp_file.unlink()
        
        return (filename, False, f"Error: {str(e)[:80]}")


def process_split_directory(
    split_name: str,
    splits_dir: Path,
    grid_with_pixels: pd.DataFrame,
    modis_dir: Path,
    max_workers: int = 8,
    dry_run: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, int]:
    """Process all training files in a split directory.
    
    Args:
        split_name: Split name ('train', 'val', 'test')
        splits_dir: Base directory containing splits
        grid_with_pixels: Grid with pixel_id column
        modis_dir: MODIS data directory
        max_workers: Number of parallel workers
        dry_run: If True, don't write files
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        Dict with counts: {'processed': int, 'success': int, 'skipped': int, 'failed': int}
    """
    split_dir = splits_dir / split_name
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist, skipping")
        return {'processed': 0, 'success': 0, 'skipped': 0, 'failed': 0}
    
    # Find all training files
    all_files = sorted(split_dir.glob("*_training.parquet"))
    
    # Filter by date range if specified
    if start_date or end_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        
        filtered_files = []
        for f in all_files:
            date_str = f.name.replace('_training.parquet', '')
            try:
                file_dt = datetime.strptime(date_str, '%Y-%m-%d')
                if start_dt and file_dt < start_dt:
                    continue
                if end_dt and file_dt > end_dt:
                    continue
                filtered_files.append(f)
            except ValueError:
                continue
        
        all_files = filtered_files
    
    if not all_files:
        print(f"No training files found in {split_dir}")
        return {'processed': 0, 'success': 0, 'skipped': 0, 'failed': 0}
    
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*80}")
    print(f"Files to process: {len(all_files)}")
    print(f"Workers: {max_workers}")
    print(f"Mode: {'DRY RUN' if dry_run else 'WRITE'}")
    print()
    
    # Process in parallel
    counts = {'processed': 0, 'success': 0, 'skipped': 0, 'failed': 0}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_file, f, grid_with_pixels, modis_dir, dry_run): f
            for f in all_files
        }
        
        # Collect results
        for future in as_completed(futures):
            filename, success, message = future.result()
            counts['processed'] += 1
            
            if success:
                counts['success'] += 1
                status = "✓"
            elif "already present" in message:
                counts['skipped'] += 1
                status = "○"
            else:
                counts['failed'] += 1
                status = "✗"
            
            # Progress update every 50 files
            if counts['processed'] % 50 == 0 or counts['processed'] == len(all_files):
                elapsed = time.time() - start_time
                rate = counts['processed'] / elapsed if elapsed > 0 else 0
                remaining = len(all_files) - counts['processed']
                eta = remaining / rate if rate > 0 else 0
                
                print(f"[{counts['processed']}/{len(all_files)}] "
                      f"Success: {counts['success']}, Skipped: {counts['skipped']}, "
                      f"Failed: {counts['failed']} | "
                      f"Rate: {rate:.1f} files/sec | ETA: {eta:.0f}s")
            
            # Show first few results and any failures
            if counts['processed'] <= 10 or not success:
                print(f"  {status} {filename}: {message}")
    
    elapsed_total = time.time() - start_time
    print(f"\n{split_name.upper()} complete in {elapsed_total/60:.1f} minutes")
    print(f"  Success: {counts['success']}")
    print(f"  Skipped: {counts['skipped']}")
    print(f"  Failed: {counts['failed']}")
    
    return counts


def main():
    parser = argparse.ArgumentParser(
        description='Append MODIS AOD features to training split files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'val', 'test'],
        default=['train', 'val', 'test'],
        help='Splits to process (default: all)'
    )
    
    parser.add_argument(
        '--grid-path',
        type=Path,
        default=DEFAULT_GRID_PATH,
        help='Path to grid parquet file'
    )
    
    parser.add_argument(
        '--modis-dir',
        type=Path,
        default=DEFAULT_MODIS_DIR,
        help='MODIS AOD data directory'
    )
    
    parser.add_argument(
        '--splits-dir',
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help='Base directory containing train/val/test splits'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing files'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date filter (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date filter (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.grid_path.exists():
        print(f"Error: Grid file not found: {args.grid_path}")
        sys.exit(1)
    
    if not args.modis_dir.exists():
        print(f"Error: MODIS directory not found: {args.modis_dir}")
        sys.exit(1)
    
    if not args.splits_dir.exists():
        print(f"Error: Splits directory not found: {args.splits_dir}")
        sys.exit(1)
    
    # Print header
    print("\n" + "="*80)
    print("APPEND MODIS AOD FEATURES TO TRAINING SPLITS")
    print("="*80)
    print(f"Grid: {args.grid_path}")
    print(f"MODIS: {args.modis_dir}")
    print(f"Splits: {args.splits}")
    print(f"Workers: {args.workers}")
    if args.start_date or args.end_date:
        print(f"Date range: {args.start_date or 'start'} to {args.end_date or 'end'}")
    print()
    
    # Load grid with pixel_id (shared across all workers)
    print("Loading grid and computing pixel_id mappings...")
    start = time.time()
    grid_with_pixels = load_grid_with_pixel_id(args.grid_path)
    elapsed = time.time() - start
    print(f"  Grid loaded: {len(grid_with_pixels)} cells")
    print(f"  Unique pixels: {grid_with_pixels['pixel_id'].nunique()}")
    print(f"  Time: {elapsed:.2f}s")
    
    # Process each split
    total_counts = {'processed': 0, 'success': 0, 'skipped': 0, 'failed': 0}
    
    for split in args.splits:
        counts = process_split_directory(
            split_name=split,
            splits_dir=args.splits_dir,
            grid_with_pixels=grid_with_pixels,
            modis_dir=args.modis_dir,
            max_workers=args.workers,
            dry_run=args.dry_run,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        
        for key in total_counts:
            total_counts[key] += counts[key]
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total files processed: {total_counts['processed']}")
    print(f"  ✓ Success: {total_counts['success']}")
    print(f"  ○ Skipped: {total_counts['skipped']}")
    print(f"  ✗ Failed: {total_counts['failed']}")
    print()
    
    if args.dry_run:
        print("DRY RUN COMPLETE - No files were modified")
    else:
        print("MODIS features successfully appended to training files!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
