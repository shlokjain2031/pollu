#!/usr/bin/env python3
"""Add Sentinel2 features to existing training.parquet files.

Appends computed Sentinel2 features (evi, ui_light, si, cloud_mask) to
existing training parquet files in artifacts/v1/training/{date}/

Usage:
    python scripts/add_sentinel2_features.py --start-date 2023-01-01 --end-date 2023-01-31 --workers 8
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nalanda.features_satellite import (
    compute_sentinel_indices,
    load_sentinel,
    find_most_recent_date,
    get_available_dates,
)

DEFAULT_SENTINEL_DIR = PROJECT_ROOT / "artifacts" / "v1" / "sentinel2"


def process_single_date(
    date_str: str,
    training_dir: Path,
    sentinel_dir: Path,
) -> Tuple[str, bool, str]:
    """Process a single date: compute and append Sentinel2 features.
    
    Args:
        date_str: Date string (YYYY-MM-DD).
        training_dir: Path to training data directory.
        sentinel_dir: Path to Sentinel2 data directory.
        
    Returns:
        Tuple of (date_str, success: bool, message: str)
    """
    date_dir = training_dir / date_str
    training_file = date_dir / 'training.parquet'
    
    if not training_file.exists():
        return (date_str, False, "File not found")
    
    try:
        # Check if features already exist
        df_existing = pd.read_parquet(training_file)
        if 'evi' in df_existing.columns:
            return (date_str, False, "Features already present")
        
        # Compute Sentinel2 features
        available = get_available_dates(sentinel_dir, 'mumbai')
        if not available:
            return (date_str, False, "No Sentinel2 data available")
        
        source_date = find_most_recent_date(date_str, available)
        if source_date is None:
            return (date_str, False, f"No Sentinel2 data before {date_str}")
        
        # Load and compute
        df_sentinel = load_sentinel(source_date, city='mumbai', sentinel_dir=sentinel_dir)
        df_sentinel = compute_sentinel_indices(df_sentinel)
        df_sentinel['sentinel_date'] = source_date
        
        # Select columns
        merge_cols = ['grid_id']
        for col in ['evi', 'ui_light', 'si', 'cloud_mask', 'sentinel_date']:
            if col in df_sentinel.columns:
                merge_cols.append(col)
        
        # Merge
        df_merged = df_existing.merge(
            df_sentinel[merge_cols],
            on='grid_id',
            how='left'
        )
        
        # Save
        df_merged.to_parquet(training_file, index=False)
        
        # Stats
        n_rows_with_features = df_merged['evi'].notna().sum()
        pct = (n_rows_with_features / len(df_merged)) * 100
        
        return (date_str, True, f"{n_rows_with_features}/{len(df_merged)} rows ({pct:.1f}%)")
    
    except Exception as e:
        return (date_str, False, f"Error: {str(e)[:50]}")


def main():
    parser = argparse.ArgumentParser(description='Add Sentinel2 features to training data (parallelized).')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--training-dir', type=Path, default=PROJECT_ROOT / 'artifacts' / 'v1' / 'training',
                        help='Training data directory')
    parser.add_argument('--sentinel-dir', type=Path, default=DEFAULT_SENTINEL_DIR,
                        help='Sentinel2 data directory')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    # Parse dates
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Generate all dates in range
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print("=" * 80)
    print(f"ADDING SENTINEL2 FEATURES (PARALLELIZED)")
    print("=" * 80)
    print(f"Date range: {args.start_date} to {args.end_date} ({len(dates)} days)")
    print(f"Training dir: {args.training_dir}")
    print(f"Workers: {args.workers}")
    print()
    
    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_date, date, args.training_dir, args.sentinel_dir): date
            for date in dates
        }
        
        # Collect results as they complete
        completed = 0
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for future in as_completed(futures):
            date_str, success, message = future.result()
            completed += 1
            
            if success:
                success_count += 1
                status = "✅"
            elif "already present" in message or "File not found" in message:
                skip_count += 1
                status = "⏭️"
            else:
                fail_count += 1
                status = "❌"
            
            # Print progress (show every 10th or last)
            if completed % 10 == 0 or completed == len(dates):
                print(f"[{completed:4d}/{len(dates)}] {date_str}: {status} {message}")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"  Total: {len(dates)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
