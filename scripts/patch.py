"""Data patching utilities for fixing existing artifacts.

This module contains patches to apply to existing training.parquet files
to fix schema issues, drop deprecated features, or correct data quality problems.

Each patch function should:
- Be idempotent (safe to run multiple times)
- Log what it's doing
- Handle missing files/columns gracefully
- Return summary statistics

Usage:
    python scripts/patch.py --patch drop_temperature_lag6h --dry-run
    python scripts/patch.py --patch drop_temperature_lag6h
    python scripts/patch.py --patch all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"


# ============================================================================
# Patch 1: Drop temperature_2m_lag6h
# ============================================================================

def patch_drop_temperature_lag6h(dry_run: bool = False) -> Dict[str, int]:
    """Drop temperature_2m_lag6h column from all training.parquet files.
    
    Reason: Low autocorrelation (~0.13) in validation tests indicates this
    lag feature doesn't provide useful signal. Removing to reduce feature count.
    
    Args:
        dry_run: If True, report what would be done without modifying files
        
    Returns:
        Dictionary with statistics:
            - files_scanned: total parquet files found
            - files_with_column: files containing the column
            - files_patched: files actually modified
            - files_skipped: files already missing the column
            - bytes_saved: total disk space saved
    """
    stats = {
        "files_scanned": 0,
        "files_with_column": 0,
        "files_patched": 0,
        "files_skipped": 0,
        "bytes_saved": 0,
    }
    
    if not TRAINING_DIR.exists():
        print(f"Training directory not found: {TRAINING_DIR}")
        return stats
    
    # Find all training.parquet files
    parquet_files = sorted(TRAINING_DIR.glob("*/training.parquet"))
    stats["files_scanned"] = len(parquet_files)
    
    if stats["files_scanned"] == 0:
        print(f"No training.parquet files found in {TRAINING_DIR}")
        return stats
    
    print(f"\nScanning {stats['files_scanned']} training.parquet files...")
    if dry_run:
        print("DRY RUN: no files will be modified\n")
    
    for parquet_path in tqdm(parquet_files, desc="Processing files"):
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_path)
            
            # Check if column exists
            if "temperature_2m_lag6h" not in df.columns:
                stats["files_skipped"] += 1
                continue
            
            stats["files_with_column"] += 1
            
            # Calculate size before
            size_before = parquet_path.stat().st_size
            
            if not dry_run:
                # Drop column
                df = df.drop(columns=["temperature_2m_lag6h"])
                
                # Write back
                df.to_parquet(parquet_path, index=False, compression="snappy")
                
                # Calculate size after
                size_after = parquet_path.stat().st_size
                stats["bytes_saved"] += (size_before - size_after)
                stats["files_patched"] += 1
            else:
                # Estimate savings
                col_bytes = df["temperature_2m_lag6h"].memory_usage(deep=True)
                stats["bytes_saved"] += int(col_bytes * 0.5)  # Conservative estimate with compression
                stats["files_patched"] += 1
                
        except Exception as e:
            print(f"\nError processing {parquet_path}: {e}")
            continue
    
    return stats


# ============================================================================
# Patch 2: Sort timestamps by (row_idx, timestamp)
# ============================================================================

def patch_sort_timestamps(dry_run: bool = False) -> Dict[str, int]:
    """Sort all training.parquet and target.parquet files by (row_idx, timestamp).
    
    Reason: Concatenation order from parallel processing can produce unordered
    timestamps within each grid cell. Sorting ensures deterministic ordering.
    
    Args:
        dry_run: If True, report what would be done without modifying files
        
    Returns:
        Dictionary with statistics:
            - files_scanned: total parquet files found
            - files_needing_sort: files that were not properly sorted
            - files_patched: files actually modified
            - files_already_sorted: files already in correct order
    """
    stats = {
        "files_scanned": 0,
        "files_needing_sort": 0,
        "files_patched": 0,
        "files_already_sorted": 0,
    }
    
    if not TRAINING_DIR.exists():
        print(f"Training directory not found: {TRAINING_DIR}")
        return stats
    
    # Find all training.parquet and target.parquet files
    training_files = sorted(TRAINING_DIR.glob("*/training.parquet"))
    target_files = sorted(TRAINING_DIR.glob("*/target.parquet"))
    parquet_files = training_files + target_files
    
    stats["files_scanned"] = len(parquet_files)
    
    if stats["files_scanned"] == 0:
        print(f"No parquet files found in {TRAINING_DIR}")
        return stats
    
    print(f"\nScanning {stats['files_scanned']} parquet files...")
    if dry_run:
        print("DRY RUN: no files will be modified\n")
    
    for parquet_path in tqdm(parquet_files, desc="Processing files"):
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_path)
            
            # Check if row_idx and timestamp columns exist
            if "row_idx" not in df.columns or "timestamp" not in df.columns:
                continue
            
            # Check if already sorted
            df_sorted = df.sort_values(['row_idx', 'timestamp']).reset_index(drop=True)
            is_sorted = df.equals(df_sorted)
            
            if is_sorted:
                stats["files_already_sorted"] += 1
                continue
            
            stats["files_needing_sort"] += 1
            
            if not dry_run:
                # Write sorted dataframe back
                df_sorted.to_parquet(parquet_path, index=False, compression="snappy")
                stats["files_patched"] += 1
            else:
                stats["files_patched"] += 1
                
        except Exception as e:
            print(f"\nError processing {parquet_path}: {e}")
            continue
    
    return stats


# ============================================================================
# Patch Registry
# ============================================================================

AVAILABLE_PATCHES = {
    "drop_temperature_lag6h": patch_drop_temperature_lag6h,
    "sort_timestamps": patch_sort_timestamps,
    # Add future patches here
}


def apply_all_patches(dry_run: bool = False) -> None:
    """Apply all available patches sequentially."""
    print(f"\n{'='*70}")
    print(f"Applying {len(AVAILABLE_PATCHES)} patches")
    print(f"{'='*70}\n")
    
    for patch_name, patch_func in AVAILABLE_PATCHES.items():
        print(f"\n{'-'*70}")
        print(f"Patch: {patch_name}")
        print(f"{'-'*70}")
        
        stats = patch_func(dry_run=dry_run)
        print_stats(stats, dry_run=dry_run)


def print_stats(stats: Dict[str, int], dry_run: bool = False) -> None:
    """Pretty print patch statistics."""
    print(f"\nResults:")
    print(f"  Files scanned:       {stats.get('files_scanned', 0):>6}")
    
    # Column drop stats
    if "files_with_column" in stats:
        print(f"  Files with column:   {stats['files_with_column']:>6}")
    
    # Sort stats
    if "files_needing_sort" in stats:
        print(f"  Files needing sort:  {stats['files_needing_sort']:>6}")
    if "files_already_sorted" in stats:
        print(f"  Already sorted:      {stats['files_already_sorted']:>6}")
    
    print(f"  Files patched:       {stats.get('files_patched', 0):>6}")
    print(f"  Files skipped:       {stats.get('files_skipped', 0):>6}")
    
    if stats.get('bytes_saved', 0) > 0:
        mb_saved = stats['bytes_saved'] / (1024 * 1024)
        print(f"  Space saved:         {mb_saved:>6.1f} MB")
    
    if dry_run:
        print(f"\n  (DRY RUN - no files were modified)")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply data patches to training artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--patch",
        choices=list(AVAILABLE_PATCHES.keys()) + ["all"],
        required=True,
        help="Patch to apply (or 'all' for all patches)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files",
    )
    
    args = parser.parse_args()
    
    # Apply patch(es)
    if args.patch == "all":
        apply_all_patches(dry_run=args.dry_run)
    else:
        patch_func = AVAILABLE_PATCHES[args.patch]
        print(f"\n{'='*70}")
        print(f"Applying patch: {args.patch}")
        print(f"{'='*70}\n")
        stats = patch_func(dry_run=args.dry_run)
        print_stats(stats, dry_run=args.dry_run)
    
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
