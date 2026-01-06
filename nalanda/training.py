"""Training data assembly module.

This module assembles the final training dataset by joining:
- Target data (PM2.5 labels with temporal lags)
- Meteorological features (OpenMeteo with temporal lags)
- Satellite features (Landsat8 + Sentinel2 indices)

The output is a single parquet file per date containing all features
joined on (row_idx, timestamp).

Output schema:
    - row_idx: int (grid cell identifier)
    - timestamp: datetime (hourly timestamp)
    - pm25_t: float (target PM2.5)
    - target_trust: float (interpolation confidence)
    - n_sensors: int (number of sensors used)
    - [meteo features]: temperature, humidity, wind, lags, etc.
    - [satellite features]: NDVI, EVI, indices, etc.

Usage:
    from nalanda.training import assemble_training_data
    
    df = assemble_training_data('2021-06-12')
    # or
    python -m nalanda.training --date 2021-06-12 --city mumbai
"""

from __future__ import annotations

import argparse
from datetime import datetime as dt
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd

from nalanda.features_meteo import (
    compute_meteo_features,
    get_meteo_feature_columns,
)
from nalanda.features_satellite import (
    compute_landsat_features,
    compute_sentinel_features,
    get_landsat_feature_columns,
    get_sentinel_feature_columns,
)


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TARGET_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"
DEFAULT_GRID_PATH = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"


# ============================================================================
# Grid mapping utilities
# ============================================================================

def load_grid_mapping(grid_path: Path = DEFAULT_GRID_PATH) -> pd.DataFrame:
    """Load row_idx <-> grid_id mapping from grid parquet.
    
    Args:
        grid_path: Path to grid parquet file.
        
    Returns:
        DataFrame with row_idx and grid_id columns.
    """
    return pd.read_parquet(grid_path, columns=['row_idx', 'grid_id'])


def get_target_grid_ids(
    target_df: pd.DataFrame,
    grid_mapping: pd.DataFrame,
) -> Set[str]:
    """Get unique grid_ids from target data.
    
    Args:
        target_df: Target DataFrame with row_idx column.
        grid_mapping: DataFrame with row_idx -> grid_id mapping.
        
    Returns:
        Set of grid_id strings.
    """
    unique_row_idx = set(target_df['row_idx'].unique())
    filtered = grid_mapping[grid_mapping['row_idx'].isin(unique_row_idx)]
    return set(filtered['grid_id'].unique())


# ============================================================================
# Data loading
# ============================================================================

def load_target(
    date: str,
    target_dir: Path = DEFAULT_TARGET_DIR,
) -> pd.DataFrame:
    """Load target parquet for a given date.
    
    Args:
        date: Date string (YYYY-MM-DD).
        target_dir: Base directory for training artifacts.
        
    Returns:
        DataFrame with row_idx, timestamp, pm25_t, target_trust, n_sensors.
        
    Raises:
        FileNotFoundError: If target file doesn't exist.
    """
    target_path = target_dir / date / "target.parquet"
    
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    
    df = pd.read_parquet(target_path)
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


# ============================================================================
# Assembly
# ============================================================================

def assemble_training_data(
    date: str,
    city: str = 'mumbai',
    target_dir: Path = DEFAULT_TARGET_DIR,
    grid_path: Path = DEFAULT_GRID_PATH,
    include_landsat_bands: bool = False,
    include_sentinel_spatial: bool = True,
    skip_missing_satellite: bool = True,
) -> pd.DataFrame:
    """Assemble complete training dataset for a given date.
    
    This is the main entry point for training data assembly. It:
    1. Loads target data
    2. Maps row_idx to grid_id
    3. Computes meteo features (filtered to target grid_ids)
    4. Computes satellite features (filtered to target grid_ids)
    5. Joins everything together
    
    Args:
        date: Target date (YYYY-MM-DD).
        city: City name.
        target_dir: Base directory for target artifacts.
        grid_path: Path to grid parquet.
        include_landsat_bands: Whether to include raw Landsat bands.
        include_sentinel_spatial: Whether to compute spatial Sentinel features.
        skip_missing_satellite: If True, fill NaN for missing satellite data
            instead of raising error.
        
    Returns:
        DataFrame with all features joined on (row_idx, timestamp).
        
    Raises:
        FileNotFoundError: If target file doesn't exist.
        ValueError: If required data is missing and skip_missing_satellite=False.
    """
    print(f"Assembling training data for {date}...")
    
    # 1. Load target data
    print("  Loading target data...")
    target_df = load_target(date, target_dir)
    print(f"    Target: {len(target_df):,} rows, {target_df['row_idx'].nunique():,} unique grids")
    
    # 2. Load grid mapping and get target grid_ids
    print("  Loading grid mapping...")
    grid_mapping = load_grid_mapping(grid_path)
    grid_ids = get_target_grid_ids(target_df, grid_mapping)
    print(f"    Grid IDs: {len(grid_ids):,}")
    
    # 3. Add grid_id to target
    target_with_grid = target_df.merge(grid_mapping, on='row_idx', how='left')
    
    # 4. Compute meteo features
    print("  Computing meteo features...")
    try:
        meteo_df = compute_meteo_features(date, city=city, grid_ids=grid_ids)
        print(f"    Meteo: {len(meteo_df):,} rows, {len(meteo_df.columns)} columns")
        
        # Rename datetime to match target's timestamp
        meteo_df = meteo_df.rename(columns={'datetime': 'timestamp'})
    except FileNotFoundError as e:
        print(f"    Warning: Meteo data not found - {e}")
        meteo_df = None
    
    # 5. Compute satellite features
    print("  Computing Landsat features...")
    try:
        landsat_df = compute_landsat_features(
            date, 
            city=city, 
            grid_ids=grid_ids,
            include_bands=include_landsat_bands,
        )
        if not landsat_df.empty:
            print(f"    Landsat: {len(landsat_df):,} rows (source: {landsat_df['landsat_date'].iloc[0]})")
        else:
            print("    Landsat: 0 rows (no data)")
            landsat_df = None
    except (FileNotFoundError, ValueError) as e:
        if skip_missing_satellite:
            print(f"    Warning: Landsat data not found - {e}")
            landsat_df = None
        else:
            raise
    
    print("  Computing Sentinel features...")
    try:
        sentinel_df = compute_sentinel_features(
            date, 
            city=city, 
            grid_ids=grid_ids,
            include_spatial=include_sentinel_spatial,
        )
        if not sentinel_df.empty:
            print(f"    Sentinel: {len(sentinel_df):,} rows (source: {sentinel_df['sentinel_date'].iloc[0]})")
        else:
            print("    Sentinel: 0 rows (no data)")
            sentinel_df = None
    except (FileNotFoundError, ValueError) as e:
        if skip_missing_satellite:
            print(f"    Warning: Sentinel data not found - {e}")
            sentinel_df = None
        else:
            raise
    
    # 6. Join everything
    print("  Joining datasets...")
    result = target_with_grid.copy()
    
    # Join meteo on (grid_id, timestamp)
    if meteo_df is not None:
        result = result.merge(
            meteo_df,
            on=['grid_id', 'timestamp'],
            how='left',
        )
        print(f"    After meteo join: {len(result):,} rows")
    
    # Join Landsat on grid_id (static per day)
    if landsat_df is not None:
        result = result.merge(
            landsat_df,
            on='grid_id',
            how='left',
        )
        print(f"    After Landsat join: {len(result):,} rows")
    
    # Join Sentinel on grid_id (static per day)
    if sentinel_df is not None:
        result = result.merge(
            sentinel_df,
            on='grid_id',
            how='left',
        )
        print(f"    After Sentinel join: {len(result):,} rows")
    
    # 7. Reorder columns
    # Keys first, then target, then features
    key_cols = ['row_idx', 'timestamp']
    target_cols = ['pm25_t', 'target_trust', 'n_sensors']
    meta_cols = ['grid_id', 'landsat_date', 'sentinel_date']
    
    # Get feature columns (everything else)
    all_cols = result.columns.tolist()
    feature_cols = [c for c in all_cols if c not in key_cols + target_cols + meta_cols]
    
    # Final column order
    final_cols = key_cols + target_cols
    for c in meta_cols:
        if c in result.columns:
            final_cols.append(c)
    final_cols.extend(feature_cols)
    
    # Filter to existing columns
    final_cols = [c for c in final_cols if c in result.columns]
    result = result[final_cols]
    
    print(f"  Final: {len(result):,} rows, {len(result.columns)} columns")
    
    return result


def save_training_data(
    df: pd.DataFrame,
    date: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save assembled training data to parquet.
    
    Args:
        df: Training DataFrame.
        date: Date string.
        output_dir: Base output directory.
        
    Returns:
        Path to saved parquet file.
    """
    output_path = output_dir / date / "training.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    
    return output_path


def get_training_feature_columns(
    include_landsat_bands: bool = False,
) -> list[str]:
    """Return list of all training feature column names.
    
    Useful for schema validation and feature selection.
    """
    cols = []
    cols.extend(get_meteo_feature_columns())
    cols.extend(get_landsat_feature_columns(include_bands=include_landsat_bands))
    cols.extend(get_sentinel_feature_columns())
    return cols


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Assemble training data for a given date"
    )
    parser.add_argument(
        '--date', '-d',
        required=True,
        help='Target date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--city', '-c',
        default='mumbai',
        help='City name (default: mumbai)',
    )
    parser.add_argument(
        '--include-bands',
        action='store_true',
        help='Include raw Landsat bands',
    )
    parser.add_argument(
        '--no-spatial',
        action='store_true',
        help='Skip Sentinel spatial features (faster)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Compute but do not save',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory',
    )
    
    args = parser.parse_args()
    
    # Assemble
    df = assemble_training_data(
        date=args.date,
        city=args.city,
        include_landsat_bands=args.include_bands,
        include_sentinel_spatial=not args.no_spatial,
    )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Training data assembled for {args.date}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nColumn summary:")
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        dtype = df[col].dtype
        print(f"  {col}: {dtype}, {null_pct:.1f}% null")
    
    # Save
    if not args.dry_run:
        output_path = save_training_data(df, args.date, args.output_dir)
        print(f"\n✓ Saved to: {output_path}")
    else:
        print("\n(dry run - not saved)")


# ============================================================================
# Split materialization
# ============================================================================

def _determine_split_for_file(
    parquet_file: Path,
    timestamp_col: str = "timestamp",
) -> Optional[str]:
    """Determine which split a parquet file belongs to based on its timestamps.
    
    Reads only the timestamp column to minimize memory usage.
    
    Args:
        parquet_file: Path to parquet file
        timestamp_col: Name of timestamp column
        
    Returns:
        Split name ('train', 'validation', 'test') or None if mixed/gap/excluded
    """
    from nalanda.splits import get_split
    
    # Read only timestamp column
    df = pd.read_parquet(parquet_file, columns=[timestamp_col])
    
    if len(df) == 0:
        return None
    
    # Check all timestamps belong to same split
    splits_in_file = set()
    for ts in df[timestamp_col]:
        try:
            split = get_split(pd.Timestamp(ts))
            if split not in ['train', 'validation', 'test']:
                # Skip gap/excluded data
                continue
            splits_in_file.add(split)
        except ValueError:
            # 2025+ or other excluded timestamps
            continue
    
    # File must belong to exactly one split
    if len(splits_in_file) == 1:
        return list(splits_in_file)[0]
    elif len(splits_in_file) == 0:
        # All timestamps are gap/excluded
        return None
    else:
        # Mixed splits in one file - need to load and filter
        return "mixed"


def _copy_file_to_split(
    source_file: Path,
    split_name: str,
    output_dirs: dict,
    timestamp_col: str = "timestamp",
) -> dict:
    """Copy a parquet file to the appropriate split directory.
    
    If file contains mixed splits, loads and filters the data.
    
    Args:
        source_file: Source parquet file
        split_name: Target split ('train', 'validation', 'test', or 'mixed')
        output_dirs: Dict mapping split names to output directories
        timestamp_col: Name of timestamp column
        
    Returns:
        Dict with file metadata (rows, min/max timestamps, target split)
    """
    from nalanda.splits import get_split_mask
    import shutil
    
    if split_name == "mixed":
        # Load full file and split it
        df = pd.read_parquet(source_file)
        
        results = []
        for target_split in ['train', 'validation', 'test']:
            mask = get_split_mask(df, target_split, timestamp_col)
            split_df = df[mask]
            
            if len(split_df) == 0:
                continue
            
            # Generate unique filename
            output_dir = output_dirs[target_split]
            output_file = output_dir / f"{source_file.parent.name}_{source_file.stem}_{target_split}.parquet"
            
            # Save filtered data
            split_df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)
            
            results.append({
                "split": target_split,
                "source": str(source_file),
                "target": str(output_file),
                "rows": len(split_df),
                "min_timestamp": str(split_df[timestamp_col].min()),
                "max_timestamp": str(split_df[timestamp_col].max()),
            })
        
        return {"type": "mixed", "results": results}
    
    elif split_name in ['train', 'validation', 'test']:
        # Simple copy - all data belongs to one split
        output_dir = output_dirs[split_name]
        output_file = output_dir / f"{source_file.parent.name}_{source_file.stem}.parquet"
        
        # Copy file
        shutil.copy2(source_file, output_file)
        
        # Read metadata
        df = pd.read_parquet(source_file, columns=[timestamp_col])
        
        return {
            "type": "copy",
            "split": split_name,
            "source": str(source_file),
            "target": str(output_file),
            "rows": len(df),
            "min_timestamp": str(df[timestamp_col].min()),
            "max_timestamp": str(df[timestamp_col].max()),
        }
    
    else:
        # Skip (gap/excluded data)
        return {"type": "skip", "source": str(source_file)}


def materialize_splits(
    input_dir: Path = DEFAULT_OUTPUT_DIR,
    output_base_dir: Optional[Path] = None,
    timestamp_col: str = "timestamp",
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> dict:
    """Materialize train/val/test splits from assembled training data.
    
    Uses file-by-file processing to avoid loading all data into memory.
    Parallelizes file processing for speed.
    
    Args:
        input_dir: Directory containing date-partitioned training parquet files
        output_base_dir: Base directory for output (creates train/, val/, test/ subdirs)
        timestamp_col: Name of timestamp column
        parallel: If True, process files in parallel
        max_workers: Max parallel workers (default: CPU count)
        
    Returns:
        Dictionary with metadata about each split (row counts, date ranges, file counts)
    """
    from nalanda.splits import validate_splits
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import hashlib
    import json
    
    if output_base_dir is None:
        output_base_dir = PROJECT_ROOT / "artifacts"
    
    output_base_dir = Path(output_base_dir)
    
    # Create output directories
    train_dir = output_base_dir / "train"
    val_dir = output_base_dir / "val"
    test_dir = output_base_dir / "test"
    
    output_dirs = {
        "train": train_dir,
        "validation": val_dir,
        "test": test_dir,
    }
    
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning training data from: {input_dir}")
    
    # Find all parquet files
    date_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if not date_dirs:
        raise ValueError(f"No date directories found in {input_dir}")
    
    all_parquet_files = []
    for date_dir in date_dirs:
        parquet_files = list(date_dir.glob("*.parquet"))
        all_parquet_files.extend(parquet_files)
    
    if not all_parquet_files:
        raise ValueError(f"No parquet files found in {input_dir}")
    
    print(f"  Found {len(all_parquet_files)} parquet files from {len(date_dirs)} dates")
    
    # Process files
    print("\nDetermining split assignments...")
    
    file_assignments = {}
    
    if parallel and len(all_parquet_files) > 1:
        print(f"  Processing in parallel (workers: {max_workers or 'auto'})...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_determine_split_for_file, pq_file, timestamp_col): pq_file
                for pq_file in all_parquet_files
            }
            
            for future in as_completed(future_to_file):
                pq_file = future_to_file[future]
                try:
                    split_name = future.result()
                    if split_name:
                        file_assignments[pq_file] = split_name
                except Exception as e:
                    print(f"  ⚠️  Error processing {pq_file.name}: {e}")
    else:
        for pq_file in all_parquet_files:
            split_name = _determine_split_for_file(pq_file, timestamp_col)
            if split_name:
                file_assignments[pq_file] = split_name
    
    # Summary of assignments
    assignment_counts = {}
    for split_name in file_assignments.values():
        assignment_counts[split_name] = assignment_counts.get(split_name, 0) + 1
    
    print(f"  File assignments:")
    for split_name, count in sorted(assignment_counts.items()):
        print(f"    {split_name}: {count} files")
    
    # Copy files to splits
    print("\nCopying files to split directories...")

    copy_results = []

    # Group files so we run target files first, then training files, then others.
    def _group_files_assignments(assignments: dict):
        target_group = []
        training_group = []
        other_group = []
        for sf, sn in assignments.items():
            name = sf.name.lower()
            if "target" in name:
                target_group.append((sf, sn))
            elif "training" in name or ("train" in name and "training" not in name):
                training_group.append((sf, sn))
            else:
                other_group.append((sf, sn))
        return [target_group, training_group, other_group]

    groups = _group_files_assignments(file_assignments)

    # Process each group sequentially (target -> training -> others). Within each group
    # we may use parallel workers if requested.
    for group_idx, group in enumerate(groups, 1):
        if not group:
            continue

        group_name = "target" if group_idx == 1 else ("training" if group_idx == 2 else "other")
        print(f"  Processing group: {group_name} ({len(group)} files)")

        if parallel and len(group) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        _copy_file_to_split,
                        source_file,
                        split_name,
                        output_dirs,
                        timestamp_col
                    ): source_file
                    for source_file, split_name in group
                }

                completed = 0
                for future in as_completed(future_to_file):
                    completed += 1
                    try:
                        result = future.result()
                        copy_results.append(result)
                        if completed % 10 == 0:
                            print(f"  Processed {completed}/{len(group)} files in group {group_name}...")
                    except Exception as e:
                        source_file = future_to_file[future]
                        print(f"  ⚠️  Error copying {source_file.name}: {e}")
        else:
            for source_file, split_name in group:
                try:
                    result = _copy_file_to_split(source_file, split_name, output_dirs, timestamp_col)
                    copy_results.append(result)
                except Exception as e:
                    print(f"  ⚠️  Error copying {source_file.name}: {e}")
    
    # Aggregate results
    print("\nAggregating split statistics...")
    
    split_stats = {
        "train": {"files": [], "rows": 0, "min_ts": None, "max_ts": None},
        "validation": {"files": [], "rows": 0, "min_ts": None, "max_ts": None},
        "test": {"files": [], "rows": 0, "min_ts": None, "max_ts": None},
    }
    
    for result in copy_results:
        if result["type"] == "mixed":
            for sub_result in result["results"]:
                split = sub_result["split"]
                split_stats[split]["files"].append(sub_result["target"])
                split_stats[split]["rows"] += sub_result["rows"]
                
                min_ts = pd.Timestamp(sub_result["min_timestamp"])
                max_ts = pd.Timestamp(sub_result["max_timestamp"])
                
                if split_stats[split]["min_ts"] is None or min_ts < split_stats[split]["min_ts"]:
                    split_stats[split]["min_ts"] = min_ts
                if split_stats[split]["max_ts"] is None or max_ts > split_stats[split]["max_ts"]:
                    split_stats[split]["max_ts"] = max_ts
        
        elif result["type"] == "copy":
            split = result["split"]
            split_stats[split]["files"].append(result["target"])
            split_stats[split]["rows"] += result["rows"]
            
            min_ts = pd.Timestamp(result["min_timestamp"])
            max_ts = pd.Timestamp(result["max_timestamp"])
            
            if split_stats[split]["min_ts"] is None or min_ts < split_stats[split]["min_ts"]:
                split_stats[split]["min_ts"] = min_ts
            if split_stats[split]["max_ts"] is None or max_ts > split_stats[split]["max_ts"]:
                split_stats[split]["max_ts"] = max_ts
    
    # Compute directory-level checksums and sizes
    print("\nComputing checksums and sizes...")
    
    splits_metadata = {}
    
    for split_name in ["train", "validation", "test"]:
        stats = split_stats[split_name]
        output_dir = output_dirs[split_name]
        
        if stats["rows"] == 0:
            print(f"  ⚠️  {split_name.upper()}: No data")
            continue
        
        # Compute total size
        total_size = sum(Path(f).stat().st_size for f in stats["files"]) / (1024 * 1024)
        
        # Compute combined checksum of all files
        hash_obj = hashlib.sha256()
        for file_path in sorted(stats["files"]):
            with open(file_path, "rb") as f:
                hash_obj.update(f.read())
        combined_hash = hash_obj.hexdigest()
        
        splits_metadata[split_name] = {
            "path": str(output_dir),
            "file_count": len(stats["files"]),
            "rows": stats["rows"],
            "min_timestamp": str(stats["min_ts"]),
            "max_timestamp": str(stats["max_ts"]),
            "size_mb": round(total_size, 2),
            "sha256": combined_hash,
        }
        
        print(f"  {split_name.upper()}:")
        print(f"    Files: {len(stats['files'])}")
        print(f"    Rows: {stats['rows']:,}")
        print(f"    Date range: {stats['min_ts']} to {stats['max_ts']}")
        print(f"    Size: {total_size:.2f} MB")
        print(f"    SHA256: {combined_hash[:16]}...")
    
    # Save metadata
    metadata_path = output_base_dir / "splits_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(splits_metadata, f, indent=2)
    
    print(f"\n✓ Split metadata saved to: {metadata_path}")
    
    return splits_metadata


if __name__ == '__main__':
    main()
