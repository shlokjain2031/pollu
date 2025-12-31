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


if __name__ == '__main__':
    main()
