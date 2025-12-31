"""OpenMeteo feature engineering for training data assembly.

This module computes derived meteorological features from hourly OpenMeteo data:
- Wind decomposition (u/v components)
- Vapor pressure deficit (VPD)
- Temperature-dewpoint difference
- Temporal lags for key variables
- Precipitation rolling sums

Optimized for 500k+ grid cells using numpy reshape operations instead of
pandas groupby (50-100x faster).

Usage:
    from nalanda.features_meteo import compute_meteo_features
    features_df = compute_meteo_features(date='2021-06-07', city='mumbai')
"""

from __future__ import annotations

from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_METEO_DIR = PROJECT_ROOT / "artifacts" / "v1" / "openmeteo"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"
DEFAULT_GRID_PATH = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"

# Variables to apply temporal lags
LAG_VARIABLES = [
    'temperature_2m',
    'relative_humidity_2m', 
    'dewpoint_2m',
    'surface_pressure',
    'wind_u_10m',
    'wind_v_10m',
    'cloud_cover',
]

# Lag hours to compute
LAG_HOURS = [1, 3, 6, 12, 24]

# Cloud cover only gets subset of lags
CLOUD_COVER_LAG_HOURS = [6, 24]


def load_hourly_meteo(
    date: str,
    city: str = 'mumbai',
    meteo_dir: Path = DEFAULT_METEO_DIR,
    grid_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Load hourly OpenMeteo data for a given date.
    
    Args:
        date: Date string in YYYY-MM-DD format.
        city: City name for path construction.
        meteo_dir: Base directory for OpenMeteo artifacts.
        grid_ids: Optional set of grid_ids to filter to. If provided,
            only rows matching these grid_ids are returned.
        
    Returns:
        DataFrame with columns: grid_id, datetime, temperature_2m, 
        relative_humidity_2m, dewpoint_2m, surface_pressure,
        wind_speed_10m, wind_direction_10m, precipitation, cloud_cover
        
    Raises:
        FileNotFoundError: If hourly file doesn't exist.
    """
    hourly_path = meteo_dir / city / f"{date}_hourly.parquet"
    
    if not hourly_path.exists():
        raise FileNotFoundError(f"Hourly meteo file not found: {hourly_path}")
    
    df = pd.read_parquet(hourly_path)
    
    # Filter to specific grid_ids if provided
    if grid_ids is not None:
        df = df[df['grid_id'].isin(grid_ids)].copy()
    
    # Ensure datetime is proper type
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df


def load_target_grid_ids(
    date: str,
    target_dir: Path = DEFAULT_TARGET_DIR,
    grid_path: Path = DEFAULT_GRID_PATH,
) -> Set[str]:
    """Load unique grid_ids from target parquet for a given date.
    
    Target data uses row_idx, so we need to map to grid_id via grid parquet.
    
    Args:
        date: Date string in YYYY-MM-DD format.
        target_dir: Base directory for training artifacts.
        grid_path: Path to grid parquet with row_idx ↔ grid_id mapping.
        
    Returns:
        Set of grid_id strings present in the target data.
        
    Raises:
        FileNotFoundError: If target or grid file doesn't exist.
    """
    target_path = target_dir / date / "target.parquet"
    
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    
    # Read unique row_idx from target
    target_df = pd.read_parquet(target_path, columns=['row_idx'])
    unique_row_idx = set(target_df['row_idx'].unique())
    
    # Load grid mapping (only row_idx and grid_id columns)
    grid_df = pd.read_parquet(grid_path, columns=['row_idx', 'grid_id'])
    
    # Filter to target row_idx and get corresponding grid_ids
    filtered = grid_df[grid_df['row_idx'].isin(unique_row_idx)]
    
    return set(filtered['grid_id'].unique())


def _compute_lag_matrix(
    matrix: np.ndarray,
    lag_hours: int,
) -> np.ndarray:
    """Compute temporal lag using numpy roll on matrix.
    
    Args:
        matrix: (n_grids, n_hours) array.
        lag_hours: Number of hours to lag.
        
    Returns:
        (n_grids, n_hours) float array with lagged values.
        Values at hours < lag_hours will be from previous periods or NaN.
    """
    # Convert to float to allow NaN
    lagged = np.roll(matrix, lag_hours, axis=1).astype(np.float64)
    # Set first lag_hours columns to NaN (no valid lag data)
    lagged[:, :lag_hours] = np.nan
    return lagged


def _compute_rolling_sum(
    matrix: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling sum along time axis using cumsum trick.
    
    Args:
        matrix: (n_grids, n_hours) array.
        window: Rolling window size in hours.
        
    Returns:
        (n_grids, n_hours) array with rolling sums.
    """
    n_grids, n_hours = matrix.shape
    
    # Pad with zeros for cumsum
    padded = np.concatenate([np.zeros((n_grids, window)), matrix], axis=1)
    cumsum = np.cumsum(padded, axis=1)
    
    # Rolling sum = cumsum[i+window] - cumsum[i]
    result = cumsum[:, window:] - cumsum[:, :-window]
    
    return result


def compute_meteo_features(
    date: str,
    city: str = 'mumbai',
    meteo_dir: Path = DEFAULT_METEO_DIR,
    include_raw_wind: bool = False,
    prev_day_df: Optional[pd.DataFrame] = None,
    grid_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Compute all meteorological features for a given date.
    
    This is the main entry point for OpenMeteo feature engineering.
    Uses numpy reshape operations for ~50x speedup over pandas groupby.
    
    Args:
        date: Date string in YYYY-MM-DD format.
        city: City name.
        meteo_dir: Base directory for OpenMeteo artifacts.
        include_raw_wind: Whether to keep raw wind_speed/direction columns.
        prev_day_df: Optional pre-loaded previous day's hourly data.
            If None, will attempt to load automatically.
        grid_ids: Optional set of grid_ids to filter to. If provided,
            only rows matching these grid_ids are processed. This dramatically
            improves performance when you only need a subset (e.g., sensor-covered
            grids from target data: ~59k grids vs 509k full grid).
        
    Returns:
        DataFrame with columns:
            - grid_id, datetime (keys)
            - temperature_2m, temperature_2m_lag1h, ..., temperature_2m_lag24h
            - relative_humidity_2m, relative_humidity_2m_lag1h, ...
            - dewpoint_2m, dewpoint_2m_lag1h, ...
            - surface_pressure, surface_pressure_lag1h, ...
            - wind_u_10m, wind_u_10m_lag1h, ...
            - wind_v_10m, wind_v_10m_lag1h, ...
            - cloud_cover, cloud_cover_lag6h, cloud_cover_lag24h
            - vpd, temp_dewpoint_diff
            - precip_last_1h, precip_last_6h, precip_last_24h
    """
    # Load current day's data (filtered if grid_ids provided)
    df = load_hourly_meteo(date, city, meteo_dir, grid_ids=grid_ids)
    
    # Try to load previous day for cross-day lags
    if prev_day_df is None:
        prev_date = (dt.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            prev_day_df = load_hourly_meteo(prev_date, city, meteo_dir, grid_ids=grid_ids)
        except FileNotFoundError:
            prev_day_df = None
    
    # Get grid count and verify structure
    n_grids = df['grid_id'].nunique()
    n_hours_current = df['datetime'].nunique()
    
    if len(df) != n_grids * n_hours_current:
        raise ValueError(
            f"Data structure mismatch: {len(df)} rows != {n_grids} grids × {n_hours_current} hours"
        )
    
    # Sort by (grid_id, datetime) for reshape
    df = df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
    
    # Compute wind components (vectorized, no reshape needed)
    direction_rad = np.deg2rad(df['wind_direction_10m'].values)
    df['wind_u_10m'] = df['wind_speed_10m'].values * np.cos(direction_rad)
    df['wind_v_10m'] = df['wind_speed_10m'].values * np.sin(direction_rad)
    
    # Compute derived signals
    T = df['temperature_2m'].values
    RH = df['relative_humidity_2m'].values
    svp = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    df['vpd'] = svp * (1 - RH / 100)
    df['temp_dewpoint_diff'] = df['temperature_2m'].values - df['dewpoint_2m'].values
    
    # Prepare for matrix operations
    if prev_day_df is not None and len(prev_day_df) > 0:
        # Sort previous day
        prev_day_df = prev_day_df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        n_hours_prev = prev_day_df['datetime'].nunique()
        
        # Add wind components to prev day
        prev_dir_rad = np.deg2rad(prev_day_df['wind_direction_10m'].values)
        prev_day_df['wind_u_10m'] = prev_day_df['wind_speed_10m'].values * np.cos(prev_dir_rad)
        prev_day_df['wind_v_10m'] = prev_day_df['wind_speed_10m'].values * np.sin(prev_dir_rad)
        
        use_combined = True
    else:
        n_hours_prev = 0
        use_combined = False
    
    # Variables to process
    base_vars = ['temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
                 'surface_pressure', 'wind_u_10m', 'wind_v_10m', 'cloud_cover',
                 'precipitation']
    
    # Create combined matrices for lag computation
    combined_matrices = {}
    if use_combined:
        for var in base_vars:
            if var in df.columns and var in prev_day_df.columns:
                prev_mat = prev_day_df[var].values.reshape(n_grids, n_hours_prev)
                curr_mat = df[var].values.reshape(n_grids, n_hours_current)
                combined_matrices[var] = np.concatenate([prev_mat, curr_mat], axis=1)
            elif var in df.columns:
                # Pad with NaN if prev day missing this var
                curr_mat = df[var].values.reshape(n_grids, n_hours_current)
                pad = np.full((n_grids, n_hours_prev), np.nan)
                combined_matrices[var] = np.concatenate([pad, curr_mat], axis=1)
    else:
        for var in base_vars:
            if var in df.columns:
                combined_matrices[var] = df[var].values.reshape(n_grids, n_hours_current)
    
    # Compute lags using matrix operations
    lag_results = {}
    
    for var in LAG_VARIABLES:
        if var not in combined_matrices:
            continue
        
        matrix = combined_matrices[var]
        
        # Determine which lags for this variable
        if var == 'cloud_cover':
            var_lag_hours = CLOUD_COVER_LAG_HOURS
        else:
            var_lag_hours = LAG_HOURS
        
        for lag_h in var_lag_hours:
            lag_col = f'{var}_lag{lag_h}h'
            lagged = _compute_lag_matrix(matrix, lag_h)
            
            # Extract only current day's portion
            if use_combined:
                lagged = lagged[:, n_hours_prev:]
            
            lag_results[lag_col] = lagged.ravel()
    
    # Compute precipitation rolling sums
    if 'precipitation' in combined_matrices:
        precip_matrix = combined_matrices['precipitation']
        
        # 1h = current value
        if use_combined:
            precip_1h = precip_matrix[:, n_hours_prev:]
        else:
            precip_1h = precip_matrix
        lag_results['precip_last_1h'] = precip_1h.ravel()
        
        # 6h rolling sum
        precip_6h = _compute_rolling_sum(precip_matrix, 6)
        if use_combined:
            precip_6h = precip_6h[:, n_hours_prev:]
        lag_results['precip_last_6h'] = precip_6h.ravel()
        
        # 24h rolling sum
        precip_24h = _compute_rolling_sum(precip_matrix, 24)
        if use_combined:
            precip_24h = precip_24h[:, n_hours_prev:]
        lag_results['precip_last_24h'] = precip_24h.ravel()
    
    # Add all computed features to DataFrame
    for col, values in lag_results.items():
        df[col] = values
    
    # Build output column order
    output_cols = ['grid_id', 'datetime']
    
    # Base variables with their lags
    for var in ['temperature_2m', 'relative_humidity_2m', 'dewpoint_2m', 
                'surface_pressure', 'wind_u_10m', 'wind_v_10m']:
        output_cols.append(var)
        for lag_h in LAG_HOURS:
            output_cols.append(f'{var}_lag{lag_h}h')
    
    # Cloud cover with subset of lags
    output_cols.append('cloud_cover')
    for lag_h in CLOUD_COVER_LAG_HOURS:
        output_cols.append(f'cloud_cover_lag{lag_h}h')
    
    # Derived signals
    output_cols.extend(['vpd', 'temp_dewpoint_diff'])
    
    # Precipitation aggregates
    output_cols.extend(['precip_last_1h', 'precip_last_6h', 'precip_last_24h'])
    
    # Optionally include raw wind
    if include_raw_wind:
        output_cols.extend(['wind_speed_10m', 'wind_direction_10m'])
    
    # Filter to existing columns
    output_cols = [c for c in output_cols if c in df.columns]
    
    return df[output_cols]


def get_meteo_feature_columns() -> List[str]:
    """Return list of all meteo feature column names (excluding keys).
    
    Useful for schema validation and feature selection.
    """
    cols = []
    
    # Base variables with lags
    for var in ['temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
                'surface_pressure', 'wind_u_10m', 'wind_v_10m']:
        cols.append(var)
        for lag_h in LAG_HOURS:
            cols.append(f'{var}_lag{lag_h}h')
    
    # Cloud cover with subset of lags
    cols.append('cloud_cover')
    for lag_h in CLOUD_COVER_LAG_HOURS:
        cols.append(f'cloud_cover_lag{lag_h}h')
    
    # Derived signals
    cols.extend(['vpd', 'temp_dewpoint_diff'])
    
    # Precipitation aggregates
    cols.extend(['precip_last_1h', 'precip_last_6h', 'precip_last_24h'])
    
    return cols


if __name__ == '__main__':
    import sys
    import time
    
    date = sys.argv[1] if len(sys.argv) > 1 else '2018-01-02'
    use_filter = '--filter' in sys.argv or '-f' in sys.argv
    
    if use_filter:
        print(f"Loading target grid_ids for {date}...")
        try:
            grid_ids = load_target_grid_ids(date)
            print(f"  Found {len(grid_ids):,} unique grid_ids in target")
        except FileNotFoundError:
            print(f"  No target file for {date}, using all grids")
            grid_ids = None
    else:
        grid_ids = None
        print("(Use --filter to process only target grid_ids)")
    
    print(f"\nComputing meteo features for {date}...")
    start = time.time()
    df = compute_meteo_features(date, grid_ids=grid_ids)
    elapsed = time.time() - start
    
    print(f"\nShape: {df.shape}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Unique grids: {df['grid_id'].nunique():,}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        print(f"  {col}: {df[col].dtype}, {null_pct:.1f}% null")
    
    print(f"\nSample (first grid cell, first 5 hours):")
    sample = df[df['grid_id'] == df['grid_id'].iloc[0]].head(5)
    print(sample[['grid_id', 'datetime', 'temperature_2m', 'temperature_2m_lag1h', 
                  'temperature_2m_lag24h', 'vpd', 'precip_last_6h']].to_string())
