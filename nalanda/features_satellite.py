"""Satellite feature engineering for training data assembly.

This module computes derived satellite features from Landsat8 and Sentinel2 artifacts:

Landsat8 (bands B1-B11):
- NDVI: Normalized Difference Vegetation Index
- MNDWI: Modified Normalized Difference Water Index
- NDBI: Normalized Difference Built-up Index
- IBI: Index-based Built-up Index
- BSI: Bare Soil Index
- All raw bands preserved

Sentinel2 (bands B2, B4, B8):
- EVI: Enhanced Vegetation Index
- UI_light: Urban Index (simplified)
- SI: Shadow Index
- cloud_mask: from QA10 band
- cloud_fraction_30m: local cloud fraction
- local_ndvi_std: 3×3 spatial standard deviation of NDVI

Temporal matching:
- For a given target date, uses the most recent satellite overpass ≤ target date
- Landsat8 has ~16-day revisit, Sentinel2 has ~5-day revisit

Usage:
    from nalanda.features_satellite import (
        compute_landsat_features,
        compute_sentinel_features,
        find_most_recent_date,
    )
    
    landsat_df = compute_landsat_features(target_date='2021-06-12', grid_ids=my_grid_ids)
    sentinel_df = compute_sentinel_features(target_date='2021-06-12', grid_ids=my_grid_ids)
"""

from __future__ import annotations

from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_LANDSAT_DIR = PROJECT_ROOT / "artifacts" / "v1" / "landsat8"
DEFAULT_SENTINEL_DIR = PROJECT_ROOT / "artifacts" / "v1" / "sentinel2"
DEFAULT_GRID_PATH = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"


# ============================================================================
# Date matching utilities
# ============================================================================

def get_available_dates(artifact_dir: Path, city: str = 'mumbai') -> List[str]:
    """Get sorted list of available dates for a satellite artifact.
    
    Args:
        artifact_dir: Base directory (e.g., artifacts/v1/landsat8)
        city: City name
        
    Returns:
        Sorted list of date strings in YYYY-MM-DD format.
    """
    city_dir = artifact_dir / city
    if not city_dir.exists():
        return []
    
    dates = []
    for f in city_dir.glob("????-??-??.parquet"):
        dates.append(f.stem)
    
    return sorted(dates)


def find_most_recent_date(
    target_date: str,
    available_dates: List[str],
) -> Optional[str]:
    """Find the most recent available date that is <= target_date.
    
    Args:
        target_date: Target date string (YYYY-MM-DD)
        available_dates: List of available date strings
        
    Returns:
        Most recent date <= target_date, or None if no valid date exists.
    """
    target_dt = dt.strptime(target_date, '%Y-%m-%d')
    
    valid_dates = [
        d for d in available_dates
        if dt.strptime(d, '%Y-%m-%d') <= target_dt
    ]
    
    if not valid_dates:
        return None
    
    return max(valid_dates)


# ============================================================================
# Landsat8 feature engineering
# ============================================================================

def load_landsat(
    date: str,
    city: str = 'mumbai',
    landsat_dir: Path = DEFAULT_LANDSAT_DIR,
    grid_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Load Landsat8 parquet for a given date.
    
    Args:
        date: Date string in YYYY-MM-DD format.
        city: City name.
        landsat_dir: Base directory for Landsat artifacts.
        grid_ids: Optional set of grid_ids to filter to.
        
    Returns:
        DataFrame with grid_id and B1-B11 bands.
        
    Raises:
        FileNotFoundError: If parquet file doesn't exist.
    """
    parquet_path = landsat_dir / city / f"{date}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Landsat parquet not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    if grid_ids is not None:
        df = df[df['grid_id'].isin(grid_ids)].copy()
    
    return df


def compute_landsat_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spectral indices from Landsat8 bands.
    
    Band mapping for Landsat8 OLI:
        B1: Coastal/Aerosol (0.43-0.45 μm)
        B2: Blue (0.45-0.51 μm)
        B3: Green (0.53-0.59 μm)
        B4: Red (0.64-0.67 μm)
        B5: NIR (0.85-0.88 μm)
        B6: SWIR1 (1.57-1.65 μm)
        B7: SWIR2 (2.11-2.29 μm)
        B8: Pan (0.50-0.68 μm) - not used for indices
        B9: Cirrus (1.36-1.38 μm)
        B10: TIRS1 (10.6-11.19 μm) - thermal
        B11: TIRS2 (11.50-12.51 μm) - thermal
    
    Args:
        df: DataFrame with B1-B11 columns.
        
    Returns:
        DataFrame with added index columns.
    """
    result = df.copy()
    
    # Avoid division by zero
    eps = 1e-10
    
    # Extract bands as arrays for vectorized computation
    red = df['B4'].values
    nir = df['B5'].values
    green = df['B3'].values
    blue = df['B2'].values
    swir1 = df['B6'].values
    swir2 = df['B7'].values
    
    # NDVI: Normalized Difference Vegetation Index
    # NDVI = (NIR - Red) / (NIR + Red)
    result['ndvi'] = (nir - red) / (nir + red + eps)
    
    # MNDWI: Modified Normalized Difference Water Index
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    result['mndwi'] = (green - swir1) / (green + swir1 + eps)
    
    # NDBI: Normalized Difference Built-up Index
    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    result['ndbi'] = (swir1 - nir) / (swir1 + nir + eps)
    
    # BSI: Bare Soil Index
    # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    bsi_num = (swir1 + red) - (nir + blue)
    bsi_den = (swir1 + red) + (nir + blue) + eps
    result['bsi'] = bsi_num / bsi_den
    
    # IBI: Index-based Built-up Index (simplified version)
    # IBI combines NDBI, SAVI, and MNDWI
    # Simplified: IBI = (2*SWIR1 / (SWIR1 + NIR)) - (NIR / (NIR + Red) + Green / (Green + SWIR1))
    # We'll use a practical approximation: IBI ≈ NDBI - (NDVI + MNDWI) / 2
    result['ibi'] = result['ndbi'] - (result['ndvi'] + result['mndwi']) / 2
    
    return result


def compute_landsat_features(
    target_date: str,
    city: str = 'mumbai',
    landsat_dir: Path = DEFAULT_LANDSAT_DIR,
    grid_ids: Optional[Set[str]] = None,
    include_bands: bool = True,
) -> pd.DataFrame:
    """Compute all Landsat8 features for a target date.
    
    Uses temporal matching: finds most recent Landsat date <= target_date.
    
    Args:
        target_date: Target date (YYYY-MM-DD).
        city: City name.
        landsat_dir: Base directory for Landsat artifacts.
        grid_ids: Optional set of grid_ids to filter to.
        include_bands: Whether to include raw band columns.
        
    Returns:
        DataFrame with columns:
            - grid_id (key)
            - landsat_date (source date used)
            - ndvi, mndwi, ndbi, bsi, ibi (indices)
            - B1-B11 (if include_bands=True)
            
    Raises:
        ValueError: If no Landsat data available for or before target_date.
    """
    # Find available dates
    available = get_available_dates(landsat_dir, city)
    
    if not available:
        raise ValueError(f"No Landsat data found in {landsat_dir / city}")
    
    # Find most recent date
    source_date = find_most_recent_date(target_date, available)
    
    if source_date is None:
        raise ValueError(f"No Landsat data available for or before {target_date}")
    
    # Load and compute
    df = load_landsat(source_date, city, landsat_dir, grid_ids)
    df = compute_landsat_indices(df)
    
    # Add source date column
    df['landsat_date'] = source_date
    
    # Select output columns
    index_cols = ['ndvi', 'mndwi', 'ndbi', 'bsi', 'ibi']
    band_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']
    
    output_cols = ['grid_id', 'landsat_date'] + index_cols
    if include_bands:
        output_cols.extend(band_cols)
    
    # Filter to existing columns (some bands may be missing)
    output_cols = [c for c in output_cols if c in df.columns]
    
    return df[output_cols]


def get_landsat_feature_columns(include_bands: bool = True) -> List[str]:
    """Return list of Landsat feature column names (excluding keys)."""
    cols = ['ndvi', 'mndwi', 'ndbi', 'bsi', 'ibi']
    if include_bands:
        cols.extend(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'])
    return cols


# ============================================================================
# Sentinel2 feature engineering
# ============================================================================

def load_sentinel(
    date: str,
    city: str = 'mumbai',
    sentinel_dir: Path = DEFAULT_SENTINEL_DIR,
    grid_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Load Sentinel2 parquet for a given date.
    
    Args:
        date: Date string in YYYY-MM-DD format.
        city: City name.
        sentinel_dir: Base directory for Sentinel artifacts.
        grid_ids: Optional set of grid_ids to filter to.
        
    Returns:
        DataFrame with grid_id, B2, B4, B8, AOT, QA10.
        
    Raises:
        FileNotFoundError: If parquet file doesn't exist.
    """
    parquet_path = sentinel_dir / city / f"{date}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Sentinel2 parquet not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    if grid_ids is not None:
        df = df[df['grid_id'].isin(grid_ids)].copy()
    
    return df


def compute_sentinel_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spectral indices from Sentinel2 bands.
    
    Band mapping for Sentinel2:
        B2: Blue (490nm)
        B4: Red (665nm)
        B8: NIR (842nm)
        AOT: Aerosol Optical Thickness
        QA10: Quality band (cloud mask info)
    
    Args:
        df: DataFrame with B2, B4, B8, AOT, QA10 columns.
        
    Returns:
        DataFrame with added index columns.
    """
    result = df.copy()
    
    eps = 1e-10
    
    blue = df['B2'].values.astype(np.float64)
    red = df['B4'].values.astype(np.float64)
    nir = df['B8'].values.astype(np.float64)
    
    # NDVI for Sentinel (needed for local_ndvi_std)
    result['ndvi_s2'] = (nir - red) / (nir + red + eps)
    
    # EVI: Enhanced Vegetation Index
    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    # Scale factor for Sentinel2: typically divide by 10000 for reflectance
    # Assuming bands are already in reflectance units
    evi_num = 2.5 * (nir - red)
    evi_den = nir + 6.0 * red - 7.5 * blue + 10000  # +10000 for scaled values
    result['evi'] = evi_num / (evi_den + eps)
    
    # UI_light: Urban Index (simplified)
    # UI = (SWIR - NIR) / (SWIR + NIR) - but we don't have SWIR in Sentinel2 subset
    # Alternative: use (NIR - Red) / (NIR + Red) inverted or NDBI proxy
    # Simplified UI using what we have: UI_light = (Red - Blue) / (Red + Blue)
    result['ui_light'] = (red - blue) / (red + blue + eps)
    
    # SI: Shadow Index
    # SI = sqrt((1 - Blue) * (1 - Green) * (1 - Red))
    # Since we don't have green, use: SI = sqrt((1 - Blue/10000) * (1 - Red/10000))
    # Clamp to avoid negative sqrt
    blue_norm = np.clip(blue / 10000, 0, 1)
    red_norm = np.clip(red / 10000, 0, 1)
    result['si'] = np.sqrt(np.maximum((1 - blue_norm) * (1 - red_norm), 0))
    
    # Cloud mask from QA10
    # QA10 encodes cloud/cirrus info - exact bit meanings depend on processor
    # Simplified: treat non-null QA10 as potential cloud flag
    # For now, create binary cloud_mask (1 = likely cloudy, 0 = clear)
    result['cloud_mask'] = (~df['QA10'].isna()).astype(np.int8)
    
    return result


def compute_local_ndvi_std(
    df: pd.DataFrame,
    grid_path: Path = DEFAULT_GRID_PATH,
    window_size: int = 3,
) -> pd.DataFrame:
    """Compute local (3×3) standard deviation of NDVI.
    
    This captures vegetation heterogeneity around each grid cell.
    
    Args:
        df: DataFrame with grid_id and ndvi_s2 columns.
        grid_path: Path to grid parquet with spatial info.
        window_size: Spatial window size (default 3 for 3×3).
        
    Returns:
        DataFrame with added local_ndvi_std column.
    """
    result = df.copy()
    
    # Load grid to get spatial structure
    grid_df = pd.read_parquet(grid_path, columns=['grid_id', 'row_idx', 'col_idx', 'row_idx_grid'])
    
    # Merge to get spatial indices
    merged = result.merge(grid_df, on='grid_id', how='left')
    
    if 'row_idx_grid' not in merged.columns or 'col_idx' not in merged.columns:
        # Fall back: no spatial info, use NaN
        result['local_ndvi_std'] = np.nan
        return result
    
    # Get grid dimensions
    max_row = merged['row_idx_grid'].max()
    max_col = merged['col_idx'].max()
    
    if pd.isna(max_row) or pd.isna(max_col):
        result['local_ndvi_std'] = np.nan
        return result
    
    # Create 2D NDVI array
    n_rows = int(max_row) + 1
    n_cols = int(max_col) + 1
    
    ndvi_grid = np.full((n_rows, n_cols), np.nan)
    
    # Fill grid with NDVI values
    valid = ~merged['ndvi_s2'].isna() & ~merged['row_idx_grid'].isna() & ~merged['col_idx'].isna()
    rows = merged.loc[valid, 'row_idx_grid'].astype(int).values
    cols = merged.loc[valid, 'col_idx'].astype(int).values
    ndvi_vals = merged.loc[valid, 'ndvi_s2'].values
    ndvi_grid[rows, cols] = ndvi_vals
    
    # Compute local std using a simple approach
    # For each cell, get 3×3 window and compute std
    half = window_size // 2
    local_std = np.full_like(ndvi_grid, np.nan)
    
    for i in range(n_rows):
        for j in range(n_cols):
            r_start = max(0, i - half)
            r_end = min(n_rows, i + half + 1)
            c_start = max(0, j - half)
            c_end = min(n_cols, j + half + 1)
            
            window = ndvi_grid[r_start:r_end, c_start:c_end]
            valid_vals = window[~np.isnan(window)]
            
            if len(valid_vals) >= 2:
                local_std[i, j] = np.std(valid_vals)
    
    # Map back to DataFrame
    std_values = np.full(len(merged), np.nan)
    std_values[valid.values] = local_std[rows, cols]
    result['local_ndvi_std'] = std_values[:len(result)]
    
    return result


def compute_cloud_fraction(
    df: pd.DataFrame,
    grid_path: Path = DEFAULT_GRID_PATH,
    window_size: int = 3,
) -> pd.DataFrame:
    """Compute local cloud fraction in 3×3 window.
    
    Args:
        df: DataFrame with grid_id and cloud_mask columns.
        grid_path: Path to grid parquet.
        window_size: Spatial window size.
        
    Returns:
        DataFrame with added cloud_fraction_30m column.
    """
    result = df.copy()
    
    # Load grid
    grid_df = pd.read_parquet(grid_path, columns=['grid_id', 'row_idx_grid', 'col_idx'])
    
    merged = result.merge(grid_df, on='grid_id', how='left')
    
    if 'row_idx_grid' not in merged.columns:
        result['cloud_fraction_30m'] = np.nan
        return result
    
    max_row = merged['row_idx_grid'].max()
    max_col = merged['col_idx'].max()
    
    if pd.isna(max_row) or pd.isna(max_col):
        result['cloud_fraction_30m'] = np.nan
        return result
    
    n_rows = int(max_row) + 1
    n_cols = int(max_col) + 1
    
    # Create cloud mask grid
    cloud_grid = np.full((n_rows, n_cols), np.nan)
    
    valid = ~merged['cloud_mask'].isna() & ~merged['row_idx_grid'].isna()
    rows = merged.loc[valid, 'row_idx_grid'].astype(int).values
    cols = merged.loc[valid, 'col_idx'].astype(int).values
    cloud_vals = merged.loc[valid, 'cloud_mask'].values
    cloud_grid[rows, cols] = cloud_vals
    
    # Compute local cloud fraction
    half = window_size // 2
    cloud_frac = np.full_like(cloud_grid, np.nan)
    
    for i in range(n_rows):
        for j in range(n_cols):
            r_start = max(0, i - half)
            r_end = min(n_rows, i + half + 1)
            c_start = max(0, j - half)
            c_end = min(n_cols, j + half + 1)
            
            window = cloud_grid[r_start:r_end, c_start:c_end]
            valid_vals = window[~np.isnan(window)]
            
            if len(valid_vals) > 0:
                cloud_frac[i, j] = np.mean(valid_vals)
    
    # Map back
    frac_values = np.full(len(merged), np.nan)
    frac_values[valid.values] = cloud_frac[rows, cols]
    result['cloud_fraction_30m'] = frac_values[:len(result)]
    
    return result


def compute_sentinel_features(
    target_date: str,
    city: str = 'mumbai',
    sentinel_dir: Path = DEFAULT_SENTINEL_DIR,
    grid_ids: Optional[Set[str]] = None,
    grid_path: Path = DEFAULT_GRID_PATH,
    include_spatial: bool = True,
) -> pd.DataFrame:
    """Compute all Sentinel2 features for a target date.
    
    Uses temporal matching: finds most recent Sentinel2 date <= target_date.
    
    Args:
        target_date: Target date (YYYY-MM-DD).
        city: City name.
        sentinel_dir: Base directory for Sentinel artifacts.
        grid_ids: Optional set of grid_ids to filter to.
        grid_path: Path to grid parquet for spatial features.
        include_spatial: Whether to compute spatial features (local_ndvi_std).
        
    Returns:
        DataFrame with columns:
            - grid_id (key)
            - sentinel_date (source date used)
            - evi, ui_light, si, cloud_mask (indices)
            - local_ndvi_std, cloud_fraction_30m (if include_spatial=True)
            
    Raises:
        ValueError: If no Sentinel2 data available for or before target_date.
    """
    available = get_available_dates(sentinel_dir, city)
    
    if not available:
        raise ValueError(f"No Sentinel2 data found in {sentinel_dir / city}")
    
    source_date = find_most_recent_date(target_date, available)
    
    if source_date is None:
        raise ValueError(f"No Sentinel2 data available for or before {target_date}")
    
    # Load and compute indices
    df = load_sentinel(source_date, city, sentinel_dir, grid_ids)
    df = compute_sentinel_indices(df)
    
    # Add spatial features if requested
    if include_spatial and grid_ids is not None and len(grid_ids) <= 10000:
        # Only compute spatial features for small subsets (expensive for full grid)
        df = compute_local_ndvi_std(df, grid_path)
        df = compute_cloud_fraction(df, grid_path)
    else:
        df['local_ndvi_std'] = np.nan
        df['cloud_fraction_30m'] = np.nan
    
    # Add source date
    df['sentinel_date'] = source_date
    
    # Select output columns
    output_cols = [
        'grid_id', 'sentinel_date',
        'evi', 'ui_light', 'si', 'cloud_mask',
        'local_ndvi_std', 'cloud_fraction_30m',
        'ndvi_s2',  # Keep for reference
    ]
    
    output_cols = [c for c in output_cols if c in df.columns]
    
    return df[output_cols]


def get_sentinel_feature_columns() -> List[str]:
    """Return list of Sentinel2 feature column names (excluding keys)."""
    return ['evi', 'ui_light', 'si', 'cloud_mask', 'local_ndvi_std', 'cloud_fraction_30m', 'ndvi_s2']


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    import time
    
    target_date = sys.argv[1] if len(sys.argv) > 1 else '2018-01-10'
    
    print(f"Computing satellite features for target date: {target_date}")
    print("=" * 60)
    
    # Landsat
    print("\n=== LANDSAT8 ===")
    try:
        start = time.time()
        landsat_df = compute_landsat_features(target_date, include_bands=False)
        elapsed = time.time() - start
        print(f"Source date: {landsat_df['landsat_date'].iloc[0]}")
        print(f"Shape: {landsat_df.shape}, Time: {elapsed:.2f}s")
        print(f"Columns: {landsat_df.columns.tolist()}")
        print(f"\nSample stats:")
        for col in ['ndvi', 'mndwi', 'ndbi', 'bsi', 'ibi']:
            if col in landsat_df.columns:
                print(f"  {col}: min={landsat_df[col].min():.3f}, max={landsat_df[col].max():.3f}, mean={landsat_df[col].mean():.3f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Sentinel2
    print("\n=== SENTINEL2 ===")
    try:
        start = time.time()
        sentinel_df = compute_sentinel_features(target_date, include_spatial=False)
        elapsed = time.time() - start
        print(f"Source date: {sentinel_df['sentinel_date'].iloc[0]}")
        print(f"Shape: {sentinel_df.shape}, Time: {elapsed:.2f}s")
        print(f"Columns: {sentinel_df.columns.tolist()}")
        print(f"\nSample stats:")
        for col in ['evi', 'ui_light', 'si', 'ndvi_s2']:
            if col in sentinel_df.columns:
                print(f"  {col}: min={sentinel_df[col].min():.3f}, max={sentinel_df[col].max():.3f}, mean={sentinel_df[col].mean():.3f}")
    except Exception as e:
        print(f"Error: {e}")
