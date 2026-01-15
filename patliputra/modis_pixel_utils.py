"""
MODIS AOD pixel mapping utilities.

At training time, use these functions to:
1. Compute MODIS pixel_id for any (lon, lat) coordinate
2. Join 30m grid cells to MODIS AOD values

DESIGN PRINCIPLE:
- MODIS retrieves AOD at ~1km resolution
- We store AOD at native resolution (~924 pixels for Mumbai)
- Each 30m grid cell deterministically maps to its parent 1km pixel
- O(1) coordinate math, no lookup table needed
"""

from typing import Tuple
import numpy as np
import pandas as pd


# Mumbai bounding box (EPSG:4326)
MUMBAI_BBOX = (72.7763, 18.8939, 72.9797, 19.2701)  # west, south, east, north

# MODIS MAIAC pixel size at ~19°N latitude (degrees)
# 1000m / 105,300m (longitude at 19°N) ≈ 0.00949°
# 1000m / 111,000m (latitude) ≈ 0.00901°
MODIS_PIXEL_SIZE_LON = 0.00949  # degrees
MODIS_PIXEL_SIZE_LAT = 0.00901  # degrees


def get_modis_pixel_id(
    lon: float, 
    lat: float, 
    bbox: Tuple[float, float, float, float] = MUMBAI_BBOX
) -> str:
    """
    Deterministically map (lon, lat) to MODIS pixel_id.
    
    O(1) coordinate math - no lookup table needed.
    
    Parameters
    ----------
    lon, lat : float
        Coordinates in EPSG:4326 degrees
    bbox : tuple
        (west, south, east, north) bounding box
    
    Returns
    -------
    str
        Pixel ID in format "{row}_{col}"
    
    Example
    -------
    >>> get_modis_pixel_id(72.85, 19.05)
    '24_7'
    """
    west, south, east, north = bbox
    
    # Column: distance from western edge (floor assignment)
    pixel_col = int((lon - west) / MODIS_PIXEL_SIZE_LON)
    
    # Row: distance from northern edge (image origin is top-left)
    pixel_row = int((north - lat) / MODIS_PIXEL_SIZE_LAT)
    
    return f'{pixel_row}_{pixel_col}'


def get_modis_pixel_id_vectorized(
    lons: np.ndarray,
    lats: np.ndarray,
    bbox: Tuple[float, float, float, float] = MUMBAI_BBOX
) -> np.ndarray:
    """
    Vectorized version of get_modis_pixel_id for arrays.
    
    Parameters
    ----------
    lons, lats : np.ndarray
        Arrays of coordinates in EPSG:4326 degrees
    bbox : tuple
        (west, south, east, north) bounding box
    
    Returns
    -------
    np.ndarray
        Array of pixel IDs as strings
    """
    west, south, east, north = bbox
    
    pixel_cols = ((lons - west) / MODIS_PIXEL_SIZE_LON).astype(int)
    pixel_rows = ((north - lats) / MODIS_PIXEL_SIZE_LAT).astype(int)
    
    # Vectorized string formatting
    return np.char.add(
        np.char.add(pixel_rows.astype(str), '_'),
        pixel_cols.astype(str)
    )


def add_modis_pixel_id_to_dataframe(
    df: pd.DataFrame,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    bbox: Tuple[float, float, float, float] = MUMBAI_BBOX,
    output_col: str = 'modis_pixel_id'
) -> pd.DataFrame:
    """
    Add MODIS pixel_id column to DataFrame with coordinates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with longitude and latitude columns
    lon_col, lat_col : str
        Column names for coordinates
    bbox : tuple
        (west, south, east, north) bounding box
    output_col : str
        Name for output column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added pixel_id column
    
    Example
    -------
    >>> df = pd.DataFrame({'lon': [72.85, 72.86], 'lat': [19.05, 19.06]})
    >>> df = add_modis_pixel_id_to_dataframe(df)
    >>> df['modis_pixel_id']
    0    24_7
    1    23_8
    """
    lons = df[lon_col].values
    lats = df[lat_col].values
    
    df = df.copy()
    df[output_col] = get_modis_pixel_id_vectorized(lons, lats, bbox)
    
    return df


def add_modis_pixel_id_from_geometry(
    gdf,  # GeoDataFrame
    bbox: Tuple[float, float, float, float] = MUMBAI_BBOX,
    output_col: str = 'modis_pixel_id'
):
    """
    Add MODIS pixel_id column to GeoDataFrame with Point geometry.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with Point geometry column
    bbox : tuple
        (west, south, east, north) bounding box
    output_col : str
        Name for output column
    
    Returns
    -------
    GeoDataFrame
        GeoDataFrame with added pixel_id column
    """
    # Extract coordinates from geometry
    lons = gdf.geometry.x.values
    lats = gdf.geometry.y.values
    
    gdf = gdf.copy()
    gdf[output_col] = get_modis_pixel_id_vectorized(lons, lats, bbox)
    
    return gdf


def join_modis_aod(
    grid_df: pd.DataFrame,
    aod_df: pd.DataFrame,
    pixel_id_col: str = 'modis_pixel_id',
    aod_cols: list = None
) -> pd.DataFrame:
    """
    Join MODIS AOD values to grid DataFrame.
    
    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid DataFrame with modis_pixel_id column
    aod_df : pd.DataFrame
        MODIS AOD DataFrame with pixel_id and AOD columns
    pixel_id_col : str
        Column name for pixel ID in grid_df
    aod_cols : list
        AOD columns to join (default: all except pixel_id)
    
    Returns
    -------
    pd.DataFrame
        Grid DataFrame with AOD columns added
    
    Example
    -------
    >>> # At training time:
    >>> grid = pd.read_parquet('artifacts/v1/grids/mumbai/grid.parquet')
    >>> grid = add_modis_pixel_id_from_geometry(grid)
    >>> 
    >>> aod = pd.read_parquet('artifacts/v1/modis_aod/mumbai/2024-01-01.parquet')
    >>> 
    >>> merged = join_modis_aod(grid, aod)
    >>> # Result: 509,209 rows with AOD values
    """
    if aod_cols is None:
        aod_cols = [c for c in aod_df.columns if c not in ['pixel_id', 'pixel_row', 'pixel_col']]
    
    # Ensure pixel_id column exists in both
    join_cols = ['pixel_id'] + aod_cols
    aod_subset = aod_df[join_cols].copy()
    
    # Join on pixel_id
    merged = grid_df.merge(
        aod_subset,
        left_on=pixel_id_col,
        right_on='pixel_id',
        how='left'
    )
    
    # Drop redundant pixel_id column from AOD
    if 'pixel_id' in merged.columns and pixel_id_col != 'pixel_id':
        merged = merged.drop(columns=['pixel_id'])
    
    return merged


# Convenience function for common use case
def load_and_join_modis_aod(
    grid_path: str,
    aod_path: str,
    bbox: Tuple[float, float, float, float] = MUMBAI_BBOX
) -> pd.DataFrame:
    """
    Load grid and MODIS AOD, compute pixel mapping, and join.
    
    One-liner for training data preparation.
    
    Parameters
    ----------
    grid_path : str
        Path to grid parquet
    aod_path : str
        Path to MODIS AOD parquet for specific date
    bbox : tuple
        Bounding box
    
    Returns
    -------
    pd.DataFrame
        Grid with AOD columns
    
    Example
    -------
    >>> df = load_and_join_modis_aod(
    ...     'artifacts/v1/grids/mumbai/grid.parquet',
    ...     'artifacts/v1/modis_aod/mumbai/2024-01-01.parquet'
    ... )
    """
    import geopandas as gpd
    
    # Load grid
    grid = gpd.read_parquet(grid_path)
    
    # Ensure WGS84 for coordinate extraction
    if grid.crs and grid.crs.to_string() != "EPSG:4326":
        grid = grid.to_crs("EPSG:4326")
    
    # Add pixel mapping
    grid = add_modis_pixel_id_from_geometry(grid, bbox)
    
    # Load AOD
    aod = pd.read_parquet(aod_path)
    
    # Join
    merged = join_modis_aod(grid, aod)
    
    return merged
