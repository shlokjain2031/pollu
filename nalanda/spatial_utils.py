"""Spatial utilities for Pollu target assembly.

This module provides:
- Coordinate transformation (WGS84 → UTM 43N)
- Grid spatial indexing via KDTree
- Efficient radius queries for sensor influence zones
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


# Mumbai is in UTM Zone 43N
WGS84_EPSG = 4326
UTM43N_EPSG = 32643


def create_wgs84_to_utm_transformer() -> Transformer:
    """Create a reusable transformer from WGS84 to Mumbai UTM (EPSG:32643).
    
    Returns:
        pyproj Transformer for lat/lon → x_utm/y_utm conversion.
    """
    return Transformer.from_crs(
        f"EPSG:{WGS84_EPSG}",
        f"EPSG:{UTM43N_EPSG}",
        always_xy=True,  # lon first, then lat
    )


def transform_coords_to_utm(
    latitudes: np.ndarray | pd.Series,
    longitudes: np.ndarray | pd.Series,
    transformer: Optional[Transformer] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform WGS84 coordinates to UTM 43N.
    
    Args:
        latitudes: Array of WGS84 latitudes.
        longitudes: Array of WGS84 longitudes.
        transformer: Optional pre-created transformer. If None, creates new one.
        
    Returns:
        Tuple of (x_utm, y_utm) arrays in meters.
    """
    if transformer is None:
        transformer = create_wgs84_to_utm_transformer()
    
    # pyproj expects (x, y) = (lon, lat) when always_xy=True
    x_utm, y_utm = transformer.transform(
        np.asarray(longitudes),
        np.asarray(latitudes),
    )
    
    return x_utm, y_utm


def add_utm_coords_to_sensors(
    sensors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add UTM coordinates to sensor DataFrame.
    
    Args:
        sensors_df: Must have 'latitude', 'longitude' columns.
        
    Returns:
        Copy of DataFrame with added 'x_utm', 'y_utm' columns.
    """
    df = sensors_df.copy()
    x_utm, y_utm = transform_coords_to_utm(df['latitude'], df['longitude'])
    df['x_utm'] = x_utm
    df['y_utm'] = y_utm
    return df


class GridIndex:
    """Fast spatial lookup for grid cells using KDTree.
    
    This class provides O(log n) nearest-neighbor queries and O(k) radius
    queries where k is the number of cells within the radius.
    
    Attributes:
        grid: The full grid DataFrame.
        tree: scipy cKDTree built on (x_m, y_m) coordinates.
        x_min: Minimum x coordinate (for bounds checking).
        y_min: Minimum y coordinate (for bounds checking).
        resolution: Grid cell size in meters.
    """
    
    def __init__(self, grid_path: str | Path):
        """Initialize grid index from parquet file.
        
        Args:
            grid_path: Path to grid parquet file. Must have columns:
                row_idx, x_m, y_m.
        """
        self.grid_path = Path(grid_path)
        self.grid = pd.read_parquet(
            grid_path,
            columns=['row_idx', 'x_m', 'y_m'],
        )
        
        # Build KDTree on projected coordinates
        coords = self.grid[['x_m', 'y_m']].values
        self.tree = cKDTree(coords)
        
        # Store grid metadata for bounds checking
        self.x_min = self.grid['x_m'].min()
        self.x_max = self.grid['x_m'].max()
        self.y_min = self.grid['y_m'].min()
        self.y_max = self.grid['y_m'].max()
        self.resolution = 30.0  # meters
        
    def snap_to_nearest(self, x_utm: float, y_utm: float) -> tuple[int, float]:
        """Find the nearest grid cell to a point.
        
        Args:
            x_utm: X coordinate in UTM meters.
            y_utm: Y coordinate in UTM meters.
            
        Returns:
            Tuple of (row_idx, distance_m).
        """
        dist, idx = self.tree.query([x_utm, y_utm], k=1)
        row_idx = int(self.grid.iloc[idx]['row_idx'])
        return row_idx, float(dist)
    
    def query_radius(
        self,
        x_utm: float,
        y_utm: float,
        radius_m: float,
    ) -> pd.DataFrame:
        """Find all grid cells within radius of a point.
        
        Args:
            x_utm: X coordinate in UTM meters.
            y_utm: Y coordinate in UTM meters.
            radius_m: Search radius in meters.
            
        Returns:
            DataFrame with columns:
                - row_idx: int, grid cell identifier
                - distance_m: float, Euclidean distance to query point
        """
        # Query ball returns indices of points within radius
        indices = self.tree.query_ball_point([x_utm, y_utm], r=radius_m)
        
        if not indices:
            return pd.DataFrame(columns=['row_idx', 'distance_m'])
        
        cells = self.grid.iloc[indices][['row_idx', 'x_m', 'y_m']].copy()
        
        # Compute exact Euclidean distance
        dx = cells['x_m'].values - x_utm
        dy = cells['y_m'].values - y_utm
        cells['distance_m'] = np.sqrt(dx**2 + dy**2)
        
        return cells[['row_idx', 'distance_m']].reset_index(drop=True)
    
    def is_within_bounds(self, x_utm: float, y_utm: float, margin_m: float = 500) -> bool:
        """Check if a point is within the grid bounds (with margin).
        
        Args:
            x_utm: X coordinate in UTM meters.
            y_utm: Y coordinate in UTM meters.
            margin_m: Allow this much distance outside grid bounds.
            
        Returns:
            True if point is within bounds.
        """
        x_ok = (self.x_min - margin_m) <= x_utm <= (self.x_max + margin_m)
        y_ok = (self.y_min - margin_m) <= y_utm <= (self.y_max + margin_m)
        return x_ok and y_ok
    
    def __repr__(self) -> str:
        return (
            f"GridIndex(cells={len(self.grid)}, "
            f"x=[{self.x_min:.0f}, {self.x_max:.0f}], "
            f"y=[{self.y_min:.0f}, {self.y_max:.0f}])"
        )


def compute_epanechnikov_trust(distance_m: float | np.ndarray, radius_m: float) -> float | np.ndarray:
    """Compute Epanechnikov kernel trust weight.
    
    τ(d) = max(0, 1 - (d/R)²)
    
    Properties:
        - τ(0) = 1 (full trust at sensor location)
        - τ(R) = 0 (zero trust at boundary)
        - Compact support (exactly 0 beyond R)
        - Flat derivative at d=0 (robust to small position errors)
    
    Args:
        distance_m: Distance in meters (scalar or array).
        radius_m: Influence radius in meters.
        
    Returns:
        Trust weight(s) in [0, 1].
    """
    d_normalized = np.asarray(distance_m) / radius_m
    trust = np.maximum(0.0, 1.0 - d_normalized**2)
    return float(trust) if np.isscalar(distance_m) else trust
