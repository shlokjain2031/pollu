"""Target data assembly for Pollu air quality modeling.

This module handles:
- Precomputing sensor influence zones (grid cells within R of each sensor)
- Building target rows by propagating sensor readings to nearby cells
- Blending overlapping sensor contributions with trust-weighted averaging
- Adding temporal lag columns for previous hours

The core assumption is that PM2.5 readings can be copied (not interpolated)
to nearby grid cells within radius R, with trust decaying by distance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nalanda.sensor_locations import load_sensor_locations
from nalanda.spatial_utils import (
    GridIndex,
    add_utm_coords_to_sensors,
    compute_epanechnikov_trust,
)


# Default configuration
DEFAULT_RADIUS_M = 250.0
DEFAULT_TRUST_KERNEL = "epanechnikov"

# Temporal lag hours
LAG_HOURS = [1, 2, 3, 6, 12, 24]


def precompute_sensor_influence_zones(
    sensors_df: pd.DataFrame,
    grid_index: GridIndex,
    radius_m: float = DEFAULT_RADIUS_M,
) -> dict[str, pd.DataFrame]:
    """Precompute all grid cells within influence zone of each sensor.
    
    This is a one-time computation that maps each sensor to the set of
    grid cells it can influence, along with distance and trust weight.
    
    Args:
        sensors_df: DataFrame with columns: site_id, x_utm, y_utm.
            Must have UTM coordinates already added.
        grid_index: GridIndex instance for spatial queries.
        radius_m: Influence radius in meters.
        
    Returns:
        Dict mapping site_id -> DataFrame with columns:
            - row_idx: int, grid cell identifier
            - distance_m: float, distance to sensor
            - trust_weight: float, Epanechnikov trust in [0, 1]
    """
    influence_zones = {}
    
    for _, sensor in sensors_df.iterrows():
        site_id = sensor['site_id']
        x_utm = sensor['x_utm']
        y_utm = sensor['y_utm']
        
        # Query all cells within radius
        cells = grid_index.query_radius(x_utm, y_utm, radius_m)
        
        if len(cells) == 0:
            # Sensor is outside grid coverage
            continue
        
        # Compute trust weights
        cells['trust_weight'] = compute_epanechnikov_trust(
            cells['distance_m'].values,
            radius_m,
        )
        
        # Store with site_id as key
        influence_zones[site_id] = cells.copy()
    
    return influence_zones


def build_target_for_sensor_timestamp(
    site_id: str,
    timestamp: pd.Timestamp,
    pm25_value: float,
    influence_zone: pd.DataFrame,
) -> pd.DataFrame:
    """Build target rows for a single sensor at a single timestamp.
    
    This propagates the sensor's PM2.5 reading to all grid cells in its
    influence zone, attaching trust weights.
    
    Args:
        site_id: Sensor identifier.
        timestamp: Observation timestamp.
        pm25_value: PM2.5 reading in µg/m³.
        influence_zone: DataFrame from precompute_sensor_influence_zones(),
            with columns: row_idx, distance_m, trust_weight.
            
    Returns:
        DataFrame with columns:
            - row_idx: int
            - timestamp: datetime64[ns]
            - pm25_raw: float (unweighted sensor value)
            - trust_weight: float
            - site_id: str (for debugging/tracing)
    """
    if influence_zone is None or len(influence_zone) == 0:
        return pd.DataFrame(columns=['row_idx', 'timestamp', 'pm25_raw', 'trust_weight', 'site_id'])
    
    result = influence_zone[['row_idx', 'trust_weight']].copy()
    result['timestamp'] = timestamp
    result['pm25_raw'] = pm25_value
    result['site_id'] = site_id
    
    return result[['row_idx', 'timestamp', 'pm25_raw', 'trust_weight', 'site_id']]


def blend_overlapping_sensors(
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    """Blend contributions from multiple sensors at overlapping grid cells.
    
    For each (row_idx, timestamp) that appears multiple times (because
    the cell is within R of multiple sensors), compute:
    - pm25_t: trust-weighted average of sensor readings
    - target_trust: sum of trust weights (can exceed 1.0)
    
    Args:
        contributions: DataFrame with columns:
            row_idx, timestamp, pm25_raw, trust_weight, site_id
            
    Returns:
        DataFrame with one row per (row_idx, timestamp):
            - row_idx: int
            - timestamp: datetime64[ns]
            - pm25_t: float, trust-weighted average PM2.5
            - target_trust: float, sum of contributing trust weights
            - n_sensors: int, number of contributing sensors
    """
    if len(contributions) == 0:
        return pd.DataFrame(columns=['row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors'])
    
    # Compute weighted pm25 contribution
    contributions = contributions.copy()
    contributions['weighted_pm25'] = contributions['pm25_raw'] * contributions['trust_weight']
    
    # Group by (row_idx, timestamp) and aggregate
    grouped = contributions.groupby(['row_idx', 'timestamp'], as_index=False).agg(
        weighted_pm25_sum=('weighted_pm25', 'sum'),
        trust_sum=('trust_weight', 'sum'),
        n_sensors=('site_id', 'nunique'),
    )
    
    # Compute trust-weighted average
    grouped['pm25_t'] = grouped['weighted_pm25_sum'] / grouped['trust_sum']
    grouped['target_trust'] = grouped['trust_sum']
    
    return grouped[['row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors']]


def build_target_for_day(
    aqi_df: pd.DataFrame,
    influence_zones: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build complete target DataFrame for a single day's AQI data.
    
    This is the main entry point for target assembly. It:
    1. Iterates over all (site_id, timestamp, pm25) rows in aqi_df
    2. Propagates each reading to its sensor's influence zone
    3. Blends overlapping contributions
    
    Note: This does NOT add temporal lag columns. Use add_temporal_lags()
    after calling this function to add lag columns.
    
    Args:
        aqi_df: Daily AQI artifact with columns: site_id, timestamp, pm25.
        influence_zones: Precomputed from precompute_sensor_influence_zones().
        
    Returns:
        DataFrame with columns:
            - row_idx: int
            - timestamp: datetime64[ns]
            - pm25_t: float, trust-weighted PM2.5
            - target_trust: float, sum of trust weights
            - n_sensors: int, number of contributing sensors
    """
    all_contributions = []
    
    # Track statistics
    n_readings = 0
    n_readings_with_zone = 0
    
    for _, row in aqi_df.iterrows():
        site_id = str(row['site_id'])
        timestamp = row['timestamp']
        pm25_value = row['pm25']
        
        n_readings += 1
        
        # Skip if sensor not in precomputed zones (outside grid)
        if site_id not in influence_zones:
            continue
        
        # Skip null PM2.5 values
        if pd.isna(pm25_value):
            continue
        
        n_readings_with_zone += 1
        
        # Build contributions for this sensor-timestamp
        contrib = build_target_for_sensor_timestamp(
            site_id=site_id,
            timestamp=timestamp,
            pm25_value=pm25_value,
            influence_zone=influence_zones[site_id],
        )
        
        if len(contrib) > 0:
            all_contributions.append(contrib)
    
    if not all_contributions:
        return pd.DataFrame(columns=['row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors'])
    
    # Concatenate all contributions
    contributions_df = pd.concat(all_contributions, ignore_index=True)
    
    # Blend overlapping sensors
    target_df = blend_overlapping_sensors(contributions_df)
    
    return target_df


def add_temporal_lags(
    target_df: pd.DataFrame,
    prev_day_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add temporal lag columns to target DataFrame.
    
    Adds pm25_t_minus_{1,2,3,6,12,24}h columns by looking back at
    previous timestamps. For the start of the day (e.g., midnight to 6am),
    lags that extend into the previous day use prev_day_df if provided.
    
    The lag operation works as follows:
    1. For each row_idx, sort by timestamp
    2. Shift pm25_t by the appropriate number of hours
    3. For cross-day lags, join with previous day's data
    
    Args:
        target_df: DataFrame with columns: row_idx, timestamp, pm25_t, ...
        prev_day_df: Optional DataFrame from previous day's target.parquet.
            If None, early-day lags that extend into previous day will be NaN.
            
    Returns:
        DataFrame with additional columns:
            - pm25_t_minus_1h: float32
            - pm25_t_minus_2h: float32
            - pm25_t_minus_3h: float32
            - pm25_t_minus_6h: float32
            - pm25_t_minus_12h: float32
            - pm25_t_minus_24h: float32
    """
    if len(target_df) == 0:
        # Add empty lag columns
        for lag_h in LAG_HOURS:
            target_df[f'pm25_t_minus_{lag_h}h'] = pd.Series(dtype='float32')
        return target_df
    
    # Ensure sorted by (row_idx, timestamp) for proper shifting
    df = target_df.sort_values(['row_idx', 'timestamp']).copy()
    
    # Initialize lag columns with NaN
    for lag_h in LAG_HOURS:
        df[f'pm25_t_minus_{lag_h}h'] = np.nan
    
    # Create combined lookup from current day and optional previous day
    current_lookup = df[['row_idx', 'timestamp', 'pm25_t']].copy()
    
    if prev_day_df is not None and len(prev_day_df) > 0:
        prev_lookup = prev_day_df[['row_idx', 'timestamp', 'pm25_t']].copy()
        lookup = pd.concat([prev_lookup, current_lookup], ignore_index=True)
    else:
        lookup = current_lookup
    
    # Create lookup dict: (row_idx, timestamp) -> pm25_t for fast access
    lookup_dict = {}
    for _, row in lookup.iterrows():
        key = (row['row_idx'], row['timestamp'])
        lookup_dict[key] = row['pm25_t']
    
    # Perform lag lookups using vectorized approach where possible
    for lag_h in LAG_HOURS:
        lag_col = f'pm25_t_minus_{lag_h}h'
        lagged_ts = df['timestamp'] - pd.Timedelta(hours=lag_h)
        
        # Lookup values from dict
        lagged_values = []
        for row_idx, ts in zip(df['row_idx'], lagged_ts):
            key = (row_idx, ts)
            lagged_values.append(lookup_dict.get(key, np.nan))
        
        df[lag_col] = pd.array(lagged_values, dtype='float32')
    
    return df


class TargetAssembler:
    """Stateful target assembler with cached sensor zones.
    
    This class provides a convenient interface for target assembly,
    caching the expensive spatial computations.
    
    Attributes:
        sensors_df: Sensor metadata with UTM coordinates.
        grid_index: Spatial index for grid lookups.
        influence_zones: Precomputed sensor influence zones.
        radius_m: Influence radius used for computation.
    """
    
    def __init__(
        self,
        sensor_json_path: str | Path,
        grid_parquet_path: str | Path,
        radius_m: float = DEFAULT_RADIUS_M,
    ):
        """Initialize target assembler.
        
        Args:
            sensor_json_path: Path to openaq_sensor_ids.json.
            grid_parquet_path: Path to grid parquet artifact.
            radius_m: Influence radius in meters.
        """
        self.radius_m = radius_m
        
        # Load and prepare sensors
        self.sensors_df = load_sensor_locations(sensor_json_path)
        self.sensors_df = add_utm_coords_to_sensors(self.sensors_df)
        
        # Build grid index
        self.grid_index = GridIndex(grid_parquet_path)
        
        # Precompute influence zones
        self.influence_zones = precompute_sensor_influence_zones(
            self.sensors_df,
            self.grid_index,
            radius_m,
        )
        
        # Compute statistics
        self.n_sensors = len(self.sensors_df)
        self.n_sensors_in_grid = len(self.influence_zones)
        self.total_influenced_cells = sum(
            len(zone) for zone in self.influence_zones.values()
        )
    
    def assemble_day(
        self,
        aqi_df: pd.DataFrame,
        prev_day_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Assemble target data for a single day.
        
        Args:
            aqi_df: Daily AQI artifact with columns: site_id, timestamp, pm25.
            prev_day_df: Optional target DataFrame from previous day, used
                for computing temporal lags that extend into previous day.
            
        Returns:
            Target DataFrame with columns:
                row_idx, timestamp, pm25_t, target_trust, n_sensors,
                pm25_t_minus_1h, pm25_t_minus_2h, pm25_t_minus_3h,
                pm25_t_minus_6h, pm25_t_minus_12h, pm25_t_minus_24h
        """
        target_df = build_target_for_day(aqi_df, self.influence_zones)
        target_df = add_temporal_lags(target_df, prev_day_df)
        return target_df
    
    def get_metadata(self) -> dict:
        """Get metadata for parquet file storage.
        
        Returns:
            Dict with assembly parameters for reproducibility.
        """
        return {
            'radius_m': str(self.radius_m),
            'trust_kernel': DEFAULT_TRUST_KERNEL,
            'n_sensors': str(self.n_sensors),
            'n_sensors_in_grid': str(self.n_sensors_in_grid),
            'grid_cells': str(len(self.grid_index.grid)),
        }
    
    def __repr__(self) -> str:
        return (
            f"TargetAssembler(R={self.radius_m}m, "
            f"sensors={self.n_sensors_in_grid}/{self.n_sensors}, "
            f"influenced_cells={self.total_influenced_cells:,})"
        )
