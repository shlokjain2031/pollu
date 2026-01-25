"""Feature assembly with on-the-fly derived feature computation.

Fetches raw observational data from DuckDB tables and computes all 64 required
derived features for inference.

Raw data sources:
- pm25_observation: Ground-truth PM2.5 observations
- meteo: Meteorological observations  
- landsat8: Landsat8 spectral bands (B1-B11)
- sentinel2: Sentinel2 spectral bands (B2, B4, B8, AOT, QA10)
- grid: Static grid reference data

Derived features computed on-the-fly:
- PM2.5 lags (6): t-1h, t-2h, t-3h, t-6h, t-12h, t-24h
- Meteo lags (44): temperature, humidity, dewpoint, pressure, wind, cloud cover at various intervals
- Satellite indices (13): NDVI, MNDWI, NDBI, BSI, IBI, EVI, UI, SI, etc.
- Spatial (2): latitude, longitude
"""

import duckdb
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .errors import FeatureAssemblyError, InvalidRequestError, PersistenceError
from .config import (
    TABLE_GRID, TABLE_PM25_OBSERVATION, TABLE_METEO,
    TABLE_LANDSAT8, TABLE_SENTINEL2,
    DUCKDB_GRID_PATH, DUCKDB_PM25_PATH, DUCKDB_METEO_PATH,
    DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH, get_db_path_for_table
)

logger = logging.getLogger(__name__)

# Import pool manager if available (optional for backward compatibility)
try:
    from takshashila.core.db_pool import PoolManager
    POOL_AVAILABLE = True
except ImportError:
    PoolManager = None
    POOL_AVAILABLE = False
    logger.warning("Connection pooling not available (kashi.db_pool not found)")

# Freshness thresholds (in seconds)
FRESHNESS_THRESHOLDS = {
    "pm25_observation": 3600,      # 1 hour
    "meteo": 3600,                 # 1 hour
    "landsat8": 1382400,           # 16 days
    "sentinel2": 432000,           # 5 days
}


class FeatureAssembler:
    """Assemble all 64 required features by computing from raw observational data.
    
    Reads from separate DuckDB files per table:
    - grid.duckdb: grid reference data
    - pm25_observations.duckdb: PM2.5 observations
    - meteo.duckdb: meteorological data
    - landsat8.duckdb: Landsat8 satellite imagery
    - sentinel2.duckdb: Sentinel2 satellite imagery
    
    Validates:
    - grid_id exists in grid table
    - All required raw data is available
    - Data is fresh (within freshness thresholds)
    - Derived features computed correctly
    """
    
    def __init__(self, db_paths: Dict[str, str], feature_schema: Dict, pool_manager=None):
        """Initialize feature assembler with paths to all DuckDB files.
        
        Args:
            db_paths: Dict mapping table names to file paths:
                {
                    "grid": "/path/to/grid.duckdb",
                    "pm25_observation": "/path/to/pm25_observations.duckdb",
                    "meteo": "/path/to/meteo.duckdb",
                    "landsat8": "/path/to/landsat8.duckdb",
                    "sentinel2": "/path/to/sentinel2.duckdb"
                }
            feature_schema: Loaded feature_schema.json dict
            pool_manager: Optional PoolManager for connection pooling (from kashi.db_pool)
        """
        self.db_paths = {k: Path(v) for k, v in db_paths.items()}
        self.required_features = feature_schema["feature_list"]
        self.n_features = len(self.required_features)
        self.pool_manager = pool_manager
        
        # Validate all required tables have paths
        required_tables = [TABLE_GRID, TABLE_PM25_OBSERVATION, TABLE_METEO, TABLE_LANDSAT8, TABLE_SENTINEL2]
        for table in required_tables:
            if table not in self.db_paths:
                raise ValueError(f"Missing db_path for table: {table}")
        
        pool_status = "with connection pooling" if pool_manager else "without connection pooling"
        logger.info(f"FeatureAssembler initialized {pool_status} with {self.n_features} features across {len(self.db_paths)} databases")
    
    def _get_connection(self, table_name: str) -> duckdb.DuckDBPyConnection:
        """Get a DuckDB connection for reading a specific table.
        
        If pool_manager is available, acquires from pool (reused connections).
        Otherwise, creates a new read-only connection (legacy mode).
        
        Args:
            table_name: Name of table (e.g., 'grid', 'pm25_observation', 'meteo')
        
        Returns:
            DuckDB connection to the file containing that table
        """
        if table_name not in self.db_paths:
            raise ValueError(f"No db_path configured for table: {table_name}")
        
        # Use pooled connection if available
        if self.pool_manager is not None:
            try:
                pool = self.pool_manager.get_pool(table_name)
                return pool.acquire()
            except Exception as e:
                logger.warning(f"Failed to acquire pooled connection for {table_name}, falling back to direct connection: {e}")
        
        # Fallback: create new connection (legacy mode)
        db_path = self.db_paths[table_name]
        return duckdb.connect(str(db_path), read_only=True)
    
    def _release_connection(self, table_name: str, conn: duckdb.DuckDBPyConnection) -> None:
        """Release a connection back to the pool or close it.
        
        Args:
            table_name: Name of table the connection was for
            conn: Connection to release
        """
        if self.pool_manager is not None:
            try:
                pool = self.pool_manager.get_pool(table_name)
                pool.release(conn)
            except Exception as e:
                logger.warning(f"Failed to release pooled connection for {table_name}: {e}")
                conn.close()
        else:
            # Legacy mode: just close the connection
            conn.close()
    
    def fetch_features(self, grid_id: str, timestamp_hour: datetime) -> pd.DataFrame:
        """Fetch and compute all 64 features for (grid_id, timestamp_hour).
        
        Args:
            grid_id: Grid cell identifier (e.g., "g00000001")
            timestamp_hour: Prediction timestamp (must be timezone-naive, hour-floored)
        
        Returns:
            Single-row DataFrame with 64 feature columns in correct order
        
        Raises:
            InvalidRequestError: If grid_id not in grid table
            FeatureAssemblyError: If any required raw data is missing or stale
        
        Note:
            timestamp_hour must be timezone-naive to match DuckDB TIMESTAMP storage.
            Use round_to_hour() to normalize input timestamps.
        """
        # Ensure timezone-naive (should already be normalized by round_to_hour)
        if timestamp_hour.tzinfo is not None:
            logger.warning(f"fetch_features received timezone-aware timestamp: {timestamp_hour}, stripping timezone")
            timestamp_hour = timestamp_hour.replace(tzinfo=None)
        
        logger.debug(f"fetch_features: grid_id={grid_id}, timestamp_hour={timestamp_hour} (type: {type(timestamp_hour)}, tzinfo: {timestamp_hour.tzinfo})")
        
        # Validate grid exists (opens grid.duckdb)
        conn_grid = self._get_connection(TABLE_GRID)
        try:
            grid_exists = conn_grid.execute(
                f"SELECT COUNT(*) FROM {TABLE_GRID} WHERE grid_id = ?",
                [grid_id]
            ).fetchone()[0] > 0
            
            if not grid_exists:
                raise InvalidRequestError(f"grid_id '{grid_id}' not found in grid table")
            
            # Fetch spatial features while we have grid connection open
            spatial_features = self._fetch_spatial_features(conn_grid, grid_id)
        finally:
            self._release_connection(TABLE_GRID, conn_grid)
        
        # Compute PM2.5 lag features (opens pm25_observations.duckdb)
        pm25_features = self._compute_pm25_lags(grid_id, timestamp_hour)
        
        # Compute meteorological features (opens meteo.duckdb)
        meteo_features = self._compute_meteo_features(grid_id, timestamp_hour)
        
        # Compute satellite features (opens landsat8.duckdb and sentinel2.duckdb)
        sat_features = self._compute_satellite_features(grid_id, timestamp_hour)
        
        # Merge all features
        all_features = {**pm25_features, **meteo_features, **sat_features, **spatial_features}
        
        # Validate completeness
        self._validate_features(all_features, grid_id, timestamp_hour)
        
        # Build DataFrame in correct column order
        features_df = pd.DataFrame([all_features])
        features_df = features_df[self.required_features]  # Reorder columns
        
        logger.debug(f"Assembled {len(all_features)} features for {grid_id} @ {timestamp_hour}")
        
        return features_df
    
    def fetch_features_batch(self, grid_ids: List[str], timestamp_hour: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch and compute all 64 features for multiple grids at once (batch mode).
        
        Optimized version that amortizes DuckDB queries across multiple grids.
        Instead of N separate queries to each table, performs single batch queries.
        
        Args:
            grid_ids: List of grid cell identifiers (e.g., ["g00000001", "g00000002", ...])
            timestamp_hour: Prediction timestamp (must be timezone-naive, hour-floored)
        
        Returns:
            Dict mapping grid_id → single-row DataFrame with 64 feature columns in correct order
            Only includes grids for which all features were successfully assembled.
            Grids with missing data are logged as warnings and excluded from result.
        
        Raises:
            FeatureAssemblyError: If batch assembly fails catastrophically
        
        Performance:
            - Single query per table (vs N queries per grid)
            - ~10x faster than sequential fetch_features() calls
            - Memory: ~100 grids × 1KB features = 1MB
        
        Examples:
            >>> features = engine.fetch_features_batch(
            ...     ["g00000001", "g00000002", "g00000003"],
            ...     datetime(2026, 1, 19, 16, 47)
            ... )
            >>> print(features["g00000001"].shape)  # (1, 64)
        """
        # Ensure timezone-naive
        if timestamp_hour.tzinfo is not None:
            logger.warning(f"fetch_features_batch received timezone-aware timestamp: {timestamp_hour}, stripping timezone")
            timestamp_hour = timestamp_hour.replace(tzinfo=None)
        
        logger.info(f"Batch fetch_features: {len(grid_ids)} grids @ {timestamp_hour}")
        
        result = {}
        
        # Validate all grids exist (single batch query to grid table)
        conn_grid = self._get_connection(TABLE_GRID)
        try:
            # Check which grids exist
            placeholders = ",".join(["?" for _ in grid_ids])
            grid_check = conn_grid.execute(
                f"SELECT grid_id FROM {TABLE_GRID} WHERE grid_id IN ({placeholders})",
                grid_ids
            ).fetchall()
            
            valid_grids = {row[0] for row in grid_check}
            invalid_grids = set(grid_ids) - valid_grids
            
            if invalid_grids:
                logger.warning(f"Skipping {len(invalid_grids)} invalid grid_ids: {invalid_grids}")
            
            grids_to_compute = list(valid_grids)
        finally:
            self._release_connection(TABLE_GRID, conn_grid)
        
        # If no valid grids, return empty dict
        if not grids_to_compute:
            logger.warning(f"No valid grids in batch of {len(grid_ids)}")
            return {}
        
        # OPTIMIZATION: Parallel batch queries with one thread per table
        # This reduces database round-trips from 35-40 per grid to ~4 batch queries total
        # Expected 10-12x speedup for large batches
        # Threads: PM2.5 table, Meteo table, Satellite tables (parallel)
        
        try:
            # Spatial features (fast, do first in main thread)
            conn_grid = self._get_connection(TABLE_GRID)
            try:
                placeholders = ",".join(["?" for _ in grids_to_compute])
                spatial_rows = conn_grid.execute(
                    f"SELECT grid_id, lat, long FROM {TABLE_GRID} WHERE grid_id IN ({placeholders})",
                    grids_to_compute
                ).fetchall()
                spatial_dict = {row[0]: {
                    'latitude_deg': row[1], 'longitude_deg': row[2]
                } for row in spatial_rows}
            finally:
                self._release_connection(TABLE_GRID, conn_grid)
            
            # Parallel feature computation: one thread per table source
            # Thread 1: PM2.5 observations
            # Thread 2: Meteorological data
            # Thread 3: Satellite data (Landsat8 + Sentinel2)
            # Configurable via env: KASHI_FEATURE_THREADS (default: 3)
            max_workers = int(os.environ.get("KASHI_FEATURE_THREADS", "3"))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit parallel tasks
                pm25_future = executor.submit(self._compute_pm25_lags_batch, grids_to_compute, timestamp_hour)
                meteo_future = executor.submit(self._compute_meteo_features_batch, grids_to_compute, timestamp_hour)
                sat_future = executor.submit(self._compute_satellite_features_batch, grids_to_compute, timestamp_hour)
                
                # Wait for all to complete and collect results
                pm25_features_dict = pm25_future.result()
                meteo_features_dict = meteo_future.result()
                sat_features_dict = sat_future.result()
            
            # Merge all features for each grid
            for grid_id in grids_to_compute:
                try:
                    all_features = {
                        **spatial_dict.get(grid_id, {}),
                        **pm25_features_dict.get(grid_id, {}),
                        **meteo_features_dict.get(grid_id, {}),
                        **sat_features_dict.get(grid_id, {})
                    }
                    
                    # Validate completeness
                    if len(all_features) == 0:
                        logger.warning(f"No features assembled for {grid_id}")
                        continue
                    
                    # Build DataFrame in correct column order
                    features_df = pd.DataFrame([all_features])
                    features_df = features_df[self.required_features]  # Reorder columns
                    result[grid_id] = features_df
                except Exception as e:
                    logger.warning(f"Failed to assemble features for {grid_id}: {e}")
                    continue
            
            logger.info(f"Batch fetch_features completed: {len(result)}/{len(grid_ids)} grids successful (parallel batch queries: 3 threads)")
        
        except Exception as e:
            logger.error(f"Batch feature assembly failed: {e}")
            # Fallback to sequential fetch_features for robustness
            logger.info("Falling back to sequential feature assembly...")
            for grid_id in grids_to_compute:
                try:
                    features_df = self.fetch_features(grid_id, timestamp_hour)
                    result[grid_id] = features_df
                except Exception as e:
                    logger.warning(f"Failed to fetch features for {grid_id}: {e}")
                    continue
            logger.info(f"Batch fetch_features completed: {len(result)}/{len(grid_ids)} grids successful (sequential fallback)")
        
        return result
    
    # =========================================================================
    # PM2.5 Feature Computation
    # =========================================================================
    
    def _compute_pm25_lags(
        self,
        grid_id: str,
        timestamp_hour: datetime
    ) -> Dict[str, float]:
        """Compute PM2.5 lag features from observations.
        
        Fetches PM2.5 observations at t-1h, t-2h, t-3h, t-6h, t-12h, t-24h
        and validates freshness.
        
        Returns:
            Dict with keys: pm25_t_minus_1h, pm25_t_minus_2h, etc.
        """
        lags = {
            "pm25_t_minus_1h": 1,
            "pm25_t_minus_2h": 2,
            "pm25_t_minus_3h": 3,
            "pm25_t_minus_6h": 6,
            "pm25_t_minus_12h": 12,
            "pm25_t_minus_24h": 24,
        }
        
        features = {}
        
        conn = self._get_connection(TABLE_PM25_OBSERVATION)
        try:
            for feature_name, hours_back in lags.items():
                target_ts = timestamp_hour - timedelta(hours=hours_back)
                
                result = conn.execute(f"""
                    SELECT pm25, timestamp
                    FROM {TABLE_PM25_OBSERVATION}
                    WHERE grid_id = ? AND timestamp = ?
                """, [grid_id, target_ts]).fetchone()
                
                if result is None:
                    raise FeatureAssemblyError(
                        grid_id=grid_id,
                        timestamp_hour=timestamp_hour,
                        missing_feature=f"pm25_observation at {target_ts}"
                    )
                
                pm25_value, obs_timestamp = result
                
                # Verify we got the exact timestamp we requested
                # (For lag features, the observation SHOULD be old - that's the point!)
                if obs_timestamp != target_ts:
                    logger.warning(
                        f"PM2.5 {feature_name}: Expected {target_ts}, got {obs_timestamp}"
                    )
                    raise FeatureAssemblyError(
                        grid_id=grid_id,
                        timestamp_hour=timestamp_hour,
                        missing_feature=f"pm25_observation timestamp mismatch: expected {target_ts}, got {obs_timestamp}"
                    )
                
                logger.debug(f"PM2.5 {feature_name}: value={pm25_value:.2f} at {obs_timestamp}")
                features[feature_name] = pm25_value
        finally:
            self._release_connection(TABLE_PM25_OBSERVATION, conn)
        
        return features
    
    # =========================================================================
    # BATCH FEATURE COMPUTATION (Parallel, one thread per table)
    # =========================================================================
    
    def _compute_pm25_lags_batch(
        self,
        grid_ids: List[str],
        timestamp_hour: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Batch PM2.5 lag computation for multiple grids.
        
        Fetches PM2.5 observations for all grids at once (single batch query per lag).
        Much faster than sequential fetch_features() calls.
        
        Returns:
            Dict mapping grid_id → Dict[feature_name, value]
        """
        conn = self._get_connection(TABLE_PM25_OBSERVATION)
        try:
            result = {}
            
            lags = {
                "pm25_t_minus_1h": 1,
                "pm25_t_minus_2h": 2,
                "pm25_t_minus_3h": 3,
                "pm25_t_minus_6h": 6,
                "pm25_t_minus_12h": 12,
                "pm25_t_minus_24h": 24,
            }
            
            # Initialize grid features
            for grid_id in grid_ids:
                result[grid_id] = {}
            
            # Batch query for each lag
            placeholders = ",".join(["?" for _ in grid_ids])
            for feature_name, hours_back in lags.items():
                target_ts = timestamp_hour - timedelta(hours=hours_back)
                
                rows = conn.execute(f"""
                    SELECT grid_id, pm25
                    FROM {TABLE_PM25_OBSERVATION}
                    WHERE grid_id IN ({placeholders}) AND timestamp = ?
                """, grid_ids + [target_ts]).fetchall()
                
                for grid_id, pm25_value in rows:
                    result[grid_id][feature_name] = pm25_value
                
                # Log grids missing this lag
                found_grids = {row[0] for row in rows}
                missing_grids = set(grid_ids) - found_grids
                if missing_grids and len(missing_grids) < 5:
                    logger.debug(f"Missing PM2.5 {feature_name} for: {missing_grids}")
            
            return result
        finally:
            self._release_connection(TABLE_PM25_OBSERVATION, conn)
    
    def _compute_meteo_features_batch(
        self,
        grid_ids: List[str],
        timestamp_hour: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Batch meteorological feature computation for multiple grids.
        
        Fetches meteo observations for all grids at once using batch queries.
        Much faster than sequential fetch_features() calls.
        
        Uses actual DuckDB column names: temperature_2m, relative_humidity_2m, etc.
        
        Returns:
            Dict mapping grid_id → Dict[feature_name, value]
        """
        conn = self._get_connection(TABLE_METEO)
        try:
            result = {}
            
            # Initialize grid features
            for grid_id in grid_ids:
                result[grid_id] = {}
            
            # Define what we need to fetch (matches _compute_meteo_features)
            # We'll fetch all lags and then assemble like the sequential version does
            placeholders = ",".join(["?" for _ in grid_ids])
            
            # Define lags for each variable (matches sequential version)
            meteo_lags = {
                "temperature_2m": [0, 1, 3, 12, 24],
                "relative_humidity_2m": [0, 1, 3, 6, 12, 24],
                "dewpoint_2m": [0, 1, 3, 6, 12, 24],
                "surface_pressure": [0, 1, 3, 6, 12, 24],
                "cloud_cover": [0, 6, 24],
                "wind_speed_10m": [0, 1, 3, 6, 12, 24],
                "wind_direction_10m": [0, 1, 3, 6, 12, 24],
            }
            
            # Fetch all observations in batch (one query per variable/lag combination)
            observations = {}  # var_name → lag → {grid_id → value}
            
            for var_name, lag_hours in meteo_lags.items():
                observations[var_name] = {}
                
                for lag in lag_hours:
                    target_ts = timestamp_hour - timedelta(hours=lag)
                    observations[var_name][lag] = {}
                    
                    # Batch query for this variable at this lag
                    rows = conn.execute(f"""
                        SELECT grid_id, {var_name}
                        FROM {TABLE_METEO}
                        WHERE grid_id IN ({placeholders}) AND timestamp = ?
                    """, grid_ids + [target_ts]).fetchall()
                    
                    for grid_id, value in rows:
                        observations[var_name][lag][grid_id] = value
            
            # Assemble lagged features for each grid (matches sequential version)
            for grid_id in grid_ids:
                for var_name, lag_hours in meteo_lags.items():
                    for lag in lag_hours:
                        # Skip wind variables - we'll convert to U/V components
                        if var_name in ["wind_speed_10m", "wind_direction_10m"]:
                            continue
                        
                        # Get value, skip if not found
                        if grid_id not in observations[var_name][lag]:
                            continue
                        
                        if lag == 0:
                            feature_name = var_name
                        else:
                            feature_name = f"{var_name}_lag{lag}h"
                        
                        result[grid_id][feature_name] = observations[var_name][lag][grid_id]
                
                # Convert wind speed/direction to U/V components (matches sequential version)
                for lag in meteo_lags["wind_speed_10m"]:
                    if grid_id not in observations["wind_speed_10m"][lag]:
                        continue
                    if grid_id not in observations["wind_direction_10m"][lag]:
                        continue
                    
                    speed = observations["wind_speed_10m"][lag][grid_id]
                    direction = observations["wind_direction_10m"][lag][grid_id]
                    
                    # Convert direction from degrees to radians
                    direction_rad = np.deg2rad(direction)
                    
                    # Compute U and V components
                    u_component = -speed * np.sin(direction_rad)
                    v_component = -speed * np.cos(direction_rad)
                    
                    # Add to features
                    if lag == 0:
                        result[grid_id]["wind_u_10m"] = u_component
                        result[grid_id]["wind_v_10m"] = v_component
                    else:
                        result[grid_id][f"wind_u_10m_lag{lag}h"] = u_component
                        result[grid_id][f"wind_v_10m_lag{lag}h"] = v_component
                
                # Compute derived meteorological features (vpd, temp_dewpoint_diff)
                if grid_id in observations["temperature_2m"][0] and \
                   grid_id in observations["dewpoint_2m"][0] and \
                   grid_id in observations["relative_humidity_2m"][0]:
                    
                    temp_0 = observations["temperature_2m"][0][grid_id]
                    dewpoint_0 = observations["dewpoint_2m"][0][grid_id]
                    rh_0 = observations["relative_humidity_2m"][0][grid_id]
                    
                    # VPD: Vapor Pressure Deficit
                    saturation_vp = 6.112 * np.exp((17.67 * temp_0) / (temp_0 + 243.5))
                    actual_vp = (rh_0 / 100.0) * saturation_vp
                    vpd = saturation_vp - actual_vp
                    result[grid_id]["vpd"] = vpd
                    
                    # Temp-Dewpoint Difference
                    result[grid_id]["temp_dewpoint_diff"] = temp_0 - dewpoint_0
            
            # Compute precipitation lags (cumulative) - batch query
            precip_dict = self._compute_precipitation_lags_batch(grid_ids, timestamp_hour)
            for grid_id in grid_ids:
                if grid_id in precip_dict:
                    result[grid_id].update(precip_dict[grid_id])
            
            return result
        finally:
            self._release_connection(TABLE_METEO, conn)
    
    def _compute_precipitation_lags_batch(
        self,
        grid_ids: List[str],
        timestamp_hour: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Batch computation of cumulative precipitation over 1h, 6h, 24h windows.
        
        Returns:
            Dict mapping grid_id → Dict[feature_name, cumulative_precip]
        """
        result = {}
        
        # Initialize grid features
        for grid_id in grid_ids:
            result[grid_id] = {}
        
        windows = {
            "precip_last_1h": 1,
            "precip_last_6h": 6,
            "precip_last_24h": 24,
        }
        
        conn = self._get_connection(TABLE_METEO)
        try:
            placeholders = ",".join(["?" for _ in grid_ids])
            
            for feature_name, hours in windows.items():
                start_ts = timestamp_hour - timedelta(hours=hours)
                
                # Batch query: sum precipitation for all grids in time window
                rows = conn.execute(f"""
                    SELECT grid_id, SUM(precipitation) as total_precip
                    FROM {TABLE_METEO}
                    WHERE grid_id IN ({placeholders}) 
                      AND timestamp > ? 
                      AND timestamp <= ?
                    GROUP BY grid_id
                """, grid_ids + [start_ts, timestamp_hour]).fetchall()
                
                for grid_id, total_precip in rows:
                    result[grid_id][feature_name] = total_precip if total_precip is not None else 0.0
                
                # Fill missing grids with 0.0
                for grid_id in grid_ids:
                    if feature_name not in result[grid_id]:
                        result[grid_id][feature_name] = 0.0
        finally:
            self._release_connection(TABLE_METEO, conn)
        
        return result
    
    def _compute_satellite_features_batch(
        self,
        grid_ids: List[str],
        timestamp_hour: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Batch satellite feature computation for multiple grids.
        
        Fetches raw Landsat8 and Sentinel2 data for all grids at once,
        then computes spectral indices in code.
        
        Returns:
            Dict mapping grid_id → Dict[feature_name, value]
        """
        result = {}
        
        # Initialize grid features
        for grid_id in grid_ids:
            result[grid_id] = {}
        
        placeholders = ",".join(["?" for _ in grid_ids])
        
        # Fetch most recent Landsat8 data for all grids
        conn_landsat = self._get_connection(TABLE_LANDSAT8)
        try:
            # Get most recent observation per grid (within 30 days)
            landsat_results = conn_landsat.execute(f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY grid_id ORDER BY timestamp DESC) as rn
                    FROM {TABLE_LANDSAT8}
                    WHERE grid_id IN ({placeholders}) AND timestamp <= ?
                ) WHERE rn = 1
            """, grid_ids + [timestamp_hour]).fetchall()
            
            # Get column names for proper mapping
            col_names = [col[0] for col in conn_landsat.execute(f"DESCRIBE {TABLE_LANDSAT8}").fetchall()]
            
            landsat_by_grid = {}
            for row in landsat_results:
                row_dict = {col_names[i]: row[i] for i in range(len(col_names))}
                grid_id = row_dict.pop("grid_id")
                landsat_ts = row_dict.pop("timestamp")
                row_dict.pop("created_at", None)
                
                # Check freshness (16-day threshold)
                days_since = (timestamp_hour - landsat_ts).days
                if days_since <= 16:
                    landsat_by_grid[grid_id] = (row_dict, landsat_ts)
        finally:
            self._release_connection(TABLE_LANDSAT8, conn_landsat)
        
        # Fetch most recent Sentinel2 data for all grids
        conn_sentinel = self._get_connection(TABLE_SENTINEL2)
        try:
            # Get most recent observation per grid (within 30 days)
            sentinel_results = conn_sentinel.execute(f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY grid_id ORDER BY timestamp DESC) as rn
                    FROM {TABLE_SENTINEL2}
                    WHERE grid_id IN ({placeholders}) AND timestamp <= ?
                ) WHERE rn = 1
            """, grid_ids + [timestamp_hour]).fetchall()
            
            # Get column names for proper mapping
            col_names = [col[0] for col in conn_sentinel.execute(f"DESCRIBE {TABLE_SENTINEL2}").fetchall()]
            
            sentinel_by_grid = {}
            for row in sentinel_results:
                row_dict = {col_names[i]: row[i] for i in range(len(col_names))}
                grid_id = row_dict.pop("grid_id")
                sentinel_ts = row_dict.pop("timestamp")
                row_dict.pop("created_at", None)
                
                # Check freshness (5-day threshold)
                days_since = (timestamp_hour - sentinel_ts).days
                if days_since <= 5:
                    sentinel_by_grid[grid_id] = (row_dict, sentinel_ts)
        finally:
            self._release_connection(TABLE_SENTINEL2, conn_sentinel)
        
        # Compute indices for grids that have data
        for grid_id in grid_ids:
            # Landsat indices
            if grid_id in landsat_by_grid:
                landsat_data, landsat_ts = landsat_by_grid[grid_id]
                landsat_indices = self._compute_landsat_indices(landsat_data)
                result[grid_id].update(landsat_indices)
                days_since = (timestamp_hour - landsat_ts).days
                result[grid_id]["days_since_landsat"] = float(days_since)
            
            # Sentinel indices
            if grid_id in sentinel_by_grid:
                sentinel_data, sentinel_ts = sentinel_by_grid[grid_id]
                sentinel_indices = self._compute_sentinel_indices(sentinel_data)
                result[grid_id].update(sentinel_indices)
                days_since = (timestamp_hour - sentinel_ts).days
                result[grid_id]["days_since_sentinel"] = float(days_since)
        

        return result
    
    # =========================================================================
    # Meteorological Feature Computation
    # =========================================================================
    
    def _compute_meteo_features(
        self,
        grid_id: str,
        timestamp_hour: datetime
    ) -> Dict[str, float]:
        """Compute all 44 meteorological features.
        
        Fetches raw observations at specified lags and computes:
        - Lagged values for temperature, humidity, pressure, wind, cloud cover
        - Derived: VPD, temp-dewpoint diff, precipitation lags
        """
        features = {}
        
        # Define lags for each variable (in hours)
        meteo_lags = {
            "temperature_2m": [0, 1, 3, 12, 24],
            "relative_humidity_2m": [0, 1, 3, 6, 12, 24],
            "dewpoint_2m": [0, 1, 3, 6, 12, 24],
            "surface_pressure": [0, 1, 3, 6, 12, 24],
            "cloud_cover": [0, 6, 24],
            "wind_speed_10m": [0, 1, 3, 6, 12, 24],  # Will convert to U/V components
            "wind_direction_10m": [0, 1, 3, 6, 12, 24],  # Will convert to U/V components
        }
        
        conn = self._get_connection(TABLE_METEO)
        try:
            # Fetch all required observations
            observations = {}
            for var_name, lag_hours in meteo_lags.items():
                observations[var_name] = {}
                
                for lag in lag_hours:
                    target_ts = timestamp_hour - timedelta(hours=lag)
                    
                    result = conn.execute(f"""
                        SELECT {var_name}, timestamp
                        FROM {TABLE_METEO}
                        WHERE grid_id = ? AND timestamp = ?
                    """, [grid_id, target_ts]).fetchone()
                    
                    if result is None:
                        raise FeatureAssemblyError(
                            grid_id=grid_id,
                            timestamp_hour=timestamp_hour,
                            missing_feature=f"meteo.{var_name} at {target_ts}"
                        )
                    
                    value, obs_timestamp = result
                    
                    # Verify we got the exact timestamp we requested
                    if obs_timestamp != target_ts:
                        logger.warning(
                            f"Meteo {var_name} lag{lag}h: Expected {target_ts}, got {obs_timestamp}"
                        )
                        raise FeatureAssemblyError(
                            grid_id=grid_id,
                            timestamp_hour=timestamp_hour,
                            missing_feature=f"meteo.{var_name} timestamp mismatch: expected {target_ts}, got {obs_timestamp}"
                        )
                    
                    observations[var_name][lag] = value
        finally:
            self._release_connection(TABLE_METEO, conn)
        
        # Assemble lagged features
        for var_name, lag_hours in meteo_lags.items():
            for lag in lag_hours:
                # Skip wind variables - we'll convert them to U/V components below
                if var_name in ["wind_speed_10m", "wind_direction_10m"]:
                    continue
                    
                if lag == 0:
                    feature_name = var_name
                else:
                    feature_name = f"{var_name}_lag{lag}h"
                
                features[feature_name] = observations[var_name][lag]
        
        # Convert wind speed/direction to U/V components
        # U component: eastward wind (negative = westward)
        # V component: northward wind (negative = southward)
        # U = -speed * sin(direction_radians)
        # V = -speed * cos(direction_radians)
        for lag in meteo_lags["wind_speed_10m"]:
            speed = observations["wind_speed_10m"][lag]
            direction = observations["wind_direction_10m"][lag]
            
            # Convert direction from degrees to radians
            direction_rad = np.deg2rad(direction)
            
            # Compute U and V components
            u_component = -speed * np.sin(direction_rad)
            v_component = -speed * np.cos(direction_rad)
            
            # Add to features
            if lag == 0:
                features["wind_u_10m"] = u_component
                features["wind_v_10m"] = v_component
            else:
                features[f"wind_u_10m_lag{lag}h"] = u_component
                features[f"wind_v_10m_lag{lag}h"] = v_component
        
        # Compute derived meteorological features
        temp_0 = observations["temperature_2m"][0]
        dewpoint_0 = observations["dewpoint_2m"][0]
        rh_0 = observations["relative_humidity_2m"][0]
        
        # VPD: Vapor Pressure Deficit
        # Simplified: VPD ≈ (100 - RH) * 0.6108 * exp(17.27 * T / (T + 237.3)) / 100
        # Using a simpler approximation for now
        saturation_vp = 6.112 * np.exp((17.67 * temp_0) / (temp_0 + 243.5))
        actual_vp = (rh_0 / 100.0) * saturation_vp
        vpd = saturation_vp - actual_vp
        features["vpd"] = vpd
        
        # Temp-Dewpoint Difference
        features["temp_dewpoint_diff"] = temp_0 - dewpoint_0
        
        # Precipitation lags (cumulative)
        precip_lags = self._compute_precipitation_lags(
            grid_id, timestamp_hour
        )
        features.update(precip_lags)
        
        return features
    
    def _compute_precipitation_lags(
        self,
        grid_id: str,
        timestamp_hour: datetime
    ) -> Dict[str, float]:
        """Compute cumulative precipitation over 1h, 6h, 24h windows."""
        precip = {}
        
        windows = {
            "precip_last_1h": 1,
            "precip_last_6h": 6,
            "precip_last_24h": 24,
        }
        
        conn = self._get_connection(TABLE_METEO)
        try:
            for feature_name, hours in windows.items():
                start_ts = timestamp_hour - timedelta(hours=hours)
                
                result = conn.execute(f"""
                    SELECT SUM(precipitation)
                    FROM {TABLE_METEO}
                    WHERE grid_id = ? AND timestamp > ? AND timestamp <= ?
                """, [grid_id, start_ts, timestamp_hour]).fetchone()
                
                if result is None or result[0] is None:
                    # No observations in window; set to 0
                    precip[feature_name] = 0.0
                else:
                    precip[feature_name] = result[0]
        finally:
            self._release_connection(TABLE_METEO, conn)
        
        return precip
    
    # =========================================================================
    # Satellite Feature Computation
    # =========================================================================
    
    def _compute_satellite_features(
        self,
        grid_id: str,
        timestamp_hour: datetime
    ) -> Dict[str, float]:
        """Compute all 13 satellite features from Landsat8 and Sentinel2.
        
        Indices computed:
        - Landsat: NDVI, MNDWI, NDBI, BSI, IBI
        - Sentinel: EVI, UI, SI, NDVI_S2
        - Plus freshness indicators
        """
        features = {}
        
        # Fetch most recent Landsat8 data <= timestamp_hour
        conn_landsat = self._get_connection(TABLE_LANDSAT8)
        try:
            landsat_result = self._find_most_recent_satellite(
                conn_landsat, TABLE_LANDSAT8, grid_id, timestamp_hour
            )
        finally:
            self._release_connection(TABLE_LANDSAT8, conn_landsat)
        
        # Fetch most recent Sentinel2 data <= timestamp_hour
        conn_sentinel = self._get_connection(TABLE_SENTINEL2)
        try:
            sentinel_result = self._find_most_recent_satellite(
                conn_sentinel, TABLE_SENTINEL2, grid_id, timestamp_hour
            )
        finally:
            self._release_connection(TABLE_SENTINEL2, conn_sentinel)
        
        # Compute Landsat indices
        if landsat_result is not None:
            landsat_data, landsat_ts = landsat_result
            landsat_indices = self._compute_landsat_indices(landsat_data)
            features.update(landsat_indices)
            
            # Days since Landsat
            days_since = (timestamp_hour - landsat_ts).days
            features["days_since_landsat"] = float(days_since)
            
            # Check Landsat freshness
            if days_since > 16:  # 16-day threshold
                raise FeatureAssemblyError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    missing_feature=f"landsat8 stale (age={days_since} days)"
                )
        else:
            raise FeatureAssemblyError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                missing_feature="landsat8 data not found"
            )
        
        # Compute Sentinel indices
        if sentinel_result is not None:
            sentinel_data, sentinel_ts = sentinel_result
            sentinel_indices = self._compute_sentinel_indices(sentinel_data)
            features.update(sentinel_indices)
            
            # Days since Sentinel
            days_since = (timestamp_hour - sentinel_ts).days
            features["days_since_sentinel"] = float(days_since)
            
            # Check Sentinel freshness
            if days_since > 5:  # 5-day threshold
                raise FeatureAssemblyError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    missing_feature=f"sentinel2 stale (age={days_since} days)"
                )
        else:
            raise FeatureAssemblyError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                missing_feature="sentinel2 data not found"
            )
        
        return features
    
    def _find_most_recent_satellite(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        grid_id: str,
        timestamp_hour: datetime,
        days_lookback: int = 30
    ) -> Optional[Tuple[Dict[str, float], datetime]]:
        """Find most recent satellite data <= timestamp_hour.
        
        Returns:
            Tuple of (data_dict, timestamp) or None if not found
        """
        # Query most recent data in past 30 days
        lookback_ts = timestamp_hour - timedelta(days=days_lookback)
        
        query = f"""
            SELECT * FROM {table_name}
            WHERE grid_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        result = conn.execute(query, [grid_id, timestamp_hour]).fetchone()
        
        if result is None:
            return None
        
        # Extract columns (first 2 are grid_id, timestamp)
        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
        col_names = [col[0] for col in columns]
        
        data_dict = {col_names[i]: result[i] for i in range(len(col_names))}
        timestamp = data_dict.pop("timestamp")
        data_dict.pop("grid_id")
        data_dict.pop("created_at", None)  # Remove metadata columns
        
        return data_dict, timestamp
    
    def _compute_landsat_indices(self, bands: Dict[str, float]) -> Dict[str, float]:
        """Compute spectral indices from Landsat8 bands.
        
        Band mapping:
            B4: Red, B5: NIR, B3: Green, B2: Blue, B6: SWIR1, B7: SWIR2
        
        Indices:
            NDVI = (B5 - B4) / (B5 + B4)
            MNDWI = (B3 - B6) / (B3 + B6)
            NDBI = (B6 - B5) / (B6 + B5)
            BSI = ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2))
            IBI = NDBI - (NDVI + MNDWI) / 2
        """
        eps = 1e-10
        
        # Extract bands (handle None values)
        b2 = float(bands.get("B2") or 0)
        b3 = float(bands.get("B3") or 0)
        b4 = float(bands.get("B4") or 0)
        b5 = float(bands.get("B5") or 0)
        b6 = float(bands.get("B6") or 0)
        b7 = float(bands.get("B7") or 0)
        
        # Indices
        ndvi = (b5 - b4) / (b5 + b4 + eps) if (b5 + b4 + eps) != 0 else 0.0
        mndwi = (b3 - b6) / (b3 + b6 + eps) if (b3 + b6 + eps) != 0 else 0.0
        ndbi = (b6 - b5) / (b6 + b5 + eps) if (b6 + b5 + eps) != 0 else 0.0
        
        bsi_num = (b6 + b4) - (b5 + b2)
        bsi_den = (b6 + b4) + (b5 + b2) + eps
        bsi = bsi_num / bsi_den if bsi_den != 0 else 0.0
        
        ibi = ndbi - (ndvi + mndwi) / 2.0
        
        # Scene cloud cover (if available in metadata)
        scene_cloud_cover = float(bands.get("scene_cloud_cover") or 0)
        
        # Local NDVI std (placeholder; would require spatial computation)
        local_ndvi_std = 0.0  # TODO: implement spatial neighborhood std
        
        return {
            "ndvi": ndvi,
            "mndwi": mndwi,
            "ndbi": ndbi,
            "bsi": bsi,
            "ibi": ibi,
            "scene_cloud_cover": scene_cloud_cover,
            "local_ndvi_std": local_ndvi_std,
        }
    
    def _compute_sentinel_indices(self, bands: Dict[str, float]) -> Dict[str, float]:
        """Compute spectral indices from Sentinel2 bands.
        
        Band mapping:
            B2: Blue (10m), B4: Red (10m), B8: NIR (10m), AOT: Aerosol, QA10: Quality
        
        Indices:
            EVI = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
            UI = (B4 + B2) / B8 (simplified)
            SI = sqrt(B4^2 + B8^2)
            NDVI_S2 = (B8 - B4) / (B8 + B4)
        """
        eps = 1e-10
        
        # Extract bands
        b2 = float(bands.get("B2") or 0)
        b4 = float(bands.get("B4") or 0)
        b8 = float(bands.get("B8") or 0)
        aot = float(bands.get("AOT") or 0)
        qa10 = int(bands.get("QA10") or 0)
        
        # EVI
        evi_den = b8 + 6*b4 - 7.5*b2 + 1
        evi = 2.5 * (b8 - b4) / evi_den if evi_den != 0 else 0.0
        
        # UI (simplified urban index)
        ui_den = b8 + eps
        ui = (b4 + b2) / ui_den if ui_den != 0 else 0.0
        
        # SI (shadow index)
        si = np.sqrt(b4**2 + b8**2)
        
        # NDVI_S2
        ndvi_s2 = (b8 - b4) / (b8 + b4 + eps) if (b8 + b4 + eps) != 0 else 0.0
        
        return {
            "evi": evi,
            "ui_light": ui,
            "si": si,
            "ndvi_s2": ndvi_s2,
        }
    
    # =========================================================================
    # Spatial Features
    # =========================================================================
    
    def _fetch_spatial_features(
        self,
        conn: duckdb.DuckDBPyConnection,
        grid_id: str
    ) -> Dict[str, float]:
        """Fetch static spatial features (lat, lon)."""
        result = conn.execute(f"""
            SELECT lat, long
            FROM {TABLE_GRID}
            WHERE grid_id = ?
        """, [grid_id]).fetchone()
        
        if result is None:
            raise FeatureAssemblyError(
                grid_id=grid_id,
                timestamp_hour=None,
                missing_feature="grid spatial data"
            )
        
        return {
            "latitude_deg": result[0],
            "longitude_deg": result[1],
        }
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def _validate_features(
        self,
        features: Dict[str, float],
        grid_id: str,
        timestamp_hour: datetime
    ) -> None:
        """Validate all required features are present and non-NULL.
        
        Raises:
            FeatureAssemblyError: If any feature is missing or NULL
        """
        for feature_name in self.required_features:
            if feature_name not in features:
                raise FeatureAssemblyError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    missing_feature=feature_name
                )
            
            if features[feature_name] is None:
                raise FeatureAssemblyError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    missing_feature=f"{feature_name} (NULL value)"
                )
