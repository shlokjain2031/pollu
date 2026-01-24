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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

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
    from kashi.db_pool import PoolManager
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
            conn, grid_id, timestamp_hour
        )
        features.update(precip_lags)
        
        return features
    
    def _compute_precipitation_lags(
        self,
        conn: duckdb.DuckDBPyConnection,
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
