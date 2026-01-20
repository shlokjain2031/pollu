"""Feature assembly from DuckDB tables.

Fetches and validates all 64 features for a (grid_id, timestamp_hour) pair.
"""
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import logging

from .errors import FeatureAssemblyError, InvalidRequestError, PersistenceError
from .config import (
    TABLE_GRIDS, TABLE_FEATURES_PM25, TABLE_FEATURES_METEO,
    TABLE_FEATURES_SATELLITE, TABLE_FEATURES_SPATIAL
)

logger = logging.getLogger(__name__)


class FeatureAssembler:
    """Assemble all required features from DuckDB feature tables.
    
    Validates:
    - grid_id exists in grids table
    - All feature table rows exist
    - All required features are non-NULL
    - Feature ordering matches schema
    """
    
    def __init__(self, db_path: str, feature_schema: Dict):
        """Initialize feature assembler.
        
        Args:
            db_path: Path to DuckDB file
            feature_schema: Loaded feature_schema.json dict
        """
        self.db_path = Path(db_path)
        self.required_features = feature_schema["feature_list"]
        self.n_features = len(self.required_features)
        
        logger.info(f"FeatureAssembler initialized with {self.n_features} features")
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a new DuckDB connection for reading."""
        return duckdb.connect(str(self.db_path), read_only=True)
    
    def fetch_features(self, grid_id: str, timestamp_hour: datetime) -> pd.DataFrame:
        """Fetch all 64 features for (grid_id, timestamp_hour).
        
        Args:
            grid_id: Grid cell identifier (e.g., "g00000001")
            timestamp_hour: Prediction timestamp (hour-floored)
        
        Returns:
            Single-row DataFrame with 64 feature columns in correct order
        
        Raises:
            InvalidRequestError: If grid_id not in grids table
            FeatureAssemblyError: If any required feature is missing or NULL
        """
        conn = self._get_connection()
        
        try:
            # 1. Validate grid exists
            grid_exists = conn.execute(
                f"SELECT COUNT(*) FROM {TABLE_GRIDS} WHERE grid_id = ?",
                [grid_id]
            ).fetchone()[0] > 0
            
            if not grid_exists:
                raise InvalidRequestError(f"grid_id '{grid_id}' not found in grids table")
            
            # 2. Fetch PM2.5 lag features
            pm25_features = self._fetch_pm25_features(conn, grid_id, timestamp_hour)
            
            # 3. Fetch meteorological features
            meteo_features = self._fetch_meteo_features(conn, grid_id, timestamp_hour)
            
            # 4. Fetch satellite features
            sat_features = self._fetch_satellite_features(conn, grid_id, timestamp_hour)
            
            # 5. Fetch spatial features (static, keyed by grid_id only)
            spatial_features = self._fetch_spatial_features(conn, grid_id)
            
            # 6. Merge all features
            all_features = {**pm25_features, **meteo_features, **sat_features, **spatial_features}
            
            # 7. Validate completeness
            self._validate_features(all_features, grid_id, timestamp_hour)
            
            # 8. Build DataFrame in correct column order
            features_df = pd.DataFrame([all_features])
            features_df = features_df[self.required_features]  # Reorder columns
            
            logger.debug(f"Assembled {len(all_features)} features for {grid_id} @ {timestamp_hour}")
            
            return features_df
        
        finally:
            conn.close()
    
    def _fetch_pm25_features(
        self,
        conn: duckdb.DuckDBPyConnection,
        grid_id: str,
        timestamp_hour: datetime
    ) -> Dict[str, float]:
        """Fetch PM2.5 lag features (6 features)."""
        result = conn.execute(f"""
            SELECT pm25_t_minus_1h, pm25_t_minus_2h, pm25_t_minus_3h,
                   pm25_t_minus_6h, pm25_t_minus_12h, pm25_t_minus_24h
            FROM {TABLE_FEATURES_PM25}
            WHERE grid_id = ? AND timestamp_hour = ?
        """, [grid_id, timestamp_hour]).fetchone()
        
        if result is None:
            raise FeatureAssemblyError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                missing_feature="features_pm25 row"
            )
        
        return {
            "pm25_t_minus_1h": result[0],
            "pm25_t_minus_2h": result[1],
            "pm25_t_minus_3h": result[2],
            "pm25_t_minus_6h": result[3],
            "pm25_t_minus_12h": result[4],
            "pm25_t_minus_24h": result[5]
        }
    
    def _fetch_meteo_features(
        self,
        conn: duckdb.DuckDBPyConnection,
        grid_id: str,
        timestamp_hour: datetime
    ) -> Dict[str, float]:
        """Fetch meteorological features (44 features)."""
        result = conn.execute(f"""
            SELECT 
                temperature_2m, temperature_2m_lag1h, temperature_2m_lag3h,
                temperature_2m_lag12h, temperature_2m_lag24h,
                relative_humidity_2m, relative_humidity_2m_lag1h, relative_humidity_2m_lag3h,
                relative_humidity_2m_lag6h, relative_humidity_2m_lag12h, relative_humidity_2m_lag24h,
                dewpoint_2m, dewpoint_2m_lag1h, dewpoint_2m_lag3h,
                dewpoint_2m_lag6h, dewpoint_2m_lag12h, dewpoint_2m_lag24h,
                surface_pressure, surface_pressure_lag1h, surface_pressure_lag3h,
                surface_pressure_lag6h, surface_pressure_lag12h, surface_pressure_lag24h,
                wind_u_10m, wind_u_10m_lag1h, wind_u_10m_lag3h,
                wind_u_10m_lag6h, wind_u_10m_lag12h, wind_u_10m_lag24h,
                wind_v_10m, wind_v_10m_lag1h, wind_v_10m_lag3h,
                wind_v_10m_lag6h, wind_v_10m_lag12h, wind_v_10m_lag24h,
                cloud_cover, cloud_cover_lag6h, cloud_cover_lag24h,
                vpd, temp_dewpoint_diff,
                precip_last_1h, precip_last_6h, precip_last_24h
            FROM {TABLE_FEATURES_METEO}
            WHERE grid_id = ? AND timestamp_hour = ?
        """, [grid_id, timestamp_hour]).fetchone()
        
        if result is None:
            raise FeatureAssemblyError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                missing_feature="features_meteo row"
            )
        
        return {
            "temperature_2m": result[0],
            "temperature_2m_lag1h": result[1],
            "temperature_2m_lag3h": result[2],
            "temperature_2m_lag12h": result[3],
            "temperature_2m_lag24h": result[4],
            "relative_humidity_2m": result[5],
            "relative_humidity_2m_lag1h": result[6],
            "relative_humidity_2m_lag3h": result[7],
            "relative_humidity_2m_lag6h": result[8],
            "relative_humidity_2m_lag12h": result[9],
            "relative_humidity_2m_lag24h": result[10],
            "dewpoint_2m": result[11],
            "dewpoint_2m_lag1h": result[12],
            "dewpoint_2m_lag3h": result[13],
            "dewpoint_2m_lag6h": result[14],
            "dewpoint_2m_lag12h": result[15],
            "dewpoint_2m_lag24h": result[16],
            "surface_pressure": result[17],
            "surface_pressure_lag1h": result[18],
            "surface_pressure_lag3h": result[19],
            "surface_pressure_lag6h": result[20],
            "surface_pressure_lag12h": result[21],
            "surface_pressure_lag24h": result[22],
            "wind_u_10m": result[23],
            "wind_u_10m_lag1h": result[24],
            "wind_u_10m_lag3h": result[25],
            "wind_u_10m_lag6h": result[26],
            "wind_u_10m_lag12h": result[27],
            "wind_u_10m_lag24h": result[28],
            "wind_v_10m": result[29],
            "wind_v_10m_lag1h": result[30],
            "wind_v_10m_lag3h": result[31],
            "wind_v_10m_lag6h": result[32],
            "wind_v_10m_lag12h": result[33],
            "wind_v_10m_lag24h": result[34],
            "cloud_cover": result[35],
            "cloud_cover_lag6h": result[36],
            "cloud_cover_lag24h": result[37],
            "vpd": result[38],
            "temp_dewpoint_diff": result[39],
            "precip_last_1h": result[40],
            "precip_last_6h": result[41],
            "precip_last_24h": result[42]
        }
    
    def _fetch_satellite_features(
        self,
        conn: duckdb.DuckDBPyConnection,
        grid_id: str,
        timestamp_hour: datetime
    ) -> Dict[str, float]:
        """Fetch satellite imagery features (12 features)."""
        result = conn.execute(f"""
            SELECT ndvi, mndwi, ndbi, bsi, ibi, scene_cloud_cover,
                   evi, ui_light, si, local_ndvi_std, ndvi_s2,
                   days_since_landsat, days_since_sentinel
            FROM {TABLE_FEATURES_SATELLITE}
            WHERE grid_id = ? AND timestamp_hour = ?
        """, [grid_id, timestamp_hour]).fetchone()
        
        if result is None:
            raise FeatureAssemblyError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                missing_feature="features_satellite row"
            )
        
        return {
            "ndvi": result[0],
            "mndwi": result[1],
            "ndbi": result[2],
            "bsi": result[3],
            "ibi": result[4],
            "scene_cloud_cover": result[5],
            "evi": result[6],
            "ui_light": result[7],
            "si": result[8],
            "local_ndvi_std": result[9],
            "ndvi_s2": result[10],
            "days_since_landsat": result[11],
            "days_since_sentinel": result[12]
        }
    
    def _fetch_spatial_features(
        self,
        conn: duckdb.DuckDBPyConnection,
        grid_id: str
    ) -> Dict[str, float]:
        """Fetch spatial/static features (2 features: lat, lon)."""
        result = conn.execute(f"""
            SELECT latitude_deg, longitude_deg
            FROM {TABLE_FEATURES_SPATIAL}
            WHERE grid_id = ?
        """, [grid_id]).fetchone()
        
        if result is None:
            raise FeatureAssemblyError(
                grid_id=grid_id,
                timestamp_hour=None,
                missing_feature="features_spatial row"
            )
        
        return {
            "latitude_deg": result[0],
            "longitude_deg": result[1]
        }
    
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
