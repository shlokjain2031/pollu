"""DuckDB schema definitions for Takshashila inference system.

Defines all tables, indices, and schema initialization logic.

TIMESTAMP POLICY:
    All TIMESTAMP columns store timezone-naive datetimes (conceptually UTC).
    DuckDB's TIMESTAMP type does not preserve timezone information.
    Use round_to_hour() from utils.py to normalize all timestamps before DB operations.
"""
import duckdb
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_grid_table(db_path: Path) -> None:
    """Create grid reference table (static spatial metadata) in separate file.
    
    Stores static grid cell reference data (coordinates).
    
    Args:
        db_path: Path to grid.duckdb file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS grid (
                grid_id VARCHAR NOT NULL PRIMARY KEY,
                lat DOUBLE NOT NULL,
                long DOUBLE NOT NULL
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_grid_lat_long ON grid(lat, long)")
        conn.commit()
        logger.info(f"Created table: grid in {db_path}")
    finally:
        conn.close()


def create_pm25_observation_table(db_path: Path) -> None:
    """Create PM2.5 observations table (raw ground-truth data) in separate file.
    
    Stores hourly PM2.5 sensor readings from monitoring stations.
    Freshness: 1 hour
    
    Args:
        db_path: Path to pm25_observations.duckdb file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pm25_observation (
                grid_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                pm25 DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (grid_id, timestamp)
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pm25_obs_grid_ts ON pm25_observation(grid_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pm25_obs_ts ON pm25_observation(timestamp)")
        conn.commit()
        logger.info(f"Created table: pm25_observation in {db_path}")
    finally:
        conn.close()


def create_meteo_table(db_path: Path) -> None:
    """Create meteorological observations table (raw data) in separate file.
    
    Stores hourly meteorological observations from weather stations/reanalysis.
    Freshness: 1 hour (current), 5 days (reanalysis)
    
    Args:
        db_path: Path to meteo.duckdb file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meteo (
                grid_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                temperature_2m DOUBLE,
                relative_humidity_2m DOUBLE,
                dewpoint_2m DOUBLE,
                surface_pressure DOUBLE,
                wind_speed_10m DOUBLE,
                wind_direction_10m DOUBLE,
                precipitation DOUBLE,
                cloud_cover DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (grid_id, timestamp)
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meteo_grid_ts ON meteo(grid_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meteo_ts ON meteo(timestamp)")
        conn.commit()
        logger.info(f"Created table: meteo in {db_path}")
    finally:
        conn.close()


def create_landsat8_table(db_path: Path) -> None:
    """Create Landsat8 imagery table (raw bands only) in separate file.
    
    Stores raw spectral bands (B1-B11) from Landsat8 OLI sensor.
    Freshness: 16 days (revisit cycle)
    
    Args:
        db_path: Path to landsat8.duckdb file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS landsat8 (
                grid_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                B1 DOUBLE, B2 DOUBLE, B3 DOUBLE, B4 DOUBLE, B5 DOUBLE,
                B6 DOUBLE, B7 DOUBLE, B8 DOUBLE, B9 DOUBLE, B10 DOUBLE, B11 DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (grid_id, timestamp)
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_landsat8_grid_ts ON landsat8(grid_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_landsat8_ts ON landsat8(timestamp)")
        conn.commit()
        logger.info(f"Created table: landsat8 in {db_path}")
    finally:
        conn.close()


def create_sentinel2_table(db_path: Path) -> None:
    """Create Sentinel2 imagery table (subset of bands) in separate file.
    
    Stores selected spectral bands (B2, B4, B8) and quality flags from Sentinel2.
    Freshness: 5 days (revisit cycle)
    
    Args:
        db_path: Path to sentinel2.duckdb file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentinel2 (
                grid_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                B2 DOUBLE,
                B4 DOUBLE,
                B8 DOUBLE,
                AOT DOUBLE,
                QA10 INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (grid_id, timestamp)
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentinel2_grid_ts ON sentinel2(grid_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentinel2_ts ON sentinel2(timestamp)")
        conn.commit()
        logger.info(f"Created table: sentinel2 in {db_path}")
    finally:
        conn.close()


def create_predictions_table(db_path: Path) -> None:
    """Create predictions table (source of truth) in separate file.
    
    Args:
        db_path: Path to predictions.duckdb file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                grid_id VARCHAR NOT NULL,
                timestamp_hour TIMESTAMP NOT NULL,
                pm25_prediction DOUBLE NOT NULL,
                model_version VARCHAR NOT NULL,
                schema_hash VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (grid_id, timestamp_hour)
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_grid_ts ON predictions(grid_id, timestamp_hour)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp_hour)")
        conn.commit()
        logger.info(f"Created table: predictions in {db_path}")
    finally:
        conn.close()


def create_all_tables(
    grid_path: Path,
    pm25_path: Path,
    meteo_path: Path,
    landsat_path: Path,
    sentinel_path: Path,
    predictions_path: Path
) -> None:
    """Create all schema tables in separate files.
    
    Raw data tables (each in separate .duckdb file):
    - grid: Static grid reference (lat/long)
    - pm25_observation: Ground-truth PM2.5 observations (hourly)
    - meteo: Meteorological observations (hourly)
    - landsat8: Landsat8 spectral bands (16-day revisit)
    - sentinel2: Sentinel2 spectral bands (5-day revisit)
    
    Output table:
    - predictions: Model predictions (source of truth)
    
    Args:
        grid_path: Path to grid.duckdb
        pm25_path: Path to pm25_observations.duckdb
        meteo_path: Path to meteo.duckdb
        landsat_path: Path to landsat8.duckdb
        sentinel_path: Path to sentinel2.duckdb
        predictions_path: Path to predictions.duckdb
    """
    create_grid_table(grid_path)
    create_pm25_observation_table(pm25_path)
    create_meteo_table(meteo_path)
    create_landsat8_table(landsat_path)
    create_sentinel2_table(sentinel_path)
    create_predictions_table(predictions_path)
    logger.info("All tables created successfully in separate files")


def load_grid_parquet(grid_db_path: Path, parquet_path: Path) -> int:
    """Load grid reference data from parquet file into grid.duckdb.
    
    Args:
        grid_db_path: Path to grid.duckdb file
        parquet_path: Path to grid.parquet
    
    Returns:
        Number of rows loaded
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Grid parquet not found: {parquet_path}")
    
    conn = duckdb.connect(str(grid_db_path))
    try:
        # Check if already loaded
        result = conn.execute("SELECT COUNT(*) FROM grid").fetchone()
        if result[0] > 0:
            logger.info(f"Grid table already contains {result[0]} rows, skipping load")
            return result[0]
        
        # Load parquet directly into DuckDB
        logger.info(f"Loading grid data from {parquet_path}")
        conn.execute(f"""
            INSERT INTO grid (grid_id, lat, long)
            SELECT grid_id, centroid_lat, centroid_lon
            FROM read_parquet('{parquet_path}')
        """)
        
        n_rows = conn.execute("SELECT COUNT(*) FROM grid").fetchone()[0]
        conn.commit()
        logger.info(f"Loaded {n_rows} grid cells into {grid_db_path}")
        
        return n_rows
    finally:
        conn.close()


def initialize_schema(
    grid_path: Path,
    pm25_path: Path,
    meteo_path: Path,
    landsat_path: Path,
    sentinel_path: Path,
    predictions_path: Path,
    grid_parquet_path: Optional[Path] = None
) -> None:
    """Initialize complete DuckDB schema across separate files and load grid data.
    
    Creates raw data tables (each in separate file):
    - grid: Static grid reference
    - pm25_observation: Ground-truth observations
    - meteo: Weather observations
    - landsat8: Landsat spectral data
    - sentinel2: Sentinel spectral data
    - predictions: Model predictions
    
    Args:
        grid_path: Path to grid.duckdb
        pm25_path: Path to pm25_observations.duckdb
        meteo_path: Path to meteo.duckdb
        landsat_path: Path to landsat8.duckdb
        sentinel_path: Path to sentinel2.duckdb
        predictions_path: Path to predictions.duckdb
        grid_parquet_path: Path to grid.parquet (optional, for loading)
    """
    # Create all tables in their respective files
    create_all_tables(grid_path, pm25_path, meteo_path, landsat_path, sentinel_path, predictions_path)
    
    # Load grid data if provided
    if grid_parquet_path and grid_parquet_path.exists():
        load_grid_parquet(grid_path, grid_parquet_path)
    
    logger.info("Schema initialized successfully across all DuckDB files")
