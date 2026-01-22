"""DuckDB schema definitions for Takshashila inference system.

Defines all tables, indices, and schema initialization logic.
"""
import duckdb
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_grid_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create grid reference table (static spatial metadata).
    
    Stores static grid cell reference data (coordinates).
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS grid (
            grid_id VARCHAR NOT NULL PRIMARY KEY,
            lat DOUBLE NOT NULL,
            long DOUBLE NOT NULL
        )
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_grid_lat_long ON grid(lat, long)")
    logger.info("Created table: grid")


def create_pm25_observation_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create PM2.5 observations table (raw ground-truth data).
    
    Stores hourly PM2.5 sensor readings from monitoring stations.
    Freshness: 1 hour
    """
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
    logger.info("Created table: pm25_observation")


def create_meteo_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create meteorological observations table (raw data).
    
    Stores hourly meteorological observations from weather stations/reanalysis.
    Freshness: 1 hour (current), 5 days (reanalysis)
    """
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
    logger.info("Created table: meteo")


def create_landsat8_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create Landsat8 imagery table (raw bands only).
    
    Stores raw spectral bands (B1-B11) from Landsat8 OLI sensor.
    Freshness: 16 days (revisit cycle)
    """
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
    logger.info("Created table: landsat8")


def create_sentinel2_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create Sentinel2 imagery table (subset of bands).
    
    Stores selected spectral bands (B2, B4, B8) and quality flags from Sentinel2.
    Freshness: 5 days (revisit cycle)
    """
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
    logger.info("Created table: sentinel2")


def create_predictions_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create predictions table (source of truth)."""
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
    logger.info("Created table: predictions")


def create_all_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all schema tables in correct order.
    
    Raw data tables:
    - grid: Static grid reference (lat/long)
    - pm25_observation: Ground-truth PM2.5 observations (hourly)
    - meteo: Meteorological observations (hourly)
    - landsat8: Landsat8 spectral bands (16-day revisit)
    - sentinel2: Sentinel2 spectral bands (5-day revisit)
    
    Output table:
    - predictions: Model predictions (source of truth)
    """
    create_grid_table(conn)
    create_pm25_observation_table(conn)
    create_meteo_table(conn)
    create_landsat8_table(conn)
    create_sentinel2_table(conn)
    create_predictions_table(conn)
    logger.info("All tables created successfully")


def load_grid_parquet(conn: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    """Load grid reference data from parquet file.
    
    Args:
        conn: DuckDB connection
        parquet_path: Path to grid.parquet
    
    Returns:
        Number of rows loaded
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Grid parquet not found: {parquet_path}")
    
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
    logger.info(f"Loaded {n_rows} grid cells")
    
    return n_rows


def initialize_schema(db_path: Path, grid_parquet_path: Optional[Path] = None) -> None:
    """Initialize complete DuckDB schema and load grid data.
    
    Creates raw data tables:
    - grid: Static grid reference
    - pm25_observation: Ground-truth observations
    - meteo: Weather observations
    - landsat8: Landsat spectral data
    - sentinel2: Sentinel spectral data
    - predictions: Model predictions
    
    Args:
        db_path: Path to DuckDB file
        grid_parquet_path: Path to grid.parquet (optional)
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(str(db_path))
    
    try:
        create_all_tables(conn)
        
        if grid_parquet_path and grid_parquet_path.exists():
            load_grid_parquet(conn, grid_parquet_path)
        
        conn.commit()
        logger.info(f"Schema initialized successfully at {db_path}")
    
    finally:
        conn.close()
