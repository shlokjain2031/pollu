"""DuckDB schema definitions for Takshashila inference system.

Defines all tables, indices, and schema initialization logic.
"""
import duckdb
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_grids_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create grids reference table.
    
    Stores 509,209 grid cells loaded from Mumbai grid parquet.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS grids (
            row_idx INTEGER PRIMARY KEY,
            grid_id VARCHAR NOT NULL UNIQUE,
            x_m DOUBLE NOT NULL,
            y_m DOUBLE NOT NULL,
            centroid_lon DOUBLE NOT NULL,
            centroid_lat DOUBLE NOT NULL,
            col_idx INTEGER NOT NULL,
            row_idx_grid INTEGER NOT NULL,
            h3_cell VARCHAR NOT NULL
        )
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_grids_grid_id ON grids(grid_id)")
    logger.info("Created table: grids")


def create_features_pm25_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create PM2.5 lag features table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features_pm25 (
            grid_id VARCHAR NOT NULL,
            timestamp_hour TIMESTAMP NOT NULL,
            pm25_t_minus_1h DOUBLE,
            pm25_t_minus_2h DOUBLE,
            pm25_t_minus_3h DOUBLE,
            pm25_t_minus_6h DOUBLE,
            pm25_t_minus_12h DOUBLE,
            pm25_t_minus_24h DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (grid_id, timestamp_hour)
        )
    """)
    logger.info("Created table: features_pm25")


def create_features_meteo_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create meteorological features table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features_meteo (
            grid_id VARCHAR NOT NULL,
            timestamp_hour TIMESTAMP NOT NULL,
            temperature_2m DOUBLE,
            temperature_2m_lag1h DOUBLE,
            temperature_2m_lag3h DOUBLE,
            temperature_2m_lag12h DOUBLE,
            temperature_2m_lag24h DOUBLE,
            relative_humidity_2m DOUBLE,
            relative_humidity_2m_lag1h DOUBLE,
            relative_humidity_2m_lag3h DOUBLE,
            relative_humidity_2m_lag6h DOUBLE,
            relative_humidity_2m_lag12h DOUBLE,
            relative_humidity_2m_lag24h DOUBLE,
            dewpoint_2m DOUBLE,
            dewpoint_2m_lag1h DOUBLE,
            dewpoint_2m_lag3h DOUBLE,
            dewpoint_2m_lag6h DOUBLE,
            dewpoint_2m_lag12h DOUBLE,
            dewpoint_2m_lag24h DOUBLE,
            surface_pressure DOUBLE,
            surface_pressure_lag1h DOUBLE,
            surface_pressure_lag3h DOUBLE,
            surface_pressure_lag6h DOUBLE,
            surface_pressure_lag12h DOUBLE,
            surface_pressure_lag24h DOUBLE,
            wind_u_10m DOUBLE,
            wind_u_10m_lag1h DOUBLE,
            wind_u_10m_lag3h DOUBLE,
            wind_u_10m_lag6h DOUBLE,
            wind_u_10m_lag12h DOUBLE,
            wind_u_10m_lag24h DOUBLE,
            wind_v_10m DOUBLE,
            wind_v_10m_lag1h DOUBLE,
            wind_v_10m_lag3h DOUBLE,
            wind_v_10m_lag6h DOUBLE,
            wind_v_10m_lag12h DOUBLE,
            wind_v_10m_lag24h DOUBLE,
            cloud_cover DOUBLE,
            cloud_cover_lag6h DOUBLE,
            cloud_cover_lag24h DOUBLE,
            vpd DOUBLE,
            temp_dewpoint_diff DOUBLE,
            precip_last_1h DOUBLE,
            precip_last_6h DOUBLE,
            precip_last_24h DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (grid_id, timestamp_hour)
        )
    """)
    logger.info("Created table: features_meteo")


def create_features_satellite_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create satellite imagery features table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features_satellite (
            grid_id VARCHAR NOT NULL,
            timestamp_hour TIMESTAMP NOT NULL,
            ndvi DOUBLE,
            mndwi DOUBLE,
            ndbi DOUBLE,
            bsi DOUBLE,
            ibi DOUBLE,
            scene_cloud_cover DOUBLE,
            evi DOUBLE,
            ui_light DOUBLE,
            si DOUBLE,
            local_ndvi_std DOUBLE,
            ndvi_s2 DOUBLE,
            days_since_landsat DOUBLE,
            days_since_sentinel DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (grid_id, timestamp_hour)
        )
    """)
    logger.info("Created table: features_satellite")


def create_features_spatial_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create spatial/static features table (lat/lon per grid)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features_spatial (
            grid_id VARCHAR NOT NULL PRIMARY KEY,
            latitude_deg DOUBLE NOT NULL,
            longitude_deg DOUBLE NOT NULL
        )
    """)
    logger.info("Created table: features_spatial")


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
    """Create all schema tables in correct order."""
    create_grids_table(conn)
    create_features_pm25_table(conn)
    create_features_meteo_table(conn)
    create_features_satellite_table(conn)
    create_features_spatial_table(conn)
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
    result = conn.execute("SELECT COUNT(*) FROM grids").fetchone()
    if result[0] > 0:
        logger.info(f"Grids table already contains {result[0]} rows, skipping load")
        return result[0]
    
    # Load parquet directly into DuckDB
    logger.info(f"Loading grid data from {parquet_path}")
    conn.execute(f"""
        INSERT INTO grids 
        SELECT row_idx, grid_id, x_m, y_m, centroid_lon, centroid_lat, 
               col_idx, row_idx_grid, h3_cell
        FROM read_parquet('{parquet_path}')
    """)
    
    n_rows = conn.execute("SELECT COUNT(*) FROM grids").fetchone()[0]
    logger.info(f"Loaded {n_rows} grid cells")
    
    return n_rows


def populate_spatial_features_from_grids(conn: duckdb.DuckDBPyConnection) -> int:
    """Populate features_spatial table from grids (one-time operation).
    
    Returns:
        Number of rows populated
    """
    # Check if already populated
    result = conn.execute("SELECT COUNT(*) FROM features_spatial").fetchone()
    if result[0] > 0:
        logger.info(f"features_spatial already contains {result[0]} rows, skipping")
        return result[0]
    
    logger.info("Populating features_spatial from grids table")
    conn.execute("""
        INSERT INTO features_spatial (grid_id, latitude_deg, longitude_deg)
        SELECT grid_id, centroid_lat, centroid_lon
        FROM grids
    """)
    
    n_rows = conn.execute("SELECT COUNT(*) FROM features_spatial").fetchone()[0]
    logger.info(f"Populated {n_rows} spatial feature rows")
    
    return n_rows


def initialize_schema(db_path: Path, grid_parquet_path: Optional[Path] = None) -> None:
    """Initialize complete DuckDB schema and load grid data.
    
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
            populate_spatial_features_from_grids(conn)
        
        conn.commit()
        logger.info(f"Schema initialized successfully at {db_path}")
    
    finally:
        conn.close()
