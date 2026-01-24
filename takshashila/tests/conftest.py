"""Pytest configuration and shared fixtures for Phase 1 tests."""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import sys
import logging

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from takshashila.core.schema import initialize_schema
from takshashila.core.config import (
    MODEL_PATH, FEATURE_SCHEMA_PATH, GRID_PARQUET_PATH,
    TABLE_GRID, TABLE_PM25_OBSERVATION, TABLE_METEO,
    TABLE_LANDSAT8, TABLE_SENTINEL2, TABLE_PREDICTIONS
)
from takshashila.core.inference import InferenceEngine
from takshashila.core.utils import setup_logging
import duckdb

# Setup logging for tests
setup_logging(level="DEBUG")


@pytest.fixture(scope="session")
def test_db_paths(tmp_path_factory):
    """Create temporary test DuckDB files with schema and mock data.
    
    Scope: session (shared across all tests)
    
    Returns:
        Dict mapping table names to file paths
    """
    # Create temp directory
    tmp_dir = tmp_path_factory.mktemp("duckdb")
    
    # Create paths for all database files
    db_paths = {
        TABLE_GRID: tmp_dir / "grid.duckdb",
        TABLE_PM25_OBSERVATION: tmp_dir / "pm25_observations.duckdb",
        TABLE_METEO: tmp_dir / "meteo.duckdb",
        TABLE_LANDSAT8: tmp_dir / "landsat8.duckdb",
        TABLE_SENTINEL2: tmp_dir / "sentinel2.duckdb",
        TABLE_PREDICTIONS: tmp_dir / "predictions.duckdb",
    }
    
    # Initialize schema across all files
    initialize_schema(
        grid_path=str(db_paths[TABLE_GRID]),
        pm25_path=str(db_paths[TABLE_PM25_OBSERVATION]),
        meteo_path=str(db_paths[TABLE_METEO]),
        landsat_path=str(db_paths[TABLE_LANDSAT8]),
        sentinel_path=str(db_paths[TABLE_SENTINEL2]),
        predictions_path=str(db_paths[TABLE_PREDICTIONS]),
        grid_parquet_path=str(GRID_PARQUET_PATH)
    )
    
    # Populate mock features (import here to avoid circular dependency)
    from takshashila.scripts.populate_mock_features import (
        generate_pm25_observation, generate_meteo_observation,
        generate_landsat8_observation, generate_sentinel2_observation,
        BASE_TIMESTAMP, GRID_IDS, NUM_GRIDS
    )
    from datetime import timedelta
    import random
    
    # === POPULATE PM2.5 OBSERVATIONS ===
    conn_pm25 = duckdb.connect(str(db_paths[TABLE_PM25_OBSERVATION]))
    try:
        conn_pm25.execute("DELETE FROM pm25_observation")
        pm25_batch = []
        base_hour = BASE_TIMESTAMP.replace(minute=0, second=0, microsecond=0)
        for grid_id in GRID_IDS:
            for hour_offset in range(168):  # 7 days × 24 hours
                timestamp = (base_hour - timedelta(hours=hour_offset)).replace(tzinfo=None)
                obs = generate_pm25_observation(grid_id, timestamp)
                pm25_batch.append((obs["grid_id"], obs["timestamp"], obs["pm25"]))
        
        conn_pm25.executemany(
            "INSERT INTO pm25_observation (grid_id, timestamp, pm25) VALUES (?, ?, ?)",
            pm25_batch
        )
        conn_pm25.commit()
    finally:
        conn_pm25.close()
    
    # === POPULATE METEOROLOGICAL DATA ===
    conn_meteo = duckdb.connect(str(db_paths[TABLE_METEO]))
    try:
        conn_meteo.execute("DELETE FROM meteo")
        meteo_batch = []
        base_hour = BASE_TIMESTAMP.replace(minute=0, second=0, microsecond=0)
        for grid_id in GRID_IDS:
            for hour_offset in range(168):  # 7 days × 24 hours
                timestamp = (base_hour - timedelta(hours=hour_offset)).replace(tzinfo=None)
                obs = generate_meteo_observation(grid_id, timestamp)
                meteo_batch.append((
                    obs["grid_id"], obs["timestamp"], obs["temperature_2m"],
                    obs["relative_humidity_2m"], obs["dewpoint_2m"], obs["surface_pressure"],
                    obs["wind_speed_10m"], obs["wind_direction_10m"],
                    obs["precipitation"], obs["cloud_cover"]
                ))
        
        conn_meteo.executemany(
            """INSERT INTO meteo (grid_id, timestamp, temperature_2m, relative_humidity_2m, 
                                   dewpoint_2m, surface_pressure, wind_speed_10m, wind_direction_10m, 
                                   precipitation, cloud_cover) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            meteo_batch
        )
        conn_meteo.commit()
    finally:
        conn_meteo.close()
    
    # === POPULATE LANDSAT8 DATA ===
    conn_landsat = duckdb.connect(str(db_paths[TABLE_LANDSAT8]))
    try:
        conn_landsat.execute("DELETE FROM landsat8")
        base_hour = BASE_TIMESTAMP.replace(minute=0, second=0, microsecond=0)
        landsat_timestamp = (base_hour - timedelta(days=5)).replace(tzinfo=None)
        landsat_batch = []
        for grid_id in GRID_IDS:
            obs = generate_landsat8_observation(grid_id, landsat_timestamp)
            landsat_batch.append((
                obs["grid_id"], obs["timestamp"],
                obs["B1"], obs["B2"], obs["B3"], obs["B4"], obs["B5"],
                obs["B6"], obs["B7"], obs["B8"], obs["B9"], obs["B10"], obs["B11"]
            ))
        
        conn_landsat.executemany(
            """INSERT INTO landsat8 (grid_id, timestamp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            landsat_batch
        )
        conn_landsat.commit()
    finally:
        conn_landsat.close()
    
    # === POPULATE SENTINEL2 DATA ===
    conn_sentinel = duckdb.connect(str(db_paths[TABLE_SENTINEL2]))
    try:
        conn_sentinel.execute("DELETE FROM sentinel2")
        base_hour = BASE_TIMESTAMP.replace(minute=0, second=0, microsecond=0)
        sentinel_timestamp = (base_hour - timedelta(days=2)).replace(tzinfo=None)
        sentinel_batch = []
        for grid_id in GRID_IDS:
            obs = generate_sentinel2_observation(grid_id, sentinel_timestamp)
            sentinel_batch.append((
                obs["grid_id"], obs["timestamp"],
                obs["B2"], obs["B4"], obs["B8"], obs["AOT"], obs["QA10"]
            ))
        
        conn_sentinel.executemany(
            """INSERT INTO sentinel2 (grid_id, timestamp, B2, B4, B8, AOT, QA10)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            sentinel_batch
        )
        conn_sentinel.commit()
    finally:
        conn_sentinel.close()
    
    yield {k: str(v) for k, v in db_paths.items()}
    
    # Cleanup handled by tmp_path_factory


@pytest.fixture(scope="session")
def inference_engine(test_db_paths):
    """Provide InferenceEngine with test databases.
    
    Scope: session (shared across all tests)
    """
    return InferenceEngine(
        db_paths=test_db_paths,
        model_path=str(MODEL_PATH),
        feature_schema_path=str(FEATURE_SCHEMA_PATH)
    )


@pytest.fixture
def test_timestamp():
    """Standard test timestamp for predictable testing."""
    return datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def valid_grid_id():
    """A valid grid_id with complete features."""
    return "g00000001"


@pytest.fixture
def invalid_grid_id():
    """An invalid grid_id not in grids table."""
    return "g99999999"


@pytest.fixture
def missing_pm25_grid_id():
    """Grid with missing PM2.5 features."""
    return "g00000100"


@pytest.fixture
def missing_meteo_grid_id():
    """Grid with missing meteorological features."""
    return "g00000200"


@pytest.fixture
def missing_sat_grid_id():
    """Grid with missing satellite features."""
    return "g00000300"


@pytest.fixture
def assembler(inference_engine):
    """Provide FeatureAssembler instance."""
    return inference_engine.feature_assembler


@pytest.fixture
def persistence(test_db_paths):
    """Provide PersistenceLayer instance.
    
    Uses the predictions database from test_db_paths.
    """
    from takshashika.core.persistence import PersistenceLayer
    return PersistenceLayer(test_db_paths[TABLE_PREDICTIONS])


@pytest.fixture
def fresh_db_connection(test_db_paths):
    """Provide a fresh DuckDB connection for individual tests.
    
    Scope: function (new connection per test)
    Uses the grid database which is read-only in tests.
    """
    conn = duckdb.connect(str(test_db_paths[TABLE_GRID]))
    yield conn
    conn.close()
