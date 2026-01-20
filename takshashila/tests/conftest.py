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

from Takshashila.phase1_minimal_inference.schema import initialize_schema
from Takshashila.phase1_minimal_inference.config import (
    MODEL_PATH, FEATURE_SCHEMA_PATH, GRID_PARQUET_PATH
)
from Takshashila.phase1_minimal_inference.inference import InferenceEngine
from Takshashila.phase1_minimal_inference.utils import setup_logging
from Takshashila.scripts.populate_mock_features import populate_features
import duckdb

# Setup logging for tests
setup_logging(level="DEBUG")


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """Create temporary test DuckDB with schema and mock data.
    
    Scope: session (shared across all tests)
    """
    # Create temp directory
    tmp_dir = tmp_path_factory.mktemp("duckdb")
    db_path = tmp_dir / "test_predictions.duckdb"
    
    # Initialize schema
    initialize_schema(db_path, GRID_PARQUET_PATH)
    
    # Populate mock features
    populate_features(db_path)
    
    yield str(db_path)
    
    # Cleanup handled by tmp_path_factory


@pytest.fixture(scope="session")
def inference_engine(test_db_path):
    """Provide InferenceEngine with test database.
    
    Scope: session (shared across all tests)
    """
    return InferenceEngine(
        db_path=test_db_path,
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
def persistence(test_db_path):
    """Provide PersistenceLayer instance."""
    from Takshashila.phase1_minimal_inference.persistence import PersistenceLayer
    return PersistenceLayer(test_db_path)


@pytest.fixture
def fresh_db_connection(test_db_path):
    """Provide a fresh DuckDB connection for individual tests.
    
    Scope: function (new connection per test)
    """
    conn = duckdb.connect(str(test_db_path))
    yield conn
    conn.close()
