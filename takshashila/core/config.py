"""Configuration constants for Takshashila inference system.

Centralizes paths, model metadata, and timeout budgets.
"""
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
TAKSHASHILA_ROOT = PROJECT_ROOT / "Takshashila"

# Model artifacts
MODEL_PATH = PROJECT_ROOT / "dwarka/models/prod_v1/model.txt"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "dwarka/models/prod_v1/feature_schema.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "dwarka/models/prod_v1/manifest.json"

# Grid artifacts
GRID_PARQUET_PATH = PROJECT_ROOT / "artifacts/v1/grids/mumbai/grid.parquet"
GRID_METADATA_PATH = PROJECT_ROOT / "artifacts/v1/grids/mumbai/metadata.json"

# DuckDB storage
DUCKDB_PATH = TAKSHASHILA_ROOT / "duckdb/predictions.duckdb"

# Model metadata
MODEL_VERSION = "prod_v1"
SCHEMA_HASH = "7431b6510f6f686a"
N_FEATURES = 64

# Timeouts (seconds)
# Phase 1 does not enforce these; defined for future phases
FEATURE_ASSEMBLY_TIMEOUT_SEC = 0.5
MODEL_INFERENCE_TIMEOUT_SEC = 0.2
PERSISTENCE_TIMEOUT_SEC = 0.3
TOTAL_INFERENCE_TIMEOUT_SEC = 2.0

# DuckDB write lock timeout
WRITE_LOCK_TIMEOUT_SEC = 2.0

# Grid metadata (from artifacts/v1/grids/mumbai/metadata.json)
GRID_ID = "mumbai"
GRID_VERSION = "v1"
N_GRID_CELLS = 509209  # Expected grid count

# Feature table names (raw data)
TABLE_GRID = "grid"
TABLE_PM25_OBSERVATION = "pm25_observation"
TABLE_METEO = "meteo"
TABLE_LANDSAT8 = "landsat8"
TABLE_SENTINEL2 = "sentinel2"
TABLE_PREDICTIONS = "predictions"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
