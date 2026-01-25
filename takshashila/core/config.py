"""Configuration constants for Takshashila inference system.

Centralizes paths, model metadata, and timeout budgets.
"""
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
TAKSHASHILA_ROOT = PROJECT_ROOT / "takshashila"

# Model artifacts
MODEL_PATH = PROJECT_ROOT / "dwarka/models/prod_v1/model.txt"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "dwarka/models/prod_v1/feature_schema.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "dwarka/models/prod_v1/manifest.json"

# Grid artifacts
GRID_PARQUET_PATH = PROJECT_ROOT / "artifacts/v1/grids/mumbai/grid.parquet"
GRID_METADATA_PATH = PROJECT_ROOT / "artifacts/v1/grids/mumbai/metadata.json"

# DuckDB storage (one file per table in duckdb/ directory)
DUCKDB_DIR = TAKSHASHILA_ROOT / "duckdb"
DUCKDB_GRID_PATH = DUCKDB_DIR / "grid.duckdb"
DUCKDB_PM25_PATH = DUCKDB_DIR / "pm25_observations.duckdb"
DUCKDB_METEO_PATH = DUCKDB_DIR / "meteo.duckdb"
DUCKDB_LANDSAT8_PATH = DUCKDB_DIR / "landsat8.duckdb"
DUCKDB_SENTINEL2_PATH = DUCKDB_DIR / "sentinel2.duckdb"
DUCKDB_PREDICTIONS_PATH = DUCKDB_DIR / "predictions.duckdb"

# Legacy path kept for backward compatibility (deprecated)
DUCKDB_PATH = DUCKDB_DIR / "pollu.duckdb"

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

# ============================================================================
# ORCHESTRATOR & BATCH JOB CONFIGURATION (Phase 2b & 3)
# ============================================================================

# Orchestrator Scheduling (cron times)
SCHEDULE_HOUR = 0  # UTC hour when cycle runs (0-23)
SCHEDULE_MINUTE = 0  # UTC minute when cycle runs (0-59)
WRITER_START_OFFSET_MINUTES = 2  # WriterJob starts 2m before hour (t=-2m)
INFERENCE_START_DELAY_SECONDS = 120  # InferenceJob starts at t=0m (+ 2 min after hour)

# Job Execution Timeouts
WRITER_JOB_TIMEOUT_SECONDS = 300  # 5 minutes for WriterJob (generous for satellites)
INFERENCE_JOB_TIMEOUT_SECONDS = 300  # 5 minutes for InferenceJob hot cache
HOT_CACHE_TIMEOUT_SECONDS = 120  # Max time to populate hot cache (5K grids)
WARM_CACHE_TIMEOUT_SECONDS = 120  # Timeout for warm cache (async, not blocking)
COLD_CACHE_TIMEOUT_SECONDS = 120  # Timeout for cold cache (async, background)

# Individual API/Stage Timeouts
OPENMETEO_TIMEOUT_SECONDS = 15  # OpenMeteo API fetch
CPCB_API_TIMEOUT_SECONDS = 15  # CPCB/data.gov.in API fetch
LANDSAT_FETCH_TIMEOUT_SECONDS = 60  # Google Earth Engine Landsat8 batch fetch
SENTINEL_FETCH_TIMEOUT_SECONDS = 60  # Google Earth Engine Sentinel2 batch fetch
DUCKDB_WRITE_TIMEOUT_SECONDS = 30  # Individual DuckDB write operation

# ============================================================================
# CACHE CONFIGURATION (Redis)
# ============================================================================

REDIS_HOST = "localhost"  # Can be overridden by env var REDIS_HOST
REDIS_PORT = 6379  # Can be overridden by env var REDIS_PORT
REDIS_DB = 0  # Which Redis database to use
REDIS_TTL_SECONDS = 3600  # 1 hour TTL for cached predictions
REDIS_CACHE_KEY_PREFIX = "cache:"  # Prefix for cache keys (e.g., "cache:g00000001")
REDIS_CONNECTION_TIMEOUT = 5  # Seconds to wait for Redis connection
REDIS_SOCKET_KEEPALIVE = True  # Keep connections alive

# ============================================================================
# PERSISTENCE QUEUE CONFIGURATION
# ============================================================================

PERSISTENCE_QUEUE_MAX_SIZE = 10000  # Max queue size before blocking worker
PERSISTENCE_QUEUE_BATCH_SIZE = 100  # Predictions per DuckDB batch write
PERSISTENCE_QUEUE_BATCH_TIMEOUT_SECONDS = 1.0  # Flush batch after this timeout (seconds)
PERSISTENCE_QUEUE_WORKER_THREADS = 1  # Background threads for sequential DuckDB writes

# ============================================================================
# WORKER CONFIGURATION
# ============================================================================

# InferenceJob Worker Counts
NUM_INFERENCE_WORKERS_HOT = 1  # Workers for hot cache (synchronous)
NUM_INFERENCE_WORKERS_WARM = 2  # Workers for warm cache (async)
NUM_INFERENCE_WORKERS_COLD = 2  # Workers for cold cache (async, background)

# Grid Batch Sizes per Worker
GRID_BATCH_SIZE_HOT = 5000  # Grids per hot cache worker (synchronous)
GRID_BATCH_SIZE_WARM = 22500  # Grids per warm cache worker (45K / 2 workers)
GRID_BATCH_SIZE_COLD = 230000  # Grids split among cold workers

# Cache Tiers
HOT_GRID_COUNT = 5000  # Synchronous, must finish before returning to coordinator
WARM_GRID_COUNT = 45000  # Async, non-critical for immediate serving
COLD_GRID_COUNT = 459209  # Remaining grids, background, best effort

# Retry Configuration
MAX_WRITER_JOB_RETRIES = 1  # Retry WriterJob once if fails
MAX_HOT_CACHE_RETRIES = 2  # Retry hot cache up to 2 times (critical)
MAX_WARM_CACHE_RETRIES = 1  # Retry warm cache once (optional)
MAX_COLD_CACHE_RETRIES = 0  # Don't retry cold cache (best effort)

# ============================================================================
# API CREDENTIALS & ENDPOINTS
# ============================================================================

import os

# API Credentials (read from environment with fallbacks)
OPENMETEO_API_KEY = os.environ.get("OPENMETEO_API_KEY", "")  # Optional for some OpenMeteo endpoints
CPCB_API_KEY = os.environ.get("CPCB_API_KEY", "579b464db66ec23bdd0000017036fe535e9c47d146f13453e9cd266b")
GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT", None)
GEE_PROJECT = os.environ.get("GEE_PROJECT", "pollu-dev")

# API Endpoints
OPENMETEO_API_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
CPCB_API_ENDPOINT = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
GEE_API_ENDPOINT = "https://earthengine.googleapis.com"  # Auto-handled by geemap/ee

# CPCB API Parameters
CPCB_CITY_FILTER = "Mumbai"  # City to fetch PM2.5 for
CPCB_POLLUTANT_FILTER = "PM2.5"  # Pollutant ID
CPCB_API_LIMIT = 100  # Max records to fetch per request

# ============================================================================
# GOOGLE EARTH ENGINE CONFIGURATION
# ============================================================================

GEE_BATCH_SIZE = 1000  # Cells per batch request to GEE
GEE_MAX_WORKERS = 8  # Concurrent requests to GEE API
GEE_RETRY_ATTEMPTS = 3  # Retry failed GEE requests
GEE_LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1"  # Landsat8 collection ID
GEE_SENTINEL_COLLECTION = "COPERNICUS/S2_SR"  # Sentinel2 collection ID

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

# Table-to-file mapping
TABLE_TO_DB_PATH = {
    TABLE_GRID: DUCKDB_GRID_PATH,
    TABLE_PM25_OBSERVATION: DUCKDB_PM25_PATH,
    TABLE_METEO: DUCKDB_METEO_PATH,
    TABLE_LANDSAT8: DUCKDB_LANDSAT8_PATH,
    TABLE_SENTINEL2: DUCKDB_SENTINEL2_PATH,
    TABLE_PREDICTIONS: DUCKDB_PREDICTIONS_PATH,
}

def get_db_path_for_table(table_name: str):
    """Get DuckDB file path for a given table name."""
    if table_name not in TABLE_TO_DB_PATH:
        raise ValueError(f"Unknown table: {table_name}. Valid tables: {list(TABLE_TO_DB_PATH.keys())}")
    return TABLE_TO_DB_PATH[table_name]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Orchestrator & Worker Logging
ORCHESTRATOR_LOG_FILE = TAKSHASHILA_ROOT / "logs" / "orchestrator.log"
WORKER_LOG_DIR = TAKSHASHILA_ROOT / "logs" / "workers"
CYCLE_ID_LOG_PREFIX = True  # Include cycle_id in all logs
WORKER_LOG_FORMAT = "[%(cycle_id)s] [%(asctime)s] [%(levelname)s] %(message)s"
