# CLAUDE.md — Takshashila

Agent operating guide for the `takshashila/` module. This document is for coding agents only.

---

## 1) Scope

Takshashila is the inference and serving layer. It has two subsystems:

- `core/` — Python inference engine (feature assembly, model prediction, DuckDB persistence)
- `distributed/` — Go coordinator + Python workers for batch hourly inference

---

## 2) Invariants

### Path naming
**Never output `kashi/` or `kashi` as a path prefix.** The module is called `takshashila`. All paths must use `takshashila/core/`, `takshashila/distributed/worker/`, or `takshashila/distributed/coordinator/`. If any design doc or comment still references `kashi/`, treat it as stale and mentally substitute `distributed/worker/`.

### Schema source of truth
`core/schema.py` is the canonical schema definition. If any other file (worker scripts, design docs) creates tables with different column types, constraints, or table names, `core/schema.py` wins.

### Feature order
The 64-feature schema is frozen. Feature order must match `dwarka/models/prod_v1/feature_schema.json`. Do not reorder, add, or remove features without explicit request.

### Execution model
All fetchers in `distributed/worker/write_job_worker.py` run **sequentially**. Do not describe or document them as parallel. The `main()` function calls each fetcher one after another.

### Model identity
The production model is `dwarka/models/prod_v1/model.txt` (LightGBM). Version string is `prod_v1`. Do not reference other version strings.

---

## 3) Mandatory read order

Before making any change to takshashila:

1. This file (`takshashila/CLAUDE.md`)
2. `takshashila/README.md`
3. `takshashila/core/config.py`
4. `takshashila/core/schema.py`
5. `takshashila/core/feature_assembler.py`
6. `takshashila/core/inference.py`
7. `takshashila/distributed/worker/orchestrator.py`
8. `takshashila/distributed/worker/write_job_worker.py`

---

## 4) Forbidden paths

Do not read or modify:
- `takshashila/tests/` — test suite is not in scope for documentation or refactoring
- `takshashila/distributed/coordinator/*_test.go` — Go test files
- `takshashila/duckdb/` — runtime DuckDB data files
- `logs/`, `artifacts/`, `__pycache__/`

---

## 5) File map

### 5.1 Core (Python inference engine)

#### `core/config.py`
Centralized configuration constants.
- Project paths (model, grid parquet, DuckDB directory).
- Model metadata: `MODEL_VERSION`, `SCHEMA_HASH`, `N_FEATURES`.
- Timeout budgets per inference phase.
- Redis config, worker counts, batch sizes.
- Orchestrator scheduling parameters.

#### `core/schema.py`
Canonical DuckDB table definitions.
- Six tables: `grid`, `pm25_observation`, `meteo`, `landsat8`, `sentinel2`, `predictions`.
- Each table created in its own `.duckdb` file.
- `initialize_schema()` creates all tables and loads grid parquet.
- All tables have `PRIMARY KEY (grid_id, timestamp)`.

#### `core/errors.py`
Error hierarchy for the inference system.
- `InferenceError` (base) → `FeatureAssemblyError`, `ModelInferenceError`, `PersistenceError`, `TimeoutError`, `InvalidRequestError`.
- Each error carries structured context (grid_id, timestamp, reason).
- Note: `TimeoutError` shadows the Python builtin.

#### `core/utils.py`
Timestamp normalization and validation utilities.
- `round_to_hour(dt)`: floors datetime to hour boundary, ensures UTC.
- `floor_to_hour(dt)`: alias for `round_to_hour`.
- `setup_logging()`: configures structured logging.
- `validate_grid_id_format(grid_id)`: checks `gNNNNNNNN` pattern.

#### `core/cache.py`
Redis cache client with local fallback.
- `RedisCacheClient`: connects to Redis, stores prediction results.
- `LocalMemoryCache`: in-process dict fallback when Redis unavailable.
- Retry logic with exponential backoff on Redis failures.

#### `core/db_pool.py`
DuckDB connection pooling.
- `DuckDBConnectionPool`: thread-safe pool for a single DuckDB file.
- `PoolManager`: manages pools for all 5 DuckDB files (grid, pm25, meteo, landsat8, sentinel2) + predictions.
- `WriterConnection`: dedicated writer connection with single-writer lock.

#### `core/feature_assembler.py`
Feature assembly from raw DuckDB data into 64 model features.
- `FeatureAssembler` class with `fetch_features(grid_id, timestamp)` and `fetch_features_batch()`.
- PM2.5 lag features: t-1h through t-168h (7 days).
- Meteorological features: raw + derived (VPD, wind components, rolling stats).
- Satellite features: spectral indices (NDVI, NDBI, NDWI, etc.) from Landsat-8 and Sentinel-2.
- Spatial features: latitude, longitude, grid metadata.
- Returns DataFrame with columns in frozen feature order.

#### `core/inference.py`
Main inference engine orchestrator.
- `InferenceEngine.predict(grid_id, timestamp)`: full pipeline (assemble → predict → persist).
- Initializes FeatureAssembler, ModelPredictor, PersistenceLayer.
- Handles cache-before-compute pattern.

#### `core/model_predictor.py`
LightGBM model loading and prediction.
- `ModelPredictor`: lazy-loads model from disk.
- `predict(features_df)`: validates shape (1×64), no NaN/inf, returns float PM2.5.
- `predict_batch()`: batch prediction support.

#### `core/persistence.py`
DuckDB persistence with single-writer semantics.
- `PersistenceLayer`: read/write predictions, grid existence checks.
- `PersistenceQueue`: async batch writer for distributed mode.
- Single-writer lock ensures serialized writes.

### 5.2 Distributed — Worker (Python)

#### `distributed/worker/orchestrator.py`
Flask app with APScheduler for hourly job scheduling.
- Scheduling: WriterJob @ t=-2m, InferenceJob @ t=0m every hour.
- Flask endpoints: `/inference` (on-demand), `/health`, `/test/trigger-writer-job`, `/test/trigger-inference-job`.
- Spawns `write_job_worker.py` and `inference_job_worker.py` as subprocesses.

#### `distributed/worker/write_job_worker.py`
Data fetcher classes (all run sequentially in `main()`):
- `MeteoFetcher`: Open-Meteo API → `meteo.duckdb`. Tile-level data broadcast to grid cells.
- `LandsatFetcher`: Google Earth Engine → `landsat8.duckdb`. Bands B1-B11 + scene_cloud_cover.
- `Sentinel2Fetcher`: Google Earth Engine → `sentinel2.duckdb`. Bands B2, B4, B8, AOT, QA10.
- `PM25Fetcher`: WAQI API → `pm25_observation.duckdb`. IDW interpolation to grid cells.

#### `distributed/worker/inference_job_worker.py`
Batch inference over grid cells.
- Processes grid cells in configurable batch sizes.
- Uses FeatureAssembler + ModelPredictor from core.
- Writes predictions via PersistenceQueue.
- Populates Redis cache after successful predictions.

#### `distributed/worker/worker.py`
Phase 2a worker: reads JSON from stdin, runs inference, writes JSON to stdout.
- Used by Go coordinator to dispatch individual predictions.

#### `distributed/worker/config.py`
Worker-specific configuration paths.

### 5.3 Distributed — Coordinator (Go)

#### `distributed/coordinator/main.go`
Go HTTP coordinator.
- `/predict` endpoint for on-demand inference.
- Manages Python worker processes via `python_client.go`.
- Redis + DuckDB integration for caching and persistence.

#### `distributed/coordinator/dedup.go`
Leader/follower deduplication for concurrent identical requests.
- First request for a (grid_id, timestamp) becomes leader and computes.
- Subsequent identical requests become followers and wait for the leader's result.

#### `distributed/coordinator/cache.go`
In-memory result cache with TTL expiry.

#### `distributed/coordinator/errors.go`
Typed prediction errors with HTTP status codes, retryability, and cacheability attributes.

#### `distributed/coordinator/health.go`
Worker health checking. Note: `waitForHealthResponse` is currently a stub.

#### `distributed/coordinator/metrics.go`
Prometheus metrics for request latency, cache hit rates, worker health, queue depth.

#### `distributed/coordinator/prediction_cache.go`
Prediction cache with async DB write-through. Note: `simulateDBWrite` is a placeholder.

### 5.4 Scripts

#### `scripts/init_duckdb.py`
Creates all DuckDB files with schema from `core/schema.py` and loads grid parquet.

#### `scripts/inspect_duckdb.py`
Read-only inspector: prints schema, row counts, and sample data for all DuckDB files.

#### `scripts/populate_mock_features.py`
Generates mock observational data for testing. Configurable grid count via `NUM_GRIDS` env var.

---

## 6) DuckDB table reference

| Table | File | Key columns |
|-------|------|------------|
| `grid` | `grid.duckdb` | grid_id (PK), lat, long |
| `pm25_observation` | `pm25_observations.duckdb` | grid_id, timestamp (PK), pm25 |
| `meteo` | `meteo.duckdb` | grid_id, timestamp (PK), temperature_2m, relative_humidity_2m, dewpoint_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation, cloud_cover |
| `landsat8` | `landsat8.duckdb` | grid_id, timestamp (PK), B1-B11, scene_cloud_cover |
| `sentinel2` | `sentinel2.duckdb` | grid_id, timestamp (PK), B2, B4, B8, AOT, QA10 (INTEGER) |
| `predictions` | `predictions.duckdb` | grid_id, timestamp_hour (PK), pm25_prediction, model_version, schema_hash |

---

## 7) Change discipline

- Make minimal targeted edits.
- Preserve interfaces unless explicitly asked to change them.
- If a file creates DuckDB tables, its schema must match `core/schema.py`.
- Do not document or reference test files.
- Do not output `kashi` or `kashi/` as a module path under any circumstance.
- If design docs still reference `kashi/`, ignore those paths and use the actual `takshashila/` paths.
- When describing data fetcher execution, always state **sequential** (not parallel).
