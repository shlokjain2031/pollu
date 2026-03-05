# Takshashila — Inference & Serving

Takshashila is the inference persistence and serving layer for hyperlocal PM2.5 predictions at 30m resolution across 509,209 grid cells for Mumbai.

---

## Architecture overview

Two subsystems work together:

1. **Core (Python)** — single-threaded inference engine  
   Feature assembly → LightGBM prediction → DuckDB persistence

2. **Distributed (Go coordinator + Python workers)** — batch hourly inference  
   APScheduler triggers WriterJob (data fetch) → InferenceJob (batch predict) → Redis cache population

```
Hourly cycle:
  t=-2m  WriterJob spawned (fetch OpenMeteo, CPCB PM2.5, GEE satellites)
  t=-1m  WriterJob completes, data in DuckDB
  t= 0m  InferenceJob spawned (assemble features → predict → cache)
  t=+5m  InferenceJob completes, Redis/DuckDB populated
```

On-demand inference is served by the Go coordinator via HTTP (`/predict`), which deduplicates concurrent identical requests (leader/follower pattern) and checks a 3-tier cache (hot/warm/cold) before dispatching to Python workers.

---

## Directory structure

```
takshashila/
├── core/                         # Python inference engine
│   ├── config.py                 # Centralized config: paths, model metadata, timeouts
│   ├── schema.py                 # DuckDB table definitions (canonical)
│   ├── errors.py                 # Error hierarchy (InferenceError subtypes)
│   ├── utils.py                  # Timestamp normalization, logging, grid validation
│   ├── cache.py                  # Redis cache + LocalMemoryCache fallback
│   ├── db_pool.py                # DuckDB connection pooling (PoolManager)
│   ├── feature_assembler.py      # Feature assembly: PM2.5 lags, meteo, satellite indices
│   ├── inference.py              # InferenceEngine: orchestrates predict pipeline
│   ├── model_predictor.py        # LightGBM model loading + prediction
│   ├── persistence.py            # DuckDB read/write + PersistenceQueue async writer
│   └── requirements.txt          # Core Python dependencies
│
├── distributed/
│   ├── coordinator/              # Go HTTP coordinator
│   │   ├── main.go               # HTTP handlers, orchestrator, Redis/DuckDB integration
│   │   ├── config.go             # Coordinator config
│   │   ├── errors.go             # Typed prediction errors with HTTP status codes
│   │   ├── queue.go              # JobQueue bounded channel
│   │   ├── cache.go              # ResultCache with TTL
│   │   ├── dedup.go              # Leader/follower deduplication
│   │   ├── health.go             # Worker health checking
│   │   ├── python_client.go      # Python worker process management
│   │   ├── metrics.go            # Prometheus metrics
│   │   ├── prediction_cache.go   # Prediction cache with async writes
│   │   ├── go.mod / go.sum       # Go module dependencies
│   │   ├── Makefile              # Build commands
│   │   └── *_test.go             # Go test files
│   │
│   └── worker/                   # Python workers
│       ├── orchestrator.py       # Flask app + APScheduler (WriterJob/InferenceJob scheduling)
│       ├── worker.py             # Phase 2a stdin/stdout JSON worker
│       ├── write_job_worker.py   # Data fetchers: MeteoFetcher, LandsatFetcher, Sentinel2Fetcher, PM25Fetcher
│       ├── inference_job_worker.py  # Batch inference with caching + PersistenceQueue
│       └── config.py             # Worker-specific config
│
├── scripts/                      # Utility scripts
│   ├── init_duckdb.py            # Initialize DuckDB schema + load grid
│   ├── inspect_duckdb.py         # Inspect DuckDB file contents
│   └── populate_mock_features.py # Generate mock test data
│
├── docs/                         # Design documents
│   ├── DESIGN.md                 # Storage & architecture design
│   ├── TECHNICAL_DESIGN.md       # Batch inference scheduler design
│   ├── CACHE_STRATEGY.md         # 3-tier caching strategy
│   ├── DEDUPLICATION.md          # Leader/follower dedup design
│   ├── ERROR_HANDLING.md         # Error taxonomy and handling
│   ├── SCALING.md                # Scaling assessment
│   └── ...                       # Additional design docs
│
└── tests/                        # Python test suite
    ├── conftest.py               # Shared fixtures
    └── test_*.py                 # Unit/integration tests
```

---

## Storage layer

DuckDB with one file per table:

| File | Table | Content |
|------|-------|---------|
| `grid.duckdb` | `grid` | 509,209 Mumbai 30m cells (grid_id, lat, long) |
| `pm25_observations.duckdb` | `pm25_observation` | Hourly PM2.5 sensor readings |
| `meteo.duckdb` | `meteo` | Hourly meteorology (8 variables) |
| `landsat8.duckdb` | `landsat8` | Landsat-8 spectral bands B1-B11 + scene_cloud_cover |
| `sentinel2.duckdb` | `sentinel2` | Sentinel-2 bands B2, B4, B8, AOT, QA10 |
| `predictions.duckdb` | `predictions` | Model predictions (pm25_prediction, model_version, schema_hash) |

Canonical schema defined in `core/schema.py`. All tables use `PRIMARY KEY (grid_id, timestamp)`.

---

## Model

- **Algorithm**: LightGBM
- **Features**: 64 (frozen schema from `dwarka/models/prod_v1/feature_schema.json`)
- **Model path**: `dwarka/models/prod_v1/model.txt`
- **Version**: `prod_v1`
- **Output**: PM2.5 µg/m³ prediction per (grid_id, timestamp_hour)

---

## Data sources

| Source | API | Frequency | Fetcher |
|--------|-----|-----------|---------|
| Meteorology | Open-Meteo Archive | Hourly | `MeteoFetcher` in `distributed/worker/write_job_worker.py` |
| PM2.5 ground truth | WAQI API | Hourly | `PM25Fetcher` in `distributed/worker/write_job_worker.py` |
| Landsat-8 imagery | Google Earth Engine | ~16-day revisit | `LandsatFetcher` in `distributed/worker/write_job_worker.py` |
| Sentinel-2 imagery | Google Earth Engine | ~5-day revisit | `Sentinel2Fetcher` in `distributed/worker/write_job_worker.py` |

All fetchers run sequentially in `write_job_worker.py main()`.

---

## Inference flow

1. **Feature assembly** (`core/feature_assembler.py`):
   - PM2.5 lag features (t-1h through t-168h)
   - Meteorological features + derived (VPD, wind components)
   - Satellite spectral indices (NDVI, NDBI, NDWI, etc.)
   - Spatial features (lat, long, grid metadata)
   - Returns 1×64 DataFrame in frozen feature order

2. **Model prediction** (`core/model_predictor.py`):
   - LightGBM `model.predict()` on assembled features
   - Validates shape (1 row, 64 features) and no NaN/inf

3. **Persistence** (`core/persistence.py`):
   - Writes prediction to `predictions.duckdb`
   - Single-writer semantics via threading lock
   - PersistenceQueue for async batch writes in distributed mode

---

## Cache strategy

- **Redis** (primary): 3-tier TTL (hot 5min, warm 1h, cold 24h)
- **LocalMemoryCache** (fallback): in-process dict when Redis unavailable
- **DuckDB** (persistence): source of truth for all predictions

---

## Running

```bash
# Initialize DuckDB schema
python3 takshashila/scripts/init_duckdb.py

# Populate mock data for testing
python3 takshashila/scripts/populate_mock_features.py

# Start orchestrator (Flask + APScheduler)
python3 -m takshashila.distributed.worker.orchestrator

# Build Go coordinator
cd takshashila/distributed/coordinator && make build

# Run Go coordinator
./takshashila-coordinator
```
