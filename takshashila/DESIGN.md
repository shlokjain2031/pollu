# Takshashila Inference: Storage & Architecture Design Document

**Version**: 1.0  
**Date**: January 17, 2026  
**Module**: Inference Engine (Takshashila)  
**Scope**: Storage layer design, database schema, and inference I/O architecture

---

## Executive Summary

Takshashila is the inference persistence and serving layer for hyperlocal PM2.5 predictions at 30m resolution. This document defines:

- **Storage Architecture**: DuckDB-based prediction storage + parquet-based feature manifests
- **Database Schema**: Predictions table + feature source tables (Landsat, MODIS, Sentinel, OpenMeteo)
- **Query Patterns**: Two primary access patterns (heatmap batch + single-cell point queries)
- **Data Flow**: Manifest-driven feature fetching → feature assembly → model prediction → storage
- **Concurrency Model**: Lock-and-share deduplication for concurrent requests
- **Scaling Target**: 1000 concurrent users on local hardware (Dell Vostro 3400)

---

## 1. Architecture Overview

### 1.1 High-Level Data Flow

```
Request (Heatmap or Point API)
    ↓
Query Predictions DB
    ├─ Found + fresh (< 1 hour old) → Return immediately
    ├─ Found + stale → Show loading screen, trigger refresh
    └─ Not found → Show loading screen, compute fresh
    ↓
Feature Assembly (concurrent request deduplication)
    ├─ Lock on (city, timestamp) to deduplicate
    ├─ Fetch PM2.5 lags from pm25_target_table
    ├─ Fetch meteorology from openmeteo_table
    ├─ Fetch Landsat indices from landsat_table (via manifest)
    ├─ Fetch Sentinel indices from sentinel_table (via manifest)
    └─ Fetch MODIS indices from modis_table (via manifest)
    ↓
Feature Vector Construction
    ├─ Assemble 64 features in frozen order (via nalanda.feature_freeze)
    ├─ Validate schema hash matches training schema
    └─ Handle missing values gracefully
    ↓
Model Prediction
    ├─ Load LightGBM model (singleton, cached at startup)
    ├─ Call model.predict(feature_matrix)
    └─ Clip output to [0, 500] µg/m³
    ↓
Store to Predictions DB
    ├─ UPDATE or INSERT prediction row
    ├─ Record model_version, schema_hash, timestamp_of_computation
    └─ Release lock on (city, timestamp)
    ↓
Return to Client
    └─ Display fresh data or show confirmation popup
```

### 1.2 Design Principles

1. **Boring Inference**: Inference module only predicts; feature assembly and DB I/O are caller's responsibility (job scheduler or serving layer)
2. **Manifest-Driven**: All feature source paths and metadata tracked via manifest contracts; inference doesn't hardcode paths
3. **Lazy Computation**: Only compute predictions when requested; no pre-computation job scheduler (yet)
4. **Lock & Share**: Concurrent requests for same (city, timestamp) share single computation via mutex/lock
5. **Stale-OK**: Show loading until fresh data; don't serve inconsistent stale data
6. **Single Version**: Per (city, grid_id, timestamp), only one prediction exists; UPDATE replaces on refresh
7. **Local Hardware Optimized**: DuckDB + parquet on local SSD; no distributed DB overhead

---

## 2. Storage Layer Design

### 2.1 Database Choice: DuckDB

**Rationale:**
- **Concurrent reads**: Handles 1000 concurrent queries without file locking (vs Parquet which serializes)
- **Indexing**: Automatic B-tree indices on query patterns (vs Parquet full-scan)
- **Append-only writes**: Efficient INSERT/UPDATE for streaming predictions
- **Parquet native**: Can query parquet files directly via `FROM 'path/*.parquet'` without import
- **Zero setup**: Single-file database on SSD; no server overhead
- **Scales to 1000 users**: ~33 req/sec queries = negligible load on 4-core CPU

**Cost**: Zero infrastructure ($0/month beyond electricity). Compare: AWS RDS ~$120/month.

### 2.2 Physical Storage Layout

```
artifacts/
├── inference.duckdb                 [Main predictions database]
│
├── v1/
│   ├── predictions/                 [Parquet storage for archive/audit]
│   │   ├── mumbai/2024-01-17.parquet
│   │   └── ...
│   │
│   ├── features/                    [Raw features, tracked by manifest]
│   │   ├── landsat/
│   │   │   ├── mumbai_2024-01-16.parquet
│   │   │   └── manifest.json
│   │   ├── modis/
│   │   ├── sentinel/
│   │   └── openmeteo/
│   │
│   └── models/
│       ├── model.txt                [LightGBM binary]
│       ├── model_metadata.json      [Schema, hyperparams, version]
│       └── feature_schema.json      [Frozen feature list]
│
└── cache/
    └── feature_cache/               [In-memory + Redis, discussed separately]
```

**Note**: DuckDB tables are created as tables in `inference.duckdb`. Feature parquets are queried directly.

---

## 3. Database Schema

### 3.1 Predictions Table

**Purpose**: Store computed PM2.5 predictions for all grid cells at all timestamps.

**Table Name**: `predictions`

**Schema**:

```sql
CREATE TABLE predictions (
  -- Primary Key
  city TEXT NOT NULL,
  grid_id TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  
  -- Prediction Output
  pm25_pred FLOAT NOT NULL,                    -- Predicted PM2.5 (µg/m³), clipped to [0, 500]
  
  -- Metadata for Interpretation
  target_trust FLOAT,                          -- Trust score of ground truth (if computed with trust weights)
  closest_sensor_dist FLOAT,                   -- Distance to nearest CPCB/OpenAQ sensor (meters)
  closest_sensor_pm25 FLOAT,                   -- PM2.5 value of closest sensor at this timestamp
  last_active_timestamp_of_closest_sensor TIMESTAMP,  -- When that sensor last reported
  
  -- Computation Metadata
  timestamp_of_computation TIMESTAMP NOT NULL, -- When prediction was computed (for staleness tracking)
  model_version TEXT NOT NULL,                 -- e.g., "v1.0.0" (for reproducibility + rollback)
  schema_hash TEXT NOT NULL,                   -- Feature schema hash (e.g., "b11f698517e52cf6")
  
  -- Constraints
  PRIMARY KEY (city, grid_id, timestamp),
  CHECK (pm25_pred >= 0 AND pm25_pred <= 500),
  CHECK (timestamp_of_computation >= timestamp)  -- Sanity check: can't predict before time
);

-- Indices for Query Performance
CREATE INDEX idx_predictions_city_timestamp ON predictions(city, timestamp);
CREATE INDEX idx_predictions_city_grid_timestamp ON predictions(city, grid_id, timestamp);
```

**Key Design Decisions**:

1. **PRIMARY KEY = (city, grid_id, timestamp)**: Uniquely identifies a prediction. Only one prediction per grid cell per hour.
2. **timestamp_of_computation**: Enables staleness detection. Query: `NOW() - timestamp_of_computation > 3600 seconds = stale`.
3. **model_version + schema_hash**: Enable model governance and rollback. Can run multiple model versions in parallel.
4. **No soft deletes**: When refreshing, UPDATE the row; don't insert new version (simplicity).

**Row Size**: ~150 bytes per row (text fields are short: city ~20B, grid_id ~36B UUID, floats ~8B each).

**Growth Rate** (with hourly updates):
- 59k grid cells × 24 hours × 365 days = 517M rows/year
- 517M × 150B = ~78GB/year
- For 1 week: ~1.5M rows, ~225MB (fits easily in SSD)

**Retention Policy**: For now, no deletion. If storage becomes issue, implement rolling 30-day window (keep last month only).

---

### 3.2 Feature Source Tables

**Purpose**: Store raw features from each data source, tracked by manifest contracts.

**Strategy**: Each feature source (Landsat, MODIS, Sentinel, OpenMeteo) has its own table in DuckDB OR is queried as parquet files directly.

**Option A: Tables in DuckDB** (recommended for query performance)

```sql
-- Landsat Signals Table
CREATE TABLE landsat_signals (
  grid_id TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  ndvi FLOAT,
  evi FLOAT,
  ndbi FLOAT,
  mndwi FLOAT,
  bsi FLOAT,
  ibi FLOAT,
  cloud_mask FLOAT,
  days_since_landsat INT,
  
  PRIMARY KEY (grid_id, timestamp)
);

-- Sentinel EVI Signals Table
CREATE TABLE sentinel_evi_signals (
  grid_id TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  evi FLOAT,
  ndvi FLOAT,
  days_since_sentinel INT,
  
  PRIMARY KEY (grid_id, timestamp)
);

-- OpenMeteo Meteorology Table
CREATE TABLE openmeteo_meteo (
  grid_id TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  temperature FLOAT,
  relative_humidity FLOAT,
  wind_speed FLOAT,
  wind_direction FLOAT,
  pressure FLOAT,
  -- Lags (computed or stored)
  temperature_lag1h FLOAT,
  temperature_lag3h FLOAT,
  -- ... more lags
  
  PRIMARY KEY (grid_id, timestamp)
);

-- PM2.5 Target (Historical) Table
CREATE TABLE pm25_target (
  grid_id TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  pm25 FLOAT,
  trust_weight FLOAT,
  
  PRIMARY KEY (grid_id, timestamp)
);

-- Indices
CREATE INDEX idx_landsat_grid_timestamp ON landsat_signals(grid_id, timestamp);
CREATE INDEX idx_sentinel_grid_timestamp ON sentinel_evi_signals(grid_id, timestamp);
CREATE INDEX idx_meteo_grid_timestamp ON openmeteo_meteo(grid_id, timestamp);
CREATE INDEX idx_pm25_grid_timestamp ON pm25_target(grid_id, timestamp);
```

**Option B: Parquet Queries via Manifest** (lower memory, but slower queries)

```python
# Manifest specifies parquet paths
manifest = {
  "landsat_signals": {
    "paths": ["artifacts/v1/features/landsat/*.parquet"],
    "columns": ["grid_id", "timestamp", "ndvi", "evi", ...],
    "freshness_interval": "16_days"
  },
  "sentinel_evi_signals": {
    "paths": ["artifacts/v1/features/sentinel/*.parquet"],
    "columns": ["grid_id", "timestamp", "evi", "ndvi", ...],
    "freshness_interval": "5_days"
  }
}

# Query example:
# SELECT * FROM 'artifacts/v1/features/landsat/*.parquet' 
# WHERE grid_id = '5432' AND timestamp >= NOW() - INTERVAL 30 DAY
```

**Recommendation**: Use **Option A (tables in DuckDB)** for:
- Faster queries (indexed lookups vs full scan)
- Easier to manage (single DB file)
- Atomic updates (all features + prediction in one transaction)

Load features from parquets into DuckDB tables during initial setup or as background jobs. Manifest tracks parquet source locations (for audit, but tables are canonical).

---

### 3.3 No Feature Cache Table

**Decision**: Don't store a "feature_cache" table. Features are fetched fresh each time from source tables (Landsat, Sentinel, OpenMeteo, PM2.5).

**Rationale**:
- Only PM2.5 and OpenMeteo need real-time updates (hourly)
- Landsat/Sentinel are updated daily or less frequently (no cache staleness risk)
- Simplifies schema (fewer tables to manage)
- Manifest tracks freshness at source level

---

## 4. Query Patterns

### 4.1 Pattern 1: Heatmap (All Grid Cells at Timestamp)

**Use Case**: User requests heatmap for mumbai at 2024-01-17T10:00:00Z.

**Query**:
```sql
SELECT city, grid_id, timestamp, pm25_pred, timestamp_of_computation, 
       target_trust, closest_sensor_dist, model_version, schema_hash
FROM predictions
WHERE city = 'mumbai' 
  AND timestamp = '2024-01-17T10:00:00Z'
ORDER BY grid_id;
```

**Expected Result**: 59k rows (one per grid cell).

**Execution**:
1. Check index `idx_predictions_city_timestamp`
2. ~10ms lookup + ~100ms serialization = ~110ms total

**Staleness Detection**:
```python
for row in result:
  staleness = (NOW - row.timestamp_of_computation).total_seconds()
  if staleness > 3600:  # > 1 hour
    is_stale = True
  else:
    is_stale = False
```

**If Stale**: Show loading screen, trigger fresh computation (background job or synchronous call to inference).

---

### 4.2 Pattern 2: Point Query (Single Grid Cell, Latest)

**Use Case**: API endpoint: "What's PM2.5 at lat 19.0760, lng 72.8777 right now?"

**Preprocessing** (in serving layer):
```python
grid_id = serving.lat_lng_to_grid_id(lat=19.0760, lng=72.8777)  # e.g., "5432"
timestamp = NOW  # Current hour (rounded down to nearest hour)
```

**Query**:
```sql
SELECT pm25_pred, timestamp_of_computation, target_trust, closest_sensor_dist, model_version
FROM predictions
WHERE city = 'mumbai' 
  AND grid_id = '5432' 
  AND timestamp = '2024-01-17T10:00:00Z'
LIMIT 1;
```

**Expected Result**: 1 row.

**Execution**:
1. Check index `idx_predictions_city_grid_timestamp`
2. ~1ms lookup

**If Not Found**: Compute fresh (inference → update DB → return).

---

## 5. Feature Freshness Tiers

**Decision**: Different features refresh at different intervals. Inference always fetches from DB; if data isn't fresh enough, defer to job scheduler (or error out gracefully).

| Feature Source | Interval | Freshness SLA | Action if Stale |
|---|---:|---|---|
| **PM2.5 Lags** (target) | Hourly | Must be < 2 hours old | Wait or error |
| **OpenMeteo Meteo** | Hourly | Must be < 3 hours old | Fallback to climatology or error |
| **Landsat Signals** | 16 days | Can be up to 30 days old | Graceful degrade (mark as missing) |
| **Sentinel EVI** | 5 days | Can be up to 14 days old | Graceful degrade (mark as missing) |
| **MODIS AOD** | Daily | Can be up to 7 days old | Graceful degrade (mark as missing) |

**Implementation**:

```python
def check_feature_freshness(manifest, feature_source, now):
  """Check if feature data is fresh enough."""
  last_update = manifest[feature_source]["last_update"]
  freshness_interval = manifest[feature_source]["freshness_interval"]
  
  age = now - last_update
  
  if age > freshness_interval * 2:  # Very stale
    return "STALE_CRITICAL"
  elif age > freshness_interval:    # Somewhat stale
    return "STALE_WARNING"
  else:
    return "FRESH"
```

**Policy**:
- FRESH: Use data directly
- STALE_WARNING: Use data but log warning + mark in prediction metadata
- STALE_CRITICAL: For critical sources (PM2.5, meteo) → error out; for optional (Landsat, Sentinel) → fill NaN

---

## 6. Concurrent Request Deduplication

### 6.1 Problem

```
Timeline:
T0: Request A: "Give me heatmap for mumbai at T10"
T0.1ms: Request B: "Give me heatmap for mumbai at T10"
T0.2ms: Request C: "Give me point for mumbai grid 5432 at T10"

Without deduplication:
- Request A fetches all 59k features, computes 59k predictions (10s)
- Request B also fetches all 59k features, computes 59k predictions (10s) ← wasteful!
- Request C fetches 1 cell features, computes 1 prediction (1s) ← OK

With deduplication:
- Request A acquires lock on (mumbai, T10), starts compute
- Request B waits on same lock
- Request C: Lower priority, waits for heatmap to finish (prioritize batch over point)
- A finishes → releases lock, B wakes and reads from DB (instant)
```

### 6.2 Implementation

**Data Structure**:
```python
import threading

# Global state
compute_locks = {}  # (city, timestamp) → threading.Lock()
compute_in_progress = set()  # {(city, timestamp), ...}
compute_results = {}  # (city, timestamp) → result

def compute_with_dedup(city, timestamp, request_type="heatmap"):
  """
  request_type: "heatmap" or "point"
  Heatmap requests have higher priority.
  """
  key = (city, timestamp)
  
  # Acquire lock (create if doesn't exist)
  if key not in compute_locks:
    compute_locks[key] = threading.Lock()
  
  lock = compute_locks[key]
  
  # Wait for lock (if another request is computing)
  with lock:
    # Check if result is ready
    if key in compute_results:
      return compute_results[key]
    
    # Mark as in progress
    if key not in compute_in_progress:
      compute_in_progress.add(key)
      
      # Fetch features
      features = fetch_all_features(city, timestamp)
      
      # Predict
      predictions = model.predict(features)
      
      # Store
      store_predictions(city, timestamp, predictions)
      
      # Cache result
      compute_results[key] = predictions
      compute_in_progress.remove(key)
    else:
      # Already computing, wait for completion
      return compute_results[key]
  
  return compute_results[key]
```

**Key Properties**:
1. **Atomic**: Only one request computes per (city, timestamp).
2. **Blocking**: Concurrent requests wait for first to complete, then read from DB.
3. **Memory efficient**: Don't cache large results; just cache in DB and other requests query it.
4. **Timeout**: Set lock timeout (e.g., 30s) to avoid deadlock if computation fails.

---

## 7. Error Handling & Graceful Degradation

| Scenario | Action |
|---|---|
| **PM2.5 lags missing** | Return error; can't predict without historical target |
| **Meteo missing** | Fallback: Use climatological average (pre-computed mean temp/humidity/wind for month/day) |
| **Landsat missing** | Fill feature with NaN; model was trained on missing values (or use 0) |
| **Sentinel missing** | Fill feature with NaN |
| **Prediction fails** | Return error with staleness metadata; show loading screen indefinitely |
| **Model corrupt** | Return 500 error; admin must fix |
| **DB down** | Return stale prediction from cache (if available) or error |

**Implementation**:
```python
def predict(city, grid_id, timestamp):
  try:
    # Check freshness
    pm25_freshness = check_feature_freshness("pm25_target", timestamp)
    if pm25_freshness == "STALE_CRITICAL":
      raise ValueError("PM2.5 data too stale")
    
    # Fetch features with fallbacks
    features = {}
    features["pm25_lags"] = fetch_pm25_lags(city, grid_id, timestamp)
    
    try:
      features["meteo"] = fetch_meteo(city, grid_id, timestamp)
    except:
      features["meteo"] = fetch_climatological_meteo(city, grid_id, timestamp)
    
    try:
      features["landsat"] = fetch_landsat(city, grid_id, timestamp)
    except:
      features["landsat"] = [NaN] * 9  # 9 landsat features
    
    try:
      features["sentinel"] = fetch_sentinel(city, grid_id, timestamp)
    except:
      features["sentinel"] = [NaN] * 2  # 2 sentinel features
    
    # Assemble + predict
    feature_vector = assemble_features(features)
    pm25 = model.predict(feature_vector)
    
    return pm25
    
  except Exception as e:
    logger.error(f"Prediction failed: {e}")
    raise PredictionError(f"Unable to compute prediction: {e}")
```

---

## 8. Concurrent User Scaling: 1000 Users

### 8.1 Request Rate Estimation

```
1000 concurrent users on Dell Vostro (4-core CPU):
- Heatmap requests: ~2/hour per user (check every 30 min)
- Point API requests: ~5/hour per user (realtime queries)
- Total: ~7000 requests/hour = ~2 req/sec

DuckDB can easily handle 2 req/sec (typical latency: 1-10ms per query).
Network: ~1000 heatmap responses/hour × 59k cells × 50B/cell = ~3GB/hour (manageable)
```

### 8.2 Memory Budget

```
Dell Vostro 3400 typical: 8-16GB RAM

Go serving layer:
  - HTTP server: ~50MB
  - 1000 concurrent connections (polling): ~1MB (negligible)
  - Model (LightGBM in memory): ~100MB
  - Feature cache (24h of hot features): ~500MB
  - DuckDB buffer pool: ~1GB (configurable)
  - Python inference subprocess: ~200MB
  
  Total: ~2GB ← Fits in 8GB, plenty of headroom
```

### 8.3 CPU Budget

```
Per request (heatmap, all 59k cells):
  - Feature fetch: 3-5s (I/O-bound, not CPU)
  - Prediction: 0.5s (CPU-bound, LightGBM vectorized)
  - DB update: 0.5s (I/O-bound)
  - Deduplication: 1-2 req/sec max (negligible CPU overhead)
  
  Total: ~5-7s per heatmap request

With 2 req/sec:
  - Heatmap: 1 req/10s × 5s = 0.5s active CPU per second
  - Point: 1.5 req/sec × 0.2s = 0.3s active CPU per second
  - Total: ~0.8s active CPU per second / 4 cores = 20% CPU usage ← Good headroom
```

---

## 9. Deployment & Initialization

### 9.1 DuckDB Schema Setup

```python
# scripts/init_inference_db.py
import duckdb

conn = duckdb.connect("artifacts/inference.duckdb")

# Create predictions table
conn.execute("""
CREATE TABLE IF NOT EXISTS predictions (
  city TEXT NOT NULL,
  grid_id TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  pm25_pred FLOAT NOT NULL,
  target_trust FLOAT,
  closest_sensor_dist FLOAT,
  closest_sensor_pm25 FLOAT,
  last_active_timestamp_of_closest_sensor TIMESTAMP,
  timestamp_of_computation TIMESTAMP NOT NULL,
  model_version TEXT NOT NULL,
  schema_hash TEXT NOT NULL,
  PRIMARY KEY (city, grid_id, timestamp),
  CHECK (pm25_pred >= 0 AND pm25_pred <= 500)
);
""")

# Create indices
conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_city_timestamp ON predictions(city, timestamp);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_city_grid_timestamp ON predictions(city, grid_id, timestamp);")

# Create feature tables (from manifest)
manifest = load_manifest("artifacts/v1/manifest.json")
for source in ["landsat_signals", "sentinel_evi_signals", "openmeteo_meteo", "pm25_target"]:
  conn.execute(f"CREATE TABLE IF NOT EXISTS {source} AS SELECT * FROM '{manifest[source]['path']}' LIMIT 0;")

conn.close()
```

### 9.2 Go Serving + Python Inference

**Architecture**:
```
Go binary (serves heatmap + point API on :8080)
  ↓ HTTP calls to
Python inference service (:5000)
  ↓ reads/writes
DuckDB (artifacts/inference.duckdb)
  ↓ reads
Feature tables (landsat, sentinel, meteo, pm25_target)
```

**Startup**:
```bash
# Initialize DB schema
python scripts/init_inference_db.py

# Start Python inference service
python -m serving.inference_service --port 5000 &

# Start Go serving
go run serving/main.go --port 8080
```

---

## 10. Future Extensions (Out of Scope)

1. **Job Scheduler**: Periodically refresh predictions (hourly/daily).
2. **Redis Cache**: Shared cache layer for multi-instance deployments.
3. **WebSocket Push**: Real-time updates instead of polling.
4. **Model Versioning**: Run multiple models in parallel (A/B testing).
5. **Feature Store**: DuckDB → proper feature store (Tecton, Feast).
6. **Distributed Inference**: Scale to multiple servers/cities.

---

## 11. Summary Table: Design Decisions

| Component | Decision | Rationale |
|---|---|---|
| **Storage** | DuckDB | Concurrent reads, indexing, zero setup |
| **Predictions Table** | (city, grid_id, timestamp) PRIMARY KEY | Single version per prediction |
| **Features** | Separate tables per source (Landsat, Sentinel, etc.) | Manifest-driven, independent refresh intervals |
| **Query Patterns** | (city, timestamp) heatmap + (city, grid_id, timestamp) point | Both optimized via indices |
| **Concurrency** | Lock-and-share deduplication | Single compute per (city, timestamp), others wait |
| **Staleness** | Show loading screen, no stale data shown | Simple UX, avoids confusion |
| **Retention** | No deletion (for now) | Keep full audit trail; delete later if needed |
| **Hardware** | Local Dell SSD | Cost savings vs cloud (~$90/month) |
| **Scaling** | 1000 concurrent users on 4-core CPU | ~20% CPU, ~2GB RAM utilization |

---

## 12. Appendix: Example Queries

### Query: Get Latest Prediction for Grid Cell

```sql
SELECT pm25_pred, timestamp_of_computation 
FROM predictions 
WHERE city = 'mumbai' 
  AND grid_id = '5432'
  AND timestamp >= NOW() - INTERVAL 1 HOUR
ORDER BY timestamp DESC 
LIMIT 1;
```

### Query: Check Staleness

```python
result = conn.execute("""
  SELECT pm25_pred, timestamp_of_computation 
  FROM predictions 
  WHERE city = 'mumbai' AND timestamp = $1
  LIMIT 1
""", [timestamp]).fetchone()

if result:
  staleness_seconds = (datetime.now() - result['timestamp_of_computation']).total_seconds()
  is_stale = staleness_seconds > 3600
else:
  is_stale = None  # Not found
```

### Query: Bulk Insert Fresh Predictions

```python
import pandas as pd

predictions_df = pd.DataFrame({
  'city': ['mumbai'] * 59000,
  'grid_id': grid_ids,
  'timestamp': [target_timestamp] * 59000,
  'pm25_pred': predictions,
  'target_trust': trust_scores,
  'closest_sensor_dist': distances,
  'closest_sensor_pm25': sensor_values,
  'last_active_timestamp_of_closest_sensor': sensor_timestamps,
  'timestamp_of_computation': [now] * 59000,
  'model_version': [model_version] * 59000,
  'schema_hash': [schema_hash] * 59000
})

conn.execute("""
  INSERT INTO predictions SELECT * FROM df
  ON CONFLICT (city, grid_id, timestamp) DO UPDATE SET
    pm25_pred = EXCLUDED.pm25_pred,
    timestamp_of_computation = EXCLUDED.timestamp_of_computation,
    model_version = EXCLUDED.model_version,
    schema_hash = EXCLUDED.schema_hash
""")
```

---

**Document End**
