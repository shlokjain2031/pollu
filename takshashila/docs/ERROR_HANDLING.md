# Takshashila: Error Handling & Graceful Degradation

**Date**: January 18, 2026  
**Status**: Final Design (Locked)  
**Audience**: Backend engineers, DevOps, on-call team

---

## Executive Summary

This document specifies error handling, recovery strategies, and graceful degradation policies for the Takshashila inference system. The system prioritizes **availability with hard guarantees on feature presence**: every request must have complete features (no missing columns), but features may be stale. When data is unavailable, the system fails fast with clear error codes rather than attempting to recover via fallbacks.

**Key Design Principles**:
1. **All features are hard requirements** — cannot omit a feature; must be present (even if stale)
2. **Stale data acceptable** — feature staleness is tolerable; feature absence is not
3. **Fail fast over fallback** — prefer explicit errors (503) over silent degradation
4. **Atomicity via versioning** — immutable, versioned prediction tables prevent partial state exposure
5. **Redis pub/sub for notifications** — stale predictions trigger async notifications for polling clients

---

## 1. Feature Criticality Matrix

**Definition**: All 64 features are **CRITICAL (Tier 1)** — a request cannot proceed without any single feature.

| Feature Group | Count | Criticality | Staleness Tolerance | Missing Data Action |
|---|---|---|---|---|
| **PM2.5 Lags** (1h, 2h, 3h, 6h, 12h, 24h) | 6 | CRITICAL | 24h | IDW interpolation; fail loudly if zero sensors |
| **MeteoBase** (temp, humidity, dewpoint, pressure, wind, cloud) | 18 | CRITICAL | 1h | Fetch from OpenMeteo cache; fail if missing |
| **MeteoLags** (1h, 3h, 6h, 12h, 24h across meteo) | 20 | CRITICAL | 12h | Fetch from OpenMeteo cache; fill NaN if unavailable |
| **Satellite** (NDVI, MNDWI, NDBI, BSI, IBI, EVI, cloud, light, SI) | 9 | CRITICAL | 16d | Fetch from cache; fill NaN if >30d stale |
| **Temporal** (days_since_landsat, days_since_sentinel, lat, lon) | 4 | CRITICAL | ∞ | Always available; compute from metadata |

**Behavior**:
- Request proceeds **only if all 64 features present** (no omissions)
- Stale features are acceptable (age ≤ staleness tolerance)
- If any feature missing → return **400 Bad Request** or **503 Service Unavailable** (see API Contracts)

---

## 2. PM2.5 Lag Computation (IDW Strategy)

**Decision**: All PM2.5 lags computed via **Inverse Distance Weighting (IDW)** across available sensors.

### 2.1 IDW Interpolation

For a grid cell at (lat, lon) and timestamp T - lag_offset:

```
pm25_lag_value = IDW(
  grid_point=(lat, lon),
  sensor_readings=[
    (sensor_id, pm25_value, distance) for sensor in available_sensors
  ],
  power=2.0  # Standard IDW power (tunable)
)
```

**Algorithm**:
```python
def compute_pm25_lag(grid_id: str, lag_hours: int, timestamp: datetime, db: DuckDB) -> float:
  """
  Compute PM2.5 lag at timestamp - lag_hours using IDW interpolation.
  Fails loudly (raises exception) if zero sensor readings available.
  """
  target_timestamp = timestamp - timedelta(hours=lag_hours)
  
  # Fetch all available sensor readings at target timestamp (within ±30min window)
  sensor_readings = db.execute(f"""
    SELECT 
      sensor_id, 
      pm25_value, 
      ST_DISTANCE(geom, grid_geom) as distance
    FROM pm25_observations
    WHERE timestamp BETWEEN '{target_timestamp - 30min}' AND '{target_timestamp + 30min}'
      AND pm25_value IS NOT NULL
  """)
  
  if len(sensor_readings) == 0:
    raise SensorDataUnavailableError(
      f"Zero sensor readings at {target_timestamp}; cannot compute PM2.5 lag for grid {grid_id}"
    )
  
  # IDW interpolation
  weights = [1.0 / (distance ** 2) for _, _, distance in sensor_readings]
  weighted_values = [value * weight for (_, value, _), weight in zip(sensor_readings, weights)]
  pm25_lag = sum(weighted_values) / sum(weights)
  
  return pm25_lag
```

### 2.2 Failure Mode: No Sensor Data

**Trigger**: Zero sensor readings within ±30min window of target timestamp.

**Action**:
1. Log error: `"Zero sensors at {timestamp}; cannot compute PM2.5 lag"`
2. **Raise exception** (do not return NaN or city-level fallback)
3. Propagate to caller (Inference service)
4. Inference returns **503 Service Unavailable** (see API Contracts)

**Rationale**: City-level aggregates destroy spatial signal; better to fail loudly.

---

## 3. Feature Assembly Error Handling

### 3.1 Missing Feature Detection

Before calling model.predict(), validate schema:

```python
def validate_feature_vector(features: dict, expected_schema: list[str]) -> bool:
  """
  Check that all 64 required features are present (no NaN in critical columns).
  """
  missing = set(expected_schema) - set(features.keys())
  nan_features = [k for k, v in features.items() if pd.isna(v)]
  
  if missing or nan_features:
    return False, {"missing_features": list(missing), "nan_features": nan_features}
  return True, {}
```

### 3.2 Action on Missing Feature

**If any feature missing at assembly time**:
1. Log with full context: `"Feature assembly failed for grid={grid_id}, timestamp={ts}; missing={list}"`
2. Return **503 Service Unavailable** with reason
3. Do NOT attempt to fill with NaN, defaults, or city-level aggregates

**Why**:
- Model trained on complete features; partial features degrade accuracy unpredictably
- Better to fail loudly than silently degrade

---

## 4. Model Failures & Fallback Strategy

### 4.1 Model Loading

**Approach**: Multiple versioned models in artifact store.

```
artifacts/v1/models/
├── model_v124.txt        # Current production
├── model_v123.txt        # Previous version
└── model_v122.txt        # Two versions back (fallback)
```

**Singleton loader** (load once at service startup):

```python
class ModelLoader:
  def __init__(self):
    self.models = {}
    self.current_version = None
  
  def load_models(self):
    """Load current + 2 fallback versions at startup."""
    versions = ["v124", "v123", "v122"]
    for v in versions:
      try:
        model = lgb.Booster(model_file=f"artifacts/v1/models/model_{v}.txt")
        self.models[v] = model
        logger.info(f"Loaded model {v}")
      except Exception as e:
        logger.error(f"Failed to load model {v}: {e}")
    
    # Current version = highest-numbered available
    self.current_version = max([v for v in self.models.keys() if v in self.models])
```

### 4.2 Prediction Failure: Fallback Chain

**Trigger**: Model.predict() raises exception (corrupt model, OOM, etc.)

**Recovery**:
1. Attempt current version → fails
2. **Fall back to v123** → attempt predict
3. If v123 fails → **fall back to v122**
4. If v122 fails → **return 503 Service Unavailable**

```python
def predict_with_fallback(features: np.ndarray) -> np.ndarray | None:
  """
  Attempt prediction on current model; fall back to older versions on failure.
  Returns None if all models fail.
  """
  versions_to_try = [loader.current_version, "v123", "v122"]
  
  for version in versions_to_try:
    if version not in loader.models:
      logger.warning(f"Model {version} not loaded; skipping")
      continue
    
    try:
      prediction = loader.models[version].predict(features)
      if version != loader.current_version:
        logger.warning(f"Fallback to model {version}; current={loader.current_version}")
      return prediction
    except Exception as e:
      logger.error(f"Model {version} predict failed: {e}")
      continue
  
  # All fallbacks exhausted
  logger.critical("All model versions failed; returning 503")
  return None
```

### 4.3 Action: All Models Failed

**Return**: 503 Service Unavailable (see API Contracts)

---

## 5. Database Failures & Versioning

### 5.1 Concurrent Write Safety via Immutable Versioning

**Problem**: Multiple requests attempting to update predictions table simultaneously; risk of partial state exposure.

**Solution**: **Immutable versioned tables** with atomic version pointer.

**Schema**:
```sql
-- Metadata table: points to current prediction version
CREATE TABLE current_version (
  key TEXT PRIMARY KEY,
  value TEXT
);
INSERT INTO current_version VALUES ('predictions', 'v124');

-- Prediction versions (immutable)
CREATE TABLE predictions_v124 (
  city TEXT,
  grid_id INT,
  timestamp TIMESTAMP,
  pm25_pred FLOAT,
  confidence FLOAT,
  model_version TEXT,
  PRIMARY KEY (city, grid_id, timestamp)
);

-- Next refresh creates predictions_v125 in isolation
CREATE TABLE predictions_v125 (
  city TEXT,
  grid_id INT,
  timestamp TIMESTAMP,
  pm25_pred FLOAT,
  confidence FLOAT,
  model_version TEXT,
  PRIMARY KEY (city, grid_id, timestamp)
);
```

**Write Flow (Batch Refresh)**:
```python
def batch_refresh_predictions(timestamp: datetime, db: DuckDB):
  """
  Build new predictions table in isolation; atomically swap version pointer.
  """
  # Get current version
  current = db.execute("SELECT value FROM current_version WHERE key='predictions'").fetchall()[0][0]
  next_version = increment_version(current)  # v124 → v125
  
  # Create new table
  db.execute(f"""
    CREATE TABLE {next_version} AS
    SELECT city, grid_id, {timestamp}, predict(features), confidence, '{MODEL_VERSION}'
    FROM [feature assembly query]
  """)
  
  # Atomic swap
  db.execute("BEGIN TRANSACTION")
  try:
    db.execute(f"UPDATE current_version SET value = '{next_version}' WHERE key='predictions'")
    db.execute("COMMIT")
    logger.info(f"Swapped to {next_version}")
  except Exception as e:
    db.execute("ROLLBACK")
    logger.error(f"Swap failed: {e}")
    raise
```

**Read Flow (Inference)**:
```python
def get_prediction(grid_id: str, timestamp: datetime, db: DuckDB) -> dict:
  """
  Readers always fetch from current version (no blocking).
  """
  current_version = db.execute("SELECT value FROM current_version WHERE key='predictions'").fetchall()[0][0]
  
  result = db.execute(f"""
    SELECT pm25_pred, confidence, model_version
    FROM {current_version}
    WHERE grid_id = {grid_id} AND timestamp = {timestamp}
  """).fetchall()
  
  return {"pm25": result[0], "confidence": result[1], "model_version": result[2]}
```

**Atomicity Guarantees**:
- Crash before COMMIT → readers stay on old version (safe)
- Crash after COMMIT → readers see new version (safe)
- No partial state ever exposed

### 5.2 Memory Pool & Request Rejection

**Problem**: Unbounded memory allocation under high load (1000 concurrent users).

**Solution**: Pre-allocate fixed memory pool; reject requests if exceeded.

```python
class MemoryPool:
  def __init__(self, max_bytes: int = 2 * 1024**3):  # 2GB
    self.max_bytes = max_bytes
    self.current_bytes = 0
    self.lock = threading.Lock()
  
  def acquire(self, required_bytes: int) -> bool:
    """
    Attempt to acquire memory from pool.
    Returns True if successful; False if would exceed limit.
    """
    with self.lock:
      if self.current_bytes + required_bytes > self.max_bytes:
        logger.warning(f"Memory pool exhausted: {self.current_bytes}/{self.max_bytes}")
        return False
      self.current_bytes += required_bytes
      return True
  
  def release(self, bytes_to_free: int):
    with self.lock:
      self.current_bytes -= bytes_to_free

# At request entry
pool = MemoryPool()
if not pool.acquire(ESTIMATED_REQUEST_MEMORY):
  return 503, {"error": "Memory pool exhausted; too many concurrent requests"}
```

### 5.3 DuckDB Lock Contention

**Expected**: Light contention (queries <1ms at 1000 users).

**If DuckDB lock timeout occurs** (Kashi refresh blocks readers):
```python
def get_prediction_with_timeout(grid_id, timestamp, db, timeout_sec=2.0):
  try:
    result = db.execute(..., timeout=timeout_sec).fetchall()
    return result
  except TimeoutError:
    logger.error(f"DuckDB timeout fetching {grid_id}@{timestamp}")
    return 503, {"error": "Database temporarily locked; try again in 1s"}
```

---

## 6. Sensor Data Gaps (PM2.5 Coverage)

### 6.1 Sensor Availability Check

Before IDW interpolation, verify minimum sensor coverage:

```python
def check_sensor_availability(target_timestamp: datetime, min_sensors: int = 1) -> bool:
  """
  Verify that at least min_sensors exist with readings at target_timestamp.
  """
  count = db.execute(f"""
    SELECT COUNT(DISTINCT sensor_id)
    FROM pm25_observations
    WHERE timestamp BETWEEN '{target_timestamp - 30min}' AND '{target_timestamp + 30min}'
      AND pm25_value IS NOT NULL
  """).fetchall()[0][0]
  
  return count >= min_sensors
```

### 6.2 Failure Mode

**If `check_sensor_availability()` returns False**:
1. Log: `"Insufficient sensor data at {timestamp}; {count} sensors available (min={min_sensors})"`
2. **Raise SensorDataUnavailableError** (do not return NaN or defaults)
3. Propagate to inference (returns 503)

---

## 7. Stale Data Handling & Message Queue

### 7.1 Staleness Definition

A feature is **stale** if age exceeds tolerance:

| Source | Tolerance | Example |
|---|---|---|
| OpenMeteo (meteo features) | 1 hour | Age > 1h → stale |
| MODIS (satellite) | 16 days | Age > 16d → stale |
| Sentinel (satellite) | 5 days | Age > 5d → stale |
| Landsat (satellite) | 16 days | Age > 16d → stale |

### 7.2 Staleness Check in Inference

```python
def assemble_features(grid_id: str, timestamp: datetime) -> dict:
  features = {}
  staleness_warnings = []
  
  # For each feature, check age
  for feature_name, source in FEATURE_SOURCE_MAP.items():
    value, age = fetch_feature(feature_name, grid_id, timestamp)
    features[feature_name] = value
    
    tolerance = STALENESS_TOLERANCE[source]
    if age > tolerance:
      staleness_warnings.append({
        "feature": feature_name,
        "source": source,
        "age_seconds": age.total_seconds(),
        "tolerance_seconds": tolerance.total_seconds()
      })
  
  return features, staleness_warnings
```

### 7.3 Stale Data Notification: Redis Pub/Sub

**When stale data served** (feature age exceeds tolerance but is still used):

```python
def notify_stale_predictions(grid_id: str, timestamp: datetime, staleness_warnings: list):
  """
  Publish stale prediction notification to Redis.
  Clients poll this topic to trigger local refresh.
  """
  redis_client = redis.Redis(host='localhost', port=6379)
  
  message = {
    "grid_id": grid_id,
    "timestamp": timestamp.isoformat(),
    "features_stale": staleness_warnings,
    "notified_at": datetime.utcnow().isoformat()
  }
  
  topic = f"stale_predictions:{timestamp.isoformat()}"
  redis_client.publish(topic, json.dumps(message))
  logger.info(f"Published to {topic}: {len(staleness_warnings)} stale features")
```

**Client Behavior**:
1. Receive prediction response (may contain stale features)
2. Subscribe to Redis topic: `stale_predictions:{timestamp}`
3. If notification received → poll for fresh prediction
4. When fresh data available → update UI

---

## 8. API Response Contracts

### 8.1 Success Response (200 OK)

```json
{
  "grid_id": "M0001",
  "timestamp": "2026-01-18T10:00:00Z",
  "pm25_prediction": 65.3,
  "confidence": 0.92,
  "model_version": "v124",
  "features_fresh": true,
  "stale_features": [],
  "metadata": {
    "computation_time_ms": 145,
    "feature_version": "v58",
    "pm25_lags_source": "idw_interpolation"
  }
}
```

### 8.2 Partial Success (200 OK with Warnings)

```json
{
  "grid_id": "M0001",
  "timestamp": "2026-01-18T10:00:00Z",
  "pm25_prediction": 65.3,
  "confidence": 0.78,
  "model_version": "v124",
  "features_fresh": false,
  "stale_features": [
    {
      "feature": "ndvi",
      "source": "landsat",
      "age_hours": 17.5,
      "tolerance_hours": 16.0
    }
  ],
  "warnings": ["Using stale Landsat data (17.5h old)"],
  "metadata": {
    "computation_time_ms": 145,
    "feature_version": "v58"
  }
}
```

### 8.3 Bad Request (400 Bad Request)

```json
{
  "error": "invalid_request",
  "message": "Missing required parameter: timestamp",
  "details": {
    "received_params": ["grid_id"],
    "required_params": ["grid_id", "timestamp", "city"]
  }
}
```

### 8.4 Service Unavailable (503 Service Unavailable)

Triggered by:
- All models failed
- PM2.5 sensor data unavailable (zero sensors)
- Memory pool exhausted
- Any required feature cannot be assembled

```json
{
  "error": "service_unavailable",
  "message": "Inference temporarily unavailable",
  "reason": "pm25_sensor_data_unavailable",
  "details": {
    "grid_id": "M0001",
    "timestamp": "2026-01-18T10:00:00Z",
    "failing_component": "idw_interpolation",
    "error_context": "Zero sensors at 2026-01-18 09:30:00Z; cannot compute PM2.5 lags"
  },
  "retry_after_seconds": 60
}
```

### 8.5 Other Error Codes

| Code | Trigger | Action |
|---|---|---|
| 400 | Missing/invalid request parameters | Client re-sends with valid params |
| 404 | Grid cell does not exist in schema | Client validates grid_id |
| 429 | Rate limit exceeded | Client backs off exponentially |
| 500 | Unexpected server error | Log; return generic error (no PII) |
| 503 | Feature assembly or model failure | Client retries with backoff |

---

## 9. Retry Logic & Client Behavior

### 9.1 Automatic Retry (Server-Side)

**For PM2.5 IDW computation** (up to 2 retries):
```python
def compute_pm25_with_retry(grid_id, lag_hours, timestamp, max_retries=2):
  for attempt in range(max_retries + 1):
    try:
      return compute_pm25_lag(grid_id, lag_hours, timestamp, db)
    except SensorDataUnavailableError as e:
      if attempt < max_retries:
        logger.info(f"PM2.5 computation failed (attempt {attempt+1}); retrying...")
        sleep(0.5 * (2 ** attempt))  # Exponential backoff: 0.5s, 1s, ...
      else:
        logger.error(f"PM2.5 computation failed after {max_retries + 1} attempts")
        raise
```

### 9.2 Client-Side Retry

**For heatmap & point APIs**:
- **On 503**: Retry after `retry_after_seconds` (default: 60s)
- **Exponential backoff**: 1s → 2s → 4s → 8s (max 4 retries)
- **On 400/404**: Do NOT retry (validation error; client must fix)
- **On 429**: Respect rate limit; back off

```python
def request_prediction_with_retry(grid_id, timestamp, max_retries=4):
  for attempt in range(max_retries):
    try:
      response = requests.get(f"/api/predict?grid_id={grid_id}&timestamp={timestamp}")
      if response.status_code == 200:
        return response.json()
      elif response.status_code == 503:
        retry_after = response.json().get("retry_after_seconds", 60)
        logger.warning(f"503 on attempt {attempt + 1}; retrying after {retry_after}s")
        time.sleep(retry_after)
      elif response.status_code in [400, 404]:
        raise ValueError(f"Invalid request: {response.json()}")
    except Exception as e:
      logger.error(f"Request failed: {e}")
      if attempt < max_retries - 1:
        time.sleep(2 ** attempt)
  
  raise RuntimeError("Max retries exhausted")
```

---

## 10. Heatmap API: Batch Optimization

### 10.1 Heatmap Request Pattern

**Request**: All 59k grid cells for a single timestamp T.

**Problem**: Recomputing all 59k cells is expensive; many cells may be fresh.

### 10.2 Solution: Selective Recomputation

```python
def get_heatmap(timestamp: datetime, db: DuckDB, refresh: bool = False):
  """
  Serve heatmap for all 59k cells at timestamp.
  Recompute only stale cells; return cached fresh cells.
  """
  current_version = db.execute("SELECT value FROM current_version WHERE key='predictions'").fetchall()[0][0]
  
  # Fetch all cells from current version
  all_predictions = db.execute(f"""
    SELECT grid_id, pm25_pred, timestamp
    FROM {current_version}
    WHERE timestamp = {timestamp}
  """).fetchall()  # 59k rows
  
  # Find which cells are stale (age > tolerance)
  stale_grid_ids = []
  for grid_id, pm25, ts in all_predictions:
    age = datetime.utcnow() - ts
    if age > STALENESS_TOLERANCE["meteo"]:  # 1 hour
      stale_grid_ids.append(grid_id)
  
  logger.info(f"Heatmap at {timestamp}: {len(all_predictions)} total, {len(stale_grid_ids)} stale")
  
  # Recompute only stale cells
  if len(stale_grid_ids) > 0:
    fresh_predictions = batch_predict(stale_grid_ids, timestamp)
    # Insert into current version (versioned table handles isolation)
    for grid_id, pred in fresh_predictions.items():
      db.execute(f"""
        UPDATE {current_version}
        SET pm25_pred = {pred['pm25']}, timestamp = {timestamp}
        WHERE grid_id = {grid_id}
      """)
  
  # Return all 59k cells (fresh + cached)
  return db.execute(f"""
    SELECT grid_id, pm25_pred, timestamp
    FROM {current_version}
    WHERE timestamp = {timestamp}
  """).fetchall()
```

### 10.3 Behavior

- **Fresh cells** (age ≤ 1h): Returned immediately from cache
- **Stale cells** (age > 1h): Recomputed, inserted, returned
- **Response time**: ~500ms–2s (depending on % stale cells)

---

## 11. Monitoring & Alerting

### 11.1 Structured Logging

Every inference request logs:

```json
{
  "timestamp": "2026-01-18T10:00:00.123Z",
  "event": "inference_complete",
  "grid_id": "M0001",
  "request_timestamp": "2026-01-18T09:00:00Z",
  "status": "success",
  "pm25_prediction": 65.3,
  "confidence": 0.92,
  "model_version": "v124",
  "features_fresh": false,
  "stale_features_count": 1,
  "stale_features": ["ndvi"],
  "feature_assembly_time_ms": 45,
  "model_inference_time_ms": 12,
  "total_time_ms": 145,
  "memory_used_bytes": 512000
}
```

On failure:

```json
{
  "timestamp": "2026-01-18T10:00:00.123Z",
  "event": "inference_failed",
  "grid_id": "M0001",
  "request_timestamp": "2026-01-18T09:00:00Z",
  "status": "error",
  "error_code": "pm25_sensor_unavailable",
  "error_message": "Zero sensors at 2026-01-18 09:30:00Z",
  "failing_component": "idw_interpolation",
  "feature_assembly_time_ms": 23,
  "total_time_ms": 25,
  "memory_used_bytes": 256000
}
```

### 11.2 Metrics to Track

| Metric | Alert Threshold | Action |
|---|---|---|
| Inference error rate | >5% over 5min | Page on-call engineer |
| Model fallback rate | >1% over 10min | Check model health; re-train if needed |
| Sensor data gaps | >10 consecutive hours | Check OpenAQ API status |
| DuckDB lock timeout | >1/min | Scale horizontally or optimize queries |
| Memory pool exhaustion | >0 events | Increase pool or reduce concurrent requests |
| Mean response time | >2s | Check feature assembly bottleneck |

### 11.3 Dashboard (Time-Series)

- Error rate per component (PM2.5, MeteoBase, MeteoLags, Satellite)
- Model fallback frequency
- Stale feature frequency (by source)
- Heatmap recomputation % (% of cells needing refresh)
- p50/p95/p99 response times
- Memory pool usage

---

## 12. Failure Scenarios & Recovery

### Scenario 1: OpenMeteo API Down

**Symptom**: All MeteoBase + MeteoLags features unavailable at refresh time.

**Impact**: Cannot assemble any request (all features critical).

**Recovery**:
1. **Immediate**: Return 503 for all requests (no features available)
2. **Rollback**: Kashi service falls back to previous OpenMeteo data (2-day cache)
3. **Client**: Polls `/api/health` endpoint until status → green
4. **Resume**: When OpenMeteo recovers, Kashi re-fetches; inference resumes

### Scenario 2: DuckDB Disk Full

**Symptom**: Kashi refresh fails; writes blocked.

**Impact**: Predictions table not updated; feature data stale.

**Recovery**:
1. Ops team: Clear old data (archive to S3), expand disk
2. Kashi: Retries refresh with exponential backoff
3. Readers: Continue serving from `current_version` (old but consistent)
4. Resume: After disk cleared, Kashi catches up

### Scenario 3: All Model Versions Corrupt

**Symptom**: Model fallback chain exhausted (v124, v123, v122 all fail).

**Impact**: Cannot predict; all requests fail.

**Recovery**:
1. **Immediate**: Return 503 for all requests
2. **Ops**: Restore model from git-tracked artifact store
3. **Reload**: Restart inference service
4. **Resume**: Fallback chain re-loads; inference resumes

### Scenario 4: 1000 Concurrent Users Spike

**Symptom**: Memory pool exhausted; requests rejected.

**Impact**: High 503 rate.

**Recovery**:
1. **Immediate**: Clients retry with backoff (503 response includes `retry_after`)
2. **Load shedding**: Reject non-critical requests (heatmap background refreshes)
3. **Scale**: Increase memory pool or spin up additional Python process
4. **Resume**: Load subsides; 503 rate drops

---

## 13. Configuration

### 13.1 Feature Criticality (in code)

```python
FEATURE_CRITICALITY = {
  "pm25_t_minus_1h": "CRITICAL",
  "pm25_t_minus_2h": "CRITICAL",
  "temperature_2m": "CRITICAL",
  "temperature_2m_lag1h": "CRITICAL",
  # ... all 64 features listed as CRITICAL
}

STALENESS_TOLERANCE_SECONDS = {
  "pm25_lags": 24 * 3600,        # 24 hours
  "meteo_base": 1 * 3600,         # 1 hour
  "meteo_lags": 12 * 3600,        # 12 hours
  "satellite": 16 * 24 * 3600,    # 16 days
  "temporal": float('inf'),        # Never stale
}
```

### 13.2 Memory Pool (config)

```python
MEMORY_POOL_MAX_BYTES = 2 * 1024**3  # 2GB (tunable per hardware)
CONCURRENT_REQUEST_LIMIT = 1000
MEMORY_PER_REQUEST_BYTES = 2 * 1024**2  # 2MB estimate
```

### 13.3 Retry Logic (config)

```python
PM25_IDW_RETRIES = 2
KASHI_REFRESH_RETRIES = 3
EXPONENTIAL_BACKOFF_BASE_SECONDS = 0.5
EXPONENTIAL_BACKOFF_MAX_SECONDS = 30
```

---

## 14. Summary Decision Tree

```
Request arrives for (grid_id, timestamp)
  │
  ├─→ Validate parameters
  │    └─→ Invalid? Return 400
  │
  ├─→ Acquire memory from pool
  │    └─→ Pool exhausted? Return 503 with "memory_exhausted"
  │
  ├─→ Assemble all 64 features
  │    │
  │    ├─→ PM2.5 lags (IDW)
  │    │    └─→ Zero sensors? Return 503 with "pm25_sensor_unavailable"
  │    │
  │    ├─→ MeteoBase + MeteoLags
  │    │    └─→ Missing from cache? Return 503 with "meteo_unavailable"
  │    │
  │    ├─→ Satellite (Landsat/Sentinel/MODIS)
  │    │    └─→ Missing? Return 503 with "satellite_unavailable"
  │    │
  │    └─→ Temporal (always available)
  │
  ├─→ Validate feature vector (no NaN, all columns present)
  │    └─→ Missing any feature? Return 503 with "feature_assembly_failed"
  │
  ├─→ Predict with fallback (v124 → v123 → v122)
  │    └─→ All models fail? Return 503 with "model_prediction_failed"
  │
  ├─→ Check staleness of features
  │    └─→ Any stale? Publish to Redis pub/sub topic "stale_predictions:{timestamp}"
  │
  ├─→ Return 200 OK with prediction + metadata
  │
  └─→ Release memory from pool
```

---

## 15. Implementation Checklist

- [ ] Add feature criticality matrix to code
- [ ] Implement sensor availability check (PM2.5 IDW)
- [ ] Implement feature vector validation
- [ ] Load multiple model versions at startup
- [ ] Implement prediction fallback logic
- [ ] Create immutable versioned prediction tables
- [ ] Implement atomic version pointer (current_version table)
- [ ] Implement memory pool + request rejection
- [ ] Add DuckDB lock timeout handling
- [ ] Implement staleness checks per feature source
- [ ] Implement Redis pub/sub notification on stale data
- [ ] Define API response contracts (JSON schemas)
- [ ] Implement structured logging (JSON format)
- [ ] Add monitoring dashboard (time-series metrics)
- [ ] Test failure scenarios (chaos engineering)
- [ ] Document runbook for each failure scenario

---

**Document Version**: 1.0  
**Last Updated**: January 18, 2026  
**Status**: Locked for Implementation
