# Takshashila: Cache Hit Strategy

**Date**: January 19, 2026  
**Status**: Final Design (Locked)  
**Audience**: Backend engineers, DevOps

---

## Executive Summary

This document specifies the two-tier caching strategy for PM2.5 predictions, sensor data, and satellite indices. The system balances **freshness** (predictions stale after 1h) against **performance** (1000 concurrent users requesting 59k-cell heatmaps in <1-5s). 

**Key Design Principles**:
1. **Per-cell-per-timestamp caching** — `(grid_id, timestamp)` as cache key
2. **Two-tier storage** — L1 (in-memory, 2GB) + L2 (DuckDB, 2GB)
3. **L2-first check** — look in DuckDB before recomputing
4. **TTL-based eviction** — cache expires 1h (predictions valid 24h; freshness enforced)
5. **Daily flush** — older prediction days purged from DuckDB
6. **Single-flight recompute** — only one request computes; others wait (deduplication)
7. **Demand-driven warm-up** — no preloading; requests trigger cache population
8. **Pre-computed IDW weights** — geometric costs amortized at startup

---

## 1. Data Inventory & Cache Scope

### 1.1 Data Types & Lifespans

| Data | Type | Size | Fresh TTL | Valid TTL | Cache Tier | Refresh |
|---|---|---|---|---|---|---|
| **PM2.5 Predictions** | float16 × 59k cells × T | ~180KB per hour | 1h | 24h | L1 + L2 | On-demand |
| **PM2.5 Sensors** | float16 × 31 sensors × hourly | ~2KB per hour | 1h | 24h | L2 only | Hourly |
| **Landsat8 Indices** | float16 × 59k × 9 bands | ~11MB | 16d | 16d | L2 only | Every 16d |
| **Sentinel2 Indices** | float16 × 59k × 5 bands | ~6MB | 5d | 5d | L2 only | Every 5d |
| **Meteo** | float16 × 59k × 8 vars | ~5MB | 1h | 24h | L2 only | Hourly |
| **IDW Weights** | float32 × 59k × 31 sensors | ~7MB | ∞ | ∞ | In-memory | Startup only |

### 1.2 Cache Budget

**Available RAM**: 2GB L1 (in-memory) + 2GB L2 (DuckDB)

**Capacity Calculation**:
- L1 capacity: 2GB ÷ (59k cells × 2 bytes/float16) = **~17 timestamps** in memory
- L2 capacity: 2GB ÷ (59k cells × 2 bytes) = **~17 timestamps** in DuckDB
- Total: **~34 hours of predictions** in memory + disk

**Implication**: Cache 59k cells for last 17 hours; older predictions flushed.

---

## 2. Two-Tier Cache Architecture

### 2.1 Cache Hierarchy

```
Request for (grid_id, timestamp)
  │
  ├─→ L1 Check (in-memory dict)
  │    └─→ Hit? Return immediately (~0.1ms)
  │    └─→ Miss? Continue to L2
  │
  ├─→ L2 Check (DuckDB predictions table)
  │    └─→ Hit + Fresh (<1h)? Load into L1, return (~1-2ms)
  │    └─→ Hit + Stale (>1h)? Continue to recompute
  │    └─→ Miss? Continue to recompute
  │
  └─→ Recompute (assemble features, run model)
       └─→ Store in L1 + L2, return (~100-200ms)
```

### 2.2 L1 Cache (In-Memory)

**Storage**: Python dict or Redis

```python
# Python in-process dict (fastest, MVP)
L1_CACHE = {
  ("M0001", "2026-01-19T10:00:00Z"): 65.3,  # (grid_id, timestamp) → pm25_pred
  ("M0001", "2026-01-19T09:00:00Z"): 68.2,
  # ... up to ~17 timestamps × 59k cells
}

# Metadata
L1_METADATA = {
  ("M0001", "2026-01-19T10:00:00Z"): {
    "timestamp": datetime,
    "created_at": datetime,
    "age_seconds": computed,
    "fresh": bool  # age < 3600
  }
}
```

**Eviction Policy**: **TTL-based** (not LRU)

```python
def evict_expired_from_l1():
  """Remove entries older than 1 hour from L1."""
  now = datetime.utcnow()
  to_delete = []
  
  for (grid_id, timestamp), created_at in L1_METADATA.items():
    age_seconds = (now - created_at).total_seconds()
    if age_seconds > 3600:  # >1 hour
      to_delete.append((grid_id, timestamp))
  
  for key in to_delete:
    del L1_CACHE[key]
    del L1_METADATA[key]
  
  logger.info(f"Evicted {len(to_delete)} stale entries from L1")
```

**Eviction Trigger**: 
- Run every 5 minutes (background task)
- On every cache miss (opportunistic cleanup)

### 2.3 L2 Cache (DuckDB)

**Storage**: `predictions_v*` table (immutable versioned tables from ERROR_HANDLING.md)

```sql
SELECT * FROM predictions_v124
WHERE grid_id = 'M0001' AND timestamp = '2026-01-19T10:00:00Z';

-- Output:
city, grid_id, timestamp, pm25_pred, confidence, model_version, created_at
```

**Age Check** (before using L2 data):

```python
def get_from_l2_if_fresh(grid_id: str, timestamp: datetime, db: DuckDB) -> float | None:
  """
  Fetch from DuckDB; return None if stale (>1h old).
  """
  current_version = db.execute("SELECT value FROM current_version WHERE key='predictions'").fetchall()[0][0]
  
  result = db.execute(f"""
    SELECT pm25_pred, created_at
    FROM {current_version}
    WHERE grid_id = {grid_id} AND timestamp = {timestamp}
  """).fetchall()
  
  if not result:
    return None  # Cache miss
  
  pm25_pred, created_at = result[0]
  age_seconds = (datetime.utcnow() - created_at).total_seconds()
  
  if age_seconds > 3600:  # >1 hour
    logger.info(f"L2 hit but stale ({age_seconds}s old); will recompute")
    return None  # Treat as miss; force recompute
  
  logger.info(f"L2 hit (fresh, {age_seconds}s old)")
  return pm25_pred
```

---

## 3. Cache Freshness Model

### 3.1 Timestamp-Based Organization

Predictions organized by **hour**:

| Timestamp | Fresh Until | Stale After | Valid Until | Action |
|---|---|---|---|---|
| 2026-01-19 10:00 | 11:00 | 11:01 | 2026-01-20 10:00 | Serve if <11:00; recompute if ≥11:00 |
| 2026-01-19 09:00 | 10:00 | 10:01 | 2026-01-20 09:00 | Serve if <10:00; recompute if ≥10:00 |
| 2026-01-18 10:00 | 2026-01-18 11:00 | 2026-01-18 11:01 | 2026-01-19 10:00 | Expired after 24h; flushed |

### 3.2 Daily Flush Schedule

**At midnight UTC (00:00)**:

```python
def daily_flush_old_predictions(db: DuckDB):
  """
  Delete predictions older than 24 hours from L2 (DuckDB).
  Frees up disk space; keeps only recent 24h in storage.
  """
  cutoff = datetime.utcnow() - timedelta(hours=24)
  current_version = db.execute("SELECT value FROM current_version WHERE key='predictions'").fetchall()[0][0]
  
  deleted = db.execute(f"""
    DELETE FROM {current_version}
    WHERE created_at < '{cutoff}'
  """).rowcount
  
  logger.info(f"Flushed {deleted} predictions older than 24h")
  
  # Vacuum to reclaim disk space
  db.execute("VACUUM")
```

**Trigger**: APScheduler cron job (same Takshashila service)

---

## 4. Cache Invalidation

### 4.1 Invalidation Trigger: Takshashila Refresh

When Takshashila refreshes feature data (OpenMeteo hourly, satellite every 5-16 days):

**Current Strategy**: **TTL-based only** (no event-based invalidation)

```python
# Takshashila publishes refresh to Redis (for Go server awareness)
redis_client.publish(f"refresh:openmeteo:2026-01-19T10:00:00Z", {
  "source": "openmeteo",
  "timestamp": "2026-01-19T10:00:00Z",
  "rows_updated": 59000
})

# Python inference service does NOT listen
# Instead: relies on TTL expiry to invalidate cache
# Predictions for T computed before Takshashila refresh are stale after 1h anyway
```

**Rationale**: 
- Simple (no cache coherence complexity)
- Predictions auto-expire 1h → guaranteed freshness window
- Takshashila refresh at T:00 → predictions stale at T:01 onward → force recompute

### 4.2 Invalidation Semantics

| Scenario | Action |
|---|---|
| Takshashila fetches meteo at 10:00; Python predicts 10:05 | Cache hit; serve fresh prediction |
| Takshashila fetches meteo at 10:00; user requests at 11:05 | L2 stale (>1h); recompute with new meteo |
| Landsat data refreshed (16-day cycle) | Predictions auto-stale after 1h anyway |
| Manual cache clear (ops intervention) | Flush entire L1 via admin endpoint |

---

## 5. Heatmap Cache Strategy

### 5.1 Request Pattern

**Heatmap API**: `GET /api/heatmap?timestamp=2026-01-19T10:00:00Z`

**Response**: All 59k cells at timestamp T

**Challenge**: Mix of fresh + stale + missing cells

### 5.2 Implementation: Block Until Fresh

```python
@app.route("/api/heatmap")
def get_heatmap(timestamp: str):
  """
  Serve heatmap for all 59k cells at timestamp.
  Block until all cells fresh (<1h old) or timeout.
  """
  target_ts = datetime.fromisoformat(timestamp)
  start_time = time.time()
  timeout_seconds = 10  # Max wait
  
  while (time.time() - start_time) < timeout_seconds:
    # Check if all cells are fresh
    all_fresh_count = db.execute(f"""
      SELECT COUNT(*) FROM {current_version}
      WHERE timestamp = '{target_ts}'
        AND created_at > datetime('now', '-1 hour')
    """).fetchall()[0][0]
    
    if all_fresh_count == 59000:  # All cells fresh
      logger.info(f"All cells fresh for {timestamp}")
      # Return heatmap
      heatmap_data = db.execute(f"""
        SELECT grid_id, pm25_pred FROM {current_version}
        WHERE timestamp = '{target_ts}'
      """).fetchall()
      return 200, {"cells": heatmap_data, "all_fresh": True}
    
    # Cells stale or missing → trigger batch recompute
    stale_grid_ids = db.execute(f"""
      SELECT grid_id FROM {current_version}
      WHERE timestamp = '{target_ts}'
        AND (created_at <= datetime('now', '-1 hour') OR created_at IS NULL)
      LIMIT 100  # Recompute in batches
    """).fetchall()
    
    if stale_grid_ids:
      logger.info(f"Recomputing {len(stale_grid_ids)} stale cells")
      batch_predict(stale_grid_ids, target_ts)
    
    # Wait and retry
    time.sleep(0.5)
  
  # Timeout: return best-effort (mix of fresh + stale)
  logger.warning(f"Heatmap timeout at {timestamp}; returning partial")
  heatmap_data = db.execute(f"""
    SELECT grid_id, pm25_pred, created_at FROM {current_version}
    WHERE timestamp = '{target_ts}'
  """).fetchall()
  return 200, {
    "cells": heatmap_data,
    "all_fresh": False,
    "warning": "Some cells stale or missing"
  }
```

**Behavior**:
- **Best case** (all cached + fresh): Return in <100ms
- **Typical case** (some stale): Recompute batch, block up to 10s
- **Timeout case** (not done): Return partial at 10s with warning

**Response Time**: <1-5s target (balanced between immediate + complete)

---

## 6. Point API Cache Strategy

### 6.1 Request Pattern

**Point API**: `GET /api/predict?grid_id=M0001&timestamp=2026-01-19T10:00:00Z`

**Single cell**, not all 59k

### 6.2 Implementation: Force Fresh (Option B)

```python
@app.route("/api/predict")
def get_prediction(grid_id: str, timestamp: str):
  """
  Single cell prediction. Force fresh (<1h old) by recomputing if stale.
  """
  target_ts = datetime.fromisoformat(timestamp)
  
  # Check L1
  if (grid_id, target_ts) in L1_CACHE:
    cached_pred = L1_CACHE[(grid_id, target_ts)]
    age = (datetime.utcnow() - L1_METADATA[(grid_id, target_ts)]["created_at"]).total_seconds()
    if age < 3600:
      logger.info(f"L1 hit ({age}s old)")
      return 200, {"pm25": cached_pred, "fresh": True}
  
  # Check L2
  fresh_pred = get_from_l2_if_fresh(grid_id, target_ts, db)
  if fresh_pred is not None:
    L1_CACHE[(grid_id, target_ts)] = fresh_pred
    logger.info("L2 hit (fresh)")
    return 200, {"pm25": fresh_pred, "fresh": True}
  
  # Recompute (L1 miss, L2 miss or stale)
  logger.info(f"Cache miss; recomputing {grid_id}@{target_ts}")
  prediction = compute_and_cache(grid_id, target_ts)
  
  return 200, {"pm25": prediction, "fresh": True}
```

**Behavior**:
- **Cache hit**: Return immediately
- **Cache miss/stale**: Recompute synchronously (~100-200ms)
- **Freshness guaranteed**: Always <1h old (or just computed)

---

## 7. Sensor Data & Satellite Data Caching

### 7.1 PM2.5 Sensor Data (L2 Only)

**Storage**: `pm25_observations` DuckDB table (updated hourly by Takshashila)

**No L1 caching** — accessed only during IDW computation for features

**Access Pattern**:
```python
def compute_pm25_lags_with_idw(grid_id: str, timestamp: datetime) -> list[float]:
  """
  Fetch sensor data from DuckDB; compute IDW for 6 lags.
  Sensor data refreshed hourly; stale after 1h.
  """
  # Fetch all available sensors at target timestamps
  for lag_hours in [1, 2, 3, 6, 12, 24]:
    target_ts = timestamp - timedelta(hours=lag_hours)
    
    sensors = db.execute(f"""
      SELECT sensor_id, pm25_value, latitude, longitude
      FROM pm25_observations
      WHERE timestamp BETWEEN '{target_ts - timedelta(minutes=30)}' 
                         AND '{target_ts + timedelta(minutes=30)}'
        AND pm25_value IS NOT NULL
    """).fetchall()
    
    # Compute IDW with pre-computed weights
    pm25_lag = idw_interpolate(grid_id, sensors)
    pm25_lags.append(pm25_lag)
  
  return pm25_lags
```

### 7.2 Satellite Data (L2 Only, Pre-Computed IDW)

**Landsat8 + Sentinel2 indices** (NDVI, EVI, NDBI, etc.)

**Pre-Computed at Startup**:

```python
class IDWWeights:
  def __init__(self, db: DuckDB):
    self.db = db
    self.weights = {}  # (grid_id, sensor_id) → weight
  
  def precompute_weights(self):
    """
    Pre-compute IDW weights for all (grid_id, sensor_id) pairs.
    Done once at startup; reused for all timestamps.
    """
    sensors = db.execute("SELECT sensor_id, latitude, longitude FROM sensors").fetchall()
    
    for grid_id in range(59000):
      grid_lat, grid_lon = get_grid_center(grid_id)
      
      for sensor_id, sensor_lat, sensor_lon in sensors:
        distance = haversine(grid_lat, grid_lon, sensor_lat, sensor_lon)
        weight = 1.0 / (distance ** 2) if distance > 0 else 1e6
        self.weights[(grid_id, sensor_id)] = weight
    
    logger.info(f"Pre-computed IDW weights for {len(self.weights)} grid-sensor pairs")
  
  def interpolate(self, grid_id: str, sensor_values: dict) -> float:
    """
    Fast IDW: lookup pre-computed weights, multiply, sum.
    """
    numerator = 0.0
    denominator = 0.0
    
    for sensor_id, value in sensor_values.items():
      weight = self.weights[(grid_id, sensor_id)]
      numerator += value * weight
      denominator += weight
    
    return numerator / denominator if denominator > 0 else 0.0
```

**Storage**: In-memory dict (7MB) at service startup

**Cache Strategy**:
- Landsat indices + Sentinel indices stored in DuckDB (L2 only)
- Fetched with satellite table queries (already indexed)
- TTL: 16 days (Landsat), 5 days (Sentinel)

---

## 8. Cache Warm-Up Strategy

### 8.1 MVP: Demand-Driven (No Preloading)

**Strategy**: Service starts with empty cache; requests populate it.

```python
class CacheManager:
  def __init__(self):
    self.L1 = {}  # Empty at startup
    self.warmup_in_progress = False
  
  def on_startup(self):
    """
    At service startup:
    1. Load IDW weights (geometry, ~7MB)
    2. Load current model (LAZy loaded later)
    3. Load current_version pointer from DuckDB
    4. L1 cache remains empty — requests will populate it
    """
    self.idw_weights = IDWWeights(db)
    self.idw_weights.precompute_weights()
    logger.info("IDW weights pre-computed; L1 cache empty; requests will warm it up")
```

**Warm-Up Flow** (first request after startup):

```
Request for (grid_id, timestamp) arrives
  ├─→ L1 miss (cache empty)
  ├─→ L2 miss (predictions table may be empty)
  ├─→ Recompute (assemble features, model predict)
  ├─→ Store in L1 + L2
  └─→ Return to user

Second request for same cell
  ├─→ L1 hit
  └─→ Return cached
```

**Benefit**: Fast service startup (no long preload phase)

**Trade-off**: First user requests are slower (~100-200ms); subsequent requests cached

### 8.2 No Periodic Warm-Up

**Decision**: Skip scheduled warm-up (hourly refresh of entire cache)

**Rationale**:
- Requests naturally refresh hot cells
- Reduces background compute load
- Cold cells (rarely requested) don't need to be in cache

---

## 9. Go Server + Python Inference Coherence

### 9.1 Cache Architecture (Option C)

```
┌─────────────────────────────────────────────┐
│ Go HTTP Server                              │
│  - Receives /api/heatmap, /api/predict      │
│  - Calls Python inference service (RPC/gRPC)│
│  - Returns response to client               │
│  - Does NOT cache locally                   │
└─────────────────────────────────────────────┘
                    ↓
        [Go reads DuckDB directly]
                    ↓
┌─────────────────────────────────────────────┐
│ Python Inference Service                    │
│  - Maintains L1 cache (in-memory)           │
│  - Reads L2 (DuckDB) on L1 miss             │
│  - Recomputes on L2 miss/stale              │
│  - Writes to L2 (DuckDB)                    │
└─────────────────────────────────────────────┘
                    ↓
        [Shared DuckDB predictions_v*]
```

**Behavior**:
- **Go server calls Python** for predictions (no caching in Go)
- **Go can read DuckDB directly** for heatmap visualization (if Python unavailable)
- **Python maintains L1** for speed
- **Both read L2** (DuckDB) as source of truth

**Coherence Model**: Eventual consistency
- Go may see slightly stale predictions (seconds old from L2)
- Python cache L1 is authoritative
- No explicit cache synchronization needed

---

## 10. Single-Flight Recompute (Deduplication Prep)

### 10.1 Design Pattern

When request for `(grid_id, timestamp)` misses cache:

```python
# Shared compute locks (per-cell-per-timestamp)
compute_locks = {}  # (grid_id, timestamp) → threading.Lock()

def get_prediction_single_flight(grid_id: str, timestamp: datetime) -> float:
  """
  Prevent duplicate recomputation via per-key lock.
  First request computes; others wait for same result.
  """
  cache_key = (grid_id, timestamp)
  
  # Check cache first
  if cache_key in L1_CACHE:
    return L1_CACHE[cache_key]
  
  # Get/create lock for this key
  if cache_key not in compute_locks:
    compute_locks[cache_key] = threading.Lock()
  
  lock = compute_locks[cache_key]
  
  # Try to acquire lock
  acquired = lock.acquire(timeout=10.0)  # Detailed in deduplication topic
  
  try:
    # Re-check cache (another thread may have filled it)
    if cache_key in L1_CACHE:
      return L1_CACHE[cache_key]
    
    # Compute
    prediction = compute_and_cache(grid_id, timestamp)
    return prediction
  finally:
    if acquired:
      lock.release()
```

**Detail**: Full locking strategy covered in next topic (Deduplication)

---

## 11. Cache Configuration

### 11.1 Cache Sizes & TTLs

```python
# L1 (In-Memory)
L1_MAX_SIZE_BYTES = 2 * 1024**3  # 2GB
L1_TTL_SECONDS = 3600  # 1 hour

# L2 (DuckDB)
L2_MAX_SIZE_BYTES = 2 * 1024**3  # 2GB
L2_RETENTION_HOURS = 24  # Flush older predictions

# Eviction Schedule
EVICTION_INTERVAL_SECONDS = 300  # Every 5 minutes

# Warm-Up
WARMUP_STRATEGY = "demand_driven"  # No preloading

# Heatmap Blocking
HEATMAP_TIMEOUT_SECONDS = 10
BATCH_RECOMPUTE_SIZE = 100  # Recompute 100 cells at a time
```

### 11.2 Metrics

```python
cache_metrics = {
  "l1_hits": 0,
  "l1_misses": 0,
  "l2_hits_fresh": 0,
  "l2_hits_stale": 0,
  "l2_misses": 0,
  "recompute_count": 0,
  "l1_size_bytes": 0,
  "l2_size_bytes": 0,
  "heatmap_timeouts": 0,
  "average_heatmap_latency_ms": 0,
  "average_point_latency_ms": 0
}
```

---

## 12. Summary Decision Tree

```
Request arrives for (grid_id, timestamp)
  │
  ├─→ Check L1 (in-memory)
  │    ├─→ Hit + fresh (<1h)? Return immediately
  │    └─→ Miss or stale? Continue
  │
  ├─→ Check L2 (DuckDB)
  │    ├─→ Hit + fresh (<1h)? Load into L1, return
  │    ├─→ Hit + stale (>1h)? Delete from L1, continue to recompute
  │    └─→ Miss? Continue to recompute
  │
  └─→ Acquire single-flight lock for (grid_id, timestamp)
       ├─→ Recompute (assemble features, model predict)
       ├─→ Store in L1 + L2
       ├─→ Release lock
       └─→ Return to user
```

**For Heatmap API**:
- Fetch all 59k cells
- For each cell, follow above tree
- Block until all fresh (<1h) or timeout (10s)
- Return all cells (may be partial on timeout)

---

## 13. Implementation Checklist

- [ ] Implement L1 cache dict + metadata (timestamps, freshness flags)
- [ ] Implement L2 freshness check (age >1h triggers recompute)
- [ ] Implement TTL-based eviction (background task every 5 min)
- [ ] Pre-compute IDW weights at service startup
- [ ] Implement heatmap blocking logic (retry loop up to 10s)
- [ ] Implement batch recomputation (recompute 100 cells at a time)
- [ ] Implement daily flush (at 00:00 UTC, delete >24h old)
- [ ] Add cache metrics (hits, misses, latencies)
- [ ] Add `/api/cache/stats` endpoint (visibility)
- [ ] Test single-flight lock (concurrent requests for same cell)
- [ ] Test heatmap blocking timeout (10s max wait)
- [ ] Benchmark: L1 hit latency (<1ms), L2 hit latency (1-2ms), recompute (100-200ms)

---

## 14. Future Enhancements (Not MVP)

### 14.1 Predictive Warm-Up

Pre-compute cells 30 minutes before they're requested (if predictable usage pattern).

### 14.2 Redis L1 Cache

Replace Python dict with Redis for:
- Multi-process sharing (future scaling)
- Durability (survive process restart)
- TTL management (Redis native)

### 14.3 Adaptive TTL

Adjust 1h TTL based on data freshness:
- If Takshashila refresh delayed → extend TTL automatically
- If user requests >1h old → serve stale + flag

---

**Document Version**: 1.0  
**Last Updated**: January 19, 2026  
**Status**: Locked for Implementation
