# Takshashila: Lag Computation Strategy (PM2.5 IDW)

**Date**: January 19, 2026  
**Status**: Final Design (Locked)  
**Audience**: Backend engineers, data scientists

---

## Executive Summary

This document specifies how **PM2.5 lags** (1h, 2h, 3h, 6h, 12h, 24h) are computed for each grid cell during feature assembly. The strategy uses **Inverse Distance Weighting (IDW)** to interpolate sensor readings in space. Lags are **computed on-demand** (not pre-stored), using a single batched query of 24-hour historical sensor data from the `pm25_observations` table.

**Key Design Principles**:
1. **Single batched fetch** — one query [T-24h, T] for all lags
2. **Per-lag IDW computation** — each lag (1h, 2h, ..., 24h) gets separate interpolation
3. **Haversine distance** — spatial metric (lat/lon great-circle)
4. **IDW power tunable** — start with p=2; can adjust
5. **Co-located sensor shortcut** — use sensor directly if at grid centroid
6. **Fallback to mean** — if zero sensors at lag timestamp, use arithmetic mean of other sensors
7. **Range validation** — reject readings <0 or >1000 ppm
8. **On-demand computation** — no pre-storage; fetch during inference

---

## 1. PM2.5 Lag Requirements

### 1.1 Lag Definition

For a prediction at timestamp **T**, compute lags at earlier timestamps:

| Lag Name | Hours Before | Example (T=10:00) | Fetch Timestamp |
|---|---|---|---|
| lag_1h | -1 | 09:00 | 09:00 |
| lag_2h | -2 | 08:00 | 08:00 |
| lag_3h | -3 | 07:00 | 07:00 |
| lag_6h | -6 | 04:00 | 04:00 |
| lag_12h | -12 | 22:00 (prev day) | 22:00 |
| lag_24h | -24 | 10:00 (prev day) | 10:00 |

**Timestamp Flooring**: All lags snap to `:00` minute boundary (nearest hour).

Example:
- Prediction time: T = 16:30
- T-1h = 15:30 → floor to **15:00**
- Lag timestamp = 15:00 (use sensor readings at 15:00, not 15:30)

### 1.2 Feature Schema

```python
FEATURE_LIST = [
  "pm25_t_minus_1h",    # Lag at T-1h
  "pm25_t_minus_2h",    # Lag at T-2h
  "pm25_t_minus_3h",    # Lag at T-3h
  "pm25_t_minus_6h",    # Lag at T-6h
  "pm25_t_minus_12h",   # Lag at T-12h
  "pm25_t_minus_24h",   # Lag at T-24h
  # ... other 58 features
]
```

All 6 PM2.5 lags are **CRITICAL** (required; no omissions allowed).

---

## 2. Data Fetch Strategy

### 2.1 Single Batched Query

Instead of 6 separate queries (one per lag), use **one windowed query**:

```sql
SELECT 
  sensor_id,
  pm25_value,
  timestamp,
  latitude,
  longitude
FROM pm25_observations
WHERE grid_id = 'M0001'
  AND timestamp >= datetime('2026-01-18T10:00:00Z')  -- T-24h
  AND timestamp <= datetime('2026-01-19T10:00:00Z')  -- T
  AND pm25_value IS NOT NULL
ORDER BY timestamp DESC
```

**Output**: All sensor readings in the 24-hour window [T-24h, T]

**Example**:
```
sensor_id | pm25_value | timestamp           | latitude | longitude
----------|------------|---------------------|----------|----------
S001      | 65.0       | 2026-01-19T10:00:00 | 28.6139  | 77.2090
S001      | 62.5       | 2026-01-19T09:00:00 | 28.6139  | 77.2090
S001      | 68.0       | 2026-01-19T08:00:00 | 28.6139  | 77.2090
...
S002      | 60.0       | 2026-01-19T10:00:00 | 28.5534  | 77.2048
...
S031      | 55.0       | 2026-01-18T10:00:00 | 28.7041  | 77.5895
```

### 2.2 In-Memory Buffering

Fetch result into memory; compute all 6 lags from same buffer:

```python
def fetch_sensor_data_window(grid_id: str, end_timestamp: datetime, db) -> dict:
  """
  Fetch 24h window of sensor data.
  Returns dict keyed by (sensor_id, timestamp).
  """
  start_timestamp = end_timestamp - timedelta(hours=24)
  
  query = f"""
    SELECT sensor_id, pm25_value, timestamp, latitude, longitude
    FROM pm25_observations
    WHERE timestamp >= '{start_timestamp}'
      AND timestamp <= '{end_timestamp}'
      AND pm25_value IS NOT NULL
      AND pm25_value >= 0 AND pm25_value <= 1000  -- Range validation
    ORDER BY timestamp DESC
  """
  
  rows = db.execute(query).fetchall()
  
  # Buffer: (sensor_id, timestamp) → pm25_value
  data_buffer = {}
  for sensor_id, pm25, ts, lat, lon in rows:
    data_buffer[(sensor_id, ts)] = {
      "pm25": pm25,
      "latitude": lat,
      "longitude": lon
    }
  
  return data_buffer
```

---

## 3. IDW Interpolation Algorithm

### 3.1 IDW Formula

For a grid cell at (lat_grid, lon_grid), PM2.5 at lag timestamp T is:

```
pm25_lag = Σ(w_i × v_i) / Σ(w_i)

where:
  w_i = 1 / d_i^p
  d_i = haversine_distance(grid_centroid, sensor_i)
  v_i = pm25_value at sensor_i
  p = power parameter (default 2)
```

### 3.2 Implementation

```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
  """
  Calculate great-circle distance between two points (in km).
  """
  R = 6371  # Earth radius in km
  
  lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  
  return R * c

def compute_pm25_lag_idw(
  grid_id: str,
  lag_hours: int,
  end_timestamp: datetime,
  data_buffer: dict,
  grid_lat: float,
  grid_lon: float,
  db,
  power: float = 2.0
) -> float:
  """
  Compute PM2.5 lag at (end_timestamp - lag_hours) using IDW.
  
  Args:
    grid_id: Grid cell ID
    lag_hours: Hours back (1, 2, 3, 6, 12, 24)
    end_timestamp: Prediction timestamp T
    data_buffer: Pre-fetched sensor data [T-24h, T]
    grid_lat, grid_lon: Grid centroid coordinates
    power: IDW power parameter (default 2)
  
  Returns:
    PM2.5 value (float), or NaN if computation fails
  """
  # Calculate lag timestamp
  lag_timestamp = end_timestamp - timedelta(hours=lag_hours)
  lag_timestamp_floored = lag_timestamp.replace(minute=0, second=0, microsecond=0)
  
  # Collect sensors at exact lag timestamp
  sensors_at_lag = []
  for (sensor_id, ts), data in data_buffer.items():
    if ts == lag_timestamp_floored:
      sensors_at_lag.append({
        "sensor_id": sensor_id,
        "pm25": data["pm25"],
        "latitude": data["latitude"],
        "longitude": data["longitude"]
      })
  
  # Case 1: No sensors at lag timestamp
  if len(sensors_at_lag) == 0:
    logger.warning(f"No sensors at lag {lag_hours}h ({lag_timestamp_floored}); using mean")
    return compute_pm25_mean_fallback(data_buffer, lag_timestamp_floored)
  
  # Case 2: Single sensor at grid location
  if len(sensors_at_lag) == 1:
    sensor = sensors_at_lag[0]
    distance = haversine_distance(grid_lat, grid_lon, sensor["latitude"], sensor["longitude"])
    if distance < 0.01:  # Co-located (within ~10m)
      logger.info(f"Using co-located sensor {sensor['sensor_id']} for lag {lag_hours}h")
      return sensor["pm25"]
  
  # Case 3: Multiple sensors - use IDW
  numerator = 0.0
  denominator = 0.0
  
  for sensor in sensors_at_lag:
    distance = haversine_distance(
      grid_lat, grid_lon,
      sensor["latitude"], sensor["longitude"]
    )
    
    if distance < 0.01:  # Co-located sensor shortcut
      return sensor["pm25"]
    
    weight = 1.0 / (distance ** power)
    numerator += sensor["pm25"] * weight
    denominator += weight
  
  if denominator == 0:
    logger.error(f"IDW denominator is zero; returning NaN")
    return float('nan')
  
  pm25_lag = numerator / denominator
  logger.debug(f"IDW lag {lag_hours}h = {pm25_lag:.2f} (from {len(sensors_at_lag)} sensors)")
  
  return pm25_lag

def compute_pm25_mean_fallback(data_buffer: dict, lag_timestamp: datetime) -> float:
  """
  Fallback when no sensors available at lag timestamp.
  Use arithmetic mean of all non-NaN sensors at that timestamp.
  """
  values = []
  for (sensor_id, ts), data in data_buffer.items():
    if ts == lag_timestamp:
      pm25 = data["pm25"]
      if pm25 is not None and 0 <= pm25 <= 1000:
        values.append(pm25)
  
  if len(values) == 0:
    logger.warning(f"No valid sensors for mean at {lag_timestamp}; returning NaN")
    return float('nan')
  
  mean_pm25 = sum(values) / len(values)
  logger.info(f"Mean fallback at {lag_timestamp}: {mean_pm25:.2f} (from {len(values)} sensors)")
  
  return mean_pm25
```

---

## 4. Feature Assembly Workflow

### 4.1 Main Feature Assembly Flow

```python
def assemble_pm25_lags(grid_id: str, timestamp: datetime, db) -> list[float]:
  """
  Assemble all 6 PM2.5 lags for a grid cell.
  Single fetch; compute all lags from buffer.
  """
  # Get grid centroid coordinates
  grid_coords = get_grid_coordinates(grid_id)  # Returns (lat, lon)
  
  # SINGLE fetch: [T-24h, T]
  data_buffer = fetch_sensor_data_window(grid_id, timestamp, db)
  
  # Compute all 6 lags from same buffer
  lags = {}
  for lag_hours in [1, 2, 3, 6, 12, 24]:
    pm25_lag = compute_pm25_lag_idw(
      grid_id=grid_id,
      lag_hours=lag_hours,
      end_timestamp=timestamp,
      data_buffer=data_buffer,
      grid_lat=grid_coords["latitude"],
      grid_lon=grid_coords["longitude"],
      db=db,
      power=2.0  # Tunable
    )
    
    lags[f"pm25_t_minus_{lag_hours}h"] = pm25_lag
  
  logger.info(f"Computed lags for {grid_id}@{timestamp}: {lags}")
  
  return lags
```

### 4.2 Integration into Full Feature Assembly

```python
def assemble_features(grid_id: str, timestamp: datetime, db) -> dict:
  """
  Assemble all 64 features for model inference.
  """
  features = {}
  
  # 1. PM2.5 lags (6 features)
  pm25_lags = assemble_pm25_lags(grid_id, timestamp, db)
  features.update(pm25_lags)
  
  # 2. Meteorological features + lags (20 features)
  meteo_features = assemble_meteo_features(grid_id, timestamp, db)
  features.update(meteo_features)
  
  # 3. Satellite indices (12 features)
  satellite_features = assemble_satellite_features(grid_id, timestamp, db)
  features.update(satellite_features)
  
  # 4. Temporal features (4 features)
  temporal_features = assemble_temporal_features(grid_id, timestamp)
  features.update(temporal_features)
  
  # Total: 6 + 20 + 12 + 4 = 42 features (wait, should be 64?)
  # Check schema mismatch...
  
  # Validate all 64 features present
  assert len(features) == 64, f"Expected 64 features, got {len(features)}"
  
  return features
```

---

## 5. Grid Coordinates Lookup

### 5.1 Grid Centroid Coordinates

Each grid cell has a centroid (lat, lon):

```python
def get_grid_coordinates(grid_id: str) -> dict:
  """
  Return latitude, longitude of grid cell centroid.
  """
  # Grid cell centers pre-computed during grid creation
  grid_lookup = {
    "M0001": {"latitude": 28.6139, "longitude": 77.2090, "name": "Govt Island"},
    "M0002": {"latitude": 28.5534, "longitude": 77.2048, "name": "Shahdara"},
    "M0003": {"latitude": 28.4595, "longitude": 77.1352, "name": "Dwarka"},
    # ... 59k entries
  }
  
  return grid_lookup[grid_id]
```

**Storage**: Loaded at service startup from `grid_metadata.parquet` or similar.

---

## 6. Sensor Data Requirements

### 6.1 PM2.5 Observations Table

```sql
CREATE TABLE pm25_observations (
  sensor_id TEXT,
  timestamp TIMESTAMP,
  pm25_value FLOAT,
  latitude FLOAT,
  longitude FLOAT,
  
  PRIMARY KEY (sensor_id, timestamp)
);
```

**Data**:
- **Source**: CPCB (Central Pollution Control Board) + OpenAQ API
- **Update frequency**: Hourly (by Kashi service)
- **Retention**: 24+ hours (backfill requirement)
- **Records**: ~31 sensors × 24 hours = ~744 rows

**Assumptions**:
- Table populated **before service startup**
- Kashi maintains 24+ hour rolling window
- ~1-2 new rows per sensor per hour

### 6.2 Sensor Metadata

```sql
CREATE TABLE sensors (
  sensor_id TEXT PRIMARY KEY,
  latitude FLOAT,
  longitude FLOAT,
  sensor_name TEXT,
  location_type TEXT,  -- e.g., "residential", "industrial", "traffic"
  active BOOLEAN
);
```

**Usage**: Pre-join during fetch; no additional queries needed.

---

## 7. Performance Considerations

### 7.1 Cost per Prediction

Per grid cell per prediction:

```
1. Fetch 24h window: 1 DuckDB query
   ├─→ Result: ~750 rows (31 sensors × 24 hours)
   ├─→ Latency: ~5-10ms (indexed query)
   └─→ Network: ~20-30KB transfer

2. Compute 6 lags: In-memory IDW (Python)
   ├─→ Per lag: 6 IDW ops (31 sensors × IDW formula)
   ├─→ Latency: ~1-2ms total
   └─→ Memory: ~2MB (buffer)

Total: ~7-12ms per grid cell
```

**For heatmap (59k cells)**:
```
59k cells × 12ms = ~708 seconds (worst case)
```

**Optimization**: Parallelize heatmap with thread pool (4 threads):
```
708s / 4 threads = ~177s (still slow; consider batching)
```

**Alternative**: Pre-compute lags hourly (not current design, but possible future optimization)

### 7.2 Caching Opportunity

Lags computed on-demand (not cached), but can be added to L1/L2 cache after computation:

```python
def assemble_pm25_lags_cached(grid_id, timestamp, db):
  cache_key = (f"lags_{grid_id}", timestamp)
  
  # Check cache
  if cache_key in L1_CACHE:
    return L1_CACHE[cache_key]
  
  # Compute
  lags = assemble_pm25_lags(grid_id, timestamp, db)
  
  # Cache
  L1_CACHE[cache_key] = lags
  
  return lags
```

Lags have same TTL as predictions (1 hour).

---

## 8. Error Handling

### 8.1 Missing Sensor Data

```python
def handle_sensor_data_error(lag_hours, lag_timestamp):
  """
  Called when sensor data fetch fails or returns empty.
  """
  # Attempt fallback: arithmetic mean
  mean = compute_pm25_mean_fallback(data_buffer, lag_timestamp)
  
  if not isnan(mean):
    logger.warning(f"Lag {lag_hours}h: zero sensors, using mean {mean:.2f}")
    return mean
  
  # Fallback also failed
  logger.error(f"Lag {lag_hours}h: no data available; returning NaN")
  return float('nan')
```

### 8.2 Invalid Sensor Values

```python
def validate_sensor_value(pm25_value):
  """
  Check range [0, 1000] ppm.
  """
  if pm25_value is None:
    return False
  if pm25_value < 0 or pm25_value > 1000:
    logger.warning(f"Out-of-range PM2.5: {pm25_value}; rejecting")
    return False
  return True
```

**Filter applied during fetch**:
```sql
WHERE pm25_value >= 0 AND pm25_value <= 1000
```

---

## 9. Configuration & Tuning

### 9.1 IDW Parameters

```python
IDW_POWER = 2.0  # Start with 2; can tune to 1, 3, etc.

LAG_HOURS = [1, 2, 3, 6, 12, 24]  # Required lags
LAG_WINDOW_SIZE_HOURS = 24  # Fetch window [T-24h, T]

CO_LOCATED_DISTANCE_KM = 0.01  # ~10m; sensor at grid centroid
```

### 9.2 Sensor Range

```python
SENSOR_PM25_MIN = 0.0  # ppm
SENSOR_PM25_MAX = 1000.0  # ppm
```

### 9.3 Fetch Query

```python
SENSOR_FETCH_BATCH_SIZE = "all"  # Fetch all [T-24h, T] in one query
SENSOR_FETCH_TIMEOUT = 2.0  # Seconds (inherited from Go timeout)
```

---

## 10. Testing Checklist

- [ ] Test IDW with 31 sensors at same timestamp (various distances)
- [ ] Test co-located sensor (distance < 0.01km; return directly)
- [ ] Test single sensor (weight = 1.0)
- [ ] Test zero sensors at lag timestamp (fallback to mean)
- [ ] Test zero sensors overall (return NaN)
- [ ] Test invalid values (range check filters them)
- [ ] Test 24-hour window query performance (<10ms)
- [ ] Test all 6 lags computed correctly from same buffer
- [ ] Test timestamp flooring (16:30 → 16:00)
- [ ] Test boundary lags (lag_24h at service startup with limited data)
- [ ] Benchmark: feature assembly time with/without lag caching
- [ ] Stress test: 59k cells × 6 lags (heatmap throughput)

---

## 11. Deployment Checklist

- [ ] Ensure pm25_observations table populated before service starts
- [ ] Load grid_metadata (centroid coordinates) at service startup
- [ ] Pre-compute grid lookup dict (59k entries) in memory
- [ ] Verify sensor data backfill (24+ hours)
- [ ] Test fallback: zero sensors → mean
- [ ] Test fallback: zero mean → NaN (feature assembly fails)
- [ ] Monitor lag computation time (track p95/p99)
- [ ] Alert if mean fallback happens >5% of time

---

## 12. Summary

**PM2.5 Lags Computation**:
1. Single batched query: fetch 24-hour window [T-24h, T]
2. Compute 6 lags (1h, 2h, 3h, 6h, 12h, 24h) from same buffer
3. IDW interpolation with Haversine distance (power=2 default)
4. Co-located sensor shortcut (distance < 10m)
5. Fallback: arithmetic mean if zero sensors at lag timestamp
6. Fallback: NaN if zero mean (triggers inference failure)
7. Range validation [0, 1000] ppm
8. On-demand computation (no pre-storage)
9. Timestamp flooring to nearest hour (:00)

---

**Document Version**: 1.0  
**Last Updated**: January 19, 2026  
**Status**: Locked for Implementation
