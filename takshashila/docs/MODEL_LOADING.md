# Takshashila: Model Loading & Versioning Strategy

**Date**: January 19, 2026  
**Status**: Final Design (Locked)  
**Audience**: Backend engineers, DevOps

---

## Executive Summary

This document specifies how the Takshashila inference service loads, caches, and manages multiple LightGBM model versions. The strategy prioritizes **simplicity for MVP** with a **lazy-load fallback chain** (load current version at startup, load fallbacks on-demand). Model versions are persisted as `.txt` files; current version is tracked in DuckDB; failed versions are recorded in `manifest.yml` for skip-list management.

**Key Design Principles**:
1. **Lazy loading** — only current model in memory; fallbacks loaded on first use
2. **Synchronous fallback** — user waits for fallback model to load from disk (~1-2s)
3. **Persistent skip list** — failed versions recorded in manifest.yml across restarts
4. **Cached predictions fallback** — serve recent predictions if all models fail
5. **Single process for MVP** — Redis shared store designed but not implemented yet

---

## 1. Model Storage Layout

### 1.1 Directory Structure

```
artifacts/v1/models/
├── model_v124.txt     # Current (in memory)
├── model_v123.txt     # Fallback 1
└── model_v122.txt     # Fallback 2
```

**Format**: LightGBM `.txt` booster file (text format, ~20-100MB per model)

**Versioning**: Linear versioning (v122, v123, v124, ...) — no branches or tags.

### 1.2 Model Metadata

Current model version tracked in DuckDB `current_version` table:

```sql
CREATE TABLE current_version (
  key TEXT PRIMARY KEY,
  value TEXT
);

-- Two rows:
INSERT INTO current_version VALUES ('predictions', 'v124');  -- Prediction table version
INSERT INTO current_version VALUES ('model', 'v124');        -- Model version
```

---

## 2. Model Loading Strategy

### 2.1 Startup (Service Initialization)

At service startup, load **only the current model version**:

```python
class ModelLoader:
  def __init__(self, db: duckdb.DuckDBPyConnection):
    self.db = db
    self.models = {}  # In-memory cache
    self.current_version = None
    self.skipped_versions = set()
  
  def load_current_model(self):
    """Load current model version at startup."""
    current_version = self.db.execute(
      "SELECT value FROM current_version WHERE key='model'"
    ).fetchall()[0][0]
    
    # Load skip list from manifest
    manifest = self._load_manifest()
    self.skipped_versions = set(manifest.get("failed_model_versions", []))
    
    # Load current model
    try:
      model_path = f"artifacts/v1/models/model_{current_version}.txt"
      model = lgb.Booster(model_file=model_path)
      self.models[current_version] = model
      self.current_version = current_version
      logger.info(f"Loaded current model: {current_version}")
      return True
    except Exception as e:
      logger.error(f"Failed to load current model {current_version}: {e}")
      # Current model failed at startup → service fails to start
      raise ModelLoadError(f"Cannot start without current model: {e}")
  
  def _load_manifest(self) -> dict:
    """Load manifest.yml to read skip list."""
    with open("manifest.yml", "r") as f:
      return yaml.safe_load(f) or {}
  
  def _save_manifest(self, manifest: dict):
    """Save manifest.yml with updated skip list."""
    with open("manifest.yml", "w") as f:
      yaml.dump(manifest, f)
```

**Startup Behavior**:
- Load current model (if fails → service fails to start)
- Load skip list from manifest.yml (failed versions from previous runs)
- Memory usage: ~50-100MB (only current model)

### 2.2 On-Demand Fallback Loading

When current model fails during prediction, attempt fallback chain **synchronously**:

```python
def predict_with_fallback(self, features: np.ndarray) -> np.ndarray | None:
  """
  Attempt prediction on current model; fall back to v123, v122 on failure.
  Loads fallback from disk synchronously (user waits ~1-2s).
  """
  fallback_chain = [self.current_version, "v123", "v122"]
  
  for version in fallback_chain:
    # Skip versions marked as failed in manifest
    if version in self.skipped_versions:
      logger.warning(f"Skipping {version} (in skip list)")
      continue
    
    # Load from memory or disk
    if version not in self.models:
      logger.info(f"Loading fallback model {version} from disk...")
      try:
        model_path = f"artifacts/v1/models/model_{version}.txt"
        model = lgb.Booster(model_file=model_path)
        self.models[version] = model  # Cache in memory
        logger.info(f"Loaded {version} from disk ({len(self.models)} models in memory)")
      except Exception as e:
        logger.error(f"Failed to load fallback {version}: {e}")
        self._mark_version_failed(version)  # Add to skip list
        continue
    
    # Attempt prediction
    try:
      prediction = self.models[version].predict(features)
      if version != self.current_version:
        logger.warning(f"Fallback to {version}; current={self.current_version}")
      return prediction
    except Exception as e:
      logger.error(f"Prediction on {version} failed: {e}")
      self._mark_version_failed(version)
      continue
  
  # All fallbacks exhausted
  logger.critical("All model versions failed or skipped")
  return None

def _mark_version_failed(self, version: str):
  """Record failed version in manifest.yml skip list."""
  manifest = self._load_manifest()
  failed_versions = set(manifest.get("failed_model_versions", []))
  failed_versions.add(version)
  manifest["failed_model_versions"] = list(failed_versions)
  manifest["last_failed_model"] = {"version": version, "timestamp": datetime.utcnow().isoformat()}
  self._save_manifest(manifest)
  self.skipped_versions.add(version)
```

**Fallback Flow**:
1. Current model fails → attempt load v123 from disk
2. v123 loads successfully → cache in memory, use for this + future requests
3. If v123 also fails → attempt v122
4. If all fail → return None (inference returns 503)

**User Experience**:
- First fallback request: ~1-2s delay (disk load + predict)
- Subsequent requests on fallback: Normal latency (in-memory cache)

---

## 3. Version Tracking & Skip List

### 3.1 Current Version Table (DuckDB)

```sql
SELECT * FROM current_version;
```

Output:
```
key           | value
--------------+-------
predictions   | v124
model         | v124
```

Updated by:
- Ops team when pushing new model: `UPDATE current_version SET value='v125' WHERE key='model'`
- Requires service restart to take effect

### 3.2 Failed Version Skip List (manifest.yml)

```yaml
model_version: "v124"
failed_model_versions:
  - "v124"   # Marked failed after persistent crashes
  - "v122"   # Failed to load at startup
prediction_version: "v124"
last_failed_model:
  version: "v124"
  timestamp: "2026-01-19T10:15:30Z"
  reason: "OOM during prediction"
```

**Behavior**:
- Service reads skip list at startup and on-demand
- When a version fails, it's added to list
- Skipped versions not attempted in fallback chain
- Persists across service restarts (ops team must clear manually if version is fixed)

**Manual Reset**:
```bash
# Clear skip list (if fixed version)
vim manifest.yml
# Remove version from failed_model_versions list
# Restart service
```

---

## 4. Model Caching in Memory

### 4.1 Lazy Cache

```python
self.models = {
  "v124": <loaded LightGBM booster>  # Always present
  # "v123": loaded only on first fallback request
  # "v122": loaded only if v123 fails
}
```

**Memory Growth**:
- Current: ~50-100MB (v124 always loaded)
- After 1st fallback: ~100-200MB (v124 + v123)
- After 2nd fallback: ~150-300MB (all 3)

**Cleanup**: Models stay in memory until service restart. No unloading after fallback.

### 4.2 Cache Hit Rate Metric

Track when fallback is needed:

```python
class ModelMetrics:
  def __init__(self):
    self.total_predictions = 0
    self.current_model_successes = 0
    self.fallback_v123_hits = 0
    self.fallback_v122_hits = 0
    self.all_models_failed = 0
  
  def record_prediction(self, version_used: str, success: bool):
    self.total_predictions += 1
    if version_used == "v124":
      self.current_model_successes += 1
    elif version_used == "v123":
      self.fallback_v123_hits += 1
    elif version_used == "v122":
      self.fallback_v122_hits += 1
    
    if not success:
      self.all_models_failed += 1

def get_metrics(self) -> dict:
  return {
    "total_predictions": self.total_predictions,
    "current_model_rate": self.current_model_successes / max(1, self.total_predictions),
    "fallback_v123_rate": self.fallback_v123_hits / max(1, self.total_predictions),
    "fallback_v122_rate": self.fallback_v122_hits / max(1, self.total_predictions),
    "failure_rate": self.all_models_failed / max(1, self.total_predictions),
  }
```

---

## 5. Graceful Degradation: All Models Failed

### 5.1 Fallback to Cached Predictions

When **all model versions fail** (all skip-listed or load errors):

```python
def predict_with_cached_fallback(self, grid_id: str, timestamp: datetime) -> dict | None:
  """
  Attempt model prediction; if all models fail, return most recent prediction for this grid.
  """
  prediction = self.predict_with_fallback(features)
  
  if prediction is None:
    logger.warning(f"All models failed; attempting cached fallback for grid {grid_id}")
    
    # Fetch most recent prediction from predictions table
    current_version = self.db.execute(
      "SELECT value FROM current_version WHERE key='predictions'"
    ).fetchall()[0][0]
    
    cached = self.db.execute(f"""
      SELECT pm25_pred, confidence, model_version, timestamp
      FROM {current_version}
      WHERE grid_id = {grid_id}
      ORDER BY timestamp DESC
      LIMIT 1
    """).fetchall()
    
    if cached:
      pm25, conf, model_ver, ts = cached[0]
      logger.warning(f"Serving cached prediction from {ts} (model {model_ver})")
      return {
        "pm25_pred": pm25,
        "confidence": conf * 0.5,  # Reduce confidence due to staleness
        "model_version": model_ver,
        "prediction_age_seconds": (datetime.utcnow() - ts).total_seconds(),
        "served_from_cache": True
      }
    
    # No cached prediction available either
    logger.critical(f"No prediction available (all models failed, no cache)")
    return None
```

**Behavior**:
1. **All models fail** → attempt to serve cached prediction
2. **Cached prediction exists** → return with reduced confidence (×0.5)
3. **No cached prediction** → return 503 Service Unavailable

### 5.2 Response with Cached Fallback

```json
{
  "grid_id": "M0001",
  "timestamp": "2026-01-19T10:00:00Z",
  "pm25_prediction": 65.3,
  "confidence": 0.46,
  "model_version": "v124",
  "prediction_age_seconds": 3600,
  "served_from_cache": true,
  "warning": "All model versions failed; serving cached prediction from 1 hour ago"
}
```

---

## 6. Metrics & Monitoring

### 6.1 Model Load Metrics

**At startup and on fallback load:**

```python
def load_model_and_record(self, version: str) -> bool:
  start = time.time()
  try:
    model_path = f"artifacts/v1/models/model_{version}.txt"
    model = lgb.Booster(model_file=model_path)
    load_time_ms = (time.time() - start) * 1000
    
    logger.info(f"Loaded model {version} in {load_time_ms:.1f}ms")
    
    # Record metric
    self.metrics["model_load_time_ms"] = {
      "version": version,
      "time_ms": load_time_ms,
      "timestamp": datetime.utcnow().isoformat()
    }
    return model
  except Exception as e:
    load_time_ms = (time.time() - start) * 1000
    logger.error(f"Failed to load {version} after {load_time_ms:.1f}ms: {e}")
    return None
```

### 6.2 Prediction Metrics

Track per-version success rate and latency:

```python
metrics = {
  "current_model_success_rate": 0.98,         # % of requests using v124
  "fallback_v123_hit_rate": 0.015,            # % of requests falling back to v123
  "fallback_v122_hit_rate": 0.003,            # % of requests falling back to v122
  "model_failure_rate": 0.002,                # % of requests with all models failed
  "average_prediction_latency_ms": {
    "v124": 12,
    "v123": 14,
    "v122": 15
  },
  "model_load_time_ms": {
    "v124": 450,    # Loaded at startup
    "v123": 520,    # Loaded on first fallback
    "v122": 510
  },
  "cached_fallback_uses": 5,  # # of times cached prediction served
  "total_predictions": 1000000
}
```

### 6.3 Alerts

| Metric | Alert Threshold | Action |
|---|---|---|
| Fallback hit rate (v123+v122) | >1% | Investigate current model health |
| Model failure rate | >0.1% | Page on-call engineer |
| Model load time | >5s | Check disk I/O; consider pre-loading |
| Cached fallback uses | >0 | Unusual; check model failures + disk space |

---

## 7. MVP Implementation Checklist

- [ ] Create `artifacts/v1/models/` directory with v124.txt, v123.txt, v122.txt
- [ ] Add `current_version` table to DuckDB schema (keys: 'model', 'predictions')
- [ ] Implement `ModelLoader` class (load current at startup, lazy-load fallbacks)
- [ ] Implement fallback chain with skip-list (manifest.yml)
- [ ] Implement cached fallback (serve from predictions table)
- [ ] Add structured logging for model loads + predictions
- [ ] Add `/health/model` endpoint (returns current version + memory status)
- [ ] Add `/metrics/model` endpoint (returns cache hit rates, latencies, failures)
- [ ] Test failure scenarios (remove model file, corrupt file, OOM)

---

## 8. Future Enhancements (Not MVP)

### 8.1 Multi-Process Model Sharing (Q10 Option B)

For future scaling to multiple Python processes:

**Redis-based model store** (design only, not implemented):

```python
# Writer (single process)
redis_client = redis.Redis(host='localhost', port=6379)

# Check if model exists in Redis
if redis_client.exists(f"models:{version}"):
  # Load from Redis (faster than disk)
  model_blob = redis_client.get(f"models:{version}")
  model = pickle.loads(model_blob)
else:
  # Load from disk, cache in Redis
  model = lgb.Booster(model_file=f"artifacts/v1/models/model_{version}.txt")
  redis_client.set(f"models:{version}", pickle.dumps(model), ex=24*3600)

# Multiple readers can load same version from Redis
```

**Rationale**: Avoid duplicate disk I/O when multiple processes load same model.

**Trade-off**: Requires Redis coordination; for MVP (single process), unnecessary.

### 8.2 Hot Model Reload

For future without service restart:

```python
# HTTP endpoint
@app.route("/admin/reload-model", methods=["POST"])
def reload_model():
  """Reload current model version from disk."""
  new_version = request.json.get("version")
  # Load new version
  # Atomic swap with threading lock
  # Return success
```

**Rationale**: Avoid downtime when deploying new model.

**Trade-off**: Requires careful synchronization; for MVP, just restart service.

### 8.3 Model Validation at Load Time

For future robustness:

```python
def load_and_validate_model(self, version: str) -> bool:
  """Load model and test-predict on dummy features."""
  try:
    model = lgb.Booster(model_file=f"artifacts/v1/models/model_{version}.txt")
    
    # Test prediction on dummy features
    dummy_features = np.zeros((1, 64))  # 64 features
    pred = model.predict(dummy_features)
    
    if pred[0] < 0 or pred[0] > 1000:  # Sanity check
      logger.error(f"Invalid prediction on dummy features: {pred}")
      return False
    
    return True
  except Exception as e:
    logger.error(f"Model validation failed: {e}")
    return False
```

**Rationale**: Catch corrupt models before they're used in production.

**Trade-off**: Adds ~100-500ms to load time; for MVP, skip.

---

## 9. Configuration

### 9.1 Code Config

```python
MODEL_DIR = "artifacts/v1/models"
CURRENT_MODEL_VERSION = None  # Loaded from DuckDB at runtime
FALLBACK_CHAIN = ["v124", "v123", "v122"]  # Versions to try in order
MODEL_LOAD_TIMEOUT_SECONDS = 30
PREDICTION_TIMEOUT_SECONDS = 10
CACHE_PREDICTIONS_FALLBACK = True  # Serve cached if all models fail
CACHE_FALLBACK_CONFIDENCE_MULTIPLIER = 0.5  # Reduce confidence on cached prediction
```

### 9.2 Environment Config

```bash
# .env
MODEL_VERSION=v124
MODEL_DIR=artifacts/v1/models
SKIP_LIST_SOURCE=manifest  # manifest or duckdb
```

---

## 10. Summary Decision Tree

```
Service starts
  │
  ├─→ Read current model version from DuckDB current_version table
  │
  ├─→ Load skip list from manifest.yml (failed versions)
  │
  ├─→ Attempt load current model from disk
  │    └─→ Failed? Service fails to start (no fallback at startup)
  │
  └─→ Service ready (only current model in memory)

Request arrives for prediction
  │
  ├─→ Assemble features
  │
  ├─→ Attempt predict on current model
  │    │
  │    ├─→ Success? Return prediction
  │    │
  │    └─→ Failed? Attempt fallback chain
  │        │
  │        ├─→ v123 in skip list? Skip to v122
  │        │
  │        ├─→ v123 not loaded? Load from disk synchronously
  │        │
  │        ├─→ v123 predict success? Return prediction (mark v123 as good)
  │        │
  │        └─→ v123 failed? Add to skip list, try v122
  │            │
  │            └─→ v122 same as v123 logic
  │
  ├─→ All models failed?
  │    │
  │    ├─→ Fetch most recent prediction from predictions table
  │    │
  │    ├─→ Found? Return with reduced confidence + warning
  │    │
  │    └─→ Not found? Return 503 Service Unavailable
  │
  └─→ Return prediction response (200 or 503)
```

---

**Document Version**: 1.0  
**Last Updated**: January 19, 2026  
**Status**: Locked for Implementation
