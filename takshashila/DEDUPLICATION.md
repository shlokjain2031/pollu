# Takshashila: Concurrent Request Deduplication

**Date**: January 19, 2026  
**Status**: Final Design (Locked)  
**Audience**: Backend engineers, DevOps

---

## Executive Summary

This document specifies **single-flight request deduplication**: when multiple concurrent requests ask for the same prediction, only one computation happens; all waiters share the result. The system uses **per-cell locking** (not per-heatmap), allowing fine-grained parallelism. Go routing layer enforces **point API priority** over heatmap requests via queue ordering.

**Key Design Principles**:
1. **Per-cell-per-timestamp locking** — `(grid_id, timestamp)` → `threading.Lock()`
2. **Broad lock scope** — protects feature assembly + model predict
3. **Request-specific timeout** — inherit Go timeout (2s point, 10s heatmap)
4. **Shared result via Future** — waiters get same result without recompute
5. **Lock cleanup immediate** — delete from dict after release
6. **Heatmap is per-cell loop** — 100 cells = 100 independent single-flight requests
7. **Point API priority** — Go enqueues point ahead of heatmap
8. **No orphan cleanup** — process pool timeout handles crashes

---

## 1. Deduplication Architecture

### 1.1 Core Idea: Single-Flight Pattern

```
Request A for (M0001, 10:00) arrives
  ├─→ No lock exists
  ├─→ Create lock, acquire it
  ├─→ Compute: assemble features + model predict
  ├─→ Store result in Future
  ├─→ Release lock

Request B for (M0001, 10:00) arrives (while A is computing)
  ├─→ Lock exists
  ├─→ Wait on lock (blocking)
  ├─→ Once lock released, read Future.result()
  ├─→ Return same result as A (no recompute)

Request C for (M0002, 10:00) arrives
  ├─→ Different cell (M0002 vs M0001)
  ├─→ Create separate lock
  ├─→ Compute independently
```

**Result**: Request A + B share 1 computation; Request C is separate.

### 1.2 Data Structures

```python
import threading
from concurrent.futures import Future
from typing import Dict, Tuple

class Deduplicator:
  """
  Manages per-cell-per-timestamp locks and result sharing.
  """
  def __init__(self):
    # Lock dict: (grid_id, timestamp) → threading.Lock()
    self.locks: Dict[Tuple[str, str], threading.Lock] = {}
    
    # Result dict: (grid_id, timestamp) → concurrent.futures.Future
    self.results: Dict[Tuple[str, str], Future] = {}
    
    # Protect access to dicts themselves
    self.dict_lock = threading.Lock()
  
  def get_or_create_lock(self, grid_id: str, timestamp: str) -> threading.Lock:
    """
    Get existing lock or create new one.
    Thread-safe.
    """
    key = (grid_id, timestamp)
    
    with self.dict_lock:
      if key not in self.locks:
        self.locks[key] = threading.Lock()
      return self.locks[key]
  
  def try_acquire(self, grid_id: str, timestamp: str, timeout: float) -> bool:
    """
    Try to acquire lock for this (grid_id, timestamp).
    Returns True if acquired; False if timeout.
    """
    lock = self.get_or_create_lock(grid_id, timestamp)
    return lock.acquire(timeout=timeout)
  
  def store_result(self, grid_id: str, timestamp: str, result) -> None:
    """Store result in Future; unblock waiters."""
    key = (grid_id, timestamp)
    
    with self.dict_lock:
      if key not in self.results:
        self.results[key] = Future()
      
      self.results[key].set_result(result)
  
  def get_result(self, grid_id: str, timestamp: str, timeout: float):
    """
    Retrieve result (blocks until available or timeout).
    """
    key = (grid_id, timestamp)
    
    with self.dict_lock:
      if key not in self.results:
        self.results[key] = Future()
      future = self.results[key]
    
    return future.result(timeout=timeout)
  
  def release_and_cleanup(self, grid_id: str, timestamp: str) -> None:
    """
    Release lock and cleanup from dict.
    """
    key = (grid_id, timestamp)
    
    lock = self.get_or_create_lock(grid_id, timestamp)
    lock.release()
    
    # Delete lock immediately (no longer needed)
    with self.dict_lock:
      if key in self.locks:
        del self.locks[key]
      # Leave result in dict for future cache hits

deduplicator = Deduplicator()
```

---

## 2. Request Flow: Single-Flight Deduplication

### 2.1 Compute Request (Lock Holder)

```python
def predict_with_dedup(grid_id: str, timestamp: str, timeout: float):
  """
  Main prediction flow with deduplication.
  timeout inherited from Go (2s for point, 10s for heatmap).
  """
  # Try to acquire lock
  acquired = deduplicator.try_acquire(grid_id, timestamp, timeout=timeout)
  
  if not acquired:
    # Another request is computing; wait for result
    try:
      result = deduplicator.get_result(grid_id, timestamp, timeout=timeout)
      logger.info(f"Dedup hit: waited for result of {grid_id}@{timestamp}")
      return result
    except concurrent.futures.TimeoutError:
      logger.error(f"Dedup timeout waiting for {grid_id}@{timestamp}")
      return None  # Return 503 to Go
  
  # Lock acquired; we are the lock holder
  logger.info(f"Lock acquired; computing {grid_id}@{timestamp}")
  
  try:
    # Broad scope: feature assembly + model
    features = assemble_features(grid_id, timestamp, timeout=timeout * 0.8)
    prediction = predict_model(features, timeout=timeout * 0.2)
    
    # Store result for waiters
    deduplicator.store_result(grid_id, timestamp, prediction)
    logger.info(f"Computation complete; {grid_id}@{timestamp}")
    
    return prediction
  
  except Exception as e:
    logger.error(f"Computation failed: {e}")
    # Store exception in Future; waiters will see it
    deduplicator.results[(grid_id, timestamp)].set_exception(e)
    return None
  
  finally:
    # Always release lock
    deduplicator.release_and_cleanup(grid_id, timestamp)
```

### 2.2 Waiting Request (Waiter)

When a request finds lock already held:

```python
# In predict_with_dedup() above:
if not acquired:
  # Lock already held by another request
  logger.info(f"Waiting for {grid_id}@{timestamp} (lock held)")
  
  try:
    # Block until result available or timeout
    result = deduplicator.get_result(grid_id, timestamp, timeout=timeout)
    logger.info(f"Dedup cache hit: {grid_id}@{timestamp}")
    return result
  
  except concurrent.futures.TimeoutError:
    logger.error(f"Timeout waiting for {grid_id}@{timestamp}")
    return None  # Return 503 to Go
  except Exception as e:
    logger.error(f"Computation failed (waiter sees exception): {e}")
    return None
```

**Behavior**:
- **Lock holder computes** → stores result in Future
- **Waiters block on Future** → unblock once result available
- **All waiters get same result** (no duplication)

---

## 3. Broad Lock Scope

Lock protects **entire computation**: feature assembly + model predict

```python
def predict_with_broad_scope(grid_id: str, timestamp: str):
  """
  Broad scope: entire feature pipeline locked.
  Prevents duplicate feature fetches AND duplicate model calls.
  """
  acquired = deduplicator.try_acquire(grid_id, timestamp, timeout=2.0)
  
  if not acquired:
    # Wait for result
    return deduplicator.get_result(grid_id, timestamp, timeout=2.0)
  
  try:
    # ALL of this is protected by the lock:
    
    # 1. Feature assembly (can be slow; don't duplicate)
    pm25_lags = compute_pm25_lags(grid_id, timestamp)
    meteo_features = fetch_meteo_features(grid_id, timestamp)
    satellite_features = fetch_satellite_features(grid_id, timestamp)
    temporal_features = compute_temporal_features(grid_id, timestamp)
    
    features = assemble_vector([pm25_lags, meteo_features, satellite_features, temporal_features])
    
    # 2. Model prediction (most expensive; definitely don't duplicate)
    prediction = model.predict(features)
    
    # 3. Store for cache + waiters
    deduplicator.store_result(grid_id, timestamp, prediction)
    
    return prediction
  
  finally:
    deduplicator.release_and_cleanup(grid_id, timestamp)
```

**Rationale**:
- Feature assembly can be slow (DuckDB reads, IDW computation)
- Protecting both prevents duplicate expensive I/O
- Waiters don't re-fetch; they just wait for result

---

## 4. Timeout Behavior

### 4.1 Timeout Sources

Lock timeout **inherited from Go**:

```python
def handle_inference_request(grid_id, timestamp, request_type):
  """
  Called by Python service from Go.
  Go has already enforced timeout; we respect it.
  """
  if request_type == "point_api":
    timeout = 2.0  # Go enforced 2s
  elif request_type == "heatmap_api":
    timeout = 10.0  # Go enforced 10s
  
  # Pass timeout to predict
  result = predict_with_dedup(grid_id, timestamp, timeout=timeout)
  return result
```

### 4.2 Two-Level Timeout

**Level 1: Go timeout** (wall-clock)
- Go enforces 2s for point API
- Go enforces 10s for heatmap API
- If Python takes longer, Go kills request

**Level 2: Python lock timeout**
- Lock.acquire(timeout=2.0) blocks for up to 2s
- If lock not acquired in 2s, give up
- Return error to Go (which has already timed out anyway)

```python
def predict_with_dedup(grid_id, timestamp, timeout):
  # Lock holder timeout
  if lock_holder:
    # Start internal clock
    start = time.time()
    
    # If compute takes >timeout, process pool will timeout
    features = assemble_features(..., timeout=timeout * 0.8)
    prediction = predict_model(..., timeout=timeout * 0.2)
    
    elapsed = time.time() - start
    if elapsed > timeout:
      logger.warning(f"Computation took {elapsed}s; exceeded timeout {timeout}s")
      # Store partial result or error
  
  # Waiter timeout
  else:
    result = deduplicator.get_result(..., timeout=timeout)
    # Unblocks when result available or timeout
```

---

## 5. Heatmap: Per-Cell Single-Flight (No Bulk Coordination)

### 5.1 Key Design Decision

**Heatmap is NOT special** — it's just a loop of point requests.

**No heatmap-level locking** — each cell independently deduped.

```python
def get_heatmap(timestamp: str, timeout: float = 10.0):
  """
  Fetch all 59k cells at timestamp.
  Each cell is independent; use shared deduplication.
  """
  all_cells = get_all_grid_ids()  # 59k cells
  results = {}
  
  # Loop: fetch each cell
  # Cells that are concurrent → dedup
  # Cells from different heatmap requests → also dedup
  
  for grid_id in all_cells:
    # Call same predict_with_dedup() for each cell
    # If Request A is computing cell M0001, and Request B also asks for M0001,
    # they BOTH use same lock → only one computation
    
    prediction = predict_with_dedup(grid_id, timestamp, timeout=timeout)
    results[grid_id] = prediction
  
  return results
```

### 5.2 Heatmap Deduplication Across Requests

```
Time T:
  Heatmap Request 1: asks for all 59k cells
    ├─→ Cell M0001: lock acquired, compute
    ├─→ Cell M0002: lock acquired, compute
    └─→ Cell M0003: lock acquired, compute
  
  (meanwhile, in parallel)
  
  Heatmap Request 2: asks for same 59k cells (identical request)
    ├─→ Cell M0001: WAIT (Request 1 computing)
    ├─→ Cell M0002: WAIT (Request 1 computing)
    └─→ Cell M0003: WAIT (Request 1 computing)
  
  Result:
    Request 1 computes all 59k cells once
    Request 2 waits on all locks, gets results from Request 1
    (Perfect dedup; 100% sharing)
```

**No explicit heatmap-level lock** — per-cell locks achieve the same result naturally.

---

## 6. Point API Priority

### 6.1 Go-Level Prioritization

Go maintains **separate queues** (or priority levels) for point vs heatmap:

```go
// Go queue structure
type RequestQueue struct {
  pointAPIJobs chan Job    // Higher priority
  heatmapJobs chan Job     // Lower priority
}

// Enqueue logic
func (q *RequestQueue) Enqueue(job Job) {
  if job.Type == "point_api" {
    q.pointAPIJobs <- job
  } else if job.Type == "heatmap_api" {
    q.heatmapJobs <- job
  }
}

// Dequeue: prioritize point API
func (q *RequestQueue) Dequeue() Job {
  select {
  case job := <-q.pointAPIJobs:
    return job
  case job := <-q.heatmapJobs:
    return job
  }
}
```

**Behavior**:
- Point API jobs go to high-priority queue
- Heatmap jobs go to low-priority queue
- Go pulls from point API first; heatmap if point empty

### 6.2 Python-Level Observance

Python sees a **single bounded job queue** (from Go's perspective):

```python
# Python just has one queue
job_queue = Queue(maxsize=100)

# Go enqueues point before heatmap
# Python processes in order: point_api_job_1, point_api_job_2, heatmap_job_1, ...

# Within the queue, point jobs are ahead
```

**No special logic in Python** — just follow queue order from Go.

### 6.3 Point API Interruptibility

If a **point API request and heatmap request compete for same cell**:

```
Heatmap is computing cell M0001 (lock held)
Point API request for M0001 arrives

Current behavior: Point API waits on lock (same as any waiter)

Desired: Point API should "have priority" (shorter wait)
```

**Implementation**: No special preemption needed

- Point API is enqueued ahead (Go level)
- By the time Point API runs, Heatmap might already be computing M0001
- Point API waits (fair, unavoidable)
- OR: Heatmap hasn't started M0001 yet; Point API gets it first

Priority is enforced at **enqueue time** (Go), not at compute time.

---

## 7. Lock Cleanup & Memory Management

### 7.1 Immediate Cleanup After Release

```python
def release_and_cleanup(self, grid_id: str, timestamp: str) -> None:
  """
  Release lock and delete from dict immediately.
  Results remain (cached by L1/L2).
  """
  key = (grid_id, timestamp)
  
  lock = self.get_or_create_lock(grid_id, timestamp)
  lock.release()
  
  # Delete lock immediately (no TTL, no delayed cleanup)
  with self.dict_lock:
    if key in self.locks:
      del self.locks[key]  # ← Freed immediately
    
    # Results dict stays populated (used by L1 cache check)
    # It will be cleared by L1 eviction (TTL-based)
```

**Memory Impact**:
- Lock dict size: at most = # of concurrent computations
- Typical: ~10-50 locks in dict at any time
- Memory: negligible (each lock ~200 bytes)

### 7.2 Result Cleanup

Results remain in `self.results` dict for cache coherence:

```python
# Don't delete results immediately
# They're checked by cache before locking:

def predict_with_cache_check(grid_id, timestamp, timeout):
  key = (grid_id, timestamp)
  
  # Check if result already in dict (cached by another request)
  with deduplicator.dict_lock:
    if key in deduplicator.results:
      try:
        result = deduplicator.results[key].result(timeout=0.1)
        logger.info("Cache hit from results dict")
        return result
      except:
        pass  # Result not ready; proceed to compute
  
  # If not cached, proceed to dedup flow
  return predict_with_dedup(grid_id, timestamp, timeout)
```

**Cleanup happens elsewhere**:
- L1 eviction (TTL-based) removes from cache
- Results dict eventually clears as cache entries expire

---

## 8. Crash Handling: Orphaned Locks

### 8.1 Lock Holder Crashes

If Request 1 (lock holder) crashes mid-compute:

```
Request 1 computes (lock held)
  ├─→ Assemble features (OK)
  └─→ Process pool times out (no response after 5s)
      └─→ Process killed

Process killed, but lock NOT released
Request 2 waits forever on lock ✗
```

**Solution**: **Process pool timeout kills job**

```python
# Process pool has implicit timeout (from Go)
future = process_pool.apply_async(
  model.predict,
  [features]
)

try:
  result = future.get(timeout=2.0)  # Go's timeout (point API)
except TimeoutError:
  logger.error("Process timeout; job killed")
  # Process killed
  # Lock holder (Request 1) exits exception handler
  # Lock released in finally block
  return None
```

**No orphaned lock** because:
1. Python gets Go's timeout
2. Process killed on timeout
3. Request 1's finally block releases lock
4. Request 2 unblocks

### 8.2 No Watchdog/Cleanup Thread

**Not needed** because:
- Go enforces timeout (wall-clock)
- Python enforces timeout (process pool)
- Exception handler always releases lock (finally block)

```python
def predict_with_dedup(...):
  acquired = deduplicator.try_acquire(...)
  
  if not acquired:
    # Wait on Future
    ...
  
  try:
    # Compute (with timeout)
    ...
  except:
    # Store exception
    ...
  finally:
    # ALWAYS runs, even on exception or timeout
    deduplicator.release_and_cleanup(...)  # ← Guaranteed
```

---

## 9. Metrics & Monitoring

### 9.1 Deduplication Metrics

```python
class DeduplicationMetrics:
  def __init__(self):
    self.total_requests = 0
    self.lock_acquired_count = 0  # Lock holders
    self.lock_waited_count = 0    # Waiters
    self.lock_wait_times = []     # Histogram
    self.dedup_ratio = 0.0        # waited / total
    self.lock_timeout_count = 0
    self.results_cache_hits = 0
    self.lock_dict_size = 0

metrics = DeduplicationMetrics()

def predict_with_dedup(grid_id, timestamp, timeout):
  metrics.total_requests += 1
  
  start = time.time()
  acquired = deduplicator.try_acquire(grid_id, timestamp, timeout=timeout)
  
  if acquired:
    metrics.lock_acquired_count += 1
  else:
    metrics.lock_waited_count += 1
    wait_time = time.time() - start
    metrics.lock_wait_times.append(wait_time)
  
  # ... rest of flow ...
  
  metrics.lock_dict_size = len(deduplicator.locks)
  metrics.dedup_ratio = metrics.lock_waited_count / max(1, metrics.total_requests)
```

### 9.2 Dashboard Metrics

```python
dedup_metrics = {
  "total_requests_per_sec": 0,
  "lock_acquisition_rate": 0.0,  # % of requests that acquire lock
  "dedup_ratio": 0.0,             # % of requests that wait (1.0 = perfect dedup)
  
  "lock_wait_time_ms": {
    "p50": 0.0,
    "p95": 0.0,
    "p99": 0.0,
  },
  
  "lock_timeout_count_per_min": 0,
  "lock_dict_size": 0,            # # of locks in memory
  "results_cache_size": 0,        # # of cached results
  
  "heatmap_dedup_ratio": 0.0,     # Among heatmap requests
  "point_api_dedup_ratio": 0.0,   # Among point API requests
}
```

---

## 10. Implementation Flow

### 10.1 Full Request Path

```python
def inference_entry_point(grid_id: str, timestamp: str, request_type: str):
  """
  Entry point from Go.
  """
  # Determine timeout (from Go)
  timeout = 2.0 if request_type == "point_api" else 10.0
  
  # Check cache first (L1 cache might have it)
  cached = L1_CACHE.get((grid_id, timestamp))
  if cached and not is_stale(cached):
    metrics.cache_hits += 1
    return cached["pm25"]
  
  # Cache miss → use dedup flow
  prediction = predict_with_dedup(grid_id, timestamp, timeout=timeout)
  
  if prediction is None:
    return 503, "inference_failed"
  
  # Store in L1 for future hits
  L1_CACHE[(grid_id, timestamp)] = {
    "pm25": prediction,
    "created_at": datetime.utcnow()
  }
  
  return 200, {"pm25": prediction}
```

### 10.2 Dedup Flow Detailed

```python
def predict_with_dedup(grid_id, timestamp, timeout):
  # Try to acquire lock
  acquired = deduplicator.try_acquire(grid_id, timestamp, timeout=timeout)
  
  if acquired:
    # ===== LOCK HOLDER PATH =====
    logger.info(f"Computing {grid_id}@{timestamp}")
    metrics.lock_acquired_count += 1
    
    try:
      # Broad scope: assemble + predict
      features = assemble_features(grid_id, timestamp, timeout=timeout*0.8)
      prediction = predict_model(features, timeout=timeout*0.2)
      
      # Store for waiters + cache
      deduplicator.store_result(grid_id, timestamp, prediction)
      logger.info(f"Computation done {grid_id}@{timestamp}")
      
      return prediction
    
    except Exception as e:
      logger.error(f"Computation error: {e}")
      deduplicator.results[(grid_id, timestamp)].set_exception(e)
      return None
    
    finally:
      deduplicator.release_and_cleanup(grid_id, timestamp)
  
  else:
    # ===== WAITER PATH =====
    logger.info(f"Waiting for {grid_id}@{timestamp}")
    metrics.lock_waited_count += 1
    
    try:
      result = deduplicator.get_result(grid_id, timestamp, timeout=timeout)
      logger.info(f"Got result (waited) {grid_id}@{timestamp}")
      return result
    
    except concurrent.futures.TimeoutError:
      logger.error(f"Timeout waiting for {grid_id}@{timestamp}")
      metrics.lock_timeout_count += 1
      return None
    
    except Exception as e:
      logger.error(f"Computation failed (saw exception): {e}")
      return None
```

---

## 11. Configuration

```python
# Deduplication
LOCK_ACQUIRE_TIMEOUT_SECONDS = 2.0  # Point API timeout
FEATURE_ASSEMBLY_TIMEOUT_RATIO = 0.8  # 80% of total timeout
COMPUTE_TIMEOUT_RATIO = 0.2  # 20% of total timeout

# Result sharing
RESULT_FUTURE_TIMEOUT_SECONDS = 2.0  # Waiter timeout on Future
RESULT_DICT_CLEANUP_SECONDS = 3600  # (unused; cleaned by L1 eviction)

# Monitoring
TRACK_LOCK_WAIT_HISTOGRAM = True
TRACK_DEDUP_RATIO = True
```

---

## 12. Summary Decision Tree

```
Request for (grid_id, timestamp) arrives
  │
  ├─→ Check L1 cache (fresh data only)
  │    ├─→ Hit? Return immediately
  │    └─→ Miss? Continue
  │
  ├─→ Try to acquire lock for (grid_id, timestamp)
  │    │
  │    ├─→ Acquired (no wait)? (lock holder)
  │    │    ├─→ Assemble features (broad scope locked)
  │    │    ├─→ Predict model
  │    │    ├─→ Store result in Future (unblock waiters)
  │    │    ├─→ Release + cleanup lock
  │    │    └─→ Return prediction
  │    │
  │    └─→ Not acquired (lock held)? (waiter)
  │         ├─→ Wait on Future.result(timeout=Go_timeout)
  │         ├─→ Unblock when result available or timeout
  │         └─→ Return same result as lock holder (or error)
  │
  └─→ Store in L1 cache, return to Go
```

---

## 13. Implementation Checklist

- [ ] Implement `Deduplicator` class (locks dict + results dict)
- [ ] Implement `predict_with_dedup()` main flow
- [ ] Implement lock acquire + release (with cleanup)
- [ ] Implement Future-based result sharing
- [ ] Implement per-cell heatmap loop (no special logic)
- [ ] Test concurrent requests for same cell (dedup)
- [ ] Test concurrent requests for different cells (no lock contention)
- [ ] Test timeout enforcement (point 2s, heatmap 10s)
- [ ] Test waiter timeout (lock holder slow)
- [ ] Test exception handling (store in Future, waiters see error)
- [ ] Test process crash (lock released in finally)
- [ ] Measure dedup ratio (expect 5:1 to 20:1 at 1000 users)
- [ ] Monitor lock wait times (p50, p95, p99)
- [ ] Stress test with 1000 concurrent heatmap requests (same timestamp)

---

**Document Version**: 1.0  
**Last Updated**: January 19, 2026  
**Status**: Locked for Implementation
