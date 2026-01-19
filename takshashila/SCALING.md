# Takshashila: Scaling to 1000 Concurrent Users

**Date**: January 19, 2026  
**Status**: Final Design (Locked)  
**Audience**: Backend engineers, DevOps, architects

---

## Executive Summary

This document specifies how Takshashila scales to **1000 concurrent client requests** by separating concerns between **Go serving layer** (handles concurrency, deduplication, routing) and **Python inference layer** (compute engine, small parallelism). The system uses **hybrid threading + multiprocessing** on a Dell Vostro 4-core laptop to avoid CPU/memory collapse under load.

**Key Design Principles**:
1. **Go is the web server** — handles 1000 concurrent requests, deduplication, timeouts
2. **Python is a compute engine** — sees small number of jobs, not raw requests
3. **Thread pool (16) for I/O** — DuckDB reads, feature loading, serialization
4. **Process pool (2) for compute** — CPU-bound model.predict()
5. **Bounded queue (100 max)** — reject excess; fail fast over queueing infinitely
6. **Connection pool (4-8) for DuckDB** — prevent lock contention
7. **Timeout enforcement** — Point API 2s, Heatmap 10s (Go enforces + Python safeguards)
8. **Load shedding** — reject all requests if CPU >90% or RAM >3.5GB

---

## 1. Architectural Separation: Go vs Python

### 1.1 Incorrect Model (Leaky Architecture)

```
┌────────────────────────────────┐
│ 1000 Concurrent Clients        │
└────────────────────────────────┘
          ↓ (HTTP)
┌────────────────────────────────┐
│ Go Serving Layer               │
│  - Routes to Python            │
└────────────────────────────────┘
          ↓ (Raw request passthrough)
┌────────────────────────────────┐
│ Python Inference Service       │
│  - Sees 1000 concurrent tasks  │ ← PROBLEM: Unbounded concurrency
│  - Thread pool explodes        │
│  - Memory collapses            │
│  - Model queue overflows       │
└────────────────────────────────┘
```

**Problem**: Python drowns in raw request volume.

### 1.2 Correct Model (Sealed Layers)

```
┌────────────────────────────────┐
│ 1000 Concurrent Clients        │
└────────────────────────────────┘
          ↓ (HTTP)
┌────────────────────────────────┐
│ Go Serving Layer               │
│  - Deduplicates requests       │
│  - Routes to Python            │
│  - Enforces timeouts (2s/10s)  │
│  - Caches results              │
│  - Fans out to Python (~10-50  │
│    actual work items)          │
└────────────────────────────────┘
          ↓ (Deduplicated work)
┌────────────────────────────────┐
│ Python Inference Service       │
│  - Sees 10-50 actual jobs      │ ← HEALTHY: Bounded parallelism
│  - Thread pool (16) + Process  │
│    pool (2)                    │
│  - DuckDB connections (4-8)    │
│  - CPU: 40-60%                 │
│  - RAM: ~3-4GB                 │
└────────────────────────────────┘
```

**Solution**: Go acts as admission control, deduplication, and gateway.

---

## 2. Go Serving Layer: Concurrency Management

### 2.1 Request Handling

Go receives up to **1000 concurrent client connections**:

```go
// Go HTTP server (high concurrency)
func (s *Server) handlePredictRequest(w http.ResponseWriter, r *http.Request) {
  gridID := r.URL.Query().Get("grid_id")
  timestamp := r.URL.Query().Get("timestamp")
  
  // Deduplicate: Has another request already asked for this?
  jobKey := (gridID, timestamp)
  
  inFlight := s.deduplicator.CheckAndRegister(jobKey)
  if inFlight != nil {
    // Another request is computing this → wait for result
    result := <-inFlight
    w.Header().Set("X-From-Cache", "deduplication")
    w.Write(result)
    return
  }
  
  // New work → call Python inference service
  result := s.pythonClient.Predict(gridID, timestamp, timeout=2*time.Second)
  
  s.deduplicator.NotifyComplete(jobKey, result)
  w.Write(result)
}

// Request deduplication
type Deduplicator struct {
  inFlight map[string]chan Result  // (grid_id, timestamp) → result channel
}

func (d *Deduplicator) CheckAndRegister(key string) chan Result {
  if existing, ok := d.inFlight[key]; ok {
    return existing  // Already computing; wait for it
  }
  
  resultChan := make(chan Result, 1)
  d.inFlight[key] = resultChan
  return nil  // New work; proceed to compute
}
```

**Behavior**:
- **Request 1** for (M0001, 10:00): Not cached → calls Python
- **Request 2** for (M0001, 10:00): See request 1 in-flight → waits for same result
- **Request 3** for (M0002, 10:00): Different cell → calls Python (separate work)

**Result**: 1000 client requests → ~10-50 actual Python calls (high deduplication)

### 2.2 Timeout Enforcement (Go)

```go
func (s *Server) PredictWithTimeout(gridID, timestamp string, timeoutSeconds int) (Result, error) {
  ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSeconds)*time.Second)
  defer cancel()
  
  result := make(chan Result, 1)
  go func() {
    result <- s.pythonClient.Predict(gridID, timestamp)
  }()
  
  select {
  case r := <-result:
    return r, nil
  case <-ctx.Done():
    return nil, errors.New("timeout")
  }
}

// Handler routing by request type
func (s *Server) Route(r *Request) {
  if r.Type == "point_api" {
    s.PredictWithTimeout(r.GridID, r.Timestamp, timeoutSeconds=2)
  } else if r.Type == "heatmap_api" {
    s.PredictWithTimeout(r.GridID, r.Timestamp, timeoutSeconds=10)
  }
}
```

**Timeouts**:
- **Point API**: 2s timeout (Go enforces)
- **Heatmap API**: 10s timeout (Go enforces)
- Python **also enforces** internal safeguards

---

## 3. Python Inference Layer: Hybrid Parallelism

### 3.1 Thread Pool + Process Pool Architecture

```
┌─────────────────────────────────────────────┐
│ Python Inference Service                    │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │ Request Queue (max 100 jobs)          │  │
│  │  - Job 1: (M0001, 10:00)              │  │
│  │  - Job 2: (M0002, 10:00)              │  │
│  │  - ...                                │  │
│  └───────────────────────────────────────┘  │
│           ↓                                  │
│  ┌───────────────────────────────────────┐  │
│  │ Thread Pool (16 threads)              │  │
│  │  - Worker 1: DuckDB read for Job 1   │  │
│  │  - Worker 2: Feature assembly for 1  │  │
│  │  - Worker 3: DuckDB read for Job 2   │  │
│  │  - ... (up to 16 concurrent I/O)      │  │
│  └───────────────────────────────────────┘  │
│           ↓                                  │
│  ┌───────────────────────────────────────┐  │
│  │ Process Pool (2 processes)            │  │
│  │  - Process 1: model.predict() Job 1  │  │
│  │  - Process 2: model.predict() Job 2  │  │
│  │  - (Jobs queue; execute sequentially) │  │
│  └───────────────────────────────────────┘  │
│           ↓                                  │
│  ┌───────────────────────────────────────┐  │
│  │ Result Serialization (thread pool)    │  │
│  │  - Serialize Job 1 result             │  │
│  │  - Return to Go                       │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 3.2 Thread Pool (16 Threads)

**Purpose**: I/O-bound work (DuckDB reads, feature loading, serialization)

**Size**: 16 threads (2× CPU cores on Dell Vostro 4-core)

**Rationale**: Threads block on I/O (DuckDB, Redis); extra threads hide latency

```python
from concurrent.futures import ThreadPoolExecutor

thread_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="io_worker")

def job_thread_worker(job):
  """
  1. Acquire DuckDB connection from pool
  2. Fetch features from DuckDB (blocking I/O)
  3. Assemble feature vector
  4. Submit to process pool
  5. Serialize result
  6. Return to Go
  """
  grid_id, timestamp = job
  
  # Step 1: Acquire connection
  with db_connection_pool.get_connection(timeout=1.0) as conn:
    
    # Step 2-3: Fetch + assemble (I/O-bound, thread-safe)
    features = assemble_features(grid_id, timestamp, conn)
    
    # Step 4: Submit to process pool
    future = process_pool.submit(model.predict, features)
    prediction = future.result(timeout=2.0)
    
    # Step 5-6: Serialize + return
    result = {"grid_id": grid_id, "pm25": prediction}
    return result
```

**Characteristics**:
- Threads are **cheap** (context switch fast)
- DuckDB is **thread-safe** (shared connection pool)
- GIL is released during I/O (blocking on disk)

### 3.3 Process Pool (2 Processes)

**Purpose**: CPU-bound work (model.predict() — no GIL contention)

**Size**: 2 processes (conservative on 4-core machine)

**Rationale**: Multiprocessing bypasses Python GIL; avoid saturating machine

```python
from multiprocessing import Pool

process_pool = Pool(processes=2)  # One per 2 cores

def model_predict_worker(features):
  """
  Run in separate process (not affected by GIL).
  Called by thread pool.
  """
  model = load_model_once()  # Cached in process
  prediction = model.predict(features)  # ~10-20ms
  return prediction
```

**Characteristics**:
- Processes have **separate memory** (model loaded once per process)
- Each process runs **~2-4 predictions/sec** (10-20ms per predict)
- 2 processes → ~4-8 predictions/sec system throughput

**Job Queueing in Process Pool**:
```
Thread 1 submits Job A → Process 1 starts (working)
Thread 2 submits Job B → Queue (Process 2 free? No, go to queue)
Thread 3 submits Job C → Queue
... (all queue)
Process 1 finishes → Job B moves to Process 1
Process 2 finishes → Job C moves to Process 2
```

**Throughput**: ~8 predictions/sec (2 processes × 4 pred/sec)

---

## 4. Request Queue & Admission Control

### 4.1 Queue Configuration

```python
from queue import Queue

request_queue = Queue(maxsize=100)  # Max 100 queued jobs

def enqueue_job(job, timeout=0.1):
  """
  Try to enqueue job; reject if queue full.
  """
  try:
    request_queue.put(job, block=False)
    return True
  except queue.Full:
    logger.warning("Request queue full; rejecting job")
    return False

# Handler
def handle_request(job):
  if not enqueue_job(job):
    return 503, {"error": "service_overloaded", "retry_after": 1}
  
  result = request_queue.get()
  return 200, result
```

**Behavior**:
- **Queue has space**: Job accepted, queued
- **Queue full (100 jobs)**: Reject immediately (503)
- **Rationale**: Fail fast over accepting unbounded load

**Why not wait?**
- Go enforces 2s/10s timeout
- If queue is full, system is overloaded
- Waiting makes it worse (users pile up)
- Better to reject fast, client retries later

### 4.2 Job Processing Loop

```python
def job_processor():
  """
  Main loop: pull jobs from queue, route to thread pool.
  """
  while True:
    job = request_queue.get()  # Block until job available
    grid_id, timestamp = job
    
    # Submit to thread pool
    future = thread_pool.submit(job_thread_worker, grid_id, timestamp)
    
    # Attach callback (results go back to Go via channel/queue)
    future.add_done_callback(on_job_complete)
```

---

## 5. DuckDB Connection Pool

### 5.1 Connection Pool Configuration

```python
from queue import Queue

class DuckDBConnectionPool:
  def __init__(self, pool_size=8, db_path="artifacts/inference.duckdb"):
    self.pool = Queue(maxsize=pool_size)
    self.db_path = db_path
    
    # Initialize pool with connections
    for _ in range(pool_size):
      conn = duckdb.connect(db_path, read_only=True)
      self.pool.put(conn)
  
  def get_connection(self, timeout=1.0):
    """
    Acquire connection from pool; timeout if none available.
    """
    try:
      conn = self.pool.get(timeout=timeout)
      return PooledConnection(conn, self)
    except queue.Empty:
      logger.error(f"No DuckDB connection available (pool full)")
      raise TimeoutError("DuckDB connection pool exhausted")
  
  def return_connection(self, conn):
    """Return connection to pool after use."""
    self.pool.put(conn)

class PooledConnection:
  """Context manager for pooled connections."""
  def __init__(self, conn, pool):
    self.conn = conn
    self.pool = pool
  
  def __enter__(self):
    return self.conn
  
  def __exit__(self, *args):
    self.pool.return_connection(self.conn)

# Usage
db_pool = DuckDBConnectionPool(pool_size=8)

with db_pool.get_connection(timeout=1.0) as conn:
  features = conn.execute("SELECT * FROM features WHERE grid_id = ?", [grid_id]).fetchall()
```

**Configuration**:
- **Pool size**: 4-8 (start with 8; tune down if contention low)
- **Timeout**: 1s (if no connection in 1s, fail request)
- **Read-only**: All threads read same DuckDB file (safe)

**Behavior**:
- Thread 1 acquires connection → reads features
- Thread 2 waits for available connection (blocking)
- Thread 1 releases → Thread 2 continues
- If all 8 in use and Thread 3 asks → timeout (1s) → fail request

---

## 6. Resource Utilization Targets

### 6.1 CPU Target: 40-60%

**At steady state with 1000 concurrent client requests**:

```
Calculation:
- 1000 client requests → ~50 actual Python jobs (high dedup)
- Thread pool: 16 threads, mostly blocked on I/O
- Process pool: 2 processes, ~8 predictions/sec
- Per prediction: ~15ms compute + ~185ms I/O = 200ms roundtrip
- Concurrency in flight: 50 jobs × (200ms roundtrip / 1000ms) = ~10 active jobs
- CPU used by 2 processes: ~(10 active / 8 max throughput) = ~1.25 processes worth = ~30% CPU
- Total (threads + processes + Python overhead): 40-60% CPU
```

**Implications**:
- Leave 40% headroom for spikes
- Room for Kashi refresh jobs (scheduler concurrency)
- Room for error/exception handling

### 6.2 Memory Target: 4GB Max

**Budget breakdown**:

| Component | Size |
|---|---|
| Python runtime | ~50MB |
| LightGBM model (v124 + v123) | ~200MB |
| L1 prediction cache (in-memory) | ~2000MB |
| L2 connection pool + thread working memory | ~500MB |
| IDW weights (pre-computed) | ~7MB |
| Process pool model copies (2×) | ~100MB |
| Other (buffers, overhead) | ~1000MB |
| **Total** | **~3900MB** |

**Margin**: ~100MB spare before OOM

---

## 7. Request Timeouts

### 7.1 Timeout Strategy

**Two-level timeout** (Go + Python):

```python
# Level 1: Go enforces wall-clock timeout
def point_api_handler():
  go_timeout = 2 seconds
  result = python_client.predict(grid_id, timestamp, timeout=go_timeout)
  # If timeout → Go returns 503 to client

# Level 2: Python enforces internal safeguard
def python_predict(grid_id, timestamp, timeout):
  # Allocate 80% to I/O, 20% to compute
  io_timeout = timeout * 0.8  # 1.6s for point API
  compute_timeout = timeout * 0.2  # 0.4s
  
  # I/O phase (thread pool)
  try:
    features = thread_pool.submit(assemble_features, ...).result(timeout=io_timeout)
  except TimeoutError:
    return 503, "feature_assembly_timeout"
  
  # Compute phase (process pool)
  try:
    prediction = process_pool.apply_async(model.predict, [features]).get(timeout=compute_timeout)
  except TimeoutError:
    return 503, "model_predict_timeout"
  
  return 200, {"pm25": prediction}
```

**Timeouts by Request Type**:

| Type | Go Timeout | I/O Timeout | Compute Timeout |
|---|---|---|---|
| Point API | 2s | 1.6s | 0.4s |
| Heatmap API | 10s | 8s | 2s |

**Behavior**:
- **Go timeout < Python timeout**: Go kills request gracefully
- **Python timeout < Go timeout**: Python returns 503 before Go kills
- Both layers protect against hanging

### 7.2 Timeout Handling

```python
def handle_timeout(job, timeout_type):
  """
  Called when timeout occurs.
  """
  logger.error(f"Timeout: {job} ({timeout_type})")
  
  # Partial cleanup
  release_database_connections()
  release_thread_resources()
  
  # Return to Go
  return 503, {
    "error": "timeout",
    "type": timeout_type,
    "retry_after_seconds": 1
  }
```

---

## 8. Load Shedding & Overload Response

### 8.1 Overload Detection

Monitor CPU and memory continuously:

```python
import psutil

def check_system_health():
  """Check if system is overloaded."""
  cpu_percent = psutil.cpu_percent(interval=0.1)
  memory_percent = psutil.virtual_memory().percent
  
  # Convert memory percent to bytes
  memory_bytes = psutil.virtual_memory().used
  
  is_overloaded = (cpu_percent > 90) or (memory_bytes > 3.5 * 1024**3)
  
  return {
    "overloaded": is_overloaded,
    "cpu_percent": cpu_percent,
    "memory_bytes": memory_bytes,
    "memory_percent": memory_percent
  }

# Check before accepting job
health = check_system_health()
if health["overloaded"]:
  return 503, {"error": "system_overloaded", "retry_after": 5}
```

### 8.2 Load Shedding Action

**When overloaded (CPU >90% or RAM >3.5GB)**:

```python
def handle_overload():
  """
  Reject all new requests; let in-flight finish.
  """
  # Option A (selected)
  logger.error("System overloaded; rejecting new requests")
  return 503, {
    "error": "system_overloaded",
    "reason": "CPU >90% or RAM >3.5GB",
    "retry_after_seconds": 5
  }

# Alternative (not selected): Partial graceful degradation
# if cpu_percent > 90:
#   reject_heatmap_requests()
#   accept_point_api_requests()
```

**Behavior**:
- All new requests return 503
- In-flight requests complete
- System naturally cools down
- Go retries with backoff (exponential)

---

## 9. Heatmap Batch Parallelization

### 9.1 Heatmap Request Flow

```python
def get_heatmap_parallel(timestamp):
  """
  Request for 59k cells at timestamp.
  Parallelize with 4 threads (preserve pool capacity).
  """
  all_cells = get_all_grid_ids()  # 59k cells
  results = {}
  
  # Use ThreadPoolExecutor for parallel cell computation
  # But limit to 4 threads (preserve 12 threads for other work)
  with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
      executor.submit(compute_single_cell, cell_id, timestamp): cell_id
      for cell_id in all_cells
    }
    
    for future in concurrent.futures.as_completed(futures, timeout=10):
      cell_id = futures[future]
      try:
        pm25 = future.result()
        results[cell_id] = pm25
      except TimeoutError:
        logger.warning(f"Timeout computing {cell_id}")
        results[cell_id] = None
  
  return results
```

**Configuration**:
- **Heatmap thread count**: 4 (not all 16)
- **Rationale**: Preserve 12 threads for concurrent point API requests
- **Timeout**: 10s (Go enforces)

---

## 10. Circuit Breaker & Fallback

### 10.1 Circuit Breaker Pattern

```python
class CircuitBreaker:
  def __init__(self, failure_threshold=5, timeout=30):
    self.failure_count = 0
    self.failure_threshold = failure_threshold
    self.timeout = timeout
    self.last_failure = None
    self.state = "closed"  # closed, open, half-open
  
  def call(self, func, *args, **kwargs):
    """
    Try to execute func; if fails repeatedly, open circuit.
    """
    if self.state == "open":
      if time.time() - self.last_failure > self.timeout:
        self.state = "half-open"
        logger.info("Circuit breaker half-open; retrying")
      else:
        raise CircuitBreakerOpenError("Service unhealthy")
    
    try:
      result = func(*args, **kwargs)
      self.failure_count = 0
      self.state = "closed"
      return result
    except Exception as e:
      self.failure_count += 1
      self.last_failure = time.time()
      if self.failure_count >= self.failure_threshold:
        self.state = "open"
        logger.error(f"Circuit breaker open after {self.failure_count} failures")
      raise

# Usage
python_breaker = CircuitBreaker(failure_threshold=5, timeout=30)

def inference_with_circuit_breaker(grid_id, timestamp):
  try:
    return python_breaker.call(python_client.predict, grid_id, timestamp)
  except CircuitBreakerOpenError:
    # Fallback: Go reads DuckDB directly
    logger.warning("Python unavailable; falling back to DuckDB")
    return read_cached_prediction(grid_id, timestamp)
```

### 10.2 Fallback Strategy (Option D)

**When Python inference service fails** (crashes, timeouts, circuit open):

```python
def inference_with_fallback(grid_id, timestamp, timeout=2.0):
  """
  1. Try Python inference (primary)
  2. If fails, Go reads DuckDB directly (fallback)
  """
  try:
    # Try Python
    result = python_client.predict(grid_id, timestamp, timeout=timeout)
    return 200, result
  except (TimeoutError, ConnectionError, CircuitBreakerOpenError) as e:
    logger.error(f"Python inference failed: {e}")
    
    # Fallback: Read most recent prediction from DuckDB
    cached = read_latest_prediction_from_duckdb(grid_id)
    if cached:
      return 200, {
        "pm25": cached["pm25"],
        "confidence": cached["confidence"] * 0.5,  # Reduced confidence
        "warning": "Served from cache (Python unavailable)",
        "age_seconds": cached["age_seconds"]
      }
    else:
      return 503, {"error": "no_prediction_available"}
```

**Behavior**:
- Python healthy → use fresh predictions
- Python down → fall back to cached DuckDB predictions
- No cached data → return 503

---

## 11. Graceful Shutdown

### 11.1 Shutdown Behavior (Option A)

```python
def shutdown_service():
  """
  Kill immediately. In-flight requests fail gracefully.
  """
  logger.info("Shutdown signal received")
  
  # Stop accepting new requests
  shutdown_event.set()
  
  # Kill threads immediately
  thread_pool.shutdown(wait=False)
  process_pool.terminate()
  
  # Close DuckDB connections
  db_pool.close_all()
  
  # Exit
  sys.exit(0)
```

**Behavior**:
- In-flight requests fail (return 503 to client)
- Go catches error, retries with backoff
- New deployment takes over
- Fast cutover (no graceful drain delay)

**Rationale**: MVP doesn't need graceful drain; fast fail/retry is acceptable

---

## 12. Monitoring & Metrics

### 12.1 Key Metrics

```python
metrics = {
  # Throughput
  "predictions_per_sec": 0.0,
  "requests_per_sec": 0.0,
  "deduplication_ratio": 0.0,  # (client_requests / python_calls)
  
  # Latency
  "p50_latency_ms": 0.0,
  "p95_latency_ms": 0.0,
  "p99_latency_ms": 0.0,
  
  # Resource
  "cpu_percent": 0.0,
  "memory_bytes": 0,
  "thread_pool_active": 0,
  "process_pool_active": 0,
  "queue_depth": 0,
  
  # Health
  "request_queue_rejections": 0,
  "timeout_errors": 0,
  "python_failures": 0,
  "fallback_uses": 0,  # Times cached fallback served
  "circuit_breaker_trips": 0,
  
  # Cache
  "l1_hit_rate": 0.0,
  "l2_hit_rate": 0.0,
  "duckdb_lock_timeouts": 0,
  
  # Connections
  "duckdb_pool_size": 8,
  "duckdb_pool_exhaustions": 0,
}
```

### 12.2 Dashboard

Plot over time:
- Requests/sec vs Predictions/sec (dedup ratio)
- CPU usage (target 40-60%)
- Memory usage (target <3.5GB)
- Latency p50/p95/p99
- Queue depth (should be <20 normally)
- Error rates (timeouts, rejections, failures)
- Cache hit rates (L1, L2)

---

## 13. Load Testing Checklist

- [ ] Simulate 1000 concurrent client requests (via Apache JMeter / wrk)
- [ ] Measure deduplication ratio (expect 20:1 to 100:1)
- [ ] Verify CPU stays 40-60%
- [ ] Verify memory stays <3.5GB
- [ ] Measure p95/p99 latencies (expect <1-5s)
- [ ] Test timeout enforcement (point 2s, heatmap 10s)
- [ ] Test queue overflow (reject requests if queue full)
- [ ] Test circuit breaker (kill Python, verify fallback)
- [ ] Stress test heatmap (all 59k cells at once)
- [ ] Profile thread pool contention (measure wait times)

---

## 14. Deployment Topology

```
┌─────────────────────────────────────────────────┐
│ Dell Vostro Laptop (4 cores, 16GB RAM)          │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │ Go HTTP Server (port 8080)               │   │
│  │  - Goroutine per client (light)          │   │
│  │  - Deduplicator (in-process map)         │   │
│  │  - Timeout enforcer                      │   │
│  │  - Circuit breaker                       │   │
│  │  - Fallback to DuckDB reader             │   │
│  └──────────────────────────────────────────┘   │
│                      ↓ (gRPC/local socket)      │
│  ┌──────────────────────────────────────────┐   │
│  │ Python Inference Process                 │   │
│  │  - Request queue (max 100)               │   │
│  │  - Thread pool (16 workers)              │   │
│  │  - Process pool (2 workers)              │   │
│  │  - DuckDB connection pool (8)            │   │
│  └──────────────────────────────────────────┘   │
│                      ↓ (file I/O)               │
│  ┌──────────────────────────────────────────┐   │
│  │ DuckDB File (artifacts/inference.duckdb) │   │
│  │  - Predictions table (read-only for GO)  │   │
│  │  - Feature tables (read by Python)       │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Deployment**: Single machine; Go + Python can restart independently

---

## 15. Configuration Summary

```python
# Thread pool (I/O)
THREAD_POOL_SIZE = 16  # 2× cores

# Process pool (CPU)
PROCESS_POOL_SIZE = 2  # Conservative

# Request queue
REQUEST_QUEUE_MAX_SIZE = 100

# DuckDB connection pool
DUCKDB_POOL_SIZE = 8
DUCKDB_POOL_TIMEOUT_SECONDS = 1.0

# Timeouts
POINT_API_TIMEOUT_SECONDS = 2
HEATMAP_API_TIMEOUT_SECONDS = 10
THREAD_IO_TIMEOUT_SECONDS = 1.6  # For point API
PROCESS_COMPUTE_TIMEOUT_SECONDS = 0.4  # For point API

# Load shedding
CPU_OVERLOAD_THRESHOLD_PERCENT = 90
MEMORY_OVERLOAD_THRESHOLD_BYTES = 3.5 * 1024**3  # 3.5GB

# Circuit breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT_SECONDS = 30

# Heatmap
HEATMAP_PARALLEL_THREADS = 4  # Preserve pool capacity
HEATMAP_TIMEOUT_SECONDS = 10
```

---

**Document Version**: 1.0  
**Last Updated**: January 19, 2026  
**Status**: Locked for Implementation
