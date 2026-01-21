# Phase 2: Coordinator + Worker Architecture - Detailed Implementation Plan

**Date**: January 20, 2026  
**Status**: Planning  
**Target**: Go coordinator + Python worker pool via bounded queue  

---

## 1. Architecture Overview

### High-Level Data Flow

```
Request (grid_id, timestamp_hour)
    ↓
[Go Coordinator]
  • Validate input
  • Check dedup map (in-memory)
  • Check cache (TTL-based, in-memory)
  • Enforce 1s timeout
  • Enqueue job to bounded queue
  • If leader: wait on Future; if follower: wait on promise
    ↓
[Bounded Job Queue] (capacity: 50–200)
  • Reject if full → overload error
    ↓
[Python Worker Pool] (fixed size: 2–4, pre-spawned)
  • Receive job from queue
  • Load features from DuckDB (read-only, from pool)
  • Check freshness
  • Run inference (LightGBM)
  • Write prediction to DuckDB (single writer, serialized)
  • Return result or error
    ↓
[Go Coordinator]
  • Receive result
  • Resolve Future/promise
  • Update cache (TTL: aligned to feature freshness)
  • Emit metrics
  • Return to client
  • Followers unblock
```

### Process Boundary (Critical)

```
Go Process (main service)
├── HTTP/REST listener (receive requests)
├── Coordinator logic
│   ├── Dedup map (in-memory)
│   ├── Cache (TTL-based, in-memory)
│   ├── Timeout enforcer
│   ├── Metrics collector (Prometheus)
│   └── Job queue (bounded)
└── Awaiter for Python results

Python Process (worker pool)
├── Worker 1 ──┐
├── Worker 2 ──┤→ [Queue receiver]
├── Worker 3 ──┤
└── Worker 4 ──┘
    Each worker:
    • Blocks on queue
    • Fetches features (DuckDB reads)
    • Runs inference
    • Writes predictions (serialized, single writer)
    • Sends result back
```

**Key Rule**: Go does NOT manage Python lifecycle. Go only sends/receives jobs.

---

## 2. Data Structures & Interfaces

### 2.1 Go: Job Payload

```go
type InferenceJob struct {
    GridID         string
    TimestampHour  time.Time
    RequestID      string           // for tracing
    ReplyChan      chan JobResult   // where to send result
    EnqueuedAt     time.Time        // for timeout tracking
}

type JobResult struct {
    PM25         float64
    Error        error
    ComputedAt   time.Time
    Source       string // "computed" | "cached" (for dedup followers)
}
```

### 2.2 Go: Dedup Map Entry

```go
type DedupeEntry struct {
    Leader       bool               // true if this request is the leader
    Future       <-chan JobResult   // shared result channel
    CreatedAt    time.Time
    TimeoutAt    time.Time
}

// Keyed by (grid_id, timestamp_hour)
var DedupeMap = make(map[DedupeKey]*DedupeEntry)
```

### 2.3 Go: Cache Entry

```go
type CacheEntry struct {
    PM25           float64
    CachedAt       time.Time
    ExpiresAt      time.Time       // TTL-based, aligned to feature freshness
    FreshnessType  string          // "pm25" | "landsat" | "sentinel" | "meteo"
}

// Keyed by (grid_id, timestamp_hour)
var Cache = make(map[CacheKey]*CacheEntry)
```

### 2.4 Go: Metrics Registry

```go
type MetricsRegistry struct {
    // Error counters
    TotalErrors              prometheus.Counter
    FeatureAssemblyErrors    prometheus.Counter
    ModelInferenceErrors     prometheus.Counter
    PersistenceErrors        prometheus.Counter
    TimeoutErrors            prometheus.Counter
    InvalidRequestErrors     prometheus.Counter
    OverloadErrors           prometheus.Counter
    
    // Timeout tracking
    InferenceTimeouts        prometheus.Counter
    RequestTimeouts          prometheus.Counter
    
    // Latency histograms (p50, p95, p99)
    EndToEndLatency          prometheus.Histogram
    PythonInferenceLatency   prometheus.Histogram
    FeatureLoadLatency       prometheus.Histogram
    DuckDBReadLatency        prometheus.Histogram
    DuckDBWriteLatency       prometheus.Histogram
    
    // In-flight work (most important leading indicator)
    InFlightJobs             prometheus.Gauge
    QueueLength              prometheus.Gauge
    ActiveWorkers            prometheus.Gauge
    
    // Cache effectiveness
    CacheHitRatio            prometheus.Gauge
    CacheMissRatio           prometheus.Gauge
    CacheEvictions           prometheus.Counter
    
    // Deduplication effectiveness
    SingleFlightLeaders      prometheus.Counter
    SingleFlightFollowers    prometheus.Counter
    DuplicateComputeAvoided  prometheus.Counter
    
    // Freshness outcomes
    FreshServed              prometheus.Counter
    StaleServed              prometheus.Counter  // only predictions, not features
    StaleRecomputeTriggered  prometheus.Counter
    MissingFeaturesCount     prometheus.Counter
    
    // Persistence integrity
    DuckDBWriteFailures      prometheus.Counter
    WriteLatencyDistribution prometheus.Histogram
    ReadAfterWriteFailures   prometheus.Counter // should be zero
}
```

### 2.5 Python: Worker Contract

```python
# Worker receives from Go:
class InferenceJob:
    grid_id: str
    timestamp_hour: datetime
    request_id: str
    
# Worker sends back:
class JobResult:
    pm25: float
    error: Optional[str]
    error_type: Optional[str]  # "FeatureAssemblyError" | "ModelInferenceError" | etc.
    computed_at: datetime
    worker_id: int
```

### 2.6 Freshness Configuration (LOCKED)

Store in a centralized location (e.g., `config.go` and `config.py`):

```
Feature Type          | Stale Threshold  | Notes
─────────────────────────────────────────────────
PM25 predictions      | > 1 hour         | Recent; weather-dependent
Meteorological data   | > 1 hour         | Weather updates hourly
Landsat imagery       | > 16 days        | 16-day revisit cycle
Sentinel imagery      | > 5 days         | 5-day revisit cycle
```

---

## 3. Go Coordinator: Detailed Modules

### 3.1 Module: `coordinator.go`

**Responsibilities**:
- Receive prediction requests
- Enforce 1-second timeout
- Check dedup map (leader vs follower)
- Check cache
- Enqueue to job queue
- Wait for results
- Emit metrics

**Key Functions**:
- `Predict(ctx context.Context, gridID string, timestampHour time.Time) (float64, error)`
- `checkDedupe(key DedupeKey) (bool, <-chan JobResult)` → returns (isLeader, resultChan)
- `checkCache(key CacheKey) (float64, bool)` → returns (value, found)
- `enqueueJob(job InferenceJob) error` → returns OverloadError if queue full
- `enforceTimeout(ctx, 1*time.Second)`
- `updateCache(key CacheKey, value float64, freshnessType string)`

**Timeout Semantics**:
- Context deadline set to `time.Now() + 1*time.Second` at request entry
- Context checked before enqueue
- Passed to Python job
- If job exceeds 1s, Python job is killed and result ignored
- Go returns `TimeoutError` to client

### 3.2 Module: `queue.go`

**Responsibilities**:
- Manage bounded job queue
- Handle full queue rejection
- Provide producer/consumer interface

**Key Functions**:
- `NewQueue(capacity int) *Queue`
- `Enqueue(job InferenceJob) error` → OverloadError if full
- `Dequeue() InferenceJob` → blocks until available
- `Length() int`
- `Close()`

**Data**:
```go
type Queue struct {
    jobs chan InferenceJob
    size int // capacity, immutable
}
```

**Rejection Logic**:
- Before enqueue, check if queue is at capacity
- If full, return `OverloadError("inference queue full")`
- Do not block; fail fast

### 3.3 Module: `dedup.go`

**Responsibilities**:
- Manage single-flight deduplication
- Track leaders vs followers
- Clean up expired entries

**Key Functions**:
- `Deduplicate(key DedupeKey, timeout time.Duration) (isLeader bool, resultChan <-chan JobResult, err error)`
- `ResolveLeader(key DedupeKey, result JobResult)`
- `CleanupExpired()`

**Data**:
```go
type DedupeMap struct {
    sync.RWMutex
    entries map[DedupeKey]*DedupeEntry
}
```

**Logic**:
- Followers see result channel immediately (blocked until leader resolves)
- Leaders get a channel to enqueue their job
- If another request arrives for same key while first is in-flight, they join as followers
- When leader completes, all followers unblock with same result

### 3.4 Module: `cache.go`

**Responsibilities**:
- In-memory TTL cache for predictions
- TTL aligned to feature freshness
- Lazy cleanup (on access, not background)

**Key Functions**:
- `Get(key CacheKey) (float64, bool)` → (value, found)
- `Set(key CacheKey, value float64, ttl time.Duration)`
- `Invalidate(key CacheKey)`
- `IsExpired(key CacheKey) bool`

**Data**:
```go
type Cache struct {
    sync.RWMutex
    entries map[CacheKey]*CacheEntry
}
```

**TTL Determination**:
- PM25 predictions → 1 hour (meteo data freshness)
- Landsat-based → 16 days
- Sentinel-based → 5 days
- On Set, look up feature freshness from config, compute expiry

### 3.5 Module: `metrics.go`

**Responsibilities**:
- Initialize Prometheus metrics
- Expose `/metrics` endpoint
- Emit counters, histograms, gauges

**Key Functions**:
- `NewMetricsRegistry() *MetricsRegistry`
- `RecordError(errorType string)`
- `RecordTimeout(timeoutType string)` → "request" or "inference"
- `RecordLatency(component string, duration time.Duration)`
- `RecordInFlightJobs(count int)`
- `RecordQueueLength(length int)`
- `RecordDedup(isLeader bool)`
- `RecordCacheHit()`
- `RecordCacheMiss()`
- `RecordFreshness(freshnessState string)` → "fresh" | "stale" | "missing"
- `ExportMetrics() http.Handler` → `/metrics` endpoint

### 3.6 Module: `python_client.go`

**Responsibilities**:
- Spawn Python worker processes
- Manage worker pool lifecycle (health checks, NOT respawn)
- Send jobs to workers
- Receive results

**Key Functions**:
- `NewPythonWorkerPool(count int, pythonScript string) *WorkerPool`
- `SendJob(job InferenceJob) <-chan JobResult`
- `HealthCheck() map[int]bool` → worker_id → alive
- `Shutdown(gracefulTimeout time.Duration)`

**Data**:
```go
type WorkerPool struct {
    workers []*Worker
    queue   chan InferenceJob
    done    chan struct{}
}

type Worker struct {
    id       int
    process  *exec.Cmd
    stdin    io.WriteCloser
    stdout   io.ReadCloser
    alive    bool
}
```

**Lifecycle**:
- Pre-spawn all workers at startup
- Each worker blocks on queue
- If worker dies, mark `alive = false`
- Do NOT respawn; let outer supervisor handle it
- Metrics track active worker count

---

## 4. Python Worker: Detailed Modules

### 4.1 Module: `worker.py`

**Responsibilities**:
- Read jobs from queue (stdin/JSON)
- Orchestrate feature loading → inference → write
- Return result (stdout/JSON)
- Handle timeout gracefully

**Key Functions**:
- `main()` → loop: read job → compute → send result
- `process_job(job: InferenceJob) → JobResult`
- `load_features_from_duckdb(grid_id, timestamp_hour) → pd.DataFrame`
- `check_freshness(features_df, timestamp_hour) → bool`
- `run_inference(features_df) → float64`
- `write_prediction_to_duckdb(grid_id, timestamp_hour, pm25) → bool`

**Timeout Handling**:
- Parent Go process sends `SIGALRM` or similar (or kill after 1s)
- Worker should catch and exit gracefully
- Any in-flight DuckDB write should roll back (or use transactions)

### 4.2 Module: `feature_loader.py`

**Responsibilities**:
- Fetch features from DuckDB (read-only, using connection pool)
- Validate freshness
- Return assembled feature DataFrame

**Key Functions**:
- `load_features(db_path, grid_id, timestamp_hour, freshness_config) → (pd.DataFrame, bool)`
  - Returns features and a flag: `is_fresh`
  - If not fresh, raises `FeatureAssemblyError` (per locked decision)
- `check_freshness_of_feature(feature_type, last_updated_timestamp) → bool`
- `get_feature_pool() → ConnectionPool`

**DuckDB Connection Pool**:
- Pool size: 4–8 connections (tunable per config)
- Read-only connections
- Reuse connections across requests
- Timeout on pool acquisition

### 4.3 Module: `inference.py` (extend Phase 1)

**Responsibilities**:
- Load model (once at startup, cached in-process)
- Run inference
- Validate output

**Key Functions**:
- Same as Phase 1, but called per job

### 4.4 Module: `persistence.py` (extend Phase 1, modify write semantics)

**Responsibilities**:
- Write predictions to DuckDB
- Use a single, serialized writer thread (NOT per-request)
- Validate write-after-read

**Key Functions**:
- `write_prediction_serialized(grid_id, timestamp_hour, pm25, model_version, schema_hash) → bool`
  - Acquires global write lock
  - Inserts prediction
  - Reads back to verify
  - Releases lock
- `PersistenceQueue` (new): Single background thread that serializes writes

---

## 5. Communication Protocol: Go ↔ Python

### 5.1 Job Submission (Go → Python)

**Medium**: Unix pipe / stdin

**Format**: Line-delimited JSON

```json
{
  "grid_id": "g00000001",
  "timestamp_hour": "2026-01-19T16:00:00Z",
  "request_id": "req-12345",
  "timeout_ms": 1000
}
```

### 5.2 Result Return (Python → Go)

**Medium**: Unix pipe / stdout

**Format**: Line-delimited JSON

```json
{
  "request_id": "req-12345",
  "pm25": 45.32,
  "error": null,
  "error_type": null,
  "computed_at": "2026-01-20T10:30:45Z",
  "worker_id": 1
}
```

**On Error**:

```json
{
  "request_id": "req-12345",
  "pm25": null,
  "error": "Feature 'features_pm25 row' missing",
  "error_type": "FeatureAssemblyError",
  "computed_at": "2026-01-20T10:30:46Z",
  "worker_id": 2
}
```

### 5.3 Framing & Parsing

- Use `bufio.Scanner` (Go) and line reading (Python)
- One job per line, one result per line
- If parsing fails, log and skip (don't crash)

---

## 6. Error Handling & Propagation

### 6.1 Error Taxonomy

**From Python Worker**:
- `InvalidRequestError` → Go returns 400 Bad Request
- `FeatureAssemblyError` → Go returns 422 (missing data)
- `ModelInferenceError` → Go returns 500 (computation failed)
- `PersistenceError` → Go returns 500 (DB write failed)

**From Go Coordinator**:
- `OverloadError` (queue full) → returns 503 Service Unavailable
- `TimeoutError` (exceeded 1s) → returns 504 Gateway Timeout
- `DedupeTimeout` (waiting for leader, leader timed out) → returns 504

### 6.2 Error Propagation Flow

```
Python error → JSON reply
    ↓
Go parses error_type field
    ↓
Go emits metrics (by error_type)
    ↓
Go resolves dedup (followers see same error)
    ↓
Go returns HTTP error to client
```

### 6.3 Write Failure Semantics

**Locked Decision**: If Python write fails:
- `PersistenceError` returned to Go
- Dedup followers receive same error
- Result is NOT cached
- Metrics track `DuckDBWriteFailures`

---

## 7. Metrics & Observability

### 7.1 Metric Emission Points

**In Go Coordinator**:
- On request entry: increment incoming counter
- On dedup check: record leader vs follower
- Before enqueue: record queue length
- On cache hit: increment cache hit counter
- On cache miss: increment cache miss counter
- Before dedup timeout: arm 1s timeout
- On job completion: record end-to-end latency
- On error: increment error counter (by type)

**In Python Worker**:
- On feature load: record latency
- On freshness check: record freshness outcome
- On inference: record latency
- On write: record latency, track failures
- On timeout: signal to Go (via exit code or flag)

**In `/metrics` endpoint**:
- Expose all Prometheus metrics
- Include current queue length, active workers, dedup map size
- Aggregate dedup leader/follower counts (periodically, not per-request)

### 7.2 Metrics Export

**Endpoint**: `GET /metrics`

**Format**: Prometheus text format (standard)

```
# HELP inference_errors_total Total inference errors
# TYPE inference_errors_total counter
inference_errors_total{error_type="FeatureAssemblyError"} 5
inference_errors_total{error_type="ModelInferenceError"} 2
...

# HELP inference_queue_length Current queue length
# TYPE inference_queue_length gauge
inference_queue_length 42

# HELP inference_latency_seconds End-to-end request latency
# TYPE inference_latency_seconds histogram
inference_latency_seconds_bucket{le="0.01"} 100
inference_latency_seconds_bucket{le="0.05"} 250
inference_latency_seconds_bucket{le="0.1"} 280
inference_latency_seconds_bucket{le="0.5"} 290
inference_latency_seconds_bucket{le="1.0"} 295
...
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (Go)

**`coordinator_test.go`**:
- Test dedup (leader vs follower)
- Test cache hit/miss/expiry
- Test timeout enforcement
- Test error propagation

**`queue_test.go`**:
- Test enqueue on empty queue
- Test dequeue blocks
- Test full queue rejection
- Test concurrent producers

**`dedup_test.go`**:
- Test leader creation
- Test follower joining
- Test result propagation
- Test cleanup of expired entries

**`cache_test.go`**:
- Test set/get
- Test TTL expiry
- Test eviction
- Test concurrent access

**`metrics_test.go`**:
- Test counter increments
- Test histogram observations
- Test gauge updates

### 8.2 Unit Tests (Python)

**`test_worker.py`**:
- Test job reading from stdin
- Test result writing to stdout
- Test error serialization

**`test_feature_loader.py`**:
- Test feature fetching (using Phase 1 mock data)
- Test freshness checks
- Test missing features → FeatureAssemblyError

**`test_persistence.py`**:
- Test write_prediction_serialized
- Test read-after-write validation
- Test concurrent write rejection (if attempted)

### 8.3 Integration Tests

**`integration_test.go`**:
- Spawn Go coordinator + Python workers
- Send requests through entire pipeline
- Verify end-to-end correctness
- Test dedup with concurrent requests
- Test cache behavior
- Test queue full rejection
- Test timeout kills Python job
- Test worker crash reduces capacity

**Scenarios**:
1. **Happy path**: Valid request → correct prediction
2. **Dedup leader**: One request computes, followers wait for result
3. **Dedup followers**: Multiple concurrent requests for same grid/timestamp
4. **Cache hit**: Subsequent request returns cached result
5. **Cache miss + recompute**: TTL expired → recompute
6. **Queue full**: New request rejected with OverloadError
7. **Timeout**: Job exceeds 1s → TimeoutError
8. **Worker crash**: One worker dies → remaining workers handle load
9. **Feature missing**: FeatureAssemblyError propagates
10. **Model error**: ModelInferenceError propagates
11. **Write failure**: PersistenceError prevents caching

### 8.4 Load Tests

**`load_test.go`**:
- Generate concurrent requests (10, 100, 1000 concurrent)
- Measure queue saturation
- Measure timeout rate under overload
- Measure dedup effectiveness (burst of same key)
- Record latency percentiles (p50, p95, p99)
- Verify error rates remain within threshold
- Track in-flight jobs (should peak at worker pool size)

### 8.5 Test Data

**Reuse from Phase 1**:
- Mock grids in `Takshashila/tests/fixtures/mock_grids.json`
- Mock features in populated DuckDB tables
- Test timestamp: `2026-01-19T16:00:00Z` (matches mock data)

---

## 9. Configuration & Environment

### 9.1 Go Configuration (`config.go`)

```go
type Config struct {
    // Worker pool
    NumWorkers    int           // default: 2
    WorkerScript  string        // path to Python worker script
    
    // Queue
    QueueCapacity int           // default: 100
    
    // Timeouts
    RequestTimeout     time.Duration // 1 second, hard bound
    WorkerStartTimeout time.Duration // 5 seconds
    
    // Cache
    CacheTTLByFeature map[string]time.Duration // pm25 → 1h, etc.
    
    // Metrics
    MetricsPort int // default: 9090
    
    // Database
    DuckDBPath string // default: "Takshashila/duckdb/predictions.duckdb"
    
    // Feature freshness thresholds
    FreshnessThresholds map[string]time.Duration
}
```

### 9.2 Python Configuration (`config.py`)

```python
CONFIG = {
    "duckdb_path": "Takshashila/duckdb/predictions.duckdb",
    "feature_schema_path": "dwarka/models/prod_v1/feature_schema.json",
    "model_path": "dwarka/models/prod_v1/model.txt",
    "db_pool_size": 4,
    "db_pool_timeout_s": 5,
    "freshness_thresholds": {
        "pm25": 3600,  # 1 hour
        "meteo": 3600,
        "landsat": 1382400,  # 16 days
        "sentinel": 432000,  # 5 days
    }
}
```

### 9.3 Environment Variables

- `INFERENCE_NUM_WORKERS` → override config
- `INFERENCE_QUEUE_CAPACITY`
- `INFERENCE_DUCKDB_PATH`
- `INFERENCE_METRICS_PORT`

---

## 10. Implementation Phases

### Phase 2.0: Core Infrastructure (Week 1) ✅ COMPLETE

1. **Go Coordinator Skeleton**:
   - Main HTTP server setup
   - Queue implementation (bounded)
   - Dedup map (in-memory)
   - Basic logging

2. **Python Worker Skeleton**:
   - Read jobs from stdin (JSON)
   - Call Phase 1 InferenceEngine
   - Return results (stdout/JSON)
   - Error handling

3. **IPC Protocol**:
   - Define JSON schema for job/result
   - Test with mock data

4. **Tests**:
   - Unit tests for queue, dedup, cache
   - Basic integration test (1 request → 1 response)

---

## Phase 2b: Advanced Features (4-Part Progressive Build)

### Phase 2b Part 1: Cache + Dedup Foundation (Week 2)

**Objective**: Implement dedup map and cache layer; wire to coordinator. Verify single-flight and caching work correctly.

**Implementation**:

1. **`dedup.go` Module**:
   - `DedupeMap` struct with RWMutex
   - `Deduplicate(key DedupeKey, timeout)` → (isLeader bool, resultChan, error)
   - `ResolveLeader(key, result)` → broadcasts to all followers
   - `CleanupExpired()` → removes timed-out entries
   - Per-request dedup key: hash(grid_id + timestamp_hour)

2. **`cache.go` Module**:
   - `Cache` struct with RWMutex
   - `Get(key)` → (value float64, found bool)
   - `Set(key, value, ttl)` with TTL calculation
   - `IsExpired(key)` → bool
   - TTL mapping: PM25→1h, Landsat→16d, Sentinel→5d, Meteo→1h

3. **`coordinator.go` Module** (core prediction logic):
   - `Predict(ctx, gridID, timestampHour)` orchestrates:
     - Check cache → if hit, return (emit CacheHitMetric)
     - Check dedup → if follower, wait on resultChan
     - If leader: enqueue job, wait for result
     - Update cache on success
     - Return to client
   - Context timeout: `time.Now() + 1*time.Second`

4. **Wiring**:
   - Coordinator receives requests
   - Cache checked before dedup
   - Dedup resolved after job completion
   - Cache updated if dedup leader succeeds

**Rigorous Testing**:

**`dedup_test.go`** (Unit Tests):
- Test 1: Leader creation and job enqueueing
- Test 2: Follower joining, receiving same result as leader
- Test 3: Multiple followers (N=5) get same result
- Test 4: Followers unblock when leader resolves
- Test 5: Expired entries cleaned up
- Test 6: Concurrent leader + follower requests
- Test 7: Dedup key uniqueness (different timestamps → different entries)
- **Pass Criteria**: All 7 tests pass, no race conditions on concurrent access

**`cache_test.go`** (Unit Tests):
- Test 1: Set and Get on empty cache
- Test 2: Cache hit on valid entry
- Test 3: Cache miss on missing key
- Test 4: TTL expiry (set with 100ms TTL, wait 150ms, expect miss)
- Test 5: IsExpired() returns true for expired entries
- Test 6: Invalidate removes entry
- Test 7: Concurrent Set/Get access (5 goroutines, 100 ops each)
- **Pass Criteria**: All 7 tests pass, no race conditions, TTL precise within 10ms

**`coordinator_test.go`** (Unit Tests - Dedup + Cache):
- Test 1: Cache hit returns immediately
- Test 2: Cache miss triggers dedup check
- Test 3: Leader enqueues job
- Test 4: Follower waits for leader result
- Test 5: Cache updated after leader completes
- Test 6: Expired cache entry triggers recompute
- Test 7: Dedup timeout returns error (follower waiting too long)
- **Pass Criteria**: All 7 tests pass, latency for cache hit < 1ms

**Integration Test** (`test_dedup_cache_integration.sh`):
- Scenario 1: Single request → cache miss → dedup leader → compute → cache set
- Scenario 2: Immediate second request → cache hit (no compute)
- Scenario 3: 5 concurrent requests for same key → 1 leader, 4 followers, shared result
- Scenario 4: Cache TTL expires → next request triggers recompute
- Scenario 5: Different grid_id → different cache entries (no collision)
- **Pass Criteria**: All 5 scenarios pass, dedup reduces compute by 80%+ on bursty load

**Success Metrics**:
- Dedup map reduces duplicate compute on 5 concurrent identical requests: ≥80% reduction
- Cache hit latency: <1ms
- Cache miss + compute latency: baseline (no overhead)
- No memory leaks on 1000 dedup entries

---

### Phase 2b Part 2: Timeout Enforcement + Error Classification (Week 2)

**Objective**: Implement 1-second hard timeout and error taxonomy. Verify timeouts kill jobs and errors propagate correctly.

**Implementation**:

1. **Timeout Enforcement in `coordinator.go`**:
   - Set context deadline: `ctx, cancel := context.WithTimeout(parentCtx, 1*time.Second)`
   - Pass timeout_ms to Python job
   - Go routine monitors job duration
   - If job exceeds 1s, kill Python process: `cmd.Process.Kill()`
   - Return `TimeoutError` to client (504 Gateway Timeout)
   - Do NOT cache timeout errors
   - Emit `InferenceTimeouts` counter metric

2. **Error Taxonomy in `main.go`** (HTTP handlers):
   - Map error_type from Python to HTTP status:
     - `InvalidRequestError` → 400 Bad Request
     - `FeatureAssemblyError` → 422 Unprocessable Entity
     - `ModelInferenceError` → 500 Internal Server Error
     - `PersistenceError` → 500 Internal Server Error
     - `TimeoutError` → 504 Gateway Timeout
     - `OverloadError` (queue full) → 503 Service Unavailable
   - Parse error_type from JSON result
   - Emit error counter: `RecordError(error_type)`

3. **Error Propagation through Dedup**:
   - If leader gets error, followers receive same error
   - Error NOT cached (only success cached)
   - Metrics track error distribution by type

4. **Timeout Signal in `python_client.go`**:
   - Spawn goroutine per job: `time.AfterFunc(1*time.Second, func() { cmd.Process.Kill() })`
   - Capture Python exit code
   - If killed by timeout, set error_type = "TimeoutError"

**Rigorous Testing**:

**`timeout_test.go`** (Unit Tests):
- Test 1: Job completes under 1s → success
- Test 2: Job exceeds 1s → timeout error, process killed
- Test 3: Timeout prevents cache update
- Test 4: Dedup followers receive timeout error (same as leader)
- Test 5: Timeout counter incremented correctly
- Test 6: Multiple concurrent timeouts counted separately
- Test 7: Timeout error HTTP status is 504
- **Pass Criteria**: All 7 tests pass, timeout enforcement within 50ms accuracy

**`error_test.go`** (Unit Tests):
- Test 1: InvalidRequestError → 400
- Test 2: FeatureAssemblyError → 422
- Test 3: ModelInferenceError → 500
- Test 4: PersistenceError → 500
- Test 5: TimeoutError → 504
- Test 6: OverloadError → 503
- Test 7: Error metrics incremented by type
- **Pass Criteria**: All 7 tests pass, error mapping correct

**`coordinator_test.go`** (Dedup + Error Integration):
- Test 1: Leader times out, followers receive error
- Test 2: Error not cached, next request retries
- Test 3: Mixed errors (some succeed, some fail) tracked separately
- Test 4: Error counter per type increases independently
- **Pass Criteria**: All 4 tests pass

**Integration Test** (`test_timeout_error_integration.sh`):
- Scenario 1: Request completes under 1s → success
- Scenario 2: Slow request (2s) → timeout error (504), process killed
- Scenario 3: 5 concurrent requests, 3 timeout, 2 succeed → metrics track both
- Scenario 4: FeatureAssemblyError from Python → 422 response
- Scenario 5: Timeout error not cached → next request retries
- **Pass Criteria**: All 5 scenarios pass, timeout < 1.1s (accuracy within 100ms)

**Success Metrics**:
- Timeout enforcement: ±50ms accuracy
- Error classification: 100% correct mapping
- Dedup error propagation: followers receive identical error to leader
- No cache pollution from errors

---

### Phase 2b Part 3: Metrics & Observability (Week 2-3)

**Objective**: Implement full Prometheus metrics registry. Verify all metrics emitted and `/metrics` endpoint works.

**Implementation**:

1. **`metrics.go` Module** (Registry of 49+ metrics):
   - **Error counters** (8):
     - `inference_errors_total{error_type="..."}`
     - Types: InvalidRequest, FeatureAssembly, ModelInference, Persistence, Timeout, Overload, RequestTotal
   - **Timeout tracking** (2):
     - `inference_request_timeouts_total`
     - `inference_inference_timeouts_total`
   - **Latency histograms** (5):
     - `inference_end_to_end_latency_seconds`
     - `inference_python_inference_latency_seconds`
     - `inference_feature_load_latency_seconds`
     - `inference_duckdb_read_latency_seconds`
     - `inference_duckdb_write_latency_seconds`
   - **In-flight gauges** (3):
     - `inference_in_flight_jobs`
     - `inference_queue_length`
     - `inference_active_workers`
   - **Cache metrics** (4):
     - `inference_cache_hits_total`
     - `inference_cache_misses_total`
     - `inference_cache_hit_ratio` (gauge)
     - `inference_cache_evictions_total`
   - **Dedup metrics** (3):
     - `inference_single_flight_leaders_total`
     - `inference_single_flight_followers_total`
     - `inference_duplicate_compute_avoided_total`
   - **Freshness metrics** (4):
     - `inference_fresh_served_total`
     - `inference_stale_served_total`
     - `inference_missing_features_total`
     - `inference_stale_recompute_triggered_total`
   - **Persistence metrics** (3):
     - `inference_duckdb_write_failures_total`
     - `inference_write_latency_distribution` (histogram)
     - `inference_read_after_write_failures_total`

2. **Metric Emission in `coordinator.go`**:
   - On request entry: increment request counter
   - On cache hit: increment `CacheHits`, update `CacheHitRatio`
   - On cache miss: increment `CacheMisses`, update `CacheMissRatio`
   - On dedup leader: increment `LeaderCount`
   - On dedup follower: increment `FollowerCount`
   - On job completion: record end-to-end latency
   - On error: increment error counter by type
   - On timeout: increment timeout counter
   - Before dequeue: record queue length gauge

3. **Metric Emission in `python_client.go`**:
   - On worker spawn: set active workers gauge
   - On job dispatch: increment in-flight jobs gauge
   - On job completion: decrement in-flight jobs gauge
   - On worker crash: decrement active workers gauge

4. **Python Worker Instrumentation** (add to `worker.py`):
   - Feature load timing: log to stderr `[METRIC] feature_load_ms=45`
   - Inference timing: log to stderr `[METRIC] inference_ms=23`
   - Write timing: log to stderr `[METRIC] duckdb_write_ms=12`
   - Go parses stderr, updates histogram

5. **`/metrics` Endpoint**:
   - Register http.HandleFunc("/metrics", ExportMetrics)
   - Expose all Prometheus metrics in text format
   - Include timestamp for each scrape

**Rigorous Testing**:

**`metrics_test.go`** (Unit Tests):
- Test 1: Counter increment on error (all 8 error types)
- Test 2: Histogram observation records correct latency bucket
- Test 3: Gauge update reflects current in-flight jobs
- Test 4: Dedup metrics: leader vs follower counting
- Test 5: Cache metrics: hit/miss/ratio accuracy
- Test 6: Freshness metrics: fresh vs stale tracking
- Test 7: Persistence metrics: write failures tracked
- Test 8: Metrics reset between tests (no pollution)
- **Pass Criteria**: All 8 tests pass, metric accuracy ±1%

**`prometheus_format_test.go`** (Integration):
- Test 1: `/metrics` endpoint responds with 200 OK
- Test 2: Response format matches Prometheus text format spec
- Test 3: All 49 metrics present in output
- Test 4: Metric labels correct (e.g., error_type tags)
- Test 5: Histogram buckets present (le="0.01", "0.05", etc.)
- Test 6: Gauge values correct
- Test 7: Counter values monotonically increasing
- **Pass Criteria**: All 7 tests pass, format validation passes

**Integration Test** (`test_metrics_integration.sh`):
- Scenario 1: 10 successful requests → metrics reflect counts
- Scenario 2: 3 cache hits, 7 misses → hit ratio = 0.3
- Scenario 3: 5 concurrent identical requests → dedup leader=1, followers=4
- Scenario 4: 2 timeout errors → timeout counter incremented
- Scenario 5: `/metrics` scrape returns all metrics
- Scenario 6: Latency histogram p95 < 200ms (baseline)
- **Pass Criteria**: All 6 scenarios pass, metrics accurate within 1%

**Load Test Metrics** (`load_test_metrics.go`):
- Run 100 requests, measure:
  - Latency histogram (p50, p95, p99)
  - Error rate distribution
  - Cache hit ratio
  - Dedup effectiveness (follower count / total)
- **Pass Criteria**: 
  - p50 < 50ms, p95 < 150ms, p99 < 300ms
  - Error rate < 1%
  - Dedup followers > 60% on bursty load

**Success Metrics**:
- All 49 metrics emitted correctly
- Prometheus format valid (passvalidator)
- Metric accuracy: ±1% of actual values
- `/metrics` latency: < 100ms
- No memory leaks in metric storage

---

### Phase 2b Part 4: Worker Robustness + Integration (Week 3-4)

**Objective**: Implement full worker pool lifecycle, connection pooling, serialized writes, and feature freshness. Run all 11 integration scenarios + load tests.

**Implementation**:

1. **Worker Pool in `python_client.go`**:
   - Pre-spawn N workers at startup
   - Health check: goroutine per worker sends heartbeat via ping/pong JSON
   - If worker dies: mark `alive = false`, decrement active workers gauge
   - No respawn: external supervisor handles restart
   - Job routing: round-robin across alive workers

2. **Python Connection Pooling** (new `db_pool.py`):
   - Pool size: 4–8 read-only DuckDB connections (config)
   - Reuse connections across requests
   - Timeout on pool acquisition: 5s
   - Auto-return connection after job completes
   - Pool stats: available, in-use, total

3. **Serialized Writes** (new `persistence_queue.py`):
   - Single background thread (PersistenceQueue) processes writes
   - Worker pushes write request to queue
   - Queue serializes: acquires lock, inserts, reads back, commits
   - Worker blocks on write until confirmation
   - Read-after-write validation: query DB to verify written value matches

4. **Feature Freshness Checks** (add to `worker.py`):
   - After loading features, check age of each feature type:
     - PM25: if age > 1h, raise FeatureAssemblyError
     - Meteo: if age > 1h, raise FeatureAssemblyError
     - Landsat: if age > 16d, raise FeatureAssemblyError
     - Sentinel: if age > 5d, raise FeatureAssemblyError
   - Log: `[METRIC] freshness_check=PASS|FAIL` to stderr
   - Emit metrics: `FreshServed`, `StaleServed`, `MissingFeaturesCount`

5. **Worker Lifecycle Logging**:
   - Go logs: `[worker-0] spawned (PID=12345)`, `[worker-0] crashed (exit_code=1)`
   - Python logs: `[worker-0] initialized`, `[worker-0] job_start`, `[worker-0] job_end`

**Rigorous Testing**:

**`python_client_test.go`** (Unit Tests):
- Test 1: Worker pool spawns N workers
- Test 2: Job routing round-robin across workers
- Test 3: Worker death marked correctly
- Test 4: Remaining workers pick up load after crash
- Test 5: Health check detects dead worker
- Test 6: Active workers gauge decremented on crash
- Test 7: Graceful shutdown waits for in-flight jobs
- **Pass Criteria**: All 7 tests pass, no resource leaks

**`db_pool_test.py`** (Unit Tests):
- Test 1: Pool initializes with 4–8 connections
- Test 2: Acquire connection from pool
- Test 3: Return connection to pool
- Test 4: Pool exhaustion (acquire all, next waits)
- Test 5: Timeout on pool acquisition (5s)
- Test 6: Concurrent acquires (10 requests, 4 connections)
- Test 7: Pool stats reflect available/in-use
- **Pass Criteria**: All 7 tests pass, no connection leaks

**`persistence_queue_test.py`** (Unit Tests):
- Test 1: Single write serialized through queue
- Test 2: Multiple writes serialize (not concurrent)
- Test 3: Read-after-write validation succeeds
- Test 4: Write failure propagated back to worker
- Test 5: Queue persists across multiple jobs
- Test 6: Worker blocks until write confirmed
- Test 7: Persistence queue graceful shutdown
- **Pass Criteria**: All 7 tests pass, write latency < 50ms

**`freshness_test.py`** (Unit Tests):
- Test 1: Fresh feature passes check
- Test 2: Stale PM25 (>1h) raises error
- Test 3: Stale Meteo (>1h) raises error
- Test 4: Stale Landsat (>16d) raises error
- Test 5: Stale Sentinel (>5d) raises error
- Test 6: Missing feature raises error
- Test 7: Freshness metrics logged to stderr
- **Pass Criteria**: All 7 tests pass, freshness enforcement correct

**Integration Tests** (`integration_test.go` with 11 Scenarios):

1. **Happy Path**: Valid request → dedup leader → feature load → freshness check → inference → write → cache → response
2. **Dedup Leader + Followers**: 5 concurrent identical requests → 1 computes, 4 wait → all receive same result
3. **Cache Hit**: Request 1 computes + caches; Request 2 (within TTL) hits cache
4. **Cache Miss + Recompute**: Request 1 caches; TTL expires; Request 2 recomputes
5. **Queue Full Rejection**: Enqueue 100 jobs (at capacity), next job rejected (503 OverloadError)
6. **Timeout Enforcement**: Job takes >1s → killed → timeout error (504)
7. **Worker Crash**: Worker 1 crashes mid-job → Worker 2 picks up remaining jobs
8. **Feature Missing**: DuckDB missing feature → FeatureAssemblyError (422)
9. **Model Inference Error**: Model fails → ModelInferenceError (500)
10. **Write Failure**: DuckDB write fails → PersistenceError (500), result not cached
11. **Graceful Shutdown**: SIGINT received → wait for in-flight jobs → workers exit cleanly

**Pass Criteria for Each**:
- Scenario 1: Prediction returned within 1s, value matches Phase 1
- Scenario 2: Followers unblock instantly once leader completes
- Scenario 3: Cache hit latency < 1ms
- Scenario 4: TTL enforcement: cache expires at correct time
- Scenario 5: 101st request returns 503, not 504/500
- Scenario 6: Timeout < 1.1s, process killed
- Scenario 7: Remaining worker handles queue
- Scenario 8: Error message mentions missing feature
- Scenario 9: Error message from model propagates
- Scenario 10: Error not cached; next request retries
- Scenario 11: No requests dropped; in-flight jobs complete

**Load Tests** (`load_test.go`):

- **Baseline Load** (10 concurrent requests):
  - p50 < 50ms, p95 < 150ms, p99 < 300ms
  - Error rate < 1%
  - Dedup followers > 60% (5 concurrent identical)

- **Sustained Load** (100 concurrent requests):
  - p50 < 100ms, p95 < 250ms, p99 < 500ms
  - Queue length peaks at capacity-1
  - Error rate < 2%
  - Cache hit ratio > 70% (repeated requests)

- **Stress Load** (1000 concurrent requests, 100 varied keys):
  - p50 < 200ms, p95 < 500ms, p99 < 1000ms
  - Queue full occasionally (503 errors < 5%)
  - Timeout rate < 1% (no systemic slow jobs)
  - Dedup effective: compute count ≤ 100 (vs. 1000 requests)

**Pass Criteria for Load Tests**:
- All latency targets met
- Error rates within threshold
- No memory leaks
- Metrics accurate (within 2%)

**Success Metrics for Phase 2b Part 4**:
- All 11 integration scenarios pass
- Load tests pass (baseline, sustained, stress)
- Worker crash recovery < 1s (remaining workers pick up)
- Feature freshness enforcement 100% accurate
- Connection pool efficiency: 80%+ reuse
- Write serialization: no conflicts, 100% accuracy
- No data corruption (read-after-write always succeeds)

---

## 11. Deployment Artifacts

### 11.1 Directory Structure

```
Takshashila/
├── phase2_coordinator/
│   ├── main.go
│   ├── coordinator.go
│   ├── queue.go
│   ├── dedup.go
│   ├── cache.go
│   ├── metrics.go
│   ├── python_client.go
│   ├── config.go
│   ├── *_test.go (all tests)
│   ├── go.mod
│   ├── go.sum
│   └── Makefile
│
├── phase2_worker/
│   ├── worker.py
│   ├── feature_loader.py
│   ├── persistence.py (modified)
│   ├── config.py
│   ├── test_*.py (all tests)
│   └── requirements.txt
│
└── tests/
    └── integration/ (shared Go + Python tests)
        ├── integration_test.go
        ├── load_test.go
        └── scenarios/ (test fixtures)
```

### 11.2 Build & Run

**Go**:
```bash
cd phase2_coordinator
go build -o takshashila-coordinator
./takshashila-coordinator -workers=4 -queue-capacity=100
```

**Python**:
```bash
cd phase2_worker
python worker.py
```

**Docker** (future, not Phase 2):
- Dockerfile for Go coordinator
- Dockerfile for Python workers
- docker-compose.yml to orchestrate both

---

## 12. Freshness Configuration (LOCKED)

**Centralized reference**:

| Feature Type | Stale Threshold | DuckDB Table | Notes |
|--------------|-----------------|--------------|-------|
| PM25 predictions | > 1 hour | predictions | served (cached), not features |
| Meteorological | > 1 hour | features_meteo | fail on stale; trigger recompute |
| Landsat imagery | > 16 days | features_satellite | fail on stale |
| Sentinel imagery | > 5 days | features_satellite | fail on stale |

**Implementation**:
- Store in `config.go` and `config.py`
- Reference from cache TTL calculator
- Reference from feature loader freshness check
- Metrics track freshness outcomes

---

## 13. Locks & Explicit Decisions

### 13.1 Locked Decisions

✅ **Bounded queue with rejection**: Full queue → OverloadError 503  
✅ **Fixed worker pool**: Pre-spawned, no dynamic sizing  
✅ **Pre-spawn workers**: No on-demand spawn  
✅ **Job payload**: grid_id, timestamp_hour, request_id, reply_handle  
✅ **No Python respawn by Go**: Mark dead, rely on outer supervisor  
✅ **Worker timeout**: Kill job after 1s, return TimeoutError  
✅ **Feature missing**: Return error immediately, not cached  
✅ **Python writes only**: Go never writes to DuckDB  
✅ **Single writer serialization**: Python uses global lock for DuckDB writes  
✅ **Dedup per-(grid_id, timestamp)**: In-memory, within one instance  
✅ **Cache TTL aligned to freshness**: Configurable per feature type  
✅ **Metrics via `/metrics`**: Prometheus format, no stdout mixing  
✅ **Freshness thresholds**: PM25 1h, Meteo 1h, Landsat 16d, Sentinel 5d  

### 13.2 Explicitly Deferred

❌ **Dynamic worker scaling**: Phase 3+  
❌ **Priority queues**: Phase 3+  
❌ **Job stealing**: Phase 3+  
❌ **Distributed queues**: Phase 3+  
❌ **Persistent queues**: Phase 3+  
❌ **Model A/B testing**: Phase 3+  
❌ **Batch API**: Phase 3+  
❌ **Distributed dedup**: Phase 3+  

---

## 14. Success Criteria for Phase 2

### 14.1 Phase 2a (Core Infrastructure) ✅ COMPLETE

- [x] Go coordinator receives requests, routes to Python workers
- [x] Python workers return predictions
- [x] Queue bounded and rejects when full
- [x] IPC protocol (line-delimited JSON) working
- [x] Phase 1 integration: critical bugs fixed
- [x] Manual integration test passes (all 5 phases)
- [x] Real PM2.5 prediction: 41.82 µg/m³

### 14.2 Phase 2b Part 1: Cache + Dedup Foundation

**Functional**:
- [ ] Dedup map creates leaders and followers correctly
- [ ] Followers wait for leader result (no duplicate compute)
- [ ] Cache stores predictions with TTL
- [ ] Cache hit returns result without computation
- [ ] Expired cache entries trigger recompute
- [ ] Dedup timeout error propagates to followers

**Non-Functional**:
- [ ] Dedup reduces duplicate compute by ≥80% on 5 concurrent identical requests
- [ ] Cache hit latency: <1ms
- [ ] Cache TTL enforcement: ±10ms accuracy
- [ ] No memory leaks on 1000 dedup entries

### 14.3 Phase 2b Part 2: Timeout + Error Classification

**Functional**:
- [ ] Jobs exceeding 1s timeout are killed
- [ ] Timeout errors return 504 status
- [ ] Error types classified correctly (400/422/500/503/504)
- [ ] Errors not cached
- [ ] Dedup followers receive same error as leader

**Non-Functional**:
- [ ] Timeout enforcement: ±50ms accuracy
- [ ] Error classification: 100% mapping accuracy
- [ ] No false positives/negatives on error categorization

### 14.4 Phase 2b Part 3: Metrics & Observability

**Functional**:
- [ ] All 49 metrics emitted correctly
- [ ] `/metrics` endpoint responds with Prometheus format
- [ ] Error counters track all error types
- [ ] Latency histograms record correct buckets
- [ ] In-flight jobs gauge reflects current load
- [ ] Cache metrics (hit/miss/ratio) accurate

**Non-Functional**:
- [ ] Prometheus format validation passes
- [ ] Metric accuracy: ±1% of actual counts
- [ ] `/metrics` endpoint latency: <100ms
- [ ] No memory growth from metric storage

### 14.5 Phase 2b Part 4: Worker Robustness + Integration

**Functional**:
- [ ] All 11 integration scenarios pass:
  1. Happy path
  2. Dedup leader + followers
  3. Cache hit
  4. Cache miss + recompute
  5. Queue full rejection (503)
  6. Timeout enforcement (504)
  7. Worker crash recovery
  8. Feature missing error (422)
  9. Model inference error (500)
  10. Write failure handling
  11. Graceful shutdown
- [ ] Worker pool health checks work
- [ ] Worker crash reduces capacity (no cascading failure)
- [ ] Connection pooling reuses connections (80%+ efficiency)
- [ ] Serialized writes: no conflicts, 100% accuracy
- [ ] Feature freshness checks: 100% enforcement
- [ ] Graceful shutdown: no requests dropped

**Non-Functional - Load Tests**:
- [ ] **Baseline** (10 concurrent): p50<50ms, p95<150ms, p99<300ms, error<1%
- [ ] **Sustained** (100 concurrent): p50<100ms, p95<250ms, p99<500ms, error<2%
- [ ] **Stress** (1000 concurrent, 100 keys): p50<200ms, p95<500ms, p99<1s, error<5%
- [ ] Dedup effective: compute count ≤ number of unique keys
- [ ] Worker crash recovery: <1s until remaining workers pick up
- [ ] No data corruption (read-after-write: 100% success)

---

## 14.6 Phase 2b Summary: Four-Part Progressive Build

| Part | Focus | Duration | Key Deliverables | Testing |
|------|-------|----------|------------------|---------|
| **Part 1** | Cache + Dedup Foundation | Week 2 | dedup.go, cache.go, coordinator.go | 7+7+7 unit tests, 5 integration scenarios |
| **Part 2** | Timeout + Error Classification | Week 2 | Timeout enforcement, error taxonomy | 7+7+4 unit tests, 5 integration scenarios, 100% error mapping |
| **Part 3** | Metrics & Observability | Week 2-3 | metrics.go, /metrics endpoint, 49 metrics | 8+7 unit tests, Prometheus format validation, load test metrics |
| **Part 4** | Worker Robustness + Integration | Week 3-4 | python_client.go, db_pool.py, persistence_queue.py, feature freshness | 7+7+7+7 unit tests, **11 integration scenarios**, 3 load tests (baseline/sustained/stress) |

**Progression**:
- Part 1 provides dedup + cache (core optimization layer)
- Part 2 adds timeout + error handling (safety layer)
- Part 3 adds observability (monitoring layer)
- Part 4 integrates everything + stress testing (robustness layer)

**Total Testing Commitment**:
- Unit tests: 44 tests across all parts
- Integration tests: 16 scenarios (5+5+0+11)
- Load tests: 3 profiles (baseline/sustained/stress)
- **Total: 63 test cases + 3 load profiles**

---

## 15. Open Questions (to be answered during coding)

1. **Go HTTP framework**: Use `http.ServeMux` (std lib)? Or `gorilla/mux`? Or `gin`?
   - Rec: std lib for simplicity

2. **JSON marshaling**: Use `encoding/json`? Or `json` with custom types?
   - Rec: std lib is fine

3. **Python JSON protocol**: `json.dumps()`/`json.loads()` with line-buffering?
   - Rec: use `json` module + `sys.stdout.flush()`

4. **Go/Python subprocess exec**: Use `exec.Command`? Handle signals?
   - Rec: Use `exec.Command`, signal parent Go process on Python crash

5. **Metrics storage during startup**: Pre-allocated or lazy?
   - Rec: Pre-allocate at startup in `NewMetricsRegistry()`

6. **Testing DB state**: Fresh mock data for each test? Shared?
   - Rec: Fresh for each integration test, reuse Phase 1 mock fixtures

7. **Graceful shutdown**: What happens to in-flight jobs?
   - Rec: Wait for in-flight jobs up to 5s, then force kill

8. **Logging**: Structured? Unstructured?
   - Rec: Unstructured initially (stdlib log), structured later if needed

---

**This plan is complete and ready for implementation. All decisions are locked, architecture is clear, test strategy is defined.**

