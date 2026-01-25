# INTEGRATION PLAN: Go Coordinator reads from Redis + DuckDB Predictions

## Architecture (Simplified - Read-Only Model)

```
┌─────────────────────────────────────────────────────────────────┐
│ Python Orchestrator (Scheduled daemon, runs hourly)             │
├─────────────────────────────────────────────────────────────────┤
│ t=-2m: WriterJob                                                │
│        └─> Fetch from APIs, write to DuckDB tables              │
│                                                                  │
│ t=0m:  InferenceJob (hot + warm + cold)                         │
│        └─> Load features, run model, write predictions          │
│        └─> Write to Redis cache (2-hour TTL)                    │
│        └─> Write to DuckDB predictions table (persistent)       │
└─────────────────────────────────────────────────────────────────┘
         │                                              
         │ Pre-computed results flow up
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Go Coordinator (Main API service - Port 8080)                   │
├─────────────────────────────────────────────────────────────────┤
│ POST /predict (grid_id, timestamp_hour):                        │
│                                                                  │
│  1. Check Redis cache                                           │
│     Key: "{grid_id}:{timestamp_hour}"                           │
│     If HIT → return value (2-3ms)                               │
│                                                                  │
│  2. Check DuckDB predictions table                              │
│     Query: SELECT pm25 WHERE grid_id=? AND timestamp_hour=?    │
│     If FOUND → return value (5-10ms)                            │
│                                                                  │
│  3. Neither found → FAIL LOUDLY with 404/error                 │
│     Status: "Prediction not available for this grid/time"       │
│     (Don't compute - that's orchestrator's job)                 │
│                                                                  │
│ Dedup + caching still applies for result cache (5-min TTL)      │
└─────────────────────────────────────────────────────────────────┘
```

## Read-Only Responsibility Split

| Component | Responsibility |
|-----------|-----------------|
| **Orchestrator** (Python) | Compute predictions (every hour), write to Redis + DuckDB |
| **Coordinator** (Go) | Serve pre-computed predictions via REST API |

---

## Implementation Steps

### Step 1: Add Redis + DuckDB clients to Go Coordinator

**File:** `coordinator/main.go`

```go
type Coordinator struct {
    // ... existing fields ...
    
    // NEW: Redis connection for hot cache
    redisClient *redis.Client
    
    // NEW: DuckDB connection for persistent cache
    duckdbConn  *duckdb.Conn
}

func NewCoordinator(config *Config) (*Coordinator, error) {
    // ... existing code ...
    
    // NEW: Initialize Redis client
    redisClient := redis.NewClient(&redis.Options{
        Addr: os.Getenv("REDIS_ADDR"),  // "localhost:6379"
    })
    
    // NEW: Initialize DuckDB predictions connection
    db, err := duckdb.Open(config.DuckDBPredictionsPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open DuckDB predictions: %w", err)
    }
    conn, err := db.Conn(context.Background())
    if err != nil {
        return nil, fmt.Errorf("failed to get DuckDB connection: %w", err)
    }
    
    return &Coordinator{
        // ... existing fields ...
        redisClient: redisClient,
        duckdbConn:  conn,
    }, nil
}
```

### Step 2: Modify handlePredict() to read from cache tiers

**File:** `coordinator/main.go` → `handlePredict()` method

Replace the `executeComputation()` call with:

```go
func (c *Coordinator) handlePredict(w http.ResponseWriter, r *http.Request) {
    // ... existing validation code ...
    
    // Generate request ID
    requestID := uuid.New().String()
    
    // ... existing context setup ...
    
    // NEW: Execute prediction with cache tiers
    pm25, path, err := c.readPredictionFromCache(ctx, req.GridID, req.TimestampHour, requestID)
    
    // ... rest of existing response code ...
}

// NEW method: Read from Redis → DuckDB, fail loudly if not found
func (c *Coordinator) readPredictionFromCache(
    ctx context.Context,
    gridID, 
    timestampHour,
    requestID string,
) (float64, string, error) {
    
    // Step 1: Try Redis cache (hot cache, 2-hour TTL)
    redisKey := fmt.Sprintf("%s:%s", gridID, timestampHour)
    pm25Str, err := c.redisClient.Get(ctx, redisKey).Result()
    if err == nil {
        pm25, _ := strconv.ParseFloat(pm25Str, 64)
        log.Printf("[%s] Redis cache HIT for %s:%s", requestID, gridID, timestampHour)
        return pm25, "redis_cache", nil
    }
    
    log.Printf("[%s] Redis cache MISS for %s:%s", requestID, gridID, timestampHour)
    
    // Step 2: Try DuckDB predictions table (persistent cache)
    query := `
        SELECT pm25 
        FROM predictions 
        WHERE grid_id = $1 AND timestamp_hour = $2
        LIMIT 1
    `
    
    var pm25 float64
    row := c.duckdbConn.QueryRowContext(ctx, query, gridID, timestampHour)
    if err := row.Scan(&pm25); err == nil {
        log.Printf("[%s] DuckDB cache HIT for %s:%s", requestID, gridID, timestampHour)
        
        // Populate Redis for next request
        go func() {
            c.redisClient.Set(context.Background(), redisKey, pm25, 2*time.Hour)
        }()
        
        return pm25, "duckdb_cache", nil
    }
    
    // Step 3: FAIL LOUDLY - prediction not available
    log.Printf("[%s] Prediction NOT FOUND for %s:%s", requestID, gridID, timestampHour)
    return 0, "not_found", NewPredictionNotAvailableError(gridID, timestampHour)
}
```

### Step 3: Define error type for missing predictions

**File:** `coordinator/errors.go`

```go
type PredictionNotAvailableError struct {
    GridID        string
    TimestampHour string
}

func NewPredictionNotAvailableError(gridID, timestampHour string) *PredictionNotAvailableError {
    return &PredictionNotAvailableError{
        GridID:        gridID,
        TimestampHour: timestampHour,
    }
}

func (e *PredictionNotAvailableError) Error() string {
    return fmt.Sprintf(
        "Prediction not available for grid_id=%s, timestamp_hour=%s. "+
        "Check if orchestrator has computed for this time.",
        e.GridID, e.TimestampHour,
    )
}

func (e *PredictionNotAvailableError) HTTPStatus() int {
    return http.StatusNotFound // 404
}
```

### Step 4: Remove dummy worker spawning

**File:** `coordinator/python_client.go`

Delete or disable:
- `WorkerPool.dispatchJobs()`
- `WorkerPool.readResults()`
- All worker spawning logic

Just keep worker pool initialization but don't spawn any workers.

---

## Configuration

Add to Go coordinator config:

```yaml
# kashi/config.yaml
coordinator:
  redis_addr: "localhost:6379"
  duckdb_predictions_path: "/Users/shlokjain/pollu/artifacts/predictions.duckdb"
```

Or via environment:
```bash
export REDIS_ADDR="localhost:6379"
export DUCKDB_PREDICTIONS_PATH="/Users/shlokjain/pollu/artifacts/predictions.duckdb"
```

---

## Behavior: What Happens on Cache Miss

### Scenario 1: Grid computed in previous hour (exists in DuckDB)
```
Request: grid_id=g123, timestamp_hour=2026-01-25T13:00:00Z (1 hour ago)

1. Redis check: MISS (cache expired or not written)
2. DuckDB check: HIT (orchestrator computed it yesterday)
3. Response: Return PM2.5 value + repopulate Redis
```

### Scenario 2: Grid not yet computed (orchestrator still running)
```
Request: grid_id=g123, timestamp_hour=2026-01-25T14:00:00Z (current hour)

1. Redis check: MISS (orchestrator still running)
2. DuckDB check: MISS (orchestrator hasn't finished)
3. Response: 404 error - "Prediction not available, retry in 1-2 minutes"
```

### Scenario 3: Grid doesn't exist in data
```
Request: grid_id=g999999, timestamp_hour=2026-01-25T13:00:00Z

1. Redis check: MISS
2. DuckDB check: MISS (grid never computed)
3. Response: 404 error - "Grid not found"
```

---

## Testing Plan

### Test 1: Redis cache hit
```bash
# Pre-populate Redis
redis-cli SET "g1:2026-01-25T14:00:00Z" 71.5

# Request
curl -X POST http://localhost:8080/predict \
  -d '{"grid_id":"g1","timestamp_hour":"2026-01-25T14:00:00Z"}'

# Expected: Returns 71.5 in ~2-3ms
```

### Test 2: DuckDB fallback
```bash
# Clear Redis
redis-cli DEL "g1:2026-01-25T14:00:00Z"

# Ensure DuckDB has value
sqlite3 artifacts/predictions.duckdb "SELECT pm25 FROM predictions WHERE grid_id='g1'"
# Returns: 71.5

# Request
curl -X POST http://localhost:8080/predict \
  -d '{"grid_id":"g1","timestamp_hour":"2026-01-25T14:00:00Z"}'

# Expected: Returns 71.5 in ~5-10ms (DuckDB read)
# Side effect: Repopulates Redis for next request
```

### Test 3: Missing prediction (fail loudly)
```bash
# Clear both Redis and DuckDB
redis-cli DEL "g999:2026-01-25T14:00:00Z"
# (Prediction doesn't exist in DuckDB)

# Request
curl -X POST http://localhost:8080/predict \
  -d '{"grid_id":"g999","timestamp_hour":"2026-01-25T14:00:00Z"}'

# Expected: 404 error
# Response: {"error": {"code": "PREDICTION_NOT_AVAILABLE", "message": "..."}}
```

---

## Implementation Effort

| Step | Component | Files | Effort | Priority |
|------|-----------|-------|--------|----------|
| 1 | Redis + DuckDB clients | main.go, config.go | 30min | P0 |
| 2 | readPredictionFromCache() | main.go | 45min | P0 |
| 3 | Error handling | errors.go | 15min | P0 |
| 4 | Remove worker spawning | python_client.go | 15min | P0 |
| 5 | Testing + integration | main.go (test file) | 1h | P1 |

**Total: 2-2.5 hours for complete integration**

---

## What's NOT Included (For Later)

- ❌ Dynamic computation in coordinator (that's orchestrator's job)
- ❌ Batch coalescing (dedup window optimization)
- ❌ Worker spawning from coordinator
- ❌ Orchestrator RPC/HTTP endpoint

## Monitoring

Log every miss and hit:
```
[req-123] Prediction MISS for g1:2026-01-25T14:00:00Z
[req-123] Checking DuckDB...
[req-123] DuckDB MISS - returning 404
```

This will help identify if:
- Orchestrator is running late
- Redis connection issues
- DuckDB corruption
- Time synchronization issues

