# Coordinator Migration Plan: From Job Router to Cache Query Handler

**Status:** Planning (When Kashi scheduler implemented)  
**Timeline:** Phase 3  
**Impact:** `takshashila/distributed/coordinator/main.go`

---

## **1. Current Architecture (Phase 2a)**

```
Coordinator (Go)
├── main.go
│   ├── HTTP handlers (/predict, /health)
│   ├── JobQueue
│   └── WorkerPool
│       ├── Spawns N Python workers (worker.py)
│       ├── Routes inference jobs to workers
│       ├── Health checker (detects crashes)
│       └── Result channels (receives predictions)
└── python_client.go (WorkerPool implementation)

Request Flow:
1. User: POST /predict {grid_id, timestamp}
2. Coordinator: Create job, enqueue to JobQueue
3. JobQueue → Dispatcher: Route to healthy worker
4. Worker: Call InferenceEngine.predict()
5. Worker: Return JSON result on stdout
6. Coordinator: Receive result via channel
7. User: Return PM25 prediction
```

---

## **2. Final Architecture (With Kashi Scheduler)**

```
Coordinator (Go)
├── main.go
│   ├── HTTP handlers (/predict, /health)
│   └── Cache query logic (Redis or DuckDB lookup)
│
Kashi subprocess (Python) - Started by Coordinator
├── orchestrator.py (APScheduler)
│   ├── @ t=-2m: WriterJob subprocess
│   │   └── Fetches from 4 APIs → writes to DuckDB
│   └── @ t=0m: InferenceJob subprocess
│       ├── Spawns local WorkerPool
│       ├── Routes jobs to workers
│       └── Writes predictions to DuckDB + cache
└── worker.py (managed by Python orchestrator, not Go)

Request Flow:
1. User: POST /predict {grid_id, timestamp}
2. Coordinator: Query cache (Redis) or DuckDB
3. If cache hit: Return prediction (instant)
4. If cache miss: Query predictions table from DuckDB (fallback)
5. User: Return PM25 prediction (from cache or DuckDB)
```

---

## **3. Changes to `main.go`**

### **3.1 Remove: Job Queue Infrastructure**

**Delete from `main.go`:**
```go
// DELETE THESE:
type Job struct {
    RequestID    string    `json:"request_id"`
    GridID       string    `json:"grid_id"`
    TimestampHour string   `json:"timestamp_hour"`
    CreatedAt    time.Time `json:"created_at"`
}

type JobQueue struct {
    queue chan Job
    mu    sync.Mutex
    open  bool
}

func NewJobQueue(size int) *JobQueue { ... }
func (jq *JobQueue) Enqueue(job Job) error { ... }
func (jq *JobQueue) Close() { ... }
```

**Rationale:** Kashi orchestrator manages its own job scheduling with APScheduler. Coordinator no longer routes inference jobs.

---

### **3.2 Remove: Worker Pool**

**Delete from `main.go`:**
```go
// DELETE THESE:
var workerPool *WorkerPool

// In main():
// workerPool, err := NewWorkerPool(numWorkers, pythonPath, queue)
// defer workerPool.Shutdown()

// In dispatchJobs():
// job := Job{...}
// workerPool.queue.Enqueue(job)
```

**Delete file:** `python_client.go` (no longer needed)

**Rationale:** Kashi has its own worker pool managed by Python orchestrator. Go coordinator doesn't spawn workers anymore.

---

### **3.3 Modify: /predict Handler**

**OLD (Phase 2a):**
```go
func handlePredict(w http.ResponseWriter, r *http.Request) {
    var req struct {
        GridID        string `json:"grid_id"`
        TimestampHour string `json:"timestamp_hour"`
    }
    json.NewDecoder(r.Body).Decode(&req)
    
    // Create job
    job := Job{
        RequestID:     uuid.New().String(),
        GridID:        req.GridID,
        TimestampHour: req.TimestampHour,
        CreatedAt:     time.Now(),
    }
    
    // Route to worker pool
    ch := workerPool.RegisterResultChannel(job.RequestID)
    workerPool.queue.Enqueue(job)
    
    // Wait for result
    select {
    case result := <-ch:
        json.NewEncoder(w).Encode(result)
    case <-time.After(30 * time.Second):
        w.WriteHeader(http.StatusGatewayTimeout)
    }
}
```

**NEW (With Kashi):**
```go
func handlePredict(w http.ResponseWriter, r *http.Request) {
    var req struct {
        GridID        string `json:"grid_id"`
        TimestampHour string `json:"timestamp_hour"`
    }
    json.NewDecoder(r.Body).Decode(&req)
    
    // Query cache first (Redis)
    prediction, found := cache.Get(req.GridID)  // from cache.go
    if found {
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]interface{}{
            "grid_id":       req.GridID,
            "pm25":          prediction.Value,
            "timestamp":     prediction.Timestamp,
            "from_cache":    true,
        })
        return
    }
    
    // Fallback: Query DuckDB predictions table
    prediction, err := queryPredictionsDB(req.GridID, req.TimestampHour)
    if err == nil {
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]interface{}{
            "grid_id":       req.GridID,
            "pm25":          prediction.Value,
            "timestamp":     prediction.Timestamp,
            "from_cache":    false,
            "stale":         true,
        })
        return
    }
    
    // No data available
    w.WriteHeader(http.StatusNotFound)
    json.NewEncoder(w).Encode(map[string]string{
        "error": "prediction not available",
    })
}
```

---

### **3.4 Add: Cache Integration**

**New file: `cache.go`**
```go
package main

import (
    "github.com/redis/go-redis/v9"
    "context"
)

type Prediction struct {
    Value     float64
    Timestamp string
}

type CacheProvider interface {
    Get(gridID string) (Prediction, bool)
    Set(gridID string, prediction Prediction, ttl time.Duration) error
}

// RedisCache implements CacheProvider
type RedisCache struct {
    client *redis.Client
}

func NewRedisCache(addr string) *RedisCache {
    return &RedisCache{
        client: redis.NewClient(&redis.Options{
            Addr: addr,
        }),
    }
}

func (rc *RedisCache) Get(gridID string) (Prediction, bool) {
    ctx := context.Background()
    val, err := rc.client.Get(ctx, "cache:"+gridID).Result()
    if err != nil {
        return Prediction{}, false
    }
    
    var pred Prediction
    json.Unmarshal([]byte(val), &pred)
    return pred, true
}

func (rc *RedisCache) Set(gridID string, prediction Prediction, ttl time.Duration) error {
    ctx := context.Background()
    data, _ := json.Marshal(prediction)
    return rc.client.Set(ctx, "cache:"+gridID, string(data), ttl).Err()
}
```

---

### **3.5 Add: DuckDB Query Helper**

**New function in `main.go`:**
```go
func queryPredictionsDB(gridID string, timestampHour string) (Prediction, error) {
    conn := duckdb.Open("takshashila/duckdb/predictions.duckdb")
    defer conn.Close()
    
    row := conn.QueryRow(
        `SELECT pm25_prediction, created_at 
         FROM predictions 
         WHERE grid_id = $1 
         AND timestamp_hour = $2
         ORDER BY created_at DESC 
         LIMIT 1`,
        gridID,
        timestampHour,
    )
    
    var pm25 float64
    var createdAt string
    err := row.Scan(&pm25, &createdAt)
    if err != nil {
        return Prediction{}, err
    }
    
    return Prediction{
        Value:     pm25,
        Timestamp: createdAt,
    }, nil
}
```

---

### **3.6 Add: Kashi Subprocess Management**

**In `main()`:**
```go
func main() {
    // ... existing setup ...
    
    // Start Kashi subprocess
    kashiCmd := exec.Command("python", "-m", "kashi")
    kashiCmd.Stdout = os.Stdout
    kashiCmd.Stderr = os.Stderr
    
    if err := kashiCmd.Start(); err != nil {
        log.Fatalf("Failed to start Kashi: %v", err)
    }
    
    log.Printf("Kashi subprocess started (PID: %d)", kashiCmd.Process.Pid)
    
    // Initialize cache
    cache := NewRedisCache("localhost:6379")
    
    // ... existing HTTP handlers ...
    
    // Wait for signals
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    
    <-sigChan
    log.Println("Shutting down...")
    
    // Kill Kashi subprocess
    kashiCmd.Process.Kill()
    kashiCmd.Wait()
}
```

---

## **4. Files to Delete**

```
✗ takshashila/distributed/coordinator/python_client.go (no longer used)
```

---

## **5. Files to Create**

```
✓ takshashila/distributed/coordinator/cache.go (Redis interface)
```

---

## **6. Files to Keep Unchanged**

```
✓ takshashila/distributed/coordinator/main.go (modified as above)
✓ takshashila/distributed/coordinator/health.go (if exists)
✓ takshashila/distributed/coordinator/middleware.go (if exists)
✓ HTTP handler setup (just remove /predict logic)
```

---

## **7. Summary of Changes**

| Component | Current | Final | Status |
|-----------|---------|-------|--------|
| JobQueue | ✓ (routes jobs) | ✗ (removed) | Delete |
| WorkerPool | ✓ (manages workers) | ✗ (removed) | Delete |
| python_client.go | ✓ (spawns workers) | ✗ (removed) | Delete |
| /predict handler | Route to worker | Query cache | Modify |
| Cache integration | None | Redis | Add |
| DuckDB fallback | None | Query predictions table | Add |
| Kashi subprocess | None | Managed by Coordinator | Add |

---

## **8. Implementation Order**

1. ✓ Implement Kashi orchestrator + WriterJob + InferenceJob
2. Add cache.go (Redis interface)
3. Add DuckDB query helper to main.go
4. Modify /predict handler to query cache
5. Add Kashi subprocess start logic to main()
6. Remove JobQueue, WorkerPool, python_client.go
7. Test end-to-end

---

## **9. Benefits**

- ✅ Coordinator becomes lightweight (cache query handler only)
- ✅ Kashi manages inference scheduling independently
- ✅ WorkerPool complexity moved to Python (easier to test)
- ✅ Cache hits are instant (no worker dispatch latency)
- ✅ Graceful degradation (fallback to DuckDB if cache unavailable)

---

**Next Phase:** Begin Kashi implementation (WriterJob, orchestrator)
