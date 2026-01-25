## INTEGRATION PLAN: Go Coordinator + Python Orchestrator + Inference Workers

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Go Coordinator (Main Service - Port 8080)                   │
├─────────────────────────────────────────────────────────────┤
│ HTTP API:                                                   │
│  - POST /predict (grid_id, timestamp_hour)                 │
│     └─> Results cache → Dedup → compute (1-sec timeout)    │
│                                                              │
│ Components:                                                 │
│  - Job Queue: Request dispatch                              │
│  - Worker Pool: Spawns Python workers (currently dummy)     │
│  - Deduplicator: Prevents duplicate computation             │
│  - Result Cache: 5-minute TTL (success + errors)            │
│  - Prediction Cache: 2-hour TTL (new Phase 1)               │
│  - Health Checker: Worker health monitoring                 │
│  - Metrics: Prometheus (path, status, versions)             │
└─────────────────────────────────────────────────────────────┘
         │
         │ (Currently spawns dummy Python workers)
         │ (NEEDS: Spawn orchestrator instead)
         ↓
    [Dummy Workers]
    (No real computation)
```

**Problem:** The Go Coordinator's `WorkerPool` spawns dummy Python workers that don't do real inference. It ignores the orchestrator entirely.

---

### Proposed Architecture (After Integration)

```
┌─────────────────────────────────────────────────────────────┐
│ Go Coordinator (Main Service - Port 8080)                   │
├─────────────────────────────────────────────────────────────┤
│ handlePredict():                                            │
│  1. Check Redis cache (via Redis client)                    │
│     - Key: f"{grid_id}:{timestamp_hour}"                    │
│     - TTL: 2 hours (hot cache)                              │
│  2. If HIT: Return immediately (1-2ms)                      │
│  3. If MISS:                                                │
│     - Check dedup + old result cache                        │
│     - If leader: Ask orchestrator to predict batch          │
│     - If follower: Wait for leader via dedup               │
│  4. Return result                                            │
│                                                              │
│ Components:                                                 │
│  - Redis Client: Direct connection to cache                 │
│  - Orchestrator RPC: Ask for batch inference                │
│  - Dedup + Cache: As before                                 │
└─────────────────────────────────────────────────────────────┘
         │
         │ (On cache MISS + leader election)
         │ Send batch predict request (JSON over HTTP/gRPC)
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Python Orchestrator (Daemon - Started by Coordinator)       │
├─────────────────────────────────────────────────────────────┤
│ New Responsibilities:                                       │
│  1. Listen for batch predict requests from Coordinator      │
│     - HTTP endpoint: /batch_predict (POST)                  │
│     - Payload: {grids: [g1, g2, ...], timestamp_hour: ""}  │
│     - Returns: {results: {g1: 71.2, g2: 68.5, ...}}        │
│                                                              │
│  2. When request arrives:                                   │
│     a. Generate cycle_id                                    │
│     b. Spawn inference_job_worker(s) for this batch         │
│     c. Wait for predictions in Redis cache                  │
│     d. Collect results and return to Coordinator            │
│                                                              │
│  3. Scheduled jobs (unchanged):                             │
│     - WriterJob @ t=-2m (fetch API data)                    │
│     - Background InferenceJob for hot/warm/cold cache       │
│                                                              │
│ Components:                                                 │
│  - APScheduler: Cron jobs (WriterJob, scheduled inference)  │
│  - Redis Client: Read predictions from cache                │
│  - HTTP Server: Accept batch predict requests               │
│  - Process Manager: Spawn inference_job_worker              │
└─────────────────────────────────────────────────────────────┘
         │
         │ Spawn worker(s) for batch prediction
         ↓
┌─────────────────────────────────────────────────────────────┐
│ inference_job_worker.py (Spawned per batch/cycle)           │
├─────────────────────────────────────────────────────────────┤
│ For each grid in batch:                                     │
│  1. Fetch 64 features (parallel batch queries)              │
│  2. Run LightGBM model                                      │
│  3. Write to Redis cache (set key = prediction)             │
│  4. Queue to PersistenceQueue for DuckDB write              │
│                                                              │
│ Performance: 1,017 grids/sec (10k batch size)              │
└─────────────────────────────────────────────────────────────┘
```

---

## INTEGRATION PLAN (Phase 1: Coordinator → Orchestrator)

### Step 1: Add Redis Client to Go Coordinator

**File:** `main.go`

```go
type Coordinator struct {
    // ... existing fields ...
    
    // NEW: Redis connection for reading cache
    redisClient *redis.Client
    
    // NEW: Orchestrator HTTP client
    orchestratorClient *http.Client
    orchestratorURL    string  // e.g., "http://localhost:9999"
}

// In NewCoordinator():
func NewCoordinator(config *Config) (*Coordinator, error) {
    // ... existing setup ...
    
    // NEW: Initialize Redis client
    redisClient := redis.NewClient(&redis.Options{
        Addr: config.RedisHost + ":" + config.RedisPort,
    })
    
    // NEW: Initialize Orchestrator HTTP client
    orchestratorURL := os.Getenv("KASHI_ORCHESTRATOR_URL")
    if orchestratorURL == "" {
        orchestratorURL = "http://localhost:9999"  // Default
    }
    
    return &Coordinator{
        // ... existing fields ...
        redisClient:     redisClient,
        orchestratorClient: &http.Client{Timeout: 10 * time.Second},
        orchestratorURL: orchestratorURL,
    }, nil
}
```

### Step 2: Modify Coordinator's handlePredict()

**File:** `main.go` → `handlePredict()` method

Current flow:
```
1. Check old cache (5-minute TTL)
2. Dedup + leader election
3. executeComputation() (dummy, returns 0)
4. Store in old cache
```

New flow:
```
1. Check Redis cache (2-hour hot cache) ← NEW
2. Check old cache (5-minute result cache)
3. Dedup + leader election
4. If leader: Ask orchestrator for batch prediction ← NEW
5. Wait for Redis cache update (orchestrator populates it)
6. Return result
```

**Changes:**

```go
func (c *Coordinator) handlePredict(w http.ResponseWriter, r *http.Request) {
    // ... existing code ...
    
    // NEW: Step 1 - Check Redis cache FIRST (hot cache, 2-hour TTL)
    redisKey := fmt.Sprintf("%s:%s", req.GridID, req.TimestampHour)
    pm25Str, err := c.redisClient.Get(ctx, redisKey).Result()
    if err == nil {
        pm25, _ := strconv.ParseFloat(pm25Str, 64)
        log.Printf("[%s] Redis cache HIT", requestID)
        // ... return response ...
        return
    }
    
    // ... existing cache + dedup logic ...
    
    // NEW: When leader is elected, ask orchestrator instead of dummy compute
    if isLeader {
        pm25, err = c.askOrchestratorForPrediction(ctx, req.GridID, req.TimestampHour)
        if err != nil {
            // Fallback: recompute or retry
        }
    }
}
```

### Step 3: Add Orchestrator Communication Method

**File:** `main.go` → New method

```go
type BatchPredictRequest struct {
    Grids         []string `json:"grids"`
    TimestampHour string   `json:"timestamp_hour"`
}

type BatchPredictResponse struct {
    Results map[string]float64 `json:"results"`  // grid_id -> pm25
    Errors  map[string]string  `json:"errors"`   // grid_id -> error (if any)
}

func (c *Coordinator) askOrchestratorForPrediction(
    ctx context.Context, 
    gridID, 
    timestampHour string,
) (float64, error) {
    
    // Collect a small batch of grids (e.g., batch of 1 for now, or dedup window)
    batch := []string{gridID}
    
    // Send request to orchestrator
    reqBody := BatchPredictRequest{
        Grids:         batch,
        TimestampHour: timestampHour,
    }
    
    jsonBody, _ := json.Marshal(reqBody)
    
    req, _ := http.NewRequestWithContext(ctx, "POST", 
        c.orchestratorURL + "/batch_predict", 
        bytes.NewReader(jsonBody))
    
    resp, err := c.orchestratorClient.Do(req)
    if err != nil {
        return 0, fmt.Errorf("orchestrator request failed: %w", err)
    }
    defer resp.Body.Close()
    
    var batchResp BatchPredictResponse
    json.NewDecoder(resp.Body).Decode(&batchResp)
    
    pm25, ok := batchResp.Results[gridID]
    if !ok {
        return 0, fmt.Errorf("orchestrator returned no result for %s", gridID)
    }
    
    return pm25, nil
}
```

---

## Integration Plan (Phase 2: Orchestrator as Server)

### Step 1: Add HTTP Server to Python Orchestrator

**File:** `orchestrator.py` → New class

```python
from flask import Flask, request, jsonify
import threading

class OrchestratorServer:
    def __init__(self, host="localhost", port=9999):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route("/batch_predict", methods=["POST"])
        def batch_predict():
            """
            Receives batch prediction request from Go Coordinator.
            
            Request:
            {
                "grids": ["g1", "g2", ...],
                "timestamp_hour": "2026-01-25T14:00:00Z"
            }
            
            Response:
            {
                "results": {"g1": 71.2, "g2": 68.5, ...},
                "errors": {"g3": "Feature assembly failed"}
            }
            """
            data = request.json
            grids = data.get("grids", [])
            timestamp = data.get("timestamp_hour")
            
            # Generate cycle_id for this batch
            cycle_id = str(uuid.uuid4())
            
            # Spawn inference_job_worker for this batch
            results = self.predict_batch(grids, timestamp, cycle_id)
            
            return jsonify(results)
    
    def predict_batch(self, grids, timestamp, cycle_id):
        """Spawn worker and wait for Redis cache to be populated."""
        
        # Calculate grid ranges for this batch
        start_idx = min(grids) if grids else 0
        end_idx = max(grids) + 1 if grids else 1
        
        # Spawn inference_job_worker
        cmd = [
            "python3",
            "-m", "takshashila.distributed.worker.inference_job_worker",
            "--cycle-id", cycle_id,
            "--timestamp-hour", timestamp,
            "--grid-batch-start", str(start_idx),
            "--grid-batch-end", str(end_idx),
            "--cache-host", "localhost",
            "--cache-port", "6379",
        ]
        
        # Run worker and wait for completion
        try:
            subprocess.run(cmd, timeout=120, check=True)
        except subprocess.TimeoutExpired:
            logger.error(f"[{cycle_id}] Batch prediction timeout")
            return {"results": {}, "errors": {g: "Timeout" for g in grids}}
        
        # Fetch results from Redis cache
        redis_client = redis.Redis(host="localhost", port=6379)
        results = {}
        errors = {}
        
        for grid_id in grids:
            key = f"{grid_id}:{timestamp}"
            pm25_str = redis_client.get(key)
            if pm25_str:
                results[str(grid_id)] = float(pm25_str)
            else:
                errors[str(grid_id)] = "Not found in cache"
        
        return {"results": results, "errors": errors}
    
    def start(self):
        """Start HTTP server in background thread."""
        thread = threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.port),
            daemon=True
        )
        thread.start()
        logger.info(f"Orchestrator server started on {self.host}:{self.port}")

# In main():
if __name__ == "__main__":
    config = load_config()
    
    # Start HTTP server
    server = OrchestratorServer(port=9999)
    server.start()
    
    # Continue with existing APScheduler setup
    scheduler = BackgroundScheduler(timezone=pytz.UTC)
    # ... rest of scheduler setup ...
```

### Step 2: Modify inference_job_worker to Write to Redis

**File:** `inference_job_worker.py` → In prediction loop

Currently:
```python
# Write to cache
cache.set(grid_id, prediction, ttl=3600)

# Queue to DuckDB
persistence_queue.put(grid_id, prediction)
```

After integration:
```python
# Write to Redis cache (hot cache, 2-hour TTL)
redis_client.set(f"{grid_id}:{timestamp_hour}", pm25, ex=7200)

# Write to old cache (result cache)
cache.set(grid_id, prediction, ttl=3600)

# Queue to DuckDB (async)
persistence_queue.put(grid_id, prediction)
```

---

## PLAN: Reading from Redis Cache

### Architectural Decision: Redis as Cache Tier

Current three-tier cache:
```
Tier 1 (In-Go memory):     Old result cache (5 min TTL) [dedup window]
Tier 2 (Orchestrator):     PredictionCache (2 hour TTL) [phase 1]
Tier 3 (DuckDB):           predictions table [persistent]
```

**Problem:** Tiers 1 & 2 are isolated per-process. Multiple coordinator instances don't share cache.

**Solution:** Add Redis as shared tier:

```
Tier 1 (Redis):            Hot cache (2 hour TTL) [shared by all coordinators]
Tier 2 (In-Go memory):     Result cache (5 min TTL) [dedup window]
Tier 3 (DuckDB):           predictions table [persistent]
```

### Implementation Steps

**1. Add Redis client to Coordinator**
```go
import "github.com/redis/go-redis/v9"

type Coordinator struct {
    redisClient *redis.Client
    // ... other fields ...
}

// In NewCoordinator():
redisClient := redis.NewClient(&redis.Options{
    Addr: os.Getenv("REDIS_ADDR"), // "localhost:6379"
})
```

**2. Modify handlePredict() to check Redis first**
```go
func (c *Coordinator) handlePredict(...) {
    // Step 1: Check Redis (hot cache)
    redisKey := fmt.Sprintf("%s:%s", req.GridID, req.TimestampHour)
    pm25Str, err := c.redisClient.Get(ctx, redisKey).Result()
    if err == nil {
        // Cache HIT
        return pm25, "cache", nil
    }
    
    // Step 2-4: Check dedup + old cache + execute computation (as before)
}
```

**3. Orchestrator writes to Redis on prediction**
```python
# In inference_job_worker.py predict loop:
redis_client = redis.Redis(host=args.cache_host, port=args.cache_port)
redis_client.set(
    f"{grid_id}:{timestamp_hour}",
    pm25,
    ex=7200  # 2 hour TTL
)
```

**4. Multiple Coordinator instances share Redis**
```
Coordinator 1 ──┐
Coordinator 2 ──┼─→ [Redis Cache] ←─┐
Coordinator 3 ──┘                     │
                                      │
                                 Orchestrator
                              (spawns workers)
```

---

## Summary of Changes

### Go Coordinator (`main.go`)
| Change | Impact | Priority |
|--------|--------|----------|
| Add Redis client | Enables hot cache checks | P0 |
| Add orchestrator HTTP client | Enable RPC to orchestrator | P0 |
| Check Redis before cache | Share cache across coordinators | P0 |
| Ask orchestrator on cache miss | Real predictions instead of dummy | P1 |

### Python Orchestrator (`orchestrator.py`)
| Change | Impact | Priority |
|--------|--------|----------|
| Add Flask HTTP server (/batch_predict) | Accept requests from coordinator | P0 |
| Spawn inference_job_worker on request | Execute batch prediction | P0 |
| Read Redis cache after worker | Collect results for response | P0 |
| Configure server port (env: KASHI_ORCHESTRATOR_PORT) | Allow customization | P1 |

### Python inference_job_worker (`inference_job_worker.py`)
| Change | Impact | Priority |
|--------|--------|----------|
| Write PM25 to Redis cache | Make results available to coordinators | P0 |
| Use timestamp_hour from args | Proper cache key formation | Already done |
| 2-hour TTL on Redis | Persistent hot cache | P0 |

---

## Testing Plan

### Test 1: Single Coordinator, Orchestrator, 10 grids
```bash
# Start orchestrator
python3 -m takshashila.distributed.worker.orchestrator &

# Start coordinator
go run ./distributed/coordinator &

# Make prediction request
curl -X POST http://localhost:8080/predict \
  -d '{"grid_id":"g1","timestamp_hour":"2026-01-25T14:00:00Z"}' \
  -H "Content-Type: application/json"

# Expected: Returns PM25 from orchestrator computation
```

### Test 2: Cache hit verification
```bash
# Same request twice, measure latency
# First request: ~100ms (compute)
# Second request: ~5ms (Redis cache)
```

### Test 3: Multiple coordinators share cache
```bash
# Start 2 coordinators
go run ./distributed/coordinator &
go run ./distributed/coordinator &

# Request on coordinator 1: computes, writes to Redis
# Request on coordinator 2: reads from Redis (no computation)
```

---

## Estimated Implementation Time

- **Phase 1 (Coordinator → Orchestrator RPC):** 2-3 hours
  - Add Redis client + handlePredict changes
  - Add orchestrator HTTP client
  - Test single request flow

- **Phase 2 (Orchestrator as Server):** 1-2 hours
  - Add Flask server to orchestrator
  - Update inference_job_worker Redis writes
  - Test batch prediction

- **Phase 3 (Tuning):** 1 hour
  - Configure pool sizes
  - Optimize batch sizes
  - Stress test with 100+ concurrent requests

**Total: 4-6 hours for complete integration**

