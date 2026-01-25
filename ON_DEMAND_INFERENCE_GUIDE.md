# On-Demand Inference: Orchestrator Endpoint Implementation

## Overview

The Go Coordinator now supports **on-demand inference** when predictions are not found in Redis or DuckDB. It sends an HTTP request to the Python Orchestrator, which spawns an inference worker asynchronously.

---

## What Was Implemented (Go Coordinator)

✅ **HTTP client** for orchestrator communication
✅ **On-demand request** when cache miss occurs (Redis + DuckDB both miss)
✅ **Polling mechanism** to wait for worker to populate Redis cache
✅ **Graceful fallback** if orchestrator fails or times out

---

## What Needs to Be Implemented (Python Orchestrator)

### File: `takshashila/distributed/worker/orchestrator.py`

Add Flask HTTP endpoint to receive on-demand inference requests from Coordinator.

### Endpoint Specification

**Route:** `POST /inference`

**Request Payload:**
```json
{
  "grids": ["g1", "g2", "..."],
  "timestamp_hour": "2026-01-25T14:00:00Z",
  "request_id": "req-uuid-123",
  "cycle_id": "cycle-uuid-456"
}
```

**Response (Immediate):**
```json
{
  "status": "queued",
  "cycle_id": "cycle-uuid-456",
  "message": "Worker spawned, check Redis cache"
}
```

### Implementation Example

```python
from flask import Flask, request, jsonify
import subprocess
import threading
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route("/inference", methods=["POST"])
def on_demand_inference():
    """
    Handle on-demand inference requests from Go Coordinator.
    Spawns inference_job_worker asynchronously and returns immediately.
    Worker writes results to Redis, coordinator polls for completion.
    """
    try:
        data = request.json
        grids = data.get("grids", [])
        timestamp_hour = data.get("timestamp_hour")
        request_id = data.get("request_id")
        cycle_id = data.get("cycle_id")
        
        logger.info(f"[{request_id}] On-demand inference request: {len(grids)} grids, timestamp={timestamp_hour}")
        
        # Parse grid IDs to get numeric range
        # Assuming grid IDs are like "g1", "g2", etc.
        grid_nums = []
        for grid_id in grids:
            try:
                num = int(grid_id.replace("g", ""))
                grid_nums.append(num)
            except ValueError:
                logger.warning(f"Invalid grid_id format: {grid_id}")
        
        if not grid_nums:
            return jsonify({
                "status": "error",
                "message": "No valid grid IDs provided"
            }), 400
        
        start_idx = min(grid_nums)
        end_idx = max(grid_nums) + 1
        
        # Spawn worker in background thread (non-blocking)
        def spawn_worker():
            try:
                cmd = [
                    "python3",
                    "-m", "takshashila.distributed.worker.inference_job_worker",
                    "--cycle-id", cycle_id,
                    "--timestamp-hour", timestamp_hour,
                    "--grid-batch-start", str(start_idx),
                    "--grid-batch-end", str(end_idx),
                    "--cache-host", "localhost",
                    "--cache-port", "6379",
                ]
                
                logger.info(f"[{request_id}] Spawning worker: grids [{start_idx}, {end_idx})")
                
                result = subprocess.run(
                    cmd,
                    timeout=120,  # 2 minutes max
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"[{request_id}] Worker completed successfully")
                else:
                    logger.error(f"[{request_id}] Worker failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"[{request_id}] Worker timeout after 120s")
            except Exception as e:
                logger.error(f"[{request_id}] Worker spawn error: {e}")
        
        # Start worker thread (daemon so it doesn't block shutdown)
        worker_thread = threading.Thread(
            target=spawn_worker,
            daemon=True,
            name=f"worker-{cycle_id[:8]}"
        )
        worker_thread.start()
        
        return jsonify({
            "status": "queued",
            "cycle_id": cycle_id,
            "message": "Worker spawned, polling Redis for results",
            "grids_count": len(grids),
        }), 200
        
    except Exception as e:
        logger.exception(f"Error handling on-demand inference: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Start Flask server
if __name__ == "__main__":
    # Run Flask in background thread
    from threading import Thread
    
    def run_flask():
        app.run(host="0.0.0.0", port=9999, debug=False)
    
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Continue with existing APScheduler setup
    # ... (rest of orchestrator code)
```

---

## Integration Points

### 1. Coordinator Flow (Go)

```
POST /predict
    ↓
Check Redis cache → MISS
    ↓
Check DuckDB → MISS
    ↓
POST http://localhost:9999/inference
    ↓
Poll Redis every 200ms (90s timeout)
    ↓
Result found → Return to client
```

### 2. Orchestrator Flow (Python)

```
POST /inference received
    ↓
Parse grid IDs → [1, 2, 3, ...]
    ↓
Spawn inference_job_worker (background thread)
    ↓
Return 200 OK immediately
    ↓
Worker computes predictions
    ↓
Worker writes to Redis + DuckDB
    ↓
Coordinator's polling finds result
```

---

## Configuration

### Environment Variables

**Coordinator (Go):**
```bash
export KASHI_ORCHESTRATOR_URL="http://localhost:9999"
```

**Orchestrator (Python):**
```bash
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="9999"
```

---

## Testing

### Test 1: Pre-populate cache (should not trigger orchestrator)

```bash
# Populate Redis
redis-cli SET "g1:2026-01-25T14:00:00Z" "71.5"

# Request
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"grid_id":"g1","timestamp_hour":"2026-01-25T14:00:00Z"}'

# Expected: Immediate response from Redis (no orchestrator call)
```

### Test 2: Cache miss (triggers on-demand)

```bash
# Clear cache
redis-cli DEL "g999:2026-01-25T14:00:00Z"

# Request
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"grid_id":"g999","timestamp_hour":"2026-01-25T14:00:00Z"}'

# Expected:
# 1. Coordinator calls orchestrator
# 2. Orchestrator spawns worker
# 3. Worker computes and writes to Redis
# 4. Coordinator polls and finds result
# 5. Returns prediction (takes ~1-10 seconds)
```

### Test 3: Orchestrator failure

```bash
# Stop orchestrator
# pkill -f orchestrator

# Request
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"grid_id":"g1","timestamp_hour":"2026-01-25T14:00:00Z"}'

# Expected: 404 error - "Prediction not available"
```

---

## Performance Characteristics

| Scenario | Latency | Throughput |
|----------|---------|------------|
| Redis cache HIT | 2-5ms | 1000+ req/s |
| DuckDB cache HIT | 5-15ms | 500+ req/s |
| On-demand inference | 1-10s | 0.1-1 req/s |

**Note:** On-demand inference is **slow** by design (spawns worker, computes features, runs model). Use scheduled orchestrator for bulk predictions.

---

## Logging

**Coordinator logs:**
```
[req-123] Prediction NOT FOUND - requesting on-demand inference for g1:2026-01-25T14:00:00Z
[req-123] Sending on-demand inference request to orchestrator (cycle_id=cycle-456)
[req-123] Orchestrator response: queued (cycle_id=cycle-456)
[req-123] Polling Redis for result (timeout=90s)
[req-123] Poll SUCCESS - found result in Redis (pm25=71.50)
[req-123] On-demand inference SUCCEEDED (pm25=71.50)
```

**Orchestrator logs:**
```
[req-123] On-demand inference request: 1 grids, timestamp=2026-01-25T14:00:00Z
[req-123] Spawning worker: grids [1, 2)
[req-123] Worker completed successfully
```

---

## Deduplication Concerns

**Problem:** Multiple concurrent requests for same grid+timestamp could spawn multiple workers.

**Solution (Phase 2 - Optional):**

Add dedup lock in orchestrator:

```python
import redis
from redis.lock import Lock

redis_client = redis.Redis(host="localhost", port=6379)

@app.route("/inference", methods=["POST"])
def on_demand_inference():
    data = request.json
    grids = data["grids"]
    timestamp_hour = data["timestamp_hour"]
    
    # Dedup key
    dedup_key = f"inference_lock:{grids[0]}:{timestamp_hour}"
    
    # Try to acquire lock
    lock = redis_client.lock(dedup_key, timeout=120, blocking_timeout=1)
    if not lock.acquire(blocking=False):
        # Someone else already computing
        return jsonify({
            "status": "deduped",
            "message": "Another worker already computing this grid"
        }), 200
    
    try:
        # Spawn worker
        spawn_worker()
    finally:
        lock.release()
    
    return jsonify({"status": "queued"})
```

---

## Next Steps

1. ✅ **Coordinator implementation** (Done - this session)
2. ⏳ **Orchestrator endpoint** (Needs implementation)
3. ⏳ **End-to-end testing** (Redis + DuckDB + on-demand flow)
4. ⏳ **Dedup logic** (Optional, Phase 2)
5. ⏳ **Metrics & monitoring** (Track on-demand request rate)

