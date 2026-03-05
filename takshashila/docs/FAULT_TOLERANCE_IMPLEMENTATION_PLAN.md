# Takshashila Fault Tolerance Implementation Plan

**Finalized:** January 24, 2026  
**Architecture:** Complete re-execution from failure point (no mid-cycle resume)  
**State Persistence:** DuckDB with minimal cycle metadata  
**Timeout Strategy:** Hard deadline on Takshashila startup per schedule, aggressive restart  

---

## 1. Cycle Execution Model

### 1.1 Normal Operation (No Failures)

```
t = -5m:00s → WriterJob spawned
    -5m:00s to -4m:00s → WriterJob executes
             (20-40s typical, 60s worst case)

t = 0m:00s → InferenceJob spawned (only if WriterJob validates atomically complete)
    0m:00s to 1m:15s → InferenceJob executes
             (50-75s typical, 85s worst case)
             | → Hot cache ready at ~55s
             | → Warm cache ready at ~85s

t = 1m:15s → Cycle complete, next cycle scheduled for t=1h
```

### 1.2 Failure Modes & Recovery

| Failure Point | Detection | Action | Outcome |
|---|---|---|---|
| WriterJob crashes mid-execution | Process exit + DuckDB validation fails | Retry WriterJob from scratch (new cycle_id) | Full re-execution, InferenceJob skipped for this cycle |
| WriterJob hangs >60s | Timeout at t=-4m:00s | Kill process, retry with fresh cycle | Aggressive restart, repeat until success |
| WriterJob completes, writes partial/corrupted | Atomic write validation fails (missing marker, checksum mismatch, job_id regression) | Treat as failure, retry WriterJob | Do not run InferenceJob |
| InferenceJob crashes mid-execution | Process exit code check | Treat as inference failure, serve stale cache | Use 1-hour-old cache if available, else notify |
| InferenceJob hangs >90s | Timeout at t=1m:30s | Kill process, don't retry; serve stale | Preserve availability: use old cache |
| Cache write partial/corrupted | Commit marker missing or checksum fails | Discard entry, recompute if needed | Safe degradation: no partial cache entries |
| Takshashila subprocess crashes repeatedly | Retry count exceeds limit (TBD, likely 2-3) | Timeout remaining until next cycle, close coordinator | Stop API server, notify owner |

---

## 2. WriterJob Atomic Write Protocol

### 2.1 Write Contract

**Invariant:** On-disk state is always consistent; partial writes are never visible.

**Implementation:**

```
For each output (meteo.parquet, pm25.parquet, landsat.parquet, sentinel.parquet):

1. **Generate Write Identifier**
   - cycle_id = uuid4() or deterministic hash(timestamp)
   - job_id = cycle_id
   - version = 1
   
2. **Write to Temporary Path**
   - temp_path = f"{output_dir}/.{job_id}.{output_name}.tmp"
   - Write data normally to temp path
   - Include in file metadata (as Parquet metadata or manifest file):
     {
       "job_id": "{job_id}",
       "scheduled_time": "{timestamp}",
       "version": 1,
       "row_count": {N},
       "checksum": "{sha256(data)}",
       "schema_version": 1,
       "completion_marker": "READY"  ← Last thing written
     }
   
3. **Fsync and Atomic Rename**
   - fsync(temp_file) → ensure all bytes on disk
   - atomic_rename(temp_path, final_path)  ← Linux rename is atomic
   
4. **Write Completion State to DuckDB**
   - INSERT INTO cycle_states (
       cycle_id, scheduled_time, output_name, 
       status, checksum, row_count, job_id, version, 
       written_at, completion_marker
     )
     VALUES (..., 'complete', ...)
   - COMMIT (implicit transaction boundary)
```

### 2.2 Read & Validation Contract

**On Read:**
```
For each file:

1. **Check file existence** → If missing, mark as partial
2. **Read metadata** (from Parquet metadata or .manifest file)
3. **Validate:**
   - completion_marker == "READY" → ✓
   - checksum(file_data) == metadata.checksum → ✓
   - metadata.row_count > 0 → ✓ (non-empty)
   - metadata.schema_version matches expected → ✓
   - metadata.job_id same as expected or newer → ✓
   - If **any check fails:** treat file as corrupted, discard

4. **Query DuckDB:**
   - SELECT * FROM cycle_states 
     WHERE cycle_id = ? AND output_name = ? 
     AND status = 'complete'
   - If no row or status != 'complete', treat as partial
   
5. **Decision:**
   - All 5 outputs valid + metadata consistent → WriterJob VALID
   - Any file invalid or metadata missing → WriterJob INVALID
```

### 2.3 Retry Strategy for WriterJob Failure

```
if WriterJob validation fails:
    retry_count += 1
    if retry_count > 3:
        log("WriterJob persistent failure after 3 retries, skipping to next cycle")
        skip_inference = True
    else:
        new_cycle_id = generate_new_id()
        spawn WriterJob with new_cycle_id
        wait for completion + validation
```

---

## 3. InferenceJob Integrity Gate

### 3.1 Pre-Execution Gate

**Before spawning InferenceJob:**

```
gate_result = validate_writer_outputs()  // Check 5 files + metadata

if gate_result.status != "VALID":
    log(f"WriterJob outputs not atomically valid: {gate_result.errors}")
    log("Retrying WriterJob instead of running InferenceJob")
    retry_writer()  → go back to §2.3
else:
    log("WriterJob outputs valid, proceeding to InferenceJob")
    spawn InferenceJob
```

**This gate is non-negotiable:** InferenceJob never runs on partially-written or corrupted state.

### 3.2 InferenceJob Execution

```
InferenceJob lifecycle:

t=0m:00s → Start
- Load WriterJob outputs (5 parquets, validated above)
- Assemble features (vectorized, 35-45s)
- Run LightGBM inference (5-10s)
- Write predictions to DuckDB

t≈0m:55s → Completion of hot cache (5K grids, synchronous, blocking)
         → /predict now serves cache hits <1ms

t≈1m:15s → Completion of warm cache (45K grids, async)
t≈1m:45s → Completion of cold cache (450K grids, async)
```

---

## 4. Cache Atomic Write Protocol

### 4.1 Cache Entry Structure

```
Table: cache_entries

Columns:
  grid_id: string (primary key component)
  cycle_id: string (primary key component)
  pm25: float
  checksum: string (sha256 of pm25 value + metadata)
  job_id: string
  version: int
  completion_marker: string ("READY" or empty)
  created_at: timestamp
  last_updated: timestamp
```

### 4.2 Write Protocol (In-Memory Cache + Table)

```
For each prediction (grid_id, pm25):

1. **Prepare entry**
   - cycle_id = current cycle identifier
   - job_id = InferenceJob identifier (same as cycle_id)
   - checksum = sha256(f"{cycle_id}:{grid_id}:{pm25}:{version}")
   - completion_marker = "" (initially empty)

2. **Write to temporary location (in-memory buffer)**
   - cache_entries_buffer[grid_id] = {pm25, checksum, job_id, version, completion_marker=""}

3. **Fsync to persistent store (batched)**
   - Collect all entries from buffer
   - Write all as single transaction to DuckDB predictions table
   - Include checksum + completion_marker in each row

4. **Atomic commit marker**
   - Once all entries written, UPDATE cache_entries SET completion_marker='READY' WHERE cycle_id=? AND job_id=?
   - COMMIT transaction

5. **In-memory cache population (after fsync)**
   - Move entries from buffer to Redis (hot cache)
   - Async populate warm/cold caches (non-blocking to cycle)
```

### 4.3 Read & Validation (Cache Hit)

```
On /predict request (grid_id, timestamp):

1. **Try in-memory cache (Redis, if available)**
   cache_entry = redis.get(f"cache:{grid_id}")
   if cache_entry exists:
       // Fast path: <1ms
       return cache_entry.pm25

2. **Cache miss: try DuckDB table**
   SELECT pm25, checksum, job_id, version, completion_marker 
   FROM cache_entries 
   WHERE grid_id = ? AND cycle_id = latest_valid_cycle
   
3. **Validate entry**
   - completion_marker == "READY" → ✓
   - checksum(pm25 || metadata) matches stored checksum → ✓
   - version is monotonic (no regression) → ✓
   - job_id is from a complete, valid cycle → ✓
   
4. **Age check**
   created_at >= (now - 1 hour) → ✓ (fresh enough)
   
5. **Decision**
   - All checks pass → return pm25
   - Any check fails → treat entry as nonexistent, fall back

6. **Fallback: stale cache**
   - If created_at < (now - 1 hour):
       if cycle_id from previous cycle exists and valid:
           return stale pm25 (note: up to 1 hour old, within tolerance)
       else:
           return 503 Service Unavailable
   - If created_at much older (t > 120m):
       return 503 + notify owner
```

---

## 5. Cycle State Persistence (DuckDB)

### 5.1 Schema

```sql
CREATE TABLE cycle_states (
  cycle_id TEXT PRIMARY KEY,
  scheduled_time TIMESTAMP NOT NULL,
  status TEXT NOT NULL,  -- 'success', 'failed', 'in_progress'
  last_success_ts TIMESTAMP,
  failure_reason TEXT,
  retry_count INT DEFAULT 0,
  writer_job_status TEXT,  -- 'pending', 'running', 'complete', 'failed'
  inference_job_status TEXT,  -- 'pending', 'running', 'complete', 'failed'
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_cycle_scheduled ON cycle_states(scheduled_time DESC);
CREATE INDEX idx_cycle_status ON cycle_states(status);
```

### 5.2 State Machine

```
START OF CYCLE:
  cycle_id ← generate()
  scheduled_time ← current schedule time
  INSERT INTO cycle_states (cycle_id, scheduled_time, status) 
    VALUES (?, ?, 'in_progress')

DURING WriterJob:
  UPDATE cycle_states SET writer_job_status = 'running' WHERE cycle_id = ?

ON WriterJob COMPLETION:
  if validation passes:
      UPDATE cycle_states SET writer_job_status = 'complete' WHERE cycle_id = ?
  else:
      UPDATE cycle_states SET writer_job_status = 'failed', 
                              failure_reason = '...' WHERE cycle_id = ?
      UPDATE cycle_states SET retry_count = retry_count + 1 WHERE cycle_id = ?

DECISION GATE:
  if writer_job_status == 'failed' AND retry_count < 3:
      new_cycle_id ← generate()
      goto START OF CYCLE (repeat WriterJob)
  else if writer_job_status == 'failed' AND retry_count >= 3:
      skip inference, mark cycle as 'failed'
  else:
      proceed to InferenceJob

DURING InferenceJob:
  UPDATE cycle_states SET inference_job_status = 'running' WHERE cycle_id = ?

ON InferenceJob COMPLETION:
  if validation passes:
      UPDATE cycle_states SET inference_job_status = 'complete',
                              status = 'success',
                              last_success_ts = now() WHERE cycle_id = ?
  else:
      UPDATE cycle_states SET inference_job_status = 'failed',
                              failure_reason = '...',
                              status = 'failed' WHERE cycle_id = ?
      // Do NOT retry inference; serve stale cache
      
CYCLE END:
  Query last successful cycle: 
    SELECT * FROM cycle_states WHERE status = 'success' 
                                  ORDER BY scheduled_time DESC LIMIT 1
  Use last_success_ts to determine stale cache tolerance
```

### 5.3 Restart Determinism

```
On coordinator restart:
  SELECT MAX(scheduled_time) as last_time FROM cycle_states WHERE status = 'success'
  current_time = now()
  
  if (current_time - last_time) < 1 hour:
      // Last successful cycle is fresh
      can_serve_from_cache = True
  else:
      // Last successful cycle is stale
      can_serve_from_cache = False (but still allow, notify operator)
  
  Next cycle = align to next scheduled time (e.g., top of next hour)
  Never attempt to resume partial cycle from cycle_states; always start fresh
```

---

## 6. Timeout and Restart Strategy

### 6.1 WriterJob Timeout

```
spawn WriterJob at t = -5m:00s
timeout = 60 seconds (must complete by t = -4m:00s)

if not completed by timeout:
    kill process
    log("WriterJob timeout, restarting")
    retry_count += 1
    if retry_count <= 3:
        sleep(0)  // Aggressive: no backoff
        spawn WriterJob again
    else:
        log("WriterJob persistent timeout after 3 retries, skipping to next cycle")
        mark cycle as 'failed', advance to next scheduled cycle
```

### 6.2 InferenceJob Timeout

```
spawn InferenceJob at t = 0m:00s (only if WriterJob validated)
timeout = 90 seconds (must complete by t = 1m:30s)

if not completed by timeout:
    kill process
    log("InferenceJob timeout, serving stale cache")
    // Do NOT retry; let /predict fallback to stale cache
    // Proceed to next cycle on schedule
    // Notify if cache is >1 hour old
```

### 6.3 Takshashila Startup Timeout

```
spawn Takshashila subprocess at t = -5m:00s (just before WriterJob)
timeout = check every 10 seconds until t = 0m:00s

if Takshashila not ready (health check fails) by t = 0m:00s:
    close coordinator
    log("FATAL: Takshashila failed to start before next scheduled job, coordinator shutting down")
    notify owner
    return 503 on /predict requests
    
// Never spin retrying indefinitely; hard deadline per schedule
```

### 6.4 Aggressive Restart Philosophy

```
Goal: Maximize availability within hard constraints

WriterJob failures: Restart immediately without backoff (seconds matter)
InferenceJob failures: Don't restart; serve stale cache (inference is not required)
Takshashila crashes: Restart once per schedule window, then fail hard
/predict timeouts: Serve cache or 503

No exponential backoff; no cascading retries; time is the circuit breaker
```

---

## 7. /predict Server Invariants

### 7.1 Service Contract

```
INVARIANT 1: /predict is never computed on-demand
  ✗ FORBIDDEN: if cache_miss, compute_inference()
  ✓ ALLOWED: if cache_miss, return stale or 503

INVARIANT 2: /predict is always served from cache or table
  - Hot cache (in-memory, <1ms)
  - Warm/cold cache (DuckDB table, <100ms typically)
  - Stale cache (DuckDB older cycle, <1 hour old allowed)
  
INVARIANT 3: Cache and table entries are atomic
  - No partial entries visible to /predict
  - Corrupt entries are silently discarded
  - Recompute only if all validation fails

INVARIANT 4: /predict never blocks on background tasks
  - Warm/cold cache population is async, non-blocking
  - Hot cache (5K) is synchronous, blocking InferenceJob
  - /predict does not wait for async cache
```

### 7.2 Response Codes

```
200 OK
  ✓ Found prediction in hot cache or warm/cold cache
  ✓ Found valid prediction in previous cycle (≤1 hour old)
  Body: { "grid_id": "...", "pm25": 42.5, "age": "0m", "source": "cache" }

206 Stale Content
  ✓ Found prediction from previous cycle (>1 hour old)
  ✓ Current cycle failed or incomplete
  Body: { "grid_id": "...", "pm25": 42.5, "age": "65m", "source": "stale_cache", "warning": "prediction is 65 minutes old" }

503 Service Unavailable
  ✗ No valid prediction available (no cache, all cycles failed)
  ✗ Takshashila startup timeout exceeded
  Body: { "error": "No valid predictions available", "last_success": "2026-01-24T14:35:00Z" }
```

---

## 8. Implementation Sequence

### **Phase 2a: Atomic Write Protocols** (Prerequisite)
- [ ] Implement WriterJob atomic write (temp → fsync → rename)
- [ ] Implement WriterJob validation (metadata + checksum)
- [ ] Implement cache atomic write (buffer → fsync → rename)
- [ ] Implement cache validation (checksum + completion marker)
- **Effort:** 2 days
- **Blocking:** Everything else

### **Phase 2b: Cycle State Persistence** (Prerequisite)
- [ ] Create `cycle_states` table in DuckDB
- [ ] Implement state machine (in_progress → success/failed)
- [ ] Implement cycle_id generation + tracking
- [ ] Add restart determinism logic
- [ ] Add query helpers (get_last_success, get_cycle_status, etc.)
- **Effort:** 1 day
- **Blocking:** Orchestrator, retry logic

### **Phase 2c: InferenceJob Integrity Gate**
- [ ] Implement WriterJob output validator
- [ ] Add integrity gate before InferenceJob spawn
- [ ] Add WriterJob retry logic on gate failure
- [ ] Implement failure reason capture
- **Effort:** 1 day
- **Dependencies:** Phase 2a, 2b
- **Blocking:** InferenceJob execution

### **Phase 2d: Orchestrator with Timeouts**
- [ ] Update Orchestrator to use cycle_states
- [ ] Add WriterJob timeout (60s, aggressive restart)
- [ ] Add InferenceJob timeout (90s, fail soft)
- [ ] Add Takshashila startup health check timeout
- [ ] Add process exit code checking
- [ ] Integrate retry logic from Phase 2b
- **Effort:** 2 days
- **Dependencies:** Phase 2b
- **Blocking:** Fault tolerance execution

### **Phase 2e: Cache Fallback & Age Validation**
- [ ] Update /predict to check cache age
- [ ] Implement stale cache fallback (1 hour tolerance)
- [ ] Implement 503 response for expired cache (>120m)
- [ ] Add operator notification on age threshold
- [ ] Add response header with cache age
- **Effort:** 1 day
- **Dependencies:** Phase 2a, 2b
- **Blocking:** Production readiness

### **Phase 2f: Crash Recovery & Observability**
- [ ] Add structured logging with cycle_id
- [ ] Add Prometheus metrics (cycle status, job durations, cache age)
- [ ] Add health check endpoint (reports last_success_ts, cache status)
- [ ] Add coordinator shutdown logic (Takshashila startup timeout)
- [ ] Add owner notification mechanism (email, Slack, etc.)
- **Effort:** 2 days
- **Dependencies:** Phase 2a-2e
- **Blocking:** Production visibility

---

## 9. Critical Implementation Notes

### **9.1 Start Time: t = -5m (Not -2m)**

The original 2-minute offset was based on WriterJob duration (20-40s).  
New 5-minute offset provides:
- 300 seconds for WriterJob (provides 5× buffer for retries)
- 60 seconds timeout + 2 retry attempts = 180 seconds worst case
- ~60 seconds slack before InferenceJob at t=0m

```
t = -5m:00s → WriterJob START (spawn + retry window)
t = -4m:00s → WriterJob TIMEOUT threshold (fail hard if not done)
t = 0m:00s  → InferenceJob START (only if WriterJob validated)
t = 1m:30s  → InferenceJob TIMEOUT threshold
t = 2m:00s  → Cycle complete, hot cache ready by now (usually ~55s)
```

### **9.2 No Mid-Cycle Resume**

State persisted in DuckDB is **for decision-making only**, never for resumption.

```
FORBIDDEN:
  cycle_states says "WriterJob crashed at 40%, resume from API call 3"
  
ALLOWED:
  cycle_states says "WriterJob failed, retry_count=1"
  → Spawn fresh WriterJob with new cycle_id, start from beginning
```

### **9.3 Satellite Value Freshness**

Landsat (16-day revisit) and Sentinel (4-5 day revisit) valid for their revisit period.  
**No per-hour staleness clock on satellite data** within a 24-hour cycle; value received at t=0 is valid through t=16d (Landsat) or next revisit (Sentinel).

Staleness applies only to the PM25 inference value, not the input satellite data.

### **9.4 WriterJob Re-run Complete Philosophy**

Always re-run from scratch, never attempt incremental updates.

```
Example: Sentinel-2 fetch partially succeeds (5/8 tiles)
  ✗ Write 5 tiles and mark "partial"
  ✓ Detect partial write via atomic validation
  → Treat entire WriterJob as failed
  → Retry WriterJob (will re-attempt all fetches)
```

This keeps logic simple and avoids complex merge/conflict resolution.

### **9.5 DuckDB as Single Source of Truth**

Use DuckDB for:
- Cycle state persistence (metadata only)
- Prediction table (backup to in-memory cache)
- Checksum + metadata validation
- Transactional commit markers

Rationale:
- Already exists, no new dependency
- Transactional semantics for atomic commits
- Survives crashes, queryable state machine
- Avoids Redis as availability coupling point

### **9.6 Coordinator Shutdown (Hard Stop)**

If Takshashila doesn't start before next scheduled job (t=0m):
```
close coordinator
return 503 on all /predict requests
notify owner immediately
```

No retry loop, no "wait for next cycle", no exponential backoff.  
Time is the circuit breaker.

---

## 10. Failure Scenario Examples

### **Scenario 1: WriterJob Crashes After 30s**

```
t = -5m:00s  → WriterJob spawned
t = -4m:30s  → Process crashes (PID killed, etc.)
t = -4m:30s  → Orchestrator detects exit code != 0
               → cycle_states: writer_job_status = 'failed'
               → retry_count += 1 (now = 1)
t = -4m:30s  → Aggressive restart (no backoff)
t = -4m:35s  → WriterJob running again (fresh execution)
t = -4m:05s  → WriterJob completes successfully
               → Validation passes: all 5 files, checksums, markers ✓
               → cycle_states: writer_job_status = 'complete'
t = 0m:00s   → InferenceJob spawned (gate passed)
t = 1m:15s   → Full cycle completes
```

### **Scenario 2: InferenceJob Hangs >90s**

```
t = 0m:00s   → InferenceJob spawned (WriterJob valid)
t = 0m:45s   → Feature assembly stalled (e.g., disk I/O contention)
t = 1m:30s   → Timeout threshold hit
               → Kill InferenceJob process
               → cycle_states: inference_job_status = 'failed'
               → Reason: "timeout after 90s"
t = 1m:35s   → /predict request arrives
               → Cache hit from previous cycle (50 minutes old)
               → Return 200 OK with age=50m
t = 2m:00s   → Next scheduled cycle starts
               → Fresh WriterJob + InferenceJob attempt
```

### **Scenario 3: WriterJob Writes Partial Data**

```
t = -5m:00s  → WriterJob starts writing
t = -4m:30s  → sentinel.parquet written successfully
               → landsat.parquet written successfully
               → meteo.parquet **partially written, process crashes**
t = -4m:30s  → Atomic rename never happens for meteo.parquet
               → Old meteo.parquet still on disk (or missing)
t = -4m:30s  → Orchestrator detects crash
               → Tries validation of WriterJob outputs
               → meteo.parquet: missing completion_marker → INVALID
               → cycle_states: writer_job_status = 'failed'
t = -4m:30s  → Retry WriterJob (fresh, all 5 sources)
t = -4m:05s  → All 5 files now valid, cycle proceeds
```

### **Scenario 4: Takshashila Process Crashes Repeatedly**

```
t = -5m:00s  → Takshashila spawned, starts initializing
t = -4m:50s  → Takshashila crashes (out of memory, segfault, etc.)
t = -4m:50s  → Coordinator detects crash, restarts aggressively
t = -4m:45s  → Takshashila crashes again
t = -4m:40s  → Coordinator detects, restarts again
... (repeats)
t = 0m:00s   → Health check: Takshashila still not ready
t = 0m:00s   → HARD STOP
               → Close coordinator
               → All /predict return 503
               → Notify owner
               → Wait for manual intervention
```

---

## 11. Success Criteria

- [x] WriterJob atomic writes with checksum + completion marker
- [x] WriterJob validation before InferenceJob
- [x] InferenceJob atomic cache writes with same protocol
- [x] Cycle state persistence in DuckDB (cycle_id, status, retry_count)
- [x] 60-second WriterJob timeout with aggressive restart (max 3 attempts)
- [x] 90-second InferenceJob timeout with fail-soft (serve stale)
- [x] /predict never computes on-demand; always cache or 503
- [x] /predict serves stale cache if ≤1 hour old
- [x] /predict returns 503 if cache >2 hours old
- [x] Takshashila startup health check with hard deadline at t=0m
- [x] Coordinator shutdown if Takshashila doesn't start by next scheduled job
- [x] Operator notification on coordinator shutdown
- [x] All failures logged with cycle_id for traceability
- [x] No mid-cycle resume; all failures trigger fresh re-execution

---

## 12. Open Implementation Questions (None)

All clarifying questions answered. Architecture is finalized.

Next: Begin Phase 2a implementation (atomic write protocols).
