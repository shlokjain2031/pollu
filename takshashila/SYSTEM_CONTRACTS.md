# Takshashila System Contracts & Invariants

**Phase 0 — System Design**  
**Date**: January 19, 2026  
**Status**: Locked Design  
**Audience**: All engineers, architects

---

## Executive Summary

This document defines the **invariants, contracts, and failure semantics** that every component in the Takshashila inference system must obey. No implementation details, no optimization tricks—only the rules that make the system correct, observable, and debuggable.

This is the **source of truth** for all subsequent phases (1–10).

---

## 1. Unit of Truth (Hard Invariant)

The atomic, independently valid unit in the system is:

```
(grid_id, timestamp_hour) → pm25_prediction_value
```

**Properties**:
- `grid_id`: Uniquely identifies a spatial grid cell (e.g., "M0001")
- `timestamp_hour`: Wall-clock time of prediction, floored to hour boundary
- `pm25_prediction_value`: Float value (in ppm); result of model inference

**Atomicity**: 
- Each key is independently queryable and valid
- No dependencies between cells or timestamps
- One cell's computation does not require another's

**Idempotence**:
- Once a value is successfully computed for a key, it must **never change** unless explicitly overwritten by the scheduler (§7)
- Recomputation is allowed **only** under freshness rules (§4)

**Invariant**: If two requests ask for identical `(grid_id, timestamp_hour)` with identical input features, the inference engine must return identical `pm25_prediction_value`.

---

## 2. Time Semantics

### 2.1 Timestamp Definition

`timestamp_hour` = wall-clock time rounded/floored to the nearest hour

**Example**:
- Request at 2026-01-19 **16:47:30** → use `2026-01-19 16:00:00`
- Request at 2026-01-19 **16:00:00** → use `2026-01-19 16:00:00`

### 2.2 Determinism Contract

Two requests with:
- Identical `grid_id`
- Identical `timestamp_hour`
- Identical input feature values

Must produce identical `pm25_prediction_value`.

**This is enforced by**:
- Deterministic model (LightGBM with fixed seed)
- Idempotent feature assembly
- No random jitter or external randomness in prediction path

### 2.3 When Recomputation is Allowed

Recomputation is allowed **only** if:

1. No value exists for `(grid_id, timestamp_hour)`, OR
2. The existing value is classified as **stale** under freshness rules (§4)

**Recomputation is forbidden if**:
- Value exists AND is **fresh**

---

## 3. Authority Chain (Strict)

Ownership and write authority are **unambiguous**:

### 3.1 Python Inference Service

**Owns**:
- Computation of predictions
- Writing results to DuckDB
- Single-flight deduplication (§6)
- Error handling during inference

**Authority**: Only Python writes to `predictions` table

**Constraint**: Python is not aware of serving layer; inference is **pure** (no HTTP, no logging to external services)

### 3.2 DuckDB (Durable Storage)

**Owns**:
- Source of truth for all predictions
- Durability across restarts
- Concurrency control (read/write isolation)

**Authority**: All reads of predictions must go through DuckDB (not cache)

**Constraint**: DuckDB is **read-only for Go**; writes happen only from Python

### 3.3 Go Serving Layer

**Owns**:
- Request routing
- Timeout enforcement
- Priority handling (point API vs heatmap)
- Deduplication of client requests (future phases)

**Authority**: None with respect to predictions; Go is **read-only**

**Constraint**: Go **never writes** predictions directly; always goes through Python

### 3.4 Caches (L1, L2, etc.)

**Status**: **Advisory only**

- Caches are optimizations
- Cache miss **must never break correctness**
- Cache hit may be stale; cache miss always checks DuckDB
- DuckDB is source of truth; caches are never authoritative

---

## 4. Freshness State Machine (Formal)

Each `(grid_id, timestamp_hour)` exists in exactly **one** state at any time:

### 4.1 State Definitions

#### **fresh**

**Conditions**:
- Prediction exists in DuckDB
- All input features are within freshness bounds (see §4.2)

**Serving action**: Serve immediately; no recompute triggered

**Cache behavior**: Can be served from cache without staleness check

#### **stale-but-acceptable**

**Conditions**:
- Prediction exists in DuckDB
- One or more input features exceed freshness bound but remain within tolerance

**Serving action**: Serve immediately with warning; recompute may be scheduled asynchronously (future scheduler)

**Cache behavior**: Serve with metadata indicating partial staleness

#### **stale-and-recompute**

**Conditions**:
- Prediction exists in DuckDB
- One or more input features exceed tolerance threshold

**Serving action**: Must trigger recompute; request may block or return error + retry hint

**Cache behavior**: Do not serve; force recompute

#### **missing**

**Conditions**:
- No prediction exists for `(grid_id, timestamp_hour)` in DuckDB

**Serving action**: Must trigger compute; always blocks until result or timeout

**Cache behavior**: N/A (no cached value)

### 4.2 Feature-Specific Freshness Thresholds

| Feature Group | Tolerance | Exceeds At | Action on Exceed |
|---|---|---|---|
| PM2.5 prediction | 1 hour | >1h old | stale-and-recompute |
| Meteorological (temp, humidity, etc.) | 1 hour | >1h old | stale-and-recompute |
| Sentinel-2 imagery | 5 days | >5d old | stale-but-acceptable |
| Landsat imagery | 16 days | >16d old | stale-but-acceptable |

**Rule**: If any input feature exceeds its threshold, the prediction is **stale-and-recompute**.

---

## 5. Failure Semantics (Fail Loudly)

Failures are **never silent**. Every error is explicit.

### 5.1 Inference Failure

**Trigger**: Model.predict() raises exception, feature assembly fails, timeout occurs, etc.

**Behavior**:
- Return explicit error code (503, 400, etc.)
- **Do not** return stale value as fallback
- **Do not** retry automatically within the request path
- **Do not** silently degrade to lower-confidence output

**Error contract**: Client receives structured error with:
- Error type (e.g., "feature_assembly_failed", "model_timeout")
- Affected grid_id and timestamp_hour
- Retry guidance (if applicable)

### 5.2 Inference Timeout

**Trigger**: Computation exceeds Go-enforced timeout (2s for point, 10s for heatmap)

**Behavior**: Treated identically to failure; return explicit error

**Guarantee**: Timeout is always **bounded and surfaced**; never infinite wait

### 5.3 DuckDB Read Failure

**Trigger**: DuckDB query fails, connection lost, disk error, etc.

**Behavior**:
- Return explicit error
- Request fails immediately; no retry or fallback
- Error is logged with full context

### 5.4 DuckDB Write Failure

**Trigger**: Write attempt fails (disk full, constraint violation, lock timeout, etc.)

**Behavior**:
- Result is **not considered committed**
- Request returns failure
- Prediction is **not** available for subsequent reads
- Retry is safe (idempotent)

### 5.5 Invalid Grid or Missing Feature

**Trigger**: grid_id does not exist in schema; required feature cannot be computed

**Behavior**:
- Return explicit error (400 Bad Request if client's fault; 503 if system's fault)
- No partial result returned
- This state may be cached as a negative result (future optimization)

---

## 6. Deduplication Contract (Single-Flight)

### 6.1 What is Deduplicated

Deduplication applies to the key:

```
(grid_id, timestamp_hour)
```

**Scope**: One Python inference service process

### 6.2 Guarantee

**Invariant**: At most **one** inference computation runs concurrently per unique `(grid_id, timestamp_hour)`

**Semantic**: If Request A and Request B arrive simultaneously for the same key:
- One request (A) becomes the **leader** and computes
- Other requests (B, C, ...) become **followers** and wait
- All followers receive the same result (or error) as the leader

**No duplication**: Compute does not run twice for the same key

### 6.3 Deduplication Does Not Guarantee Exactly-Once Across Restarts

**Clarification**: If Python service crashes and restarts, a new request for the same key may recompute the value (idempotent, no data loss).

### 6.4 Implementation Pattern (Conceptual)

```python
# Pseudo-code
inflight = {}  # (grid_id, timestamp_hour) → Future

def predict(grid_id, timestamp_hour):
  key = (grid_id, timestamp_hour)
  
  if key in inflight:
    # Follower path
    return inflight[key].result()  # Wait for leader
  
  # Leader path
  future = Future()
  inflight[key] = future
  
  try:
    result = compute_inference(grid_id, timestamp_hour)
    future.set_result(result)
    return result
  except Exception as e:
    future.set_exception(e)
    raise
  finally:
    del inflight[key]  # Cleanup
```

---

## 7. Scheduler Contract (Defined Now, Implemented Later)

Scheduler behavior is **constrained and predictable** even before implementation.

### 7.1 Scheduler is a Producer, Not Privileged

**Invariant**: Scheduler is **not a special component**

- Scheduler jobs use the same inference code path as on-demand requests
- Scheduler jobs are deduplicated like any other request
- Scheduler has no privileged write access

### 7.2 Scheduler Recompute Rules

**Scheduler may recompute** (write to DuckDB) **only if**:
- Target prediction is missing, OR
- Target prediction is **stale** (matches §4 stale-and-recompute criteria)

**Scheduler must not recompute** if:
- Target prediction is **fresh**

### 7.3 Last Writer Wins (Limited Scope)

**For stale predictions only**: If scheduler and on-demand request both attempt to recompute the same key, last writer's result is committed to DuckDB.

**For fresh predictions**: Scheduler sees fresh value; does not write.

### 7.4 Future: Scheduler Enqueue Behavior

Scheduler will eventually enqueue jobs into the same job queue as on-demand requests (Phase 10). No special prioritization.

---

## 8. Partial Result Policy (Forward-Compatible)

Partial responses are **valid and expected** in future heatmap APIs, but must follow strict semantics.

### 8.1 Partial Response Definition

A **partial response** is one where:
- Some requested cells have predictions
- Some cells do not (missing or error)
- Response is returned (not rejected entirely)

### 8.2 Partial Response Requirements

If a partial response is returned:

1. **Explicit marking**: Each cell must be marked with state (fresh, stale, missing, error)
2. **Loud error metadata**: Missing/error cells must include error reason
3. **No silent success**: Partial ≠ silently acceptable

**Example (future heatmap response)**:
```json
{
  "results": [
    {"grid_id": "M0001", "pm25": 65.3, "state": "fresh"},
    {"grid_id": "M0002", "pm25": null, "state": "error", "reason": "feature_assembly_failed"},
    {"grid_id": "M0003", "pm25": 70.0, "state": "stale-but-acceptable"}
  ]
}
```

### 8.3 Not for Point API

Point API (single cell) must return **either complete success or complete failure**, no partial.

---

## 9. Timeout & Bounds (Hard Limits)

### 9.1 All Waits are Bounded

**Invariant**: Every blocking operation has a timeout.

**Examples**:
- Inference computation: bounded by Go-enforced timeout (2s for point, 10s for heatmap)
- DuckDB query: bounded (e.g., 1s)
- Lock acquisition: bounded (e.g., 2s)
- Follower wait on Future: bounded (inherit Go timeout)

### 9.2 Timeout Properties

**Timeout must be**:
- Long enough for correct inference in steady state (worst case: 200ms)
- Short enough that failure is distinguishable from waiting (not >30s)
- Explicitly configured, not magical

### 9.3 Timeout Surfacing

Timeouts **always surface as errors**, never as:
- Silent fallback to stale data
- Retry loop (in request path)
- Infinite wait

---

## 10. Explicit Non-Goals (Phase 0)

The following are **intentionally excluded** from Phase 0 design:

- **Heatmap APIs**: Will be added in Phase X; not in scope now
- **Distributed / multi-node coordination**: Single-machine only
- **Redis or external caches**: Not in scope for MVP
- **Strong consistency across multiple cells**: Per-cell consistency only
- **Latency SLAs**: No targets yet; focus on correctness
- **Auto-scaling**: Single Python process, fixed thread pool
- **Geo-replication**: Not applicable

---

## 11. Phase 0 Deliverable Checklist

This document is the **source of truth** for all later phases.

Every subsequent implementation must:

- [ ] Obey the unit of truth (§1)
- [ ] Respect time semantics (§2)
- [ ] Follow the authority chain (§3)
- [ ] Implement the freshness state machine (§4)
- [ ] Handle failures loudly (§5)
- [ ] Enforce single-flight deduplication (§6)
- [ ] Respect scheduler constraints (§7, even if not yet implemented)
- [ ] Support partial responses (§8, even if not yet used)
- [ ] Enforce bounded timeouts (§9)
- [ ] Stay out of non-goals (§10)

---

## 12. Phase 0 Contract Sign-Off

**Contract Version**: 1.0  
**Date Locked**: January 19, 2026  
**Status**: Source of Truth

This document is **invariant** across all phases. Changes require explicit re-lock.

**Not in scope for Phase 0**:
- Implementation
- Code examples
- Performance targets
- Operational procedures

**Scope for Phase 0**: Rules, invariants, failure semantics, authority, freshness states.

---

**End of SYSTEM_CONTRACTS.md**
