# Backwards Compatibility & Stability Policy
# Items 40-41: Treat metric names and labels as stable API

## Core Principle

Prometheus metric names and labels are part of our stable API. **Breaking changes will cause alerts to fire incorrectly, SLO burn rates to become invalid, and client dashboards to break.**

## Stability Guarantees

### ✅ Safe Operations (Backwards Compatible)

1. **Adding new metrics**
   - Example: Add `worker_pool_utilization_percent` alongside existing metrics
   - Safe: Existing queries unaffected
   - Action: Document the new metric

2. **Adding new label values to existing CounterVecs/GaugeVecs**
   - Example: Add new error_type "RateLimitExceeded" to `errors_total{error_type}`
   - Safe: Existing label values unchanged
   - Action: Update METRICS.md with new enum value
   - Note: Only safe if cardinality remains controlled

3. **Adding new optional scrape targets**
   - Safe: Existing targets still work
   - Action: Update prometheus.yaml

4. **Changing metric documentation**
   - Safe: Doesn't affect queries
   - Action: Update METRICS.md

### ❌ Breaking Changes (FORBIDDEN after release)

1. **Renaming existing metrics**
   - ❌ FORBIDDEN: Renames existing metric, breaks PromQL queries
   - Example: Renaming `request_latency_seconds` → `request_duration_seconds`
   - Impact: All dashboards, alerts, SLO rules fail
   - **Alternative**: Deprecate old metric, add new metric, run both in parallel for transition period

2. **Changing metric type**
   - ❌ FORBIDDEN: Counter → Gauge breaks rate calculations
   - Example: Changing `requests_total` (Counter) → `requests_current` (Gauge)
   - Impact: All `rate()` and `increase()` queries fail
   - **Alternative**: Create new metric with correct type, deprecate old

3. **Removing metric labels**
   - ❌ FORBIDDEN: Breaks queries using that label
   - Example: Removing `path` from `requests_total{path,status}`
   - Impact: PromQL queries referencing label fail
   - **Alternative**: Keep label, add new metric without label if needed

4. **Adding required labels to existing metrics**
   - ❌ FORBIDDEN (Item 41): Breaking change
   - Example: Adding `version` label to `requests_total{path,status,version}`
   - Impact: Existing alerts/SLOs expecting only path/status labels break
   - **Alternative**: Create new metric variant, keep old metric unchanged

5. **Removing cardinality controls**
   - ❌ FORBIDDEN: Per-request dynamic labels break cardinality guarantees
   - Example: Adding request_id label to any metric
   - Impact: Unbounded cardinality → Prometheus memory explosion
   - **Alternative**: Keep cardinality constraints, add structured logs for request tracing

6. **Changing metric bucket boundaries**
   - ❌ FORBIDDEN (Item 40): Breaks histogram semantics
   - Example: Changing histogram buckets from [0.005, 0.01, ...] to [0.01, 0.02, ...]
   - Impact: p50/p95/p99 calculations invalid
   - **Alternative**: Create new histogram metric, deprecate old in parallel

## Migration Strategy for Required Breaking Changes

If a breaking change is absolutely necessary (e.g., critical architectural shift):

1. **Parallel Metrics Period (1-2 weeks)**
   - Emit both old and new metric names simultaneously
   - Allow clients time to migrate queries/dashboards
   - Example: Emit both `request_latency_seconds` and `request_duration_seconds`

2. **Deprecation Period (2-4 weeks)**
   - Mark old metric as deprecated in METRICS.md
   - Document migration path in CHANGELOG
   - Continue emitting old metric
   - Send deprecation notices to ops team

3. **Removal Period**
   - Set end-of-life date (at least 4 weeks after initial deprecation)
   - Remove old metric
   - Update all documentation

4. **Communication**
   - Email all teams using metrics
   - Update runbooks and dashboards
   - Hold SRE review meeting

## Versioning Policy

**Item 40: No versioning prefixes**

❌ DO NOT use metric versioning:
```
request_latency_seconds_v2  ← WRONG
request_duration_v2_seconds ← WRONG
takshashila:request_latency_v2:seconds ← WRONG
```

✅ USE direct metric names:
```
request_latency_seconds  ← CORRECT
queue_depth              ← CORRECT
worker_job_duration_seconds ← CORRECT
```

**Rationale**: Version prefixes in metric names are:
- Confusing (which version is "current"?)
- Difficult to maintain (versioning spread across code)
- Not idiomatic Prometheus

**Instead**: Use metric replacement strategy above.

## Label Stability (Item 41)

### Current Stable Labels (Locked)

These labels are part of our public API and **cannot be changed**:

| Metric | Labels | Status |
|--------|--------|--------|
| `requests_total` | path (cache\|dedup\|compute), status (success\|error) | **LOCKED** |
| `errors_total` | error_type (6 values) | **LOCKED** |
| `cache_hits_total` | cache_result (hit) | **LOCKED** |
| `cache_misses_total` | cache_result (miss) | **LOCKED** |
| `cache_evictions_total` | reason (ttl_expired, manual_invalidate) | **LOCKED** |
| `dedup_wait_latency_seconds` | (none) | **LOCKED** |
| `leader_compute_latency_seconds` | (none) | **LOCKED** |
| `cache_hit_latency_seconds` | (none) | **LOCKED** |
| `request_latency_seconds` | (none) | **LOCKED** |
| `queue_wait_latency_seconds` | (none) | **LOCKED** |
| `worker_job_duration_seconds` | (none) | **LOCKED** |

### Adding New Label Values (Safe)

New enum values can be added to existing labeled metrics:

**Example: Adding new error type**
```
// BEFORE
errors_total{error_type="InvalidRequest"} 42
errors_total{error_type="GridNotFound"} 15
errors_total{error_type="Unknown"} 3

// AFTER (ADD value, don't remove)
errors_total{error_type="InvalidRequest"} 42
errors_total{error_type="GridNotFound"} 15
errors_total{error_type="Unknown"} 3
errors_total{error_type="NewErrorType"} 7  ← SAFE: New value

// DON'T DO THIS:
errors_total{error_type="InvalidRequest"} 42  ← REMOVED: BREAKING
// errors_total{error_type="GridNotFound"}   ← REMOVED: BREAKING
```

### Renaming Label Values (FORBIDDEN)

❌ DO NOT rename label values:
```
// BEFORE
requests_total{path="cache"} 100

// AFTER
requests_total{path="cache_lookup"} 100  ← BREAKING: path value changed
```

**Impact**: Queries filtering by `path="cache"` silently start returning zero.

## Documentation of Changes

All metric changes must be documented in:

1. **METRICS.md** - Update metric entry
2. **CHANGELOG.md** - Record version and date
3. **Release Notes** - Summarize for operators
4. **Runbooks** - Update SLO queries if affected

Example entry:
```markdown
## Version 1.1.0 (2026-02-15)

### New Metrics
- `worker_pool_utilization_percent` (gauge) - Current worker pool utilization

### Deprecated Metrics
- `worker_compute_latency_seconds` (renamed to `leader_compute_latency_seconds`)
  - Migration: Update PromQL queries from `worker_compute_latency_seconds` to `leader_compute_latency_seconds`
  - End of life: 2026-03-15

### Changed Labels
- `errors_total`: Added new error_type value `RateLimitExceeded`

### Unchanged (Stable)
- All existing metric names
- All existing label keys (path, status, error_type, etc.)
```

## Testing Backwards Compatibility

**Item 38: Use real Prometheus registry in tests**

Tests must verify:
1. Metric names match expected values
2. Label keys/values unchanged
3. Metric types stable
4. Cardinality controlled

```go
// Example test (from metrics_test.go)
func TestBackwardsCompatibility_MetricNames(t *testing.T) {
    if globalMetrics == nil {
        globalMetrics = NewMetricsWithLabels()
    }
    
    // Verify metric names haven't changed
    expectedMetrics := []string{
        "requests_total",
        "cache_hit_latency_seconds",
        "dedup_wait_latency_seconds",
        "errors_total",
        // ... all metrics
    }
    
    // Query /metrics endpoint, verify all expected metrics present
    resp, _ := http.Get("http://localhost:8080/metrics")
    body, _ := io.ReadAll(resp.Body)
    metricsText := string(body)
    
    for _, metric := range expectedMetrics {
        if !strings.Contains(metricsText, metric) {
            t.Errorf("expected metric %s not found", metric)
        }
    }
}
```

## Historical Changes (Version Log)

### Version 1.0.0 (Current - 2026-01-22)

**Initial Release**

Stable metrics (locked):
- `requests_total{path,status}`
- `cache_hit_latency_seconds`
- `dedup_wait_latency_seconds`
- `leader_compute_latency_seconds`
- `request_latency_seconds`
- `queue_depth`
- `queue_wait_latency_seconds`
- `jobs_queued_total`
- `jobs_completed_total`
- `jobs_timed_out_total`
- `workers_active`
- `worker_job_duration_seconds`
- `worker_restarts_total`
- `worker_errors_total`
- `dedup_leaders_total`
- `dedup_followers_total`
- `dedup_active_entries`
- `dedup_cleanup_total`
- `deduped_requests_total`
- `inflight_leader_computations`
- `cache_hits_total{cache_result}`
- `cache_misses_total{cache_result}`
- `cache_evictions_total{reason}`
- `cache_entries`
- `errors_total{error_type}`

All labels and metric types locked per Items 40-41.

## Summary

**Item 40-41 Compliance**:
- ✅ Metric names treated as stable API
- ✅ No versioning prefixes
- ✅ Backwards compatibility guaranteed for v1.0.0
- ✅ Migration strategy defined for future breaking changes
- ✅ All label combinations locked until migration period
- ✅ Adding new metrics safe
- ✅ Changing existing metrics forbidden after release
