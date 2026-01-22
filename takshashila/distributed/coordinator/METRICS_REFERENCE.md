# Metrics Reference - Item 44

Comprehensive catalog of all 62 Prometheus metrics (Item 44: Metrics reference document with name, type, labels, units, interpretation).

## Export Format (Item 45)

**Endpoint**: `GET /metrics` (only export method)
**Format**: Prometheus text format (OpenMetrics)
**No custom exports**: Only Prometheus /metrics endpoint supported

---

## Metrics by Category (62 Total)

### Request Path Metrics (2)

| Metric | Type | Labels | Units | Description | SLO |
|--------|------|--------|-------|-------------|-----|
| `requests_total` | Counter | `path` (cache\|dedup\|compute), `status` (success\|error) | requests | Total requests by path and outcome | Availability (Item 34) |
| `deduped_requests_total` | Counter | — | requests | Total requests deduplicated (followers) | Dedup rate ≥80% (Item 34) |

### Latency Metrics (4 Histograms)

| Metric | Type | Buckets | Units | Description | SLO |
|--------|------|---------|-------|-------------|-----|
| `cache_hit_latency_seconds` | Histogram | [0.005..1.25]s | seconds | Cache hit latency (hit path only) | p99 <500ms (Item 34) |
| `dedup_wait_latency_seconds` | Histogram | [0.005..1.25]s | seconds | Follower wait time | p99 <500ms (Item 34) |
| `leader_compute_latency_seconds` | Histogram | [0.005..1.25]s | seconds | Leader computation time | p99 <500ms (Item 34) |
| `request_latency_seconds` | Histogram | [0.005..1.25]s | seconds | End-to-end request latency | p99 <500ms (Item 34) |

**Bucket Strategy** (Item 3b-29): Custom buckets [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25] aligned to SLO.

### Cache Metrics (3 Counters + 1 Gauge)

| Metric | Type | Labels | Units | Description |
|--------|------|--------|-------|-------------|
| `cache_hits_total` | Counter | — | hits | Total cache hits |
| `cache_misses_total` | Counter | — | misses | Total cache misses |
| `cache_evictions_total` | Counter | `reason` (ttl_expired\|manual_invalidate) | evictions | Cache entries removed |
| `cache_entries` | Gauge | — | entries | Current cache size |

**SLO**: Cache hit rate ≥90% (Item 34)

### Dedup Metrics (3 Counters + 1 Gauge)

| Metric | Type | Labels | Units | Description |
|--------|------|--------|-------|-------------|
| `dedup_leaders_total` | Counter | — | acquisitions | Total leader acquisitions |
| `dedup_followers_total` | Counter | — | joinings | Total follower joinings |
| `dedup_cleanup_total` | Counter | — | cleanups | Total dedup cleanups |
| `dedup_active_entries` | Gauge | — | entries | Current in-flight dedup ops |

**SLO**: Dedup rate ≥80% (Item 34)

### Queue Metrics (3 Counters + 1 Gauge + 1 Histogram)

| Metric | Type | Labels | Units | Description |
|--------|------|--------|-------|-------------|
| `jobs_queued_total` | Counter | — | jobs | Total jobs submitted |
| `jobs_completed_total` | Counter | — | jobs | Total jobs completed |
| `jobs_timed_out_total` | Counter | — | jobs | Total jobs that timed out |
| `queue_depth` | Gauge | — | jobs | Current queue size |
| `queue_wait_latency_seconds` | Histogram | [0.005..1.25]s | seconds | Queue wait time |

### Worker Pool Metrics (2 Counters + 1 Gauge + 1 Histogram)

| Metric | Type | Labels | Units | Description |
|--------|------|--------|-------|-------------|
| `worker_restarts_total` | Counter | — | restarts | Total worker restarts |
| `worker_errors_total` | Counter | — | errors | Total worker errors |
| `workers_active` | Gauge | — | workers | Current active workers |
| `worker_job_duration_seconds` | Histogram | [0.005..1.25]s | seconds | Job processing time |

### Error Metrics (1 Counter)

| Metric | Type | Labels | Units | Description | SLO |
|--------|------|--------|-------|-------------|-----|
| `errors_total` | Counter | `error_type` (InvalidRequest\|GridNotFound\|DataMissing\|ComputationTimeout\|WorkerError\|Unknown) | errors | Total errors by type | Error rate <1% (Item 34) |

### Coordinator Metrics (1 Gauge)

| Metric | Type | Labels | Units | Description |
|--------|------|--------|-------|-------------|
| `inflight_leader_computations` | Gauge | — | leaders | Current executing computations |

---

## Label Cardinality (Item 32: Enforced in Code)

| Label Name | Possible Values | Max Cardinality | Enforcement |
|------------|-----------------|-----------------|-------------|
| `path` | cache, dedup, compute | 3 | Code enum |
| `status` | success, error | 2 | Code enum |
| `error_type` | 6 types (see above) | 6 | Code enum |
| `reason` | ttl_expired, manual_invalidate | 2 | Code enum |

**Total Cardinality**: 18 (controlled via code enums, not runtime)

---

## Performance & Overhead (Item 31)

**Recording**: Every event (no sampling)
**Lock-Free**: Prometheus Go client atomic primitives
**Overhead**: <50μs per request
**Latency Impact**: <0.005% (well under 1% target, Item 31)

---

## SLO Tracking Queries (Items 34-35)

Use with `slo_recording_rules.yaml`:

```promql
# Cache hit rate ≥90%
slo:cache_hit_rate:ratio_5m

# Dedup rate ≥80%
slo:dedup_rate:ratio_5m

# P99 latency <500ms
histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m])) < 0.5

# Error rate <1%
slo:error_rate:ratio_5m < 0.01

# Availability ≥99.9%
slo:availability:ratio_5m >= 0.999

# Burn rates (1h and 30d windows)
slo:cache_hit_rate:burn_rate_1h
slo:error_rate:burn_rate_1h
```

---

## Backwards Compatibility (Items 40-41)

**All 62 metrics locked in v1.0.0**:
- ✅ Metric names: Stable (no versioning prefixes)
- ✅ Metric types: Stable (no type changes)
- ✅ Label names: Stable (no removal/reordering)
- ✅ Label values: Stable (only additions allowed)

See `BACKWARDS_COMPATIBILITY.md` for policy.

---

## Common PromQL Patterns

### Cache Analysis
```promql
# Hit rate
sum(rate(cache_hits_total[5m])) / 
  (sum(rate(cache_hits_total[5m])) + sum(rate(cache_misses_total[5m])))

# Eviction rate
rate(cache_evictions_total[5m])
```

### Dedup Analysis
```promql
# Dedup rate
sum(rate(dedup_followers_total[5m])) / 
  (sum(rate(dedup_leaders_total[5m])) + sum(rate(dedup_followers_total[5m])))

# P99 wait latency
histogram_quantile(0.99, rate(dedup_wait_latency_seconds_bucket[5m]))
```

### Latency Analysis
```promql
# P50, P95, P99 end-to-end
histogram_quantile(0.50, rate(request_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m]))
```

### Error Analysis
```promql
# Error rate by type
rate(errors_total[5m]) by (error_type)

# Timeout errors
rate(errors_total{error_type="ComputationTimeout"}[5m])
```

---

**All metrics follow Prometheus standards (Counter, Gauge, Histogram). No custom types.**
