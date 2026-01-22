package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Custom histogram buckets aligned to 1s SLO
// Tightly resolves sub-100ms cache paths and 1s timeout edge
var histogramBuckets = []float64{0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25}

// Metrics holds all Prometheus metrics for the coordinator
type Metrics struct {
	// Request Metrics
	// requests_total: Counter with path={cache|dedup|compute}, status={success|error}
	RequestsTotal *prometheus.CounterVec

	// Latency Metrics (separate histograms per path, success-only)
	CacheHitLatencySeconds      prometheus.Histogram // Hit path only
	DedupWaitLatencySeconds     prometheus.Histogram // Follower wait time
	LeaderComputeLatencySeconds prometheus.Histogram // Leader computation
	RequestLatencySeconds       prometheus.Histogram // All requests end-to-end

	// Cache Metrics
	CacheHitsTotal     *prometheus.CounterVec // Labels: cache_result
	CacheMissesTotal   *prometheus.CounterVec // Labels: cache_result
	CacheEvictionTotal *prometheus.CounterVec // Labels: reason
	CacheEntriesGauge  prometheus.Gauge

	// Dedup Metrics
	DedupLeadersTotal          prometheus.Counter
	DedupFollowersTotal        prometheus.Counter
	DedupCleanupTotal          prometheus.Counter
	DedupActiveEntriesGauge    prometheus.Gauge
	InflightLeaderComputations prometheus.Gauge // Current leaders executing

	// Deduped Requests Counter (followers only)
	DedupedRequestsTotal prometheus.Counter

	// Error Metrics
	ErrorsTotal *prometheus.CounterVec // Labels: error_type

	// Worker Pool & Queue Metrics
	QueueDepthGauge          prometheus.Gauge
	QueueWaitLatencySeconds  prometheus.Histogram
	JobsQueuedTotal          prometheus.Counter
	JobsCompletedTotal       prometheus.Counter
	JobsTimedOutTotal        prometheus.Counter
	WorkersActiveGauge       prometheus.Gauge
	WorkerRestartsTotal      prometheus.Counter
	WorkerErrorsTotal        prometheus.Counter
	WorkerJobDurationSeconds prometheus.Histogram
}

// NewMetrics creates and registers all Prometheus metrics
func NewMetrics() *Metrics {
	m := &Metrics{
		// Request metrics with path, status, model_version, feature_version labels
		// Item 26: model_version and feature_version allow tracking model/feature rollouts
		// Note: Low-cardinality labels only (typically 2-3 active model versions)
		RequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "requests_total",
				Help: "Total prediction requests by path (cache|dedup|compute), status (success|error), model_version, and feature_version",
			},
			[]string{"path", "status", "model_version", "feature_version"},
		),

		// Latency histograms - separate per path, success-only
		CacheHitLatencySeconds: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "cache_hit_latency_seconds",
				Help:    "Cache hit latency (cache_result=hit path only)",
				Buckets: histogramBuckets,
			},
		),

		DedupWaitLatencySeconds: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "dedup_wait_latency_seconds",
				Help:    "Follower wait time from join to completion",
				Buckets: histogramBuckets,
			},
		),

		LeaderComputeLatencySeconds: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "leader_compute_latency_seconds",
				Help:    "Leader computation time (from acquire to result)",
				Buckets: histogramBuckets,
			},
		),

		RequestLatencySeconds: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "request_latency_seconds",
				Help:    "End-to-end request latency (all outcomes)",
				Buckets: histogramBuckets,
			},
		),

		// Cache metrics
		CacheHitsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cache_hits_total",
				Help: "Cache hits by result type",
			},
			[]string{"cache_result"},
		),

		CacheMissesTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cache_misses_total",
				Help: "Cache misses",
			},
			[]string{"cache_result"},
		),

		CacheEvictionTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cache_evictions_total",
				Help: "Cache evictions by reason (ttl_expired, manual_invalidate)",
			},
			[]string{"reason"},
		),

		CacheEntriesGauge: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "cache_entries",
				Help: "Current number of entries in the result cache",
			},
		),

		// Dedup metrics
		DedupLeadersTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "dedup_leaders_total",
				Help: "Total leader acquisitions",
			},
		),

		DedupFollowersTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "dedup_followers_total",
				Help: "Total follower joinings",
			},
		),

		DedupCleanupTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "dedup_cleanup_total",
				Help: "Total dedup entries cleaned up after completion",
			},
		),

		DedupActiveEntriesGauge: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "dedup_active_entries",
				Help: "Current number of active dedup entries in flight",
			},
		),

		InflightLeaderComputations: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "inflight_leader_computations",
				Help: "Current number of leader computations in flight",
			},
		),

		// Deduped requests (followers only)
		DedupedRequestsTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "deduped_requests_total",
				Help: "Total requests that were deduplicated (followers)",
			},
		),

		// Error metrics with error_type, model_version, feature_version labels
		// Item 26: model_version and feature_version for debugging model/feature issues
		ErrorsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "errors_total",
				Help: "Total errors by type, model_version, and feature_version",
			},
			[]string{"error_type", "model_version", "feature_version"},
		),

		// Worker pool & queue metrics
		QueueDepthGauge: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "queue_depth",
				Help: "Current number of jobs in the queue",
			},
		),

		QueueWaitLatencySeconds: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "queue_wait_latency_seconds",
				Help:    "Time job waits in queue before worker picks it up",
				Buckets: histogramBuckets,
			},
		),

		JobsQueuedTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "jobs_queued_total",
				Help: "Total jobs queued",
			},
		),

		JobsCompletedTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "jobs_completed_total",
				Help: "Total jobs completed successfully",
			},
		),

		JobsTimedOutTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "jobs_timed_out_total",
				Help: "Total jobs that exceeded execution deadline (Item 25, distinct from worker_restarts_total)",
			},
		),

		WorkersActiveGauge: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "workers_active",
				Help: "Current number of active workers (Item 25)",
			},
		),

		// Item 25: worker_restarts_total increments ONLY on crash/health-check failure,
		// NOT on job-level timeouts (those are tracked in jobs_timed_out_total)
		WorkerRestartsTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "worker_restarts_total",
				Help: "Total worker process restarts due to crash or health-check failure (Item 25)",
			},
		),

		// Item 25: worker_errors_total counts errors returned by worker (inference exceptions, etc.)
		// Does NOT count crashes (those are worker_restarts_total)
		WorkerErrorsTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "worker_errors_total",
				Help: "Total worker errors (non-crash exceptions) (Item 25)",
			},
		),

		WorkerJobDurationSeconds: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "worker_job_duration_seconds",
				Help:    "End-to-end job processing time in worker",
				Buckets: histogramBuckets,
			},
		),
	}

	return m
}

// MetricsWithLabels wraps metrics (for backwards compatibility with tests)
type MetricsWithLabels struct {
	*Metrics
}

// NewMetricsWithLabels creates metrics with label support
func NewMetricsWithLabels() *MetricsWithLabels {
	return &MetricsWithLabels{
		Metrics: NewMetrics(),
	}
}

// CacheResult enum for label values
const (
	CacheResultHit  = "hit"
	CacheResultMiss = "miss"
)

// EvictionReason enum for label values
const (
	EvictionReasonTTLExpired       = "ttl_expired"
	EvictionReasonManualInvalidate = "manual_invalidate"
)

// DedupRole enum for label values
const (
	DedupRoleLeader   = "leader"
	DedupRoleFollower = "follower"
)
