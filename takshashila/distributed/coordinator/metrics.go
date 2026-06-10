package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var globalMetrics *Metrics

// Buckets resolve sub-100ms cache paths and the multi-second inference edge.
var histogramBuckets = []float64{0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0}

// Metrics holds all Prometheus metrics for the coordinator.
type Metrics struct {
	// requests_total: path={cache|redis|dedup|inference|resolve|validation}, status={success|error}
	RequestsTotal *prometheus.CounterVec
	ErrorsTotal   *prometheus.CounterVec // error_type, model_version, feature_version

	RequestLatencySeconds       prometheus.Histogram
	CacheHitLatencySeconds      prometheus.Histogram
	DedupWaitLatencySeconds     prometheus.Histogram
	LeaderComputeLatencySeconds prometheus.Histogram

	CacheHitsTotal     *prometheus.CounterVec
	CacheMissesTotal   *prometheus.CounterVec
	CacheEvictionTotal *prometheus.CounterVec
	CacheEntriesGauge  prometheus.Gauge

	DedupLeadersTotal          prometheus.Counter
	DedupFollowersTotal        prometheus.Counter
	DedupCleanupTotal          prometheus.Counter
	DedupActiveEntriesGauge    prometheus.Gauge
	DedupedRequestsTotal       prometheus.Counter
	InflightLeaderComputations prometheus.Gauge
}

// NewMetrics creates and registers all Prometheus metrics.
// Must be called at most once per process (prometheus panics on re-registration).
func NewMetrics() *Metrics {
	return &Metrics{
		RequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "requests_total",
			Help: "Prediction requests by serving path, status, model_version, feature_version",
		}, []string{"path", "status", "model_version", "feature_version"}),

		ErrorsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "errors_total",
			Help: "Errors by type, model_version, feature_version",
		}, []string{"error_type", "model_version", "feature_version"}),

		RequestLatencySeconds: promauto.NewHistogram(prometheus.HistogramOpts{
			Name: "request_latency_seconds", Help: "End-to-end /predict latency", Buckets: histogramBuckets,
		}),
		CacheHitLatencySeconds: promauto.NewHistogram(prometheus.HistogramOpts{
			Name: "cache_hit_latency_seconds", Help: "Result cache hit latency", Buckets: histogramBuckets,
		}),
		DedupWaitLatencySeconds: promauto.NewHistogram(prometheus.HistogramOpts{
			Name: "dedup_wait_latency_seconds", Help: "Follower wait time from join to completion", Buckets: histogramBuckets,
		}),
		LeaderComputeLatencySeconds: promauto.NewHistogram(prometheus.HistogramOpts{
			Name: "leader_compute_latency_seconds", Help: "Leader inference call duration", Buckets: histogramBuckets,
		}),

		CacheHitsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "cache_hits_total", Help: "Result cache hits",
		}, []string{"cache_result"}),
		CacheMissesTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "cache_misses_total", Help: "Result cache misses",
		}, []string{"cache_result"}),
		CacheEvictionTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "cache_evictions_total", Help: "Result cache evictions by reason",
		}, []string{"reason"}),
		CacheEntriesGauge: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "cache_entries", Help: "Current result cache entries",
		}),

		DedupLeadersTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "dedup_leaders_total", Help: "Total leader acquisitions",
		}),
		DedupFollowersTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "dedup_followers_total", Help: "Total follower joins",
		}),
		DedupCleanupTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "dedup_cleanup_total", Help: "Total dedup entries cleaned up",
		}),
		DedupActiveEntriesGauge: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "dedup_active_entries", Help: "In-flight dedup entries",
		}),
		DedupedRequestsTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "deduped_requests_total", Help: "Requests served as dedup followers",
		}),
		InflightLeaderComputations: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "inflight_leader_computations", Help: "In-flight kashi calls",
		}),
	}
}

// Label values for cache metrics.
const (
	CacheResultHit  = "hit"
	CacheResultMiss = "miss"
)

// Label values for eviction reasons.
const (
	EvictionReasonTTLExpired       = "ttl_expired"
	EvictionReasonManualInvalidate = "manual_invalidate"
)
