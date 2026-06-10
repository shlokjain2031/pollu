package main

import (
	"os"
	"testing"
)

// TestMain initializes the shared metrics once: prometheus panics on
// duplicate registration, so tests must not call NewMetrics themselves.
func TestMain(m *testing.M) {
	globalMetrics = NewMetrics()
	os.Exit(m.Run())
}

// TestMetricsSmoke exercises every metric used on the request path.
func TestMetricsSmoke(t *testing.T) {
	m := globalMetrics

	m.RequestsTotal.WithLabelValues("cache", "success", "test", "test").Inc()
	m.ErrorsTotal.WithLabelValues("NotInCoverage", "test", "test").Inc()
	m.RequestLatencySeconds.Observe(0.05)
	m.CacheHitLatencySeconds.Observe(0.001)
	m.DedupWaitLatencySeconds.Observe(0.01)
	m.LeaderComputeLatencySeconds.Observe(0.5)
	m.CacheHitsTotal.WithLabelValues(CacheResultHit).Inc()
	m.CacheMissesTotal.WithLabelValues(CacheResultMiss).Inc()
	m.CacheEvictionTotal.WithLabelValues(EvictionReasonTTLExpired).Inc()
	m.CacheEntriesGauge.Set(1)
	m.DedupLeadersTotal.Inc()
	m.DedupFollowersTotal.Inc()
	m.DedupCleanupTotal.Inc()
	m.DedupActiveEntriesGauge.Set(0)
	m.DedupedRequestsTotal.Inc()
	m.InflightLeaderComputations.Set(0)
}
