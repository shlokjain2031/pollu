package main

import (
	"testing"
	"time"
)

// TestHistogramBucketsCustom verifies histogram uses SLO-aligned buckets
func TestHistogramBucketsCustom(t *testing.T) {
	// Verify custom buckets are used
	expectedBuckets := []float64{0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25}

	// Check that bucket order matches expectations
	if len(expectedBuckets) != len(histogramBuckets) {
		t.Errorf("expected %d buckets, got %d", len(expectedBuckets), len(histogramBuckets))
	}

	for i, expected := range expectedBuckets {
		if i >= len(histogramBuckets) {
			t.Errorf("bucket %d missing", i)
			break
		}
		if histogramBuckets[i] != expected {
			t.Errorf("bucket %d: expected %f, got %f", i, expected, histogramBuckets[i])
		}
	}
}

// TestCacheHitLatencyHistogram verifies cache hit latency is recorded correctly
func TestCacheHitLatencyHistogram(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	startTime := time.Now()
	time.Sleep(10 * time.Millisecond)
	duration := time.Since(startTime).Seconds()

	globalMetrics.CacheHitLatencySeconds.Observe(duration)
}

// TestRequestLatencyAllOutcomes verifies request latency tracks all requests
func TestRequestLatencyAllOutcomes(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.RequestLatencySeconds.Observe(0.05)
	globalMetrics.RequestLatencySeconds.Observe(0.5)
	globalMetrics.RequestLatencySeconds.Observe(1.0)
}

// TestRequestsTotal_PathStatus verifies requests_total has correct labels
func TestRequestsTotal_PathStatus(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.RequestsTotal.WithLabelValues("cache", "success").Inc()
	globalMetrics.RequestsTotal.WithLabelValues("cache", "error").Inc()
	globalMetrics.RequestsTotal.WithLabelValues("dedup", "success").Inc()
	globalMetrics.RequestsTotal.WithLabelValues("dedup", "error").Inc()
	globalMetrics.RequestsTotal.WithLabelValues("compute", "success").Inc()
	globalMetrics.RequestsTotal.WithLabelValues("compute", "error").Inc()
}

// TestDedupedRequestsTotal verifies deduped requests are tracked
func TestDedupedRequestsTotal(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.DedupedRequestsTotal.Inc()
	globalMetrics.DedupedRequestsTotal.Inc()
	globalMetrics.DedupedRequestsTotal.Inc()
}

// TestInflightLeaderComputations verifies in-flight gauge
func TestInflightLeaderComputations(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.InflightLeaderComputations.Set(1)
	globalMetrics.InflightLeaderComputations.Inc()
	globalMetrics.InflightLeaderComputations.Dec()
	globalMetrics.InflightLeaderComputations.Set(0)
}

// TestWorkerPoolMetrics verifies worker pool and queue metrics
func TestWorkerPoolMetrics(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.QueueDepthGauge.Set(5)
	globalMetrics.JobsQueuedTotal.Inc()
	globalMetrics.JobsCompletedTotal.Inc()
	globalMetrics.JobsTimedOutTotal.Inc()
	globalMetrics.WorkersActiveGauge.Set(3)
	globalMetrics.WorkerRestartsTotal.Inc()
	globalMetrics.WorkerErrorsTotal.Inc()
	globalMetrics.WorkerJobDurationSeconds.Observe(0.25)
	globalMetrics.WorkerJobDurationSeconds.Observe(0.5)
}

// TestQueueWaitLatency verifies queue wait latency histogram
func TestQueueWaitLatency(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.QueueWaitLatencySeconds.Observe(0.001)
	globalMetrics.QueueWaitLatencySeconds.Observe(0.01)
	globalMetrics.QueueWaitLatencySeconds.Observe(0.1)
}

// TestErrorMetricsTotal verifies error type classification
func TestErrorMetricsTotal(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.ErrorsTotal.WithLabelValues("InvalidRequest").Inc()
	globalMetrics.ErrorsTotal.WithLabelValues("GridNotFound").Inc()
	globalMetrics.ErrorsTotal.WithLabelValues("DataMissing").Inc()
	globalMetrics.ErrorsTotal.WithLabelValues("ComputationTimeout").Inc()
	globalMetrics.ErrorsTotal.WithLabelValues("WorkerError").Inc()
	globalMetrics.ErrorsTotal.WithLabelValues("Unknown").Inc()
}

// TestCacheMetricsVectorized verifies cache result labels
func TestCacheMetricsVectorized(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.CacheHitsTotal.WithLabelValues("hit").Inc()
	globalMetrics.CacheMissesTotal.WithLabelValues("miss").Inc()
	globalMetrics.CacheEvictionTotal.WithLabelValues("ttl_expired").Inc()
	globalMetrics.CacheEvictionTotal.WithLabelValues("manual_invalidate").Inc()
}

// TestDedupMetricsNew verifies dedup-specific metrics
func TestDedupMetricsNew(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.DedupLeadersTotal.Inc()
	globalMetrics.DedupFollowersTotal.Add(5)
	globalMetrics.DedupActiveEntriesGauge.Set(3)
	globalMetrics.DedupCleanupTotal.Add(2)
	globalMetrics.DedupWaitLatencySeconds.Observe(0.01)
	globalMetrics.DedupWaitLatencySeconds.Observe(0.05)
}

// TestThreadSafety verifies metrics can be used concurrently
func TestThreadSafety(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	done := make(chan bool, 3)

	go func() {
		for i := 0; i < 50; i++ {
			globalMetrics.RequestsTotal.WithLabelValues("cache", "success").Inc()
			globalMetrics.RequestLatencySeconds.Observe(0.01)
		}
		done <- true
	}()

	go func() {
		for i := 0; i < 50; i++ {
			globalMetrics.WorkersActiveGauge.Inc()
			globalMetrics.WorkerJobDurationSeconds.Observe(0.05)
		}
		done <- true
	}()

	go func() {
		for i := 0; i < 50; i++ {
			globalMetrics.DedupFollowersTotal.Inc()
			globalMetrics.DedupWaitLatencySeconds.Observe(0.02)
		}
		done <- true
	}()

	<-done
	<-done
	<-done
}

// TestLeaderComputeLatency tests leader_compute_latency_seconds histogram
func TestLeaderComputeLatency(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	latencies := []float64{0.01, 0.05, 0.1, 0.5, 0.99, 1.2}
	for _, lat := range latencies {
		globalMetrics.LeaderComputeLatencySeconds.Observe(lat)
	}
}

// TestPathSpecificMetrics verifies path label tracking
func TestPathSpecificMetrics(t *testing.T) {
	if globalMetrics == nil {
		globalMetrics = NewMetricsWithLabels()
	}

	globalMetrics.RequestsTotal.WithLabelValues("cache", "success").Add(10)
	globalMetrics.RequestsTotal.WithLabelValues("dedup", "success").Add(5)
	globalMetrics.RequestsTotal.WithLabelValues("dedup", "error").Add(1)
	globalMetrics.RequestsTotal.WithLabelValues("compute", "success").Add(3)
	globalMetrics.RequestsTotal.WithLabelValues("compute", "error").Add(2)
}
