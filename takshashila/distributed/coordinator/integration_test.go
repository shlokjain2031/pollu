package main

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Integration Scenario 1: Cache hit (warm cache, no dedup, no compute)
func TestScenario1_CacheHit(t *testing.T) {
	t.Log("Scenario 1: Cache hit - warm cache bypasses dedup and computation")

	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		return 55.5, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	gridID := "g00000001"
	timestampHour := "2026-01-21T12:00:00Z"

	// Warm up cache with first request
	pm25_1, err := mc.executePrediction(ctx, gridID, timestampHour, "req-warmup")
	if err != nil {
		t.Fatalf("Warmup failed: %v", err)
	}
	if pm25_1 != 55.5 {
		t.Errorf("Expected pm25=55.5, got %.2f", pm25_1)
	}

	if mc.computeCount != 1 {
		t.Fatalf("Expected 1 computation for warmup, got %d", mc.computeCount)
	}

	// Issue 10 subsequent requests - all should hit cache
	for i := 0; i < 10; i++ {
		requestID := fmt.Sprintf("req-cache-%d", i)
		pm25, err := mc.executePrediction(ctx, gridID, timestampHour, requestID)
		if err != nil {
			t.Errorf("Request %s failed: %v", requestID, err)
		}
		if pm25 != 55.5 {
			t.Errorf("Request %s got wrong pm25: %.2f", requestID, pm25)
		}
	}

	// Verify still only 1 computation (all cache hits)
	if mc.computeCount != 1 {
		t.Errorf("Expected 1 computation total (10 cache hits), got %d", mc.computeCount)
	}

	// Verify stats
	cacheStats := mc.cache.Stats()
	if cacheStats.TotalEntries != 1 {
		t.Errorf("Expected 1 cache entry, got %d", cacheStats.TotalEntries)
	}

	dedupStats := mc.dedup.Stats()
	if dedupStats.ActiveComputations != 0 {
		t.Errorf("Expected 0 active dedup computations, got %d", dedupStats.ActiveComputations)
	}

	t.Log("✅ Scenario 1 passed: 10 requests, 1 computation, 10 cache hits")
}

// Integration Scenario 2: Dedup bursty load (5 concurrent identical requests → 1 compute, 4 followers)
func TestScenario2_DedupBurstLoad(t *testing.T) {
	t.Log("Scenario 2: Dedup burst load - 5 concurrent identical requests deduplicated to 1 computation")

	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		// Simulate realistic computation time
		time.Sleep(200 * time.Millisecond)
		return 42.0, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	gridID := "g00000001"
	timestampHour := "2026-01-21T15:00:00Z"

	numRequests := 5
	var wg sync.WaitGroup
	results := make([]float64, numRequests)
	errors := make([]error, numRequests)

	startTime := time.Now()

	// Launch 5 concurrent identical requests
	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			requestID := fmt.Sprintf("burst-req-%d", idx)
			pm25, err := mc.executePrediction(ctx, gridID, timestampHour, requestID)
			results[idx] = pm25
			errors[idx] = err
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(startTime)

	// Verify all requests succeeded with same result
	for i := 0; i < numRequests; i++ {
		if errors[i] != nil {
			t.Errorf("Request %d failed: %v", i, errors[i])
		}
		if results[i] != 42.0 {
			t.Errorf("Request %d got wrong pm25: %.2f", i, results[i])
		}
	}

	// Verify only 1 computation happened (4 deduplicated)
	if mc.computeCount != 1 {
		t.Errorf("Expected 1 computation (4 deduplicated), got %d", mc.computeCount)
	}

	// Calculate dedup rate
	dedupRate := float64(numRequests-int(mc.computeCount)) / float64(numRequests) * 100
	if dedupRate < 80.0 {
		t.Errorf("Dedup rate %.1f%% below target 80%%", dedupRate)
	}

	// Verify elapsed time is ~200ms (not 5 * 200ms = 1s)
	if elapsed > 400*time.Millisecond {
		t.Errorf("Expected ~200ms latency (parallel), got %v", elapsed)
	}

	t.Logf("✅ Scenario 2 passed: 5 requests → 1 computation (%.0f%% dedup), latency=%v", dedupRate, elapsed)
}

// Integration Scenario 3: Mixed workload (cache hits, dedup, fresh computations)
func TestScenario3_MixedWorkload(t *testing.T) {
	t.Log("Scenario 3: Mixed workload - cache hits, deduplication, and fresh computations")

	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		time.Sleep(50 * time.Millisecond)
		// Return different values per grid
		if gridID == "g00000001" {
			return 10.0, nil
		} else if gridID == "g00000002" {
			return 20.0, nil
		}
		return 30.0, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	timestamp1 := "2026-01-21T12:00:00Z"
	timestamp2 := "2026-01-21T13:00:00Z"

	// Phase 1: Warm cache for grid1@t1 and grid2@t1
	pm25, err := mc.executePrediction(ctx, "g00000001", timestamp1, "warmup-g1-t1")
	if err != nil || pm25 != 10.0 {
		t.Fatalf("Warmup g1-t1 failed: pm25=%.2f, err=%v", pm25, err)
	}

	pm25, err = mc.executePrediction(ctx, "g00000002", timestamp1, "warmup-g2-t1")
	if err != nil || pm25 != 20.0 {
		t.Fatalf("Warmup g2-t1 failed: pm25=%.2f, err=%v", pm25, err)
	}

	if mc.computeCount != 2 {
		t.Fatalf("Expected 2 computations after warmup, got %d", mc.computeCount)
	}

	// Phase 2: Mixed workload (concurrent)
	var wg sync.WaitGroup
	results := make(map[string]float64)
	var mu sync.Mutex

	// 3 cache hits for grid1@t1
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			pm25, _ := mc.executePrediction(ctx, "g00000001", timestamp1, fmt.Sprintf("cache-g1-t1-%d", idx))
			mu.Lock()
			results[fmt.Sprintf("cache-g1-t1-%d", idx)] = pm25
			mu.Unlock()
		}(i)
	}

	// 2 cache hits for grid2@t1
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			pm25, _ := mc.executePrediction(ctx, "g00000002", timestamp1, fmt.Sprintf("cache-g2-t1-%d", idx))
			mu.Lock()
			results[fmt.Sprintf("cache-g2-t1-%d", idx)] = pm25
			mu.Unlock()
		}(i)
	}

	// 5 concurrent requests for grid3@t2 (dedup burst)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			pm25, _ := mc.executePrediction(ctx, "g00000003", timestamp2, fmt.Sprintf("dedup-g3-t2-%d", idx))
			mu.Lock()
			results[fmt.Sprintf("dedup-g3-t2-%d", idx)] = pm25
			mu.Unlock()
		}(i)
	}

	wg.Wait()

	// Verify results
	for i := 0; i < 3; i++ {
		if results[fmt.Sprintf("cache-g1-t1-%d", i)] != 10.0 {
			t.Errorf("Cache hit g1-t1-%d failed", i)
		}
	}
	for i := 0; i < 2; i++ {
		if results[fmt.Sprintf("cache-g2-t1-%d", i)] != 20.0 {
			t.Errorf("Cache hit g2-t1-%d failed", i)
		}
	}
	for i := 0; i < 5; i++ {
		if results[fmt.Sprintf("dedup-g3-t2-%d", i)] != 30.0 {
			t.Errorf("Dedup g3-t2-%d failed", i)
		}
	}

	// Verify total computations: warmup(2) + grid3@t2(1) = 3
	if mc.computeCount != 3 {
		t.Errorf("Expected 3 computations, got %d", mc.computeCount)
	}

	cacheStats := mc.cache.Stats()
	if cacheStats.TotalEntries != 3 {
		t.Errorf("Expected 3 cache entries, got %d", cacheStats.TotalEntries)
	}

	t.Log("✅ Scenario 3 passed: 10 requests (5 cache hits, 5 dedup) → 3 computations")
}

// Integration Scenario 4: Cache expiry during burst
func TestScenario4_CacheExpiryDuringBurst(t *testing.T) {
	t.Log("Scenario 4: Cache expiry - first request hits cache, TTL expires, next request recomputes")

	computeCallCount := 0
	var mu sync.Mutex

	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		mu.Lock()
		computeCallCount++
		callNum := computeCallCount
		mu.Unlock()

		time.Sleep(50 * time.Millisecond)

		// Return different value on second computation
		if callNum == 1 {
			return 100.0, nil
		}
		return 200.0, nil
	}

	// Create coordinator with short TTL
	mc := &MockCoordinator{
		dedup:       NewDeduplicator(),
		cache:       NewResultCache(300 * time.Millisecond),
		computeFunc: computeFunc,
	}

	ctx := context.Background()
	gridID := "g00000001"
	timestampHour := "2026-01-21T12:00:00Z"

	// Phase 1: First computation
	pm25_1, err := mc.executePrediction(ctx, gridID, timestampHour, "req-1")
	if err != nil || pm25_1 != 100.0 {
		t.Fatalf("First request failed: pm25=%.2f, err=%v", pm25_1, err)
	}

	// Phase 2: Immediate cache hit
	pm25_2, err := mc.executePrediction(ctx, gridID, timestampHour, "req-2")
	if err != nil || pm25_2 != 100.0 {
		t.Errorf("Immediate cache hit failed: pm25=%.2f, err=%v", pm25_2, err)
	}

	mu.Lock()
	computesBefore := computeCallCount
	mu.Unlock()

	if computesBefore != 1 {
		t.Errorf("Expected 1 computation before expiry, got %d", computesBefore)
	}

	// Phase 3: Wait for cache expiry
	time.Sleep(400 * time.Millisecond)

	// Phase 4: Request after expiry - should recompute
	pm25_3, err := mc.executePrediction(ctx, gridID, timestampHour, "req-3")
	if err != nil {
		t.Fatalf("Request after expiry failed: %v", err)
	}
	if pm25_3 != 200.0 {
		t.Errorf("Expected recomputed pm25=200.0, got %.2f", pm25_3)
	}

	mu.Lock()
	computesAfter := computeCallCount
	mu.Unlock()

	if computesAfter != 2 {
		t.Errorf("Expected 2 computations after expiry, got %d", computesAfter)
	}

	t.Log("✅ Scenario 4 passed: Cache hit → expiry → recomputation")
}

// Integration Scenario 5: Dedup + Cache stress test (100 concurrent requests, verify ≥80% dedup)
func TestScenario5_StressTest(t *testing.T) {
	t.Log("Scenario 5: Stress test - 100 concurrent requests across 5 unique keys")

	var computeCount int32

	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		atomic.AddInt32(&computeCount, 1)
		time.Sleep(100 * time.Millisecond)

		// Return predictable value based on grid
		switch gridID {
		case "g00000001":
			return 10.0, nil
		case "g00000002":
			return 20.0, nil
		case "g00000003":
			return 30.0, nil
		case "g00000004":
			return 40.0, nil
		case "g00000005":
			return 50.0, nil
		default:
			return 0.0, fmt.Errorf("unknown grid")
		}
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	timestampHour := "2026-01-21T18:00:00Z"

	numRequests := 100
	grids := []string{"g00000001", "g00000002", "g00000003", "g00000004", "g00000005"}

	var wg sync.WaitGroup
	results := make([]float64, numRequests)
	errors := make([]error, numRequests)

	startTime := time.Now()

	// Launch 100 concurrent requests (20 per grid)
	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			gridID := grids[idx%len(grids)]
			requestID := fmt.Sprintf("stress-req-%03d", idx)

			pm25, err := mc.executePrediction(ctx, gridID, timestampHour, requestID)
			results[idx] = pm25
			errors[idx] = err
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(startTime)

	// Verify all requests succeeded
	errorCount := 0
	for i := 0; i < numRequests; i++ {
		if errors[i] != nil {
			errorCount++
			t.Errorf("Request %d failed: %v", i, errors[i])
		}
	}

	if errorCount > 0 {
		t.Fatalf("%d/%d requests failed", errorCount, numRequests)
	}

	// Verify results match expected grid values
	expectedValues := map[string]float64{
		"g00000001": 10.0,
		"g00000002": 20.0,
		"g00000003": 30.0,
		"g00000004": 40.0,
		"g00000005": 50.0,
	}

	for i := 0; i < numRequests; i++ {
		gridID := grids[i%len(grids)]
		expectedPM25 := expectedValues[gridID]
		if results[i] != expectedPM25 {
			t.Errorf("Request %d (grid=%s) got pm25=%.2f, expected %.2f", i, gridID, results[i], expectedPM25)
		}
	}

	// Verify dedup effectiveness
	actualComputes := atomic.LoadInt32(&computeCount)
	expectedComputes := int32(5) // One per unique grid

	if actualComputes != expectedComputes {
		t.Errorf("Expected %d computations (1 per grid), got %d", expectedComputes, actualComputes)
	}

	// Calculate dedup rate
	dedupRate := float64(numRequests-int(actualComputes)) / float64(numRequests) * 100
	if dedupRate < 80.0 {
		t.Errorf("Dedup rate %.1f%% below target 80%% (100 requests → %d computations)", dedupRate, actualComputes)
	}

	// Verify cache entries
	cacheStats := mc.cache.Stats()
	if cacheStats.TotalEntries != 5 {
		t.Errorf("Expected 5 cache entries, got %d", cacheStats.TotalEntries)
	}

	// Verify elapsed time is reasonable (should be ~100ms for parallel execution, not 100*100ms)
	if elapsed > 500*time.Millisecond {
		t.Errorf("Expected ~100-200ms for parallel execution, got %v", elapsed)
	}

	t.Logf("✅ Scenario 5 passed: 100 requests → %d computations (%.0f%% dedup), latency=%v", actualComputes, dedupRate, elapsed)
}
