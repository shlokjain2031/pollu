package main

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// MockComputeFunc simulates the computation (replaces worker pool)
type MockComputeFunc func(ctx context.Context, gridID, timestampHour string) (float64, error)

// MockCoordinator wraps coordinator logic for testing without HTTP/worker pool
type MockCoordinator struct {
	dedup       *Deduplicator
	cache       *ResultCache
	computeFunc MockComputeFunc
	computeCount int32 // Track number of actual computations
}

func NewMockCoordinator(computeFunc MockComputeFunc) *MockCoordinator {
	return &MockCoordinator{
		dedup:       NewDeduplicator(),
		cache:       NewResultCache(5 * time.Minute),
		computeFunc: computeFunc,
	}
}

// executePrediction mimics the real coordinator's executePrediction
func (mc *MockCoordinator) executePrediction(ctx context.Context, gridID, timestampHour, requestID string) (float64, error) {
	// Construct dedup/cache key
	key := DedupKey{
		GridID:         gridID,
		TimestampHour:  timestampHour,
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Step 1: Check cache
	cacheKey := CacheKey{
		GridID:         key.GridID,
		TimestampHour:  key.TimestampHour,
		ModelVersion:   key.ModelVersion,
		FeatureVersion: key.FeatureVersion,
	}

	if pm25, cachedErr, hit := mc.cache.Get(cacheKey); hit {
		if cachedErr != nil {
			return 0, cachedErr
		}
		return pm25, nil
	}

	// Step 2: Attempt to acquire leader
	entry, isLeader := mc.dedup.AcquireLeader(ctx, key, requestID)

	if !isLeader {
		// We're a follower - wait for leader
		result, err := mc.dedup.WaitForResult(ctx, entry)
		if err != nil {
			return 0, err
		}
		return result.PM25, result.Error
	}

	// We're the leader - execute computation
	atomic.AddInt32(&mc.computeCount, 1)
	pm25, err := mc.computeFunc(ctx, gridID, timestampHour)

	// Complete dedup
	dedupResult := &DedupResult{
		PM25:  pm25,
		Error: err,
	}
	mc.dedup.Complete(key, dedupResult)

	// Cache result only if successful
	if err == nil {
		mc.cache.Set(cacheKey, pm25)
	}

	return pm25, err
}

// Test 1: Cache hit avoids computation
func TestCoordinatorCacheHit(t *testing.T) {
	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		return 50.0, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	gridID := "g00000001"
	timestampHour := "2026-01-21T12:00:00Z"

	// First request - cache miss, computes
	pm25_1, err := mc.executePrediction(ctx, gridID, timestampHour, "req-001")
	if err != nil {
		t.Fatalf("First request failed: %v", err)
	}
	if pm25_1 != 50.0 {
		t.Errorf("Expected pm25=50.0, got %.2f", pm25_1)
	}

	// Second request - cache hit, no computation
	pm25_2, err := mc.executePrediction(ctx, gridID, timestampHour, "req-002")
	if err != nil {
		t.Fatalf("Second request failed: %v", err)
	}
	if pm25_2 != 50.0 {
		t.Errorf("Expected pm25=50.0, got %.2f", pm25_2)
	}

	// Verify only 1 computation happened
	if mc.computeCount != 1 {
		t.Errorf("Expected 1 computation, got %d", mc.computeCount)
	}
}

// Test 2: Dedup reduces concurrent identical requests to single computation
func TestCoordinatorDedupBurst(t *testing.T) {
	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		// Simulate slow computation
		time.Sleep(100 * time.Millisecond)
		return 75.5, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	gridID := "g00000001"
	timestampHour := "2026-01-21T12:00:00Z"

	numRequests := 50
	var wg sync.WaitGroup
	results := make([]float64, numRequests)

	// Launch 50 concurrent identical requests
	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			requestID := fmt.Sprintf("req-%03d", idx)
			pm25, err := mc.executePrediction(ctx, gridID, timestampHour, requestID)
			if err != nil {
				t.Errorf("Request %s failed: %v", requestID, err)
			}
			results[idx] = pm25
		}(i)
	}

	wg.Wait()

	// Verify all requests got same result
	for i, pm25 := range results {
		if pm25 != 75.5 {
			t.Errorf("Request %d got wrong result: %.2f", i, pm25)
		}
	}

	// Verify only 1 computation happened (dedup worked)
	if mc.computeCount != 1 {
		t.Errorf("Expected 1 computation (dedup), got %d", mc.computeCount)
	}
}

// Test 3: Different grid IDs bypass dedup
func TestCoordinatorDifferentGrids(t *testing.T) {
	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		// Return different values per grid
		if gridID == "g00000001" {
			return 10.0, nil
		}
		return 20.0, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	timestampHour := "2026-01-21T12:00:00Z"

	// Request for grid 1
	pm25_1, err := mc.executePrediction(ctx, "g00000001", timestampHour, "req-001")
	if err != nil {
		t.Fatalf("Grid 1 request failed: %v", err)
	}
	if pm25_1 != 10.0 {
		t.Errorf("Expected grid1 pm25=10.0, got %.2f", pm25_1)
	}

	// Request for grid 2
	pm25_2, err := mc.executePrediction(ctx, "g00000002", timestampHour, "req-002")
	if err != nil {
		t.Fatalf("Grid 2 request failed: %v", err)
	}
	if pm25_2 != 20.0 {
		t.Errorf("Expected grid2 pm25=20.0, got %.2f", pm25_2)
	}

	// Verify 2 computations (different grids = different keys)
	if mc.computeCount != 2 {
		t.Errorf("Expected 2 computations, got %d", mc.computeCount)
	}
}

// Test 4: Computation error not cached
func TestCoordinatorErrorNotCached(t *testing.T) {
	callCount := 0
	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		callCount++
		if callCount == 1 {
			return 0, fmt.Errorf("grid not found")
		}
		return 60.0, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()
	gridID := "g00000001"
	timestampHour := "2026-01-21T12:00:00Z"

	// First request - error
	_, err := mc.executePrediction(ctx, gridID, timestampHour, "req-001")
	if err == nil {
		t.Fatal("Expected error from first request")
	}

	// Second request - should recompute (error wasn't cached)
	pm25_2, err := mc.executePrediction(ctx, gridID, timestampHour, "req-002")
	if err != nil {
		t.Fatalf("Second request failed: %v", err)
	}
	if pm25_2 != 60.0 {
		t.Errorf("Expected pm25=60.0, got %.2f", pm25_2)
	}

	// Verify 2 computations happened (error not cached)
	if mc.computeCount != 2 {
		t.Errorf("Expected 2 computations, got %d", mc.computeCount)
	}
}

// Test 5: Cache expiry forces recomputation
func TestCoordinatorCacheExpiry(t *testing.T) {
	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		return 30.0, nil
	}

	// Create coordinator with short TTL
	mc := &MockCoordinator{
		dedup:       NewDeduplicator(),
		cache:       NewResultCache(200 * time.Millisecond),
		computeFunc: computeFunc,
	}

	ctx := context.Background()
	gridID := "g00000001"
	timestampHour := "2026-01-21T12:00:00Z"

	// First request - computes and caches
	pm25_1, err := mc.executePrediction(ctx, gridID, timestampHour, "req-001")
	if err != nil {
		t.Fatalf("First request failed: %v", err)
	}
	if pm25_1 != 30.0 {
		t.Errorf("Expected pm25=30.0, got %.2f", pm25_1)
	}

	// Second request immediately - cache hit
	_, err = mc.executePrediction(ctx, gridID, timestampHour, "req-002")
	if err != nil {
		t.Fatalf("Second request failed: %v", err)
	}
	if mc.computeCount != 1 {
		t.Errorf("Expected 1 computation after immediate request, got %d", mc.computeCount)
	}

	// Wait for cache expiry
	time.Sleep(250 * time.Millisecond)

	// Third request - cache expired, recomputes
	pm25_3, err := mc.executePrediction(ctx, gridID, timestampHour, "req-003")
	if err != nil {
		t.Fatalf("Third request failed: %v", err)
	}
	if pm25_3 != 30.0 {
		t.Errorf("Expected pm25=30.0, got %.2f", pm25_3)
	}

	// Verify recomputation happened
	if mc.computeCount != 2 {
		t.Errorf("Expected 2 computations after expiry, got %d", mc.computeCount)
	}
}

// Test 6: Context timeout propagates to followers
func TestCoordinatorFollowerTimeout(t *testing.T) {
	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		// Simulate very slow computation
		time.Sleep(5 * time.Second)
		return 100.0, nil
	}

	mc := NewMockCoordinator(computeFunc)

	gridID := "g00000001"
	timestampHour := "2026-01-21T12:00:00Z"

	// Leader starts (long timeout)
	leaderCtx, leaderCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer leaderCancel()

	leaderDone := make(chan struct{})
	go func() {
		mc.executePrediction(leaderCtx, gridID, timestampHour, "req-leader")
		close(leaderDone)
	}()

	// Give leader time to acquire lock
	time.Sleep(50 * time.Millisecond)

	// Follower with short timeout
	followerCtx, followerCancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer followerCancel()

	_, err := mc.executePrediction(followerCtx, gridID, timestampHour, "req-follower")

	// Follower should timeout
	if err != context.DeadlineExceeded {
		t.Errorf("Expected context.DeadlineExceeded, got %v", err)
	}

	// Leader should still be running
	select {
	case <-leaderDone:
		// Leader finished (or we'll cancel it below)
	case <-time.After(100 * time.Millisecond):
		// Expected - leader still running
	}

	// Cancel leader to avoid test hanging
	leaderCancel()
}

// Test 7: Stats reflect dedup and cache state
func TestCoordinatorStats(t *testing.T) {
	computeFunc := func(ctx context.Context, gridID, timestampHour string) (float64, error) {
		time.Sleep(50 * time.Millisecond)
		return 80.0, nil
	}

	mc := NewMockCoordinator(computeFunc)

	ctx := context.Background()

	// Initial stats
	dedupStats := mc.dedup.Stats()
	cacheStats := mc.cache.Stats()

	if dedupStats.ActiveComputations != 0 {
		t.Errorf("Expected 0 active computations initially, got %d", dedupStats.ActiveComputations)
	}
	if cacheStats.TotalEntries != 0 {
		t.Errorf("Expected 0 cache entries initially, got %d", cacheStats.TotalEntries)
	}

	// Execute prediction
	pm25, err := mc.executePrediction(ctx, "g00000001", "2026-01-21T12:00:00Z", "req-001")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if pm25 != 80.0 {
		t.Errorf("Expected pm25=80.0, got %.2f", pm25)
	}

	// After completion, dedup should be cleaned up, cache should have 1 entry
	dedupStats = mc.dedup.Stats()
	cacheStats = mc.cache.Stats()

	if dedupStats.ActiveComputations != 0 {
		t.Errorf("Expected 0 active computations after completion, got %d", dedupStats.ActiveComputations)
	}

	if cacheStats.TotalEntries != 1 {
		t.Errorf("Expected 1 cache entry, got %d", cacheStats.TotalEntries)
	}
}
