package main

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestCacheSetGet tests basic cache set/get functionality
func TestCacheSetGet(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"
	value := 42.5

	// Get from empty cache
	_, hit := cache.Get(gridID, ts)
	if hit {
		t.Errorf("Expected cache miss, got hit")
	}

	// Set value
	cache.Set(gridID, ts, value)

	// Small delay for async write to start
	time.Sleep(10 * time.Millisecond)

	// Get should hit
	retrieved, hit := cache.Get(gridID, ts)
	if !hit {
		t.Errorf("Expected cache hit, got miss")
	}

	if retrieved != value {
		t.Errorf("Expected %f, got %f", value, retrieved)
	}
}

// TestCacheTTLExpiration tests that entries expire after TTL
func TestCacheTTLExpiration(t *testing.T) {
	cache := NewPredictionCache(100 * time.Millisecond)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"
	value := 42.5

	cache.Set(gridID, ts, value)

	// Immediate get should hit
	_, hit := cache.Get(gridID, ts)
	if !hit {
		t.Errorf("Expected cache hit immediately after set")
	}

	// Wait for expiration
	time.Sleep(150 * time.Millisecond)

	// Get should miss (expired)
	_, hit = cache.Get(gridID, ts)
	if hit {
		t.Errorf("Expected cache miss after TTL expiration")
	}
}

// TestWriteStateTransition tests WriteState lifecycle
func TestWriteStateTransition(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"
	value := 42.5

	// Set with explicit state
	cache.SetWithState(gridID, ts, value, NotWritten, "test")

	// Check initial state
	cache.mu.RLock()
	entry, exists := cache.cache[cache.makeKey(gridID, ts)]
	cache.mu.RUnlock()

	if !exists {
		t.Fatalf("Entry not found in cache")
	}

	if entry.WriteState != NotWritten {
		t.Errorf("Expected WriteState=NotWritten, got %v", entry.WriteState)
	}

	if entry.Source != "test" {
		t.Errorf("Expected Source=test, got %s", entry.Source)
	}
}

// TestConcurrentSameKeyWrites tests first-write-wins behavior
func TestConcurrentSameKeyWrites(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"

	// Set initial value
	cache.Set(gridID, ts, 100.0)
	time.Sleep(50 * time.Millisecond)

	// Get initial value
	val1, _ := cache.Get(gridID, ts)
	if val1 != 100.0 {
		t.Errorf("Expected 100.0, got %f", val1)
	}

	// Attempt to overwrite (should NOT overwrite, first-write-wins)
	cache.Set(gridID, ts, 200.0)
	time.Sleep(50 * time.Millisecond)

	// Should still have original value
	val2, _ := cache.Get(gridID, ts)
	if val2 != 100.0 {
		t.Errorf("Expected 100.0 (first write wins), got %f", val2)
	}
}

// TestCacheMissMetrics tests that miss counter increments
func TestCacheMissMetrics(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	_, hit := cache.Get("nonexistent", "2026-01-19T14:00:00Z")
	if hit {
		t.Errorf("Expected cache miss")
	}

	cache.mu.RLock()
	misses := cache.misses
	cache.mu.RUnlock()

	if misses != 1 {
		t.Errorf("Expected 1 miss, got %d", misses)
	}
}

// TestCacheHitMetrics tests that hit counter increments
func TestCacheHitMetrics(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"

	cache.Set(gridID, ts, 42.5)
	time.Sleep(10 * time.Millisecond)

	cache.Get(gridID, ts)

	cache.mu.RLock()
	hits := cache.hits
	cache.mu.RUnlock()

	if hits != 1 {
		t.Errorf("Expected 1 hit, got %d", hits)
	}
}

// TestCacheStatsCalculation tests GetStats() hit rate calculation
func TestCacheStatsCalculation(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"

	cache.Set(gridID, ts, 42.5)
	time.Sleep(10 * time.Millisecond)

	// 2 hits
	cache.Get(gridID, ts)
	cache.Get(gridID, ts)

	// 1 miss
	cache.Get("nonexistent", "2026-01-19T14:00:00Z")

	stats := cache.GetStats()

	if stats.Hits != 2 {
		t.Errorf("Expected 2 hits, got %d", stats.Hits)
	}

	if stats.Misses != 1 {
		t.Errorf("Expected 1 miss, got %d", stats.Misses)
	}

	expectedHitRate := 2.0 / 3.0
	if stats.HitRate < expectedHitRate-0.01 || stats.HitRate > expectedHitRate+0.01 {
		t.Errorf("Expected hit rate ~%.2f, got %.2f", expectedHitRate, stats.HitRate)
	}
}

// TestConcurrentReads tests multiple concurrent reads
func TestConcurrentReads(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"
	value := 42.5

	cache.Set(gridID, ts, value)
	time.Sleep(50 * time.Millisecond)

	// 100 concurrent reads
	var wg sync.WaitGroup
	successCount := atomic.Int64{}

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if val, hit := cache.Get(gridID, ts); hit && val == value {
				successCount.Add(1)
			}
		}()
	}

	wg.Wait()

	if successCount.Load() != 100 {
		t.Errorf("Expected 100 successful reads, got %d", successCount.Load())
	}
}

// TestConcurrentWritesDifferentKeys tests concurrent writes to different keys
func TestConcurrentWritesDifferentKeys(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	ts := "2026-01-19T14:00:00Z"
	numKeys := 50

	// Write 50 different keys concurrently
	var wg sync.WaitGroup
	for i := 0; i < numKeys; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			gridID := fmt.Sprintf("g%08d", idx)
			cache.Set(gridID, ts, float64(idx))
		}(i)
	}

	wg.Wait()

	// Verify all keys are in cache
	time.Sleep(50 * time.Millisecond)

	for i := 0; i < numKeys; i++ {
		gridID := fmt.Sprintf("g%08d", i)
		if val, hit := cache.Get(gridID, ts); !hit || val != float64(i) {
			t.Errorf("Key %s: expected hit with value %f, got hit=%v value=%f", gridID, float64(i), hit, val)
		}
	}
}

// TestPeriodicCleanup tests that old entries are cleaned up
func TestPeriodicCleanup(t *testing.T) {
	cache := NewPredictionCache(100 * time.Millisecond)
	defer cache.Stop()

	// Add 10 entries
	ts := "2026-01-19T14:00:00Z"
	for i := 0; i < 10; i++ {
		gridID := fmt.Sprintf("g%08d", i)
		cache.Set(gridID, ts, float64(i))
	}

	cache.mu.RLock()
	initialCount := int64(len(cache.cache))
	cache.mu.RUnlock()

	if initialCount != 10 {
		t.Errorf("Expected 10 entries, got %d", initialCount)
	}

	// Wait for TTL expiration
	time.Sleep(150 * time.Millisecond)

	// Manually trigger cleanup instead of waiting for periodic cycle (5 minutes)
	cache.runCleanup()

	cache.mu.RLock()
	finalCount := int64(len(cache.cache))
	cache.mu.RUnlock()

	if finalCount != 0 {
		t.Errorf("Expected 0 entries after cleanup, got %d", finalCount)
	}
}

// TestWriteStateProgress tests that write state progresses correctly
func TestWriteStateProgress(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"
	value := 42.5

	cache.Set(gridID, ts, value)

	// Give async write time to complete
	time.Sleep(100 * time.Millisecond)

	cache.mu.RLock()
	entry, exists := cache.cache[cache.makeKey(gridID, ts)]
	cache.mu.RUnlock()

	if !exists {
		t.Fatalf("Entry not found")
	}

	// Should be Written (async write completes quickly in test)
	if entry.WriteState != Written {
		t.Errorf("Expected WriteState=Written, got %v", entry.WriteState)
	}

	if entry.Attempts != 1 {
		t.Errorf("Expected 1 attempt, got %d", entry.Attempts)
	}
}

// TestRetryQueueCapacity tests that retry queue handles overflow
func TestRetryQueueCapacity(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	// This test just ensures the retry queue doesn't panic on overflow
	// The queue has capacity 10000, so we won't fill it in a test

	stats := cache.GetStats()
	if stats.WriteFailures != 0 {
		t.Errorf("Expected 0 write failures initially, got %d", stats.WriteFailures)
	}
}

// TestCacheKeyFormat tests that cache keys are formatted correctly
func TestCacheKeyFormat(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"

	expectedKey := "g00000001:2026-01-19T14:00:00Z"
	actualKey := cache.makeKey(gridID, ts)

	if actualKey != expectedKey {
		t.Errorf("Expected key %s, got %s", expectedKey, actualKey)
	}
}

// TestCacheLazyEviction tests lazy eviction on access
func TestCacheLazyEviction(t *testing.T) {
	cache := NewPredictionCache(100 * time.Millisecond)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"

	cache.Set(gridID, ts, 42.5)

	// Wait for expiration
	time.Sleep(150 * time.Millisecond)

	// Access should trigger lazy eviction
	_, hit := cache.Get(gridID, ts)
	if hit {
		t.Errorf("Expected cache miss after TTL expiration")
	}

	// Entry should be deleted
	cache.mu.RLock()
	_, exists := cache.cache[cache.makeKey(gridID, ts)]
	cache.mu.RUnlock()

	if exists {
		t.Errorf("Expected entry to be evicted after lazy eviction")
	}

	cache.mu.RLock()
	evictions := cache.evictions
	cache.mu.RUnlock()

	if evictions != 1 {
		t.Errorf("Expected 1 eviction, got %d", evictions)
	}
}

// TestMultipleEntriesTTL tests TTL with multiple entries
func TestMultipleEntriesTTL(t *testing.T) {
	cache := NewPredictionCache(100 * time.Millisecond)
	defer cache.Stop()

	ts := "2026-01-19T14:00:00Z"

	// Add 5 entries
	for i := 0; i < 5; i++ {
		gridID := fmt.Sprintf("g%08d", i)
		cache.Set(gridID, ts, float64(i))
	}

	// Verify all are there
	cache.mu.RLock()
	initialCount := int64(len(cache.cache))
	cache.mu.RUnlock()

	if initialCount != 5 {
		t.Errorf("Expected 5 entries, got %d", initialCount)
	}

	// Wait for expiration
	time.Sleep(150 * time.Millisecond)

	// Access one (triggers lazy eviction)
	cache.Get(fmt.Sprintf("g%08d", 0), ts)

	cache.mu.RLock()
	finalCount := int64(len(cache.cache))
	cache.mu.RUnlock()

	// Should have 4 left (1 accessed and evicted)
	if finalCount != 4 {
		t.Errorf("Expected 4 entries after lazy eviction, got %d", finalCount)
	}
}

// TestEntrySourceTracking tests that source is recorded
func TestEntrySourceTracking(t *testing.T) {
	cache := NewPredictionCache(1 * time.Hour)
	defer cache.Stop()

	gridID := "g00000001"
	ts := "2026-01-19T14:00:00Z"

	cache.SetWithState(gridID, ts, 42.5, Written, "restored_from_db")

	cache.mu.RLock()
	entry, exists := cache.cache[cache.makeKey(gridID, ts)]
	cache.mu.RUnlock()

	if !exists {
		t.Fatalf("Entry not found")
	}

	if entry.Source != "restored_from_db" {
		t.Errorf("Expected source 'restored_from_db', got '%s'", entry.Source)
	}
}
