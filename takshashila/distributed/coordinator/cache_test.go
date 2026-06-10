package main

import (
	"sync"
	"testing"
	"time"
)

// Test 1: Cache miss on empty cache
func TestCacheMiss(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	key := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	pm25, cachedErr, hit := cache.Get(key)

	if hit {
		t.Error("Expected cache miss on empty cache")
	}

	if pm25 != 0 {
		t.Errorf("Expected pm25=0 on miss, got %.2f", pm25)
	}

	if cachedErr != nil {
		t.Errorf("Expected nil error on miss, got %v", cachedErr)
	}
}

// Test 2: Cache hit after set
func TestCacheHit(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	key := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Set value
	cache.Set(key, 42.5)

	// Get value
	pm25, cachedErr, hit := cache.Get(key)

	if !hit {
		t.Fatal("Expected cache hit after set")
	}

	if pm25 != 42.5 {
		t.Errorf("Expected pm25=42.5, got %.2f", pm25)
	}

	if cachedErr != nil {
		t.Errorf("Expected nil error for successful result, got %v", cachedErr)
	}
}

// Test 3: Cache expiry after TTL
func TestCacheExpiry(t *testing.T) {
	ttl := 200 * time.Millisecond
	cache := NewResultCache(ttl)

	key := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Set value
	cache.Set(key, 65.3)

	// Immediate get - should hit
	pm25, _, hit := cache.Get(key)
	if !hit {
		t.Fatal("Expected cache hit immediately after set")
	}
	if pm25 != 65.3 {
		t.Errorf("Expected pm25=65.3, got %.2f", pm25)
	}

	// Wait for TTL to expire
	time.Sleep(ttl + 50*time.Millisecond)

	// Get again - should miss due to expiry
	_, _, hit = cache.Get(key)
	if hit {
		t.Error("Expected cache miss after TTL expiry")
	}
}

// Test 4: Concurrent access (read/write race freedom)
func TestCacheConcurrentAccess(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	numGoroutines := 100
	var wg sync.WaitGroup

	// Concurrent writes
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			key := CacheKey{
				Location:       "g00000001",
				TimestampHour:  "2026-01-21T12:00:00Z",
				ModelVersion:   "prod_v1",
				FeatureVersion: "v1",
			}

			cache.Set(key, float64(idx))
		}(i)
	}

	// Concurrent reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			key := CacheKey{
				Location:       "g00000001",
				TimestampHour:  "2026-01-21T12:00:00Z",
				ModelVersion:   "prod_v1",
				FeatureVersion: "v1",
			}

			cache.Get(key)
		}()
	}

	wg.Wait()

	// No race detector failures = success
	// Final value should be from one of the writes
	key := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	_, _, hit := cache.Get(key)
	if !hit {
		t.Error("Expected at least one write to succeed")
	}
}

// Test 5: Evict expired entries
func TestCacheEvictExpired(t *testing.T) {
	ttl := 100 * time.Millisecond
	cache := NewResultCache(ttl)

	// Set 5 entries
	for i := 0; i < 5; i++ {
		key := CacheKey{
			Location:       "g0000000" + string(rune('1'+i)),
			TimestampHour:  "2026-01-21T12:00:00Z",
			ModelVersion:   "prod_v1",
			FeatureVersion: "v1",
		}
		cache.Set(key, float64(i*10))
	}

	// Verify 5 entries
	stats := cache.Stats()
	if stats.TotalEntries != 5 {
		t.Errorf("Expected 5 entries, got %d", stats.TotalEntries)
	}

	// Wait for expiry
	time.Sleep(ttl + 50*time.Millisecond)

	// Evict expired
	evicted := cache.EvictExpired()

	if evicted != 5 {
		t.Errorf("Expected to evict 5 entries, got %d", evicted)
	}

	// Verify cache is empty
	stats = cache.Stats()
	if stats.TotalEntries != 0 {
		t.Errorf("Expected 0 entries after eviction, got %d", stats.TotalEntries)
	}
}

// Test 6: Invalidate specific entry
func TestCacheInvalidate(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	key1 := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	key2 := CacheKey{
		Location:       "g00000002",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Set two entries
	cache.Set(key1, 10.0)
	cache.Set(key2, 20.0)

	// Verify both exist
	stats := cache.Stats()
	if stats.TotalEntries != 2 {
		t.Fatalf("Expected 2 entries, got %d", stats.TotalEntries)
	}

	// Invalidate key1
	cache.Invalidate(key1)

	// Verify key1 is gone, key2 remains
	_, _, hit1 := cache.Get(key1)
	if hit1 {
		t.Error("Expected key1 to be invalidated")
	}

	pm25_2, _, hit2 := cache.Get(key2)
	if !hit2 {
		t.Fatal("Expected key2 to still exist")
	}
	if pm25_2 != 20.0 {
		t.Errorf("Expected key2 pm25=20.0, got %.2f", pm25_2)
	}

	stats = cache.Stats()
	if stats.TotalEntries != 1 {
		t.Errorf("Expected 1 entry after invalidation, got %d", stats.TotalEntries)
	}
}

// Test 7: Clear all entries
func TestCacheClear(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	// Set multiple entries
	for i := 0; i < 10; i++ {
		key := CacheKey{
			Location:       "g0000000" + string(rune('1'+i)),
			TimestampHour:  "2026-01-21T12:00:00Z",
			ModelVersion:   "prod_v1",
			FeatureVersion: "v1",
		}
		cache.Set(key, float64(i*10))
	}

	// Verify 10 entries
	stats := cache.Stats()
	if stats.TotalEntries != 10 {
		t.Errorf("Expected 10 entries, got %d", stats.TotalEntries)
	}

	// Clear cache
	cache.Clear()

	// Verify cache is empty
	stats = cache.Stats()
	if stats.TotalEntries != 0 {
		t.Errorf("Expected 0 entries after clear, got %d", stats.TotalEntries)
	}

	// Verify all keys are gone
	for i := 0; i < 10; i++ {
		key := CacheKey{
			Location:       "g0000000" + string(rune('1'+i)),
			TimestampHour:  "2026-01-21T12:00:00Z",
			ModelVersion:   "prod_v1",
			FeatureVersion: "v1",
		}
		_, _, hit := cache.Get(key)
		if hit {
			t.Errorf("Expected key %d to be cleared", i)
		}
	}
}
