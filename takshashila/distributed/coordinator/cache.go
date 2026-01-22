package main

import (
	"sync"
	"time"
)

var globalMetrics *MetricsWithLabels

// CacheKey uniquely identifies a cached prediction
// MUST match DedupKey exactly (no request_id)
type CacheKey struct {
	GridID         string
	TimestampHour  string
	ModelVersion   string
	FeatureVersion string
}

// String returns a stable string representation for map keys
func (ck CacheKey) String() string {
	return ck.GridID + ":" + ck.TimestampHour + ":" + ck.ModelVersion + ":" + ck.FeatureVersion
}

// CacheEntry holds a cached prediction or error with TTL
type CacheEntry struct {
	PM25      float64
	Error     PredictionError // nil for successful results
	CachedAt  time.Time
	ExpiresAt time.Time
}

// IsExpired returns true if the entry has passed its TTL
func (ce *CacheEntry) IsExpired(now time.Time) bool {
	return now.After(ce.ExpiresAt)
}

// IsError returns true if this entry caches an error
func (ce *CacheEntry) IsError() bool {
	return ce.Error != nil
}

// ResultCache is a TTL-based cache for prediction results
// Thread-safe for concurrent reads and writes
type ResultCache struct {
	entries map[string]*CacheEntry
	mu      sync.RWMutex
	ttl     time.Duration
}

// NewResultCache creates a new cache with the specified TTL
func NewResultCache(ttl time.Duration) *ResultCache {
	return &ResultCache{
		entries: make(map[string]*CacheEntry),
		ttl:     ttl,
	}
}

// Get retrieves a cached value if present and not expired
// Returns (pm25, error, true) if cache hit (error may be nil for success)
// Returns (0, nil, false) if cache miss or expired
func (rc *ResultCache) Get(key CacheKey) (float64, PredictionError, bool) {
	start := time.Now()
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	keyStr := key.String()
	entry, exists := rc.entries[keyStr]

	if !exists {
		// Cache miss - record in counters only (not in latency histograms)
		if globalMetrics != nil {
			globalMetrics.CacheMissesTotal.WithLabelValues(CacheResultMiss).Inc()
		}
		return 0, nil, false
	}

	// Check expiration
	if entry.IsExpired(time.Now()) {
		// Expired - treat as miss
		// Note: We don't delete here to avoid write lock
		// Cleanup happens lazily in Set() or via periodic sweep
		if globalMetrics != nil {
			globalMetrics.CacheMissesTotal.WithLabelValues(CacheResultMiss).Inc()
		}
		return 0, nil, false
	}

	// Cache hit - record metrics
	if globalMetrics != nil {
		elapsed := time.Since(start).Seconds()
		// Record in cache_hit_latency_seconds histogram (success-only)
		globalMetrics.CacheHitLatencySeconds.Observe(elapsed)
		// Record in cache_hits_total counter
		globalMetrics.CacheHitsTotal.WithLabelValues(CacheResultHit).Inc()
	}

	return entry.PM25, entry.Error, true
}

// Set stores a successful result in the cache with TTL
func (rc *ResultCache) Set(key CacheKey, pm25 float64) {
	rc.SetWithTTL(key, pm25, nil, rc.ttl)
}

// SetError stores an error in the cache with custom TTL
// Only cacheable errors are stored (based on error.Cacheable())
func (rc *ResultCache) SetError(key CacheKey, err PredictionError) {
	if err == nil || !err.Cacheable() {
		return // Don't cache nil or non-cacheable errors
	}

	rc.SetWithTTL(key, 0, err, err.CacheTTL())
}

// SetWithTTL stores a value or error with custom TTL (internal method)
func (rc *ResultCache) SetWithTTL(key CacheKey, pm25 float64, err PredictionError, ttl time.Duration) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	now := time.Now()
	keyStr := key.String()

	entry := &CacheEntry{
		PM25:      pm25,
		Error:     err,
		CachedAt:  now,
		ExpiresAt: now.Add(ttl),
	}

	rc.entries[keyStr] = entry
}

// Invalidate removes a specific entry from the cache
// Used when we know data is stale (e.g., feature freshness violation)
func (rc *ResultCache) Invalidate(key CacheKey) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	keyStr := key.String()
	if _, exists := rc.entries[keyStr]; exists {
		delete(rc.entries, keyStr)
		// Record eviction
		if globalMetrics != nil {
			globalMetrics.CacheEvictionTotal.WithLabelValues(EvictionReasonManualInvalidate).Inc()
		}
	}
}

// EvictExpired removes all expired entries (for periodic cleanup)
// Returns number of entries evicted
func (rc *ResultCache) EvictExpired() int {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	now := time.Now()
	evicted := 0

	for keyStr, entry := range rc.entries {
		if entry.IsExpired(now) {
			delete(rc.entries, keyStr)
			evicted++
			// Record eviction
			if globalMetrics != nil {
				globalMetrics.CacheEvictionTotal.WithLabelValues(EvictionReasonTTLExpired).Inc()
			}
		}
	}

	return evicted
}

// Clear removes all entries from the cache
func (rc *ResultCache) Clear() {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	rc.entries = make(map[string]*CacheEntry)
}

// Stats returns current cache statistics (for observability)
func (rc *ResultCache) Stats() CacheStats {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	now := time.Now()
	stats := CacheStats{
		TotalEntries:   len(rc.entries),
		ExpiredEntries: 0,
	}

	for _, entry := range rc.entries {
		if entry.IsExpired(now) {
			stats.ExpiredEntries++
		}
	}

	// Update gauge metric
	if globalMetrics != nil {
		globalMetrics.CacheEntriesGauge.Set(float64(stats.TotalEntries))
	}

	return stats
}

// CacheStats holds cache metrics
type CacheStats struct {
	TotalEntries   int
	ExpiredEntries int
}
