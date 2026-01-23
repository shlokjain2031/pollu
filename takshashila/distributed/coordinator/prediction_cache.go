package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// WriteState tracks the lifecycle of a cached prediction's DB write
type WriteState int

const (
	NotWritten  WriteState = iota // Cached but not written yet
	Writing                       // Async write in progress
	Written                       // Successfully written to DB
	WriteFailed                   // Write failed, queued for retry
)

func (ws WriteState) String() string {
	switch ws {
	case NotWritten:
		return "NotWritten"
	case Writing:
		return "Writing"
	case Written:
		return "Written"
	case WriteFailed:
		return "WriteFailed"
	default:
		return "Unknown"
	}
}

// PredictionCacheEntry tracks a cached prediction with write state
type PredictionCacheEntry struct {
	Value       float64
	CachedAt    time.Time
	WriteState  WriteState
	Source      string // "computed", "restored_from_db", etc.
	Attempts    int    // Number of write attempts
	LastAttempt time.Time
}

// LeaderResult is sent over a channel to wake up followers
type LeaderResult struct {
	Value float64
	Error error
}

// PredictionCache provides in-memory caching with async DB writes
type PredictionCache struct {
	mu          sync.RWMutex
	cache       map[string]*PredictionCacheEntry
	ttl         time.Duration
	maxSize     int64 // Maximum memory in bytes
	retryQueue  chan *RetryEntry
	stopCleanup chan struct{}
	stopRetry   chan struct{}

	// Metrics
	hits           int64
	misses         int64
	evictions      int64
	writeSuccesses int64
	writeFailures  int64
}

// RetryEntry tracks a failed write for retry
type RetryEntry struct {
	GridID        string
	TimestampHour string
	Value         float64
	Attempts      int
	NextRetry     time.Time
}

// NewPredictionCache creates a new prediction cache with 2-hour TTL
func NewPredictionCache(ttl time.Duration) *PredictionCache {
	pc := &PredictionCache{
		cache:       make(map[string]*PredictionCacheEntry),
		ttl:         ttl,
		maxSize:     1024 * 1024 * 1024, // 1GB limit
		retryQueue:  make(chan *RetryEntry, 10000),
		stopCleanup: make(chan struct{}),
		stopRetry:   make(chan struct{}),
	}

	// Start periodic cleanup goroutine (every 5 minutes)
	go pc.periodicCleanup()

	// Start retry queue processor
	go pc.processRetryQueue()

	return pc
}

// Stop shuts down background goroutines
func (pc *PredictionCache) Stop() {
	close(pc.stopCleanup)
	close(pc.stopRetry)
}

// Get retrieves a cached prediction if available and not expired
func (pc *PredictionCache) Get(gridID, timestampHour string) (float64, bool) {
	key := pc.makeKey(gridID, timestampHour)

	pc.mu.RLock()
	entry, exists := pc.cache[key]
	pc.mu.RUnlock()

	if !exists {
		pc.recordMiss()
		return 0, false
	}

	// Check TTL (lazy eviction)
	if time.Since(entry.CachedAt) > pc.ttl {
		pc.evict(key)
		pc.recordMiss()
		return 0, false
	}

	pc.recordHit()
	return entry.Value, true
}

// Set caches a prediction and triggers async DB write
// Implements first-write-wins: only caches if key doesn't exist
func (pc *PredictionCache) Set(gridID, timestampHour string, value float64) {
	key := pc.makeKey(gridID, timestampHour)

	pc.mu.Lock()
	// Check if entry already exists (first-write-wins)
	if _, exists := pc.cache[key]; exists {
		pc.mu.Unlock()
		// Already cached by another goroutine, don't overwrite
		return
	}

	entry := &PredictionCacheEntry{
		Value:      value,
		CachedAt:   time.Now(),
		WriteState: NotWritten,
		Source:     "computed",
	}

	pc.cache[key] = entry
	pc.mu.Unlock()

	// Trigger async write (does not block response)
	go pc.asyncWriteToDB(gridID, timestampHour, value, key)
}

// SetWithState allows setting specific write state (for testing/restoration)
func (pc *PredictionCache) SetWithState(gridID, timestampHour string, value float64, state WriteState, source string) {
	key := pc.makeKey(gridID, timestampHour)

	entry := &PredictionCacheEntry{
		Value:      value,
		CachedAt:   time.Now(),
		WriteState: state,
		Source:     source,
	}

	pc.mu.Lock()
	pc.cache[key] = entry
	pc.mu.Unlock()
}

// asyncWriteToDB writes prediction to DuckDB in background
func (pc *PredictionCache) asyncWriteToDB(gridID, timestampHour string, value float64, key string) {
	startTime := time.Now()

	// Mark as Writing
	pc.mu.Lock()
	if entry, exists := pc.cache[key]; exists {
		entry.WriteState = Writing
		entry.Attempts++
		entry.LastAttempt = time.Now()
	}
	pc.mu.Unlock()

	// Simulate DB write (replace with actual persistence layer call)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// TODO: Replace with actual DB write
	// err := persistence.WritePrediction(ctx, gridID, timestampHour, value)
	_ = ctx
	err := pc.simulateDBWrite(gridID, timestampHour, value)

	// Record latency
	if globalMetrics != nil {
		globalMetrics.CacheWriteLatencySeconds.Observe(time.Since(startTime).Seconds())
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	entry, exists := pc.cache[key]
	if !exists {
		return // Entry was evicted during write
	}

	if err == nil {
		entry.WriteState = Written
		pc.writeSuccesses++

		if globalMetrics != nil {
			globalMetrics.CacheWriteSuccessTotal.Inc()
		}

		log.Printf("[CACHE] Successfully wrote to DB: %s (attempts=%d)", key, entry.Attempts)
	} else {
		entry.WriteState = WriteFailed
		pc.writeFailures++

		if globalMetrics != nil {
			globalMetrics.CacheWriteFailureTotal.Inc()
		}

		// Queue for retry if under max attempts
		if entry.Attempts < 5 {
			delay := pc.calculateRetryDelay(entry.Attempts)
			retryEntry := &RetryEntry{
				GridID:        gridID,
				TimestampHour: timestampHour,
				Value:         value,
				Attempts:      entry.Attempts,
				NextRetry:     time.Now().Add(delay),
			}

			select {
			case pc.retryQueue <- retryEntry:
				if globalMetrics != nil {
					globalMetrics.CacheWriteRetryTotal.Inc()
				}
				log.Printf("[CACHE] Queued retry for %s (attempt %d, delay=%v): %v",
					key, entry.Attempts, delay, err)
			default:
				log.Printf("[CACHE] Retry queue full, dropping %s: %v", key, err)
			}
		} else {
			log.Printf("[CACHE] Max retry attempts exceeded for %s: %v", key, err)
		}
	}
}

// calculateRetryDelay returns exponential backoff capped at 5 minutes
func (pc *PredictionCache) calculateRetryDelay(attempts int) time.Duration {
	// 30s, 60s, 120s, 240s, 300s (capped)
	delay := 30 * time.Second
	for i := 1; i < attempts && i < 5; i++ {
		delay *= 2
	}

	maxDelay := 5 * time.Minute
	if delay > maxDelay {
		delay = maxDelay
	}

	return delay
}

// processRetryQueue processes failed writes with retry logic
func (pc *PredictionCache) processRetryQueue() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	var pending []*RetryEntry

	for {
		select {
		case <-pc.stopRetry:
			return

		case entry := <-pc.retryQueue:
			pending = append(pending, entry)

		case <-ticker.C:
			now := time.Now()
			var stillPending []*RetryEntry

			for _, entry := range pending {
				if now.After(entry.NextRetry) {
					// Retry the write
					go pc.asyncWriteToDB(entry.GridID, entry.TimestampHour, entry.Value,
						pc.makeKey(entry.GridID, entry.TimestampHour))
				} else {
					stillPending = append(stillPending, entry)
				}
			}

			pending = stillPending
		}
	}
}

// simulateDBWrite simulates a DB write (replace with actual implementation)
func (pc *PredictionCache) simulateDBWrite(gridID, timestampHour string, value float64) error {
	// TODO: Replace with actual PersistenceLayer call
	// For now, simulate success
	time.Sleep(10 * time.Millisecond)
	return nil
}

// periodicCleanup runs every 5 minutes to evict expired entries
func (pc *PredictionCache) periodicCleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-pc.stopCleanup:
			return
		case <-ticker.C:
			pc.runCleanup()
		}
	}
}

// runCleanup evicts expired entries and reports aggregate metrics
func (pc *PredictionCache) runCleanup() {
	now := time.Now()
	evictedCount := 0
	bytesFreed := int64(0)

	pc.mu.Lock()
	defer pc.mu.Unlock()

	for key, entry := range pc.cache {
		if now.Sub(entry.CachedAt) > pc.ttl {
			delete(pc.cache, key)
			evictedCount++
			bytesFreed += pc.estimateEntrySize(entry)
		}
	}

	pc.evictions += int64(evictedCount)

	if evictedCount > 0 {
		log.Printf("[CACHE] Periodic cleanup: evicted %d entries, freed %d bytes",
			evictedCount, bytesFreed)

		if globalMetrics != nil {
			globalMetrics.CacheEvictions.Add(float64(evictedCount))
		}
	}
}

// evict removes a single entry (called during lazy eviction)
func (pc *PredictionCache) evict(key string) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	delete(pc.cache, key)
	pc.evictions++

	if globalMetrics != nil {
		globalMetrics.CacheEvictions.Inc()
	}
}

// estimateEntrySize estimates memory used by a cache entry
func (pc *PredictionCache) estimateEntrySize(entry *PredictionCacheEntry) int64 {
	// Key (40) + Value (8) + CachedAt (24) + WriteState (8) + Source (16) + overhead (40)
	return 136
}

// makeKey creates a cache key from grid_id and timestamp
func (pc *PredictionCache) makeKey(gridID, timestampHour string) string {
	return fmt.Sprintf("%s:%s", gridID, timestampHour)
}

// recordHit increments hit counter
func (pc *PredictionCache) recordHit() {
	pc.mu.Lock()
	pc.hits++
	pc.mu.Unlock()

	if globalMetrics != nil {
		globalMetrics.CacheHitsTotal.WithLabelValues("prediction").Inc()
	}
}

// recordMiss increments miss counter
func (pc *PredictionCache) recordMiss() {
	pc.mu.Lock()
	pc.misses++
	pc.mu.Unlock()

	if globalMetrics != nil {
		globalMetrics.CacheMissesTotal.WithLabelValues("prediction").Inc()
	}
}

// GetStats returns current cache statistics
func (pc *PredictionCache) GetStats() PredictionCacheStats {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	stats := PredictionCacheStats{
		Hits:           pc.hits,
		Misses:         pc.misses,
		Evictions:      pc.evictions,
		WriteSuccesses: pc.writeSuccesses,
		WriteFailures:  pc.writeFailures,
		EntryCount:     int64(len(pc.cache)),
	}

	if stats.Hits+stats.Misses > 0 {
		stats.HitRate = float64(stats.Hits) / float64(stats.Hits+stats.Misses)
	}

	return stats
}

// PredictionCacheStats holds cache performance metrics
type PredictionCacheStats struct {
	Hits           int64
	Misses         int64
	Evictions      int64
	WriteSuccesses int64
	WriteFailures  int64
	EntryCount     int64
	HitRate        float64
}
