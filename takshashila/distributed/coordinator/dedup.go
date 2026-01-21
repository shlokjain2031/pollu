package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DedupKey uniquely identifies a prediction computation
// Does NOT include request_id (many requests can map to same computation)
type DedupKey struct {
	GridID          string
	TimestampHour   string
	ModelVersion    string
	FeatureVersion  string
}

// String returns a stable string representation for map keys
func (dk DedupKey) String() string {
	return fmt.Sprintf("%s:%s:%s:%s", dk.GridID, dk.TimestampHour, dk.ModelVersion, dk.FeatureVersion)
}

// DedupEntry tracks a single in-flight computation
// Leader creates this entry; followers join and wait on done_chan
type DedupEntry struct {
	Key              DedupKey
	LeaderRequestID  string
	CreatedAt        time.Time
	FollowerCount    int
	Done             chan struct{} // Closed when computation finishes
	Result           *DedupResult  // Set when computation finishes
	mu               sync.Mutex    // Protects FollowerCount and Result
}

// DedupResult holds the outcome of a computation
type DedupResult struct {
	PM25  float64
	Error error
}

// Deduplicator manages in-flight computation deduplication
// Thread-safe for concurrent leader creation and follower joins
type Deduplicator struct {
	entries map[string]*DedupEntry // key = DedupKey.String()
	mu      sync.RWMutex
}

// NewDeduplicator creates a new deduplicator
func NewDeduplicator() *Deduplicator {
	return &Deduplicator{
		entries: make(map[string]*DedupEntry),
	}
}

// AcquireLeader attempts to become the leader for this computation
// Returns (entry, true) if caller is the leader (must execute computation)
// Returns (entry, false) if another leader exists (caller is follower, must wait on entry.Done)
func (d *Deduplicator) AcquireLeader(ctx context.Context, key DedupKey, requestID string) (*DedupEntry, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()

	keyStr := key.String()
	
	// Check if entry already exists
	if entry, exists := d.entries[keyStr]; exists {
		// Another request is already the leader - we're a follower
		entry.mu.Lock()
		entry.FollowerCount++
		entry.mu.Unlock()
		return entry, false
	}

	// We're the first - become the leader
	entry := &DedupEntry{
		Key:             key,
		LeaderRequestID: requestID,
		CreatedAt:       time.Now(),
		FollowerCount:   0,
		Done:            make(chan struct{}),
		Result:          nil,
	}

	d.entries[keyStr] = entry
	return entry, true
}

// Complete marks a computation as finished and wakes all followers
// Caller MUST be the leader (not enforced, but violating this breaks invariants)
// Cleanup is immediate to prevent memory leaks
func (d *Deduplicator) Complete(key DedupKey, result *DedupResult) {
	d.mu.Lock()
	defer d.mu.Unlock()

	keyStr := key.String()
	entry, exists := d.entries[keyStr]
	if !exists {
		// Already cleaned up or never existed - safe to ignore
		return
	}

	// Set result
	entry.mu.Lock()
	entry.Result = result
	entry.mu.Unlock()

	// Wake all followers
	close(entry.Done)

	// Immediate cleanup to avoid leaks
	delete(d.entries, keyStr)
}

// WaitForResult blocks until the leader completes or context times out
// Returns (result, error) where error is:
// - ctx.Err() if timeout
// - result.Error if computation failed
// - nil if computation succeeded
func (d *Deduplicator) WaitForResult(ctx context.Context, entry *DedupEntry) (*DedupResult, error) {
	select {
	case <-entry.Done:
		// Leader finished - return result
		entry.mu.Lock()
		result := entry.Result
		entry.mu.Unlock()
		
		if result == nil {
			// This should never happen if Complete() was called correctly
			return nil, fmt.Errorf("dedup: entry completed without result")
		}
		
		return result, result.Error
		
	case <-ctx.Done():
		// Timeout - follower gives up
		return nil, ctx.Err()
	}
}

// Stats returns current deduplication statistics (for observability)
func (d *Deduplicator) Stats() DedupStats {
	d.mu.RLock()
	defer d.mu.RUnlock()

	stats := DedupStats{
		ActiveComputations: len(d.entries),
		TotalFollowers:     0,
	}

	for _, entry := range d.entries {
		entry.mu.Lock()
		stats.TotalFollowers += entry.FollowerCount
		entry.mu.Unlock()
	}

	return stats
}

// DedupStats holds deduplication metrics
type DedupStats struct {
	ActiveComputations int
	TotalFollowers     int
}
