package main

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Test 1: Leader creation - first request becomes leader
func TestDedupLeaderCreation(t *testing.T) {
	dedup := NewDeduplicator()
	ctx := context.Background()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	entry, isLeader := dedup.AcquireLeader(ctx, key, "req-001")

	if !isLeader {
		t.Fatal("Expected first request to become leader")
	}

	if entry.LeaderRequestID != "req-001" {
		t.Errorf("Expected LeaderRequestID=req-001, got %s", entry.LeaderRequestID)
	}

	if entry.FollowerCount != 0 {
		t.Errorf("Expected FollowerCount=0, got %d", entry.FollowerCount)
	}

	// Verify entry is in map
	stats := dedup.Stats()
	if stats.ActiveComputations != 1 {
		t.Errorf("Expected 1 active computation, got %d", stats.ActiveComputations)
	}
}

// Test 2: Followers joining - subsequent requests become followers
func TestDedupFollowersJoining(t *testing.T) {
	dedup := NewDeduplicator()
	ctx := context.Background()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// First request - leader
	entry1, isLeader1 := dedup.AcquireLeader(ctx, key, "req-001")
	if !isLeader1 {
		t.Fatal("Expected first request to become leader")
	}

	// Second request - follower
	entry2, isLeader2 := dedup.AcquireLeader(ctx, key, "req-002")
	if isLeader2 {
		t.Fatal("Expected second request to be follower")
	}

	// Third request - follower
	entry3, isLeader3 := dedup.AcquireLeader(ctx, key, "req-003")
	if isLeader3 {
		t.Fatal("Expected third request to be follower")
	}

	// All entries should point to same dedup entry
	if entry1 != entry2 || entry2 != entry3 {
		t.Fatal("Expected all entries to point to same dedup entry")
	}

	// Verify follower count
	entry1.mu.Lock()
	followerCount := entry1.FollowerCount
	entry1.mu.Unlock()

	if followerCount != 2 {
		t.Errorf("Expected 2 followers, got %d", followerCount)
	}

	// Verify stats
	stats := dedup.Stats()
	if stats.ActiveComputations != 1 {
		t.Errorf("Expected 1 active computation, got %d", stats.ActiveComputations)
	}
	if stats.TotalFollowers != 2 {
		t.Errorf("Expected 2 total followers, got %d", stats.TotalFollowers)
	}
}

// Test 3: Cleanup after leader completion
func TestDedupCleanupAfterCompletion(t *testing.T) {
	dedup := NewDeduplicator()
	ctx := context.Background()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Acquire leader
	entry, isLeader := dedup.AcquireLeader(ctx, key, "req-001")
	if !isLeader {
		t.Fatal("Expected to be leader")
	}

	// Verify entry exists
	stats := dedup.Stats()
	if stats.ActiveComputations != 1 {
		t.Fatalf("Expected 1 active computation before completion, got %d", stats.ActiveComputations)
	}

	// Complete computation
	result := &DedupResult{
		PM25:  42.5,
		Error: nil,
	}
	dedup.Complete(key, result)

	// Verify immediate cleanup
	stats = dedup.Stats()
	if stats.ActiveComputations != 0 {
		t.Errorf("Expected 0 active computations after completion, got %d", stats.ActiveComputations)
	}

	// Verify done channel is closed
	select {
	case <-entry.Done:
		// Expected - channel should be closed
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected Done channel to be closed")
	}

	// Verify result is set
	entry.mu.Lock()
	if entry.Result == nil {
		t.Fatal("Expected result to be set")
	}
	if entry.Result.PM25 != 42.5 {
		t.Errorf("Expected PM25=42.5, got %.2f", entry.Result.PM25)
	}
	entry.mu.Unlock()
}

// Test 4: Follower wait with successful result
func TestDedupFollowerWaitSuccess(t *testing.T) {
	dedup := NewDeduplicator()
	ctx := context.Background()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Leader acquires
	_, isLeader := dedup.AcquireLeader(ctx, key, "req-001")
	if !isLeader {
		t.Fatal("Expected to be leader")
	}

	// Follower acquires
	entry, isFollower := dedup.AcquireLeader(ctx, key, "req-002")
	if isFollower {
		t.Fatal("Expected to be follower")
	}

	// Follower waits in background
	done := make(chan struct{})
	var followerResult *DedupResult
	var followerErr error

	go func() {
		followerResult, followerErr = dedup.WaitForResult(ctx, entry)
		close(done)
	}()

	// Leader completes after 100ms
	time.Sleep(100 * time.Millisecond)
	result := &DedupResult{
		PM25:  65.3,
		Error: nil,
	}
	dedup.Complete(key, result)

	// Wait for follower to receive result
	select {
	case <-done:
		// Expected
	case <-time.After(1 * time.Second):
		t.Fatal("Follower did not receive result in time")
	}

	// Verify follower got correct result
	if followerErr != nil {
		t.Fatalf("Expected no error, got %v", followerErr)
	}
	if followerResult.PM25 != 65.3 {
		t.Errorf("Expected PM25=65.3, got %.2f", followerResult.PM25)
	}
}

// Test 5: Follower wait with error result
func TestDedupFollowerWaitError(t *testing.T) {
	dedup := NewDeduplicator()
	ctx := context.Background()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Leader acquires
	_, isLeader := dedup.AcquireLeader(ctx, key, "req-001")
	if !isLeader {
		t.Fatal("Expected to be leader")
	}

	// Follower acquires
	entry, isFollower := dedup.AcquireLeader(ctx, key, "req-002")
	if isFollower {
		t.Fatal("Expected to be follower")
	}

	// Follower waits in background
	done := make(chan struct{})
	var followerErr error

	go func() {
		_, followerErr = dedup.WaitForResult(ctx, entry)
		close(done)
	}()

	// Leader completes with error
	time.Sleep(50 * time.Millisecond)
	result := &DedupResult{
		PM25:  0,
		Error: fmt.Errorf("computation failed: grid not found"),
	}
	dedup.Complete(key, result)

	// Wait for follower
	<-done

	// Verify follower got error
	if followerErr == nil {
		t.Fatal("Expected error from follower")
	}
	if followerErr.Error() != "computation failed: grid not found" {
		t.Errorf("Expected specific error, got %v", followerErr)
	}
}

// Test 6: Follower timeout
func TestDedupFollowerTimeout(t *testing.T) {
	dedup := NewDeduplicator()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Leader acquires
	_, isLeader := dedup.AcquireLeader(context.Background(), key, "req-001")
	if !isLeader {
		t.Fatal("Expected to be leader")
	}

	// Follower acquires with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	entry, isFollower := dedup.AcquireLeader(ctx, key, "req-002")
	if isFollower {
		t.Fatal("Expected to be follower")
	}

	// Follower waits (leader never completes)
	_, err := dedup.WaitForResult(ctx, entry)

	// Verify timeout error
	if err != context.DeadlineExceeded {
		t.Errorf("Expected context.DeadlineExceeded, got %v", err)
	}
}

// Test 7: Concurrent dedup with burst of identical requests
func TestDedupConcurrentBurst(t *testing.T) {
	dedup := NewDeduplicator()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	numRequests := 100
	var leaderCount int32
	var followerCount int32
	var wg sync.WaitGroup

	// Launch 100 concurrent requests
	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(reqID int) {
			defer wg.Done()

			ctx := context.Background()
			requestID := fmt.Sprintf("req-%03d", reqID)

			entry, isLeader := dedup.AcquireLeader(ctx, key, requestID)

			if isLeader {
				atomic.AddInt32(&leaderCount, 1)

				// Simulate computation
				time.Sleep(50 * time.Millisecond)

				// Complete
				result := &DedupResult{
					PM25:  100.0,
					Error: nil,
				}
				dedup.Complete(key, result)
			} else {
				atomic.AddInt32(&followerCount, 1)

				// Wait for leader
				_, err := dedup.WaitForResult(ctx, entry)
				if err != nil {
					t.Errorf("Follower %s failed: %v", requestID, err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify exactly 1 leader
	if leaderCount != 1 {
		t.Errorf("Expected exactly 1 leader, got %d", leaderCount)
	}

	// Verify 99 followers
	if followerCount != 99 {
		t.Errorf("Expected 99 followers, got %d", followerCount)
	}

	// Verify cleanup happened
	stats := dedup.Stats()
	if stats.ActiveComputations != 0 {
		t.Errorf("Expected 0 active computations after burst, got %d", stats.ActiveComputations)
	}
}
