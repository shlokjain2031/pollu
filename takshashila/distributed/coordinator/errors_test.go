package main

import (
	"context"
	"testing"
	"time"
)

// Test 1: InvalidRequest error properties
func TestErrorInvalidRequest(t *testing.T) {
	err := NewInvalidRequestError("missing grid_id parameter")

	if err.Type() != ErrorTypeInvalidRequest {
		t.Errorf("Expected type InvalidRequest, got %v", err.Type())
	}

	if err.HTTPStatus() != 400 {
		t.Errorf("Expected HTTP 400, got %d", err.HTTPStatus())
	}

	if err.Retryable() {
		t.Error("InvalidRequest should not be retryable")
	}

	if err.Cacheable() {
		t.Error("InvalidRequest should not be cacheable")
	}

	if err.Error() != "missing grid_id parameter" {
		t.Errorf("Unexpected error message: %s", err.Error())
	}
}

// Test 2: NotInCoverage error properties
func TestErrorNotInCoverage(t *testing.T) {
	err := NewNotInCoverageError("g99999999")

	if err.Type() != ErrorTypeNotInCoverage {
		t.Errorf("Expected type NotInCoverage, got %v", err.Type())
	}

	if err.HTTPStatus() != 404 {
		t.Errorf("Expected HTTP 404, got %d", err.HTTPStatus())
	}

	if err.Retryable() {
		t.Error("NotInCoverage should not be retryable")
	}

	if !err.Cacheable() {
		t.Error("NotInCoverage should be cacheable")
	}

	if err.CacheTTL() != 5*time.Minute {
		t.Errorf("Expected 5 minute TTL, got %v", err.CacheTTL())
	}
}

// Test 3: DataMissing error properties
func TestErrorDataMissing(t *testing.T) {
	err := NewDataMissingError("PM2.5 observations not available for grid g00000001")

	if err.Type() != ErrorTypeDataMissing {
		t.Errorf("Expected type DataMissing, got %v", err.Type())
	}

	if err.HTTPStatus() != 503 {
		t.Errorf("Expected HTTP 503, got %d", err.HTTPStatus())
	}

	if !err.Retryable() {
		t.Error("DataMissing should be retryable")
	}

	if !err.Cacheable() {
		t.Error("DataMissing should be cacheable")
	}

	if err.CacheTTL() != 45*time.Second {
		t.Errorf("Expected 45 second TTL, got %v", err.CacheTTL())
	}
}

// Test 4: ComputationTimeout error properties
func TestErrorComputationTimeout(t *testing.T) {
	err := NewComputationTimeoutError("Computation timed out after 1 second")

	if err.Type() != ErrorTypeComputationTimeout {
		t.Errorf("Expected type ComputationTimeout, got %v", err.Type())
	}

	if err.HTTPStatus() != 504 {
		t.Errorf("Expected HTTP 504, got %d", err.HTTPStatus())
	}

	if !err.Retryable() {
		t.Error("ComputationTimeout should be retryable")
	}

	if !err.Cacheable() {
		t.Error("ComputationTimeout should be cacheable")
	}

	if err.CacheTTL() != 5*time.Second {
		t.Errorf("Expected 5 second TTL, got %v", err.CacheTTL())
	}
}

// Test 5: WorkerError properties
func TestErrorWorkerError(t *testing.T) {
	err := NewWorkerError("Database connection failed")

	if err.Type() != ErrorTypeWorkerError {
		t.Errorf("Expected type WorkerError, got %v", err.Type())
	}

	if err.HTTPStatus() != 500 {
		t.Errorf("Expected HTTP 500, got %d", err.HTTPStatus())
	}

	if !err.Retryable() {
		t.Error("WorkerError should be retryable")
	}

	if err.Cacheable() {
		t.Error("WorkerError should not be cacheable")
	}
}

// Test 6: Unknown error properties
func TestErrorUnknown(t *testing.T) {
	err := NewUnknownError("Unexpected error occurred")

	if err.Type() != ErrorTypeUnknown {
		t.Errorf("Expected type Unknown, got %v", err.Type())
	}

	if err.HTTPStatus() != 500 {
		t.Errorf("Expected HTTP 500, got %d", err.HTTPStatus())
	}

	if !err.Retryable() {
		t.Error("Unknown should be retryable")
	}

	if err.Cacheable() {
		t.Error("Unknown should not be cacheable")
	}
}

// Test 7: ClassifyError for context.DeadlineExceeded
func TestClassifyErrorTimeout(t *testing.T) {
	err := context.DeadlineExceeded

	classified := ClassifyError(err)

	if classified.Type() != ErrorTypeComputationTimeout {
		t.Errorf("Expected ComputationTimeout, got %v", classified.Type())
	}

	if classified.Error() != "Computation timed out after 1 second" {
		t.Errorf("Unexpected error message: %s", classified.Error())
	}
}

// Test 8: ClassifyError for already classified error
func TestClassifyErrorPreserves(t *testing.T) {
	original := NewNotInCoverageError("g00000001")

	classified := ClassifyError(original)

	if classified != original {
		t.Error("ClassifyError should preserve PredictionError as-is")
	}

	if classified.Type() != ErrorTypeNotInCoverage {
		t.Errorf("Expected NotInCoverage, got %v", classified.Type())
	}
}

// Test 9: ErrorResponse structure for retryable error
func TestErrorResponseRetryable(t *testing.T) {
	err := NewDataMissingError("PM2.5 data missing")

	resp := NewErrorResponse(err)

	if resp.ErrorType != "DataMissing" {
		t.Errorf("Expected error_type=DataMissing, got %s", resp.ErrorType)
	}

	if resp.HTTPStatus != 503 {
		t.Errorf("Expected http_status=503, got %d", resp.HTTPStatus)
	}

	if !resp.Retryable {
		t.Error("Expected retryable=true")
	}

	if resp.RetryAfter == nil {
		t.Fatal("Expected retry_after to be set")
	}

	if *resp.RetryAfter != 45 {
		t.Errorf("Expected retry_after=45s, got %d", *resp.RetryAfter)
	}

	if resp.TTL == nil {
		t.Fatal("Expected ttl to be set")
	}

	if *resp.TTL != 45 {
		t.Errorf("Expected ttl=45s, got %d", *resp.TTL)
	}
}

// Test 10: ErrorResponse structure for non-retryable error
func TestErrorResponseNonRetryable(t *testing.T) {
	err := NewNotInCoverageError("g99999999")

	resp := NewErrorResponse(err)

	if resp.ErrorType != "NotInCoverage" {
		t.Errorf("Expected error_type=NotInCoverage, got %s", resp.ErrorType)
	}

	if resp.HTTPStatus != 404 {
		t.Errorf("Expected http_status=404, got %d", resp.HTTPStatus)
	}

	if resp.Retryable {
		t.Error("Expected retryable=false")
	}

	// NotInCoverage is cacheable but non-retryable, so no retry_after
	if resp.RetryAfter != nil {
		t.Error("Expected retry_after to be nil for non-retryable error")
	}
}

// Test 11: Cache stores NotInCoverage with 5 minute TTL
func TestCacheErrorNotInCoverage(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	key := CacheKey{
		Location:       "g99999999",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	err := NewNotInCoverageError("g99999999")
	cache.SetError(key, err)

	// Retrieve from cache
	pm25, cachedErr, hit := cache.Get(key)

	if !hit {
		t.Fatal("Expected cache hit for NotInCoverage error")
	}

	if pm25 != 0 {
		t.Errorf("Expected pm25=0 for error, got %.2f", pm25)
	}

	if cachedErr == nil {
		t.Fatal("Expected cached error, got nil")
	}

	if cachedErr.Type() != ErrorTypeNotInCoverage {
		t.Errorf("Expected NotInCoverage error, got %v", cachedErr.Type())
	}

	// Verify TTL is 5 minutes
	if cachedErr.CacheTTL() != 5*time.Minute {
		t.Errorf("Expected 5 minute TTL, got %v", cachedErr.CacheTTL())
	}
}

// Test 12: Cache does NOT store WorkerError
func TestCacheErrorNotCached(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	key := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Try to cache WorkerError (not cacheable)
	err := NewWorkerError("Database connection failed")
	cache.SetError(key, err)

	// Verify it was NOT cached
	_, _, hit := cache.Get(key)

	if hit {
		t.Error("Expected cache miss - WorkerError should not be cached")
	}

	stats := cache.Stats()
	if stats.TotalEntries != 0 {
		t.Errorf("Expected 0 cache entries, got %d", stats.TotalEntries)
	}
}

// Test 13: Dedup propagates error to followers
func TestDedupErrorPropagation(t *testing.T) {
	dedup := NewDeduplicator()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	ctx := context.Background()

	// Leader acquires
	_, isLeader1 := dedup.AcquireLeader(ctx, key, "req-leader")
	if !isLeader1 {
		t.Fatal("Expected first request to be leader")
	}

	// Follower joins
	entry2, isLeader2 := dedup.AcquireLeader(ctx, key, "req-follower")
	if isLeader2 {
		t.Fatal("Expected second request to be follower")
	}

	// Leader completes with error
	testErr := NewDataMissingError("PM2.5 data missing")
	go func() {
		time.Sleep(50 * time.Millisecond)
		dedup.Complete(key, &DedupResult{PM25: 0, Error: testErr})
	}()

	// Follower waits
	result, err := dedup.WaitForResult(ctx, entry2)

	// WaitForResult returns (result, result.Error) when leader completes with error
	if err == nil {
		t.Fatal("Expected error from WaitForResult, got nil")
	}

	if result == nil {
		t.Fatal("Expected result object, got nil")
	}

	if result.Error == nil {
		t.Fatal("Expected error in result, got nil")
	}

	// Verify error type matches
	predErr, ok := result.Error.(PredictionError)
	if !ok {
		t.Fatalf("Expected PredictionError, got %T", result.Error)
	}

	if predErr.Type() != ErrorTypeDataMissing {
		t.Errorf("Expected DataMissing error, got %v", predErr.Type())
	}

	// Verify dedup entry is cleaned up
	stats := dedup.Stats()
	if stats.ActiveComputations != 0 {
		t.Errorf("Expected 0 active computations after completion, got %d", stats.ActiveComputations)
	}
}

// Test 14: Timeout error converts to ComputationTimeout
func TestTimeoutClassification(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	<-ctx.Done()

	predErr := ClassifyError(ctx.Err())

	if predErr.Type() != ErrorTypeComputationTimeout {
		t.Errorf("Expected ComputationTimeout, got %v", predErr.Type())
	}

	if predErr.HTTPStatus() != 504 {
		t.Errorf("Expected HTTP 504, got %d", predErr.HTTPStatus())
	}
}

// Test 15: DataMissing error cached with 45s TTL
func TestDataMissingCacheTTL(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	key := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	err := NewDataMissingError("PM2.5 data missing")
	cache.SetError(key, err)

	// Get should hit
	_, cachedErr, hit := cache.Get(key)
	if !hit {
		t.Fatal("Expected cache hit")
	}

	if cachedErr == nil {
		t.Fatal("Expected cached error, got nil")
	}

	if cachedErr.CacheTTL() != 45*time.Second {
		t.Errorf("Expected 45s TTL, got %v", cachedErr.CacheTTL())
	}

	// Wait for expiry
	time.Sleep(100 * time.Millisecond) // Longer than 45s would be too slow for tests

	// For this test, we just verify the TTL property
	// Full expiry testing is covered in cache_test.go
}

// Test 16: ComputationTimeout cached with 5s TTL
func TestComputationTimeoutCacheTTL(t *testing.T) {
	cache := NewResultCache(5 * time.Minute)

	key := CacheKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	err := NewComputationTimeoutError("Computation timed out")
	cache.SetError(key, err)

	// Get should hit
	_, cachedErr, hit := cache.Get(key)
	if !hit {
		t.Fatal("Expected cache hit")
	}

	if cachedErr.CacheTTL() != 5*time.Second {
		t.Errorf("Expected 5s TTL, got %v", cachedErr.CacheTTL())
	}
}

// Test 17: Leader completes with error, follower receives error immediately
func TestLeaderTimeoutUnblocksFollowers(t *testing.T) {
	dedup := NewDeduplicator()

	key := DedupKey{
		Location:       "g00000001",
		TimestampHour:  "2026-01-21T12:00:00Z",
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	ctx := context.Background()

	// Leader acquires
	_, isLeader := dedup.AcquireLeader(ctx, key, "req-leader")
	if !isLeader {
		t.Fatal("Expected first request to be leader")
	}

	// Follower acquires (joins existing entry)
	entry, isLeader2 := dedup.AcquireLeader(ctx, key, "req-follower")
	if isLeader2 {
		t.Fatal("Expected second request to be follower, got leader")
	}

	// Start follower wait
	waitDone := make(chan struct{})
	var waitErr error
	var waitResult *DedupResult

	go func() {
		waitResult, waitErr = dedup.WaitForResult(ctx, entry)
		close(waitDone)
	}()

	// Let follower start waiting
	time.Sleep(20 * time.Millisecond)

	// Leader completes with timeout error
	timeoutErr := NewComputationTimeoutError("Computation timed out")
	dedup.Complete(key, &DedupResult{PM25: 0, Error: timeoutErr})

	// Wait for follower to unblock
	select {
	case <-waitDone:
		// Follower unblocked successfully
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Follower did not unblock within 200ms")
	}

	// Verify follower received the error
	if waitErr == nil {
		t.Fatal("Expected error from WaitForResult")
	}

	if waitResult == nil {
		t.Fatal("Expected result object")
	}

	predErr, ok := waitResult.Error.(PredictionError)
	if !ok {
		t.Fatalf("Expected PredictionError, got %T", waitResult.Error)
	}

	if predErr.Type() != ErrorTypeComputationTimeout {
		t.Errorf("Expected ComputationTimeout, got %v", predErr.Type())
	}
}
