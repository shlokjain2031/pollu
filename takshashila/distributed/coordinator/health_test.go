package main

import (
	"testing"
	"time"
)

// TestHealthCheckSuccess verifies that a healthy worker responds correctly
func TestHealthCheckSuccess(t *testing.T) {
	// Create health checker with aggressive intervals for testing
	hc := &HealthChecker{
		health:           make(map[string]*WorkerHealth),
		interval:         100 * time.Millisecond,
		timeout:          500 * time.Millisecond,
		failureThreshold: 2,
		stopChan:         make(chan struct{}),
	}

	// Manually initialize worker health
	workerID := "test-worker-0"
	hc.health[workerID] = &WorkerHealth{
		WorkerID:            workerID,
		State:               "Healthy",
		ConsecutiveFailures: 0,
		LastHealthCheckTime: time.Now(),
		LastStateChange:     time.Now(),
	}

	// Record a success
	hc.recordSuccess(workerID, 50*time.Millisecond)

	// Verify state
	if hc.GetWorkerHealth(workerID) != "Healthy" {
		t.Errorf("Expected Healthy, got %s", hc.GetWorkerHealth(workerID))
	}

	// Verify failure counter reset
	hc.mu.RLock()
	failures := hc.health[workerID].ConsecutiveFailures
	hc.mu.RUnlock()

	if failures != 0 {
		t.Errorf("Expected 0 failures after success, got %d", failures)
	}

	t.Log("✓ Test passed: Health check success")
}

// TestHealthCheckTimeout verifies that timeout marks worker unhealthy
func TestHealthCheckTimeout(t *testing.T) {
	hc := &HealthChecker{
		health:           make(map[string]*WorkerHealth),
		interval:         100 * time.Millisecond,
		timeout:          100 * time.Millisecond,
		failureThreshold: 2,
		stopChan:         make(chan struct{}),
	}

	workerID := "test-worker-1"
	hc.health[workerID] = &WorkerHealth{
		WorkerID:            workerID,
		State:               "Healthy",
		ConsecutiveFailures: 0,
		LastHealthCheckTime: time.Now(),
		LastStateChange:     time.Now(),
	}

	// Record first failure
	hc.recordFailure(workerID)
	if hc.GetWorkerHealth(workerID) != "Healthy" {
		t.Errorf("Expected Healthy after 1 failure, got %s", hc.GetWorkerHealth(workerID))
	}

	// Record second failure (should trigger Unhealthy)
	hc.recordFailure(workerID)
	if hc.GetWorkerHealth(workerID) != "Unhealthy" {
		t.Errorf("Expected Unhealthy after 2 failures, got %s", hc.GetWorkerHealth(workerID))
	}

	t.Log("✓ Test passed: Health check timeout")
}

// TestConsecutiveFailureCounter verifies counter semantics
func TestConsecutiveFailureCounter(t *testing.T) {
	hc := &HealthChecker{
		health:           make(map[string]*WorkerHealth),
		interval:         100 * time.Millisecond,
		timeout:          100 * time.Millisecond,
		failureThreshold: 2,
		stopChan:         make(chan struct{}),
	}

	workerID := "test-worker-2"
	hc.health[workerID] = &WorkerHealth{
		WorkerID:            workerID,
		State:               "Healthy",
		ConsecutiveFailures: 0,
		LastHealthCheckTime: time.Now(),
		LastStateChange:     time.Now(),
	}

	// Failure 1
	hc.recordFailure(workerID)
	hc.mu.RLock()
	failures := hc.health[workerID].ConsecutiveFailures
	hc.mu.RUnlock()
	if failures != 1 {
		t.Errorf("Expected 1 failure, got %d", failures)
	}

	// Success resets counter
	hc.recordSuccess(workerID, 50*time.Millisecond)
	hc.mu.RLock()
	failures = hc.health[workerID].ConsecutiveFailures
	hc.mu.RUnlock()
	if failures != 0 {
		t.Errorf("Expected 0 failures after success, got %d", failures)
	}

	// Failure 1 again
	hc.recordFailure(workerID)
	hc.mu.RLock()
	failures = hc.health[workerID].ConsecutiveFailures
	hc.mu.RUnlock()
	if failures != 1 {
		t.Errorf("Expected 1 failure (counter reset), got %d", failures)
	}

	// Failure 2
	hc.recordFailure(workerID)
	if hc.GetWorkerHealth(workerID) != "Unhealthy" {
		t.Errorf("Expected Unhealthy after 2 consecutive, got %s", hc.GetWorkerHealth(workerID))
	}

	t.Log("✓ Test passed: Consecutive failure counter")
}

// TestIsWorkerHealthy verifies health query
func TestIsWorkerHealthy(t *testing.T) {
	hc := &HealthChecker{
		health:           make(map[string]*WorkerHealth),
		interval:         100 * time.Millisecond,
		timeout:          100 * time.Millisecond,
		failureThreshold: 2,
		stopChan:         make(chan struct{}),
	}

	workerID := "test-worker-3"
	hc.health[workerID] = &WorkerHealth{
		WorkerID:            workerID,
		State:               "Healthy",
		ConsecutiveFailures: 0,
		LastHealthCheckTime: time.Now(),
		LastStateChange:     time.Now(),
	}

	// Initially healthy
	if !hc.IsWorkerHealthy(workerID) {
		t.Errorf("Expected IsWorkerHealthy=true initially")
	}

	// After threshold failures
	hc.recordFailure(workerID)
	hc.recordFailure(workerID)

	if hc.IsWorkerHealthy(workerID) {
		t.Errorf("Expected IsWorkerHealthy=false after failures")
	}

	t.Log("✓ Test passed: IsWorkerHealthy query")
}

// TestHealthyWorkerCount verifies aggregation
func TestHealthyWorkerCount(t *testing.T) {
	hc := &HealthChecker{
		health:           make(map[string]*WorkerHealth),
		interval:         100 * time.Millisecond,
		timeout:          100 * time.Millisecond,
		failureThreshold: 2,
		stopChan:         make(chan struct{}),
	}

	// Add 4 workers, all initially healthy
	for i := 0; i < 4; i++ {
		workerID := "worker-" + string(rune('0'+i))
		hc.health[workerID] = &WorkerHealth{
			WorkerID:            workerID,
			State:               "Healthy",
			ConsecutiveFailures: 0,
			LastHealthCheckTime: time.Now(),
			LastStateChange:     time.Now(),
		}
	}

	if hc.HealthyWorkerCount() != 4 {
		t.Errorf("Expected 4 healthy workers, got %d", hc.HealthyWorkerCount())
	}

	// Mark 2 as unhealthy
	hc.recordFailure("worker-0")
	hc.recordFailure("worker-0")
	hc.recordFailure("worker-1")
	hc.recordFailure("worker-1")

	if hc.HealthyWorkerCount() != 2 {
		t.Errorf("Expected 2 healthy workers after 2 failures, got %d", hc.HealthyWorkerCount())
	}

	if hc.UnhealthyWorkerCount() != 2 {
		t.Errorf("Expected 2 unhealthy workers, got %d", hc.UnhealthyWorkerCount())
	}

	t.Log("✓ Test passed: Healthy worker count")
}

// TestGetAllWorkerHealth verifies snapshot query
func TestGetAllWorkerHealth(t *testing.T) {
	hc := &HealthChecker{
		health:           make(map[string]*WorkerHealth),
		interval:         100 * time.Millisecond,
		timeout:          100 * time.Millisecond,
		failureThreshold: 2,
		stopChan:         make(chan struct{}),
	}

	// Add 3 workers
	for i := 0; i < 3; i++ {
		workerID := "worker-" + string(rune('0'+i))
		hc.health[workerID] = &WorkerHealth{
			WorkerID:            workerID,
			State:               "Healthy",
			ConsecutiveFailures: 0,
			LastHealthCheckTime: time.Now(),
			LastStateChange:     time.Now(),
		}
	}

	// Mark one as unhealthy
	hc.recordFailure("worker-1")
	hc.recordFailure("worker-1")

	snapshot := hc.GetAllWorkerHealth()

	if len(snapshot) != 3 {
		t.Errorf("Expected snapshot of 3 workers, got %d", len(snapshot))
	}

	if snapshot["worker-0"] != "Healthy" || snapshot["worker-2"] != "Healthy" {
		t.Errorf("Expected workers 0 and 2 to be Healthy")
	}

	if snapshot["worker-1"] != "Unhealthy" {
		t.Errorf("Expected worker 1 to be Unhealthy")
	}

	t.Log("✓ Test passed: Get all worker health")
}

// TestHealthStats verifies statistics aggregation
func TestHealthStats(t *testing.T) {
	hc := &HealthChecker{
		health:           make(map[string]*WorkerHealth),
		interval:         100 * time.Millisecond,
		timeout:          100 * time.Millisecond,
		failureThreshold: 2,
		stopChan:         make(chan struct{}),
	}

	// Add 5 workers
	for i := 0; i < 5; i++ {
		workerID := "worker-" + string(rune('0'+i))
		hc.health[workerID] = &WorkerHealth{
			WorkerID:            workerID,
			State:               "Healthy",
			ConsecutiveFailures: 0,
			LastHealthCheckTime: time.Now(),
			LastStateChange:     time.Now(),
		}
	}

	// Mark 2 as unhealthy
	hc.recordFailure("worker-0")
	hc.recordFailure("worker-0")
	hc.recordFailure("worker-3")
	hc.recordFailure("worker-3")

	stats := hc.GetStats()

	if stats.TotalWorkers != 5 {
		t.Errorf("Expected total=5, got %d", stats.TotalWorkers)
	}

	if stats.HealthyWorkers != 3 {
		t.Errorf("Expected healthy=3, got %d", stats.HealthyWorkers)
	}

	if stats.UnhealthyWorkers != 2 {
		t.Errorf("Expected unhealthy=2, got %d", stats.UnhealthyWorkers)
	}

	t.Log("✓ Test passed: Health statistics")
}
