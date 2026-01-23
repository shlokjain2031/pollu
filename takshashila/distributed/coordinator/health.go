package main

import (
	"encoding/json"
	"log"
	"sync"
	"time"
)

// HealthCheckMessage represents a health check request sent to worker
type HealthCheckMessage struct {
	Type string `json:"type"` // "health_check"
	Ts   int64  `json:"ts"`   // Unix milliseconds
}

// HealthCheckResponse represents the response from worker
type HealthCheckResponse struct {
	Type string `json:"type"` // "health_response"
	Pong bool   `json:"pong"`
}

// WorkerHealth tracks health state of a single worker
type WorkerHealth struct {
	WorkerID            string
	State               string    // "Healthy" or "Unhealthy"
	ConsecutiveFailures int       // 0, 1, 2+
	LastHealthCheckTime time.Time // When we last performed a check
	LastStateChange     time.Time // When state last transitioned
	mu                  sync.RWMutex
}

// HealthChecker manages all worker health checks
type HealthChecker struct {
	workers          []*PythonWorker
	health           map[string]*WorkerHealth // keyed by worker ID
	interval         time.Duration
	timeout          time.Duration
	failureThreshold int
	mu               sync.RWMutex
	stopChan         chan struct{}
	done             sync.WaitGroup
}

// NewHealthChecker creates and starts health checking for all workers
func NewHealthChecker(
	workers []*PythonWorker,
	intervalMs int,
	timeoutMs int,
	failureThreshold int,
) *HealthChecker {
	hc := &HealthChecker{
		workers:          workers,
		health:           make(map[string]*WorkerHealth),
		interval:         time.Duration(intervalMs) * time.Millisecond,
		timeout:          time.Duration(timeoutMs) * time.Millisecond,
		failureThreshold: failureThreshold,
		stopChan:         make(chan struct{}),
	}

	// Initialize health records for all workers
	for _, w := range workers {
		hc.health[w.ID] = &WorkerHealth{
			WorkerID:            w.ID,
			State:               "Healthy",
			ConsecutiveFailures: 0,
			LastHealthCheckTime: time.Now(),
			LastStateChange:     time.Now(),
		}
	}

	// Start health check goroutine for each worker
	for _, w := range workers {
		hc.done.Add(1)
		go hc.checkWorkerHealth(w)
	}

	log.Printf("HealthChecker started for %d workers (interval=%dms, timeout=%dms, threshold=%d)",
		len(workers), intervalMs, timeoutMs, failureThreshold)

	return hc
}

// checkWorkerHealth periodically checks a single worker
// Runs in background goroutine per worker
func (hc *HealthChecker) checkWorkerHealth(worker *PythonWorker) {
	defer hc.done.Done()

	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()

	for {
		select {
		case <-hc.stopChan:
			log.Printf("[%s] Health check goroutine stopping", worker.ID)
			return
		case <-ticker.C:
			hc.performHealthCheck(worker)
		}
	}
}

// performHealthCheck sends a health check to a worker and waits for response
func (hc *HealthChecker) performHealthCheck(worker *PythonWorker) {
	startTime := time.Now()

	// Create health check message
	healthCheckMsg := HealthCheckMessage{
		Type: "health_check",
		Ts:   startTime.UnixMilli(),
	}

	msgJSON, err := json.Marshal(healthCheckMsg)
	if err != nil {
		log.Printf("[%s] Failed to marshal health check: %v", worker.ID, err)
		hc.recordFailure(worker.ID)
		return
	}

	// Send to worker stdin
	worker.mu.Lock()
	if !worker.active {
		worker.mu.Unlock()
		log.Printf("[%s] Worker inactive, skipping health check", worker.ID)
		hc.recordFailure(worker.ID)
		return
	}

	_, err = worker.stdin.Write(append(msgJSON, '\n'))
	worker.mu.Unlock()

	if err != nil {
		log.Printf("[%s] Failed to send health check: %v", worker.ID, err)
		hc.recordFailure(worker.ID)
		return
	}

	// Wait for response with timeout
	responseChan := make(chan bool, 1)
	go hc.waitForHealthResponse(worker.ID, responseChan)

	select {
	case success := <-responseChan:
		elapsed := time.Since(startTime)
		if success {
			log.Printf("[%s] Health check passed in %v", worker.ID, elapsed)
			hc.recordSuccess(worker.ID, elapsed)
		} else {
			log.Printf("[%s] Health check failed (invalid response)", worker.ID)
			hc.recordFailure(worker.ID)
		}

	case <-time.After(hc.timeout):
		log.Printf("[%s] Health check timeout after %v", worker.ID, hc.timeout)
		hc.recordFailure(worker.ID)
	}
}

// waitForHealthResponse waits for a health response
// This is a simplified version that assumes responses arrive in order
// In Part 2, this will be enhanced with proper request/response matching
func (hc *HealthChecker) waitForHealthResponse(workerID string, responseChan chan bool) {
	// Simplified for Part 1: immediately signal success
	// Actual response validation happens in readResults() in python_client.go
	responseChan <- true
}

// recordSuccess marks a successful health check
func (hc *HealthChecker) recordSuccess(workerID string, latency time.Duration) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	h := hc.health[workerID]
	oldState := h.State

	h.ConsecutiveFailures = 0
	h.State = "Healthy"
	h.LastHealthCheckTime = time.Now()

	if oldState != "Healthy" {
		h.LastStateChange = time.Now()
		log.Printf("[%s] ✓ Health status transition: %s → Healthy", workerID, oldState)
	}

	// Update metrics
	if globalMetrics != nil {
		globalMetrics.WorkerHealthStatus.WithLabelValues(workerID).Set(1)
		globalMetrics.WorkerHealthChecksTotal.WithLabelValues(workerID).Inc()
		globalMetrics.WorkerHealthCheckLatencySeconds.WithLabelValues(workerID).Observe(latency.Seconds())
	}
}

// recordFailure marks a failed health check
func (hc *HealthChecker) recordFailure(workerID string) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	h := hc.health[workerID]
	oldState := h.State
	h.ConsecutiveFailures++
	h.LastHealthCheckTime = time.Now()

	newState := h.State

	// Check if we've hit the failure threshold
	if h.ConsecutiveFailures >= hc.failureThreshold {
		h.State = "Unhealthy"
		newState = "Unhealthy"
		h.LastStateChange = time.Now()

		if oldState != "Unhealthy" {
			log.Printf("[%s] ✗ Health status transition: %s → Unhealthy (failures: %d)",
				workerID, oldState, h.ConsecutiveFailures)
		}
	}

	// Update metrics
	if globalMetrics != nil {
		if newState == "Unhealthy" {
			globalMetrics.WorkerHealthStatus.WithLabelValues(workerID).Set(0)
		}
		globalMetrics.WorkerHealthCheckFailuresTotal.WithLabelValues(workerID).Inc()
	}
}

// GetWorkerHealth returns current health state of a worker
func (hc *HealthChecker) GetWorkerHealth(workerID string) string {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	if h, ok := hc.health[workerID]; ok {
		return h.State
	}
	return "Unknown"
}

// IsWorkerHealthy returns true if worker is in Healthy state
func (hc *HealthChecker) IsWorkerHealthy(workerID string) bool {
	return hc.GetWorkerHealth(workerID) == "Healthy"
}

// HealthyWorkerCount returns the number of workers in Healthy state
func (hc *HealthChecker) HealthyWorkerCount() int {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	count := 0
	for _, h := range hc.health {
		if h.State == "Healthy" {
			count++
		}
	}
	return count
}

// UnhealthyWorkerCount returns the number of workers in Unhealthy state
func (hc *HealthChecker) UnhealthyWorkerCount() int {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	count := 0
	for _, h := range hc.health {
		if h.State == "Unhealthy" {
			count++
		}
	}
	return count
}

// GetAllWorkerHealth returns a snapshot of all worker health states
func (hc *HealthChecker) GetAllWorkerHealth() map[string]string {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	result := make(map[string]string)
	for workerID, h := range hc.health {
		result[workerID] = h.State
	}
	return result
}

// Stop gracefully stops all health checking
func (hc *HealthChecker) Stop() {
	log.Println("Stopping HealthChecker...")
	close(hc.stopChan)
	hc.done.Wait()
	log.Println("HealthChecker stopped")
}

// HealthStats represents health statistics
type HealthStats struct {
	TotalWorkers     int
	HealthyWorkers   int
	UnhealthyWorkers int
	WorkerStates     map[string]string
}

// GetStats returns current health statistics
func (hc *HealthChecker) GetStats() HealthStats {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	workerStates := make(map[string]string)
	healthyCount := 0

	for workerID, h := range hc.health {
		workerStates[workerID] = h.State
		if h.State == "Healthy" {
			healthyCount++
		}
	}

	return HealthStats{
		TotalWorkers:     len(hc.health),
		HealthyWorkers:   healthyCount,
		UnhealthyWorkers: len(hc.health) - healthyCount,
		WorkerStates:     workerStates,
	}
}

// PrintStats logs current health statistics
func (hc *HealthChecker) PrintStats() {
	stats := hc.GetStats()
	log.Printf("Health Statistics: Total=%d, Healthy=%d, Unhealthy=%d",
		stats.TotalWorkers, stats.HealthyWorkers, stats.UnhealthyWorkers)
	for workerID, state := range stats.WorkerStates {
		log.Printf("  [%s] %s", workerID, state)
	}
}
