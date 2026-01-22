package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Coordinator is the main coordinator service
type Coordinator struct {
	config *Config
	queue  *JobQueue
	pool   *WorkerPool
	dedup  *Deduplicator
	cache  *ResultCache
}

// PredictRequest is the HTTP request payload
type PredictRequest struct {
	GridID        string `json:"grid_id"`
	TimestampHour string `json:"timestamp_hour"`
}

// PredictResponse is the HTTP response payload
type PredictResponse struct {
	RequestID     string         `json:"request_id"`
	PM25          float64        `json:"pm25,omitempty"`
	Error         *ErrorResponse `json:"error,omitempty"`
}

// NewCoordinator creates a new coordinator instance
func NewCoordinator(config *Config) (*Coordinator, error) {
	// Create queue
	queue := NewJobQueue(config.QueueCapacity)

	// Create worker pool
	pool, err := NewWorkerPool(config.NumWorkers, config.PythonWorkerPath, queue)
	if err != nil {
		return nil, fmt.Errorf("failed to create worker pool: %w", err)
	}

	// Create deduplicator
	dedup := NewDeduplicator()

	// Create result cache with 5-minute TTL (Part 1)
	cache := NewResultCache(5 * time.Minute)

	return &Coordinator{
		config: config,
		queue:  queue,
		pool:   pool,
		dedup:  dedup,
		cache:  cache,
	}, nil
}

// handlePredict handles POST /predict requests
// Orchestrates: Cache Check → Dedup → Compute → Cache Store
// Part 3b: Instruments with request_latency_seconds and path/status labels
func (c *Coordinator) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Record request entry time for end-to-end latency
	startTime := time.Now()

	// Parse request
	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.GridID == "" || req.TimestampHour == "" {
		http.Error(w, "Missing required fields: grid_id, timestamp_hour", http.StatusBadRequest)
		return
	}

	// Generate request ID
	requestID := uuid.New().String()

	log.Printf("[%s] Received prediction request (grid_id=%s, timestamp=%s)",
		requestID, req.GridID, req.TimestampHour)

	// Create context with 1-second timeout (Part 2)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// Execute prediction with dedup + cache
	pm25, path, err := c.executePredictionWithPath(ctx, req.GridID, req.TimestampHour, requestID)

	// Record end-to-end latency
	if globalMetrics != nil {
		elapsed := time.Since(startTime).Seconds()
		globalMetrics.RequestLatencySeconds.Observe(elapsed)
	}

	// Build response
	resp := PredictResponse{
		RequestID: requestID,
	}

	if err != nil {
		// Classify error and build structured response
		predErr := ClassifyError(err)
		errorResp := NewErrorResponse(predErr)
		resp.Error = &errorResp

		// Record metrics with path/status labels
		if globalMetrics != nil {
			globalMetrics.RequestsTotal.WithLabelValues(path, "error").Inc()
			globalMetrics.ErrorsTotal.WithLabelValues(predErr.Type().String()).Inc()
		}

		// Set HTTP status from error
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(predErr.HTTPStatus())
		json.NewEncoder(w).Encode(resp)
		return
	}

	// Success - record with path/status labels
	resp.PM25 = pm25
	if globalMetrics != nil {
		globalMetrics.RequestsTotal.WithLabelValues(path, "success").Inc()
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resp)
}

// executePredictionWithPath orchestrates cache check, dedup, and computation
// Returns: (pm25, path, error) where path is "cache", "dedup", or "compute"
func (c *Coordinator) executePredictionWithPath(ctx context.Context, gridID, timestampHour, requestID string) (float64, string, error) {
	// Construct dedup/cache key (model_version and feature_version hardcoded for now)
	key := DedupKey{
		GridID:         gridID,
		TimestampHour:  timestampHour,
		ModelVersion:   "prod_v1",
		FeatureVersion: "v1",
	}

	// Step 1: Check cache (both success and cacheable errors)
	cacheKey := CacheKey{
		GridID:         key.GridID,
		TimestampHour:  key.TimestampHour,
		ModelVersion:   key.ModelVersion,
		FeatureVersion: key.FeatureVersion,
	}

	if pm25, cachedErr, hit := c.cache.Get(cacheKey); hit {
		if cachedErr != nil {
			log.Printf("[%s] Cache HIT (error) for %s: %v", requestID, key.String(), cachedErr)
			return 0, "cache", cachedErr
		}
		log.Printf("[%s] Cache HIT for %s", requestID, key.String())
		return pm25, "cache", nil
	}

	log.Printf("[%s] Cache MISS for %s", requestID, key.String())

	// Step 2: Attempt to acquire leader
	entry, isLeader := c.dedup.AcquireLeader(ctx, key, requestID)

	if !isLeader {
		// We're a follower - wait for leader to finish
		log.Printf("[%s] FOLLOWER: waiting for leader %s", requestID, entry.LeaderRequestID)

		// Track deduped request
		if globalMetrics != nil {
			globalMetrics.DedupedRequestsTotal.Inc()
		}

		result, err := c.dedup.WaitForResult(ctx, entry)
		if err != nil {
			log.Printf("[%s] FOLLOWER: wait failed: %v", requestID, err)
			return 0, "dedup", err
		}

		log.Printf("[%s] FOLLOWER: received result from leader (pm25=%.2f)", requestID, result.PM25)
		return result.PM25, "dedup", result.Error
	}

	// We're the leader - execute computation
	log.Printf("[%s] LEADER: executing computation for %s", requestID, key.String())

	// Track inflight leader computation
	if globalMetrics != nil {
		globalMetrics.InflightLeaderComputations.Inc()
		defer globalMetrics.InflightLeaderComputations.Dec()
	}

	pm25, err := c.executeComputation(ctx, gridID, timestampHour, requestID)

	// Classify error if present
	var predErr PredictionError
	if err != nil {
		predErr = ClassifyError(err)
	}

	// Complete dedup (wakes followers)
	dedupResult := &DedupResult{
		PM25:  pm25,
		Error: predErr,
	}
	c.dedup.Complete(key, dedupResult)

	// Cache result based on error type
	if predErr == nil {
		// Success - cache with default TTL
		c.cache.Set(cacheKey, pm25)
		log.Printf("[%s] LEADER: cached result (pm25=%.2f)", requestID, pm25)
	} else if predErr.Cacheable() {
		// Cacheable error - cache with error-specific TTL
		c.cache.SetError(cacheKey, predErr)
		log.Printf("[%s] LEADER: cached error %s (ttl=%v)", requestID, predErr.Type(), predErr.CacheTTL())
	} else {
		// Non-cacheable error - don't cache
		log.Printf("[%s] LEADER: error not cached: %s", requestID, predErr.Type())
	}

	return pm25, "compute", predErr
}

// executePrediction orchestrates cache check, dedup, and computation (legacy)
func (c *Coordinator) executePrediction(ctx context.Context, gridID, timestampHour, requestID string) (float64, error) {
	pm25, _, err := c.executePredictionWithPath(ctx, gridID, timestampHour, requestID)
	return pm25, err
}

// executeComputation submits job to worker pool and waits for result
// Instruments leader_compute_latency_seconds (execution time from queue to completion)
func (c *Coordinator) executeComputation(ctx context.Context, gridID, timestampHour, requestID string) (float64, error) {
	startTime := time.Now()
	
	// Create job
	job := Job{
		GridID:        gridID,
		TimestampHour: timestampHour,
		RequestID:     requestID,
	}

	// Register result channel before enqueuing
	resultChan := c.pool.RegisterResultChannel(requestID)
	defer c.pool.UnregisterResultChannel(requestID)

	// Enqueue job
	if err := c.queue.Enqueue(job); err != nil {
		log.Printf("[%s] Failed to enqueue job: %v", requestID, err)
		return 0, fmt.Errorf("queue full")
	}

	log.Printf("[%s] Enqueued job (queue length: %d)", requestID, c.queue.Len())

	// Wait for result or timeout
	select {
	case result := <-resultChan:
		// Record leader computation latency
		if globalMetrics != nil {
			elapsed := time.Since(startTime).Seconds()
			globalMetrics.LeaderComputeLatencySeconds.Observe(elapsed)
		}

		if result.Error != "" {
			// Convert worker error to typed error based on error_type
			return 0, c.convertWorkerError(result)
		}
		return result.PM25, nil

	case <-ctx.Done():
		// Request timeout (context deadline exceeded)
		return 0, ctx.Err()
	}
}

// convertWorkerError converts a WorkerResult error to a PredictionError
func (c *Coordinator) convertWorkerError(result WorkerResult) error {
	switch result.ErrorType {
	case "InvalidRequest":
		return NewInvalidRequestError(result.Error)
	case "GridNotFound":
		// Extract grid ID from error message if possible
		return NewGridNotFoundError(result.Error)
	case "DataMissing":
		return NewDataMissingError(result.Error)
	case "WorkerError":
		return NewWorkerError(result.Error)
	case "Unknown":
		return NewUnknownError(result.Error)
	default:
		// Fallback for errors without type
		return NewUnknownError(result.Error)
	}
}

// handleHealth handles GET /health requests
func (c *Coordinator) handleHealth(w http.ResponseWriter, r *http.Request) {
	dedupStats := c.dedup.Stats()
	cacheStats := c.cache.Stats()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "healthy",
		"queue_length": c.queue.Len(),
		"num_workers":  c.config.NumWorkers,
		"dedup": map[string]int{
			"active_computations": dedupStats.ActiveComputations,
			"total_followers":     dedupStats.TotalFollowers,
		},
		"cache": map[string]int{
			"total_entries":   cacheStats.TotalEntries,
			"expired_entries": cacheStats.ExpiredEntries,
		},
	})
}

// handleMetrics exposes Prometheus metrics at /metrics
func (c *Coordinator) handleMetrics(w http.ResponseWriter, r *http.Request) {
	promhttp.Handler().ServeHTTP(w, r)
}

// Shutdown gracefully shuts down the coordinator
func (c *Coordinator) Shutdown() {
	log.Println("Shutting down coordinator...")
	c.pool.Shutdown()
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Initialize metrics globally
	globalMetrics = NewMetricsWithLabels()

	// Load config
	config := DefaultConfig()
	log.Printf("Starting coordinator with config: port=%s, workers=%d, queue_capacity=%d",
		config.ServerPort, config.NumWorkers, config.QueueCapacity)

	// Create coordinator
	coordinator, err := NewCoordinator(config)
	if err != nil {
		log.Fatalf("Failed to create coordinator: %v", err)
	}

	// Setup HTTP handlers
	http.HandleFunc("/predict", coordinator.handlePredict)
	http.HandleFunc("/health", coordinator.handleHealth)
	http.HandleFunc("/metrics", coordinator.handleMetrics)

	// Create HTTP server
	server := &http.Server{
		Addr:         ":" + config.ServerPort,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	// Start server in background
	go func() {
		log.Printf("Starting HTTP server on port %s", config.ServerPort)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("HTTP server failed: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Println("Received shutdown signal")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}

	coordinator.Shutdown()

	log.Println("Shutdown complete")
}
