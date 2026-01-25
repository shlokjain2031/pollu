package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/google/uuid"
	_ "github.com/marcboeker/go-duckdb"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
)

// Coordinator is the main coordinator service
// Item 26: Stores model/feature versions read at startup from environment
type Coordinator struct {
	config          *Config
	queue           *JobQueue
	pool            *WorkerPool
	dedup           *Deduplicator
	cache           *ResultCache
	predictionCache *PredictionCache // Phase 1: In-memory cache with async writes
	modelVersion    string           // Read from MODEL_VERSION env var (Item 26)
	featureVersion  string           // Read from FEATURE_VERSION env var (Item 26)

	// NEW: Redis connection for hot cache (2-hour TTL)
	redisClient *redis.Client

	// NEW: DuckDB connection for persistent predictions cache
	duckdbConn *sql.DB

	// NEW: Orchestrator HTTP client for on-demand inference
	orchestratorClient *http.Client
	orchestratorURL    string
}

// PredictRequest is the HTTP request payload
type PredictRequest struct {
	GridID        string `json:"grid_id"`
	TimestampHour string `json:"timestamp_hour"`
}

// PredictResponse is the HTTP response payload
type PredictResponse struct {
	RequestID string         `json:"request_id"`
	PM25      float64        `json:"pm25,omitempty"`
	Source    string         `json:"source,omitempty"` // "redis", "duckdb", or "inference"
	Error     *ErrorResponse `json:"error,omitempty"`
}

// NewCoordinator creates a new coordinator instance
// Item 26: Reads ModelVersion and FeatureVersion from environment at startup
func NewCoordinator(config *Config) (*Coordinator, error) {
	// Item 26: Read versions from environment (not request-scoped)
	modelVersion := os.Getenv("MODEL_VERSION")
	if modelVersion == "" {
		modelVersion = "unknown" // Default if not set
	}
	featureVersion := os.Getenv("FEATURE_VERSION")
	if featureVersion == "" {
		featureVersion = "unknown" // Default if not set
	}

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

	// NEW: Initialize Redis client for hot cache
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379" // Default
	}
	redisClient := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	// Test Redis connection
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		log.Printf("WARNING: Redis connection failed: %v (predictions will fallback to DuckDB)", err)
	}

	// NEW: Initialize DuckDB predictions connection
	duckdbPath := os.Getenv("DUCKDB_PREDICTIONS_PATH")
	if duckdbPath == "" {
		duckdbPath = "/Users/shlokjain/pollu/takshashila/duckdb/predictions.duckdb" // Default
	}
	duckdbConn, err := sql.Open("duckdb", duckdbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB predictions: %w", err)
	}

	// Test DuckDB connection
	if err := duckdbConn.Ping(); err != nil {
		log.Printf("WARNING: DuckDB connection failed: %v", err)
	}

	// NEW: Initialize Orchestrator HTTP client
	orchestratorURL := os.Getenv("KASHI_ORCHESTRATOR_URL")
	if orchestratorURL == "" {
		orchestratorURL = "http://localhost:9999" // Default
	}
	orchestratorClient := &http.Client{
		Timeout: 120 * time.Second, // 2 minutes for inference completion
	}
	log.Printf("Orchestrator configured at: %s", orchestratorURL)

	return &Coordinator{
		config:             config,
		queue:              queue,
		pool:               pool,
		dedup:              dedup,
		cache:              cache,
		modelVersion:       modelVersion,   // Item 26: Stored at startup
		featureVersion:     featureVersion, // Item 26: Stored at startup
		redisClient:        redisClient,
		duckdbConn:         duckdbConn,
		orchestratorClient: orchestratorClient,
		orchestratorURL:    orchestratorURL,
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

	// Item 27: Log request_id in structured logs (not in metrics)
	log.Printf("[%s] [START] Received prediction request (grid_id=%s, timestamp=%s, model=%s, feature=%s)",
		requestID, req.GridID, req.TimestampHour, c.modelVersion, c.featureVersion)

	// Try to read from cache FIRST (Redis → DuckDB → on-demand inference)
	// Use 120-second timeout to allow on-demand inference to complete
	cacheCtx, cacheCancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cacheCancel()
	pm25, source, err := c.readPredictionFromCache(cacheCtx, req.GridID, req.TimestampHour, requestID)

	// If found in cache, return immediately (no workers needed)
	if err == nil {
		elapsed := time.Since(startTime).Milliseconds()
		resp := PredictResponse{
			RequestID: requestID,
			PM25:      pm25,
			Source:    source,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		log.Printf("[%s] [END] Served from cache (Source: %s) in %dms", requestID, source, elapsed)
		return
	}

	// Cache miss - now check if workers are available for on-demand inference
	if c.pool.healthChecker.HealthyWorkerCount() == 0 {
		// No workers - return cached error (PredictionNotAvailableError)
		log.Printf("[%s] No healthy workers available and cache miss", requestID)
		predErr := ClassifyError(err)
		errorResp := NewErrorResponse(predErr)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(predErr.HTTPStatus())
		json.NewEncoder(w).Encode(map[string]interface{}{"error": errorResp})
		return
	}

	// Create context with 1-second timeout (Part 2)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// Execute prediction with dedup + cache (workers ARE healthy)
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

		// Item 26: Record metrics with path, status, model_version, feature_version labels
		// request_id is NOT included (forbidden by Item 27, but logged above for structured logs)
		if globalMetrics != nil {
			globalMetrics.RequestsTotal.WithLabelValues(path, "error", c.modelVersion, c.featureVersion).Inc()
			globalMetrics.ErrorsTotal.WithLabelValues(predErr.Type().String(), c.modelVersion, c.featureVersion).Inc()
		}

		// Set HTTP status from error
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(predErr.HTTPStatus())
		json.NewEncoder(w).Encode(resp)
		return
	}

	// Success - record with path, status, model_version, feature_version labels
	resp.PM25 = pm25
	resp.Source = "inference" // Came from worker inference
	if globalMetrics != nil {
		globalMetrics.RequestsTotal.WithLabelValues(path, "success", c.modelVersion, c.featureVersion).Inc()
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

	// Step 1: Check OLD cache (both success and cacheable errors)
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

	// NEW: Read from cache tiers (Redis → DuckDB) instead of computing
	pm25, source, err := c.readPredictionFromCache(ctx, gridID, timestampHour, requestID)
	_ = source // Dedup path doesn't need to track source separately

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

	// Phase 1: Cache in OLD result cache based on error type
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

// readPredictionFromCache reads pre-computed predictions from Redis → DuckDB
// DOES NOT compute - fails loudly with 404 if prediction not found
// This is the read-only path: orchestrator computes, coordinator serves
func (c *Coordinator) readPredictionFromCache(
	ctx context.Context,
	gridID,
	timestampHour,
	requestID string,
) (float64, string, error) { // Returns: (pm25, source, error)

	// Step 1: Try Redis cache (hot cache, 2-hour TTL)
	redisKey := fmt.Sprintf("%s:%s", gridID, timestampHour)
	pm25Str, err := c.redisClient.Get(ctx, redisKey).Result()
	if err == nil {
		pm25, parseErr := strconv.ParseFloat(pm25Str, 64)
		if parseErr == nil {
			log.Printf("[%s] Redis cache HIT for %s:%s (pm25=%.2f)", requestID, gridID, timestampHour, pm25)
			return pm25, "redis", nil
		}
		log.Printf("[%s] Redis cache value parse error: %v", requestID, parseErr)
	}

	log.Printf("[%s] Redis cache MISS for %s:%s", requestID, gridID, timestampHour)

	// Step 2: Try DuckDB predictions table (persistent cache)
	query := `
		SELECT pm25_prediction 
		FROM predictions 
		WHERE grid_id = ? AND timestamp_hour = ?
		LIMIT 1
	`

	var pm25 float64
	row := c.duckdbConn.QueryRowContext(ctx, query, gridID, timestampHour)
	if err := row.Scan(&pm25); err == nil {
		log.Printf("[%s] DuckDB cache HIT for %s:%s (pm25=%.2f)", requestID, gridID, timestampHour, pm25)

		// Populate Redis for next request (async, fire-and-forget)
		go func() {
			bgCtx := context.Background()
			c.redisClient.Set(bgCtx, redisKey, pm25, 2*time.Hour)
		}()

		return pm25, "duckdb", nil
	}

	// Step 3: On-demand inference via orchestrator
	// TODO: Test on-demand inference with grids that have complete feature data
	// Current blockers: Feature tables incomplete for most grids
	// Test case: Find grid with all 64 required features populated in feature tables
	log.Printf("[%s] Prediction NOT FOUND - requesting on-demand inference for %s:%s", requestID, gridID, timestampHour)

	pm25, err = c.requestInferenceFromOrchestrator(ctx, gridID, timestampHour, requestID)
	if err == nil {
		log.Printf("[%s] On-demand inference SUCCEEDED (pm25=%.2f)", requestID, pm25)
		return pm25, "inference", nil
	}

	// Step 4: FAIL LOUDLY - on-demand inference failed
	log.Printf("[%s] On-demand inference FAILED: %v", requestID, err)
	return 0, "", NewPredictionNotAvailableError(gridID, timestampHour)
}

// requestInferenceFromOrchestrator sends on-demand inference request to orchestrator
// Orchestrator spawns worker asynchronously, coordinator polls Redis for result
func (c *Coordinator) requestInferenceFromOrchestrator(
	ctx context.Context,
	gridID,
	timestampHour,
	requestID string,
) (float64, error) {

	// Build request payload
	cycleID := uuid.New().String()
	reqBody := map[string]interface{}{
		"grids":          []string{gridID},
		"timestamp_hour": timestampHour,
		"request_id":     requestID,
		"cycle_id":       cycleID,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send POST request to orchestrator
	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		c.orchestratorURL+"/inference",
		bytes.NewReader(jsonBody),
	)
	if err != nil {
		return 0, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	log.Printf("[%s] Sending on-demand inference request to orchestrator (cycle_id=%s)", requestID, cycleID)

	resp, err := c.orchestratorClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("orchestrator request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("orchestrator returned status %d", resp.StatusCode)
	}

	// Parse response
	var orchResp struct {
		Status  string `json:"status"`
		CycleID string `json:"cycle_id"`
		Message string `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&orchResp); err != nil {
		return 0, fmt.Errorf("failed to decode orchestrator response: %w", err)
	}

	log.Printf("[%s] Orchestrator response: %s (cycle_id=%s)", requestID, orchResp.Status, orchResp.CycleID)

	// Poll Redis for result (worker writes to cache asynchronously)
	return c.pollRedisForResult(ctx, gridID, timestampHour, requestID, 90*time.Second)
}

// pollRedisForResult polls Redis cache every 200ms until result appears or timeout
func (c *Coordinator) pollRedisForResult(
	ctx context.Context,
	gridID,
	timestampHour,
	requestID string,
	timeout time.Duration,
) (float64, error) {

	redisKey := fmt.Sprintf("%s:%s", gridID, timestampHour)
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	log.Printf("[%s] Polling Redis for result (timeout=%v)", requestID, timeout)

	for {
		select {
		case <-ticker.C:
			// Check Redis cache
			pm25Str, err := c.redisClient.Get(ctx, redisKey).Result()
			if err == nil && pm25Str != "" {
				pm25, parseErr := strconv.ParseFloat(pm25Str, 64)
				if parseErr == nil {
					log.Printf("[%s] Poll SUCCESS - found result in Redis (pm25=%.2f)", requestID, pm25)
					return pm25, nil
				}
			}

			// Check timeout
			if time.Now().After(deadline) {
				log.Printf("[%s] Poll TIMEOUT after %v", requestID, timeout)
				return 0, fmt.Errorf("inference timed out after %v", timeout)
			}

		case <-ctx.Done():
			log.Printf("[%s] Poll CANCELLED (context done)", requestID)
			return 0, ctx.Err()
		}
	}
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
	log.Println("Coordinator shutdown complete")
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Initialize metrics globally
	globalMetrics = NewMetricsWithLabels()

	// Load config
	config := DefaultConfig()
	log.Printf("Starting coordinator with config: port=%s, workers=%d, queue_capacity=%d",
		config.ServerPort, config.NumWorkers, config.QueueCapacity)

	// TODO: Spawn orchestrator as subprocess
	// Currently orchestrator is started manually. Should:
	// 1. Spawn: subprocess.Popen(['python3', '-m', 'takshashila.distributed.worker.orchestrator'])
	// 2. Monitor orchestrator health via /health endpoint
	// 3. Restart orchestrator if it crashes
	// 4. Kill orchestrator on coordinator shutdown
	// 5. Log orchestrator output for debugging

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
