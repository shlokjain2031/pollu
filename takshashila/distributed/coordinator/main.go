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
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	_ "github.com/marcboeker/go-duckdb"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
)

// utcLogWriter wraps os.Stderr and formats all output with UTC timestamps
type utcLogWriter struct{}

func (w *utcLogWriter) Write(p []byte) (n int, err error) {
	// Get current UTC time in the format: 2026/01/26 16:08:51
	now := time.Now().UTC()
	timestamp := now.Format("2006/01/02 15:04:05")

	// Prepend timestamp to output
	output := fmt.Sprintf("%s UTC %s", timestamp, string(p))
	return os.Stderr.WriteString(output)
}

// OrchestratorManager manages the Python orchestrator subprocess lifecycle
type OrchestratorManager struct {
	cmd          *exec.Cmd
	stopChan     chan struct{}
	restartChan  chan struct{}
	pythonPath   string
	modulePath   string
	healthURL    string
	maxRestarts  int
	restartDelay time.Duration
	restartCount int
	shutdownOnce sync.Once
}

// NewOrchestratorManager creates a new orchestrator manager
func NewOrchestratorManager() *OrchestratorManager {
	pythonPath := os.Getenv("PYTHON_PATH")
	if pythonPath == "" {
		// Use venv Python if it exists, otherwise fallback to system python3
		pythonPath = "/Users/shlokjain/pollu/.venv/bin/python3"
		if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
			pythonPath = "python3"
		}
	}

	orchURL := os.Getenv("KASHI_ORCHESTRATOR_URL")
	if orchURL == "" {
		orchURL = "http://localhost:9999"
	}

	return &OrchestratorManager{
		stopChan:     make(chan struct{}),
		restartChan:  make(chan struct{}, 1),
		pythonPath:   pythonPath,
		modulePath:   "takshashila.distributed.worker.orchestrator",
		healthURL:    orchURL + "/health",
		maxRestarts:  10,              // Max 10 restarts before giving up
		restartDelay: 5 * time.Second, // Wait 5 seconds between restarts
	}
}

// Start spawns the orchestrator subprocess and monitors it
func (om *OrchestratorManager) Start() error {
	log.Println("[Orchestrator] Starting Python orchestrator subprocess...")

	// Spawn orchestrator process
	if err := om.spawnProcess(); err != nil {
		return fmt.Errorf("failed to spawn orchestrator: %w", err)
	}

	log.Printf("[Orchestrator] Started subprocess (PID: %d)", om.cmd.Process.Pid)

	// Give orchestrator 3 seconds to start Flask server before health monitoring
	time.Sleep(3 * time.Second)

	// Start health monitoring in background
	go om.monitorHealth()

	// Start restart handler in background
	go om.handleRestarts()

	return nil
}

// spawnProcess spawns the Python orchestrator subprocess
func (om *OrchestratorManager) spawnProcess() error {
	om.cmd = exec.Command(om.pythonPath, "-m", om.modulePath)

	// Pass environment variables and ensure PYTHONPATH is set
	env := os.Environ()
	pythonPathSet := false
	for _, e := range env {
		if strings.HasPrefix(e, "PYTHONPATH=") {
			pythonPathSet = true
			break
		}
	}
	if !pythonPathSet {
		// Add /Users/shlokjain/pollu to PYTHONPATH
		env = append(env, "PYTHONPATH=/Users/shlokjain/pollu")
	}
	om.cmd.Env = env

	// Redirect stdout/stderr to coordinator logs with [Orchestrator] prefix
	om.cmd.Stdout = &prefixedWriter{prefix: "[Orchestrator] [STDOUT] "}
	om.cmd.Stderr = &prefixedWriter{prefix: "[Orchestrator] [STDERR] "}

	// Start process
	if err := om.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start process: %w", err)
	}

	return nil
}

// monitorHealth periodically checks orchestrator health and triggers restart on failure
func (om *OrchestratorManager) monitorHealth() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	// Wait 15 seconds before first health check (allow orchestrator to start)
	time.Sleep(15 * time.Second)

	for {
		select {
		case <-om.stopChan:
			log.Println("[Orchestrator] Health monitor stopped")
			return

		case <-ticker.C:
			// Check process is still running
			if om.cmd == nil || om.cmd.Process == nil {
				log.Println("[Orchestrator] Process not running, triggering restart")
				om.triggerRestart()
				continue
			}

			// Check process state
			if om.cmd.ProcessState != nil && om.cmd.ProcessState.Exited() {
				log.Printf("[Orchestrator] Process exited (exit code: %d), triggering restart", om.cmd.ProcessState.ExitCode())
				om.triggerRestart()
				continue
			}

			// Ping health endpoint
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			req, err := http.NewRequestWithContext(ctx, "GET", om.healthURL, nil)
			resp, err := http.DefaultClient.Do(req)
			if err != nil || (resp != nil && resp.StatusCode != http.StatusOK) {
				log.Printf("[Orchestrator] Health check failed (err=%v), triggering restart", err)
				om.triggerRestart()
				if resp != nil {
					resp.Body.Close()
				}
				continue
			}

			if resp != nil {
				resp.Body.Close()
			}
		}
	}
}

// triggerRestart sends restart signal (non-blocking)
func (om *OrchestratorManager) triggerRestart() {
	select {
	case om.restartChan <- struct{}{}:
		// Restart signal sent
	default:
		// Already pending restart
	}
}

// handleRestarts processes restart requests with rate limiting
func (om *OrchestratorManager) handleRestarts() {
	for {
		select {
		case <-om.stopChan:
			log.Println("[Orchestrator] Restart handler stopped")
			return

		case <-om.restartChan:
			if om.restartCount >= om.maxRestarts {
				log.Printf("[Orchestrator] Max restarts (%d) exceeded, giving up", om.maxRestarts)
				return
			}

			om.restartCount++
			log.Printf("[Orchestrator] Restarting subprocess (attempt %d/%d)", om.restartCount, om.maxRestarts)

			// Kill existing process if still running
			if om.cmd != nil && om.cmd.Process != nil {
				log.Printf("[Orchestrator] Killing existing process (PID: %d)", om.cmd.Process.Pid)
				om.cmd.Process.Kill()
				om.cmd.Wait() // Wait for process to exit
			}

			// Wait before restarting
			log.Printf("[Orchestrator] Waiting %v before restart...", om.restartDelay)
			time.Sleep(om.restartDelay)

			// Spawn new process
			if err := om.spawnProcess(); err != nil {
				log.Printf("[Orchestrator] Restart failed: %v", err)
				// Retry after delay
				time.AfterFunc(om.restartDelay, func() {
					om.triggerRestart()
				})
				continue
			}

			log.Printf("[Orchestrator] Restart successful (PID: %d)", om.cmd.Process.Pid)
		}
	}
}

// Shutdown gracefully stops the orchestrator subprocess
func (om *OrchestratorManager) Shutdown() {
	om.shutdownOnce.Do(func() {
		log.Println("[Orchestrator] Shutting down subprocess...")

		// Stop monitors
		close(om.stopChan)

		// Send SIGTERM to orchestrator
		if om.cmd != nil && om.cmd.Process != nil {
			log.Printf("[Orchestrator] Sending SIGTERM to PID %d", om.cmd.Process.Pid)
			om.cmd.Process.Signal(syscall.SIGTERM)

			// Wait up to 10 seconds for graceful shutdown
			done := make(chan error, 1)
			go func() {
				done <- om.cmd.Wait()
			}()

			select {
			case <-time.After(10 * time.Second):
				log.Println("[Orchestrator] Graceful shutdown timeout, killing process")
				om.cmd.Process.Kill()
				om.cmd.Wait()
			case err := <-done:
				if err != nil {
					log.Printf("[Orchestrator] Process exited with error: %v", err)
				} else {
					log.Println("[Orchestrator] Process exited gracefully")
				}
			}
		}

		log.Println("[Orchestrator] Shutdown complete")
	})
}

// prefixedWriter wraps log output with a prefix
type prefixedWriter struct {
	prefix string
}

func (pw *prefixedWriter) Write(p []byte) (n int, err error) {
	log.Print(pw.prefix + string(p))
	return len(p), nil
}

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

// normalizeGridID converts grid IDs to 8-digit zero-padded format with 'g' prefix
// Examples: "g1" → "g00000001", "g000001" → "g00000001", "g00000001" → "g00000001"
func normalizeGridID(gridID string) string {
	// Remove 'g' prefix if present
	numericPart := gridID
	if len(gridID) > 0 && gridID[0] == 'g' {
		numericPart = gridID[1:]
	}

	// Parse as integer and re-format with 8-digit zero-padding
	gridNum, err := strconv.Atoi(numericPart)
	if err != nil {
		// If parsing fails, return original (shouldn't happen in normal operation)
		return gridID
	}

	return fmt.Sprintf("g%08d", gridNum)
}

// constructRedisCacheKey builds the cache key to match simplify_job_worker format
// Format: "prediction:{grid_id}:{timestamp_hour_iso}"
// The simplify worker writes with .strftime('%Y-%m-%dT%H:%M:%SZ') which produces Z suffix
func constructRedisCacheKey(gridID, timestampHourStr string) string {
	// Parse the incoming timestamp (typically "2026-01-25T13:00:00Z")
	parsedTime, err := time.Parse("2006-01-02T15:04:05Z", timestampHourStr)
	if err != nil {
		// Fallback: try without Z
		parsedTime, err = time.Parse("2006-01-02T15:04:05", timestampHourStr)
		if err != nil {
			// If parsing fails, use as-is (shouldn't happen)
			return fmt.Sprintf("prediction:%s:%s", gridID, timestampHourStr)
		}
		// Convert naive time to UTC
		parsedTime = parsedTime.UTC()
	}

	// Format with Z suffix to match SimplifyJob worker format
	isoTimeStr := parsedTime.UTC().Format("2006-01-02T15:04:05Z")
	return fmt.Sprintf("prediction:%s:%s", gridID, isoTimeStr)
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

	// NEW: Initialize DuckDB PM2.5 observations connection
	duckdbPath := os.Getenv("DUCKDB_PREDICTIONS_PATH")
	if duckdbPath == "" {
		duckdbPath = "/Users/shlokjain/pollu/takshashila/duckdb/pm25_observations.duckdb" // Default
	}
	duckdbConn, err := sql.Open("duckdb", duckdbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB PM2.5 observations: %w", err)
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

	// Record request entry time for end-to-end latency (duration measurement)
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

	// Normalize grid ID to 8-digit zero-padded format
	normalizedGridID := normalizeGridID(req.GridID)

	// Generate request ID
	requestID := uuid.New().String()

	// Item 27: Log request_id in structured logs (not in metrics)
	log.Printf("[%s] [START] Received prediction request (grid_id=%s → %s, timestamp=%s, model=%s, feature=%s)",
		requestID, req.GridID, normalizedGridID, req.TimestampHour, c.modelVersion, c.featureVersion)

	// Try to read from cache FIRST (Redis → DuckDB within ±1 hour)
	// Use 10-second timeout for cache reads
	cacheCtx, cacheCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cacheCancel()
	pm25, source, err := c.readPredictionFromCache(cacheCtx, normalizedGridID, req.TimestampHour, requestID)

	// If found in cache (within ±1 hour), return immediately
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

	// Cache miss (no data within ±1 hour) - return not found error
	log.Printf("[%s] No data found within ±1 hour of %s", requestID, req.TimestampHour)
	predErr := ClassifyError(err)
	errorResp := NewErrorResponse(predErr)
	elapsed := time.Since(startTime).Milliseconds()
	
	if globalMetrics != nil {
		globalMetrics.RequestsTotal.WithLabelValues("cache_miss", "not_found", c.modelVersion, c.featureVersion).Inc()
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(predErr.HTTPStatus())
	json.NewEncoder(w).Encode(map[string]interface{}{"error": errorResp})
	log.Printf("[%s] [END] Cache miss (no data within ±1 hour) in %dms", requestID, elapsed)
	return
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

	// Step 1: Try Redis cache first (hot cache, 2-hour TTL)
	// Try exact match first
	redisKey := constructRedisCacheKey(gridID, timestampHour)
	pm25Str, err := c.redisClient.Get(ctx, redisKey).Result()
	if err == nil {
		pm25, parseErr := strconv.ParseFloat(pm25Str, 64)
		if parseErr == nil {
			log.Printf("[%s] Redis cache HIT for %s (pm25=%.2f)", requestID, redisKey, pm25)
			return pm25, "redis", nil
		}
		log.Printf("[%s] Redis cache value parse error: %v", requestID, parseErr)
	}

	log.Printf("[%s] Redis cache MISS for %s", requestID, redisKey)

	// Step 2: Try DuckDB pm25_observations table (persistent cache)
	// Serve whatever is there - get the latest PM2.5 value for this grid_id
	query := `
		SELECT pm25
		FROM pm25_observations 
		WHERE grid_id = ?
		ORDER BY timestamp DESC
		LIMIT 1
	`

	var pm25 float64
	row := c.duckdbConn.QueryRowContext(ctx, query, gridID)
	if err := row.Scan(&pm25); err == nil {
		log.Printf("[%s] DuckDB cache HIT for %s (pm25=%.2f)", requestID, gridID, pm25)

		// Populate Redis for next request (async, fire-and-forget)
		go func() {
			bgCtx := context.Background()
			c.redisClient.Set(bgCtx, redisKey, pm25, 2*time.Hour)
		}()

		return pm25, "duckdb", nil
	}

	// Step 3: No data found - return not found
	log.Printf("[%s] No prediction found for grid %s", requestID, gridID)
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

	redisKey := constructRedisCacheKey(gridID, timestampHour)
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
	// Configure logger to use UTC timestamps only
	log.SetFlags(log.Lmicroseconds)
	log.SetOutput(&utcLogWriter{})

	// Initialize metrics globally
	globalMetrics = NewMetricsWithLabels()

	// Load config
	config := DefaultConfig()
	log.Printf("Starting coordinator with config: port=%s, workers=%d, queue_capacity=%d",
		config.ServerPort, config.NumWorkers, config.QueueCapacity)

	// Start orchestrator subprocess manager
	orchManager := NewOrchestratorManager()
	if err := orchManager.Start(); err != nil {
		log.Fatalf("Failed to start orchestrator: %v", err)
	}
	defer orchManager.Shutdown()

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
	orchManager.Shutdown()

	log.Println("Shutdown complete")
}
