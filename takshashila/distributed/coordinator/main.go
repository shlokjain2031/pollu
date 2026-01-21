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
)

// Coordinator is the main coordinator service
type Coordinator struct {
	config *Config
	queue  *JobQueue
	pool   *WorkerPool
}

// PredictRequest is the HTTP request payload
type PredictRequest struct {
	GridID        string `json:"grid_id"`
	TimestampHour string `json:"timestamp_hour"`
}

// PredictResponse is the HTTP response payload
type PredictResponse struct {
	RequestID string  `json:"request_id"`
	PM25      float64 `json:"pm25,omitempty"`
	Error     string  `json:"error,omitempty"`
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

	return &Coordinator{
		config: config,
		queue:  queue,
		pool:   pool,
	}, nil
}

// handlePredict handles POST /predict requests
func (c *Coordinator) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

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

	log.Printf("Received prediction request: %s (grid_id=%s, timestamp=%s)",
		requestID, req.GridID, req.TimestampHour)

	// Create job
	job := Job{
		GridID:        req.GridID,
		TimestampHour: req.TimestampHour,
		RequestID:     requestID,
	}

	// Register result channel before enqueuing
	resultChan := c.pool.RegisterResultChannel(requestID)
	defer c.pool.UnregisterResultChannel(requestID)

	// Enqueue job
	if err := c.queue.Enqueue(job); err != nil {
		log.Printf("Failed to enqueue job %s: %v", requestID, err)

		resp := PredictResponse{
			RequestID: requestID,
			Error:     "Queue is full (server overloaded)",
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(resp)
		return
	}

	log.Printf("Enqueued job %s (queue length: %d)", requestID, c.queue.Len())

	// Wait for result with timeout
	select {
	case result := <-resultChan:
		log.Printf("Received result for request %s: pm25=%.2f, error=%s",
			requestID, result.PM25, result.Error)

		resp := PredictResponse{
			RequestID: requestID,
			PM25:      result.PM25,
			Error:     result.Error,
		}

		w.Header().Set("Content-Type", "application/json")
		if result.Error != "" {
			w.WriteHeader(http.StatusInternalServerError)
		} else {
			w.WriteHeader(http.StatusOK)
		}
		json.NewEncoder(w).Encode(resp)

	case <-time.After(c.config.RequestTimeout):
		log.Printf("Request %s timed out after %v", requestID, c.config.RequestTimeout)

		resp := PredictResponse{
			RequestID: requestID,
			Error:     "Request timed out",
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusGatewayTimeout)
		json.NewEncoder(w).Encode(resp)
	}
}

// handleHealth handles GET /health requests
func (c *Coordinator) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "healthy",
		"queue_length": c.queue.Len(),
		"num_workers":  c.config.NumWorkers,
	})
}

// Shutdown gracefully shuts down the coordinator
func (c *Coordinator) Shutdown() {
	log.Println("Shutting down coordinator...")
	c.pool.Shutdown()
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

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
