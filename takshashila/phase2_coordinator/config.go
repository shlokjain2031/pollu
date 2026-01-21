package main

import "time"

// Config holds all configuration for the Phase 2a coordinator
type Config struct {
	// HTTP server
	ServerPort string

	// Worker management
	NumWorkers       int
	PythonWorkerPath string

	// Queue settings
	QueueCapacity int

	// Timeouts (for Phase 2b, but defined now)
	RequestTimeout time.Duration
}

// DefaultConfig returns hardcoded defaults for Phase 2a
func DefaultConfig() *Config {
	return &Config{
		ServerPort:       "8080",
		NumWorkers:       2,
		PythonWorkerPath: "../phase2_worker/worker.py",
		QueueCapacity:    100,
		RequestTimeout:   1 * time.Second,
	}
}
