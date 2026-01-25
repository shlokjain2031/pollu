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

	// NEW: Redis configuration
	RedisAddr string

	// NEW: DuckDB predictions path
	DuckDBPredictionsPath string
}

// DefaultConfig returns hardcoded defaults for Phase 2a
func DefaultConfig() *Config {
	return &Config{
		ServerPort:            "8080",
		NumWorkers:            2,
		PythonWorkerPath:      "../worker/worker.py",
		QueueCapacity:         100,
		RequestTimeout:        1 * time.Second,
		RedisAddr:             "localhost:6379",
		DuckDBPredictionsPath: "takshashila/duckdb/predictions.duckdb",
	}
}
