package main

import (
	"os"
	"strconv"
	"time"
)

// Config holds coordinator configuration, read from environment with defaults.
type Config struct {
	ServerPort string

	RedisAddr string
	RedisTTL  time.Duration // TTL for predictions written to Redis

	InferenceURL           string        // kashi inference service base URL
	InferenceTimeout       time.Duration // per-call budget for kashi requests
	MaxConcurrentInference int           // bound on in-flight kashi calls

	RequestTimeout time.Duration // total budget for one /predict request
	ResultCacheTTL time.Duration // in-memory result cache TTL
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func envDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}
	return def
}

// DefaultConfig returns the production configuration.
func DefaultConfig() *Config {
	return &Config{
		ServerPort:             envStr("PORT", "8080"),
		RedisAddr:              envStr("REDIS_ADDR", "localhost:6379"),
		RedisTTL:               envDur("REDIS_TTL", time.Hour),
		InferenceURL:           envStr("KASHI_URL", "http://localhost:9999"),
		InferenceTimeout:       envDur("INFERENCE_TIMEOUT", 10*time.Second),
		MaxConcurrentInference: envInt("MAX_CONCURRENT_INFERENCE", 256),
		RequestTimeout:         envDur("REQUEST_TIMEOUT", 15*time.Second),
		ResultCacheTTL:         envDur("RESULT_CACHE_TTL", 5*time.Minute),
	}
}
