package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
)

// utcLogWriter prefixes all log output with UTC timestamps.
type utcLogWriter struct{}

func (w *utcLogWriter) Write(p []byte) (int, error) {
	ts := time.Now().UTC().Format("2006/01/02 15:04:05")
	return os.Stderr.WriteString(fmt.Sprintf("%s UTC %s", ts, p))
}

// corsMiddleware allows browser clients (the pollu-web map) to call the API.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	log.SetFlags(log.Lmicroseconds)
	log.SetOutput(&utcLogWriter{})

	globalMetrics = NewMetrics()
	cfg := DefaultConfig()
	log.Printf("starting coordinator: port=%s kashi=%s redis=%s max_inference=%d",
		cfg.ServerPort, cfg.InferenceURL, cfg.RedisAddr, cfg.MaxConcurrentInference)

	redisClient := redis.NewClient(&redis.Options{
		Addr:         cfg.RedisAddr,
		DialTimeout:  500 * time.Millisecond,
		ReadTimeout:  500 * time.Millisecond,
		WriteTimeout: 500 * time.Millisecond,
		PoolSize:     200,
	})
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		log.Printf("WARNING: redis unreachable (%v) - serving without hot cache until it returns", err)
	}

	coordinator := NewCoordinator(cfg, redisClient, NewInferenceClient(cfg))

	watchCtx, stopWatch := context.WithCancel(context.Background())
	defer stopWatch()
	go coordinator.watchDependencies(watchCtx)

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", coordinator.handlePredict)
	mux.HandleFunc("/health", coordinator.handleHealth)
	mux.Handle("/metrics", promhttp.Handler())

	server := &http.Server{
		Addr:           ":" + cfg.ServerPort,
		Handler:        corsMiddleware(mux),
		ReadTimeout:    5 * time.Second,
		WriteTimeout:   cfg.RequestTimeout + 5*time.Second,
		IdleTimeout:    60 * time.Second,
		MaxHeaderBytes: 16 << 10,
	}

	go func() {
		log.Printf("listening on :%s", cfg.ServerPort)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("server failed: %v", err)
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Println("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
	log.Println("shutdown complete")
}
