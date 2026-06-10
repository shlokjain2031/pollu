package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// InferenceClient talks to the kashi inference service. In-flight calls are
// bounded by a semaphore so a flood of unique cache misses cannot overwhelm
// the Python service.
type InferenceClient struct {
	baseURL string
	client  *http.Client
	sem     chan struct{}
	timeout time.Duration
}

// InferenceMeta describes the model and coverage served by kashi.
type InferenceMeta struct {
	PointCount     int    `json:"point_count"`
	ModelVersion   string `json:"model_version"`
	FeatureVersion string `json:"feature_version"`
}

func NewInferenceClient(cfg *Config) *InferenceClient {
	transport := &http.Transport{
		MaxIdleConns:        cfg.MaxConcurrentInference,
		MaxIdleConnsPerHost: cfg.MaxConcurrentInference,
		IdleConnTimeout:     90 * time.Second,
	}
	return &InferenceClient{
		baseURL: cfg.InferenceURL,
		client:  &http.Client{Transport: transport}, // per-request ctx carries the timeout
		sem:     make(chan struct{}, cfg.MaxConcurrentInference),
		timeout: cfg.InferenceTimeout,
	}
}

// Predict computes PM2.5 for one (lat, lon, hour) via kashi.
// All returned errors are PredictionErrors.
func (ic *InferenceClient) Predict(ctx context.Context, lat, lon float64, timestampHour string) (float64, error) {
	select {
	case ic.sem <- struct{}{}:
		defer func() { <-ic.sem }()
	case <-ctx.Done():
		return 0, NewComputationTimeoutError("inference queue full, request timed out waiting for a slot")
	}

	body := map[string]any{"coords": [][2]float64{{lat, lon}}, "timestamp_hour": timestampHour}
	var resp struct {
		Predictions []float64 `json:"predictions"`
	}
	if err := ic.post(ctx, "/inference", body, &resp); err != nil {
		return 0, err
	}

	if len(resp.Predictions) != 1 {
		return 0, NewWorkerError(fmt.Sprintf("inference returned %d predictions, want 1", len(resp.Predictions)))
	}
	return resp.Predictions[0], nil
}

// Meta fetches coverage/model metadata.
func (ic *InferenceClient) Meta(ctx context.Context) (*InferenceMeta, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ic.baseURL+"/meta", nil)
	if err != nil {
		return nil, err
	}
	resp, err := ic.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("meta returned %d", resp.StatusCode)
	}
	var meta InferenceMeta
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

// Healthy reports whether kashi answers its health endpoint.
func (ic *InferenceClient) Healthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ic.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := ic.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// post sends a JSON request and decodes a JSON response, mapping transport
// and HTTP failures onto PredictionErrors.
func (ic *InferenceClient) post(ctx context.Context, path string, body, out any) error {
	callCtx, cancel := context.WithTimeout(ctx, ic.timeout)
	defer cancel()

	payload, err := json.Marshal(body)
	if err != nil {
		return NewUnknownError(err.Error())
	}
	req, err := http.NewRequestWithContext(callCtx, http.MethodPost, ic.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return NewUnknownError(err.Error())
	}
	req.Header.Set("Content-Type", "application/json")

	if globalMetrics != nil {
		globalMetrics.InflightLeaderComputations.Inc()
		defer globalMetrics.InflightLeaderComputations.Dec()
	}

	resp, err := ic.client.Do(req)
	if err != nil {
		if callCtx.Err() == context.DeadlineExceeded || ctx.Err() == context.DeadlineExceeded {
			return NewComputationTimeoutError(fmt.Sprintf("inference call timed out after %v", ic.timeout))
		}
		log.Printf("inference unreachable: %v", err) // detail stays server-side
		return NewServiceUnavailableError("inference service unavailable, retry shortly")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		detail := kashiErrorDetail(resp.Body)
		switch {
		case resp.StatusCode == http.StatusNotFound:
			return NewNotInCoverageError(detail)
		case resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusUnprocessableEntity:
			return NewInvalidRequestError(detail)
		case resp.StatusCode >= 500:
			return NewServiceUnavailableError(fmt.Sprintf("inference service error %d: %s", resp.StatusCode, detail))
		default:
			return NewWorkerError(fmt.Sprintf("inference service returned %d: %s", resp.StatusCode, detail))
		}
	}

	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return NewWorkerError(fmt.Sprintf("invalid inference response: %v", err))
	}
	return nil
}

// kashiErrorDetail extracts FastAPI's {"detail": ...} message, if present.
func kashiErrorDetail(r io.Reader) string {
	raw, _ := io.ReadAll(io.LimitReader(r, 4096))
	var parsed struct {
		Detail string `json:"detail"`
	}
	if json.Unmarshal(raw, &parsed) == nil && parsed.Detail != "" {
		return parsed.Detail
	}
	return string(raw)
}
