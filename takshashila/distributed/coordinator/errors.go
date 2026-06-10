package main

import (
	"fmt"
	"time"
)

// ErrorType represents the classification of prediction errors
type ErrorType int

const (
	ErrorTypeInvalidRequest ErrorType = iota
	ErrorTypeNotInCoverage
	ErrorTypeDataMissing
	ErrorTypeComputationTimeout
	ErrorTypeWorkerError
	ErrorTypePredictionNotAvailable
	ErrorTypeServiceUnavailable
	ErrorTypeUnknown
)

// String returns the error type as a string
func (e ErrorType) String() string {
	switch e {
	case ErrorTypeInvalidRequest:
		return "InvalidRequest"
	case ErrorTypeNotInCoverage:
		return "NotInCoverage"
	case ErrorTypeDataMissing:
		return "DataMissing"
	case ErrorTypeComputationTimeout:
		return "ComputationTimeout"
	case ErrorTypeWorkerError:
		return "WorkerError"
	case ErrorTypePredictionNotAvailable:
		return "PredictionNotAvailable"
	case ErrorTypeServiceUnavailable:
		return "ServiceUnavailable"
	case ErrorTypeUnknown:
		return "Unknown"
	default:
		return "Unknown"
	}
}

// PredictionError is the interface for all prediction errors
type PredictionError interface {
	error
	Type() ErrorType
	HTTPStatus() int
	Retryable() bool
	Cacheable() bool
	CacheTTL() time.Duration
}

// predictionError is the concrete implementation
type predictionError struct {
	errType    ErrorType
	message    string
	httpStatus int
	retryable  bool
	cacheable  bool
	cacheTTL   time.Duration
}

// Error implements the error interface
func (e *predictionError) Error() string {
	return e.message
}

// Type returns the error type
func (e *predictionError) Type() ErrorType {
	return e.errType
}

// HTTPStatus returns the HTTP status code
func (e *predictionError) HTTPStatus() int {
	return e.httpStatus
}

// Retryable returns whether the error is retryable
func (e *predictionError) Retryable() bool {
	return e.retryable
}

// Cacheable returns whether the error should be cached
func (e *predictionError) Cacheable() bool {
	return e.cacheable
}

// CacheTTL returns the cache TTL for this error
func (e *predictionError) CacheTTL() time.Duration {
	return e.cacheTTL
}

// Error constructors

// NewInvalidRequestError creates a 400 non-retryable error (not cached)
func NewInvalidRequestError(message string) PredictionError {
	return &predictionError{
		errType:    ErrorTypeInvalidRequest,
		message:    message,
		httpStatus: 400,
		retryable:  false,
		cacheable:  false,
		cacheTTL:   0,
	}
}

// NewNotInCoverageError creates a 404 non-retryable error (cached 5 minutes)
// for a coordinate outside the model's coverage area.
func NewNotInCoverageError(message string) PredictionError {
	return &predictionError{
		errType:    ErrorTypeNotInCoverage,
		message:    message,
		httpStatus: 404,
		retryable:  false,
		cacheable:  true,
		cacheTTL:   5 * time.Minute,
	}
}

// NewServiceUnavailableError creates a 503 retryable error (cached 10s to
// shed load while the inference service is down or restarting)
func NewServiceUnavailableError(message string) PredictionError {
	return &predictionError{
		errType:    ErrorTypeServiceUnavailable,
		message:    message,
		httpStatus: 503,
		retryable:  true,
		cacheable:  true,
		cacheTTL:   10 * time.Second,
	}
}

// NewDataMissingError creates a 503 retryable error (cached 30-60s)
func NewDataMissingError(message string) PredictionError {
	return &predictionError{
		errType:    ErrorTypeDataMissing,
		message:    message,
		httpStatus: 503,
		retryable:  true,
		cacheable:  true,
		cacheTTL:   45 * time.Second, // 45s as middle ground
	}
}

// NewPredictionNotAvailableError creates a 404 retryable error for missing predictions
// Used when prediction doesn't exist in Redis or DuckDB (orchestrator hasn't computed yet)
// Retryable because orchestrator may still be computing, cacheable for brief period to avoid hammering
func NewPredictionNotAvailableError(gridID, timestampHour string) PredictionError {
	return &predictionError{
		errType:    ErrorTypePredictionNotAvailable,
		message:    fmt.Sprintf("Prediction not available for grid_id=%s, timestamp_hour=%s. Orchestrator may still be computing, retry in 30-60 seconds.", gridID, timestampHour),
		httpStatus: 404,
		retryable:  true,
		cacheable:  true,
		cacheTTL:   30 * time.Second, // Cache miss briefly to avoid hammering DB
	}
}

// NewComputationTimeoutError creates a 504 retryable error (cached 5s for load shedding)
func NewComputationTimeoutError(message string) PredictionError {
	return &predictionError{
		errType:    ErrorTypeComputationTimeout,
		message:    message,
		httpStatus: 504,
		retryable:  true,
		cacheable:  true,
		cacheTTL:   5 * time.Second,
	}
}

// NewWorkerError creates a 500 retryable error (not cached)
func NewWorkerError(message string) PredictionError {
	return &predictionError{
		errType:    ErrorTypeWorkerError,
		message:    message,
		httpStatus: 500,
		retryable:  true,
		cacheable:  false,
		cacheTTL:   0,
	}
}

// NewUnknownError creates a 500 retryable error (not cached)
func NewUnknownError(message string) PredictionError {
	return &predictionError{
		errType:    ErrorTypeUnknown,
		message:    message,
		httpStatus: 500,
		retryable:  true,
		cacheable:  false,
		cacheTTL:   0,
	}
}

// ErrorResponse represents the structured error response sent to clients
type ErrorResponse struct {
	ErrorType  string `json:"error_type"`
	Message    string `json:"message"`
	HTTPStatus int    `json:"http_status"`
	Retryable  bool   `json:"retryable"`
	RetryAfter *int   `json:"retry_after,omitempty"` // seconds
	TTL        *int   `json:"ttl,omitempty"`         // seconds
}

// NewErrorResponse creates a structured error response from a PredictionError
func NewErrorResponse(err PredictionError) ErrorResponse {
	resp := ErrorResponse{
		ErrorType:  err.Type().String(),
		Message:    err.Error(),
		HTTPStatus: err.HTTPStatus(),
		Retryable:  err.Retryable(),
	}

	// Add retry_after for retryable errors
	if err.Retryable() && err.Cacheable() {
		ttlSeconds := int(err.CacheTTL().Seconds())
		resp.RetryAfter = &ttlSeconds
		resp.TTL = &ttlSeconds
	}

	return resp
}

// ClassifyError converts a generic error into a PredictionError
// This is used to wrap errors from the worker
func ClassifyError(err error) PredictionError {
	if err == nil {
		return nil
	}

	// If already a PredictionError, return as-is
	if predErr, ok := err.(PredictionError); ok {
		return predErr
	}

	// Check for context.DeadlineExceeded
	if err.Error() == "context deadline exceeded" {
		return NewComputationTimeoutError("Computation timed out after 1 second")
	}

	// Otherwise treat as unknown error
	return NewUnknownError(err.Error())
}
