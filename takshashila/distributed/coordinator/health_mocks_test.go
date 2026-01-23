package main

import (
	"io"
	"sync"
)

// Test doubles for health.go unit testing

// MockGauge implements Gauge for testing
type MockGauge struct {
	value float64
}

func (mg *MockGauge) Set(v float64) {
	mg.value = v
}

// MockCounter implements Counter for testing
type MockCounter struct {
	value int64
}

func (mc *MockCounter) Inc() {
	mc.value++
}

// MockHistogram implements Histogram for testing
type MockHistogram struct {
	values []float64
}

func (mh *MockHistogram) Observe(v float64) {
	mh.values = append(mh.values, v)
}

// MockLabeledGauge implements LabeledGauge for testing
type MockLabeledGauge struct {
	gauges map[string]*MockGauge
	mu     sync.Mutex
}

func (mlg *MockLabeledGauge) WithLabelValues(labels ...string) Gauge {
	mlg.mu.Lock()
	defer mlg.mu.Unlock()
	key := labels[0]
	if mlg.gauges == nil {
		mlg.gauges = make(map[string]*MockGauge)
	}
	if _, ok := mlg.gauges[key]; !ok {
		mlg.gauges[key] = &MockGauge{}
	}
	return mlg.gauges[key]
}

// MockLabeledCounter implements LabeledCounter for testing
type MockLabeledCounter struct {
	counters map[string]*MockCounter
	mu       sync.Mutex
}

func (mlc *MockLabeledCounter) WithLabelValues(labels ...string) Counter {
	mlc.mu.Lock()
	defer mlc.mu.Unlock()
	key := labels[0]
	if mlc.counters == nil {
		mlc.counters = make(map[string]*MockCounter)
	}
	if _, ok := mlc.counters[key]; !ok {
		mlc.counters[key] = &MockCounter{}
	}
	return mlc.counters[key]
}

// MockLabeledHistogram implements LabeledHistogram for testing
type MockLabeledHistogram struct {
	histograms map[string]*MockHistogram
	mu         sync.Mutex
}

func (mlh *MockLabeledHistogram) WithLabelValues(labels ...string) Histogram {
	mlh.mu.Lock()
	defer mlh.mu.Unlock()
	key := labels[0]
	if mlh.histograms == nil {
		mlh.histograms = make(map[string]*MockHistogram)
	}
	if _, ok := mlh.histograms[key]; !ok {
		mlh.histograms[key] = &MockHistogram{}
	}
	return mlh.histograms[key]
}

// TestPythonWorker is a test double for PythonWorker
type TestPythonWorker struct {
	ID     string
	cmd    interface{} // unused in tests
	stdin  *TestStdin
	stdout io.ReadCloser
	mu     sync.Mutex
	active bool
}

// TestStdin captures writes for testing
type TestStdin struct {
	writes [][]byte
	mu     sync.Mutex
}

func (ts *TestStdin) Write(p []byte) (n int, err error) {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	ts.writes = append(ts.writes, append([]byte{}, p...))
	return len(p), nil
}

func (ts *TestStdin) Close() error {
	return nil
}

// Gauge interface for Prometheus-like metrics
type Gauge interface {
	Set(float64)
}

// Counter interface for Prometheus-like metrics
type Counter interface {
	Inc()
}

// Histogram interface for Prometheus-like metrics
type Histogram interface {
	Observe(float64)
}

// LabeledGauge interface
type LabeledGauge interface {
	WithLabelValues(labels ...string) Gauge
}

// LabeledCounter interface
type LabeledCounter interface {
	WithLabelValues(labels ...string) Counter
}

// LabeledHistogram interface
type LabeledHistogram interface {
	WithLabelValues(labels ...string) Histogram
}

// MetricsCollector stub
type MetricsCollector struct {
	WorkerHealthStatus              LabeledGauge
	WorkerHealthChecksTotal         LabeledCounter
	WorkerHealthCheckFailuresTotal  LabeledCounter
	WorkerHealthCheckLatencySeconds LabeledHistogram
}
