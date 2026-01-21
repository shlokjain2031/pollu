package main

import (
	"errors"
	"sync"
)

// Job represents a prediction request to be processed
type Job struct {
	GridID        string `json:"grid_id"`
	TimestampHour string `json:"timestamp_hour"`
	RequestID     string `json:"request_id"`
}

// JobQueue is a bounded queue for job requests
type JobQueue struct {
	capacity int
	queue    chan Job
	mu       sync.RWMutex
	closed   bool
}

// NewJobQueue creates a new bounded queue with the specified capacity
func NewJobQueue(capacity int) *JobQueue {
	return &JobQueue{
		capacity: capacity,
		queue:    make(chan Job, capacity),
		closed:   false,
	}
}

// Enqueue adds a job to the queue
// Returns error if queue is full or closed
func (jq *JobQueue) Enqueue(job Job) error {
	jq.mu.RLock()
	defer jq.mu.RUnlock()

	if jq.closed {
		return errors.New("queue is closed")
	}

	select {
	case jq.queue <- job:
		return nil
	default:
		return errors.New("queue is full")
	}
}

// Dequeue removes and returns a job from the queue
// Blocks until a job is available or queue is closed
func (jq *JobQueue) Dequeue() (Job, bool) {
	job, ok := <-jq.queue
	return job, ok
}

// Len returns the current number of jobs in the queue
func (jq *JobQueue) Len() int {
	return len(jq.queue)
}

// Close closes the queue
func (jq *JobQueue) Close() {
	jq.mu.Lock()
	defer jq.mu.Unlock()

	if !jq.closed {
		jq.closed = true
		close(jq.queue)
	}
}
