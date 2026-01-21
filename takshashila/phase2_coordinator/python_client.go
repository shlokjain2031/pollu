package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
)

// WorkerResult represents the result from a Python worker
type WorkerResult struct {
	RequestID string  `json:"request_id"`
	PM25      float64 `json:"pm25"`
	Error     string  `json:"error"`
	WorkerID  string  `json:"worker_id"`
}

// PythonWorker represents a single Python worker process
type PythonWorker struct {
	ID      string
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  io.ReadCloser
	scanner *bufio.Scanner
	mu      sync.Mutex
	active  bool
}

// WorkerPool manages a pool of Python workers
type WorkerPool struct {
	workers       []*PythonWorker
	queue         *JobQueue
	resultChans   map[string]chan WorkerResult
	resultChansMu sync.RWMutex
	wg            sync.WaitGroup
}

// NewWorkerPool creates a new pool of Python workers
func NewWorkerPool(numWorkers int, pythonPath string, queue *JobQueue) (*WorkerPool, error) {
	pool := &WorkerPool{
		workers:     make([]*PythonWorker, 0, numWorkers),
		queue:       queue,
		resultChans: make(map[string]chan WorkerResult),
	}

	// Spawn workers
	for i := 0; i < numWorkers; i++ {
		worker, err := spawnWorker(fmt.Sprintf("worker-%d", i), pythonPath)
		if err != nil {
			// Clean up already spawned workers
			pool.Shutdown()
			return nil, fmt.Errorf("failed to spawn worker %d: %w", i, err)
		}
		pool.workers = append(pool.workers, worker)

		// Start result reader for this worker
		pool.wg.Add(1)
		go pool.readResults(worker)

		log.Printf("Spawned worker: %s", worker.ID)
	}

	// Start job dispatcher
	pool.wg.Add(1)
	go pool.dispatchJobs()

	return pool, nil
}

// spawnWorker starts a single Python worker process
func spawnWorker(id, pythonPath string) (*PythonWorker, error) {
	cmd := exec.Command("python", pythonPath)

	// Set environment variable so worker knows its ID
	cmd.Env = append(os.Environ(), fmt.Sprintf("WORKER_ID=%s", id))

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, err
	}

	// Capture stderr to see worker errors
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdin.Close()
		stdout.Close()
		return nil, err
	}

	// Log stderr in background
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("[%s stderr] %s", id, scanner.Text())
		}
	}()

	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		return nil, err
	}

	log.Printf("[%s] Process started (PID: %d)", id, cmd.Process.Pid)

	log.Printf("[%s] Process started (PID: %d)", id, cmd.Process.Pid)

	worker := &PythonWorker{
		ID:      id,
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		scanner: bufio.NewScanner(stdout),
		active:  true,
	}

	return worker, nil
}

// dispatchJobs reads jobs from the queue and sends them to workers round-robin
func (wp *WorkerPool) dispatchJobs() {
	defer wp.wg.Done()

	workerIdx := 0
	for job := range wp.queue.queue {
		// Round-robin worker selection
		worker := wp.workers[workerIdx]
		workerIdx = (workerIdx + 1) % len(wp.workers)

		// Send job to worker
		if err := wp.sendJob(worker, job); err != nil {
			log.Printf("Failed to send job %s to worker %s: %v", job.RequestID, worker.ID, err)

			// Send error result
			wp.resultChansMu.RLock()
			if ch, ok := wp.resultChans[job.RequestID]; ok {
				ch <- WorkerResult{
					RequestID: job.RequestID,
					Error:     fmt.Sprintf("worker communication error: %v", err),
					WorkerID:  worker.ID,
				}
			}
			wp.resultChansMu.RUnlock()
		}
	}
}

// sendJob sends a job to a specific worker
func (wp *WorkerPool) sendJob(worker *PythonWorker, job Job) error {
	worker.mu.Lock()
	defer worker.mu.Unlock()

	if !worker.active {
		return fmt.Errorf("worker %s is not active", worker.ID)
	}

	jobJSON, err := json.Marshal(job)
	if err != nil {
		return err
	}

	_, err = worker.stdin.Write(append(jobJSON, '\n'))
	if err != nil {
		worker.active = false
		return err
	}

	log.Printf("Sent job %s to worker %s", job.RequestID, worker.ID)
	return nil
}

// readResults continuously reads results from a worker
func (wp *WorkerPool) readResults(worker *PythonWorker) {
	defer wp.wg.Done()

	for worker.scanner.Scan() {
		line := worker.scanner.Text()

		var result WorkerResult
		if err := json.Unmarshal([]byte(line), &result); err != nil {
			log.Printf("Failed to parse result from worker %s: %v", worker.ID, err)
			continue
		}

		log.Printf("Received result from worker %s for request %s", worker.ID, result.RequestID)

		// Send result to the waiting channel
		wp.resultChansMu.RLock()
		if ch, ok := wp.resultChans[result.RequestID]; ok {
			ch <- result
		} else {
			log.Printf("No result channel found for request %s", result.RequestID)
		}
		wp.resultChansMu.RUnlock()
	}

	if err := worker.scanner.Err(); err != nil {
		log.Printf("Error reading from worker %s: %v", worker.ID, err)
	}

	worker.mu.Lock()
	worker.active = false
	worker.mu.Unlock()

	log.Printf("[%s] Process exited (scanning stopped)", worker.ID)
}

// RegisterResultChannel registers a channel to receive the result for a request
func (wp *WorkerPool) RegisterResultChannel(requestID string) chan WorkerResult {
	wp.resultChansMu.Lock()
	defer wp.resultChansMu.Unlock()

	ch := make(chan WorkerResult, 1)
	wp.resultChans[requestID] = ch
	return ch
}

// UnregisterResultChannel removes a result channel
func (wp *WorkerPool) UnregisterResultChannel(requestID string) {
	wp.resultChansMu.Lock()
	defer wp.resultChansMu.Unlock()

	if ch, ok := wp.resultChans[requestID]; ok {
		close(ch)
		delete(wp.resultChans, requestID)
	}
}

// Shutdown gracefully shuts down all workers
func (wp *WorkerPool) Shutdown() {
	log.Println("Shutting down worker pool...")

	// Close queue to stop dispatcher
	wp.queue.Close()

	// Close worker stdin to signal shutdown
	for _, worker := range wp.workers {
		worker.mu.Lock()
		if worker.stdin != nil {
			worker.stdin.Close()
		}
		worker.mu.Unlock()
	}

	// Wait for all goroutines to finish
	wp.wg.Wait()

	// Kill any remaining processes
	for _, worker := range wp.workers {
		if worker.cmd != nil && worker.cmd.Process != nil {
			worker.cmd.Process.Kill()
		}
	}

	log.Println("Worker pool shutdown complete")
}
