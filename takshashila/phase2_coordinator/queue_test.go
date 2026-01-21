package main

import (
	"testing"
)

func TestNewJobQueue(t *testing.T) {
	capacity := 10
	queue := NewJobQueue(capacity)

	if queue == nil {
		t.Fatal("NewJobQueue returned nil")
	}

	if queue.capacity != capacity {
		t.Errorf("Expected capacity %d, got %d", capacity, queue.capacity)
	}

	if queue.Len() != 0 {
		t.Errorf("Expected empty queue, got length %d", queue.Len())
	}
}

func TestJobQueue_Enqueue(t *testing.T) {
	queue := NewJobQueue(2)

	job1 := Job{GridID: "g1", TimestampHour: "2026-01-19T16:00:00Z", RequestID: "req1"}
	job2 := Job{GridID: "g2", TimestampHour: "2026-01-19T17:00:00Z", RequestID: "req2"}
	job3 := Job{GridID: "g3", TimestampHour: "2026-01-19T18:00:00Z", RequestID: "req3"}

	// Enqueue first job
	if err := queue.Enqueue(job1); err != nil {
		t.Errorf("Failed to enqueue job1: %v", err)
	}

	if queue.Len() != 1 {
		t.Errorf("Expected queue length 1, got %d", queue.Len())
	}

	// Enqueue second job
	if err := queue.Enqueue(job2); err != nil {
		t.Errorf("Failed to enqueue job2: %v", err)
	}

	if queue.Len() != 2 {
		t.Errorf("Expected queue length 2, got %d", queue.Len())
	}

	// Try to enqueue third job (should fail - queue full)
	if err := queue.Enqueue(job3); err == nil {
		t.Error("Expected error when enqueueing to full queue, got nil")
	}
}

func TestJobQueue_Dequeue(t *testing.T) {
	queue := NewJobQueue(5)

	job1 := Job{GridID: "g1", TimestampHour: "2026-01-19T16:00:00Z", RequestID: "req1"}
	job2 := Job{GridID: "g2", TimestampHour: "2026-01-19T17:00:00Z", RequestID: "req2"}

	// Enqueue jobs
	queue.Enqueue(job1)
	queue.Enqueue(job2)

	// Dequeue first job
	dequeued, ok := queue.Dequeue()
	if !ok {
		t.Fatal("Dequeue returned not ok")
	}

	if dequeued.RequestID != job1.RequestID {
		t.Errorf("Expected request ID %s, got %s", job1.RequestID, dequeued.RequestID)
	}

	if queue.Len() != 1 {
		t.Errorf("Expected queue length 1, got %d", queue.Len())
	}

	// Dequeue second job
	dequeued, ok = queue.Dequeue()
	if !ok {
		t.Fatal("Dequeue returned not ok")
	}

	if dequeued.RequestID != job2.RequestID {
		t.Errorf("Expected request ID %s, got %s", job2.RequestID, dequeued.RequestID)
	}

	if queue.Len() != 0 {
		t.Errorf("Expected empty queue, got length %d", queue.Len())
	}
}

func TestJobQueue_Close(t *testing.T) {
	queue := NewJobQueue(5)

	job := Job{GridID: "g1", TimestampHour: "2026-01-19T16:00:00Z", RequestID: "req1"}

	// Enqueue before close
	if err := queue.Enqueue(job); err != nil {
		t.Errorf("Failed to enqueue before close: %v", err)
	}

	// Close queue
	queue.Close()

	// Try to enqueue after close (should fail)
	if err := queue.Enqueue(job); err == nil {
		t.Error("Expected error when enqueueing to closed queue, got nil")
	}

	// Dequeue should still work for already enqueued items
	dequeued, ok := queue.Dequeue()
	if !ok {
		t.Error("Expected to dequeue existing item from closed queue")
	}

	if dequeued.RequestID != job.RequestID {
		t.Errorf("Expected request ID %s, got %s", job.RequestID, dequeued.RequestID)
	}

	// Dequeue from empty closed queue should return not ok
	_, ok = queue.Dequeue()
	if ok {
		t.Error("Expected not ok when dequeueing from empty closed queue")
	}
}

func TestJobQueue_FIFO(t *testing.T) {
	queue := NewJobQueue(10)

	jobs := []Job{
		{GridID: "g1", RequestID: "req1"},
		{GridID: "g2", RequestID: "req2"},
		{GridID: "g3", RequestID: "req3"},
	}

	// Enqueue all jobs
	for _, job := range jobs {
		if err := queue.Enqueue(job); err != nil {
			t.Fatalf("Failed to enqueue job %s: %v", job.RequestID, err)
		}
	}

	// Dequeue and verify FIFO order
	for i, expectedJob := range jobs {
		dequeued, ok := queue.Dequeue()
		if !ok {
			t.Fatalf("Dequeue %d returned not ok", i)
		}

		if dequeued.RequestID != expectedJob.RequestID {
			t.Errorf("Job %d: expected request ID %s, got %s",
				i, expectedJob.RequestID, dequeued.RequestID)
		}
	}
}
