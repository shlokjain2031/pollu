# Persistence Queue Implementation Plan

**Status:** Design Phase  
**Date:** January 24, 2026  
**Scope:** Simple serial queue for predictions table writes  
**Goal:** Decouple write latency from response latency, fix 94% write failure rate

---

## 1. Architecture Overview

### Component: `PersistenceQueue`

```
User Request
    ↓
InferenceEngine.predict()
    ├─ Feature assembly (30ms)
    ├─ Model inference (5ms)
    ├─ queue.enqueue(prediction) (~0.1ms) ← RETURNS TO USER
    └─ Return response (35ms total)
    
    ↓
PersistenceQueue (background thread)
    ├─ Dequeue prediction
    ├─ Write to DB (10-20ms sequential, no lock contention)
    ├─ Track metrics
    └─ Handle errors (retry with exponential backoff)
```

### Key Properties

- **Unbounded queue** (no size limit, memory sufficient)
- **Single background thread** (sequential writes, no contention)
- **Strict FIFO ordering** (predictions written in order received)
- **Queue owns lifecycle** (thread management internal to queue)
- **Encapsulated failure handling** (retries, notifications internal)

---

## 2. Data Structures

### 2.1 Prediction Item (to enqueue)

```python
@dataclass
class PredictionItem:
    """Single prediction to persist."""
    grid_id: str                    # Grid cell ID
    timestamp_hour: datetime        # Prediction timestamp
    pm25_prediction: float          # Model output
    model_version: str              # Model identifier
    schema_hash: str                # Feature schema hash
    
    # Internal tracking
    enqueued_at: datetime           # When added to queue
    retry_count: int = 0            # Number of write failures
    last_error: Optional[str] = None  # Last error message
```

### 2.2 Queue Metrics (tracked continuously)

```python
@dataclass
class PersistenceQueueMetrics:
    """Metrics for queue health and performance."""
    
    # Counts
    total_enqueued: int = 0         # Total predictions enqueued
    total_written: int = 0          # Successfully persisted
    total_failed: int = 0           # Failed permanently (dropped)
    queue_size: int = 0             # Current items waiting
    
    # Latency (milliseconds)
    last_write_latency_ms: float = 0.0
    avg_write_latency_ms: float = 0.0
    max_write_latency_ms: float = 0.0
    
    # Error tracking
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    error_count_last_hour: int = 0
    
    # Thread health
    worker_thread_alive: bool = False
    worker_thread_restarts: int = 0
    last_thread_restart: Optional[datetime] = None
    
    def report(self) -> str:
        """Human-readable metrics summary."""
        return (
            f"Queue size: {self.queue_size} | "
            f"Enqueued: {self.total_enqueued} | "
            f"Written: {self.total_written} | "
            f"Failed: {self.total_failed} | "
            f"Avg latency: {self.avg_write_latency_ms:.2f}ms | "
            f"Worker restarts: {self.worker_thread_restarts}"
        )
```

### 2.3 Queue State Exception

```python
class PersistenceQueueException(Exception):
    """Raised when queue lifecycle/state is invalid."""
    pass

class WorkerThreadException(Exception):
    """Raised when worker thread fails after max retries."""
    pass
```

---

## 3. PersistenceQueue Class

### 3.1 Constructor & Initialization

```python
class PersistenceQueue:
    """FIFO queue for sequential prediction persistence.
    
    Owns:
    - Queue data structure (unbounded)
    - Background worker thread
    - Thread lifecycle management (start, stop, restart)
    - Failure handling (retry, notifications)
    - Metrics tracking
    - Error logging
    """
    
    def __init__(
        self,
        db_path: str,
        initial_retry_delay_ms: float = 100,
        max_retry_delay_ms: float = 10000,
        max_retries_before_notify: int = 3,
    ):
        """Initialize queue.
        
        Args:
            db_path: Path to predictions.duckdb
            initial_retry_delay_ms: Starting delay for exponential backoff
            max_retry_delay_ms: Maximum backoff delay
            max_retries_before_notify: After this many failures, notify owner
        
        State initialized:
        - Queue created (empty)
        - Metrics created
        - Background thread created and started
        - Database connection verified
        """
```

**Validation:**
- Verify `db_path` exists and is readable
- Verify predictions table exists in database
- Verify can create write connection
- Raise `PersistenceQueueException` if validation fails

### 3.2 Public Methods

#### `enqueue(prediction: PredictionItem) -> None`

```python
def enqueue(self, prediction: PredictionItem) -> None:
    """Add prediction to persistence queue.
    
    Args:
        prediction: PredictionItem to persist
    
    Behavior:
    - Immediately appends to FIFO queue (~0.1ms)
    - Sets enqueued_at timestamp
    - Updates metrics.queue_size
    - Returns immediately (non-blocking)
    - Raises PersistenceQueueException if queue is closed
    
    Thread safety:
    - Thread-safe (uses queue.Queue internally)
    """
```

**Implementation notes:**
- Wrap prediction in PredictionItem if needed (or require caller to create it)
- Log at DEBUG level: "Enqueued prediction: {grid_id} @ {timestamp}"
- Increment `metrics.total_enqueued`
- Update `metrics.queue_size`

#### `metrics() -> PersistenceQueueMetrics`

```python
def metrics(self) -> PersistenceQueueMetrics:
    """Get current queue metrics (thread-safe).
    
    Returns:
    - Copy of current metrics (safe to read without blocking)
    
    No blocking, no side effects.
    """
```

#### `is_healthy() -> bool`

```python
def is_healthy(self) -> bool:
    """Check if queue and worker thread are operational.
    
    Returns:
    - True if worker thread is alive
    - False if thread died (caller should investigate)
    
    Used by monitoring/health checks.
    """
```

#### `shutdown(timeout_sec: float = 30) -> bool`

```python
def shutdown(self, timeout_sec: float = 30) -> bool:
    """Gracefully shutdown queue and flush pending writes.
    
    Behavior:
    1. Stop accepting new enqueues (raise if attempted)
    2. Worker continues draining queue
    3. Wait up to timeout_sec for queue to empty
    4. Stop worker thread
    5. Close database connection
    
    Args:
        timeout_sec: Max time to wait for queue flush
    
    Returns:
    - True if queue flushed successfully
    - False if timeout (some items may not be written)
    
    Logs:
    - "Shutdown initiated"
    - "Shutdown complete: {N} items written, {M} items remaining"
    
    Side effects:
    - Queue is closed (no further enqueues allowed)
    - Cannot restart queue after shutdown
    """
```

**Implementation notes:**
- Set `_shutdown_requested = True`
- Reject new enqueues if shutdown requested
- Join worker thread with timeout
- Return True if queue empty and thread stopped, False if timeout
- Always close DB connection in finally block

### 3.3 Internal Methods

#### `_worker_loop() -> None` (runs in background thread)

```python
def _worker_loop(self) -> None:
    """Main loop for background persistence worker.
    
    Behavior:
    1. Loop while not shutdown or queue not empty
    2. Get next item from queue (blocking, timeout=1s)
    3. Attempt write with retry logic
    4. Update metrics
    5. Log results
    
    Error handling:
    - On write failure: put item back in queue
    - Exponential backoff: wait before retry
    - After 3 failures: notify owner via callback
    - Thread continues (doesn't exit on error)
    
    Lifecycle:
    - Runs in dedicated thread
    - Graceful shutdown on _shutdown_requested flag
    - Logs startup and shutdown
    """
```

**Algorithm:**

```
while not _shutdown_requested or not queue.empty():
    try:
        # Get next item (wait up to 1 second)
        item = queue.get(timeout=1)
        
        # Attempt write
        write_time = measure_time(lambda: _write_item(item))
        
        # Success
        metrics.total_written += 1
        metrics.queue_size = queue.qsize()
        metrics.last_write_latency_ms = write_time
        update_avg_latency(write_time)
        
        log(f"Wrote: {item.grid_id} ({write_time:.2f}ms)")
        
    except queue.Empty:
        # Timeout waiting for item, continue
        continue
        
    except Exception as e:
        # Write failed, retry with backoff
        item.retry_count += 1
        item.last_error = str(e)
        
        if item.retry_count <= max_retries_before_notify:
            # Put back in queue, wait exponential backoff
            backoff = exponential_backoff(item.retry_count)
            log(f"Write failed for {item.grid_id}: {e}, retry after {backoff}ms")
            sleep(backoff)
            queue.put(item)
        else:
            # Max retries exceeded, drop and notify
            metrics.total_failed += 1
            log(f"DROPPED {item.grid_id} after {item.retry_count} failures: {e}")
            notify_owner_of_failure(item)
```

#### `_write_item(item: PredictionItem) -> None`

```python
def _write_item(self, item: PredictionItem) -> None:
    """Write single prediction to database.
    
    Behavior:
    1. Get write connection from pool (or new connection)
    2. Execute INSERT statement
    3. Commit transaction
    4. Close connection
    
    Args:
        item: PredictionItem with all required fields
    
    Raises:
        Exception: If INSERT or commit fails
    
    Note:
    - Uses WriterConnection (non-pooled, dedicated)
    - Sequential execution prevents lock contention
    """
```

**Implementation notes:**
- Use `WriterConnection` if available (from Part 2), or create new connection
- SQL: `INSERT INTO predictions (grid_id, timestamp_hour, pm25_prediction, model_version, schema_hash) VALUES (?, ?, ?, ?, ?)`
- Commit after insert
- Let exceptions propagate (caller handles retry logic)

#### `_exponential_backoff(retry_count: int) -> float`

```python
def _exponential_backoff(self, retry_count: int) -> float:
    """Calculate delay for nth retry.
    
    Formula:
    delay = min(initial_delay * 2^(retry_count - 1), max_delay)
    
    Examples (initial=100ms, max=10000ms):
    - Retry 1: 100ms
    - Retry 2: 200ms
    - Retry 3: 400ms
    - Retry 4: 800ms
    - ...
    - Retry 7+: 10000ms (capped)
    
    Returns:
    - Delay in milliseconds
    """
```

#### `_notify_owner_of_failure(item: PredictionItem) -> None`

```python
def _notify_owner_of_failure(self, item: PredictionItem) -> None:
    """Notify caller that prediction write failed permanently.
    
    Behavior:
    - Log ERROR message
    - Update metrics.last_error_time, last_error_message
    - Increment metrics.error_count_last_hour
    - Call optional failure callback (if provided)
    
    Note:
    - Called after 3 failed retry attempts
    - Item is dropped (not written to DB)
    - Thread continues (doesn't stop)
    """
```

#### `_restart_worker_thread() -> None` (optional, for future resilience)

```python
def _restart_worker_thread(self) -> None:
    """Restart worker thread if it died unexpectedly.
    
    Behavior:
    - Check if thread is alive
    - If dead, create new thread and start
    - Log restart event
    - Increment metrics.worker_thread_restarts
    
    Called by:
    - Periodic health check (every 10s)
    - enqueue() if worker is dead
    """
```

---

## 4. Integration Points

### 4.1 InferenceEngine Integration

**Location:** `takshashila/core/inference.py`

```python
class InferenceEngine:
    def __init__(self, ..., use_persistence_queue: bool = True):
        """Initialize inference engine.
        
        Args:
            use_persistence_queue: If True, use async queue for writes
                                   If False, write immediately (legacy mode)
        
        Creates:
        - PersistenceQueue if use_persistence_queue=True
        - Stores as self._persistence_queue
        """
    
    def predict(self, grid_id: str, timestamp_hour: datetime) -> float:
        """Compute PM2.5 prediction.
        
        Modified behavior with queue:
        1. Assemble features (same as before)
        2. Run model (same as before)
        3. If queue enabled:
            - Create PredictionItem
            - queue.enqueue(item) (~0.1ms)
            - Return prediction immediately
        4. Else (legacy):
            - Write to DB directly (blocking)
            - Return prediction
        
        User gets response in ~35ms (with queue) vs ~50ms (without queue)
        """
```

**Question deferred to worker implementation phase:**
- Should PersistenceQueue be injected from outside, or created internally by InferenceEngine?

### 4.2 Testing Integration

**Unit tests location:** `takshashila/tests/test_persistence_queue.py`

Tests to implement (see section 5.1 for details):
1. Enqueue and dequeue basic cycle
2. FIFO ordering (multiple items)
3. Concurrent enqueues (10+ parallel threads)
4. Metrics tracking (accurate counts and latency)
5. Write failure and retry logic
6. Item re-enqueueing on write failure
7. Max retries exceeded (item dropped)
8. Owner notification after 3 failures
9. Graceful shutdown (queue drained before exit)
10. Shutdown timeout (items remaining)
11. Thread restart on death
12. Database connection failures
13. Predicate item validation

### 4.3 Load Test Integration

**Location:** Extend `takshashila/load_test.py`

Add test variant:
- Run same 3 load profiles (Baseline, Sustained, Stress) WITH persistence queue
- Compare metrics:
  - Success rate (expect: 100% instead of 5%)
  - Response latency (expect: ~35ms instead of blocked on write)
  - Write latency distribution (should be sequential, no lock waits)
  - Queue size over time (should drain smoothly)

---

## 5. Testing Strategy

### 5.1 Unit Tests (13 tests total)

**File:** `takshashila/tests/test_persistence_queue.py`

#### Test 1: Basic Enqueue/Dequeue

```python
def test_enqueue_and_write_single_prediction(test_db_paths):
    """Test basic enqueue and background write cycle."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Enqueue
    item = PredictionItem(
        grid_id='g00000001',
        timestamp_hour=datetime(2026, 1, 19, 14, 0, 0),
        pm25_prediction=45.0,
        model_version='v1',
        schema_hash='hash123'
    )
    queue.enqueue(item)
    
    # Wait for background write
    time.sleep(0.5)
    
    # Verify written
    conn = duckdb.connect(test_db_paths['predictions'], read_only=True)
    result = conn.execute(
        "SELECT pm25_prediction FROM predictions WHERE grid_id = ?",
        ['g00000001']
    ).fetchone()
    assert result[0] == 45.0
    
    # Check metrics
    assert queue.metrics().total_enqueued == 1
    assert queue.metrics().total_written == 1
    
    queue.shutdown()
```

#### Test 2: FIFO Ordering

```python
def test_strict_fifo_ordering(test_db_paths):
    """Test that predictions are written in FIFO order."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Enqueue 10 items in order
    for i in range(10):
        item = PredictionItem(
            grid_id=f'g{i:08d}',
            timestamp_hour=datetime(2026, 1, 19, 14, 0, 0),
            pm25_prediction=float(40 + i),
            model_version='v1',
            schema_hash='hash123'
        )
        queue.enqueue(item)
    
    # Wait for all to write
    time.sleep(1)
    
    # Verify order in DB (by checking insertion order)
    conn = duckdb.connect(test_db_paths['predictions'], read_only=True)
    results = conn.execute(
        "SELECT grid_id, pm25_prediction FROM predictions ORDER BY rowid"
    ).fetchall()
    
    assert len(results) == 10
    for i, (grid_id, pm25) in enumerate(results):
        assert grid_id == f'g{i:08d}'
        assert pm25 == float(40 + i)
    
    queue.shutdown()
```

#### Test 3: Concurrent Enqueues (10+ threads)

```python
def test_concurrent_enqueues(test_db_paths):
    """Test thread-safety of enqueue with 10+ parallel threads."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    num_items_per_thread = 100
    num_threads = 10
    
    def enqueue_items(thread_id):
        for i in range(num_items_per_thread):
            item = PredictionItem(
                grid_id=f't{thread_id}_i{i}',
                timestamp_hour=datetime(2026, 1, 19, 14, 0, 0),
                pm25_prediction=float(thread_id * 100 + i),
                model_version='v1',
                schema_hash='hash123'
            )
            queue.enqueue(item)
    
    threads = [Thread(target=enqueue_items, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Wait for writes
    time.sleep(2)
    
    # Verify all written
    assert queue.metrics().total_enqueued == num_threads * num_items_per_thread
    assert queue.metrics().total_written == num_threads * num_items_per_thread
    
    queue.shutdown()
```

#### Test 4: Metrics Accuracy

```python
def test_metrics_tracking(test_db_paths):
    """Test that metrics are tracked accurately."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Enqueue 5 items
    for i in range(5):
        item = PredictionItem(...)
        queue.enqueue(item)
    
    # Check metrics
    metrics = queue.metrics()
    assert metrics.total_enqueued == 5
    assert metrics.queue_size >= 0
    
    # Wait for write
    time.sleep(0.5)
    
    # Check after write
    metrics = queue.metrics()
    assert metrics.total_written == 5
    assert metrics.queue_size == 0
    assert metrics.last_write_latency_ms > 0
    assert metrics.avg_write_latency_ms > 0
    
    queue.shutdown()
```

#### Test 5: Write Failure and Retry

```python
def test_write_failure_and_retry(test_db_paths):
    """Test that failed writes are retried with exponential backoff."""
    queue = PersistenceQueue(
        test_db_paths['predictions'],
        initial_retry_delay_ms=10,  # Fast for testing
        max_retries_before_notify=3
    )
    
    # Mock write failure (corrupt DB connection)
    original_write = queue._write_item
    call_count = [0]
    
    def failing_write(item):
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("Simulated write failure")
        return original_write(item)
    
    queue._write_item = failing_write
    
    # Enqueue item
    item = PredictionItem(grid_id='g00000001', ...)
    queue.enqueue(item)
    
    # Wait for retries
    time.sleep(1)
    
    # Verify retried and eventually succeeded
    assert call_count[0] >= 3
    assert queue.metrics().total_written == 1
    
    queue.shutdown()
```

#### Test 6: Item Re-enqueueing

```python
def test_item_requeued_on_failure(test_db_paths):
    """Test that failed items are put back in queue."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Track re-enqueues
    original_put = queue._queue.put
    put_count = [0]
    
    def counting_put(item):
        put_count[0] += 1
        return original_put(item)
    
    queue._queue.put = counting_put
    
    # Force a write failure
    original_write = queue._write_item
    def failing_write(item):
        if item.retry_count == 0:
            raise Exception("First attempt fails")
        return original_write(item)
    
    queue._write_item = failing_write
    
    # Enqueue
    item = PredictionItem(grid_id='g00000001', ...)
    queue.enqueue(item)
    
    # Wait for retry cycle
    time.sleep(0.5)
    
    # Should have been re-put in queue
    assert put_count[0] >= 2  # Initial enqueue + re-queue
    
    queue.shutdown()
```

#### Test 7: Max Retries Exceeded (Item Dropped)

```python
def test_max_retries_drops_item(test_db_paths):
    """Test that items are dropped after max retries exceeded."""
    queue = PersistenceQueue(
        test_db_paths['predictions'],
        initial_retry_delay_ms=10,
        max_retries_before_notify=2
    )
    
    # Permanently failing write
    queue._write_item = lambda item: (_ for _ in ()).throw(Exception("Permanent failure"))
    
    # Enqueue
    item = PredictionItem(grid_id='g00000001', ...)
    queue.enqueue(item)
    
    # Wait for retries to exhaust
    time.sleep(1)
    
    # Should be dropped
    assert queue.metrics().total_failed == 1
    assert queue.metrics().total_written == 0
    
    queue.shutdown()
```

#### Test 8: Owner Notification After 3 Failures

```python
def test_owner_notified_after_max_retries(test_db_paths):
    """Test that owner is notified when prediction fails permanently."""
    notifications = []
    
    def notify_failure(item):
        notifications.append(item)
    
    queue = PersistenceQueue(
        test_db_paths['predictions'],
        initial_retry_delay_ms=10,
        max_retries_before_notify=3
    )
    queue._notify_owner_of_failure = notify_failure
    
    # Permanently failing write
    queue._write_item = lambda item: (_ for _ in ()).throw(Exception("Fail"))
    
    # Enqueue
    item = PredictionItem(grid_id='g00000001', ...)
    queue.enqueue(item)
    
    # Wait
    time.sleep(1)
    
    # Should have notified
    assert len(notifications) == 1
    assert notifications[0].grid_id == 'g00000001'
    
    queue.shutdown()
```

#### Test 9: Graceful Shutdown (Queue Drained)

```python
def test_graceful_shutdown_flushes_queue(test_db_paths):
    """Test that shutdown waits for queue to drain."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Enqueue 5 items
    for i in range(5):
        item = PredictionItem(grid_id=f'g{i:08d}', ...)
        queue.enqueue(item)
    
    # Shutdown with timeout
    success = queue.shutdown(timeout_sec=5)
    
    # Verify drained
    assert success == True
    assert queue.metrics().queue_size == 0
    assert queue.metrics().total_written == 5
```

#### Test 10: Shutdown Timeout (Items Remaining)

```python
def test_shutdown_timeout_partial_drain(test_db_paths):
    """Test that shutdown times out if queue not fully drained."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Slow write
    original_write = queue._write_item
    def slow_write(item):
        time.sleep(0.5)
        return original_write(item)
    queue._write_item = slow_write
    
    # Enqueue 10 items (will take 5+ seconds to write)
    for i in range(10):
        item = PredictionItem(grid_id=f'g{i:08d}', ...)
        queue.enqueue(item)
    
    # Shutdown with short timeout
    success = queue.shutdown(timeout_sec=1)
    
    # Should timeout
    assert success == False
    # Some items written, some remaining
    assert queue.metrics().total_written < 10
```

#### Test 11: Thread Restart on Death

```python
def test_worker_thread_restart_on_death(test_db_paths):
    """Test that worker thread is restarted if it dies."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Verify thread started
    assert queue.is_healthy() == True
    
    # Kill the thread (simulate crash)
    original_thread = queue._worker_thread
    original_thread._stop()  # Force stop
    time.sleep(0.5)
    
    # Verify thread is dead
    assert original_thread.is_alive() == False
    
    # Trigger restart (optional, depends on implementation)
    # ...
    
    # Should be restarted
    # assert queue.is_healthy() == True
    
    queue.shutdown()
```

#### Test 12: Database Connection Failures

```python
def test_database_connection_failure_handling(test_db_paths):
    """Test recovery from database connection failures."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Simulate DB connection failure
    original_write = queue._write_item
    failure_count = [0]
    
    def failing_then_succeeding_write(item):
        failure_count[0] += 1
        if failure_count[0] <= 2:
            raise Exception("DB connection lost")
        return original_write(item)
    
    queue._write_item = failing_then_succeeding_write
    
    # Enqueue
    item = PredictionItem(grid_id='g00000001', ...)
    queue.enqueue(item)
    
    # Wait for recovery
    time.sleep(1)
    
    # Should recover and write
    assert queue.metrics().total_written == 1
    
    queue.shutdown()
```

#### Test 13: Prediction Item Validation

```python
def test_prediction_item_fields_required(test_db_paths):
    """Test that all required fields are present in PredictionItem."""
    queue = PersistenceQueue(test_db_paths['predictions'])
    
    # Valid item
    valid_item = PredictionItem(
        grid_id='g00000001',
        timestamp_hour=datetime(2026, 1, 19, 14, 0, 0),
        pm25_prediction=45.0,
        model_version='v1',
        schema_hash='hash123'
    )
    queue.enqueue(valid_item)
    
    # Invalid items
    with pytest.raises(Exception):  # Missing grid_id
        PredictionItem(
            timestamp_hour=datetime(2026, 1, 19, 14, 0, 0),
            pm25_prediction=45.0,
            model_version='v1',
            schema_hash='hash123'
        )
    
    queue.shutdown()
```

### 5.2 Load Test Integration

**File:** Extend `takshashila/load_test.py`

**New test variant:** `run_load_tests_with_persistence_queue()`

```python
def run_load_tests_with_persistence_queue(db_paths: Dict[str, str]):
    """Run same load profiles WITH persistence queue enabled.
    
    Compare against baseline (test_load.py without queue):
    
    Profile 1: Baseline (10 concurrent, 30s)
    - Expected P50: <50ms (vs 26.6ms without queue, but still <100ms)
    - Expected P99: <200ms (vs 138ms without queue)
    - Expected success rate: 100% (vs 5% without queue) ✅
    
    Profile 2: Sustained (100 concurrent, 60s)
    - Expected throughput: 100+ req/s (vs 0 without queue)
    - Expected success rate: 100% (vs 0% without queue) ✅
    - Expected queue size: steady state (drains at write rate)
    
    Profile 3: Stress (1000 concurrent, 30s)
    - Expected throughput: 500+ req/s (vs 0 without queue)
    - Expected success rate: 100% (vs 0% without queue) ✅
    - Expected queue size: grows then drains
    
    Metrics to track:
    - Response latency (should be ~35ms, not blocked on write)
    - Write latency (sequential, no lock contention)
    - Queue size over time (histogram)
    - Error rate (should be 0%)
    - Worker thread health
    - Metrics report from queue
    """
```

**Test implementation notes:**
- Use `InferenceSimulator` but enable `use_persistence_queue=True`
- Run both variants (with/without queue) in same suite
- Compare side-by-side metrics
- Verify success rate improvement from 5% → 100%

---

## 6. Implementation Checklist

### Phase 1: Core Implementation

- [ ] Create `PersistenceQueue` class skeleton
- [ ] Implement queue data structure (unbounded, FIFO)
- [ ] Implement `enqueue()` method
- [ ] Implement `metrics()` method
- [ ] Implement `is_healthy()` method
- [ ] Implement `shutdown()` method
- [ ] Create `PredictionItem` dataclass
- [ ] Create `PersistenceQueueMetrics` dataclass
- [ ] Create exception classes

### Phase 2: Worker Thread

- [ ] Implement `_worker_loop()` main logic
- [ ] Implement `_write_item()` database writes
- [ ] Implement `_exponential_backoff()` calculation
- [ ] Implement `_notify_owner_of_failure()` callback
- [ ] Add thread start/stop lifecycle
- [ ] Add graceful shutdown logic

### Phase 3: Testing

- [ ] Create `test_persistence_queue.py` with all 13 unit tests
- [ ] Verify all tests pass
- [ ] Extend `load_test.py` with queue variant
- [ ] Run load tests with and without queue
- [ ] Compare metrics side-by-side
- [ ] Document results

### Phase 4: Integration

- [ ] Integrate with `InferenceEngine` (deferred to worker phase)
- [ ] Add configuration options (retry delays, timeouts)
- [ ] Add monitoring/metrics export
- [ ] Add documentation

---

## 7. Configuration & Constants

### Recommended Defaults

```python
PERSISTENCE_QUEUE_DEFAULTS = {
    'initial_retry_delay_ms': 100,      # Start backoff at 100ms
    'max_retry_delay_ms': 10_000,       # Cap at 10 seconds
    'max_retries_before_notify': 3,     # Notify after 3 failures
    'worker_get_timeout_sec': 1.0,      # Block on queue.get for 1s
    'shutdown_timeout_sec': 30.0,       # Wait 30s for graceful shutdown
}
```

### Retry Backoff Schedule

```
Attempt 1: 100ms
Attempt 2: 200ms
Attempt 3: 400ms
Attempt 4: 800ms
Attempt 5: 1,600ms
Attempt 6: 3,200ms
Attempt 7+: 10,000ms (capped)
```

---

## 8. Monitoring & Observability

### Logging Levels

**DEBUG:**
- Each enqueue: "Enqueued {grid_id} @ {timestamp}"
- Each write: "Wrote {grid_id} in {latency}ms"
- Queue drains on shutdown

**INFO:**
- Queue initialization: "PersistenceQueue created for {db_path}"
- Worker thread start/stop
- Shutdown initiated/complete

**WARNING:**
- Write retry: "Write failed for {grid_id}, retry {n}/3 in {backoff}ms"
- Slow writes: "Write latency {ms}ms exceeds threshold"

**ERROR:**
- Max retries exceeded: "DROPPED {grid_id} after 3 failures: {error}"
- Worker thread died: "Worker thread crashed: {error}"
- Notification sent to owner

### Metrics Export

```python
# Available via queue.metrics().report()
metrics_output = (
    "Queue size: 5 | "
    "Enqueued: 1000 | "
    "Written: 995 | "
    "Failed: 1 | "
    "Avg latency: 15.32ms | "
    "Worker restarts: 0"
)
```

### Health Checks

```python
# Simple boolean
queue.is_healthy()  # True if worker thread alive
```

---

## 9. Error Recovery Strategy

### Transient Failures (Network, Temporary Lock)
→ Retry with exponential backoff (attempt 1, 2, 3...)

### Permanent Failures (Data Validation Error, Disk Full)
→ After 3 retries, drop and notify owner

### Worker Thread Death
→ (Optional) Restart thread with automatic recovery

### Service Shutdown
→ Graceful: drain queue before exit (30s timeout)

---

## 10. Future Enhancements (Not In Scope)

- **Persistence to disk** (write queue to file on disk for crash recovery)
- **Batching** (if sequential writes become bottleneck, batch 10-100 items)
- **Distributed queue** (RabbitMQ/Redis if multi-service architecture needed)
- **Monitoring service** (external health check, alert on worker thread death)
- **Metrics storage** (Prometheus/Grafana integration)

---

## Summary

**PersistenceQueue is a simple, focused component:**
- Unbounded FIFO queue for async prediction writes
- Single background worker thread (sequential writes, no contention)
- Encapsulated failure handling (retry with exponential backoff)
- Graceful shutdown (flush queue before exit)
- Full observability (metrics, logging, health checks)
- 13 unit tests + load test validation

**Expected improvement:**
- Write success rate: 5% → 100%
- Response latency: ~35ms (decoupled from write latency)
- Handles 100+ concurrent users without errors

**Next phase:** Integration with InferenceEngine and running load tests to validate the fix.
