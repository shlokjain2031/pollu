"""DuckDB persistence layer with single-writer enforcement.

Implements read/write operations with validation per SYSTEM_CONTRACTS.md §3 and §5.

TIMESTAMP POLICY:
    All timestamps passed to persistence methods must be timezone-naive (conceptually UTC).
    This matches DuckDB's TIMESTAMP column storage behavior.
"""
import duckdb
import threading
import time
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from .errors import PersistenceError
from .config import WRITE_LOCK_TIMEOUT_SEC

logger = logging.getLogger(__name__)

# Import pool manager if available (optional for backward compatibility)
try:
    from takshashila.core.db_pool import PoolManager
    POOL_AVAILABLE = True
except ImportError:
    PoolManager = None
    POOL_AVAILABLE = False
    logger.warning("Connection pooling not available (kashi.db_pool not found)")

# Retry configuration for read operations during concurrent writes
MAX_READ_RETRIES = 5
READ_RETRY_BASE_DELAY_MS = 10  # Start with 10ms


class PersistenceLayer:
    """DuckDB persistence with single-writer semantics.
    
    Enforces:
    - Single writer at a time (application-level lock)
    - Write validation via read-back
    - Explicit error handling (no silent failures)
    
    Thread-safe for concurrent readers, serialized writes.
    """
    
    def __init__(self, db_path: str, pool_manager=None):
        """Initialize persistence layer.
        
        Args:
            db_path: Path to DuckDB file
            pool_manager: Optional PoolManager for connection pooling (from kashi.db_pool)
        """
        self.db_path = Path(db_path)
        self._write_lock = threading.Lock()
        self.pool_manager = pool_manager
        
        if not self.db_path.exists():
            logger.warning(f"DuckDB file does not exist: {db_path}. "
                         "Run init_duckdb.py to initialize schema.")
        
        pool_status = "with connection pooling" if pool_manager else "without connection pooling"
        logger.info(f"PersistenceLayer initialized {pool_status} for {db_path}")
    
    def _get_connection(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        """Get a DuckDB connection.
        
        If pool_manager is available and read_only=True, acquires from pool.
        Write operations always use new connections (not pooled).
        
        Args:
            read_only: If True, open in read-only mode for concurrent reads
        
        Returns:
            DuckDB connection (pooled if available and read_only, otherwise new)
        """
        # Use pooled connection for reads if available
        if read_only and self.pool_manager is not None:
            try:
                pool = self.pool_manager.get_pool('predictions')
                return pool.acquire()
            except Exception as e:
                logger.warning(f"Failed to acquire pooled connection for predictions, falling back to direct connection: {e}")
        
        # Fallback or write: create new connection
        conn = duckdb.connect(str(self.db_path), read_only=read_only)
        
        # Enable WAL mode for better concurrent access (write-ahead logging)
        # This allows multiple readers while safely serializing writes
        try:
            conn.execute("PRAGMA enable_object_cache=true")
        except:
            pass  # WAL may not be supported in all versions
        
        return conn
    
    def _release_connection(self, conn: duckdb.DuckDBPyConnection, read_only: bool = False) -> None:
        """Release a connection back to the pool or close it.
        
        Args:
            conn: Connection to release
            read_only: Whether this was a read-only connection
        """
        if read_only and self.pool_manager is not None:
            try:
                pool = self.pool_manager.get_pool('predictions')
                pool.release(conn)
            except Exception as e:
                logger.warning(f"Failed to release pooled connection for predictions: {e}")
                conn.close()
        else:
            # Write connections or legacy mode: just close
            conn.close()
    
    def _execute_with_retry(self, query: str, params: list = None, read_only: bool = False) -> Optional[tuple]:
        """Execute a query with retry logic for read operations during concurrent writes.
        
        When multiple workers try to read while another is writing, DuckDB may return
        lock errors. This method retries with exponential backoff.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            read_only: If True, use read-only connection and retry on lock errors
        
        Returns:
            Query result (first row) or None if no result
        
        Raises:
            PersistenceError: On persistent failures
        """
        if params is None:
            params = []
        
        if not read_only:
            # Write operations don't retry, they should fail fast
            conn = self._get_connection(read_only=False)
            try:
                return conn.execute(query, params).fetchone()
            finally:
                self._release_connection(conn, read_only=False)
        
        # Read operations with retry on lock conflicts
        last_error = None
        for attempt in range(MAX_READ_RETRIES):
            try:
                conn = self._get_connection(read_only=True)
                try:
                    return conn.execute(query, params).fetchone()
                finally:
                    self._release_connection(conn, read_only=True)
            except Exception as e:
                error_msg = str(e)
                # Check if this is a lock error
                if "lock" in error_msg.lower() or "Conflicting" in error_msg:
                    last_error = e
                    if attempt < MAX_READ_RETRIES - 1:
                        # Exponential backoff with jitter
                        delay_ms = READ_RETRY_BASE_DELAY_MS * (2 ** attempt)
                        delay_s = delay_ms / 1000.0
                        logger.debug(f"Lock conflict on read (attempt {attempt+1}/{MAX_READ_RETRIES}), "
                                   f"retrying in {delay_ms}ms...")
                        time.sleep(delay_s)
                        continue
                # Non-lock errors should fail immediately
                raise e
        
        # All retries exhausted
        raise PersistenceError(
            operation="read_with_retry",
            grid_id="unknown",
            timestamp_hour=None,
            reason=f"Failed after {MAX_READ_RETRIES} retries: {last_error}"
        )
    
    def read_prediction(self, grid_id: str, timestamp_hour: datetime) -> Optional[float]:
        """Read existing prediction from database.
        
        Args:
            grid_id: Grid cell identifier (e.g., "g00000001")
            timestamp_hour: Prediction timestamp (hour-floored)
        
        Returns:
            PM2.5 prediction value, or None if not found
        
        Raises:
            PersistenceError: On database read failure
        """
        try:
            result = self._execute_with_retry("""
                SELECT pm25_prediction 
                FROM predictions 
                WHERE grid_id = ? AND timestamp_hour = ?
            """, [grid_id, timestamp_hour], read_only=True)
            
            if result is None:
                return None
            
            return float(result[0])
        
        except Exception as e:
            raise PersistenceError(
                operation="read",
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                reason=str(e)
            )
    
    def write_prediction(
        self,
        grid_id: str,
        timestamp_hour: datetime,
        pm25_value: float,
        model_version: str,
        schema_hash: str
    ) -> None:
        """Write prediction to database with validation.
        
        Enforces single-writer semantics and validates write via read-back.
        
        Args:
            grid_id: Grid cell identifier
            timestamp_hour: Prediction timestamp (hour-floored)
            pm25_value: PM2.5 prediction value
            model_version: Model version used (e.g., "prod_v1")
            schema_hash: Feature schema hash
        
        Raises:
            PersistenceError: On write failure or validation failure
        """
        # Acquire write lock
        acquired = self._write_lock.acquire(timeout=WRITE_LOCK_TIMEOUT_SEC)
        if not acquired:
            raise PersistenceError(
                operation="write",
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                reason=f"Could not acquire write lock within {WRITE_LOCK_TIMEOUT_SEC}s"
            )
        
        try:
            conn = self._get_connection()
            
            try:
                # Insert prediction
                conn.execute("""
                    INSERT INTO predictions 
                        (grid_id, timestamp_hour, pm25_prediction, model_version, schema_hash)
                    VALUES (?, ?, ?, ?, ?)
                """, [grid_id, timestamp_hour, pm25_value, model_version, schema_hash])
                
                conn.commit()
                
                # Validate write via read-back
                if not self._validate_write(conn, grid_id, timestamp_hour, pm25_value):
                    raise PersistenceError(
                        operation="write_validation",
                        grid_id=grid_id,
                        timestamp_hour=timestamp_hour,
                        reason="Read-back validation failed (value mismatch)"
                    )
                
                logger.info(
                    f"Wrote prediction: grid_id={grid_id}, ts={timestamp_hour}, "
                    f"pm25={pm25_value:.2f}"
                )
            
            except PersistenceError:
                # Re-raise our own errors
                raise
            
            except Exception as e:
                raise PersistenceError(
                    operation="write",
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    reason=str(e)
                )
            
            finally:
                self._release_connection(conn, read_only=False)
        
        finally:
            self._write_lock.release()
    
    def _validate_write(
        self,
        conn: duckdb.DuckDBPyConnection,
        grid_id: str,
        timestamp_hour: datetime,
        expected_value: float
    ) -> bool:
        """Validate write by reading back the value.
        
        Args:
            conn: Active DuckDB connection
            grid_id: Grid cell identifier
            timestamp_hour: Prediction timestamp
            expected_value: Expected PM2.5 value
        
        Returns:
            True if read-back matches expected value
        """
        result = conn.execute("""
            SELECT pm25_prediction 
            FROM predictions 
            WHERE grid_id = ? AND timestamp_hour = ?
        """, [grid_id, timestamp_hour]).fetchone()
        
        if result is None:
            return False
        
        actual_value = float(result[0])
        
        # Use small epsilon for float comparison
        return abs(actual_value - expected_value) < 1e-6
    
    def grid_exists(self, grid_id: str) -> bool:
        """Check if grid_id exists in grid table.
        
        Args:
            grid_id: Grid cell identifier
        
        Returns:
            True if grid exists, False otherwise
        
        Raises:
            PersistenceError: On database read failure
        """
        try:
            result = self._execute_with_retry("""
                SELECT COUNT(*) 
                FROM grid 
                WHERE grid_id = ?
            """, [grid_id], read_only=True)
            
            return result[0] > 0
        
        except Exception as e:
            raise PersistenceError(
                operation="grid_lookup",
                grid_id=grid_id,
                timestamp_hour=None,
                reason=str(e)
            )


# ============================================================================
# PERSISTENCE QUEUE - ASYNC SEQUENTIAL DUCKDB WRITER
# ============================================================================

class PersistenceQueue:
    """Minimal thread-safe FIFO queue for sequential DuckDB writes.
    
    Purpose:
    - Accept predictions from multiple worker threads (non-blocking)
    - Batch predictions for efficient DuckDB inserts
    - Write sequentially to predictions table
    - Background writer thread handles all I/O
    
    Flow:
    ```
    Worker 1: queue.put(grid_id=g001, pm25=45.2)  [returns immediately]
    Worker 2: queue.put(grid_id=g002, pm25=48.1)  [returns immediately]
    Worker 3: queue.put(grid_id=g003, pm25=42.5)  [returns immediately]
    
    Background writer thread:
      Every batch_size items or batch_timeout_seconds:
        UPSERT INTO predictions VALUES (g001, ...), (g002, ...), ...
    ```
    """
    
    def __init__(self, db_path: str, batch_size: int = 100, batch_timeout_seconds: float = 1.0):
        """Initialize queue.
        
        Args:
            db_path: Path to predictions.duckdb
            batch_size: Write N predictions per batch (efficient for DuckDB)
            batch_timeout_seconds: Flush batch after timeout even if not full
        """
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        
        # Thread-safe FIFO queue (maxsize prevents unbounded memory growth)
        self.queue = queue.Queue(maxsize=10000)
        
        # Control flag for graceful shutdown
        self.running = True
        
        # Persistent write connection (shared across batches, created on first use)
        self._write_conn = None
        self._write_conn_lock = threading.RLock()
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        logger.info(
            f"PersistenceQueue initialized (batch_size={batch_size}, "
            f"timeout={batch_timeout_seconds}s, db={db_path})"
        )
    
    def put(
        self,
        grid_id: str,
        timestamp_hour: datetime,
        pm25_value: float,
        model_version: str,
        schema_hash: str
    ) -> None:
        """Add prediction to queue (non-blocking).
        
        Returns immediately. Write happens in background.
        
        Args:
            grid_id: Grid cell identifier
            timestamp_hour: Prediction timestamp
            pm25_value: PM2.5 prediction
            model_version: Model version
            schema_hash: Feature schema hash
        
        Raises:
            queue.Full: If queue is full (backpressure - slows workers)
        """
        item = {
            'grid_id': grid_id,
            'timestamp_hour': timestamp_hour,
            'pm25_value': pm25_value,
            'model_version': model_version,
            'schema_hash': schema_hash
        }
        
        try:
            self.queue.put(item, block=True)  # Block if queue full (backpressure)
        except queue.Full:
            logger.error(f"PersistenceQueue full, dropping prediction for {grid_id}")
    
    def _writer_loop(self) -> None:
        """Background thread: read from queue, batch, write to DuckDB.
        
        Collects predictions into batches and writes to DB every batch_size items
        or after batch_timeout_seconds, whichever comes first.
        """
        while self.running:
            batch = []
            deadline = time.time() + self.batch_timeout_seconds
            
            # Collect batch_size items or until timeout
            while len(batch) < self.batch_size:
                timeout = deadline - time.time()
                
                if timeout <= 0:
                    break
                
                try:
                    # Get next item with timeout
                    item = self.queue.get(timeout=timeout)
                    batch.append(item)
                except queue.Empty:
                    break
            
            # Write batch if not empty
            if batch:
                self._write_batch(batch)
    
    def _get_write_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create persistent write connection with WAL enabled.
        
        Creates connection once and reuses it for all batch writes to avoid
        connection pool exhaustion and "different configuration" errors.
        
        Returns:
            DuckDB connection in write mode with WAL enabled
        """
        self._write_conn_lock.acquire()
        try:
            if self._write_conn is None:
                self._write_conn = duckdb.connect(str(self.db_path), read_only=False)
                logger.debug(f"DuckDB write connection created for {self.db_path}")
            
            return self._write_conn
        finally:
            self._write_conn_lock.release()
    
    def _write_batch(self, batch: list) -> None:
        """Write batch of predictions to DuckDB in single transaction using UPSERT.
        
        Uses persistent connection and WAL mode to avoid connection conflicts.
        Idempotent: if prediction already exists, updates it.
        
        Args:
            batch: List of prediction dicts
        """
        try:
            conn = self._get_write_connection()
            
            # UPSERT: insert or replace if exists (idempotent)
            # DuckDB: INSERT OR REPLACE requires unique constraint on (grid_id, timestamp_hour)
            for item in batch:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions 
                        (grid_id, timestamp_hour, pm25_prediction, model_version, schema_hash)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    item['grid_id'],
                    item['timestamp_hour'],
                    item['pm25_value'],
                    item['model_version'],
                    item['schema_hash']
                ])
            
            conn.commit()
            logger.info(f"PersistenceQueue wrote {len(batch)} predictions to DuckDB")
        
        except Exception as e:
            logger.error(f"PersistenceQueue write failed for batch of {len(batch)}: {e}")
    
    def _close_write_connection(self) -> None:
        """Close persistent write connection."""
        self._write_conn_lock.acquire()
        try:
            if self._write_conn is not None:
                try:
                    self._write_conn.close()
                except:
                    pass
                self._write_conn = None
        finally:
            self._write_conn_lock.release()
    
    def shutdown(self, timeout_seconds: int = 5) -> None:
        """Gracefully shutdown queue and flush remaining items.
        
        Args:
            timeout_seconds: Max time to wait for writer thread
        """
        logger.info("PersistenceQueue shutting down...")
        
        self.running = False
        self.writer_thread.join(timeout=timeout_seconds)
        
        # Flush any remaining items
        if not self.queue.empty():
            logger.info(f"Flushing {self.queue.qsize()} remaining items...")
            batch = []
            while True:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            
            if batch:
                self._write_batch(batch)
        
        # Close persistent connection
        self._close_write_connection()
        
        logger.info("PersistenceQueue shutdown complete")
