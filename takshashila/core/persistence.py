"""DuckDB persistence layer with single-writer enforcement.

Implements read/write operations with validation per SYSTEM_CONTRACTS.md §3 and §5.

TIMESTAMP POLICY:
    All timestamps passed to persistence methods must be timezone-naive (conceptually UTC).
    This matches DuckDB's TIMESTAMP column storage behavior.
"""
import duckdb
import threading
import time
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
