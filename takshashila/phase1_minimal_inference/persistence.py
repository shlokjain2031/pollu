"""DuckDB persistence layer with single-writer enforcement.

Implements read/write operations with validation per SYSTEM_CONTRACTS.md §3 and §5.
"""
import duckdb
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from .errors import PersistenceError
from .config import WRITE_LOCK_TIMEOUT_SEC

logger = logging.getLogger(__name__)


class PersistenceLayer:
    """DuckDB persistence with single-writer semantics.
    
    Enforces:
    - Single writer at a time (application-level lock)
    - Write validation via read-back
    - Explicit error handling (no silent failures)
    
    Thread-safe for concurrent readers, serialized writes.
    """
    
    def __init__(self, db_path: str):
        """Initialize persistence layer.
        
        Args:
            db_path: Path to DuckDB file
        """
        self.db_path = Path(db_path)
        self._write_lock = threading.Lock()
        
        if not self.db_path.exists():
            logger.warning(f"DuckDB file does not exist: {db_path}. "
                         "Run init_duckdb.py to initialize schema.")
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a new DuckDB connection.
        
        DuckDB supports multiple reader connections but only one writer.
        We enforce single-writer at application level via _write_lock.
        """
        return duckdb.connect(str(self.db_path))
    
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
            conn = self._get_connection()
            
            try:
                result = conn.execute("""
                    SELECT pm25_prediction 
                    FROM predictions 
                    WHERE grid_id = ? AND timestamp_hour = ?
                """, [grid_id, timestamp_hour]).fetchone()
                
                if result is None:
                    return None
                
                return float(result[0])
            
            finally:
                conn.close()
        
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
                conn.close()
        
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
        """Check if grid_id exists in grids table.
        
        Args:
            grid_id: Grid cell identifier
        
        Returns:
            True if grid exists, False otherwise
        
        Raises:
            PersistenceError: On database read failure
        """
        try:
            conn = self._get_connection()
            
            try:
                result = conn.execute("""
                    SELECT COUNT(*) 
                    FROM grids 
                    WHERE grid_id = ?
                """, [grid_id]).fetchone()
                
                return result[0] > 0
            
            finally:
                conn.close()
        
        except Exception as e:
            raise PersistenceError(
                operation="grid_lookup",
                grid_id=grid_id,
                timestamp_hour=None,
                reason=str(e)
            )
