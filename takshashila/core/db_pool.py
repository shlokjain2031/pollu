"""
Connection pooling for DuckDB files in Kashi.

Part 1 of 3-part implementation:
- 5 independent pools (meteo, pm25_observations, landsat8, sentinel2, predictions)
- Thread-safe acquisition/release using threading.Lock
- Health checks on acquire
- Metrics tracking
- Dedicated writer connection (non-pooled)
"""

import duckdb
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Track connection pool metrics."""
    
    connections_created: int = 0
    connections_in_use: int = 0
    connections_idle: int = 0
    total_acquires: int = 0
    total_releases: int = 0
    total_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    health_check_failures: int = 0
    
    def report(self) -> str:
        """Format metrics as human-readable string."""
        avg_wait = self.total_wait_time_ms / max(1, self.total_acquires)
        return (
            f"Created: {self.connections_created} | "
            f"In-use: {self.connections_in_use} | "
            f"Idle: {self.connections_idle} | "
            f"Acquires: {self.total_acquires} | "
            f"Releases: {self.total_releases} | "
            f"Avg wait: {avg_wait:.2f}ms | "
            f"Max wait: {self.max_wait_time_ms:.2f}ms | "
            f"Health failures: {self.health_check_failures}"
        )


class DuckDBConnectionPool:
    """
    Thread-safe connection pool for a single DuckDB file.
    
    Design:
    - Lazy initialization: connections created on-demand
    - Health checks: corrupted connections replaced automatically
    - Prepared statement caching: query text cached for reuse
    - Blocking wait: if pool exhausted, wait for release
    """
    
    def __init__(self, db_path: str, pool_size: int, file_name: str):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to DuckDB file
            pool_size: Maximum connections in pool
            file_name: Descriptive name for logging
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.file_name = file_name
        
        self._idle_connections: deque = deque()
        self._in_use: set = set()  # Track connection IDs
        self._lock = threading.Lock()
        self._prepared_stmts: Dict[int, str] = {}  # query_hash -> query_text
        self.metrics = PoolMetrics()
        
        logger.info(f"Pool({file_name}): Initialized with size={pool_size}")
    
    def acquire(self) -> duckdb.DuckDBPyConnection:
        """
        Acquire a connection from the pool.
        
        Algorithm:
        1. Try to get idle connection
        2. If none available and pool not full, create new connection
        3. If full, wait (Semaphore-style blocking)
        4. Health check acquired connection
        5. Replace if corrupted, retry
        """
        start_time = datetime.now()
        
        with self._lock:
            # Try idle connection first
            while self._idle_connections:
                conn = self._idle_connections.popleft()
                if self._is_healthy(conn):
                    self._in_use.add(id(conn))
                    self.metrics.connections_in_use = len(self._in_use)
                    self.metrics.connections_idle = len(self._idle_connections)
                    self._record_acquire(start_time)
                    return conn
                else:
                    # Connection corrupted, discard
                    self.metrics.health_check_failures += 1
                    logger.warning(f"Pool({self.file_name}): Discarded unhealthy connection")
            
            # No idle connection available
            if len(self._in_use) + len(self._idle_connections) < self.pool_size:
                # Create new connection
                conn = self._create_connection()
                self._in_use.add(id(conn))
                self.metrics.connections_created += 1
                self.metrics.connections_in_use = len(self._in_use)
                self._record_acquire(start_time)
                logger.debug(f"Pool({self.file_name}): Created new connection "
                            f"({self.metrics.connections_created}/{self.pool_size})")
                return conn
            
            # Pool exhausted, wait for a connection to be released
            logger.debug(f"Pool({self.file_name}): Pool exhausted, waiting for release...")
        
        # Block outside the lock to allow releases
        while True:
            threading.Event().wait(0.1)  # Sleep 100ms
            with self._lock:
                if self._idle_connections:
                    conn = self._idle_connections.popleft()
                    if self._is_healthy(conn):
                        self._in_use.add(id(conn))
                        self.metrics.connections_in_use = len(self._in_use)
                        self.metrics.connections_idle = len(self._idle_connections)
                        self._record_acquire(start_time)
                        return conn
                    else:
                        self.metrics.health_check_failures += 1
                elif len(self._in_use) + len(self._idle_connections) < self.pool_size:
                    conn = self._create_connection()
                    self._in_use.add(id(conn))
                    self.metrics.connections_created += 1
                    self.metrics.connections_in_use = len(self._in_use)
                    self._record_acquire(start_time)
                    return conn
    
    def release(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Return a connection to the pool."""
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                self._in_use.remove(conn_id)
                self._idle_connections.append(conn)
                self.metrics.total_releases += 1
                self.metrics.connections_in_use = len(self._in_use)
                self.metrics.connections_idle = len(self._idle_connections)
            else:
                logger.warning(f"Pool({self.file_name}): Attempted to release unknown connection")
    
    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection."""
        conn = duckdb.connect(self.db_path, read_only=False)
        return conn
    
    def _is_healthy(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """
        Validate connection is alive.
        
        Quick check: SELECT 1
        """
        try:
            conn.execute("SELECT 1").fetchall()
            return True
        except Exception as e:
            logger.warning(f"Pool({self.file_name}): Health check failed: {e}")
            return False
    
    def _record_acquire(self, start_time: datetime) -> None:
        """Record acquire timing metrics."""
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.total_acquires += 1
        self.metrics.total_wait_time_ms += elapsed_ms
        self.metrics.max_wait_time_ms = max(self.metrics.max_wait_time_ms, elapsed_ms)
    
    def prepare_statement(self, query: str) -> str:
        """
        Cache prepared statement metadata.
        
        Note: DuckDB doesn't have explicit prepared statements like PostgreSQL,
        but we can cache query text for repeated use and measure amortized parsing.
        """
        query_hash = hash(query)
        if query_hash not in self._prepared_stmts:
            self._prepared_stmts[query_hash] = query
        return self._prepared_stmts[query_hash]
    
    def stats(self) -> str:
        """Return human-readable pool statistics."""
        return self.metrics.report()
    
    def close_all(self) -> None:
        """Close all connections in the pool (for shutdown)."""
        with self._lock:
            for conn in self._idle_connections:
                conn.close()
            self._idle_connections.clear()
            self._in_use.clear()
            logger.info(f"Pool({self.file_name}): Closed all connections")


class PoolManager:
    """
    Manage all pools across Kashi.
    
    Pools:
    - meteo: 4 connections
    - pm25_observations: 4 connections
    - landsat8: 4 connections
    - sentinel2: 4 connections
    - predictions: 8 connections
    """
    
    def __init__(self, base_path: str):
        """Initialize all pools."""
        self.base_path = base_path
        self.pools: Dict[str, DuckDBConnectionPool] = {}
        
        # Feature table pools (4 connections each)
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']:
            db_path = f"{base_path}/{table}.duckdb"
            self.pools[table] = DuckDBConnectionPool(db_path, pool_size=4, file_name=table)
        
        # Predictions pool (8 connections)
        db_path = f"{base_path}/predictions.duckdb"
        self.pools['predictions'] = DuckDBConnectionPool(db_path, pool_size=8, file_name='predictions')
        
        logger.info(f"PoolManager initialized with {len(self.pools)} pools")
    
    def get_pool(self, table_name: str) -> DuckDBConnectionPool:
        """Get the pool for a specific table."""
        if table_name not in self.pools:
            raise ValueError(f"No pool for table {table_name}. Available: {list(self.pools.keys())}")
        return self.pools[table_name]
    
    def stats(self) -> str:
        """Return all pool statistics."""
        lines = ["=== Connection Pool Statistics ==="]
        for name, pool in self.pools.items():
            lines.append(f"{name}: {pool.stats()}")
        return "\n".join(lines)
    
    def close_all(self) -> None:
        """Close all pools (for shutdown)."""
        for pool in self.pools.values():
            pool.close_all()


class WriterConnection:
    """
    Dedicated connection for WriterJob bulk inserts.
    
    NOT pooled: long transactions, isolation matters, no benefit to pooling.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = threading.Lock()
    
    def get(self) -> duckdb.DuckDBPyConnection:
        """Get or create the dedicated writer connection."""
        with self._lock:
            if self._conn is None:
                self._conn = duckdb.connect(self.db_path, read_only=False)
                logger.info(f"WriterConnection: Created for {self.db_path}")
            return self._conn
    
    def close(self) -> None:
        """Close the writer connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                logger.info(f"WriterConnection: Closed")
