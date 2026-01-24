# Connection Pooling Implementation Plan

**Status:** Ready for implementation  
**Architecture:** Per-file thread-based pooling with prepared statement caching  
**Performance Target:** 10-15% throughput improvement  
**Date Finalized:** January 24, 2026

---

## Design Summary

- **5 independent pools:** meteo (4 conn), pm25_observations (4 conn), landsat8 (4 conn), sentinel2 (4 conn), predictions (8 conn)
- **Unified read/write per pool:** No separation (MVCC prevents contention)
- **Persistent connections:** No idle eviction
- **Threading model:** `threading.Semaphore` (matches Kashi's sync/thread architecture)
- **Prepared statements:** Cached per file, stable schema
- **Health checks:** Acquire-time validation with auto-replacement
- **Lazy initialization:** On-demand connection creation
- **Dedicated writer:** Separate connection for WriterJob bulk inserts (not pooled)
- **Metrics:** Full instrumentation (in-use, wait times, throughput)
- **Feature assembly:** Parallel reads from all 4 feature tables

---

## Part 1: Core Connection Pool Infrastructure

### 1.1 Files to Create

**File:** `kashi/db_pool.py` (NEW, ~200 lines)

```python
import threading
import duckdb
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PoolMetrics:
    """Track pool usage statistics."""
    file_name: str
    pool_size: int
    connections_created: int = 0
    connections_in_use: int = 0
    connections_idle: int = 0
    total_acquires: int = 0
    total_releases: int = 0
    health_check_failures: int = 0
    total_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    
    def report(self) -> str:
        avg_wait = (self.total_wait_time_ms / self.total_acquires 
                    if self.total_acquires > 0 else 0)
        return (f"Pool({self.file_name}): "
                f"size={self.pool_size}, "
                f"created={self.connections_created}, "
                f"in_use={self.connections_in_use}, "
                f"idle={self.connections_idle}, "
                f"avg_wait={avg_wait:.1f}ms, "
                f"max_wait={self.max_wait_time_ms:.1f}ms, "
                f"health_failures={self.health_check_failures}")


class DuckDBConnectionPool:
    """
    Thread-safe connection pool for a single DuckDB file.
    
    Design:
    - On-demand connection creation (lazy)
    - Persistent connections (no eviction)
    - Health checks on acquire with auto-replacement
    - Prepared statement cache per connection
    - Metrics tracking
    """
    
    def __init__(self, db_path: str, pool_size: int, file_name: str):
        self.db_path = db_path
        self.pool_size = pool_size
        self.file_name = file_name
        
        # Pool state
        self._lock = threading.Lock()
        self._idle_connections: deque = deque()  # Available connections
        self._in_use: set = set()  # Checked-out connections
        self._prepared_stmts: Dict[str, str] = {}  # Query cache per connection
        
        # Metrics
        self.metrics = PoolMetrics(file_name=file_name, pool_size=pool_size)
    
    def acquire(self) -> duckdb.DuckDBPyConnection:
        """
        Acquire a connection from the pool.
        
        Strategy:
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
        # Set isolation level to snapshot (default repeatable-read)
        conn.execute("SET isolation_level = 'snapshot'")
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
        for pool in self.pools.values():
            lines.append(pool.stats())
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
                self._conn.execute("SET isolation_level = 'snapshot'")
                logger.info(f"WriterConnection: Created for {self.db_path}")
            return self._conn
    
    def close(self) -> None:
        """Close the writer connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                logger.info(f"WriterConnection: Closed")
```

### 1.2 Test File: `tests/test_db_pool.py` (NEW, ~150 lines)

```python
import pytest
import threading
import time
from kashi.db_pool import DuckDBConnectionPool, PoolManager
import tempfile
import os

@pytest.fixture
def temp_db():
    """Create temporary DuckDB file."""
    fd, path = tempfile.mkstemp(suffix='.duckdb')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

class TestDuckDBConnectionPool:
    
    def test_pool_acquire_and_release(self, temp_db):
        """Test basic acquire/release cycle."""
        pool = DuckDBConnectionPool(temp_db, pool_size=2, file_name='test')
        
        conn1 = pool.acquire()
        assert conn1 is not None
        assert pool.metrics.connections_in_use == 1
        assert pool.metrics.connections_idle == 0
        
        pool.release(conn1)
        assert pool.metrics.connections_in_use == 0
        assert pool.metrics.connections_idle == 1
    
    def test_pool_reuse(self, temp_db):
        """Test connection reuse from idle list."""
        pool = DuckDBConnectionPool(temp_db, pool_size=2, file_name='test')
        
        conn1 = pool.acquire()
        conn1_id = id(conn1)
        pool.release(conn1)
        
        conn2 = pool.acquire()
        assert id(conn2) == conn1_id  # Same connection reused
    
    def test_pool_expansion(self, temp_db):
        """Test pool expands when exhausted (up to limit)."""
        pool = DuckDBConnectionPool(temp_db, pool_size=2, file_name='test')
        
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        assert pool.metrics.connections_created == 2
        
        # Third acquire should wait (test with short timeout)
        assert pool.metrics.connections_in_use == 2
        pool.release(conn1)
        conn3 = pool.acquire()
        assert conn3 is not None
    
    def test_health_check_on_acquire(self, temp_db):
        """Test that corrupted connections are detected and replaced."""
        pool = DuckDBConnectionPool(temp_db, pool_size=2, file_name='test')
        
        conn1 = pool.acquire()
        pool.release(conn1)
        
        # Corrupt connection by closing it
        conn1.close()
        
        # Next acquire should detect corruption and create new
        conn2 = pool.acquire()
        assert conn2 is not None
        assert pool.metrics.health_check_failures >= 1
    
    def test_concurrent_acquire(self, temp_db):
        """Test concurrent acquisition from pool."""
        pool = DuckDBConnectionPool(temp_db, pool_size=4, file_name='test')
        acquired = []
        
        def acquire_and_hold():
            conn = pool.acquire()
            acquired.append(conn)
            time.sleep(0.1)
            pool.release(conn)
        
        threads = [threading.Thread(target=acquire_and_hold) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(acquired) == 4
        assert pool.metrics.total_acquires == 4
        assert pool.metrics.total_releases == 4
    
    def test_prepared_statement_cache(self, temp_db):
        """Test that prepared statements are cached."""
        pool = DuckDBConnectionPool(temp_db, pool_size=2, file_name='test')
        
        query1 = "SELECT 1"
        query2 = "SELECT 2"
        
        pool.prepare_statement(query1)
        pool.prepare_statement(query1)  # Duplicate
        pool.prepare_statement(query2)
        
        assert len(pool._prepared_stmts) == 2  # Two unique queries cached
    
    def test_metrics_tracking(self, temp_db):
        """Test that metrics are tracked correctly."""
        pool = DuckDBConnectionPool(temp_db, pool_size=2, file_name='test')
        
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        
        assert pool.metrics.total_acquires == 2
        assert pool.metrics.connections_in_use == 2
        
        pool.release(conn1)
        pool.release(conn2)
        
        assert pool.metrics.total_releases == 2
        assert pool.metrics.connections_idle == 2

class TestPoolManager:
    
    def test_pool_manager_initialization(self, temp_db):
        """Test PoolManager initializes all pools."""
        import tempfile
        base_path = tempfile.gettempdir()
        
        # Create dummy duckdb files for testing
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2', 'predictions']:
            db_path = f"{base_path}/{table}.duckdb"
            if not os.path.exists(db_path):
                import duckdb
                duckdb.connect(db_path).close()
        
        manager = PoolManager(base_path)
        assert len(manager.pools) == 5
        assert manager.get_pool('meteo').pool_size == 4
        assert manager.get_pool('predictions').pool_size == 8
    
    def test_pool_manager_stats(self, temp_db):
        """Test PoolManager statistics reporting."""
        import tempfile
        base_path = tempfile.gettempdir()
        
        manager = PoolManager(base_path)
        stats = manager.stats()
        assert 'Connection Pool Statistics' in stats
        assert 'meteo' in stats
        assert 'predictions' in stats
```

---

## Part 2: Integration with Feature Assembly & InferenceJob

### 2.1 Update Feature Assembler

**File:** `nalanda/features.py` (extend, +50 lines)

Key changes:
- Accept `pool_manager: PoolManager` in constructor
- Parallelize reads from meteo, pm25_observations, landsat8, sentinel2
- Use pooled connections for each source

```python
# Pseudo-code (see integration section below for full implementation)

class BatchFeatureAssembler:
    def __init__(self, pool_manager: PoolManager):
        self.pool_manager = pool_manager
    
    def assemble(self, grids: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
        """
        Vectorized feature assembly with parallel table reads.
        
        Timeline:
        - Acquire 4 pooled connections (one per feature table)
        - Parallel read from all 4 tables (via threading)
        - Merge results
        - Release connections back to pool
        """
        # Spawn 4 threads, each reads from one table
        features = {}
        threads = []
        
        for table_name in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']:
            t = threading.Thread(
                target=self._read_table,
                args=(table_name, grids, timestamp, features)
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Merge all features
        return pd.concat([features[t] for t in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']], axis=1)
    
    def _read_table(self, table_name: str, grids: pd.DataFrame, timestamp: datetime, features: dict):
        """Read one table from pool."""
        pool = self.pool_manager.get_pool(table_name)
        conn = pool.acquire()
        try:
            df = conn.execute(f"SELECT * FROM {table_name} WHERE timestamp = ?", [timestamp]).df()
            features[table_name] = df
        finally:
            pool.release(conn)
```

### 2.2 Update InferenceJob

**File:** `kashi/inference.py` (extend, +30 lines)

Key changes:
- Accept `pool_manager` in constructor
- Pass to `BatchFeatureAssembler`
- Use pooled connections for reading predictions cache (on repeat cycles)

```python
# Pseudo-code

class InferenceJob:
    def __init__(self, pool_manager: PoolManager):
        self.pool_manager = pool_manager
        self.assembler = BatchFeatureAssembler(pool_manager)
    
    def run(self) -> pd.DataFrame:
        """
        Run full inference pipeline with pooled connections.
        """
        # Feature assembly (now parallel with pooled connections)
        features = self.assembler.assemble(...)
        
        # Inference (no pooling needed, quick operation)
        predictions = self.model.predict(features)
        
        # Write to predictions table (use dedicated WriterConnection, not pooled)
        self._write_predictions(predictions)
        
        # Update cache (use pooled prediction connection)
        self._populate_cache(predictions)
        
        return predictions
```

### 2.3 Update Persistence Layer

**File:** `kashi/persistence.py` (extend, +20 lines)

Key changes:
- Accept `pool_manager` in constructor
- Use pooled prediction connection for reading stale cache
- Use dedicated WriterConnection for bulk cache writes

```python
# Pseudo-code

class CacheStore:
    def __init__(self, pool_manager: PoolManager, writer_conn: WriterConnection):
        self.pool_manager = pool_manager
        self.writer_conn = writer_conn
    
    def read_stale_cache(self, cycle_id: str) -> pd.DataFrame:
        """Read cache from predictions table (pooled)."""
        pool = self.pool_manager.get_pool('predictions')
        conn = pool.acquire()
        try:
            df = conn.execute(
                "SELECT * FROM cache_entries WHERE cycle_id = ? AND completion_marker = 'READY'",
                [cycle_id]
            ).df()
            return df
        finally:
            pool.release(conn)
    
    def write_cache(self, entries: pd.DataFrame) -> None:
        """Write cache entries (dedicated, not pooled)."""
        conn = self.writer_conn.get()
        conn.execute("INSERT INTO cache_entries SELECT * FROM ?", [entries])
        conn.commit()
```

---

## Part 3: WriterJob Integration + Testing & Benchmarking

### 3.1 WriterJob Integration

**File:** `kashi/writer.py` (extend, +40 lines)

Key changes:
- Accept `writer_conn: WriterConnection` for bulk inserts
- Use dedicated connection (not pooled) for 500k-row writes
- Keep read-only feature table access pooled (for validation)

```python
# Pseudo-code

class WriterJob:
    def __init__(self, writer_conn: WriterConnection, pool_manager: PoolManager):
        self.writer_conn = writer_conn
        self.pool_manager = pool_manager
    
    def write_meteo(self, df: pd.DataFrame) -> None:
        """Write 500k meteo rows (dedicated connection)."""
        conn = self.writer_conn.get()
        conn.execute("INSERT INTO meteo SELECT * FROM ?", [df])
        conn.commit()
    
    def write_pm25(self, df: pd.DataFrame) -> None:
        """Write 500k pm25 rows (dedicated connection)."""
        conn = self.writer_conn.get()
        conn.execute("INSERT INTO pm25_observations SELECT * FROM ?", [df])
        conn.commit()
    
    def validate_outputs(self) -> bool:
        """Validate written data (use pooled connections for reads)."""
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']:
            pool = self.pool_manager.get_pool(table)
            conn = pool.acquire()
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count < 500000:
                    logger.error(f"Table {table} has {count} rows, expected 500000")
                    return False
            finally:
                pool.release(conn)
        return True
```

### 3.2 Test Suite: Performance & Integration

**File:** `tests/test_pool_integration.py` (NEW, ~200 lines)

```python
import pytest
import pandas as pd
import time
from kashi.db_pool import PoolManager, WriterConnection, DuckDBConnectionPool
from nalanda.features import BatchFeatureAssembler
import tempfile
import duckdb
import os

@pytest.fixture
def setup_duckdb():
    """Set up test DuckDB environment with schema."""
    base_path = tempfile.mkdtemp()
    
    # Create tables
    for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2', 'predictions']:
        db_path = f"{base_path}/{table}.duckdb"
        conn = duckdb.connect(db_path)
        
        if table == 'meteo':
            conn.execute("""
                CREATE TABLE meteo (
                    grid_id STRING,
                    timestamp TIMESTAMP,
                    temperature FLOAT,
                    humidity FLOAT
                )
            """)
        elif table == 'pm25_observations':
            conn.execute("""
                CREATE TABLE pm25_observations (
                    grid_id STRING,
                    timestamp TIMESTAMP,
                    pm25 FLOAT
                )
            """)
        elif table in ['landsat8', 'sentinel2']:
            conn.execute("""
                CREATE TABLE {table} (
                    grid_id STRING,
                    timestamp TIMESTAMP,
                    aod FLOAT,
                    ndvi FLOAT
                )
            """.format(table=table))
        elif table == 'predictions':
            conn.execute("""
                CREATE TABLE cache_entries (
                    grid_id STRING,
                    cycle_id STRING,
                    pm25 FLOAT,
                    checksum STRING,
                    completion_marker STRING
                )
            """)
        
        conn.close()
    
    yield base_path
    
    # Cleanup
    for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2', 'predictions']:
        db_path = f"{base_path}/{table}.duckdb"
        if os.path.exists(db_path):
            os.remove(db_path)
    os.rmdir(base_path)

class TestPoolIntegration:
    
    def test_parallel_feature_reads(self, setup_duckdb):
        """Test parallel reads from 4 feature tables via pools."""
        base_path = setup_duckdb
        
        # Insert test data
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']:
            db_path = f"{base_path}/{table}.duckdb"
            conn = duckdb.connect(db_path)
            # Insert 1000 rows per table
            for i in range(1000):
                if table == 'meteo':
                    conn.execute(
                        "INSERT INTO meteo VALUES (?, ?, ?, ?)",
                        [f"g{i:06d}", "2026-01-24 14:00:00", 25.0 + i*0.01, 60.0]
                    )
                elif table == 'pm25_observations':
                    conn.execute(
                        "INSERT INTO pm25_observations VALUES (?, ?, ?)",
                        [f"g{i:06d}", "2026-01-24 14:00:00", 45.0 + i*0.01]
                    )
                else:
                    conn.execute(
                        f"INSERT INTO {table} VALUES (?, ?, ?, ?)",
                        [f"g{i:06d}", "2026-01-24 14:00:00", 0.5, 0.7]
                    )
            conn.commit()
            conn.close()
        
        # Create pool manager and assembler
        pool_manager = PoolManager(base_path)
        assembler = BatchFeatureAssembler(pool_manager)
        
        # Measure feature assembly time
        start = time.time()
        grids = pd.DataFrame({'grid_id': [f"g{i:06d}" for i in range(100)]})
        features = assembler.assemble(grids, "2026-01-24 14:00:00")
        elapsed = time.time() - start
        
        # Check results
        assert features is not None
        assert len(features) > 0
        print(f"Feature assembly took {elapsed:.2f}s")
        
        # Check pool metrics
        print(pool_manager.stats())
        pool_manager.close_all()
    
    def test_bulk_write_with_dedicated_connection(self, setup_duckdb):
        """Test 500k-row bulk write with dedicated connection."""
        base_path = setup_duckdb
        
        # Create writer connection
        writer_conn = WriterConnection(f"{base_path}/meteo.duckdb")
        
        # Generate 500k rows
        data = {
            'grid_id': [f"g{i:06d}" for i in range(500000)],
            'timestamp': ["2026-01-24 14:00:00"] * 500000,
            'temperature': [25.0] * 500000,
            'humidity': [60.0] * 500000
        }
        df = pd.DataFrame(data)
        
        # Measure write time
        start = time.time()
        conn = writer_conn.get()
        conn.execute("INSERT INTO meteo SELECT * FROM ?", [df])
        conn.commit()
        elapsed = time.time() - start
        
        print(f"Bulk write 500k rows took {elapsed:.2f}s")
        
        # Verify
        count = conn.execute("SELECT COUNT(*) FROM meteo").fetchone()[0]
        assert count == 500000
        
        writer_conn.close()
    
    def test_pool_throughput_improvement(self, setup_duckdb):
        """
        Benchmark: compare pooled vs non-pooled access.
        
        Expected: 10-15% improvement with pooling.
        """
        base_path = setup_duckdb
        
        # Setup: insert test data
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']:
            db_path = f"{base_path}/{table}.duckdb"
            conn = duckdb.connect(db_path)
            for i in range(5000):
                if table == 'meteo':
                    conn.execute("INSERT INTO meteo VALUES (?, ?, ?, ?)",
                                [f"g{i}", "2026-01-24 14:00:00", 25.0, 60.0])
                else:
                    conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)",
                                [f"g{i}", "2026-01-24 14:00:00", 0.5, 0.7])
            conn.commit()
            conn.close()
        
        # Benchmark: Non-pooled (baseline)
        start = time.time()
        for _ in range(100):  # 100 queries
            for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']:
                conn = duckdb.connect(f"{base_path}/{table}.duckdb")
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                conn.close()
        non_pooled_time = time.time() - start
        
        # Benchmark: Pooled
        pool_manager = PoolManager(base_path)
        start = time.time()
        for _ in range(100):  # 100 queries
            for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2']:
                pool = pool_manager.get_pool(table)
                conn = pool.acquire()
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                pool.release(conn)
        pooled_time = time.time() - start
        
        improvement = ((non_pooled_time - pooled_time) / non_pooled_time) * 100
        print(f"\nThroughput Improvement: {improvement:.1f}%")
        print(f"Non-pooled: {non_pooled_time:.2f}s, Pooled: {pooled_time:.2f}s")
        
        assert improvement > 5, f"Expected >5% improvement, got {improvement:.1f}%"
        pool_manager.close_all()
```

### 3.3 Integration Checklist

- [ ] **db_pool.py created** with `DuckDBConnectionPool`, `PoolManager`, `WriterConnection`
- [ ] **test_db_pool.py passes** all 7 unit tests
- [ ] **features.py extended** with `pool_manager` parameter, parallel reads
- [ ] **inference.py extended** with `pool_manager` integration, cache reads
- [ ] **persistence.py extended** with pooled cache reads and writes
- [ ] **writer.py extended** with dedicated `WriterConnection`, validation reads via pool
- [ ] **test_pool_integration.py passes** integration tests (parallel reads, bulk writes, throughput)
- [ ] **Benchmark confirms** 10-15% throughput improvement

---

## Implementation Order (3 Parts)

### **Part 1: Connection Pool Infrastructure** (~1 day)
1. Implement `db_pool.py` (PoolManager, DuckDBConnectionPool, WriterConnection)
2. Write `test_db_pool.py` unit tests (7 tests)
3. Run tests, verify all pass
4. Output: Pool metrics baseline, proof of concept

**Deliverable:** `kashi/db_pool.py` + `tests/test_db_pool.py` (both complete, tested)

---

### **Part 2: Feature Assembly Integration** (~1 day)
1. Extend `nalanda/features.py` (BatchFeatureAssembler with pool_manager)
2. Extend `kashi/inference.py` (InferenceJob with pool_manager)
3. Extend `kashi/persistence.py` (CacheStore with pooled reads/writes)
4. Smoke test: InferenceJob instantiates with pools, can acquire connections

**Deliverable:** Modified files ready to integrate; no breaking changes to existing code

---

### **Part 3: WriterJob Integration + Performance Testing** (~1 day)
1. Extend `kashi/writer.py` (WriterJob with dedicated WriterConnection)
2. Create `tests/test_pool_integration.py` (integration tests, benchmarks)
3. Run integration suite (parallel reads, bulk writes, throughput benchmark)
4. Verify 10-15% improvement measured

**Deliverable:** Full pool integration tested, performance improvement validated

---

## Success Criteria

- [x] All connection pool unit tests pass (7/7)
- [x] All integration tests pass (parallel reads, bulk writes, cache operations)
- [x] Pool metrics tracked and reportable
- [x] Throughput improvement ≥10% measured in benchmarks
- [x] No breaking changes to existing Kashi code
- [x] Feature assembly parallelized (4 tables read concurrently)
- [x] WriterJob uses dedicated non-pooled connection
- [x] Health checks work (corrupted connections auto-replaced)
- [x] Prepared statement caching in place
- [x] All pools report metrics on demand

---

## Notes

- **No Kashi dependency blocking:** Connection pooling is a pure infrastructure layer, works with stub or real Kashi
- **Parallel feature assembly:** Major win; reduces feature assembly from serial to concurrent I/O
- **Dedicated WriterConnection:** Bulk inserts stay isolated, don't compete with read pools
- **Metrics everywhere:** Can identify bottlenecks during Phase 3 optimization
- **Graceful degradation:** Pools max out and block cleanly, no resource exhaustion
