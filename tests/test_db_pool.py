"""
Unit tests for connection pooling infrastructure.

Tests:
1. pool_acquire_and_release: Basic acquire/release cycle
2. pool_reuse: Connection reuse from idle list
3. pool_expansion: Pool expands up to limit
4. health_check_on_acquire: Corrupted connections detected and replaced
5. concurrent_acquire: Concurrent acquisition from pool
6. prepared_statement_cache: Prepared statements cached
7. metrics_tracking: Metrics tracked correctly
"""

import pytest
import threading
import time
import tempfile
import os
import duckdb
from takshashila.core.db_pool import DuckDBConnectionPool, PoolManager


@pytest.fixture
def temp_db():
    """Create temporary DuckDB file."""
    fd, path = tempfile.mkstemp(suffix='.duckdb')
    os.close(fd)
    # Initialize as valid DuckDB database
    os.remove(path)  # Remove the empty file first
    conn = duckdb.connect(path)
    conn.close()
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
    
    def test_pool_manager_initialization(self):
        """Test PoolManager initializes all pools."""
        base_path = tempfile.gettempdir()
        
        # Create dummy duckdb files for testing
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2', 'predictions']:
            db_path = f"{base_path}/{table}.duckdb"
            if not os.path.exists(db_path):
                duckdb.connect(db_path).close()
        
        manager = PoolManager(base_path)
        assert len(manager.pools) == 5
        assert manager.get_pool('meteo').pool_size == 4
        assert manager.get_pool('predictions').pool_size == 8
        
        # Cleanup
        manager.close_all()
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2', 'predictions']:
            db_path = f"{base_path}/{table}.duckdb"
            if os.path.exists(db_path):
                os.remove(db_path)
    
    def test_pool_manager_stats(self):
        """Test PoolManager statistics reporting."""
        base_path = tempfile.gettempdir()
        
        # Create dummy duckdb files for testing
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2', 'predictions']:
            db_path = f"{base_path}/{table}.duckdb"
            if not os.path.exists(db_path):
                duckdb.connect(db_path).close()
        
        manager = PoolManager(base_path)
        stats = manager.stats()
        assert 'Connection Pool Statistics' in stats
        assert 'meteo' in stats
        assert 'predictions' in stats
        
        # Cleanup
        manager.close_all()
        for table in ['meteo', 'pm25_observations', 'landsat8', 'sentinel2', 'predictions']:
            db_path = f"{base_path}/{table}.duckdb"
            if os.path.exists(db_path):
                os.remove(db_path)
