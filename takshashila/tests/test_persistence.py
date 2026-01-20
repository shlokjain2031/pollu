"""Tests for persistence layer."""
import pytest
from datetime import datetime, timezone
import threading
import time

from Takshashila.phase1_minimal_inference.persistence import PersistenceLayer
from Takshashila.phase1_minimal_inference.errors import PersistenceError
from Takshashila.phase1_minimal_inference.config import MODEL_VERSION, SCHEMA_HASH


class TestPersistenceLayer:
    """Test DuckDB persistence operations."""
    
    def test_read_nonexistent_prediction(self, test_db_path, valid_grid_id, test_timestamp):
        """Test reading prediction that doesn't exist returns None."""
        persistence = PersistenceLayer(test_db_path)
        
        # Use a timestamp that has no prediction
        future_ts = datetime(2099, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = persistence.read_prediction(valid_grid_id, future_ts)
        
        assert result is None
    
    def test_write_and_read_prediction(self, persistence):
        """Test write/read cycle with validation."""
        # Use unique grid_id to avoid conflicts with other tests
        grid_id = "g00001000"
        timestamp = datetime(2026, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
        pm25_value = 65.5
        
        # Write prediction
        persistence.write_prediction(
            grid_id=grid_id,
            timestamp_hour=timestamp,
            pm25_value=pm25_value,
            model_version=MODEL_VERSION,
            schema_hash=SCHEMA_HASH
        )
        
        # Read back
        result = persistence.read_prediction(grid_id, timestamp)
        
        assert result is not None
        assert abs(result - pm25_value) < 1e-6
    
    def test_write_duplicate_raises_error(self, test_db_path, valid_grid_id):
        """Test writing duplicate (grid_id, timestamp) raises error."""
        persistence = PersistenceLayer(test_db_path)
        
        ts = datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc)
        pm25_value = 70.0
        
        # First write succeeds
        persistence.write_prediction(
            grid_id=valid_grid_id,
            timestamp_hour=ts,
            pm25_value=pm25_value,
            model_version=MODEL_VERSION,
            schema_hash=SCHEMA_HASH
        )
        
        # Second write should fail (PRIMARY KEY violation)
        with pytest.raises(PersistenceError) as exc_info:
            persistence.write_prediction(
                grid_id=valid_grid_id,
                timestamp_hour=ts,
                pm25_value=pm25_value + 10.0,
                model_version=MODEL_VERSION,
                schema_hash=SCHEMA_HASH
            )
        
        assert exc_info.value.operation == "write"
    
    def test_grid_exists(self, test_db_path, valid_grid_id, invalid_grid_id):
        """Test grid_exists checks grids table."""
        persistence = PersistenceLayer(test_db_path)
        
        assert persistence.grid_exists(valid_grid_id) is True
        assert persistence.grid_exists(invalid_grid_id) is False
    
    def test_write_lock_serialization(self, test_db_path, valid_grid_id):
        """Test single-writer lock enforces serialization."""
        persistence = PersistenceLayer(test_db_path)
        
        write_order = []
        
        def writer(writer_id, delay):
            """Simulate slow write."""
            time.sleep(delay)
            ts = datetime(2026, 1, 21, writer_id, 0, 0, tzinfo=timezone.utc)
            persistence.write_prediction(
                grid_id=valid_grid_id,
                timestamp_hour=ts,
                pm25_value=float(writer_id),
                model_version=MODEL_VERSION,
                schema_hash=SCHEMA_HASH
            )
            write_order.append(writer_id)
        
        # Launch two writers concurrently
        t1 = threading.Thread(target=writer, args=(10, 0.1))
        t2 = threading.Thread(target=writer, args=(11, 0.0))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Both should complete successfully
        assert len(write_order) == 2
        assert 10 in write_order
        assert 11 in write_order
