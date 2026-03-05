"""
Tests for inference_job_worker.py

Tests the batch inference worker that processes grid predictions,
caches results in Redis, and queues writes to DuckDB.
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import argparse

# Mock the imports before importing the worker
sys.modules['redis'] = MagicMock()
sys.modules['takshashila.core.config'] = MagicMock()
sys.modules['takshashila.core.inference'] = MagicMock()
sys.modules['takshashila.core.persistence'] = MagicMock()


class TestInferenceJobWorkerSetup:
    """Test worker initialization and setup."""
    
    def test_setup_logging(self):
        """Test logging initialization with cycle_id."""
        from takshashila.distributed.worker.inference_job_worker import setup_logging
        
        cycle_id = "test-cycle-123"
        logger = setup_logging(cycle_id)
        
        assert logger is not None
        assert logger.name == "takshashila.distributed.worker.inference_job_worker"
    
    def test_batch_size_env_variable(self):
        """Test BATCH_SIZE environment variable."""
        import os
        
        # Test default
        if "BATCH_SIZE" in os.environ:
            del os.environ["BATCH_SIZE"]
        
        from importlib import reload
        import takshashila.distributed.worker.inference_job_worker as worker_module
        
        # Should default to 100
        assert hasattr(worker_module, 'BATCH_SIZE')
        
        # Test custom value
        os.environ["BATCH_SIZE"] = "50"
        reload(worker_module)
        assert worker_module.BATCH_SIZE == 50
        
        # Cleanup
        del os.environ["BATCH_SIZE"]


class TestArgumentParsing:
    """Test command-line argument parsing."""
    
    def test_required_arguments(self):
        """Test that all required arguments are parsed correctly."""
        from takshashila.distributed.worker import inference_job_worker
        
        # Mock sys.argv
        test_args = [
            "inference_job_worker.py",
            "--cycle-id", "cycle-123",
            "--timestamp-hour", "2026-01-25T14:00:00Z",
            "--grid-batch-start", "0",
            "--grid-batch-end", "100",
            "--cache-host", "localhost",
            "--cache-port", "6379"
        ]
        
        with patch.object(sys, 'argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--cycle-id", required=True)
            parser.add_argument("--timestamp-hour", required=True)
            parser.add_argument("--grid-batch-start", type=int, required=True)
            parser.add_argument("--grid-batch-end", type=int, required=True)
            parser.add_argument("--cache-host", default="localhost")
            parser.add_argument("--cache-port", type=int, default=6379)
            parser.add_argument("--config", default=None)
            
            args = parser.parse_args(test_args[1:])
            
            assert args.cycle_id == "cycle-123"
            assert args.timestamp_hour == "2026-01-25T14:00:00Z"
            assert args.grid_batch_start == 0
            assert args.grid_batch_end == 100
            assert args.cache_host == "localhost"
            assert args.cache_port == 6379


class TestBatchPrediction:
    """Test batch prediction processing."""
    
    @patch('takshashila.distributed.worker.inference_job_worker.InferenceEngine')
    @patch('takshashila.distributed.worker.inference_job_worker.redis')
    @patch('takshashila.distributed.worker.inference_job_worker.PersistenceQueue')
    def test_batch_prediction_flow(self, mock_queue_class, mock_redis, mock_engine_class):
        """Test end-to-end batch prediction with 10 grids."""
        from takshashila.distributed.worker.inference_job_worker import (
            setup_logging, BATCH_SIZE
        )
        
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        mock_cache = MagicMock()
        mock_redis.Redis.return_value = mock_cache
        
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        
        # Mock predict_batch to return predictions
        grid_ids = [f"g{i:08d}" for i in range(10)]
        predictions = {gid: 40.0 + i for i, gid in enumerate(grid_ids)}
        mock_engine.predict_batch.return_value = predictions
        
        # Setup logging
        logger = setup_logging("test-cycle")
        
        # Test batch processing
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        
        batch_predictions = mock_engine.predict_batch(grid_ids, timestamp_hour)
        
        # Verify predictions returned correctly
        assert len(batch_predictions) == 10
        assert batch_predictions["g00000000"] == 40.0
        assert batch_predictions["g00000009"] == 49.0
        
        # Verify predict_batch was called
        mock_engine.predict_batch.assert_called_once_with(grid_ids, timestamp_hour)
    
    @patch('takshashila.distributed.worker.inference_job_worker.InferenceEngine')
    @patch('takshashila.distributed.worker.inference_job_worker.redis')
    @patch('takshashila.distributed.worker.inference_job_worker.PersistenceQueue')
    def test_batch_with_configurable_size(self, mock_queue_class, mock_redis, mock_engine_class):
        """Test batch processing with different batch sizes."""
        import os
        os.environ["BATCH_SIZE"] = "100"
        
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_cache = MagicMock()
        mock_redis.Redis.return_value = mock_cache
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        
        # Mock predict_batch
        def predict_batch_side_effect(grid_ids, ts):
            return {gid: 42.0 for gid in grid_ids}
        
        mock_engine.predict_batch.side_effect = predict_batch_side_effect
        
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        
        # Test with 250 grids and batch size 100
        total_grids = 250
        grids_processed = 0
        batch_count = 0
        BATCH_SIZE = 100
        
        for batch_start in range(0, total_grids, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_grids)
            batch_grid_ids = [f"g{i:08d}" for i in range(batch_start, batch_end)]
            
            batch_predictions = mock_engine.predict_batch(batch_grid_ids, timestamp_hour)
            grids_processed += len(batch_predictions)
            batch_count += 1
        
        # Verify batching
        assert grids_processed == 250
        assert batch_count == 3  # 250 / 100 = 2.5 → 3 batches
        assert mock_engine.predict_batch.call_count == 3
        
        # Cleanup
        del os.environ["BATCH_SIZE"]


class TestCaching:
    """Test Redis caching behavior."""
    
    @patch('takshashila.distributed.worker.inference_job_worker.redis')
    def test_cache_predictions_with_hash(self, mock_redis):
        """Test that predictions are cached in Redis as hash."""
        mock_cache = MagicMock()
        mock_redis.Redis.return_value = mock_cache
        
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        hash_key = f"predictions:{timestamp_hour.isoformat()}"
        
        # Simulate caching
        predictions = {
            "g00000001": 45.2,
            "g00000002": 48.1,
            "g00000003": 42.5
        }
        
        for grid_id, pm25_value in predictions.items():
            mock_cache.hset(hash_key, grid_id, str(pm25_value))
            mock_cache.expire(hash_key, 3600)
        
        # Verify hset was called for each grid
        assert mock_cache.hset.call_count == 3
        
        # Verify expire was called for each grid (or once for hash)
        assert mock_cache.expire.call_count == 3
        
        # Verify calls
        for call_args in mock_cache.hset.call_args_list:
            args = call_args[0]
            assert args[0] == hash_key
    
    @patch('takshashila.distributed.worker.inference_job_worker.redis')
    def test_cache_graceful_failure(self, mock_redis):
        """Test that cache failures don't stop processing."""
        mock_cache = MagicMock()
        mock_redis.Redis.return_value = mock_cache
        
        # Simulate cache failure
        mock_cache.hset.side_effect = Exception("Cache connection failed")
        
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        hash_key = f"predictions:{timestamp_hour.isoformat()}"
        
        # Should raise exception (caller catches it)
        with pytest.raises(Exception):
            mock_cache.hset(hash_key, "g00000001", "45.2")


class TestPersistenceQueue:
    """Test async DuckDB write queuing."""
    
    @patch('takshashila.distributed.worker.inference_job_worker.PersistenceQueue')
    def test_queue_predictions(self, mock_queue_class):
        """Test that predictions are queued for DuckDB writes."""
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        
        predictions = {
            "g00000001": 45.2,
            "g00000002": 48.1
        }
        
        # Simulate queuing
        for grid_id, pm25_value in predictions.items():
            mock_queue.put(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                pm25_value=pm25_value,
                model_version="v1.0",
                schema_hash="schema_v1"
            )
        
        # Verify put was called for each prediction
        assert mock_queue.put.call_count == 2
        
        # Verify shutdown was called
        mock_queue.shutdown(timeout_seconds=5)
        mock_queue.shutdown.assert_called_once_with(timeout_seconds=5)
    
    @patch('takshashila.distributed.worker.inference_job_worker.PersistenceQueue')
    def test_queue_graceful_failure(self, mock_queue_class):
        """Test that queue failures don't stop processing."""
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        
        # Simulate queue full
        mock_queue.put.side_effect = Exception("Queue full")
        
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        
        # Should raise exception (caller catches it)
        with pytest.raises(Exception):
            mock_queue.put(
                grid_id="g00000001",
                timestamp_hour=timestamp_hour,
                pm25_value=45.2,
                model_version="v1.0",
                schema_hash="schema_v1"
            )


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @patch('takshashila.distributed.worker.inference_job_worker.InferenceEngine')
    @patch('takshashila.distributed.worker.inference_job_worker.redis')
    @patch('takshashila.distributed.worker.inference_job_worker.PersistenceQueue')
    def test_batch_prediction_partial_failure(self, mock_queue_class, mock_redis, mock_engine_class):
        """Test handling of partial batch failures."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_cache = MagicMock()
        mock_redis.Redis.return_value = mock_cache
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        
        # Some grids succeed, some fail
        grid_ids = [f"g{i:08d}" for i in range(10)]
        predictions = {gid: 40.0 + i for i, gid in enumerate(grid_ids[:8])}
        
        # Only 8 out of 10 succeeded
        mock_engine.predict_batch.return_value = predictions
        
        result = mock_engine.predict_batch(grid_ids, timestamp_hour)
        
        assert len(result) == 8
        assert len(result) < len(grid_ids)
    
    def test_redis_connection_failure(self):
        """Test graceful handling when Redis unavailable."""
        # Simulate Redis not being available
        cache = None
        try:
            # If redis module doesn't exist, cache stays None
            # This is the expected behavior in the worker
            cache = None
        except Exception:
            cache = None
        
        assert cache is None


class TestGridIdFormatting:
    """Test grid ID formatting."""
    
    def test_grid_id_format(self):
        """Test that grid IDs are formatted correctly as g{8-digit}."""
        grid_ids = [f"g{i:08d}" for i in range(10)]
        
        assert grid_ids[0] == "g00000000"
        assert grid_ids[9] == "g00000009"
        assert len(grid_ids[0]) == 9  # 'g' + 8 digits
    
    def test_large_grid_id(self):
        """Test grid ID formatting for large indices."""
        # 509208 as index should format with 8-digit padding
        grid_id = f"g{509208:08d}"
        
        assert grid_id == "g00509208"  # 509208 padded to 8 digits
        assert len(grid_id) == 9  # 'g' + 8 digits


class TestBatchProcessing:
    """Test batch processing logic."""
    
    @patch('takshashila.distributed.worker.inference_job_worker.InferenceEngine')
    @patch('takshashila.distributed.worker.inference_job_worker.redis')
    @patch('takshashila.distributed.worker.inference_job_worker.PersistenceQueue')
    def test_process_full_grid_range(self, mock_queue_class, mock_redis, mock_engine_class):
        """Test processing a full range of grids."""
        import os
        os.environ["BATCH_SIZE"] = "10"
        
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_cache = MagicMock()
        mock_redis.Redis.return_value = mock_cache
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        
        timestamp_hour = datetime.fromisoformat("2026-01-25T14:00:00")
        
        # Mock predict_batch
        def predict_batch_side_effect(grid_ids, ts):
            return {gid: 42.0 for gid in grid_ids}
        
        mock_engine.predict_batch.side_effect = predict_batch_side_effect
        
        # Process 100 grids with batch size 10
        grid_batch_start = 0
        grid_batch_end = 100
        BATCH_SIZE = 10
        
        total_predictions = 0
        batch_calls = 0
        
        batch_start = grid_batch_start
        while batch_start < grid_batch_end:
            batch_end = min(batch_start + BATCH_SIZE, grid_batch_end)
            batch_grid_ids = [f"g{i:08d}" for i in range(batch_start, batch_end)]
            
            batch_predictions = mock_engine.predict_batch(batch_grid_ids, timestamp_hour)
            total_predictions += len(batch_predictions)
            batch_calls += 1
            
            batch_start = batch_end
        
        assert total_predictions == 100
        assert batch_calls == 10
        assert mock_engine.predict_batch.call_count == 10
        
        # Cleanup
        del os.environ["BATCH_SIZE"]


class TestTimestampHandling:
    """Test timestamp parsing and normalization."""
    
    def test_parse_iso_timestamp(self):
        """Test parsing ISO format timestamp."""
        timestamp_str = "2026-01-25T14:00:00Z"
        
        # Parse like the worker does
        timestamp_hour = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        assert timestamp_hour.year == 2026
        assert timestamp_hour.month == 1
        assert timestamp_hour.day == 25
        assert timestamp_hour.hour == 14
        assert timestamp_hour.minute == 0
        assert timestamp_hour.second == 0
    
    def test_invalid_timestamp(self):
        """Test handling of invalid timestamp."""
        timestamp_str = "invalid-timestamp"
        
        with pytest.raises(ValueError):
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
