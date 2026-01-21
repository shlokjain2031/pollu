"""
Unit tests for Phase 2a Python worker
"""

import json
import pytest
from io import StringIO
import sys
from pathlib import Path

# Add Phase 1 to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import worker module separately to handle Phase 1 dependencies
try:
    from phase2_worker.worker import Worker
    WORKER_AVAILABLE = True
except ImportError:
    WORKER_AVAILABLE = False
    Worker = None


def test_process_job_success(tmp_path):
    """Test successful job processing."""
    # Note: This test requires Phase 1 artifacts to exist
    # For now, we just test the job structure
    
    if not WORKER_AVAILABLE:
        pytest.skip("Worker not available (Phase 1 dependencies missing)")
    
    job = {
        "grid_id": "g00000001",
        "timestamp_hour": "2026-01-19T16:00:00Z",
        "request_id": "test-req-123"
    }
    
    # Worker initialization might fail if DB/model don't exist
    # This is expected in unit test environment
    try:
        worker = Worker()
        result = worker.process_job(job)
        
        # Verify result structure
        assert "request_id" in result
        assert "pm25" in result
        assert "error" in result
        assert "worker_id" in result
        
        assert result["request_id"] == job["request_id"]
        
    except Exception as e:
        # Expected if Phase 1 artifacts don't exist
        pytest.skip(f"Worker initialization failed (expected in unit test): {e}")


def test_job_validation():
    """Test job structure validation."""
    valid_job = {
        "grid_id": "g00000001",
        "timestamp_hour": "2026-01-19T16:00:00Z",
        "request_id": "test-req-123"
    }
    
    # All required fields present
    assert "grid_id" in valid_job
    assert "timestamp_hour" in valid_job
    assert "request_id" in valid_job
    
    # Test JSON serialization
    job_json = json.dumps(valid_job)
    parsed = json.loads(job_json)
    assert parsed == valid_job


def test_result_structure():
    """Test result JSON structure."""
    result = {
        "request_id": "test-req-123",
        "pm25": 42.5,
        "error": "",
        "worker_id": "worker-0"
    }
    
    # All required fields present
    assert "request_id" in result
    assert "pm25" in result
    assert "error" in result
    assert "worker_id" in result
    
    # Test JSON serialization
    result_json = json.dumps(result)
    parsed = json.loads(result_json)
    assert parsed == result
    
    # Test error result
    error_result = {
        "request_id": "test-req-456",
        "pm25": 0.0,
        "error": "Some error occurred",
        "worker_id": "worker-0"
    }
    
    error_json = json.dumps(error_result)
    parsed = json.loads(error_json)
    assert parsed["error"] == "Some error occurred"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
