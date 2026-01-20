"""Tests for error classes and their attributes."""
import pytest
from datetime import datetime, timezone

from Takshashila.phase1_minimal_inference.errors import (
    InferenceError, FeatureAssemblyError, ModelInferenceError,
    PersistenceError, TimeoutError, InvalidRequestError
)


class TestErrorClasses:
    """Test all error class attributes and message formatting."""
    
    def test_inference_error_base(self):
        """Test base InferenceError class."""
        err = InferenceError("test error")
        assert str(err) == "test error"
        assert isinstance(err, Exception)
    
    def test_feature_assembly_error_attributes(self):
        """Test FeatureAssemblyError has correct attributes."""
        grid_id = "g00000001"
        timestamp = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        missing_feature = "pm25_t_minus_1h"
        
        err = FeatureAssemblyError(grid_id, timestamp, missing_feature)
        
        assert err.grid_id == grid_id
        assert err.timestamp_hour == timestamp
        assert err.missing_feature == missing_feature
        assert "pm25_t_minus_1h" in str(err)
        assert "g00000001" in str(err)
    
    def test_model_inference_error_attributes(self):
        """Test ModelInferenceError has correct attributes."""
        grid_id = "g00000001"
        timestamp = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        reason = "Model returned NaN"
        
        err = ModelInferenceError(grid_id, timestamp, reason)
        
        assert err.grid_id == grid_id
        assert err.timestamp_hour == timestamp
        assert err.reason == reason
        assert "Model returned NaN" in str(err)
        assert "g00000001" in str(err)
    
    def test_persistence_error_attributes(self):
        """Test PersistenceError has correct attributes."""
        operation = "write"
        grid_id = "g00000001"
        timestamp = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        reason = "Connection failed"
        
        err = PersistenceError(operation, grid_id, timestamp, reason)
        
        assert err.operation == operation
        assert err.grid_id == grid_id
        assert err.timestamp_hour == timestamp
        assert err.reason == reason
        assert "write" in str(err)
        assert "Connection failed" in str(err)
    
    def test_timeout_error_attributes(self):
        """Test TimeoutError has correct attributes."""
        grid_id = "g00000001"
        timestamp = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
        timeout_sec = 2.0
        
        err = TimeoutError(grid_id, timestamp, timeout_sec)
        
        assert err.grid_id == grid_id
        assert err.timestamp_hour == timestamp
        assert err.timeout_sec == timeout_sec
        assert "2.0" in str(err) or "2" in str(err)
    
    def test_invalid_request_error_attributes(self):
        """Test InvalidRequestError has correct attributes."""
        reason = "grid_id 'invalid' not found"
        
        err = InvalidRequestError(reason)
        
        assert err.reason == reason
        assert "grid_id 'invalid' not found" in str(err)
    
    def test_all_errors_inherit_from_inference_error(self):
        """Test all custom errors inherit from InferenceError."""
        assert issubclass(FeatureAssemblyError, InferenceError)
        assert issubclass(ModelInferenceError, InferenceError)
        assert issubclass(PersistenceError, InferenceError)
        assert issubclass(TimeoutError, InferenceError)
        assert issubclass(InvalidRequestError, InferenceError)
