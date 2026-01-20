"""Tests for model predictor."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from Takshashila.phase1_minimal_inference.model_predictor import ModelPredictor
from Takshashila.phase1_minimal_inference.errors import ModelInferenceError
from Takshashila.phase1_minimal_inference.config import MODEL_PATH, N_FEATURES


class TestModelPredictor:
    """Test LightGBM model loading and prediction."""
    
    @pytest.fixture
    def predictor(self):
        """Create ModelPredictor instance."""
        return ModelPredictor(str(MODEL_PATH))
    
    @pytest.fixture
    def mock_features_df(self):
        """Create mock single-row feature DataFrame with realistic values."""
        # Create 64 random features (simplified for testing)
        data = {
            f"feature_{i}": [np.random.uniform(0, 100)] 
            for i in range(N_FEATURES)
        }
        return pd.DataFrame(data)
    
    def test_model_loads_successfully(self, predictor):
        """Test model loads from disk without error."""
        predictor.load_model()
        assert predictor.model is not None
    
    def test_predict_valid_input(self, predictor, assembler, valid_grid_id, test_timestamp):
        """Test prediction with real features from assembler."""
        # Get real features
        features_df = assembler.fetch_features(valid_grid_id, test_timestamp)
        
        # Run prediction
        result = predictor.predict(features_df, valid_grid_id, test_timestamp)
        
        # Check output type and validity
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)
        assert result >= 0  # PM2.5 can't be negative
    
    def test_predict_wrong_row_count_raises_error(self, predictor):
        """Test prediction with wrong number of rows raises error."""
        # Create multi-row DataFrame
        data = {f"feature_{i}": [1.0, 2.0, 3.0] for i in range(N_FEATURES)}
        features_df = pd.DataFrame(data)
        
        with pytest.raises(ModelInferenceError) as exc_info:
            predictor.predict(features_df)
        
        assert "Expected 1 row" in str(exc_info.value)
    
    def test_predict_wrong_feature_count_raises_error(self, predictor):
        """Test prediction with wrong number of features raises error."""
        # Create DataFrame with wrong feature count
        data = {f"feature_{i}": [1.0] for i in range(10)}  # Only 10 features
        features_df = pd.DataFrame(data)
        
        with pytest.raises(ModelInferenceError) as exc_info:
            predictor.predict(features_df)
        
        assert "Expected 64 features" in str(exc_info.value)
    
    def test_predict_with_nan_raises_error(self, predictor):
        """Test prediction with NaN values raises error."""
        data = {f"feature_{i}": [np.nan if i == 0 else 1.0] for i in range(N_FEATURES)}
        features_df = pd.DataFrame(data)
        
        with pytest.raises(ModelInferenceError) as exc_info:
            predictor.predict(features_df)
        
        assert "NaN" in str(exc_info.value)
    
    def test_predict_with_inf_raises_error(self, predictor):
        """Test prediction with infinite values raises error."""
        data = {f"feature_{i}": [np.inf if i == 0 else 1.0] for i in range(N_FEATURES)}
        features_df = pd.DataFrame(data)
        
        with pytest.raises(ModelInferenceError) as exc_info:
            predictor.predict(features_df)
        
        assert "infinite" in str(exc_info.value).lower()
    
    def test_lazy_loading(self, predictor):
        """Test model is lazy-loaded on first predict call."""
        assert predictor.model is None
        
        # Load explicitly
        predictor.load_model()
        assert predictor.model is not None
        
        # Loading again should be no-op
        predictor.load_model()
        assert predictor.model is not None
