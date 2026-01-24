"""Tests for feature assembler."""
import pytest
from datetime import datetime, timezone

from takshashila.core.feature_assembler import FeatureAssembler
from takshashila.core.errors import FeatureAssemblyError, InvalidRequestError
from takshashila.core.config import FEATURE_SCHEMA_PATH
import json


class TestFeatureAssembler:
    """Test feature assembly from DuckDB."""
    
    @pytest.fixture
    def feature_schema(self):
        """Load feature schema."""
        with open(FEATURE_SCHEMA_PATH, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def assembler(self, test_db_paths, feature_schema):
        """Create FeatureAssembler instance."""
        return FeatureAssembler(test_db_paths, feature_schema)
    
    def test_fetch_features_valid_grid(self, assembler, valid_grid_id, test_timestamp):
        """Test fetching features for valid grid with complete data."""
        features_df = assembler.fetch_features(valid_grid_id, test_timestamp)
        
        # Check shape
        assert features_df.shape[0] == 1  # Single row
        assert features_df.shape[1] == 64  # 64 features
        
        # Check no null values
        assert features_df.isnull().sum().sum() == 0
        
        # Check column order matches schema
        expected_cols = assembler.required_features
        assert list(features_df.columns) == expected_cols
    
    def test_fetch_features_invalid_grid_raises_error(self, assembler, invalid_grid_id, test_timestamp):
        """Test fetching features for invalid grid_id raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            assembler.fetch_features(invalid_grid_id, test_timestamp)
        
        assert invalid_grid_id in str(exc_info.value)
    
    def test_fetch_features_missing_pm25_raises_error(self, assembler, missing_pm25_grid_id, test_timestamp):
        """Test missing PM2.5 features raises FeatureAssemblyError."""
        with pytest.raises(FeatureAssemblyError) as exc_info:
            assembler.fetch_features(missing_pm25_grid_id, test_timestamp)
        
        assert exc_info.value.grid_id == missing_pm25_grid_id
        assert exc_info.value.timestamp_hour == test_timestamp
        assert "pm25_observation" in str(exc_info.value)
    
    def test_fetch_features_missing_meteo_raises_error(self, assembler, missing_meteo_grid_id, test_timestamp):
        """Test missing meteorological features raises FeatureAssemblyError."""
        with pytest.raises(FeatureAssemblyError) as exc_info:
            assembler.fetch_features(missing_meteo_grid_id, test_timestamp)
        
        assert exc_info.value.grid_id == missing_meteo_grid_id
        assert "meteo" in str(exc_info.value)
    
    def test_fetch_features_missing_satellite_raises_error(self, assembler, missing_sat_grid_id, test_timestamp):
        """Test missing satellite features raises FeatureAssemblyError."""
        with pytest.raises(FeatureAssemblyError) as exc_info:
            assembler.fetch_features(missing_sat_grid_id, test_timestamp)
        
        assert exc_info.value.grid_id == missing_sat_grid_id
        assert "landsat8" in str(exc_info.value) or "sentinel2" in str(exc_info.value)
    
    def test_feature_count_matches_schema(self, assembler):
        """Test assembler knows correct number of features."""
        assert assembler.n_features == 64
        assert len(assembler.required_features) == 64
