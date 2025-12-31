"""Tests for nalanda/training.py (training data assembly)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalanda.training import (
    load_grid_mapping,
    get_target_grid_ids,
    load_target,
    assemble_training_data,
    save_training_data,
    get_training_feature_columns,
    DEFAULT_TARGET_DIR,
    DEFAULT_GRID_PATH,
    DEFAULT_OUTPUT_DIR,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def grid_mapping():
    """Load grid mapping fixture."""
    return load_grid_mapping()


@pytest.fixture
def target_df():
    """Load target data for test date."""
    return load_target('2021-06-12')


@pytest.fixture
def training_df():
    """Full assembled training data for test date."""
    return assemble_training_data('2021-06-12', skip_missing_satellite=True)


# ============================================================================
# Grid mapping tests
# ============================================================================

class TestGridMapping:
    """Tests for grid_id <-> row_idx mapping utilities."""
    
    def test_load_grid_mapping_returns_dataframe(self, grid_mapping):
        """Grid mapping loads as DataFrame."""
        assert isinstance(grid_mapping, pd.DataFrame)
    
    def test_grid_mapping_has_required_columns(self, grid_mapping):
        """Grid mapping has row_idx and grid_id columns."""
        assert 'row_idx' in grid_mapping.columns
        assert 'grid_id' in grid_mapping.columns
    
    def test_grid_mapping_row_idx_is_unique(self, grid_mapping):
        """Each row_idx appears exactly once."""
        assert grid_mapping['row_idx'].is_unique
    
    def test_grid_mapping_grid_id_is_unique(self, grid_mapping):
        """Each grid_id appears exactly once."""
        assert grid_mapping['grid_id'].is_unique
    
    def test_grid_mapping_size_matches_grid(self, grid_mapping):
        """Grid mapping has expected ~509k rows."""
        assert len(grid_mapping) > 500_000


class TestGetTargetGridIds:
    """Tests for extracting grid_ids from target data."""
    
    def test_returns_set(self, target_df, grid_mapping):
        """get_target_grid_ids returns a set."""
        result = get_target_grid_ids(target_df, grid_mapping)
        assert isinstance(result, set)
    
    def test_returns_nonempty(self, target_df, grid_mapping):
        """Returns non-empty set for valid target."""
        result = get_target_grid_ids(target_df, grid_mapping)
        assert len(result) > 0
    
    def test_count_matches_unique_row_idx(self, target_df, grid_mapping):
        """Number of grid_ids matches unique row_idx count."""
        result = get_target_grid_ids(target_df, grid_mapping)
        expected = target_df['row_idx'].nunique()
        assert len(result) == expected
    
    def test_grid_ids_are_strings(self, target_df, grid_mapping):
        """All returned grid_ids are strings."""
        result = get_target_grid_ids(target_df, grid_mapping)
        for grid_id in result:
            assert isinstance(grid_id, str)


# ============================================================================
# Target loading tests
# ============================================================================

class TestLoadTarget:
    """Tests for target data loading."""
    
    def test_load_target_returns_dataframe(self, target_df):
        """Target loads as DataFrame."""
        assert isinstance(target_df, pd.DataFrame)
    
    def test_target_has_required_columns(self, target_df):
        """Target has all required columns."""
        required = ['row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors']
        for col in required:
            assert col in target_df.columns
    
    def test_target_timestamp_is_datetime(self, target_df):
        """Timestamp column is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(target_df['timestamp'])
    
    def test_target_has_expected_row_count(self, target_df):
        """Target has ~76k rows for test date."""
        assert len(target_df) > 70_000
        assert len(target_df) < 100_000
    
    def test_target_row_idx_is_integer(self, target_df):
        """row_idx is integer type."""
        assert pd.api.types.is_integer_dtype(target_df['row_idx'])
    
    def test_target_pm25_in_valid_range(self, target_df):
        """pm25_t values are in reasonable range."""
        assert target_df['pm25_t'].min() >= 0
        assert target_df['pm25_t'].max() < 500  # reasonable upper bound
    
    def test_load_missing_date_raises(self):
        """Loading nonexistent date raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_target('1999-01-01')


# ============================================================================
# Assembly tests
# ============================================================================

class TestAssembleTrainingData:
    """Tests for full training data assembly."""
    
    def test_returns_dataframe(self, training_df):
        """Assembly returns DataFrame."""
        assert isinstance(training_df, pd.DataFrame)
    
    def test_row_count_matches_target(self, training_df, target_df):
        """Row count equals target row count (1:1 join)."""
        assert len(training_df) == len(target_df)
    
    def test_has_key_columns(self, training_df):
        """Has row_idx and timestamp keys."""
        assert 'row_idx' in training_df.columns
        assert 'timestamp' in training_df.columns
    
    def test_has_target_columns(self, training_df):
        """Has target label columns."""
        assert 'pm25_t' in training_df.columns
        assert 'target_trust' in training_df.columns
        assert 'n_sensors' in training_df.columns
    
    def test_has_meteo_features(self, training_df):
        """Has meteorological features."""
        assert 'temperature_2m' in training_df.columns
        assert 'wind_u_10m' in training_df.columns
        assert 'vpd' in training_df.columns
    
    def test_has_meteo_lags(self, training_df):
        """Has temporal lag features."""
        assert 'temperature_2m_lag1h' in training_df.columns
        assert 'temperature_2m_lag24h' in training_df.columns
    
    def test_has_landsat_indices(self, training_df):
        """Has Landsat-derived indices."""
        assert 'ndvi' in training_df.columns
        assert 'ndbi' in training_df.columns
        assert 'bsi' in training_df.columns
    
    def test_has_sentinel_features(self, training_df):
        """Has Sentinel-derived features."""
        assert 'evi' in training_df.columns
        assert 'cloud_mask' in training_df.columns
        assert 'cloud_fraction_30m' in training_df.columns
    
    def test_has_satellite_dates(self, training_df):
        """Has metadata about satellite source dates."""
        assert 'landsat_date' in training_df.columns
        assert 'sentinel_date' in training_df.columns
    
    def test_column_order_keys_first(self, training_df):
        """row_idx and timestamp are first columns."""
        cols = training_df.columns.tolist()
        assert cols[0] == 'row_idx'
        assert cols[1] == 'timestamp'
    
    def test_no_duplicate_rows(self, training_df):
        """No duplicate (row_idx, timestamp) combinations."""
        dups = training_df.duplicated(subset=['row_idx', 'timestamp'])
        assert not dups.any()
    
    def test_meteo_features_low_nulls(self, training_df):
        """Meteo features have 0% nulls (perfect coverage)."""
        meteo_cols = [c for c in training_df.columns if 'temperature' in c or 'wind' in c]
        for col in meteo_cols:
            null_pct = training_df[col].isna().mean()
            assert null_pct == 0, f"{col} has {null_pct*100:.1f}% nulls"
    
    def test_landsat_features_low_nulls(self, training_df):
        """Landsat indices have low nulls."""
        landsat_cols = ['ndvi', 'ndbi', 'mndwi', 'bsi', 'ibi']
        for col in landsat_cols:
            if col in training_df.columns:
                null_pct = training_df[col].isna().mean()
                assert null_pct < 0.1, f"{col} has {null_pct*100:.1f}% nulls"


class TestSaveTrainingData:
    """Tests for saving training data."""
    
    def test_save_creates_file(self, training_df, tmp_path):
        """save_training_data creates parquet file."""
        output_path = save_training_data(training_df, '2021-06-12', tmp_path)
        assert output_path.exists()
    
    def test_saved_file_is_valid_parquet(self, training_df, tmp_path):
        """Saved file is readable parquet."""
        output_path = save_training_data(training_df, '2021-06-12', tmp_path)
        loaded = pd.read_parquet(output_path)
        assert len(loaded) == len(training_df)
    
    def test_save_roundtrip_preserves_columns(self, training_df, tmp_path):
        """Columns are preserved after save/load."""
        output_path = save_training_data(training_df, '2021-06-12', tmp_path)
        loaded = pd.read_parquet(output_path)
        assert set(loaded.columns) == set(training_df.columns)


# ============================================================================
# Feature column catalog tests
# ============================================================================

class TestGetTrainingFeatureColumns:
    """Tests for feature column list utility."""
    
    def test_returns_list(self):
        """Returns a list."""
        result = get_training_feature_columns()
        assert isinstance(result, list)
    
    def test_returns_nonempty(self):
        """Returns non-empty list."""
        result = get_training_feature_columns()
        assert len(result) > 0
    
    def test_includes_meteo_cols(self):
        """Includes temperature feature."""
        result = get_training_feature_columns()
        assert 'temperature_2m' in result
    
    def test_includes_landsat_cols(self):
        """Includes Landsat indices."""
        result = get_training_feature_columns()
        assert 'ndvi' in result
    
    def test_includes_sentinel_cols(self):
        """Includes Sentinel features."""
        result = get_training_feature_columns()
        assert 'evi' in result
    
    def test_with_landsat_bands(self):
        """Can include raw Landsat bands."""
        without = get_training_feature_columns(include_landsat_bands=False)
        with_bands = get_training_feature_columns(include_landsat_bands=True)
        assert len(with_bands) > len(without)


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_produces_valid_data(self, training_df):
        """Full pipeline produces usable training data."""
        # Has features and targets
        assert 'pm25_t' in training_df.columns
        assert 'temperature_2m' in training_df.columns
        
        # Reasonable number of features
        assert len(training_df.columns) > 50
        
        # No all-null columns
        for col in training_df.columns:
            assert training_df[col].notna().any(), f"{col} is all null"
    
    def test_different_dates_possible(self):
        """Can assemble for different dates (if target exists)."""
        # Just test 2021-06-12 since we know it exists
        df = assemble_training_data('2021-06-12')
        assert len(df) > 0
    
    def test_skip_missing_satellite_fills_nulls(self):
        """skip_missing_satellite=True fills NaN for missing data."""
        # This relies on our test data structure
        df = assemble_training_data('2021-06-12', skip_missing_satellite=True)
        # Should still have satellite columns even if source data is old
        assert 'ndvi' in df.columns
        assert 'evi' in df.columns


# ============================================================================
# Data quality tests
# ============================================================================

class TestDataQuality:
    """Tests for data quality and consistency."""
    
    def test_timestamp_coverage(self, training_df):
        """Timestamps cover expected hourly range."""
        timestamps = training_df['timestamp']
        min_hour = timestamps.dt.hour.min()
        max_hour = timestamps.dt.hour.max()
        # Should have range of hours
        assert max_hour - min_hour >= 12  # at least 12 hours of coverage
    
    def test_grid_coverage(self, training_df):
        """Multiple grid cells covered."""
        unique_grids = training_df['row_idx'].nunique()
        assert unique_grids > 1000
    
    def test_temperature_in_reasonable_range(self, training_df):
        """Temperature values are physically reasonable."""
        temp = training_df['temperature_2m']
        assert temp.min() > -50  # °C
        assert temp.max() < 60   # °C
    
    def test_humidity_in_valid_range(self, training_df):
        """Humidity in 0-100% range."""
        rh = training_df['relative_humidity_2m']
        assert rh.min() >= 0
        assert rh.max() <= 100
    
    def test_ndvi_in_valid_range(self, training_df):
        """NDVI in theoretical -1 to 1 range."""
        ndvi = training_df['ndvi'].dropna()
        assert ndvi.min() >= -1
        assert ndvi.max() <= 1
    
    def test_evi_in_valid_range(self, training_df):
        """EVI in reasonable range."""
        evi = training_df['evi'].dropna()
        assert evi.min() > -1
        assert evi.max() < 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
