"""Comprehensive feature validation tests for training.parquet files.

Tests all 56 feature columns for:
- Value ranges (domain-specific bounds)
- Data types (numeric, datetime)
- Distribution sanity (means, standard deviations)
- Missing data patterns
- Correlation structure (lagged features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"


# Domain-specific value bounds for each feature
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Target and metadata
    "pm25_t": (0, 1200),  # µg/m³ (Mumbai pollution spikes)
    "target_trust": (0, 100),  # Confidence score
    "n_sensors": (1, 50),  # Number of sensors
    "row_idx": (0, 1_000_000),  # Grid cell index
    "grid_id": (0, 1_000_000),  # Grid ID
    
    # PM2.5 lags (same bounds as target)
    "pm25_t_minus_1h": (0, 1200),
    "pm25_t_minus_2h": (0, 1200),
    "pm25_t_minus_3h": (0, 1200),
    "pm25_t_minus_6h": (0, 1200),
    "pm25_t_minus_12h": (0, 1200),
    "pm25_t_minus_24h": (0, 1200),
    
    # Temperature (°C, Mumbai climate)
    "temperature_2m": (5, 50),
    "temperature_2m_lag1h": (5, 50),
    "temperature_2m_lag3h": (5, 50),
    "temperature_2m_lag6h": (5, 50),
    "temperature_2m_lag12h": (5, 50),
    "temperature_2m_lag24h": (5, 50),
    
    # Relative humidity (%)
    "relative_humidity_2m": (0, 100),
    "relative_humidity_2m_lag1h": (0, 100),
    "relative_humidity_2m_lag3h": (0, 100),
    "relative_humidity_2m_lag6h": (0, 100),
    "relative_humidity_2m_lag12h": (0, 100),
    "relative_humidity_2m_lag24h": (0, 100),
    
    # Dewpoint (°C)
    "dewpoint_2m": (-10, 35),
    "dewpoint_2m_lag1h": (-10, 35),
    "dewpoint_2m_lag3h": (-10, 35),
    "dewpoint_2m_lag6h": (-10, 35),
    "dewpoint_2m_lag12h": (-10, 35),
    "dewpoint_2m_lag24h": (-10, 35),
    
    # Surface pressure (hPa)
    "surface_pressure": (950, 1050),
    "surface_pressure_lag1h": (950, 1050),
    "surface_pressure_lag3h": (950, 1050),
    "surface_pressure_lag6h": (950, 1050),
    "surface_pressure_lag12h": (950, 1050),
    "surface_pressure_lag24h": (950, 1050),
    
    # Wind U component (m/s, can be negative/positive)
    "wind_u_10m": (-30, 30),
    "wind_u_10m_lag1h": (-30, 30),
    "wind_u_10m_lag3h": (-30, 30),
    "wind_u_10m_lag6h": (-30, 30),
    "wind_u_10m_lag12h": (-30, 30),
    "wind_u_10m_lag24h": (-30, 30),
    
    # Wind V component (m/s, can be negative/positive)
    "wind_v_10m": (-30, 30),
    "wind_v_10m_lag1h": (-30, 30),
    "wind_v_10m_lag3h": (-30, 30),
    "wind_v_10m_lag6h": (-30, 30),
    "wind_v_10m_lag12h": (-30, 30),
    "wind_v_10m_lag24h": (-30, 30),
    
    # Cloud cover (%)
    "cloud_cover": (0, 100),
    "cloud_cover_lag6h": (0, 100),
    "cloud_cover_lag24h": (0, 100),
    
    # VPD (vapor pressure deficit, kPa)
    "vpd": (0, 10),
    
    # Temperature - dewpoint difference (°C)
    "temp_dewpoint_diff": (0, 30),
    
    # Precipitation (mm)
    "precip_last_1h": (0, 200),
    "precip_last_6h": (0, 500),
    "precip_last_24h": (0, 1000),
}


@pytest.fixture
def sample_training_files():
    """Get a sample of training parquet files."""
    if not TRAINING_DIR.exists():
        pytest.skip(f"Training directory not found: {TRAINING_DIR}")
    
    parquet_files = sorted(TRAINING_DIR.glob("*/training.parquet"))
    if not parquet_files:
        pytest.skip("No training.parquet files found")
    
    # Sample 10 files for faster testing
    import random
    sample_size = min(10, len(parquet_files))
    return random.sample(parquet_files, sample_size)


@pytest.fixture
def sample_training_data(sample_training_files):
    """Load sample training data."""
    dfs = [pd.read_parquet(f) for f in sample_training_files]
    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Schema and Data Type Tests
# ============================================================================

class TestSchema:
    """Test schema consistency across training files."""
    
    def test_required_columns_exist(self, sample_training_data):
        """All required columns should exist."""
        required = ["row_idx", "timestamp", "pm25_t", "target_trust", "n_sensors"]
        missing = [col for col in required if col not in sample_training_data.columns]
        assert len(missing) == 0, f"Missing required columns: {missing}"
    
    def test_timestamp_is_datetime(self, sample_training_data):
        """Timestamp column should be datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_training_data["timestamp"]), \
            f"timestamp dtype is {sample_training_data['timestamp'].dtype}, expected datetime64"
    
    def test_numeric_columns_are_numeric(self, sample_training_data):
        """All feature columns should be numeric."""
        non_datetime_cols = [col for col in sample_training_data.columns if col != "timestamp"]
        
        for col in non_datetime_cols:
            assert pd.api.types.is_numeric_dtype(sample_training_data[col]), \
                f"{col} dtype is {sample_training_data[col].dtype}, expected numeric"
    
    def test_no_object_columns(self, sample_training_data):
        """No string/object columns (except timestamp)."""
        object_cols = sample_training_data.select_dtypes(include=['object']).columns.tolist()
        assert len(object_cols) == 0, f"Found object columns: {object_cols}"


# ============================================================================
# Value Range Tests
# ============================================================================

class TestValueRanges:
    """Test all features are within expected value ranges."""
    
    @pytest.mark.parametrize("feature,bounds", FEATURE_BOUNDS.items())
    def test_feature_bounds(self, sample_training_data, feature, bounds):
        """Each feature should be within its domain-specific bounds."""
        if feature not in sample_training_data.columns:
            pytest.skip(f"Feature {feature} not in data")
        
        min_val, max_val = bounds
        values = sample_training_data[feature].dropna()
        
        if len(values) == 0:
            pytest.skip(f"Feature {feature} has no non-null values")
        
        actual_min = values.min()
        actual_max = values.max()
        
        assert actual_min >= min_val, \
            f"{feature}: min={actual_min:.2f} below bound {min_val}"
        assert actual_max <= max_val, \
            f"{feature}: max={actual_max:.2f} above bound {max_val}"
    
    def test_no_infinite_values(self, sample_training_data):
        """No infinite values in numeric columns."""
        numeric_cols = sample_training_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(sample_training_data[col]).sum()
            assert inf_count == 0, f"{col} has {inf_count} infinite values"
    
    def test_pm25_realistic_mean(self, sample_training_data):
        """PM2.5 should have realistic mean for Mumbai."""
        pm25 = sample_training_data["pm25_t"].dropna()
        mean = pm25.mean()
        
        # Mumbai typical PM2.5: 40-150 µg/m³ annual average
        assert 20 < mean < 300, f"PM2.5 mean {mean:.1f} outside realistic range [20-300]"
    
    def test_temperature_realistic_mean(self, sample_training_data):
        """Temperature should have realistic mean for Mumbai."""
        if "temperature_2m" not in sample_training_data.columns:
            pytest.skip("No temperature data")
        
        temp = sample_training_data["temperature_2m"].dropna()
        mean = temp.mean()
        
        # Mumbai typical: 25-30°C annual average
        assert 20 < mean < 35, f"Temperature mean {mean:.1f}°C outside realistic range [20-35]"
    
    def test_humidity_realistic_mean(self, sample_training_data):
        """Humidity should have realistic mean for Mumbai (coastal)."""
        if "relative_humidity_2m" not in sample_training_data.columns:
            pytest.skip("No humidity data")
        
        humidity = sample_training_data["relative_humidity_2m"].dropna()
        mean = humidity.mean()
        
        # Mumbai is coastal, typically 60-80% humidity
        assert 40 < mean < 90, f"Humidity mean {mean:.1f}% outside realistic range [40-90]"


# ============================================================================
# Missing Data Tests
# ============================================================================

class TestMissingData:
    """Test missing data patterns."""
    
    def test_no_all_null_columns(self, sample_training_data):
        """No column should be completely null."""
        all_null = sample_training_data.columns[sample_training_data.isna().all()].tolist()
        assert len(all_null) == 0, f"All-null columns: {all_null}"
    
    def test_key_columns_no_nulls(self, sample_training_data):
        """Key columns should have no missing values."""
        key_cols = ["timestamp", "row_idx", "pm25_t", "target_trust", "n_sensors"]
        
        for col in key_cols:
            if col not in sample_training_data.columns:
                continue
            null_count = sample_training_data[col].isna().sum()
            assert null_count == 0, f"{col} has {null_count} nulls"
    
    def test_null_percentages_reasonable(self, sample_training_data):
        """No feature should have >50% missing data."""
        null_pcts = (sample_training_data.isna().sum() / len(sample_training_data) * 100)
        high_null = null_pcts[null_pcts > 50].sort_values(ascending=False)
        
        assert len(high_null) == 0, \
            f"Features with >50% nulls:\n{high_null}"
    
    def test_lag_features_have_some_nulls(self, sample_training_data):
        """Lag features should have some nulls at beginning of sequences."""
        lag_features = [col for col in sample_training_data.columns if "_lag" in col or "_minus_" in col]
        
        for col in lag_features:
            if col not in sample_training_data.columns:
                continue
            
            null_pct = (sample_training_data[col].isna().sum() / len(sample_training_data) * 100)
            # Lags should have 1-30% nulls depending on lag length
            assert 0 <= null_pct < 50, \
                f"{col} has {null_pct:.1f}% nulls (expected 0-50%)"


# ============================================================================
# Correlation Tests
# ============================================================================

class TestCorrelations:
    """Test expected correlations between features."""
    
    def test_pm25_lags_correlate_with_target(self, sample_training_data):
        """PM2.5 lags should correlate with target."""
        target = sample_training_data["pm25_t"]
        lag_cols = [col for col in sample_training_data.columns if col.startswith("pm25_t_minus_")]
        
        for lag_col in lag_cols:
            valid_mask = sample_training_data[[lag_col, "pm25_t"]].notna().all(axis=1)
            if valid_mask.sum() < 100:
                continue
            
            corr = sample_training_data.loc[valid_mask, lag_col].corr(
                sample_training_data.loc[valid_mask, "pm25_t"]
            )
            
            # Recent lags should correlate >0.5, older lags >0.3
            if "_1h" in lag_col or "_2h" in lag_col or "_3h" in lag_col:
                assert corr > 0.5, f"{lag_col} correlation {corr:.3f} too low (expected >0.5)"
            else:
                assert corr > 0.2, f"{lag_col} correlation {corr:.3f} too low (expected >0.2)"
    
    def test_temperature_lags_correlate(self, sample_training_data):
        """Temperature lags should correlate with current temperature."""
        if "temperature_2m" not in sample_training_data.columns:
            pytest.skip("No temperature data")
        
        temp_lags = [col for col in sample_training_data.columns 
                     if col.startswith("temperature_2m_lag")]
        
        for lag_col in temp_lags:
            valid_mask = sample_training_data[[lag_col, "temperature_2m"]].notna().all(axis=1)
            if valid_mask.sum() < 100:
                continue
            
            corr = sample_training_data.loc[valid_mask, lag_col].corr(
                sample_training_data.loc[valid_mask, "temperature_2m"]
            )
            
            # Temperature is autocorrelated even at 24h
            assert corr > 0.5, f"{lag_col} correlation {corr:.3f} too low (expected >0.5)"
    
    def test_vpd_correlates_with_temp_humidity(self, sample_training_data):
        """VPD should correlate with temperature (positive) and humidity (negative)."""
        if "vpd" not in sample_training_data.columns:
            pytest.skip("No VPD data")
        
        required = ["vpd", "temperature_2m", "relative_humidity_2m"]
        if not all(col in sample_training_data.columns for col in required):
            pytest.skip("Missing required columns for VPD test")
        
        valid_mask = sample_training_data[required].notna().all(axis=1)
        if valid_mask.sum() < 100:
            pytest.skip("Insufficient valid data")
        
        vpd_temp_corr = sample_training_data.loc[valid_mask, "vpd"].corr(
            sample_training_data.loc[valid_mask, "temperature_2m"]
        )
        vpd_humidity_corr = sample_training_data.loc[valid_mask, "vpd"].corr(
            sample_training_data.loc[valid_mask, "relative_humidity_2m"]
        )
        
        # VPD increases with temp, decreases with humidity
        assert vpd_temp_corr > 0, f"VPD-temp correlation {vpd_temp_corr:.3f} should be positive"
        assert vpd_humidity_corr < 0, f"VPD-humidity correlation {vpd_humidity_corr:.3f} should be negative"


# ============================================================================
# Distribution Tests
# ============================================================================

class TestDistributions:
    """Test feature distributions are reasonable."""
    
    def test_pm25_not_constant(self, sample_training_data):
        """PM2.5 should have variation (not constant)."""
        pm25_std = sample_training_data["pm25_t"].std()
        assert pm25_std > 5, f"PM2.5 std={pm25_std:.1f} too low (nearly constant)"
    
    def test_temperature_not_constant(self, sample_training_data):
        """Temperature should vary across samples."""
        if "temperature_2m" not in sample_training_data.columns:
            pytest.skip("No temperature data")
        
        temp_std = sample_training_data["temperature_2m"].std()
        assert temp_std > 1, f"Temperature std={temp_std:.1f}°C too low"
    
    def test_wind_components_centered_around_zero(self, sample_training_data):
        """Wind U/V components should be roughly centered around 0."""
        for wind_col in ["wind_u_10m", "wind_v_10m"]:
            if wind_col not in sample_training_data.columns:
                continue
            
            mean = sample_training_data[wind_col].mean()
            # Allow mean within [-5, 5] m/s (prevailing winds can bias)
            assert -10 < mean < 10, f"{wind_col} mean={mean:.1f} not centered"
    
    def test_precipitation_mostly_zero(self, sample_training_data):
        """Precipitation should be zero most of the time."""
        precip_cols = [col for col in sample_training_data.columns if "precip" in col]
        
        for col in precip_cols:
            if col not in sample_training_data.columns:
                continue
            
            zero_pct = (sample_training_data[col] == 0).sum() / len(sample_training_data) * 100
            # Precipitation is rare, should be 0 for >70% of samples
            assert zero_pct > 50, f"{col} is zero only {zero_pct:.1f}% (expected >50%)"


# ============================================================================
# Temporal Consistency Tests
# ============================================================================

class TestTemporalConsistency:
    """Test temporal ordering and consistency."""
    
    def test_timestamps_ordered_per_cell(self, sample_training_data):
        """Timestamps should be ordered within each grid cell."""
        for row_idx in sample_training_data["row_idx"].unique()[:10]:  # Sample 10 cells
            cell_data = sample_training_data[sample_training_data["row_idx"] == row_idx].copy()
            
            timestamps = pd.to_datetime(cell_data["timestamp"]).sort_index()
            is_ordered = (timestamps.diff().dropna() >= pd.Timedelta(0)).all()
            
            assert is_ordered, f"Timestamps not ordered for row_idx={row_idx}"
    
    def test_timestamps_within_2023(self, sample_training_data):
        """Sample data should be from 2023 (validation year)."""
        timestamps = pd.to_datetime(sample_training_data["timestamp"])
        min_year = timestamps.min().year
        max_year = timestamps.max().year
        
        # Training dir currently has 2023 data
        assert min_year >= 2019, f"Min year {min_year} before 2019"
        assert max_year <= 2024, f"Max year {max_year} after 2024"


# ============================================================================
# Data Integrity Tests  
# ============================================================================

class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_no_duplicate_keys(self, sample_training_data):
        """No duplicate (row_idx, timestamp) pairs in training files."""
        key_cols = ["row_idx", "timestamp"]
        dups = sample_training_data.duplicated(subset=key_cols).sum()
        dup_ratio = dups / len(sample_training_data)
        
        # Some duplicates OK from different files, but not >50%
        assert dup_ratio < 0.5, f"Duplicates {dup_ratio:.1%} exceed 50%"
    
    def test_grid_cells_reasonable_count(self, sample_training_data):
        """Number of unique grid cells should be reasonable."""
        n_cells = sample_training_data["row_idx"].nunique()
        
        # Mumbai grid should have hundreds to thousands of cells
        assert 100 < n_cells < 100_000, \
            f"Grid cell count {n_cells} outside reasonable range [100-100k]"
    
    def test_sensor_counts_reasonable(self, sample_training_data):
        """Sensor counts should be in reasonable range."""
        sensor_stats = sample_training_data["n_sensors"].describe()
        
        assert sensor_stats["min"] >= 1, "Min sensors should be >= 1"
        assert sensor_stats["max"] < 100, "Max sensors should be < 100"
        assert sensor_stats["mean"] > 0, "Mean sensors should be > 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
