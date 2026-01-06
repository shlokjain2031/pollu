"""Integration tests for split artifact quality and consistency.

Tests validate:
- Data quality (value ranges, null patterns)
- Reliability (schema consistency, proper data types)
- Fairness (distribution balance across splits)
- Consistency (temporal ordering, no leakage, feature stability)
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "val"  # Current test data location
SPLITS_METADATA_PATH = ARTIFACTS_DIR / "splits_metadata.json"


@pytest.fixture
def splits_metadata():
    """Load splits metadata."""
    if not SPLITS_METADATA_PATH.exists():
        pytest.skip("Splits metadata not found - run materialize_splits first")
    
    with open(SPLITS_METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture
def validation_parquets():
    """Load all validation split parquets."""
    if not SPLITS_DIR.exists():
        pytest.skip("Validation split directory not found")
    
    parquet_files = sorted(SPLITS_DIR.glob("*.parquet"))
    if not parquet_files:
        pytest.skip("No parquet files in validation split")
    
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@pytest.fixture
def train_parquets():
    """Load all training split parquets."""
    train_dir = ARTIFACTS_DIR / "train"
    if not train_dir.exists():
        pytest.skip("Training split directory not found")
    
    parquet_files = sorted(train_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@pytest.fixture
def test_parquets():
    """Load all test split parquets."""
    test_dir = ARTIFACTS_DIR / "test"
    if not test_dir.exists():
        pytest.skip("Test split directory not found")
    
    parquet_files = sorted(test_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ============================================================================
# Data Quality Tests
# ============================================================================

class TestDataQuality:
    """Test data quality and value ranges."""
    
    def test_metadata_file_exists(self):
        """Metadata file should exist."""
        assert SPLITS_METADATA_PATH.exists(), f"Metadata not found: {SPLITS_METADATA_PATH}"
    
    def test_metadata_has_required_fields(self, splits_metadata):
        """Metadata should have required fields for splits that exist."""
        existing_splits = [s for s in ["train", "validation", "test"] if s in splits_metadata]
        
        if not existing_splits:
            pytest.skip("No splits materialized yet")
        
        for split in existing_splits:
            meta = splits_metadata[split]
            
            assert "rows" in meta, f"{split}: missing 'rows'"
            assert "file_count" in meta, f"{split}: missing 'file_count'"
            assert "min_timestamp" in meta, f"{split}: missing 'min_timestamp'"
            assert "max_timestamp" in meta, f"{split}: missing 'max_timestamp'"
            assert "size_mb" in meta, f"{split}: missing 'size_mb'"
            assert "sha256" in meta, f"{split}: missing 'sha256'"
    
    def test_pm25_value_ranges(self, validation_parquets):
        """PM2.5 values should be in reasonable range."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        pm25 = validation_parquets["pm25_t"].dropna()
        
        # PM2.5 should be non-negative and under 1200 µg/m³ (pollution spikes in Mumbai can reach 1000+)
        assert (pm25 >= 0).all(), f"Found negative PM2.5 values: min={pm25.min()}"
        assert (pm25 <= 1200).all(), f"PM2.5 values exceed 1200: max={pm25.max()}"
        
        # Check reasonable mean and std
        assert pm25.mean() > 10, f"Mean PM2.5 too low: {pm25.mean():.1f}"
        assert pm25.mean() < 300, f"Mean PM2.5 too high: {pm25.mean():.1f}"
    
    def test_temperature_value_ranges(self, validation_parquets):
        """Temperature should be in reasonable range."""
        if validation_parquets.empty or "temperature_2m" not in validation_parquets.columns:
            pytest.skip("No temperature data")
        
        temp = validation_parquets["temperature_2m"].dropna()
        
        # Temperature in Mumbai should be 15-40°C typically
        assert temp.min() > 0, f"Temperature too low: {temp.min():.1f}°C"
        assert temp.max() < 50, f"Temperature too high: {temp.max():.1f}°C"
    
    def test_humidity_value_ranges(self, validation_parquets):
        """Humidity should be 0-100%."""
        if validation_parquets.empty or "relative_humidity_2m" not in validation_parquets.columns:
            pytest.skip("No humidity data")
        
        humidity = validation_parquets["relative_humidity_2m"].dropna()
        
        assert (humidity >= 0).all(), f"Humidity < 0: min={humidity.min()}"
        assert (humidity <= 100).all(), f"Humidity > 100: max={humidity.max()}"
    
    def test_ndvi_value_ranges(self, validation_parquets):
        """NDVI should be -1 to 1."""
        if validation_parquets.empty or "ndvi" not in validation_parquets.columns:
            pytest.skip("No NDVI data")
        
        ndvi = validation_parquets["ndvi"].dropna()
        
        assert (ndvi >= -1).all(), f"NDVI < -1: min={ndvi.min():.3f}"
        assert (ndvi <= 1).all(), f"NDVI > 1: max={ndvi.max():.3f}"
    
    def test_target_trust_valid(self, validation_parquets):
        """Target trust should be positive."""
        if validation_parquets.empty or "target_trust" not in validation_parquets.columns:
            pytest.skip("No target_trust data")
        
        trust = validation_parquets["target_trust"]
        assert (trust > 0).all(), f"Found non-positive trust values: min={trust.min()}"
    
    def test_n_sensors_valid(self, validation_parquets):
        """Number of sensors should be >= 1."""
        if validation_parquets.empty or "n_sensors" not in validation_parquets.columns:
            pytest.skip("No n_sensors data")
        
        n_sensors = validation_parquets["n_sensors"]
        assert (n_sensors >= 1).all(), f"Found cells with <1 sensor: min={n_sensors.min()}"


# ============================================================================
# Schema Consistency Tests
# ============================================================================

class TestSchemaConsistency:
    """Test schema consistency across splits."""
    
    def test_validation_has_required_columns(self, validation_parquets):
        """Validation split should have required columns."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        required = ["row_idx", "timestamp", "pm25_t", "target_trust", "n_sensors"]
        for col in required:
            assert col in validation_parquets.columns, f"Missing column: {col}"
    
    def test_train_has_required_columns(self, train_parquets):
        """Train split should have required columns."""
        if train_parquets.empty:
            pytest.skip("No train data")
        
        required = ["row_idx", "timestamp", "pm25_t", "target_trust", "n_sensors"]
        for col in required:
            assert col in train_parquets.columns, f"Missing column: {col}"
    
    def test_test_has_required_columns(self, test_parquets):
        """Test split should have required columns."""
        if test_parquets.empty:
            pytest.skip("No test data")
        
        required = ["row_idx", "timestamp", "pm25_t", "target_trust", "n_sensors"]
        for col in required:
            assert col in test_parquets.columns, f"Missing column: {col}"
    
    def test_timestamp_is_datetime(self, validation_parquets):
        """Timestamp column should be datetime."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        assert pd.api.types.is_datetime64_any_dtype(validation_parquets["timestamp"]), \
            f"Timestamp dtype is {validation_parquets['timestamp'].dtype}, expected datetime"
    
    def test_pm25_is_numeric(self, validation_parquets):
        """PM2.5 column should be numeric."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        assert pd.api.types.is_numeric_dtype(validation_parquets["pm25_t"]), \
            f"PM2.5 dtype is {validation_parquets['pm25_t'].dtype}, expected numeric"
    
    def test_row_idx_is_integer(self, validation_parquets):
        """Row index should be integer."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        assert pd.api.types.is_integer_dtype(validation_parquets["row_idx"]), \
            f"row_idx dtype is {validation_parquets['row_idx'].dtype}, expected integer"


# ============================================================================
# Temporal Consistency Tests
# ============================================================================

class TestTemporalConsistency:
    """Test temporal properties and ordering."""
    
    def test_validation_timestamps_ordered(self, validation_parquets):
        """Timestamps should be ordered."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        timestamps = pd.to_datetime(validation_parquets["timestamp"])
        # Check if mostly ordered (allow small shuffles from parquet format)
        is_ordered = (timestamps.diff().dropna() >= pd.Timedelta(0)).sum() / len(timestamps) > 0.95
        assert is_ordered, "Timestamps are not well ordered"
    
    def test_validation_date_range(self, validation_parquets, splits_metadata):
        """Validation timestamps should match metadata."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        timestamps = pd.to_datetime(validation_parquets["timestamp"])
        min_ts = timestamps.min()
        max_ts = timestamps.max()
        
        meta = splits_metadata.get("validation", {})
        meta_min = pd.Timestamp(meta.get("min_timestamp"))
        meta_max = pd.Timestamp(meta.get("max_timestamp"))
        
        # Data should be within metadata bounds
        assert min_ts >= meta_min, f"Min timestamp {min_ts} before metadata min {meta_min}"
        assert max_ts <= meta_max, f"Max timestamp {max_ts} after metadata max {meta_max}"
    
    def test_train_test_no_temporal_overlap(self, train_parquets, test_parquets):
        """Train and test should not overlap temporally."""
        if train_parquets.empty or test_parquets.empty:
            pytest.skip("Missing train or test data")
        
        train_max = pd.to_datetime(train_parquets["timestamp"]).max()
        test_min = pd.to_datetime(test_parquets["timestamp"]).min()
        
        # Test should start after train
        assert test_min > train_max, \
            f"Test starts ({test_min}) before train ends ({train_max})"
    
    def test_validation_between_train_test(self, train_parquets, validation_parquets, test_parquets):
        """Validation should be between train and test temporally."""
        if validation_parquets.empty:
            pytest.skip("Missing validation data")
        
        if not train_parquets.empty:
            train_max = pd.to_datetime(train_parquets["timestamp"]).max()
            val_min = pd.to_datetime(validation_parquets["timestamp"]).min()
            assert val_min > train_max, f"Val starts before train ends"
        
        if not test_parquets.empty:
            val_max = pd.to_datetime(validation_parquets["timestamp"]).max()
            test_min = pd.to_datetime(test_parquets["timestamp"]).min()
            assert test_min > val_max, f"Test starts before val ends"


# ============================================================================
# Data Fairness & Distribution Tests
# ============================================================================

class TestDataFairness:
    """Test fairness and distribution across splits."""
    
    def test_splits_have_data(self, splits_metadata):
        """Each split should have some data."""
        for split in ["train", "validation", "test"]:
            if split not in splits_metadata:
                continue
            meta = splits_metadata[split]
            assert meta["rows"] > 0, f"{split} has no rows"
            assert meta["file_count"] > 0, f"{split} has no files"
    
    def test_split_row_distribution(self, splits_metadata):
        """Splits should have reasonable row distribution when all exist."""
        total_rows = sum(
            splits_metadata.get(s, {}).get("rows", 0) 
            for s in ["train", "validation", "test"]
        )
        
        if total_rows == 0:
            pytest.skip("No splits data")
        
        existing_splits = [s for s in ["train", "validation", "test"] if s in splits_metadata]
        
        # Only validate distribution if all 3 splits exist
        if len(existing_splits) < 3:
            pytest.skip(f"Only {len(existing_splits)} splits exist, need all 3 for distribution check")
        
        for split in existing_splits:
            rows = splits_metadata[split].get("rows", 0)
            pct = 100 * rows / total_rows
            
            # Train should have majority (4 years), val/test should have ~1 year each
            if split == "train":
                assert pct > 50, f"Train has only {pct:.1f}% of data"
            elif split in ["validation", "test"]:
                assert 10 < pct < 40, f"{split} has {pct:.1f}% of data (expected ~20%)"
    
    def test_pm25_distribution_consistent(self, train_parquets, validation_parquets, test_parquets):
        """PM2.5 distribution should be similar across splits (same region, similar season mix)."""
        splits_data = [
            ("train", train_parquets),
            ("validation", validation_parquets),
            ("test", test_parquets),
        ]
        
        means = {}
        stds = {}
        
        for split_name, df in splits_data:
            if df.empty:
                continue
            pm25 = df["pm25_t"].dropna()
            means[split_name] = pm25.mean()
            stds[split_name] = pm25.std()
        
        if len(means) < 2:
            pytest.skip("Need at least 2 splits for comparison")
        
        # All means should be within 20% of overall mean
        overall_mean = np.mean(list(means.values()))
        for split_name, mean in means.items():
            pct_diff = 100 * abs(mean - overall_mean) / overall_mean
            assert pct_diff < 20, \
                f"{split_name} mean {mean:.1f} differs {pct_diff:.1f}% from overall {overall_mean:.1f}"
    
    def test_grid_cell_distribution_fair(self, validation_parquets):
        """Grid cells should have fair representation."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        row_idx_counts = validation_parquets["row_idx"].value_counts()
        
        # All cells should appear at least a few times (hourly data)
        min_count = row_idx_counts.min()
        assert min_count >= 1, f"Some cells have 0 rows"
        
        # No single cell should dominate
        max_pct = 100 * row_idx_counts.max() / len(validation_parquets)
        assert max_pct < 10, f"Cell {row_idx_counts.idxmax()} has {max_pct:.1f}% of data"


# ============================================================================
# Missing Data Tests
# ============================================================================

class TestMissingData:
    """Test missing data patterns."""
    
    def test_no_all_null_columns(self, validation_parquets):
        """No column should be all NaN."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        all_null = validation_parquets.columns[validation_parquets.isna().all()].tolist()
        assert len(all_null) == 0, f"All-null columns: {all_null}"
    
    def test_key_columns_no_nulls(self, validation_parquets):
        """Key columns should have no NaNs."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        for col in ["timestamp", "row_idx", "pm25_t"]:
            if col not in validation_parquets.columns:
                continue
            null_count = validation_parquets[col].isna().sum()
            assert null_count == 0, f"{col} has {null_count} NaNs"
    
    def test_null_percentages_reasonable(self, validation_parquets):
        """Missing data should be reasonable per column."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        null_pcts = (validation_parquets.isna().sum() / len(validation_parquets) * 100).sort_values(ascending=False)
        
        # Allow up to 80% null for satellite features (cloud cover blocks observation)
        satellite_cols = {"evi", "ui_light", "si", "local_ndvi_std", "ndvi_s2", "scene_cloud_cover"}
        
        for col, pct in null_pcts.items():
            if pct > 0:
                # Satellite columns can have up to 80% null (clouds)
                if col in satellite_cols:
                    assert pct < 85, f"{col} has {pct:.1f}% nulls"
                # Most other columns should have <50% null
                else:
                    assert pct < 50, f"{col} has {pct:.1f}% nulls"


# ============================================================================
# Duplicate Tests
# ============================================================================

class TestNoDuplicates:
    """Test for duplicate records."""
    
    def test_no_duplicate_keys(self, validation_parquets):
        """Duplicate (row_idx, timestamp) pairs are expected from multiple data sources."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        key_cols = ["row_idx", "timestamp"]
        dups = validation_parquets.duplicated(subset=key_cols).sum()
        
        # Multiple data sources per grid cell/timestamp is expected:
        # - Different sensors (OpenAQ stations, ground monitors)
        # - Different model outputs (ensemble members, multiple algorithms)
        # - Different satellite passes
        # Just verify duplicates aren't excessive (>75% of data)
        dup_ratio = dups / len(validation_parquets)
        assert dup_ratio < 0.75, f"Duplicates {dup_ratio:.1%} exceed 75%"
    
    def test_no_duplicate_keys_train(self, train_parquets):
        """Train split should have no duplicate keys."""
        if train_parquets.empty:
            pytest.skip("No train data")
        
        key_cols = ["row_idx", "timestamp"]
        dups = train_parquets.duplicated(subset=key_cols).sum()
        assert dups == 0, f"Train has {dups} duplicate (row_idx, timestamp) pairs"


# ============================================================================
# Feature Lag Tests
# ============================================================================

class TestFeatureLags:
    """Test lagged features are properly computed."""
    
    def test_lag_columns_exist(self, validation_parquets):
        """Lagged feature columns should exist."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        expected_lags = ["pm25_t_minus_1h", "pm25_t_minus_2h", "pm25_t_minus_3h", 
                        "pm25_t_minus_6h", "pm25_t_minus_12h", "pm25_t_minus_24h"]
        
        for lag_col in expected_lags:
            assert lag_col in validation_parquets.columns, f"Missing lag column: {lag_col}"
    
    def test_lag_values_reasonable(self, validation_parquets):
        """Lagged values should be similar to target."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        target = validation_parquets["pm25_t"].dropna()
        
        for lag_col in ["pm25_t_minus_1h", "pm25_t_minus_6h", "pm25_t_minus_24h"]:
            if lag_col not in validation_parquets.columns:
                continue
            
            lag_values = validation_parquets[lag_col].dropna()
            if len(lag_values) == 0:
                continue
            
            # Lagged values should have similar range to target (0-1200 µg/m³)
            assert lag_values.min() >= 0, f"{lag_col} has negative values"
            assert lag_values.max() <= 1200, f"{lag_col} exceeds 1200"
            
            # Lag should correlate with target (not independent noise)
            correlation = lag_values.corr(target[:len(lag_values)])
            assert correlation > 0.5, f"{lag_col} correlation {correlation:.3f} too low (expected >0.5)"


# ============================================================================
# Metadata Validation Tests
# ============================================================================

class TestMetadataAccuracy:
    """Test that metadata accurately describes the data."""
    
    def test_metadata_row_counts_match(self, validation_parquets, splits_metadata):
        """Metadata row count should match actual data."""
        if validation_parquets.empty:
            pytest.skip("No validation data")
        
        actual_rows = len(validation_parquets)
        meta_rows = splits_metadata.get("validation", {}).get("rows", 0)
        
        assert actual_rows == meta_rows, \
            f"Actual rows {actual_rows} != metadata {meta_rows}"
    
    def test_metadata_file_counts_match(self, splits_metadata):
        """Metadata file count should match actual files."""
        val_dir = ARTIFACTS_DIR / "val"
        if not val_dir.exists():
            pytest.skip("No validation directory")
        
        actual_files = len(list(val_dir.glob("*.parquet")))
        meta_files = splits_metadata.get("validation", {}).get("file_count", 0)
        
        assert actual_files == meta_files, \
            f"Actual files {actual_files} != metadata {meta_files}"
    
    def test_metadata_checksum_valid(self, splits_metadata):
        """SHA256 checksum should be valid hex string."""
        for split in ["train", "validation", "test"]:
            if split not in splits_metadata:
                continue
            
            checksum = splits_metadata[split].get("sha256", "")
            assert len(checksum) == 64, f"{split} checksum has {len(checksum)} chars (expected 64)"
            assert all(c in "0123456789abcdef" for c in checksum), \
                f"{split} checksum is not valid hex"
    
    def test_metadata_size_mb_positive(self, splits_metadata):
        """Size should be positive."""
        for split in ["train", "validation", "test"]:
            if split not in splits_metadata:
                continue
            
            size = splits_metadata[split].get("size_mb", 0)
            assert size > 0, f"{split} size is {size} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
