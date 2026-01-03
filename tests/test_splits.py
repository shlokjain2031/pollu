"""Unit tests for temporal split module.

Tests enforce:
- No overlap between train/val/test
- 48h gaps between splits
- Correct timestamp assignment
- Seasonal coverage
- Leakage detection
"""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime

from nalanda.splits import (
    get_split,
    is_in_split,
    get_split_mask,
    validate_splits,
    assert_no_leakage,
    SPLIT_RANGES,
    GAP_1,
    GAP_2,
)


class TestSplitBoundaries:
    """Test exact split boundary definitions."""
    
    def test_train_boundaries(self):
        """Train split is 2019-01-01 to 2022-12-29."""
        start, end = SPLIT_RANGES["train"]
        assert start == pd.Timestamp("2019-01-01 00:00:00")
        assert end == pd.Timestamp("2022-12-29 23:59:59")
    
    def test_validation_boundaries(self):
        """Validation split is 2023-01-02 to 2023-12-30."""
        start, end = SPLIT_RANGES["validation"]
        assert start == pd.Timestamp("2023-01-02 00:00:00")
        assert end == pd.Timestamp("2023-12-30 23:59:59")
    
    def test_test_boundaries(self):
        """Test split is 2024-01-03 to 2024-12-31."""
        start, end = SPLIT_RANGES["test"]
        assert start == pd.Timestamp("2024-01-03 00:00:00")
        assert end == pd.Timestamp("2024-12-31 23:59:59")
    
    def test_gap_1_is_48_hours(self):
        """First gap (train→val) is exactly 48 hours."""
        duration = (GAP_1[1] - GAP_1[0]).total_seconds() / 3600
        # Note: gap includes both endpoints, so it's actually ~48h worth of timestamps
        assert duration >= 47.9  # Allow small floating point tolerance
    
    def test_gap_2_is_48_hours(self):
        """Second gap (val→test) is exactly 48 hours."""
        duration = (GAP_2[1] - GAP_2[0]).total_seconds() / 3600
        assert duration >= 47.9


class TestGetSplit:
    """Test get_split function for correct assignment."""
    
    def test_train_timestamps(self):
        """Timestamps in train range return 'train'."""
        assert get_split(pd.Timestamp("2019-01-01")) == "train"
        assert get_split(pd.Timestamp("2020-06-15")) == "train"
        assert get_split(pd.Timestamp("2022-12-29")) == "train"
    
    def test_validation_timestamps(self):
        """Timestamps in validation range return 'validation'."""
        assert get_split(pd.Timestamp("2023-01-02")) == "validation"
        assert get_split(pd.Timestamp("2023-07-15")) == "validation"
        assert get_split(pd.Timestamp("2023-12-30")) == "validation"
    
    def test_test_timestamps(self):
        """Timestamps in test range return 'test'."""
        assert get_split(pd.Timestamp("2024-01-03")) == "test"
        assert get_split(pd.Timestamp("2024-08-20")) == "test"
        assert get_split(pd.Timestamp("2024-12-31")) == "test"
    
    def test_gap_timestamps(self):
        """Timestamps in gap ranges return 'gap'."""
        assert get_split(pd.Timestamp("2022-12-30")) == "gap"
        assert get_split(pd.Timestamp("2022-12-31")) == "gap"
        assert get_split(pd.Timestamp("2023-12-31")) == "gap"
        assert get_split(pd.Timestamp("2024-01-01")) == "gap"
    
    def test_excluded_timestamps(self):
        """2017-2018 timestamps return 'excluded'."""
        assert get_split(pd.Timestamp("2017-06-15")) == "excluded"
        assert get_split(pd.Timestamp("2018-12-31")) == "excluded"
    
    def test_2025_raises_error(self):
        """2025 timestamps raise ValueError (partial year)."""
        with pytest.raises(ValueError, match="2025 or later"):
            get_split(pd.Timestamp("2025-06-15"))
        
        with pytest.raises(ValueError, match="2025 or later"):
            get_split(pd.Timestamp("2026-01-01"))
    
    def test_string_input(self):
        """get_split accepts string timestamps."""
        assert get_split("2020-06-15") == "train"
        assert get_split("2023-06-15") == "validation"
        assert get_split("2024-06-15") == "test"


class TestIsInSplit:
    """Test is_in_split helper function."""
    
    def test_correct_split(self):
        """Returns True when timestamp is in specified split."""
        assert is_in_split(pd.Timestamp("2020-06-15"), "train")
        assert is_in_split(pd.Timestamp("2023-06-15"), "validation")
        assert is_in_split(pd.Timestamp("2024-06-15"), "test")
    
    def test_incorrect_split(self):
        """Returns False when timestamp is not in specified split."""
        assert not is_in_split(pd.Timestamp("2020-06-15"), "validation")
        assert not is_in_split(pd.Timestamp("2023-06-15"), "test")
        assert not is_in_split(pd.Timestamp("2024-06-15"), "train")
    
    def test_gap_split(self):
        """Gap timestamps identified correctly."""
        assert is_in_split(pd.Timestamp("2022-12-30"), "gap")
        assert not is_in_split(pd.Timestamp("2020-06-15"), "gap")


class TestNoOverlap:
    """Test that splits have no temporal overlap."""
    
    def test_train_and_validation_no_overlap(self):
        """Train and validation ranges don't overlap."""
        train_start, train_end = SPLIT_RANGES["train"]
        val_start, val_end = SPLIT_RANGES["validation"]
        
        assert train_end < val_start
    
    def test_validation_and_test_no_overlap(self):
        """Validation and test ranges don't overlap."""
        val_start, val_end = SPLIT_RANGES["validation"]
        test_start, test_end = SPLIT_RANGES["test"]
        
        assert val_end < test_start
    
    def test_train_and_test_no_overlap(self):
        """Train and test ranges don't overlap."""
        train_start, train_end = SPLIT_RANGES["train"]
        test_start, test_end = SPLIT_RANGES["test"]
        
        assert train_end < test_start
    
    def test_every_timestamp_unique_split(self):
        """Every timestamp belongs to exactly one split."""
        # Sample timestamps across the range
        dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
        
        for ts in dates:
            splits_assigned = []
            for split_name in ["train", "validation", "test", "gap"]:
                if is_in_split(ts, split_name):
                    splits_assigned.append(split_name)
            
            # Each timestamp should be in exactly one split
            assert len(splits_assigned) == 1, f"{ts} assigned to {splits_assigned}"


class TestGapEnforcement:
    """Test that 48h gaps are properly enforced."""
    
    def test_gap_1_separates_train_and_val(self):
        """First gap prevents train/val adjacency."""
        train_end = SPLIT_RANGES["train"][1]
        val_start = SPLIT_RANGES["validation"][0]
        
        gap_hours = (val_start - train_end).total_seconds() / 3600
        assert gap_hours >= 48
    
    def test_gap_2_separates_val_and_test(self):
        """Second gap prevents val/test adjacency."""
        val_end = SPLIT_RANGES["validation"][1]
        test_start = SPLIT_RANGES["test"][0]
        
        gap_hours = (test_start - val_end).total_seconds() / 3600
        assert gap_hours >= 48
    
    def test_gap_1_contents(self):
        """Gap 1 contains exactly 2022-12-30 and 2022-12-31."""
        assert GAP_1[0] == pd.Timestamp("2022-12-30 00:00:00")
        assert GAP_1[1] == pd.Timestamp("2022-12-31 23:59:59")
    
    def test_gap_2_contents(self):
        """Gap 2 contains 2023-12-31, 2024-01-01, and 2024-01-02."""
        assert GAP_2[0] == pd.Timestamp("2023-12-31 00:00:00")
        assert GAP_2[1] == pd.Timestamp("2024-01-02 23:59:59")


class TestGetSplitMask:
    """Test DataFrame masking by split."""
    
    def test_train_mask(self):
        """Train mask selects only train timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2024-12-31", freq="30D")
        })
        
        mask = get_split_mask(df, "train", "timestamp")
        train_df = df[mask]
        
        # All selected timestamps should be in train split
        for ts in train_df["timestamp"]:
            assert get_split(ts) == "train"
    
    def test_validation_mask(self):
        """Validation mask selects only validation timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2024-12-31", freq="30D")
        })
        
        mask = get_split_mask(df, "validation", "timestamp")
        val_df = df[mask]
        
        for ts in val_df["timestamp"]:
            assert get_split(ts) == "validation"
    
    def test_test_mask(self):
        """Test mask selects only test timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2024-12-31", freq="30D")
        })
        
        mask = get_split_mask(df, "test", "timestamp")
        test_df = df[mask]
        
        for ts in test_df["timestamp"]:
            assert get_split(ts) == "test"
    
    def test_masks_are_exclusive(self):
        """Train/val/test masks don't overlap."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2024-12-31", freq="D")
        })
        
        train_mask = get_split_mask(df, "train", "timestamp")
        val_mask = get_split_mask(df, "validation", "timestamp")
        test_mask = get_split_mask(df, "test", "timestamp")
        
        # No row should be in multiple splits
        assert not (train_mask & val_mask).any()
        assert not (train_mask & test_mask).any()
        assert not (val_mask & test_mask).any()


class TestValidateSplits:
    """Test validate_splits function."""
    
    def test_valid_dataframe(self):
        """DataFrame with all valid splits passes validation."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2024-12-31", freq="D"),
            "value": range(len(pd.date_range("2019-01-01", "2024-12-31", freq="D")))
        })
        
        result = validate_splits(df, timestamp_col="timestamp", strict=False)
        assert result["valid"]
        assert result["total_rows"] == len(df)
        assert len(result["issues"]) == 0
    
    def test_dataframe_with_2025_data_fails(self):
        """DataFrame with 2025 timestamps fails validation."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2025-06-30", freq="30D"),
            "value": range(len(pd.date_range("2019-01-01", "2025-06-30", freq="30D")))
        })
        
        result = validate_splits(df, timestamp_col="timestamp", strict=False)
        assert not result["valid"]
        assert result["unassigned_rows"] > 0
    
    def test_strict_mode_raises_on_issues(self):
        """Strict mode raises AssertionError on validation failures."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2025-06-30", freq="30D"),
            "value": range(len(pd.date_range("2019-01-01", "2025-06-30", freq="30D")))
        })
        
        with pytest.raises(AssertionError, match="Split validation failed"):
            validate_splits(df, timestamp_col="timestamp", strict=True)
    
    def test_split_counts(self):
        """Validation returns correct split counts."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2024-12-31", freq="D"),
            "value": range(len(pd.date_range("2019-01-01", "2024-12-31", freq="D")))
        })
        
        result = validate_splits(df, timestamp_col="timestamp", strict=False)
        
        # Verify counts
        assert result["split_counts"]["train"] > 0
        assert result["split_counts"]["validation"] > 0
        assert result["split_counts"]["test"] > 0
        
        # Total should match
        total = sum(result["split_counts"].values())
        assert total == result["total_rows"]


class TestAssertNoLeakage:
    """Test assert_no_leakage function."""
    
    def test_valid_splits_pass(self):
        """Properly separated train/val/test pass leakage check."""
        train_df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2022-12-29", freq="D")
        })
        val_df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-02", "2023-12-30", freq="D")
        })
        test_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-03", "2024-12-31", freq="D")
        })
        
        # Should not raise
        assert_no_leakage(train_df, val_df, test_df, timestamp_col="timestamp")
    
    def test_overlapping_timestamps_fail(self):
        """Overlapping timestamps between splits fail."""
        train_df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2023-01-05", freq="D")  # Overlaps with val
        })
        val_df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-02", "2023-12-30", freq="D")
        })
        test_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-03", "2024-12-31", freq="D")
        })
        
        with pytest.raises(AssertionError, match="must be before"):
            assert_no_leakage(train_df, val_df, test_df, timestamp_col="timestamp")
    
    def test_insufficient_gap_fails(self):
        """Less than 48h gap between splits fails."""
        # Create train that ends on 2023-01-01 (only 24h before val start)
        train_df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2023-01-01", freq="D")
        })
        val_df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-02", "2023-12-30", freq="D")
        })
        test_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-03", "2024-12-31", freq="D")
        })
        
        with pytest.raises(AssertionError, match="must be >= 48h"):
            assert_no_leakage(train_df, val_df, test_df, timestamp_col="timestamp")
    
    def test_wrong_temporal_order_fails(self):
        """Splits in wrong temporal order fail."""
        # Swap train and val
        train_df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-02", "2023-12-30", freq="D")
        })
        val_df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2022-12-29", freq="D")
        })
        test_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-03", "2024-12-31", freq="D")
        })
        
        with pytest.raises(AssertionError, match="must be before"):
            assert_no_leakage(train_df, val_df, test_df, timestamp_col="timestamp")


class TestSeasonalCoverage:
    """Test that splits have proper seasonal coverage."""
    
    def test_train_has_4_years(self):
        """Train split spans approximately 4 years."""
        start, end = SPLIT_RANGES["train"]
        years = (end - start).days / 365.25
        assert 3.9 <= years <= 4.1  # Allow small tolerance
    
    def test_validation_has_1_year(self):
        """Validation split spans approximately 1 year."""
        start, end = SPLIT_RANGES["validation"]
        years = (end - start).days / 365.25
        assert 0.9 <= years <= 1.1
    
    def test_test_has_1_year(self):
        """Test split spans approximately 1 year."""
        start, end = SPLIT_RANGES["test"]
        years = (end - start).days / 365.25
        assert 0.9 <= years <= 1.1
    
    def test_all_months_in_train(self):
        """Train split contains all 12 months (multiple times)."""
        dates = pd.date_range(
            SPLIT_RANGES["train"][0],
            SPLIT_RANGES["train"][1],
            freq="D"
        )
        months = set(dates.month)
        assert len(months) == 12
    
    def test_all_months_in_validation(self):
        """Validation split contains all 12 months."""
        dates = pd.date_range(
            SPLIT_RANGES["validation"][0],
            SPLIT_RANGES["validation"][1],
            freq="D"
        )
        months = set(dates.month)
        assert len(months) == 12
    
    def test_all_months_in_test(self):
        """Test split contains all 12 months."""
        dates = pd.date_range(
            SPLIT_RANGES["test"][0],
            SPLIT_RANGES["test"][1],
            freq="D"
        )
        months = set(dates.month)
        assert len(months) == 12


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_boundary_timestamps_exclusive(self):
        """Timestamps at exact boundaries are correctly assigned."""
        # Last timestamp of train should be in train
        assert get_split(pd.Timestamp("2022-12-29 23:59:59")) == "train"
        
        # First timestamp of gap should be in gap
        assert get_split(pd.Timestamp("2022-12-30 00:00:00")) == "gap"
        
        # Last timestamp of gap should be in gap
        assert get_split(pd.Timestamp("2022-12-31 23:59:59")) == "gap"
        
        # First timestamp of val should be in val
        assert get_split(pd.Timestamp("2023-01-02 00:00:00")) == "validation"
    
    def test_single_timestamp_dataframe(self):
        """DataFrame with single timestamp validates correctly."""
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2020-06-15")]})
        result = validate_splits(df, timestamp_col="timestamp", strict=False)
        
        assert result["valid"]
        assert result["split_counts"]["train"] == 1
    
    def test_empty_dataframe(self):
        """Empty DataFrame validates without errors."""
        df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]")})
        result = validate_splits(df, timestamp_col="timestamp", strict=False)
        
        assert result["total_rows"] == 0
        assert all(count == 0 for count in result["split_counts"].values())
    
    def test_missing_timestamp_column_raises(self):
        """Missing timestamp column raises ValueError."""
        df = pd.DataFrame({"date": [pd.Timestamp("2020-06-15")]})
        
        with pytest.raises(ValueError, match="not found"):
            validate_splits(df, timestamp_col="timestamp", strict=False)


class TestMaterializeSplits:
    """Test split materialization functionality."""
    
    def test_materialize_splits_creates_directories(self, tmp_path):
        """Materialize splits creates train/val/test directories."""
        from nalanda.training import materialize_splits
        
        # Create sample training data
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create data spanning all splits
        dates = pd.date_range("2019-01-01", "2024-12-31", freq="30D")
        df = pd.DataFrame({
            "timestamp": dates,
            "row_idx": range(len(dates)),
            "pm25_t": [50.0] * len(dates),
        })
        
        # Save as date-partitioned files (simulate real structure)
        for date in ["2020-01-01", "2023-06-01", "2024-06-01"]:
            date_dir = input_dir / date
            date_dir.mkdir()
            subset = df[df["timestamp"].dt.date == pd.Timestamp(date).date()]
            if len(subset) > 0:
                subset.to_parquet(date_dir / "data.parquet")
        
        # Also save full dataset as single file for testing
        date_dir = input_dir / "2020-06-01"
        date_dir.mkdir(exist_ok=True)
        df.to_parquet(date_dir / "full.parquet")
        
        output_dir = tmp_path / "output"
        
        # Materialize splits
        splits_meta = materialize_splits(
            input_dir=input_dir,
            output_base_dir=output_dir,
        )
        
        # Check directories exist
        assert (output_dir / "train").exists()
        assert (output_dir / "val").exists()
        assert (output_dir / "test").exists()
        
        # Check files exist
        assert (output_dir / "train" / "train.parquet").exists()
        assert (output_dir / "val" / "validation.parquet").exists()
        assert (output_dir / "test" / "test.parquet").exists()
        
        # Check metadata
        assert "train" in splits_meta
        assert "validation" in splits_meta
        assert "test" in splits_meta
        
        assert splits_meta["train"]["rows"] > 0
        assert splits_meta["validation"]["rows"] > 0
        assert splits_meta["test"]["rows"] > 0
    
    def test_materialize_splits_no_overlap(self, tmp_path):
        """Materialized splits have no timestamp overlap."""
        from nalanda.training import materialize_splits
        
        # Create sample data
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        date_dir = input_dir / "2020-01-01"
        date_dir.mkdir()
        
        dates = pd.date_range("2019-01-01", "2024-12-31", freq="7D")
        df = pd.DataFrame({
            "timestamp": dates,
            "row_idx": range(len(dates)),
            "pm25_t": [50.0] * len(dates),
        })
        df.to_parquet(date_dir / "data.parquet")
        
        output_dir = tmp_path / "output"
        
        # Materialize
        materialize_splits(input_dir=input_dir, output_base_dir=output_dir)
        
        # Load splits
        train_df = pd.read_parquet(output_dir / "train" / "train.parquet")
        val_df = pd.read_parquet(output_dir / "val" / "validation.parquet")
        test_df = pd.read_parquet(output_dir / "test" / "test.parquet")
        
        # Check no overlap
        from nalanda.splits import assert_no_leakage
        assert_no_leakage(train_df, val_df, test_df, timestamp_col="timestamp")
    
    def test_materialize_splits_metadata_saved(self, tmp_path):
        """Materialize splits saves metadata JSON."""
        from nalanda.training import materialize_splits
        import json
        
        # Create sample data
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        date_dir = input_dir / "2020-01-01"
        date_dir.mkdir()
        
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", "2024-12-31", freq="30D"),
            "row_idx": range(len(pd.date_range("2019-01-01", "2024-12-31", freq="30D"))),
            "pm25_t": [50.0] * len(pd.date_range("2019-01-01", "2024-12-31", freq="30D")),
        })
        df.to_parquet(date_dir / "data.parquet")
        
        output_dir = tmp_path / "output"
        
        # Materialize
        materialize_splits(input_dir=input_dir, output_base_dir=output_dir)
        
        # Check metadata file exists
        metadata_path = output_dir / "splits_metadata.json"
        assert metadata_path.exists()
        
        # Load and validate metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert "train" in metadata
        assert "validation" in metadata
        assert "test" in metadata
        
        # Check required fields
        for split in ["train", "validation", "test"]:
            assert "rows" in metadata[split]
            assert "columns" in metadata[split]
            assert "min_timestamp" in metadata[split]
            assert "max_timestamp" in metadata[split]
            assert "sha256" in metadata[split]
            assert "size_mb" in metadata[split]
            assert len(metadata[split]["sha256"]) == 64  # SHA256 hex length

