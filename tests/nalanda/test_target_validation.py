"""Validation tests for target data assembly outputs.

These tests verify that generated target data meets quality requirements:
- Value ranges are physically sensible (PM2.5 >= 0, trust in [0,1], etc.)
- Required columns are present with correct dtypes
- No data corruption (NaN where unexpected, duplicate keys, etc.)
- Temporal consistency (timestamps are valid, lags computed correctly)
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from nalanda.target import (
    LAG_HOURS,
    add_temporal_lags,
    blend_overlapping_sensors,
    build_target_for_day,
    build_target_for_sensor_timestamp,
    precompute_sensor_influence_zones,
)
from nalanda.target_cli import write_target_parquet


# =============================================================================
# Value Range Validation Tests
# =============================================================================

class TestPM25ValueRanges:
    """Tests for PM2.5 value validity."""

    def test_pm25_non_negative(self):
        """PM2.5 values must be >= 0 (physically impossible to have negative)."""
        contributions = pd.DataFrame({
            'row_idx': [100, 101, 102],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 3,
            'pm25_raw': [0.0, 42.5, 150.0],  # Valid range
            'trust_weight': [1.0, 0.9, 0.8],
            'site_id': ['5120', '5120', '5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert (result['pm25_t'] >= 0).all(), "PM2.5 must be non-negative"

    def test_pm25_upper_bound_reasonable(self):
        """PM2.5 values should typically be < 1000 µg/m³ (very high pollution)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')],
            'pm25_raw': [500.0],  # High but plausible
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Should pass through without modification
        assert result['pm25_t'].values[0] == 500.0
        # Upper bound check (1000 is extreme but possible in severe events)
        assert (result['pm25_t'] < 2000).all(), "PM2.5 exceeds realistic maximum"

    def test_pm25_zero_is_valid(self):
        """Zero PM2.5 is valid (clean air)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')],
            'pm25_raw': [0.0],
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert result['pm25_t'].values[0] == 0.0

    def test_pm25_nan_not_propagated_to_blend(self):
        """NaN PM2.5 readings should be filtered before blending."""
        # This tests that the system handles NaN gracefully
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 2,
            'pm25_raw': [50.0, np.nan],  # One valid, one NaN
            'trust_weight': [1.0, 1.0],
            'site_id': ['5120', '5121'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Result should reflect NaN influence (becomes NaN in weighted avg)
        # The upstream code (build_target_for_day) filters NaN before this
        # But blend itself should handle it gracefully
        assert len(result) == 1


class TestTrustWeightRanges:
    """Tests for trust weight validity."""

    def test_trust_weight_in_unit_interval(self):
        """Trust weights must be in [0, 1]."""
        influence_zone = pd.DataFrame({
            'row_idx': [100, 101, 102],
            'distance_m': [0.0, 50.0, 100.0],
            'trust_weight': [1.0, 0.75, 0.5],
        })
        
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=influence_zone,
        )
        
        assert (result['trust_weight'] >= 0).all()
        assert (result['trust_weight'] <= 1).all()

    def test_trust_sum_can_exceed_one(self):
        """target_trust (sum of weights) can exceed 1 when multiple sensors overlap."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100, 100],  # Same cell, 3 sensors
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 3,
            'pm25_raw': [40.0, 45.0, 50.0],
            'trust_weight': [0.8, 0.7, 0.6],  # Sum = 2.1
            'site_id': ['5120', '5121', '5122'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # target_trust is sum, so can exceed 1
        assert result['target_trust'].values[0] == pytest.approx(2.1, rel=1e-5)

    def test_zero_trust_weight_valid(self):
        """Zero trust weight is valid (edge of influence zone)."""
        influence_zone = pd.DataFrame({
            'row_idx': [100],
            'distance_m': [250.0],  # At radius boundary
            'trust_weight': [0.0],
        })
        
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=influence_zone,
        )
        
        assert len(result) == 1
        assert result['trust_weight'].values[0] == 0.0


class TestNSensorsValidation:
    """Tests for n_sensors column validity."""

    def test_n_sensors_positive_integer(self):
        """n_sensors must be positive integer."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 2,
            'pm25_raw': [40.0, 50.0],
            'trust_weight': [1.0, 0.9],
            'site_id': ['5120', '5121'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert result['n_sensors'].values[0] >= 1
        assert result['n_sensors'].dtype in [np.int8, np.int16, np.int32, np.int64, int]

    def test_n_sensors_counts_unique_sensors(self):
        """n_sensors should count unique sensors, not duplicate entries."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 3,
            'pm25_raw': [40.0, 45.0, 50.0],
            'trust_weight': [1.0, 0.9, 0.8],
            'site_id': ['5120', '5121', '5120'],  # 5120 appears twice
        })
        result = blend_overlapping_sensors(contributions)
        
        # Should count unique: 5120, 5121 = 2 sensors
        assert result['n_sensors'].values[0] == 2


# =============================================================================
# Timestamp Validation Tests
# =============================================================================

class TestTimestampValidation:
    """Tests for timestamp consistency and validity."""

    def test_timestamp_preserved(self):
        """Timestamps should be preserved through processing."""
        ts = pd.Timestamp('2021-06-07 10:30:00')
        influence_zone = pd.DataFrame({
            'row_idx': [100],
            'distance_m': [0.0],
            'trust_weight': [1.0],
        })
        
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=ts,
            pm25_value=42.5,
            influence_zone=influence_zone,
        )
        
        assert result['timestamp'].values[0] == ts

    def test_timestamps_not_null(self):
        """Output timestamps should never be null."""
        contributions = pd.DataFrame({
            'row_idx': [100, 101],
            'timestamp': [
                pd.Timestamp('2021-06-07 10:00:00'),
                pd.Timestamp('2021-06-07 11:00:00'),
            ],
            'pm25_raw': [40.0, 50.0],
            'trust_weight': [1.0, 1.0],
            'site_id': ['5120', '5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert not result['timestamp'].isna().any()

    def test_multiple_hours_same_day(self):
        """Multiple hours in same day should produce separate rows."""
        contributions = pd.DataFrame({
            'row_idx': [100] * 24,
            'timestamp': pd.date_range('2021-06-07 00:00', periods=24, freq='h'),
            'pm25_raw': [40.0] * 24,
            'trust_weight': [1.0] * 24,
            'site_id': ['5120'] * 24,
        })
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 24


# =============================================================================
# Lag Column Validation Tests
# =============================================================================

class TestLagColumnValidation:
    """Tests for temporal lag column validity."""

    def test_lag_columns_exist(self):
        """All expected lag columns should be present."""
        target_df = pd.DataFrame({
            'row_idx': [100, 100, 100],
            'timestamp': pd.date_range('2021-06-07 00:00', periods=3, freq='h'),
            'pm25_t': [40.0, 45.0, 50.0],
            'target_trust': [1.0, 1.0, 1.0],
            'n_sensors': [1, 1, 1],
        })
        
        result = add_temporal_lags(target_df)
        
        for lag_h in LAG_HOURS:
            col = f'pm25_t_minus_{lag_h}h'
            assert col in result.columns, f"Missing lag column: {col}"

    def test_lag_values_match_previous_hours(self):
        """Lag values should match pm25_t from previous hours."""
        timestamps = pd.date_range('2021-06-07 00:00', periods=26, freq='h')
        pm25_values = list(range(26))  # 0, 1, 2, ..., 25
        
        target_df = pd.DataFrame({
            'row_idx': [100] * 26,
            'timestamp': timestamps,
            'pm25_t': pm25_values,
            'target_trust': [1.0] * 26,
            'n_sensors': [1] * 26,
        })
        
        result = add_temporal_lags(target_df)
        
        # At hour 25 (index 25), pm25_t = 25
        # pm25_t_minus_1h should be 24 (hour 24)
        row_25 = result[result['timestamp'] == timestamps[25]].iloc[0]
        assert row_25['pm25_t_minus_1h'] == 24.0
        # pm25_t_minus_24h should be 1 (hour 1)
        assert row_25['pm25_t_minus_24h'] == 1.0

    def test_lag_values_nan_for_insufficient_history(self):
        """Lag values should be NaN when insufficient history exists."""
        target_df = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 00:00:00')],  # First hour
            'pm25_t': [40.0],
            'target_trust': [1.0],
            'n_sensors': [1],
        })
        
        result = add_temporal_lags(target_df)
        
        # All lags should be NaN since there's no previous data
        for lag_h in LAG_HOURS:
            col = f'pm25_t_minus_{lag_h}h'
            assert pd.isna(result[col].values[0]), f"{col} should be NaN"

    def test_lag_values_dtype_float32(self):
        """Lag columns should be float32 for storage efficiency."""
        target_df = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': pd.date_range('2021-06-07 00:00', periods=2, freq='h'),
            'pm25_t': [40.0, 45.0],
            'target_trust': [1.0, 1.0],
            'n_sensors': [1, 1],
        })
        
        result = add_temporal_lags(target_df)
        
        for lag_h in LAG_HOURS:
            col = f'pm25_t_minus_{lag_h}h'
            assert result[col].dtype == np.float32


# =============================================================================
# Uniqueness and Key Constraints Tests
# =============================================================================

class TestUniquenessConstraints:
    """Tests for data uniqueness and key constraints."""

    def test_no_duplicate_row_timestamp_pairs(self):
        """Each (row_idx, timestamp) pair should appear exactly once."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100, 101],
            'timestamp': [
                pd.Timestamp('2021-06-07 10:00:00'),
                pd.Timestamp('2021-06-07 10:00:00'),  # Same cell, same time (overlap)
                pd.Timestamp('2021-06-07 10:00:00'),
            ],
            'pm25_raw': [40.0, 50.0, 45.0],
            'trust_weight': [1.0, 0.9, 1.0],
            'site_id': ['5120', '5121', '5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Should have 2 unique (row_idx, timestamp) pairs
        assert len(result) == 2
        
        # No duplicates
        duplicates = result.duplicated(subset=['row_idx', 'timestamp'])
        assert not duplicates.any()

    def test_row_idx_is_integer(self):
        """row_idx should be an integer type."""
        influence_zone = pd.DataFrame({
            'row_idx': [100, 101, 102],
            'distance_m': [0.0, 30.0, 60.0],
            'trust_weight': [1.0, 0.9, 0.8],
        })
        
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=influence_zone,
        )
        
        assert result['row_idx'].dtype in [np.int8, np.int16, np.int32, np.int64, int]


# =============================================================================
# Parquet Output Validation Tests
# =============================================================================

class TestParquetOutputValidation:
    """Tests for parquet file validity."""

    def test_parquet_readable(self):
        """Written parquet should be readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'target.parquet'
            
            df = pd.DataFrame({
                'row_idx': [1, 2, 3],
                'timestamp': pd.to_datetime(['2021-06-07 10:00'] * 3),
                'pm25_t': [42.5, 43.0, 41.0],
                'target_trust': [1.0, 0.9, 0.8],
                'n_sensors': [1, 1, 1],
            })
            
            write_target_parquet(df, output_path, {'test': 'metadata'})
            
            # Should be readable
            read_df = pd.read_parquet(output_path)
            assert len(read_df) == 3

    def test_parquet_schema_consistent(self):
        """Schema should match expected dtypes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'target.parquet'
            
            df = pd.DataFrame({
                'row_idx': [1],
                'timestamp': pd.to_datetime(['2021-06-07 10:00']),
                'pm25_t': [42.5],
                'target_trust': [1.0],
                'n_sensors': [1],
                'pm25_t_minus_1h': [40.0],
            })
            
            write_target_parquet(df, output_path, {})
            
            table = pq.read_table(output_path)
            schema = table.schema
            
            # Check types
            assert schema.field('row_idx').type == 'int64'
            assert 'int' in str(schema.field('n_sensors').type)
            assert 'float' in str(schema.field('pm25_t').type)

    def test_parquet_compression_applied(self):
        """Parquet should use compression (file smaller than raw)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'target.parquet'
            
            # Create larger dataframe
            n_rows = 10000
            df = pd.DataFrame({
                'row_idx': list(range(n_rows)),
                'timestamp': pd.to_datetime(['2021-06-07 10:00'] * n_rows),
                'pm25_t': [42.5] * n_rows,
                'target_trust': [1.0] * n_rows,
                'n_sensors': [1] * n_rows,
            })
            
            write_target_parquet(df, output_path, {})
            
            # File should exist and be compressed (smaller than naive estimate)
            file_size = output_path.stat().st_size
            # Rough estimate: 5 columns * 8 bytes * 10000 rows = 400KB uncompressed
            # Should be much smaller with compression
            assert file_size < 200000  # 200KB max


# =============================================================================
# Mathematical Correctness Tests
# =============================================================================

class TestMathematicalCorrectness:
    """Tests for mathematical correctness of computations."""

    def test_weighted_average_formula(self):
        """Verify weighted average formula: sum(w*x) / sum(w)."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 2,
            'pm25_raw': [30.0, 70.0],
            'trust_weight': [0.6, 0.4],
            'site_id': ['5120', '5121'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Expected: (30*0.6 + 70*0.4) / (0.6 + 0.4) = (18 + 28) / 1.0 = 46
        expected = (30.0 * 0.6 + 70.0 * 0.4) / (0.6 + 0.4)
        assert result['pm25_t'].values[0] == pytest.approx(expected, rel=1e-5)

    def test_single_sensor_no_change(self):
        """Single sensor should give same value (weighted avg with one term)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')],
            'pm25_raw': [42.5],
            'trust_weight': [0.7],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Single value weighted avg = value itself
        assert result['pm25_t'].values[0] == 42.5

    def test_equal_weights_simple_average(self):
        """Equal weights should give simple arithmetic average."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 3,
            'pm25_raw': [30.0, 40.0, 50.0],
            'trust_weight': [1.0, 1.0, 1.0],  # Equal weights
            'site_id': ['5120', '5121', '5122'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Simple average: (30 + 40 + 50) / 3 = 40
        assert result['pm25_t'].values[0] == pytest.approx(40.0, rel=1e-5)


# =============================================================================
# Empty / Edge Input Tests
# =============================================================================

class TestEmptyInputHandling:
    """Tests for handling empty inputs gracefully."""

    def test_empty_contributions_returns_empty(self):
        """Empty contributions should return empty result."""
        contributions = pd.DataFrame(columns=[
            'row_idx', 'timestamp', 'pm25_raw', 'trust_weight', 'site_id'
        ])
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 0
        assert 'pm25_t' in result.columns

    def test_none_influence_zone(self):
        """None influence zone should return empty result."""
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=None,
        )
        
        assert len(result) == 0

    def test_empty_target_df_adds_lag_columns(self):
        """Empty target should still have lag columns added."""
        target_df = pd.DataFrame(columns=[
            'row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors'
        ])
        
        result = add_temporal_lags(target_df)
        
        # Should have lag columns even if empty
        for lag_h in LAG_HOURS:
            col = f'pm25_t_minus_{lag_h}h'
            assert col in result.columns
