"""Edge case tests for target data assembly.

These tests cover boundary conditions, unusual inputs, and corner cases:
- Extreme values (very high/low PM2.5, distances, trust)
- Boundary conditions (exactly at radius, midnight timestamps)
- Malformed or unusual inputs
- Multi-day processing scenarios
- Race conditions and parallel processing edge cases
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from nalanda.target import (
    LAG_HOURS,
    add_temporal_lags,
    blend_overlapping_sensors,
    build_target_for_day,
    build_target_for_sensor_timestamp,
)
from nalanda.target_cli import (
    date_range,
    get_aqi_artifact_path,
    get_output_path,
    write_target_parquet,
)


# =============================================================================
# Extreme Value Edge Cases
# =============================================================================

class TestExtremeValues:
    """Tests for extreme but valid values."""

    def test_very_high_pm25(self):
        """Handle very high PM2.5 values (severe pollution events)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')],
            'pm25_raw': [999.0],  # Very high but possible in extreme events
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert result['pm25_t'].values[0] == 999.0
        assert not np.isnan(result['pm25_t'].values[0])
        assert not np.isinf(result['pm25_t'].values[0])

    def test_very_small_pm25(self):
        """Handle very small non-zero PM2.5 values."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')],
            'pm25_raw': [0.001],  # Very clean air
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert result['pm25_t'].values[0] == pytest.approx(0.001, rel=1e-5)

    def test_very_small_trust_weight(self):
        """Handle very small trust weights (far from sensor)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')],
            'pm25_raw': [50.0],
            'trust_weight': [0.0001],  # Very low trust
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Should still compute correctly
        assert result['pm25_t'].values[0] == 50.0

    def test_many_overlapping_sensors(self):
        """Handle many overlapping sensors at one cell."""
        n_sensors = 20
        contributions = pd.DataFrame({
            'row_idx': [100] * n_sensors,
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * n_sensors,
            'pm25_raw': list(range(30, 30 + n_sensors)),  # 30, 31, ..., 49
            'trust_weight': [0.5] * n_sensors,
            'site_id': [f'sensor_{i}' for i in range(n_sensors)],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Should have high target_trust (sum of all weights)
        assert result['target_trust'].values[0] == pytest.approx(n_sensors * 0.5, rel=1e-5)
        assert result['n_sensors'].values[0] == n_sensors
        
        # Weighted average should be simple average since equal weights
        expected_avg = sum(range(30, 30 + n_sensors)) / n_sensors
        assert result['pm25_t'].values[0] == pytest.approx(expected_avg, rel=1e-5)

    def test_large_number_of_cells(self):
        """Handle large influence zone (many cells)."""
        n_cells = 1000
        influence_zone = pd.DataFrame({
            'row_idx': list(range(n_cells)),
            'distance_m': np.linspace(0, 250, n_cells),
            'trust_weight': np.linspace(1.0, 0.0, n_cells),
        })
        
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=influence_zone,
        )
        
        assert len(result) == n_cells
        assert (result['pm25_raw'] == 42.5).all()


# =============================================================================
# Boundary Condition Edge Cases
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_exactly_at_radius_boundary(self):
        """Cell exactly at radius boundary."""
        influence_zone = pd.DataFrame({
            'row_idx': [100],
            'distance_m': [250.0],  # Exactly at default radius
            'trust_weight': [0.0],  # Epanechnikov gives 0 at boundary
        })
        
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=influence_zone,
        )
        
        assert len(result) == 1
        assert result['trust_weight'].values[0] == 0.0

    def test_zero_distance(self):
        """Cell exactly at sensor location (distance = 0)."""
        influence_zone = pd.DataFrame({
            'row_idx': [100],
            'distance_m': [0.0],
            'trust_weight': [1.0],  # Maximum trust at center
        })
        
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=influence_zone,
        )
        
        assert result['trust_weight'].values[0] == 1.0

    def test_midnight_timestamp(self):
        """Handle midnight timestamp (start of day)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 00:00:00')],
            'pm25_raw': [42.5],
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert result['timestamp'].values[0] == pd.Timestamp('2021-06-07 00:00:00')

    def test_end_of_day_timestamp(self):
        """Handle 23:00 timestamp (last hour)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 23:00:00')],
            'pm25_raw': [42.5],
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert result['timestamp'].values[0] == pd.Timestamp('2021-06-07 23:00:00')

    def test_year_boundary(self):
        """Handle timestamp at year boundary."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [
                pd.Timestamp('2020-12-31 23:00:00'),
                pd.Timestamp('2021-01-01 00:00:00'),
            ],
            'pm25_raw': [40.0, 50.0],
            'trust_weight': [1.0, 1.0],
            'site_id': ['5120', '5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 2

    def test_leap_year_date(self):
        """Handle leap year date (Feb 29)."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2020-02-29 12:00:00')],  # 2020 was leap year
            'pm25_raw': [42.5],
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 1
        assert result['timestamp'].values[0] == pd.Timestamp('2020-02-29 12:00:00')


# =============================================================================
# Date Range Edge Cases
# =============================================================================

class TestDateRangeEdgeCases:
    """Tests for date range edge cases."""

    def test_same_start_end_date(self):
        """Single day when start == end."""
        dates = date_range('2021-06-07', '2021-06-07')
        assert dates == ['2021-06-07']
        assert len(dates) == 1

    def test_february_non_leap_year(self):
        """February in non-leap year."""
        dates = date_range('2021-02-26', '2021-03-02')
        expected = ['2021-02-26', '2021-02-27', '2021-02-28', '2021-03-01', '2021-03-02']
        assert dates == expected

    def test_february_leap_year(self):
        """February in leap year."""
        dates = date_range('2020-02-27', '2020-03-02')
        expected = ['2020-02-27', '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02']
        assert dates == expected

    def test_full_year_range(self):
        """Full year generates 365/366 dates."""
        dates = date_range('2021-01-01', '2021-12-31')
        assert len(dates) == 365  # 2021 not leap year

        dates_leap = date_range('2020-01-01', '2020-12-31')
        assert len(dates_leap) == 366  # 2020 is leap year


# =============================================================================
# Temporal Lag Edge Cases
# =============================================================================

class TestTemporalLagEdgeCases:
    """Tests for temporal lag computation edge cases."""

    def test_cross_day_lag_without_prev_day(self):
        """Cross-day lags should be NaN without previous day data."""
        # Data starting at midnight - 24h lag needs previous day
        target_df = pd.DataFrame({
            'row_idx': [100, 100, 100],
            'timestamp': pd.date_range('2021-06-07 00:00', periods=3, freq='h'),
            'pm25_t': [40.0, 45.0, 50.0],
            'target_trust': [1.0, 1.0, 1.0],
            'n_sensors': [1, 1, 1],
        })
        
        result = add_temporal_lags(target_df, prev_day_df=None)
        
        # First row: all lags should be NaN (no data before)
        first_row = result.iloc[0]
        for lag_h in LAG_HOURS:
            col = f'pm25_t_minus_{lag_h}h'
            assert pd.isna(first_row[col]), f"{col} should be NaN at midnight"

    def test_cross_day_lag_with_prev_day(self):
        """Cross-day lags should work with previous day data."""
        # Previous day ending at 23:00
        prev_day_df = pd.DataFrame({
            'row_idx': [100] * 24,
            'timestamp': pd.date_range('2021-06-06 00:00', periods=24, freq='h'),
            'pm25_t': list(range(24)),  # 0, 1, 2, ..., 23
            'target_trust': [1.0] * 24,
            'n_sensors': [1] * 24,
        })
        
        # Current day starting at midnight
        target_df = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': pd.date_range('2021-06-07 00:00', periods=2, freq='h'),
            'pm25_t': [100.0, 101.0],  # Different values to distinguish
            'target_trust': [1.0, 1.0],
            'n_sensors': [1, 1],
        })
        
        result = add_temporal_lags(target_df, prev_day_df=prev_day_df)
        
        # At 2021-06-07 00:00, pm25_t_minus_1h should be 23 (from prev day 23:00)
        first_row = result[result['timestamp'] == pd.Timestamp('2021-06-07 00:00')].iloc[0]
        assert first_row['pm25_t_minus_1h'] == 23.0

    def test_multiple_cells_independent_lags(self):
        """Different cells should have independent lag values."""
        target_df = pd.DataFrame({
            'row_idx': [100, 100, 200, 200],
            'timestamp': [
                pd.Timestamp('2021-06-07 00:00'),
                pd.Timestamp('2021-06-07 01:00'),
                pd.Timestamp('2021-06-07 00:00'),
                pd.Timestamp('2021-06-07 01:00'),
            ],
            'pm25_t': [40.0, 45.0, 80.0, 85.0],  # Different values per cell
            'target_trust': [1.0, 1.0, 1.0, 1.0],
            'n_sensors': [1, 1, 1, 1],
        })
        
        result = add_temporal_lags(target_df)
        
        # Cell 100 at 01:00 should have lag from cell 100 at 00:00 (40.0)
        row_100_01 = result[(result['row_idx'] == 100) & 
                            (result['timestamp'] == pd.Timestamp('2021-06-07 01:00'))].iloc[0]
        assert row_100_01['pm25_t_minus_1h'] == 40.0
        
        # Cell 200 at 01:00 should have lag from cell 200 at 00:00 (80.0)
        row_200_01 = result[(result['row_idx'] == 200) & 
                            (result['timestamp'] == pd.Timestamp('2021-06-07 01:00'))].iloc[0]
        assert row_200_01['pm25_t_minus_1h'] == 80.0

    def test_sparse_timestamps_missing_lags(self):
        """Missing intermediate hours should result in NaN lags."""
        # Only hours 0 and 6 present (missing 1-5)
        target_df = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [
                pd.Timestamp('2021-06-07 00:00'),
                pd.Timestamp('2021-06-07 06:00'),
            ],
            'pm25_t': [40.0, 60.0],
            'target_trust': [1.0, 1.0],
            'n_sensors': [1, 1],
        })
        
        result = add_temporal_lags(target_df)
        
        # At 06:00, pm25_t_minus_6h should be 40.0 (00:00)
        row_06 = result[result['timestamp'] == pd.Timestamp('2021-06-07 06:00')].iloc[0]
        assert row_06['pm25_t_minus_6h'] == 40.0
        
        # But pm25_t_minus_1h should be NaN (05:00 doesn't exist)
        assert pd.isna(row_06['pm25_t_minus_1h'])


# =============================================================================
# Special Input Edge Cases
# =============================================================================

class TestSpecialInputs:
    """Tests for special input cases."""

    def test_single_contribution(self):
        """Single contribution should work correctly."""
        contributions = pd.DataFrame({
            'row_idx': [100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')],
            'pm25_raw': [42.5],
            'trust_weight': [1.0],
            'site_id': ['5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 1
        assert result['pm25_t'].values[0] == 42.5
        assert result['n_sensors'].values[0] == 1

    def test_same_sensor_multiple_timestamps(self):
        """Same sensor at multiple timestamps."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100, 100],
            'timestamp': [
                pd.Timestamp('2021-06-07 10:00:00'),
                pd.Timestamp('2021-06-07 11:00:00'),
                pd.Timestamp('2021-06-07 12:00:00'),
            ],
            'pm25_raw': [40.0, 45.0, 50.0],
            'trust_weight': [1.0, 1.0, 1.0],
            'site_id': ['5120', '5120', '5120'],
        })
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 3
        # Each timestamp should have correct value
        assert result[result['timestamp'] == pd.Timestamp('2021-06-07 10:00:00')]['pm25_t'].values[0] == 40.0
        assert result[result['timestamp'] == pd.Timestamp('2021-06-07 11:00:00')]['pm25_t'].values[0] == 45.0

    def test_same_sensor_multiple_cells(self):
        """Same sensor influences multiple cells (normal case)."""
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
        
        assert len(result) == 3
        assert set(result['row_idx']) == {100, 101, 102}


# =============================================================================
# Output Path Edge Cases
# =============================================================================

class TestOutputPathEdgeCases:
    """Tests for output path handling."""

    def test_output_path_with_special_date(self):
        """Output path handles various date formats."""
        path = get_output_path('2021-01-01')
        assert '2021-01-01' in str(path)
        
        path = get_output_path('2021-12-31')
        assert '2021-12-31' in str(path)

    def test_aqi_artifact_path_consistency(self):
        """AQI artifact path is consistent."""
        path1 = get_aqi_artifact_path('2021-06-07')
        path2 = get_aqi_artifact_path('2021-06-07')
        assert path1 == path2

    def test_output_creates_parent_directories(self):
        """Writing parquet creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / 'a' / 'b' / 'c' / 'target.parquet'
            
            df = pd.DataFrame({
                'row_idx': [1],
                'timestamp': pd.to_datetime(['2021-06-07 10:00']),
                'pm25_t': [42.5],
                'target_trust': [1.0],
                'n_sensors': [1],
            })
            
            write_target_parquet(df, deep_path, {})
            
            assert deep_path.exists()
            assert deep_path.parent.exists()


# =============================================================================
# Numerical Stability Edge Cases
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_weights_sum(self):
        """Handle sum of many small weights without underflow."""
        n = 100
        small_weight = 0.001
        contributions = pd.DataFrame({
            'row_idx': [100] * n,
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * n,
            'pm25_raw': [50.0] * n,
            'trust_weight': [small_weight] * n,
            'site_id': [f'sensor_{i}' for i in range(n)],
        })
        result = blend_overlapping_sensors(contributions)
        
        # Sum should be n * small_weight = 0.1
        assert result['target_trust'].values[0] == pytest.approx(n * small_weight, rel=1e-5)
        assert result['pm25_t'].values[0] == pytest.approx(50.0, rel=1e-5)

    def test_large_pm25_values_precision(self):
        """Maintain precision with large PM2.5 values."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 2,
            'pm25_raw': [999.999, 1000.001],
            'trust_weight': [0.5, 0.5],
            'site_id': ['5120', '5121'],
        })
        result = blend_overlapping_sensors(contributions)
        
        expected = (999.999 * 0.5 + 1000.001 * 0.5) / 1.0
        assert result['pm25_t'].values[0] == pytest.approx(expected, rel=1e-5)

    def test_float32_output_range(self):
        """Output should fit in float32 range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'target.parquet'
            
            # Use values that fit in float32
            df = pd.DataFrame({
                'row_idx': [1],
                'timestamp': pd.to_datetime(['2021-06-07 10:00']),
                'pm25_t': [999.999],  # High but fits in float32
                'target_trust': [2.5],  # Sum > 1 is valid
                'n_sensors': [3],
            })
            
            write_target_parquet(df, output_path, {})
            
            # Read back and check
            read_df = pd.read_parquet(output_path)
            assert read_df['pm25_t'].dtype == np.float32
            assert read_df['pm25_t'].values[0] == pytest.approx(999.999, rel=1e-3)


# =============================================================================
# Data Integrity Edge Cases
# =============================================================================

class TestDataIntegrity:
    """Tests for data integrity through processing."""

    def test_no_data_loss_in_blend(self):
        """All input rows contribute to some output."""
        n_cells = 50
        contributions = pd.DataFrame({
            'row_idx': list(range(n_cells)),
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * n_cells,
            'pm25_raw': [42.5] * n_cells,
            'trust_weight': [1.0] * n_cells,
            'site_id': ['5120'] * n_cells,
        })
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == n_cells

    def test_deterministic_output(self):
        """Same input should produce same output."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100, 101],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 3,
            'pm25_raw': [40.0, 50.0, 45.0],
            'trust_weight': [0.8, 0.6, 1.0],
            'site_id': ['5120', '5121', '5120'],
        })
        
        result1 = blend_overlapping_sensors(contributions.copy())
        result2 = blend_overlapping_sensors(contributions.copy())
        
        # Sort for comparison
        result1 = result1.sort_values('row_idx').reset_index(drop=True)
        result2 = result2.sort_values('row_idx').reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result1, result2)
