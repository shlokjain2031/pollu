#!/usr/bin/env python3
"""
Unit tests for nalanda/features_meteo.py

Tests meteorological feature engineering including:
- Wind decomposition (u/v components)
- Vapor pressure deficit (VPD) computation
- Temperature-dewpoint difference
- Temporal lag correctness
- Precipitation rolling sums
- Cross-day lag handling
- Output schema validation

Usage:
    pytest tests/test_features_meteo.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalanda.features_meteo import (
    _compute_lag_matrix,
    _compute_rolling_sum,
    compute_meteo_features,
    get_meteo_feature_columns,
    load_hourly_meteo,
    load_target_grid_ids,
    LAG_HOURS,
    LAG_VARIABLES,
)


# Test constants
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "v1" / "openmeteo" / "mumbai"
TARGET_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"
EXPECTED_HOURS_PER_DAY = 24
# Use small subset for fast tests (100 random grids)
TEST_GRID_SAMPLE_SIZE = 100


def get_available_dates() -> list[str]:
    """Find available hourly meteo artifact dates."""
    if not ARTIFACTS_DIR.exists():
        return []
    files = sorted(ARTIFACTS_DIR.glob("*_hourly.parquet"))
    return [f.stem.replace("_hourly", "") for f in files]


def get_sample_grid_ids(date: str, n: int = TEST_GRID_SAMPLE_SIZE) -> set[str]:
    """Get a small sample of grid_ids for fast testing.
    
    First tries to use target grid_ids if available, otherwise samples from
    the full grid.
    """
    import random
    
    # Try to load from target
    target_path = TARGET_DIR / date / "target.parquet"
    if target_path.exists():
        try:
            grid_ids = load_target_grid_ids(date)
            if len(grid_ids) <= n:
                return grid_ids
            return set(random.sample(list(grid_ids), n))
        except Exception:
            pass
    
    # Fall back to sampling from full data
    df = load_hourly_meteo(date)
    all_grids = list(df['grid_id'].unique())
    return set(random.sample(all_grids, min(n, len(all_grids))))


@pytest.fixture
def available_dates() -> list[str]:
    """Fixture providing available dates."""
    dates = get_available_dates()
    if not dates:
        pytest.skip("No hourly OpenMeteo artifacts found")
    return dates


@pytest.fixture
def sample_date(available_dates: list[str]) -> str:
    """Fixture providing a single date for testing."""
    # Prefer a date that has a previous day available for cross-day lag tests
    for i, date in enumerate(available_dates):
        if i > 0:
            return date
    return available_dates[0]


@pytest.fixture
def sample_grid_ids(sample_date: str) -> set[str]:
    """Fixture providing a small sample of grid_ids for fast testing."""
    return get_sample_grid_ids(sample_date)


@pytest.fixture
def two_consecutive_dates(available_dates: list[str]) -> tuple[str, str]:
    """Fixture providing two consecutive dates for cross-day tests."""
    from datetime import datetime as dt, timedelta
    
    for date in available_dates[1:]:
        prev_date = (dt.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        if prev_date in available_dates:
            return prev_date, date
    
    pytest.skip("No consecutive dates found for cross-day lag tests")


class TestLagMatrix:
    """Test the _compute_lag_matrix helper function."""
    
    def test_lag_shifts_values_correctly(self):
        """Lag should shift values to the right."""
        # Create simple test matrix: 3 grids, 5 hours
        # Values are just hour indices for easy verification
        matrix = np.array([
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ], dtype=np.float64)
        
        lagged = _compute_lag_matrix(matrix, lag_hours=1)
        
        # First column should be NaN (no previous data)
        assert np.all(np.isnan(lagged[:, 0]))
        
        # Column 1 should have original column 0 values
        np.testing.assert_array_equal(lagged[:, 1], [0, 10, 20])
        
        # Column 4 should have original column 3 values
        np.testing.assert_array_equal(lagged[:, 4], [3, 13, 23])
    
    def test_lag_3_hours(self):
        """3-hour lag should shift by 3 positions."""
        matrix = np.arange(24).reshape(1, 24).astype(np.float64)
        
        lagged = _compute_lag_matrix(matrix, lag_hours=3)
        
        # First 3 columns should be NaN
        assert np.all(np.isnan(lagged[:, :3]))
        
        # Column 3 should have original column 0 value
        assert lagged[0, 3] == 0
        
        # Column 10 should have original column 7 value
        assert lagged[0, 10] == 7
    
    def test_lag_preserves_shape(self):
        """Output shape should match input shape."""
        matrix = np.random.rand(100, 24)
        
        for lag in [1, 3, 6, 12, 24]:
            lagged = _compute_lag_matrix(matrix, lag)
            assert lagged.shape == matrix.shape


class TestRollingSum:
    """Test the _compute_rolling_sum helper function."""
    
    def test_rolling_sum_window_1(self):
        """Window of 1 should return original values."""
        matrix = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
        
        result = _compute_rolling_sum(matrix, window=1)
        
        np.testing.assert_array_almost_equal(result, matrix)
    
    def test_rolling_sum_window_3(self):
        """Window of 3 should sum 3 values (with ramp-up at start)."""
        matrix = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.float64)
        
        result = _compute_rolling_sum(matrix, window=3)
        
        # Rolling sum ramps up at start: [1, 2, 3, 3, 3, 3]
        # Position 0: sum of [0, 0, 1] = 1
        # Position 1: sum of [0, 1, 1] = 2
        # Position 2+: sum of [1, 1, 1] = 3
        np.testing.assert_array_almost_equal(result, [[1, 2, 3, 3, 3, 3]])
    
    def test_rolling_sum_varying_values(self):
        """Test rolling sum with varying values."""
        matrix = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.float64)
        
        result = _compute_rolling_sum(matrix, window=3)
        
        # Window 3: sum of [i-2, i-1, i] for each position
        # Position 0: sum of [0, 0, 0] = 0 (padding with zeros)
        # Position 1: sum of [0, 0, 1] = 1
        # Position 2: sum of [0, 1, 2] = 3
        # Position 3: sum of [1, 2, 3] = 6
        # etc.
        expected = [[0, 1, 3, 6, 9, 12]]
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rolling_sum_preserves_shape(self):
        """Output shape should match input shape."""
        matrix = np.random.rand(100, 24)
        
        for window in [1, 6, 24]:
            result = _compute_rolling_sum(matrix, window)
            assert result.shape == matrix.shape


class TestWindDecomposition:
    """Test wind u/v component computation."""
    
    def test_north_wind(self, sample_date: str, sample_grid_ids: set[str]):
        """North wind (0°) should have u=speed, v=0."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        # Find rows with near-zero wind direction
        raw = load_hourly_meteo(sample_date, grid_ids=sample_grid_ids)
        raw = raw.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        north_mask = (raw['wind_direction_10m'] < 5) | (raw['wind_direction_10m'] > 355)
        if north_mask.sum() > 0:
            idx = raw[north_mask].index[0]
            speed = raw.loc[idx, 'wind_speed_10m']
            
            df_sorted = df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
            u = df_sorted.loc[idx, 'wind_u_10m']
            v = df_sorted.loc[idx, 'wind_v_10m']
            
            # u should be close to speed, v should be close to 0
            np.testing.assert_almost_equal(u, speed, decimal=1)
            np.testing.assert_almost_equal(v, 0, decimal=1)
    
    def test_east_wind(self, sample_date: str, sample_grid_ids: set[str]):
        """East wind (90°) should have u≈0, v≈speed."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        raw = load_hourly_meteo(sample_date, grid_ids=sample_grid_ids)
        raw = raw.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        # Find rows with wind direction close to 90°
        east_mask = (raw['wind_direction_10m'] > 88) & (raw['wind_direction_10m'] < 92)
        if east_mask.sum() > 0:
            idx = raw[east_mask].index[0]
            speed = raw.loc[idx, 'wind_speed_10m']
            direction = raw.loc[idx, 'wind_direction_10m']
            
            df_sorted = df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
            u = df_sorted.loc[idx, 'wind_u_10m']
            v = df_sorted.loc[idx, 'wind_v_10m']
            
            # For 90° wind: u = speed * cos(90°) ≈ 0, v = speed * sin(90°) ≈ speed
            # Use relaxed tolerance since wind direction may not be exactly 90°
            assert abs(u) < speed * 0.1, f"u component should be small for east wind (got {u})"
            assert abs(v - speed) < speed * 0.1, f"v component should be close to speed for east wind"
    
    def test_wind_magnitude_preserved(self, sample_date: str, sample_grid_ids: set[str]):
        """sqrt(u² + v²) should equal original wind speed."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        raw = load_hourly_meteo(sample_date, grid_ids=sample_grid_ids)
        raw = raw.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        df = df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        reconstructed = np.sqrt(df['wind_u_10m']**2 + df['wind_v_10m']**2)
        
        np.testing.assert_array_almost_equal(
            reconstructed.values,
            raw['wind_speed_10m'].values,
            decimal=5
        )


class TestVPD:
    """Test vapor pressure deficit computation."""
    
    def test_vpd_non_negative(self, sample_date: str, sample_grid_ids: set[str]):
        """VPD should always be non-negative."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        assert (df['vpd'] >= 0).all(), "VPD should be non-negative"
    
    def test_vpd_reasonable_range(self, sample_date: str, sample_grid_ids: set[str]):
        """VPD should be in reasonable range (0-10 kPa)."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        assert df['vpd'].max() < 10, "VPD too high (> 10 kPa)"
        assert df['vpd'].min() >= 0, "VPD negative"
    
    def test_vpd_formula(self, sample_date: str, sample_grid_ids: set[str]):
        """VPD should follow saturation vapor pressure formula."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        raw = load_hourly_meteo(sample_date, grid_ids=sample_grid_ids)
        raw = raw.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        df = df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        # Manually compute VPD
        T = raw['temperature_2m'].values
        RH = raw['relative_humidity_2m'].values
        svp = 0.6108 * np.exp(17.27 * T / (T + 237.3))
        expected_vpd = svp * (1 - RH / 100)
        
        np.testing.assert_array_almost_equal(
            df['vpd'].values,
            expected_vpd,
            decimal=5
        )


class TestTempDewpointDiff:
    """Test temperature-dewpoint difference computation."""
    
    def test_temp_dewpoint_diff_non_negative(self, sample_date: str, sample_grid_ids: set[str]):
        """Temp - dewpoint should typically be non-negative."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        # Allow small negative values due to measurement errors
        assert (df['temp_dewpoint_diff'] >= -1).all(), "Temp-dewpoint diff too negative"
    
    def test_temp_dewpoint_diff_formula(self, sample_date: str, sample_grid_ids: set[str]):
        """Temp-dewpoint diff should be T - Td."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        raw = load_hourly_meteo(sample_date, grid_ids=sample_grid_ids)
        raw = raw.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        df = df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        expected = raw['temperature_2m'].values - raw['dewpoint_2m'].values
        
        np.testing.assert_array_almost_equal(
            df['temp_dewpoint_diff'].values,
            expected,
            decimal=5
        )


class TestTemporalLags:
    """Test temporal lag computation correctness."""
    
    def test_lag1h_equals_previous_hour(self, sample_date: str, sample_grid_ids: set[str]):
        """1-hour lag should equal previous hour's value."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        # Get first grid cell
        first_grid = df['grid_id'].iloc[0]
        g0 = df[df['grid_id'] == first_grid].sort_values('datetime').reset_index(drop=True)
        
        # At hour 1, lag1h should equal hour 0 value
        np.testing.assert_almost_equal(
            g0.loc[1, 'temperature_2m_lag1h'],
            g0.loc[0, 'temperature_2m'],
            decimal=5
        )
        
        # At hour 5, lag1h should equal hour 4 value
        np.testing.assert_almost_equal(
            g0.loc[5, 'temperature_2m_lag1h'],
            g0.loc[4, 'temperature_2m'],
            decimal=5
        )
    
    def test_lag3h_equals_3_hours_ago(self, sample_date: str, sample_grid_ids: set[str]):
        """3-hour lag should equal value from 3 hours ago."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        first_grid = df['grid_id'].iloc[0]
        g0 = df[df['grid_id'] == first_grid].sort_values('datetime').reset_index(drop=True)
        
        # At hour 10, lag3h should equal hour 7 value
        np.testing.assert_almost_equal(
            g0.loc[10, 'temperature_2m_lag3h'],
            g0.loc[7, 'temperature_2m'],
            decimal=5
        )
    
    def test_lag12h_equals_12_hours_ago(self, sample_date: str, sample_grid_ids: set[str]):
        """12-hour lag should equal value from 12 hours ago."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        first_grid = df['grid_id'].iloc[0]
        g0 = df[df['grid_id'] == first_grid].sort_values('datetime').reset_index(drop=True)
        
        # At hour 15, lag12h should equal hour 3 value
        np.testing.assert_almost_equal(
            g0.loc[15, 'temperature_2m_lag12h'],
            g0.loc[3, 'temperature_2m'],
            decimal=5
        )
    
    def test_all_lag_variables_present(self, sample_date: str, sample_grid_ids: set[str]):
        """All lag variables should have lag columns."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        for var in LAG_VARIABLES:
            assert var in df.columns, f"Base variable {var} missing"
            
            # Cloud cover only has subset of lags
            if var == 'cloud_cover':
                for lag_h in [6, 24]:
                    col = f'{var}_lag{lag_h}h'
                    assert col in df.columns, f"Missing lag column: {col}"
            else:
                for lag_h in LAG_HOURS:
                    col = f'{var}_lag{lag_h}h'
                    assert col in df.columns, f"Missing lag column: {col}"


class TestCrossDayLags:
    """Test that lags crossing day boundaries work correctly."""
    
    def test_24h_lag_uses_previous_day(self, two_consecutive_dates: tuple[str, str]):
        """24-hour lag at hour 0 should use previous day's hour 0."""
        prev_date, curr_date = two_consecutive_dates
        
        # Use small sample of grids for speed
        sample_grids = get_sample_grid_ids(curr_date)
        
        # Load previous day
        prev_df = load_hourly_meteo(prev_date, grid_ids=sample_grids)
        prev_df = prev_df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        # Compute features for current day
        df = compute_meteo_features(curr_date, grid_ids=sample_grids)
        
        first_grid = df['grid_id'].iloc[0]
        
        # Get hour 0 of current day
        curr_hour0 = df[(df['grid_id'] == first_grid)].sort_values('datetime').iloc[0]
        
        # Get hour 0 of previous day
        prev_hour0 = prev_df[(prev_df['grid_id'] == first_grid)].sort_values('datetime').iloc[0]
        
        # 24h lag at hour 0 should equal previous day hour 0
        np.testing.assert_almost_equal(
            curr_hour0['temperature_2m_lag24h'],
            prev_hour0['temperature_2m'],
            decimal=5,
            err_msg="24h lag at hour 0 should equal previous day hour 0"
        )
    
    def test_12h_lag_at_hour_0_uses_previous_day(self, two_consecutive_dates: tuple[str, str]):
        """12-hour lag at hour 0 should use previous day's hour 12."""
        prev_date, curr_date = two_consecutive_dates
        
        # Use small sample of grids for speed
        sample_grids = get_sample_grid_ids(curr_date)
        
        prev_df = load_hourly_meteo(prev_date, grid_ids=sample_grids)
        prev_df = prev_df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        df = compute_meteo_features(curr_date, grid_ids=sample_grids)
        
        first_grid = df['grid_id'].iloc[0]
        
        # Get hour 0 of current day
        curr = df[(df['grid_id'] == first_grid)].sort_values('datetime').reset_index(drop=True)
        
        # Get hour 12 of previous day
        prev = prev_df[(prev_df['grid_id'] == first_grid)].sort_values('datetime').reset_index(drop=True)
        
        np.testing.assert_almost_equal(
            curr.loc[0, 'temperature_2m_lag12h'],
            prev.loc[12, 'temperature_2m'],
            decimal=5,
            err_msg="12h lag at hour 0 should equal previous day hour 12"
        )


class TestPrecipitationAggregates:
    """Test precipitation rolling sum computation."""
    
    def test_precip_1h_equals_current(self, sample_date: str, sample_grid_ids: set[str]):
        """1-hour precip should equal current hour precipitation."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        raw = load_hourly_meteo(sample_date, grid_ids=sample_grid_ids)
        raw = raw.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        df = df.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        np.testing.assert_array_almost_equal(
            df['precip_last_1h'].values,
            raw['precipitation'].values,
            decimal=5
        )
    
    def test_precip_6h_is_rolling_sum(self, sample_date: str, sample_grid_ids: set[str]):
        """6-hour precip should be sum of last 6 hours."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        raw = load_hourly_meteo(sample_date, grid_ids=sample_grid_ids)
        raw = raw.sort_values(['grid_id', 'datetime']).reset_index(drop=True)
        
        first_grid = df['grid_id'].iloc[0]
        g0_raw = raw[raw['grid_id'] == first_grid].reset_index(drop=True)
        g0 = df[df['grid_id'] == first_grid].sort_values('datetime').reset_index(drop=True)
        
        # At hour 10, precip_last_6h should be sum of hours 5-10
        expected = g0_raw.loc[5:10, 'precipitation'].sum()
        actual = g0.loc[10, 'precip_last_6h']
        
        np.testing.assert_almost_equal(actual, expected, decimal=5)
    
    def test_precip_columns_present(self, sample_date: str, sample_grid_ids: set[str]):
        """All precipitation aggregate columns should be present."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        assert 'precip_last_1h' in df.columns
        assert 'precip_last_6h' in df.columns
        assert 'precip_last_24h' in df.columns


class TestOutputSchema:
    """Test output DataFrame schema and structure."""
    
    def test_has_key_columns(self, sample_date: str, sample_grid_ids: set[str]):
        """Output should have grid_id and datetime columns."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        assert 'grid_id' in df.columns
        assert 'datetime' in df.columns
    
    def test_correct_row_count(self, sample_date: str, sample_grid_ids: set[str]):
        """Output should have grid_count × 24 rows."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        n_grids = df['grid_id'].nunique()
        n_hours = df['datetime'].nunique()
        
        assert len(df) == n_grids * n_hours
        assert n_hours == EXPECTED_HOURS_PER_DAY
    
    def test_filtered_grid_count(self, sample_date: str, sample_grid_ids: set[str]):
        """Output should have only the filtered grid cells."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        n_grids = df['grid_id'].nunique()
        assert n_grids == len(sample_grid_ids), f"Expected {len(sample_grid_ids)} grids, got {n_grids}"
    
    def test_get_meteo_feature_columns_matches_output(self, sample_date: str, sample_grid_ids: set[str]):
        """get_meteo_feature_columns() should list all feature columns."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        feature_cols = get_meteo_feature_columns()
        
        # All listed columns should be present
        for col in feature_cols:
            assert col in df.columns, f"Listed column {col} not in output"
    
    def test_no_duplicate_rows(self, sample_date: str, sample_grid_ids: set[str]):
        """Should not have duplicate (grid_id, datetime) pairs."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        n_unique = df.groupby(['grid_id', 'datetime']).ngroups
        assert n_unique == len(df), "Duplicate (grid_id, datetime) pairs found"


class TestDataQuality:
    """Test data quality constraints."""
    
    def test_no_all_null_columns(self, sample_date: str, sample_grid_ids: set[str]):
        """No feature column should be entirely null."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        for col in df.columns:
            assert not df[col].isna().all(), f"Column {col} is entirely null"
    
    def test_temperature_in_range(self, sample_date: str, sample_grid_ids: set[str]):
        """Temperature should be in realistic range."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        assert df['temperature_2m'].min() > -50, "Temperature too low"
        assert df['temperature_2m'].max() < 60, "Temperature too high"
    
    def test_humidity_lag_values_in_range(self, sample_date: str, sample_grid_ids: set[str]):
        """Humidity lag values should be in 0-100 range."""
        df = compute_meteo_features(sample_date, grid_ids=sample_grid_ids)
        
        for lag_h in LAG_HOURS:
            col = f'relative_humidity_2m_lag{lag_h}h'
            if col in df.columns:
                valid = df[col].dropna()
                assert valid.min() >= 0, f"{col} has values < 0"
                assert valid.max() <= 100, f"{col} has values > 100"
