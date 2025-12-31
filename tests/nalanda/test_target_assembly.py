"""Tests for Phase 2: Target assembly core logic."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalanda.sensor_locations import load_sensor_locations
from nalanda.spatial_utils import GridIndex, add_utm_coords_to_sensors
from nalanda.target import (
    LAG_HOURS,
    TargetAssembler,
    add_temporal_lags,
    blend_overlapping_sensors,
    build_target_for_day,
    build_target_for_sensor_timestamp,
    precompute_sensor_influence_zones,
)


# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
SENSOR_JSON = PROJECT_ROOT / "cache" / "openaq_sensor_ids.json"
GRID_PARQUET = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"
AQI_ARTIFACT = PROJECT_ROOT / "artifacts" / "v1" / "aqi" / "2021-06-07.parquet"


class TestPrecomputeInfluenceZones:
    """Tests for sensor influence zone precomputation."""
    
    @pytest.fixture
    def sensors_with_utm(self):
        """Load sensors with UTM coordinates."""
        sensors = load_sensor_locations(SENSOR_JSON)
        return add_utm_coords_to_sensors(sensors)
    
    @pytest.fixture
    def grid_index(self):
        """Create grid index fixture."""
        return GridIndex(GRID_PARQUET)
    
    def test_precompute_returns_dict(self, sensors_with_utm, grid_index):
        """Verify precompute returns a dict."""
        zones = precompute_sensor_influence_zones(sensors_with_utm, grid_index)
        assert isinstance(zones, dict)
        
    def test_precompute_has_expected_sensors(self, sensors_with_utm, grid_index):
        """Verify most sensors have influence zones."""
        zones = precompute_sensor_influence_zones(sensors_with_utm, grid_index)
        # At least 20 sensors should be in grid
        assert len(zones) >= 20
        
    def test_influence_zone_has_required_columns(self, sensors_with_utm, grid_index):
        """Verify influence zones have required columns."""
        zones = precompute_sensor_influence_zones(sensors_with_utm, grid_index)
        
        for site_id, zone in zones.items():
            assert 'row_idx' in zone.columns
            assert 'distance_m' in zone.columns
            assert 'trust_weight' in zone.columns
            
    def test_trust_weights_in_valid_range(self, sensors_with_utm, grid_index):
        """Verify trust weights are in [0, 1]."""
        zones = precompute_sensor_influence_zones(sensors_with_utm, grid_index)
        
        for site_id, zone in zones.items():
            assert (zone['trust_weight'] >= 0).all()
            assert (zone['trust_weight'] <= 1).all()
            
    def test_trust_weight_decreases_with_distance(self, sensors_with_utm, grid_index):
        """Verify trust decreases with distance."""
        zones = precompute_sensor_influence_zones(sensors_with_utm, grid_index)
        
        for site_id, zone in zones.items():
            if len(zone) > 10:
                # Sort by distance and check trust is non-increasing
                sorted_zone = zone.sort_values('distance_m')
                trust_diff = np.diff(sorted_zone['trust_weight'].values)
                # Allow small numerical tolerance for ties
                assert (trust_diff <= 1e-6).all(), f"Trust not decreasing for {site_id}"


class TestBuildTargetForSensorTimestamp:
    """Tests for single sensor-timestamp target building."""
    
    @pytest.fixture
    def sample_influence_zone(self):
        """Create a sample influence zone."""
        return pd.DataFrame({
            'row_idx': [100, 101, 102, 103],
            'distance_m': [0.0, 30.0, 60.0, 90.0],
            'trust_weight': [1.0, 0.99, 0.94, 0.87],
        })
    
    def test_builds_correct_rows(self, sample_influence_zone):
        """Verify correct number of rows generated."""
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=sample_influence_zone,
        )
        
        assert len(result) == 4
        
    def test_has_required_columns(self, sample_influence_zone):
        """Verify output has required columns."""
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=sample_influence_zone,
        )
        
        required = {'row_idx', 'timestamp', 'pm25_raw', 'trust_weight', 'site_id'}
        assert required.issubset(set(result.columns))
        
    def test_pm25_value_propagated(self, sample_influence_zone):
        """Verify PM2.5 value is same for all rows."""
        pm25 = 42.5
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=pm25,
            influence_zone=sample_influence_zone,
        )
        
        assert (result['pm25_raw'] == pm25).all()
        
    def test_empty_zone_returns_empty(self):
        """Verify empty zone returns empty DataFrame."""
        empty_zone = pd.DataFrame(columns=['row_idx', 'distance_m', 'trust_weight'])
        result = build_target_for_sensor_timestamp(
            site_id='5120',
            timestamp=pd.Timestamp('2021-06-07 10:00:00'),
            pm25_value=42.5,
            influence_zone=empty_zone,
        )
        
        assert len(result) == 0


class TestBlendOverlappingSensors:
    """Tests for multi-sensor blending."""
    
    def test_single_sensor_passthrough(self):
        """Verify single sensor contributions pass through."""
        contributions = pd.DataFrame({
            'row_idx': [100, 101],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 2,
            'pm25_raw': [42.5, 42.5],
            'trust_weight': [1.0, 0.9],
            'site_id': ['5120', '5120'],
        })
        
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 2
        assert (result['pm25_t'] == 42.5).all()
        assert result.loc[result['row_idx'] == 100, 'target_trust'].values[0] == 1.0
        
    def test_two_sensors_weighted_average(self):
        """Verify two sensors blend correctly."""
        # Same cell, same timestamp, two sensors
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 2,
            'pm25_raw': [40.0, 60.0],
            'trust_weight': [0.5, 0.5],
            'site_id': ['5120', '5121'],
        })
        
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 1
        # Weighted average: (40*0.5 + 60*0.5) / (0.5 + 0.5) = 50
        assert result['pm25_t'].values[0] == 50.0
        # Trust sum: 0.5 + 0.5 = 1.0
        assert result['target_trust'].values[0] == 1.0
        assert result['n_sensors'].values[0] == 2
        
    def test_unequal_weights(self):
        """Verify unequal weights produce correct blend."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [pd.Timestamp('2021-06-07 10:00:00')] * 2,
            'pm25_raw': [40.0, 60.0],
            'trust_weight': [0.8, 0.2],  # First sensor closer
            'site_id': ['5120', '5121'],
        })
        
        result = blend_overlapping_sensors(contributions)
        
        # Weighted average: (40*0.8 + 60*0.2) / (0.8 + 0.2) = 44
        assert abs(result['pm25_t'].values[0] - 44.0) < 0.001
        # Trust sum: 0.8 + 0.2 = 1.0
        assert result['target_trust'].values[0] == 1.0
        
    def test_multiple_timestamps_separate(self):
        """Verify different timestamps stay separate."""
        contributions = pd.DataFrame({
            'row_idx': [100, 100],
            'timestamp': [
                pd.Timestamp('2021-06-07 10:00:00'),
                pd.Timestamp('2021-06-07 11:00:00'),
            ],
            'pm25_raw': [40.0, 60.0],
            'trust_weight': [1.0, 1.0],
            'site_id': ['5120', '5120'],
        })
        
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 2  # Two separate timestamp rows
        
    def test_empty_input_returns_empty(self):
        """Verify empty input returns empty DataFrame."""
        contributions = pd.DataFrame(columns=['row_idx', 'timestamp', 'pm25_raw', 'trust_weight', 'site_id'])
        result = blend_overlapping_sensors(contributions)
        
        assert len(result) == 0


class TestBuildTargetForDay:
    """Tests for full day target assembly."""
    
    @pytest.fixture
    def assembler(self):
        """Create target assembler fixture."""
        return TargetAssembler(SENSOR_JSON, GRID_PARQUET)
    
    def test_assembles_real_data(self, assembler):
        """Verify assembly works on real AQI artifact."""
        aqi_df = pd.read_parquet(AQI_ARTIFACT)
        result = assembler.assemble_day(aqi_df)
        
        assert len(result) > 0
        
    def test_output_has_required_columns(self, assembler):
        """Verify output has required columns."""
        aqi_df = pd.read_parquet(AQI_ARTIFACT)
        result = assembler.assemble_day(aqi_df)
        
        required = {'row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors'}
        assert required.issubset(set(result.columns))
        
    def test_pm25_values_reasonable(self, assembler):
        """Verify PM2.5 values are in reasonable range."""
        aqi_df = pd.read_parquet(AQI_ARTIFACT)
        result = assembler.assemble_day(aqi_df)
        
        # PM2.5 should be positive and < 1000 (reasonable upper bound)
        assert (result['pm25_t'] >= 0).all()
        assert (result['pm25_t'] < 1000).all()
        
    def test_trust_values_positive(self, assembler):
        """Verify trust values are positive."""
        aqi_df = pd.read_parquet(AQI_ARTIFACT)
        result = assembler.assemble_day(aqi_df)
        
        assert (result['target_trust'] > 0).all()
        
    def test_row_expansion(self, assembler):
        """Verify row count increases from sensor readings to cell coverage."""
        aqi_df = pd.read_parquet(AQI_ARTIFACT)
        result = assembler.assemble_day(aqi_df)
        
        # Should have more rows than original (sensors × cells per sensor)
        # But less than sensors × max_cells × timestamps due to overlap
        n_input_rows = len(aqi_df)
        n_output_rows = len(result)
        
        print(f"Input rows: {n_input_rows}, Output rows: {n_output_rows}")
        print(f"Expansion ratio: {n_output_rows / n_input_rows:.1f}x")
        
        # Should expand by roughly 100-200x (each sensor has ~200 cells)
        assert n_output_rows > n_input_rows * 50


class TestTargetAssembler:
    """Tests for TargetAssembler class."""
    
    def test_initialization(self):
        """Verify assembler initializes correctly."""
        assembler = TargetAssembler(SENSOR_JSON, GRID_PARQUET)
        
        assert assembler.n_sensors >= 20
        assert assembler.n_sensors_in_grid >= 20
        assert assembler.total_influenced_cells > 4000
        
    def test_metadata(self):
        """Verify metadata is generated correctly."""
        assembler = TargetAssembler(SENSOR_JSON, GRID_PARQUET, radius_m=250.0)
        metadata = assembler.get_metadata()
        
        assert metadata['radius_m'] == '250.0'
        assert metadata['trust_kernel'] == 'epanechnikov'
        assert int(metadata['n_sensors']) >= 20
        
    def test_custom_radius(self):
        """Verify custom radius affects influence zones."""
        assembler_small = TargetAssembler(SENSOR_JSON, GRID_PARQUET, radius_m=100.0)
        assembler_large = TargetAssembler(SENSOR_JSON, GRID_PARQUET, radius_m=300.0)
        
        # Larger radius should have more influenced cells
        assert assembler_large.total_influenced_cells > assembler_small.total_influenced_cells


class TestTemporalLags:
    """Tests for temporal lag computation."""
    
    def test_lag_columns_added(self):
        """Verify all lag columns are added to target DataFrame."""
        # Create simple target DataFrame
        target_df = pd.DataFrame({
            'row_idx': [1, 1, 1, 2, 2, 2],
            'timestamp': pd.to_datetime([
                '2021-06-07 00:00:00', '2021-06-07 01:00:00', '2021-06-07 02:00:00',
                '2021-06-07 00:00:00', '2021-06-07 01:00:00', '2021-06-07 02:00:00',
            ]),
            'pm25_t': [10.0, 15.0, 20.0, 30.0, 35.0, 40.0],
            'target_trust': [0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
            'n_sensors': [1, 1, 1, 2, 2, 2],
        })
        
        result = add_temporal_lags(target_df)
        
        # Check all lag columns exist
        for lag_h in LAG_HOURS:
            assert f'pm25_t_minus_{lag_h}h' in result.columns
    
    def test_lag_values_correct_within_day(self):
        """Verify lag values are computed correctly for same-day lookups."""
        # Create target with hourly data
        target_df = pd.DataFrame({
            'row_idx': [1] * 6,
            'timestamp': pd.to_datetime([
                '2021-06-07 00:00:00', '2021-06-07 01:00:00', '2021-06-07 02:00:00',
                '2021-06-07 03:00:00', '2021-06-07 06:00:00', '2021-06-07 12:00:00',
            ]),
            'pm25_t': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'target_trust': [0.9] * 6,
            'n_sensors': [1] * 6,
        })
        
        result = add_temporal_lags(target_df)
        
        # Get row for 12:00
        row_12 = result[result['timestamp'] == pd.Timestamp('2021-06-07 12:00:00')].iloc[0]
        
        # pm25_t_minus_6h should be value at 06:00 = 50.0
        assert row_12['pm25_t_minus_6h'] == 50.0
        
        # pm25_t_minus_12h should be value at 00:00 = 10.0
        assert row_12['pm25_t_minus_12h'] == 10.0
    
    def test_lag_values_nan_when_missing(self):
        """Verify lag values are NaN when historical data is not available."""
        target_df = pd.DataFrame({
            'row_idx': [1, 1],
            'timestamp': pd.to_datetime(['2021-06-07 00:00:00', '2021-06-07 01:00:00']),
            'pm25_t': [10.0, 20.0],
            'target_trust': [0.9, 0.9],
            'n_sensors': [1, 1],
        })
        
        result = add_temporal_lags(target_df)
        
        # At 00:00, the 1h lag should be NaN (no previous data)
        row_00 = result[result['timestamp'] == pd.Timestamp('2021-06-07 00:00:00')].iloc[0]
        assert pd.isna(row_00['pm25_t_minus_1h'])
        
        # At 01:00, the 1h lag should be 10.0 (from 00:00)
        row_01 = result[result['timestamp'] == pd.Timestamp('2021-06-07 01:00:00')].iloc[0]
        assert row_01['pm25_t_minus_1h'] == 10.0
    
    def test_lag_with_previous_day(self):
        """Verify 24h lag uses previous day's data when provided."""
        # Previous day data
        prev_day_df = pd.DataFrame({
            'row_idx': [1],
            'timestamp': pd.to_datetime(['2021-06-06 12:00:00']),
            'pm25_t': [100.0],
            'target_trust': [0.9],
            'n_sensors': [1],
        })
        
        # Current day data
        target_df = pd.DataFrame({
            'row_idx': [1],
            'timestamp': pd.to_datetime(['2021-06-07 12:00:00']),
            'pm25_t': [50.0],
            'target_trust': [0.9],
            'n_sensors': [1],
        })
        
        result = add_temporal_lags(target_df, prev_day_df=prev_day_df)
        
        # pm25_t_minus_24h should be 100.0 from previous day
        assert result.iloc[0]['pm25_t_minus_24h'] == 100.0
    
    def test_lag_dtypes_are_float32(self):
        """Verify lag columns have float32 dtype."""
        target_df = pd.DataFrame({
            'row_idx': [1, 1],
            'timestamp': pd.to_datetime(['2021-06-07 00:00:00', '2021-06-07 01:00:00']),
            'pm25_t': [10.0, 20.0],
            'target_trust': [0.9, 0.9],
            'n_sensors': [1, 1],
        })
        
        result = add_temporal_lags(target_df)
        
        for lag_h in LAG_HOURS:
            assert result[f'pm25_t_minus_{lag_h}h'].dtype == np.float32
    
    def test_lag_empty_dataframe(self):
        """Verify empty DataFrame returns empty with lag columns."""
        target_df = pd.DataFrame(columns=['row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors'])
        
        result = add_temporal_lags(target_df)
        
        assert len(result) == 0
        for lag_h in LAG_HOURS:
            assert f'pm25_t_minus_{lag_h}h' in result.columns
    
    def test_lag_multiple_grid_cells(self):
        """Verify lags work correctly with multiple grid cells."""
        target_df = pd.DataFrame({
            'row_idx': [1, 1, 2, 2],
            'timestamp': pd.to_datetime([
                '2021-06-07 00:00:00', '2021-06-07 01:00:00',
                '2021-06-07 00:00:00', '2021-06-07 01:00:00',
            ]),
            'pm25_t': [10.0, 20.0, 100.0, 200.0],
            'target_trust': [0.9, 0.9, 0.8, 0.8],
            'n_sensors': [1, 1, 2, 2],
        })
        
        result = add_temporal_lags(target_df)
        
        # Cell 1 at 01:00 should have pm25_t_minus_1h = 10.0
        cell1_01 = result[(result['row_idx'] == 1) & 
                          (result['timestamp'] == pd.Timestamp('2021-06-07 01:00:00'))].iloc[0]
        assert cell1_01['pm25_t_minus_1h'] == 10.0
        
        # Cell 2 at 01:00 should have pm25_t_minus_1h = 100.0 (not cell 1's value)
        cell2_01 = result[(result['row_idx'] == 2) & 
                          (result['timestamp'] == pd.Timestamp('2021-06-07 01:00:00'))].iloc[0]
        assert cell2_01['pm25_t_minus_1h'] == 100.0


class TestAssemblerWithLags:
    """Tests for TargetAssembler with temporal lags."""
    
    @pytest.fixture
    def assembler(self):
        """Create assembler fixture."""
        return TargetAssembler(SENSOR_JSON, GRID_PARQUET)
    
    @pytest.fixture
    def aqi_df(self):
        """Load sample AQI data."""
        return pd.read_parquet(AQI_ARTIFACT)
    
    def test_assemble_day_includes_lags(self, assembler, aqi_df):
        """Verify assemble_day includes lag columns."""
        result = assembler.assemble_day(aqi_df)
        
        for lag_h in LAG_HOURS:
            assert f'pm25_t_minus_{lag_h}h' in result.columns
    
    def test_assemble_day_with_prev_day(self, assembler, aqi_df):
        """Verify assemble_day accepts previous day data."""
        # First day without previous
        day1 = assembler.assemble_day(aqi_df, prev_day_df=None)
        
        # Second day with previous
        day2 = assembler.assemble_day(aqi_df, prev_day_df=day1)
        
        # Both should have lag columns
        for lag_h in LAG_HOURS:
            assert f'pm25_t_minus_{lag_h}h' in day2.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
