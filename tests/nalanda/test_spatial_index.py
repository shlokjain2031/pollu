"""Tests for Phase 1: Sensor metadata and grid spatial index."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalanda.sensor_locations import load_sensor_locations, get_sensor_by_site_id
from nalanda.spatial_utils import (
    GridIndex,
    add_utm_coords_to_sensors,
    compute_epanechnikov_trust,
    transform_coords_to_utm,
)


# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
SENSOR_JSON = PROJECT_ROOT / "cache" / "openaq_sensor_ids.json"
GRID_PARQUET = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"


class TestSensorLocations:
    """Tests for sensor metadata loading."""
    
    def test_load_sensors_returns_dataframe(self):
        """Verify load_sensor_locations returns a DataFrame."""
        sensors = load_sensor_locations(SENSOR_JSON)
        assert isinstance(sensors, pd.DataFrame)
        
    def test_load_sensors_has_required_columns(self):
        """Verify required columns are present."""
        sensors = load_sensor_locations(SENSOR_JSON)
        required = {'site_id', 'latitude', 'longitude', 'location_name'}
        assert required.issubset(set(sensors.columns))
        
    def test_load_sensors_filters_missing_site_id(self):
        """Verify sensors without site_id are filtered out."""
        sensors = load_sensor_locations(SENSOR_JSON, require_site_id=True)
        assert sensors['site_id'].notna().all()
        assert (sensors['site_id'] != '').all()
        
    def test_load_sensors_count(self):
        """Verify expected number of sensors loaded."""
        sensors = load_sensor_locations(SENSOR_JSON)
        # We expect ~28 sensors with site_id from the JSON
        assert 20 <= len(sensors) <= 40, f"Got {len(sensors)} sensors"
        
    def test_sensor_coordinates_in_mumbai(self):
        """Verify all sensors are within Mumbai bounds."""
        sensors = load_sensor_locations(SENSOR_JSON)
        # Mumbai rough bounds
        assert (sensors['latitude'] >= 18.5).all()
        assert (sensors['latitude'] <= 19.5).all()
        assert (sensors['longitude'] >= 72.5).all()
        assert (sensors['longitude'] <= 73.5).all()
        
    def test_get_sensor_by_site_id(self):
        """Verify sensor lookup by site_id works."""
        sensors = load_sensor_locations(SENSOR_JSON)
        
        # Pick a known site_id
        known_id = sensors.iloc[0]['site_id']
        result = get_sensor_by_site_id(sensors, known_id)
        
        assert result is not None
        assert result['site_id'] == known_id
        
    def test_get_sensor_by_invalid_id_returns_none(self):
        """Verify invalid site_id returns None."""
        sensors = load_sensor_locations(SENSOR_JSON)
        result = get_sensor_by_site_id(sensors, 'nonexistent_id_99999')
        assert result is None


class TestCoordinateTransform:
    """Tests for WGS84 to UTM coordinate transformation."""
    
    def test_transform_known_point(self):
        """Verify transformation of a known Mumbai point."""
        # Roughly center of Mumbai: Gateway of India
        lat, lon = 18.9220, 72.8347
        x_utm, y_utm = transform_coords_to_utm(
            np.array([lat]),
            np.array([lon]),
        )
        
        # UTM 43N coordinates should be in expected range
        # x ~ 275,000m, y ~ 2,093,000m
        assert 260_000 < x_utm[0] < 290_000, f"x_utm={x_utm[0]}"
        assert 2_080_000 < y_utm[0] < 2_140_000, f"y_utm={y_utm[0]}"
        
    def test_add_utm_coords_to_sensors(self):
        """Verify UTM coordinates are added to sensor DataFrame."""
        sensors = load_sensor_locations(SENSOR_JSON)
        sensors_utm = add_utm_coords_to_sensors(sensors)
        
        assert 'x_utm' in sensors_utm.columns
        assert 'y_utm' in sensors_utm.columns
        assert sensors_utm['x_utm'].notna().all()
        assert sensors_utm['y_utm'].notna().all()


class TestGridIndex:
    """Tests for grid spatial indexing."""
    
    @pytest.fixture
    def grid_index(self):
        """Create grid index fixture."""
        return GridIndex(GRID_PARQUET)
    
    def test_grid_loads_correctly(self, grid_index):
        """Verify grid is loaded with expected size."""
        assert len(grid_index.grid) == 509_209
        
    def test_grid_has_required_columns(self, grid_index):
        """Verify grid has row_idx, x_m, y_m."""
        required = {'row_idx', 'x_m', 'y_m'}
        assert required.issubset(set(grid_index.grid.columns))
        
    def test_snap_to_nearest_returns_valid_row_idx(self, grid_index):
        """Verify nearest snap returns a valid row_idx."""
        # Use center of grid
        x_center = (grid_index.x_min + grid_index.x_max) / 2
        y_center = (grid_index.y_min + grid_index.y_max) / 2
        
        row_idx, dist = grid_index.snap_to_nearest(x_center, y_center)
        
        assert row_idx in grid_index.grid['row_idx'].values
        assert dist >= 0
        assert dist < 30  # Should be within one cell
        
    def test_snap_all_sensors_to_grid(self, grid_index):
        """Verify all sensors snap to valid grid cells."""
        sensors = load_sensor_locations(SENSOR_JSON)
        sensors = add_utm_coords_to_sensors(sensors)
        
        sensors_within_grid = 0
        sensors_outside_grid = []
        
        for _, sensor in sensors.iterrows():
            row_idx, dist = grid_index.snap_to_nearest(sensor['x_utm'], sensor['y_utm'])
            
            # Row idx should always be valid (nearest cell exists)
            assert row_idx in grid_index.grid['row_idx'].values, \
                f"Sensor {sensor['site_id']} snapped to invalid row_idx {row_idx}"
            
            # Track sensors within vs outside grid
            if dist < 100:
                sensors_within_grid += 1
            else:
                sensors_outside_grid.append((sensor['site_id'], dist))
        
        # Most sensors should be within grid (allow some in Thane, etc.)
        assert sensors_within_grid >= 20, \
            f"Only {sensors_within_grid} sensors within grid, outside: {sensors_outside_grid}"
        
        # Log sensors outside grid for visibility
        if sensors_outside_grid:
            print(f"\nSensors outside grid: {sensors_outside_grid}")
                
    def test_query_radius_returns_expected_count(self, grid_index):
        """Verify radius query returns approximately π(R/30)² cells."""
        # Use center of grid
        x_center = (grid_index.x_min + grid_index.x_max) / 2
        y_center = (grid_index.y_min + grid_index.y_max) / 2
        
        radius_m = 250.0
        cells = grid_index.query_radius(x_center, y_center, radius_m)
        
        # Expected: π * (250/30)² ≈ 218 cells
        expected = math.pi * (radius_m / 30) ** 2
        
        assert len(cells) > 0
        assert 0.7 * expected < len(cells) < 1.3 * expected, \
            f"Expected ~{expected:.0f} cells, got {len(cells)}"
            
    def test_query_radius_all_within_distance(self, grid_index):
        """Verify all returned cells are within the query radius."""
        x_center = (grid_index.x_min + grid_index.x_max) / 2
        y_center = (grid_index.y_min + grid_index.y_max) / 2
        
        radius_m = 250.0
        cells = grid_index.query_radius(x_center, y_center, radius_m)
        
        assert (cells['distance_m'] <= radius_m).all()
        
    def test_query_radius_sensor_influence_zones(self, grid_index):
        """Verify each sensor's influence zone has reasonable coverage."""
        sensors = load_sensor_locations(SENSOR_JSON)
        sensors = add_utm_coords_to_sensors(sensors)
        
        radius_m = 250.0
        sensors_with_coverage = 0
        sensors_without_coverage = []
        
        for _, sensor in sensors.iterrows():
            cells = grid_index.query_radius(sensor['x_utm'], sensor['y_utm'], radius_m)
            
            if len(cells) > 100:
                sensors_with_coverage += 1
            else:
                sensors_without_coverage.append((sensor['site_id'], len(cells)))
        
        # At least 20 sensors should be inside grid with proper coverage
        # Some sensors (e.g., in Thane) may be outside the actual grid area
        assert sensors_with_coverage >= 20, \
            f"Only {sensors_with_coverage} sensors have full influence zone coverage"
        
        if sensors_without_coverage:
            print(f"\nSensors without full coverage (outside grid): {sensors_without_coverage}")


class TestTrustKernel:
    """Tests for Epanechnikov trust kernel."""
    
    def test_trust_at_zero_distance(self):
        """Trust should be 1.0 at sensor location."""
        assert compute_epanechnikov_trust(0.0, 250.0) == 1.0
        
    def test_trust_at_boundary(self):
        """Trust should be 0.0 at radius boundary."""
        assert compute_epanechnikov_trust(250.0, 250.0) == 0.0
        
    def test_trust_beyond_boundary(self):
        """Trust should be 0.0 beyond radius."""
        assert compute_epanechnikov_trust(300.0, 250.0) == 0.0
        
    def test_trust_at_half_radius(self):
        """Trust at R/2 should be 1 - 0.25 = 0.75."""
        trust = compute_epanechnikov_trust(125.0, 250.0)
        assert abs(trust - 0.75) < 0.001
        
    def test_trust_vectorized(self):
        """Verify vectorized computation works."""
        distances = np.array([0, 125, 250, 300])
        trusts = compute_epanechnikov_trust(distances, 250.0)
        
        expected = np.array([1.0, 0.75, 0.0, 0.0])
        np.testing.assert_array_almost_equal(trusts, expected)
        
    def test_trust_monotonically_decreasing(self):
        """Trust should decrease with distance."""
        distances = np.linspace(0, 250, 100)
        trusts = compute_epanechnikov_trust(distances, 250.0)
        
        # Check monotonically decreasing
        assert np.all(np.diff(trusts) <= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
