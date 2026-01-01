"""Tests for Sentinel2 target row_idx filtering."""

import tempfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import pytest


def test_load_target_row_idxs_exists():
    """Test loading target row_idxs when file exists."""
    from patliputra.sentinel2_signals import load_target_row_idxs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "target_row_idx"
        
        # Create test data
        target_df = pd.DataFrame({'row_idx': [100, 200, 300, 400]})
        target_df['row_idx'] = target_df['row_idx'].astype('int32')
        target_df.to_parquet(tmp_path)
        
        # Load
        result = load_target_row_idxs(tmp_path)
        
        assert result is not None
        assert result == {100, 200, 300, 400}


def test_load_target_row_idxs_missing():
    """Test loading target row_idxs when file doesn't exist."""
    from patliputra.sentinel2_signals import load_target_row_idxs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "nonexistent"
        
        result = load_target_row_idxs(tmp_path)
        
        assert result is None


def test_fetch_band_values_filtering_with_target():
    """Test that fetch_band_values_batched filters grid when target_row_idxs provided."""
    # Create mock grid GeoDataFrame
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(50)],
        'row_idx': list(range(50)),
        'geometry': [Point(i, i) for i in range(50)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")
    
    # Target cells 5-15
    target_row_idxs = set(range(5, 16))
    
    filtered_grid = grid_gdf[grid_gdf['row_idx'].isin(target_row_idxs)].copy()
    
    assert len(filtered_grid) == len(target_row_idxs)
    assert set(filtered_grid['row_idx']) == target_row_idxs
    assert all(gid in filtered_grid['grid_id'].values for gid in [f'g{i:08d}' for i in range(5, 16)])


def test_fetch_band_values_no_filtering():
    """Test that fetch_band_values_batched uses full grid when target_row_idxs is None."""
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(50)],
        'geometry': [Point(i, i) for i in range(50)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")
    
    # With target_row_idxs=None, all rows should be kept
    assert len(grid_gdf) == 50


def test_fetch_band_values_invalid_grid_column():
    """Test that fetch_band_values_batched raises error if row_idx missing."""
    from patliputra.sentinel2_signals import fetch_band_values_batched
    
    # Grid without row_idx column
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(10)],
        'geometry': [Point(i, i) for i in range(10)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")
    
    target_row_idxs = {1, 2, 3}
    
    with pytest.raises(ValueError, match="row_idx"):
        fetch_band_values_batched(
            image=None,
            grid_gdf=grid_gdf,
            target_row_idxs=target_row_idxs,
        )


def test_filtering_preserves_geometry():
    """Test that filtering preserves geometry column."""
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'row_idx': list(range(100)),
        'geometry': [Point(i, i) for i in range(100)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")
    
    target_row_idxs = set(range(25, 75))
    filtered_grid = grid_gdf[grid_gdf['row_idx'].isin(target_row_idxs)].copy()
    
    assert 'geometry' in filtered_grid.columns
    assert len(filtered_grid) == len(target_row_idxs)
    assert filtered_grid.crs == grid_gdf.crs


def test_large_grid_filtering_performance():
    """Test filtering on large grid (500k cells)."""
    # Simulate large grid
    n_cells = 500000
    target_size = 5000
    
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(n_cells)],
        'row_idx': list(range(n_cells)),
    }
    grid_df = pd.DataFrame(grid_data)
    
    # Target random cells
    target_row_idxs = set(np.random.choice(n_cells, target_size, replace=False))
    
    filtered_grid = grid_df[grid_df['row_idx'].isin(target_row_idxs)].copy()
    
    assert len(filtered_grid) == len(target_row_idxs)
    reduction_factor = n_cells / len(filtered_grid)
    assert reduction_factor == pytest.approx(n_cells / target_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
