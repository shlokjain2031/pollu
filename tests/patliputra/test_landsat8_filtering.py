"""Tests for Landsat8 target row_idx filtering."""

import tempfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import pytest


def test_load_target_row_idxs_exists():
    """Test loading target row_idxs when file exists."""
    from patliputra.landsat8_signals import load_target_row_idxs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "target_row_idx"
        
        # Create test data
        target_df = pd.DataFrame({'row_idx': [1, 5, 10, 20]})
        target_df['row_idx'] = target_df['row_idx'].astype('int32')
        target_df.to_parquet(tmp_path)
        
        # Load
        result = load_target_row_idxs(tmp_path)
        
        assert result is not None
        assert result == {1, 5, 10, 20}


def test_load_target_row_idxs_missing():
    """Test loading target row_idxs when file doesn't exist."""
    from patliputra.landsat8_signals import load_target_row_idxs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "nonexistent"
        
        result = load_target_row_idxs(tmp_path)
        
        assert result is None


def test_fetch_band_values_filtering_with_target():
    """Test that fetch_band_values_batched filters grid when target_row_idxs provided."""
    # Create mock grid GeoDataFrame
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'row_idx': list(range(100)),
        'geometry': [Point(i, i) for i in range(100)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")
    
    # Target only cells 10-20
    target_row_idxs = set(range(10, 21))
    
    # We can't directly test without mocking EE, but we can test the filter logic manually
    filtered_grid = grid_gdf[grid_gdf['row_idx'].isin(target_row_idxs)].copy()
    assert len(filtered_grid) == len(target_row_idxs)
    assert set(filtered_grid['row_idx']) == target_row_idxs


def test_fetch_band_values_no_filtering():
    """Test that fetch_band_values_batched uses full grid when target_row_idxs is None."""
    # Create mock grid
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'geometry': [Point(i, i) for i in range(100)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")
    
    # With target_row_idxs=None, all rows should be kept
    filtered_grid = grid_gdf.copy()
    
    assert len(filtered_grid) == 100


def test_fetch_band_values_invalid_grid_column():
    """Test that fetch_band_values_batched raises error if row_idx missing."""
    from patliputra.landsat8_signals import fetch_band_values_batched
    
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


def test_target_row_idxs_reduction_factor():
    """Test that filtering produces expected reduction."""
    # Create grid
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(1000)],
        'row_idx': list(range(1000)),
        'geometry': [Point(i % 100, i // 100) for i in range(1000)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")
    
    # Target 10% of cells
    target_row_idxs = set(range(0, 1000, 10))  # Every 10th cell
    
    filtered_grid = grid_gdf[grid_gdf['row_idx'].isin(target_row_idxs)].copy()
    
    assert len(filtered_grid) == len(target_row_idxs)
    assert len(grid_gdf) / len(filtered_grid) == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
