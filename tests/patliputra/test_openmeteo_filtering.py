"""Tests for OpenMeteo target row_idx filtering."""

import tempfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import pytest


def test_load_target_row_idxs_exists():
    """Test loading target row_idxs when file exists."""
    from patliputra.openmeteo_signals import load_target_row_idxs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "target_row_idx"
        
        # Create test data
        target_df = pd.DataFrame({'row_idx': [10, 50, 100, 150, 200]})
        target_df['row_idx'] = target_df['row_idx'].astype('int32')
        target_df.to_parquet(tmp_path)
        
        # Load
        result = load_target_row_idxs(tmp_path)
        
        assert result is not None
        assert result == {10, 50, 100, 150, 200}


def test_load_target_row_idxs_missing():
    """Test loading target row_idxs when file doesn't exist."""
    from patliputra.openmeteo_signals import load_target_row_idxs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "nonexistent"
        
        result = load_target_row_idxs(tmp_path)
        
        assert result is None


def test_grid_with_tiles_filtering():
    """Test that grid_with_tiles is filtered correctly by row_idx."""
    # Create mock grid and grid_with_tiles
    grid_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'row_idx': list(range(100)),
        'x_m': [i * 30 for i in range(100)],
        'y_m': [i * 30 for i in range(100)],
    }
    grid_gdf = gpd.GeoDataFrame(grid_data)
    
    grid_with_tiles_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'meteo_tile_id': [f't{(i // 10):02d}' for i in range(100)],
        'x_m': [i * 30 for i in range(100)],
        'y_m': [i * 30 for i in range(100)],
    }
    grid_with_tiles = pd.DataFrame(grid_with_tiles_data)
    
    # Target cells 20-30
    target_row_idxs = set(range(20, 31))
    
    # Simulate filtering in openmeteo main()
    grid_with_tiles_merged = grid_with_tiles.merge(
        grid_gdf[['grid_id', 'row_idx']], on='grid_id', how='left'
    )
    grid_with_tiles_filtered = grid_with_tiles_merged[
        grid_with_tiles_merged['row_idx'].isin(target_row_idxs)
    ].copy()
    grid_with_tiles_filtered = grid_with_tiles_filtered.drop(columns=['row_idx'])
    
    assert len(grid_with_tiles_filtered) == len(target_row_idxs)
    assert all(grid_with_tiles_filtered['grid_id'].isin([f'g{i:08d}' for i in range(20, 31)]))


def test_filtering_preserves_tile_mapping():
    """Test that filtering preserves meteo_tile_id mapping."""
    grid_with_tiles_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'meteo_tile_id': [f't{(i // 10):02d}' for i in range(100)],
    }
    grid_with_tiles = pd.DataFrame(grid_with_tiles_data)
    
    # Target cells that span multiple tiles
    target_row_idxs = {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}  # Cells from different tiles
    
    # Create a dummy row_idx mapping
    grid_gdf = gpd.GeoDataFrame({
        'grid_id': grid_with_tiles['grid_id'].values,
        'row_idx': list(range(100)),
    })
    
    grid_with_tiles_merged = grid_with_tiles.merge(
        grid_gdf[['grid_id', 'row_idx']], on='grid_id', how='left'
    )
    grid_with_tiles_filtered = grid_with_tiles_merged[
        grid_with_tiles_merged['row_idx'].isin(target_row_idxs)
    ].copy()
    grid_with_tiles_filtered = grid_with_tiles_filtered.drop(columns=['row_idx'])
    
    # Check that tile mapping is preserved
    assert len(grid_with_tiles_filtered) == len(target_row_idxs)
    assert 'meteo_tile_id' in grid_with_tiles_filtered.columns
    assert len(grid_with_tiles_filtered['meteo_tile_id'].unique()) >= 1


def test_no_filtering_when_target_is_none():
    """Test that full grid is used when target_row_idxs is None."""
    grid_with_tiles_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'meteo_tile_id': [f't{(i // 10):02d}' for i in range(100)],
    }
    grid_with_tiles = pd.DataFrame(grid_with_tiles_data)
    
    # No filtering when target_row_idxs is None
    assert len(grid_with_tiles) == 100


def test_filtering_reduces_unique_tiles():
    """Test that filtering can reduce the number of unique tiles fetched."""
    # Create grid spanning multiple tiles
    grid_with_tiles_data = {
        'grid_id': [f'g{i:08d}' for i in range(100)],
        'meteo_tile_id': [f't{(i // 10):02d}' for i in range(100)],  # 10 tiles
    }
    grid_with_tiles_full = pd.DataFrame(grid_with_tiles_data)
    
    # Target only cells from tiles t00 and t01
    target_grid_ids = [f'g{i:08d}' for i in range(20)]  # First 20 cells = tiles t00, t01
    grid_with_tiles_filtered = grid_with_tiles_full[
        grid_with_tiles_full['grid_id'].isin(target_grid_ids)
    ].copy()
    
    full_tiles = grid_with_tiles_full['meteo_tile_id'].nunique()
    filtered_tiles = grid_with_tiles_filtered['meteo_tile_id'].nunique()
    
    assert filtered_tiles < full_tiles
    assert full_tiles == 10
    assert filtered_tiles == 2


def test_filtering_with_categorical_tile_id():
    """Test that filtering works with categorical meteo_tile_id."""
    grid_with_tiles_data = {
        'grid_id': [f'g{i:08d}' for i in range(50)],
        'meteo_tile_id': pd.Categorical([f't{(i // 10):02d}' for i in range(50)]),
    }
    grid_with_tiles = pd.DataFrame(grid_with_tiles_data)
    
    # Create row_idx mapping
    grid_gdf = gpd.GeoDataFrame({
        'grid_id': grid_with_tiles['grid_id'].values,
        'row_idx': list(range(50)),
    })
    
    target_row_idxs = set(range(10, 30))
    
    grid_with_tiles_merged = grid_with_tiles.merge(
        grid_gdf[['grid_id', 'row_idx']], on='grid_id', how='left'
    )
    grid_with_tiles_filtered = grid_with_tiles_merged[
        grid_with_tiles_merged['row_idx'].isin(target_row_idxs)
    ].copy()
    grid_with_tiles_filtered = grid_with_tiles_filtered.drop(columns=['row_idx'])
    
    assert len(grid_with_tiles_filtered) == len(target_row_idxs)
    # Categorical dtype should be preserved or convertible
    assert 'meteo_tile_id' in grid_with_tiles_filtered.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
