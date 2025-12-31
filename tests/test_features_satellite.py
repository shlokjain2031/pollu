#!/usr/bin/env python3
"""
Unit tests for nalanda/features_satellite.py

Tests satellite feature engineering including:
- Landsat8 spectral indices (NDVI, MNDWI, NDBI, BSI, IBI)
- Sentinel2 indices (EVI, UI_light, SI)
- Temporal date matching
- Cloud mask and spatial features
- Grid filtering

Usage:
    pytest tests/test_features_satellite.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalanda.features_satellite import (
    compute_landsat_features,
    compute_landsat_indices,
    compute_sentinel_features,
    compute_sentinel_indices,
    find_most_recent_date,
    get_available_dates,
    get_landsat_feature_columns,
    get_sentinel_feature_columns,
    load_landsat,
    load_sentinel,
)


# Test constants
PROJECT_ROOT = Path(__file__).parent.parent
LANDSAT_DIR = PROJECT_ROOT / "artifacts" / "v1" / "landsat8" / "mumbai"
SENTINEL_DIR = PROJECT_ROOT / "artifacts" / "v1" / "sentinel2" / "mumbai"
TEST_GRID_SAMPLE_SIZE = 100


def get_landsat_dates() -> list[str]:
    """Find available Landsat8 dates."""
    if not LANDSAT_DIR.exists():
        return []
    return sorted(f.stem for f in LANDSAT_DIR.glob("????-??-??.parquet"))


def get_sentinel_dates() -> list[str]:
    """Find available Sentinel2 dates."""
    if not SENTINEL_DIR.exists():
        return []
    return sorted(f.stem for f in SENTINEL_DIR.glob("????-??-??.parquet"))


def get_sample_grid_ids(date: str, source: str = 'landsat') -> set[str]:
    """Get a small sample of grid_ids for fast testing."""
    import random
    
    if source == 'landsat':
        df = load_landsat(date)
    else:
        df = load_sentinel(date)
    
    all_grids = list(df['grid_id'].unique())
    return set(random.sample(all_grids, min(TEST_GRID_SAMPLE_SIZE, len(all_grids))))


@pytest.fixture
def landsat_dates() -> list[str]:
    """Fixture providing available Landsat dates."""
    dates = get_landsat_dates()
    if not dates:
        pytest.skip("No Landsat8 artifacts found")
    return dates


@pytest.fixture
def sentinel_dates() -> list[str]:
    """Fixture providing available Sentinel2 dates."""
    dates = get_sentinel_dates()
    if not dates:
        pytest.skip("No Sentinel2 artifacts found")
    return dates


@pytest.fixture
def sample_landsat_date(landsat_dates: list[str]) -> str:
    """Fixture providing a single Landsat date."""
    return landsat_dates[0]


@pytest.fixture
def sample_sentinel_date(sentinel_dates: list[str]) -> str:
    """Fixture providing a single Sentinel2 date."""
    return sentinel_dates[0]


@pytest.fixture
def sample_grid_ids(sample_landsat_date: str) -> set[str]:
    """Fixture providing sample grid_ids."""
    return get_sample_grid_ids(sample_landsat_date, 'landsat')


# ============================================================================
# Date matching tests
# ============================================================================

class TestDateMatching:
    """Test temporal date matching logic."""
    
    def test_find_most_recent_exact_match(self):
        """Should return exact date if available."""
        available = ['2018-01-01', '2018-01-17', '2018-02-02']
        result = find_most_recent_date('2018-01-17', available)
        assert result == '2018-01-17'
    
    def test_find_most_recent_before_target(self):
        """Should return most recent date before target."""
        available = ['2018-01-01', '2018-01-17', '2018-02-02']
        result = find_most_recent_date('2018-01-20', available)
        assert result == '2018-01-17'
    
    def test_find_most_recent_no_valid_date(self):
        """Should return None if no date <= target."""
        available = ['2018-01-17', '2018-02-02']
        result = find_most_recent_date('2018-01-10', available)
        assert result is None
    
    def test_find_most_recent_empty_list(self):
        """Should return None for empty list."""
        result = find_most_recent_date('2018-01-10', [])
        assert result is None
    
    def test_get_available_dates(self, landsat_dates: list[str]):
        """Should return sorted list of dates."""
        assert len(landsat_dates) > 0
        # Dates should be sorted
        assert landsat_dates == sorted(landsat_dates)


# ============================================================================
# Landsat8 tests
# ============================================================================

class TestLandsatIndices:
    """Test Landsat8 spectral index computation."""
    
    def test_ndvi_range(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """NDVI should be in [-1, 1] range."""
        df = compute_landsat_features(sample_landsat_date, grid_ids=sample_grid_ids)
        
        assert df['ndvi'].min() >= -1.1, "NDVI too low"
        assert df['ndvi'].max() <= 1.1, "NDVI too high"
    
    def test_mndwi_range(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """MNDWI should be in [-1, 1] range."""
        df = compute_landsat_features(sample_landsat_date, grid_ids=sample_grid_ids)
        
        assert df['mndwi'].min() >= -1.1, "MNDWI too low"
        assert df['mndwi'].max() <= 1.1, "MNDWI too high"
    
    def test_ndbi_range(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """NDBI should be in [-1, 1] range."""
        df = compute_landsat_features(sample_landsat_date, grid_ids=sample_grid_ids)
        
        assert df['ndbi'].min() >= -1.1, "NDBI too low"
        assert df['ndbi'].max() <= 1.1, "NDBI too high"
    
    def test_ndvi_formula(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """NDVI should follow (NIR - Red) / (NIR + Red) formula."""
        df = load_landsat(sample_landsat_date, grid_ids=sample_grid_ids)
        df = compute_landsat_indices(df)
        
        eps = 1e-10
        expected = (df['B5'] - df['B4']) / (df['B5'] + df['B4'] + eps)
        
        np.testing.assert_array_almost_equal(
            df['ndvi'].values,
            expected.values,
            decimal=5
        )
    
    def test_all_indices_present(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """All expected indices should be present."""
        df = compute_landsat_features(sample_landsat_date, grid_ids=sample_grid_ids)
        
        for idx in ['ndvi', 'mndwi', 'ndbi', 'bsi', 'ibi']:
            assert idx in df.columns, f"Missing index: {idx}"
    
    def test_include_bands(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """Should include raw bands when requested."""
        df = compute_landsat_features(
            sample_landsat_date, 
            grid_ids=sample_grid_ids, 
            include_bands=True
        )
        
        for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
            assert band in df.columns, f"Missing band: {band}"
    
    def test_exclude_bands(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """Should exclude raw bands when not requested."""
        df = compute_landsat_features(
            sample_landsat_date, 
            grid_ids=sample_grid_ids, 
            include_bands=False
        )
        
        for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
            assert band not in df.columns, f"Band should be excluded: {band}"
    
    def test_landsat_date_column(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """Should include source date column."""
        df = compute_landsat_features(sample_landsat_date, grid_ids=sample_grid_ids)
        
        assert 'landsat_date' in df.columns
        assert df['landsat_date'].iloc[0] == sample_landsat_date


class TestLandsatFiltering:
    """Test Landsat grid filtering."""
    
    def test_filtered_row_count(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """Filtered output should have correct row count."""
        df = compute_landsat_features(sample_landsat_date, grid_ids=sample_grid_ids)
        
        assert len(df) == len(sample_grid_ids)
    
    def test_filtered_grid_ids_match(self, sample_landsat_date: str, sample_grid_ids: set[str]):
        """Filtered output should only contain requested grid_ids."""
        df = compute_landsat_features(sample_landsat_date, grid_ids=sample_grid_ids)
        
        output_grids = set(df['grid_id'].unique())
        assert output_grids == sample_grid_ids


# ============================================================================
# Sentinel2 tests
# ============================================================================

class TestSentinelIndices:
    """Test Sentinel2 spectral index computation."""
    
    def test_evi_computed(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """EVI should be computed."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        assert 'evi' in df.columns
        assert not df['evi'].isna().all(), "EVI is all NaN"
    
    def test_ndvi_s2_range(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """Sentinel NDVI should be in [-1, 1] range."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        valid = df['ndvi_s2'].dropna()
        assert valid.min() >= -1.1, "NDVI_S2 too low"
        assert valid.max() <= 1.1, "NDVI_S2 too high"
    
    def test_ui_light_computed(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """UI_light should be computed."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        assert 'ui_light' in df.columns
    
    def test_si_range(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """Shadow Index should be in [0, 1] range."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        valid = df['si'].dropna()
        assert valid.min() >= 0, "SI should be non-negative"
        assert valid.max() <= 1.1, "SI too high"
    
    def test_cloud_mask_binary(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """Cloud mask should be binary (0 or 1)."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        valid = df['cloud_mask'].dropna()
        unique_vals = set(valid.unique())
        assert unique_vals.issubset({0, 1}), f"Cloud mask should be binary, got: {unique_vals}"
    
    def test_sentinel_date_column(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """Should include source date column."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        assert 'sentinel_date' in df.columns
        assert df['sentinel_date'].iloc[0] == sample_sentinel_date
    
    def test_all_indices_present(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """All expected Sentinel indices should be present."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        for idx in ['evi', 'ui_light', 'si', 'cloud_mask', 'ndvi_s2']:
            assert idx in df.columns, f"Missing index: {idx}"


class TestSentinelSpatialFeatures:
    """Test Sentinel2 spatial feature computation."""
    
    def test_local_ndvi_std_column_exists(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """local_ndvi_std column should exist."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=True
        )
        
        assert 'local_ndvi_std' in df.columns
    
    def test_cloud_fraction_column_exists(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """cloud_fraction_30m column should exist."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=True
        )
        
        assert 'cloud_fraction_30m' in df.columns
    
    def test_local_ndvi_std_non_negative(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """local_ndvi_std should be non-negative."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=True
        )
        
        valid = df['local_ndvi_std'].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0, "std should be non-negative"
    
    def test_cloud_fraction_range(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """cloud_fraction_30m should be in [0, 1] range."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=True
        )
        
        valid = df['cloud_fraction_30m'].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0, "cloud fraction should be >= 0"
            assert valid.max() <= 1, "cloud fraction should be <= 1"


class TestSentinelFiltering:
    """Test Sentinel grid filtering."""
    
    def test_filtered_row_count(self, sample_sentinel_date: str, sample_grid_ids: set[str]):
        """Filtered output should have correct row count."""
        df = compute_sentinel_features(
            sample_sentinel_date, 
            grid_ids=sample_grid_ids,
            include_spatial=False
        )
        
        assert len(df) == len(sample_grid_ids)


# ============================================================================
# Schema tests
# ============================================================================

class TestFeatureColumns:
    """Test feature column helper functions."""
    
    def test_get_landsat_feature_columns_with_bands(self):
        """Should return all feature columns including bands."""
        cols = get_landsat_feature_columns(include_bands=True)
        
        assert 'ndvi' in cols
        assert 'B1' in cols
        assert 'B11' in cols
    
    def test_get_landsat_feature_columns_without_bands(self):
        """Should return only index columns."""
        cols = get_landsat_feature_columns(include_bands=False)
        
        assert 'ndvi' in cols
        assert 'B1' not in cols
    
    def test_get_sentinel_feature_columns(self):
        """Should return all Sentinel feature columns."""
        cols = get_sentinel_feature_columns()
        
        assert 'evi' in cols
        assert 'local_ndvi_std' in cols
        assert 'cloud_mask' in cols


# ============================================================================
# Temporal matching integration tests
# ============================================================================

class TestTemporalMatching:
    """Test temporal matching with real data."""
    
    def test_landsat_uses_most_recent(self, landsat_dates: list[str], sample_grid_ids: set[str]):
        """Should use most recent Landsat date <= target."""
        if len(landsat_dates) < 2:
            pytest.skip("Need at least 2 Landsat dates")
        
        # Use a target date after the first available date
        target = landsat_dates[1]  # Second available date
        df = compute_landsat_features(target, grid_ids=sample_grid_ids)
        
        # Should use target date itself (exact match)
        assert df['landsat_date'].iloc[0] == target
    
    def test_landsat_temporal_gap(self, landsat_dates: list[str], sample_grid_ids: set[str]):
        """Should handle target date between available dates."""
        if len(landsat_dates) < 2:
            pytest.skip("Need at least 2 Landsat dates")
        
        from datetime import datetime as dt, timedelta
        
        # Create a target date 5 days after first date
        first_date = dt.strptime(landsat_dates[0], '%Y-%m-%d')
        target = (first_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        df = compute_landsat_features(target, grid_ids=sample_grid_ids)
        
        # Should use first date (most recent <= target)
        assert df['landsat_date'].iloc[0] == landsat_dates[0]
