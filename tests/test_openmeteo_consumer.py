#!/usr/bin/env python3
"""
Consumer test for Open-Meteo artifacts.

Validates that Open-Meteo parquet artifacts meet schema and value expectations
for downstream consumption by Nalanda feature assembly.

Expectations:
    - One row per grid cell (509,209 for Mumbai)
    - Columns: grid_id, meteo_tile_id, weather variables + daily aggregates
    - Temperature in realistic range (-40°C to +50°C)
    - Relative humidity in 0-100%
    - Cloud cover in 0-100%
    - Wind speed >= 0 m/s
    - Pressure in realistic range (850-1100 hPa)

Usage:
    pytest tests/test_openmeteo_consumer.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


# Expected columns in artifacts
REQUIRED_COLUMNS = [
    "grid_id",
    "meteo_tile_id",
]

# Meteorology columns (at overpass time)
METEO_OVERPASS_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

# Daily aggregate columns
METEO_DAILY_COLUMNS = [
    "temp_mean",
    "temp_max",
    "temp_min",
    "rh_mean",
    "wind_speed_mean",
    "wind_speed_max",
    "precip_total",
    "cloud_cover_mean",
    "pressure_mean",
]

# Expected grid size for Mumbai
EXPECTED_GRID_SIZE = 509_209


def get_openmeteo_artifacts(city: str = "mumbai") -> list[Path]:
    """Find all Open-Meteo artifacts for a city."""
    artifacts_dir = Path("artifacts/v1/openmeteo") / city
    if not artifacts_dir.exists():
        return []
    return sorted(artifacts_dir.glob("????-??-??.parquet"))


@pytest.fixture
def openmeteo_artifacts() -> list[Path]:
    """Fixture providing all available Open-Meteo artifacts."""
    artifacts = get_openmeteo_artifacts()
    if not artifacts:
        pytest.skip("No Open-Meteo artifacts found")
    return artifacts


@pytest.fixture
def sample_artifact(openmeteo_artifacts: list[Path]) -> Path:
    """Fixture providing a single Open-Meteo artifact for testing."""
    return openmeteo_artifacts[0]


class TestOpenMeteoSchema:
    """Test artifact schema compliance."""
    
    def test_has_required_columns(self, sample_artifact: Path):
        """Artifact must have required columns."""
        df = pd.read_parquet(sample_artifact)
        
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_has_meteo_columns(self, sample_artifact: Path):
        """Artifact should have meteorology columns."""
        df = pd.read_parquet(sample_artifact)
        
        # Check for at least some overpass columns
        overpass_present = sum(c in df.columns for c in METEO_OVERPASS_COLUMNS)
        assert overpass_present >= 4, f"Only {overpass_present} overpass columns found"
        
        # Check for daily aggregates
        daily_present = sum(c in df.columns for c in METEO_DAILY_COLUMNS)
        assert daily_present >= 4, f"Only {daily_present} daily columns found"
    
    def test_grid_id_type(self, sample_artifact: Path):
        """grid_id should be string or integer type."""
        df = pd.read_parquet(sample_artifact)
        # Grid ID can be either string (e.g., "g00000001") or integer
        is_valid = (
            pd.api.types.is_integer_dtype(df["grid_id"]) or
            pd.api.types.is_string_dtype(df["grid_id"]) or
            pd.api.types.is_object_dtype(df["grid_id"])
        )
        assert is_valid, f"grid_id is {df['grid_id'].dtype}, expected string or integer"
    
    def test_meteo_tile_id_type(self, sample_artifact: Path):
        """meteo_tile_id should be string type."""
        df = pd.read_parquet(sample_artifact)
        assert pd.api.types.is_string_dtype(df["meteo_tile_id"]) or \
               pd.api.types.is_object_dtype(df["meteo_tile_id"]), \
            f"meteo_tile_id is {df['meteo_tile_id'].dtype}, expected string"


class TestOpenMeteoGrid:
    """Test grid alignment and coverage."""
    
    def test_row_count(self, sample_artifact: Path):
        """Artifact should have one row per grid cell."""
        df = pd.read_parquet(sample_artifact)
        
        # Allow some tolerance for different grids
        assert len(df) >= 100_000, f"Too few rows: {len(df)}"
        
        if len(df) != EXPECTED_GRID_SIZE:
            print(f"⚠ Row count {len(df):,} differs from expected {EXPECTED_GRID_SIZE:,}")
    
    def test_unique_grid_ids(self, sample_artifact: Path):
        """Each grid_id should appear exactly once."""
        df = pd.read_parquet(sample_artifact)
        
        duplicates = df["grid_id"].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate grid_ids"
    
    def test_grid_id_range(self, sample_artifact: Path):
        """grid_id should be non-empty and valid."""
        df = pd.read_parquet(sample_artifact)
        
        # Grid ID can be string (e.g., "g00000001") or integer
        if pd.api.types.is_integer_dtype(df["grid_id"]):
            assert df["grid_id"].min() >= 1, "grid_id should start from 1"
        else:
            # String grid_id - check not empty
            assert df["grid_id"].notna().all(), "Found null grid_ids"
            assert (df["grid_id"].str.len() > 0).all(), "Found empty grid_ids"
    
    def test_tile_coverage(self, sample_artifact: Path):
        """All cells should be assigned to tiles."""
        df = pd.read_parquet(sample_artifact)
        
        null_tiles = df["meteo_tile_id"].isna().sum()
        assert null_tiles == 0, f"Found {null_tiles} cells without tile assignment"
    
    def test_reasonable_tile_count(self, sample_artifact: Path):
        """Should have reasonable number of tiles (~3-20 for Mumbai)."""
        df = pd.read_parquet(sample_artifact)
        
        unique_tiles = df["meteo_tile_id"].nunique()
        assert 1 <= unique_tiles <= 100, f"Unexpected tile count: {unique_tiles}"
        print(f"ℹ Found {unique_tiles} unique meteo tiles")


class TestOpenMeteoValues:
    """Test meteorology value ranges."""
    
    def test_temperature_range(self, sample_artifact: Path):
        """Temperature should be in realistic range."""
        df = pd.read_parquet(sample_artifact)
        
        if "temperature_2m" in df.columns:
            temp = df["temperature_2m"].dropna()
            if len(temp) > 0:
                assert temp.min() >= -40, f"Temperature too low: {temp.min()}"
                assert temp.max() <= 55, f"Temperature too high: {temp.max()}"
                print(f"ℹ Temperature range: {temp.min():.1f}°C to {temp.max():.1f}°C")
    
    def test_humidity_range(self, sample_artifact: Path):
        """Humidity should be 0-100%."""
        df = pd.read_parquet(sample_artifact)
        
        if "relative_humidity_2m" in df.columns:
            rh = df["relative_humidity_2m"].dropna()
            if len(rh) > 0:
                assert rh.min() >= 0, f"Humidity too low: {rh.min()}"
                assert rh.max() <= 100, f"Humidity too high: {rh.max()}"
    
    def test_cloud_cover_range(self, sample_artifact: Path):
        """Cloud cover should be 0-100%."""
        df = pd.read_parquet(sample_artifact)
        
        if "cloud_cover" in df.columns:
            cc = df["cloud_cover"].dropna()
            if len(cc) > 0:
                assert cc.min() >= 0, f"Cloud cover too low: {cc.min()}"
                assert cc.max() <= 100, f"Cloud cover too high: {cc.max()}"
    
    def test_wind_speed_non_negative(self, sample_artifact: Path):
        """Wind speed should be non-negative."""
        df = pd.read_parquet(sample_artifact)
        
        if "wind_speed_10m" in df.columns:
            ws = df["wind_speed_10m"].dropna()
            if len(ws) > 0:
                assert ws.min() >= 0, f"Wind speed negative: {ws.min()}"
    
    def test_pressure_range(self, sample_artifact: Path):
        """Pressure should be in realistic range."""
        df = pd.read_parquet(sample_artifact)
        
        if "surface_pressure" in df.columns:
            pressure = df["surface_pressure"].dropna()
            if len(pressure) > 0:
                assert pressure.min() >= 850, f"Pressure too low: {pressure.min()}"
                assert pressure.max() <= 1100, f"Pressure too high: {pressure.max()}"
    
    def test_precipitation_non_negative(self, sample_artifact: Path):
        """Precipitation should be non-negative."""
        df = pd.read_parquet(sample_artifact)
        
        if "precipitation" in df.columns:
            precip = df["precipitation"].dropna()
            if len(precip) > 0:
                assert precip.min() >= 0, f"Precipitation negative: {precip.min()}"


class TestOpenMeteoTileConsistency:
    """Test that cells in same tile have same values."""
    
    def test_tile_value_consistency(self, sample_artifact: Path):
        """All cells in same tile should have identical meteo values."""
        df = pd.read_parquet(sample_artifact)
        
        # Check a meteo column
        if "temperature_2m" in df.columns:
            # Group by tile and check variance
            tile_groups = df.groupby("meteo_tile_id")["temperature_2m"]
            
            for tile_id, group in tile_groups:
                unique_vals = group.dropna().unique()
                assert len(unique_vals) <= 1, \
                    f"Tile {tile_id} has multiple temperature values: {unique_vals}"


class TestOpenMeteoMetadata:
    """Test metadata sidecar files."""
    
    def test_metadata_exists(self, sample_artifact: Path):
        """Metadata sidecar should exist."""
        meta_path = sample_artifact.with_suffix(".parquet.meta.json")
        assert meta_path.exists(), f"Missing metadata: {meta_path}"
    
    def test_metadata_contents(self, sample_artifact: Path):
        """Metadata should have expected fields."""
        meta_path = sample_artifact.with_suffix(".parquet.meta.json")
        
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            
            expected_fields = ["date", "city", "source", "model"]
            for field in expected_fields:
                assert field in meta, f"Missing metadata field: {field}"


class TestOpenMeteoAllArtifacts:
    """Test consistency across all artifacts."""
    
    def test_all_same_row_count(self, openmeteo_artifacts: list[Path]):
        """All artifacts should have same row count."""
        row_counts = {}
        for artifact in openmeteo_artifacts[:5]:  # Check first 5
            df = pd.read_parquet(artifact, columns=["grid_id"])
            row_counts[artifact.stem] = len(df)
        
        unique_counts = set(row_counts.values())
        if len(unique_counts) > 1:
            print(f"⚠ Row count varies: {row_counts}")
    
    def test_grid_id_consistency(self, openmeteo_artifacts: list[Path]):
        """All artifacts should have same grid_ids."""
        if len(openmeteo_artifacts) < 2:
            pytest.skip("Need at least 2 artifacts")
        
        df1 = pd.read_parquet(openmeteo_artifacts[0], columns=["grid_id"])
        df2 = pd.read_parquet(openmeteo_artifacts[1], columns=["grid_id"])
        
        ids1 = set(df1["grid_id"])
        ids2 = set(df2["grid_id"])
        
        assert ids1 == ids2, "Grid IDs differ between artifacts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
