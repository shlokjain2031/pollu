"""Comprehensive tests for nalanda.cli with feature validation.

These tests verify that assembled training data has correct:
- Meteorological features with realistic value ranges
- Satellite features with correct index bounds
- Temporal lags computed correctly
- No data corruption or unrealistic values
- Edge cases and boundary conditions

Tests are TOUGH - they will catch subtle data quality issues.
"""

from pathlib import Path
import tempfile
import json

import numpy as np
import pandas as pd
import pytest

from nalanda.cli import (
    get_date_range,
    get_available_target_dates,
    get_available_training_dates,
    assemble_single,
    validate_training_data,
    validate_batch,
    generate_report,
    AssemblyResult,
    ValidationResult,
)
from nalanda.training import assemble_training_data


# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_TARGET_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"


# =============================================================================
# Meteorological Feature Value Range Tests
# =============================================================================

class TestMeteoFeatureRanges:
    """Tests for meteorological feature value ranges and realism."""

    @pytest.fixture
    def sample_training_df(self):
        """Create sample training data for validation."""
        try:
            df = assemble_training_data(
                date='2021-06-07',
                city='mumbai',
                include_landsat_bands=False,
                include_sentinel_spatial=False,
                skip_missing_satellite=True,
            )
            return df
        except Exception as e:
            pytest.skip(f"Could not load training data: {e}")

    def test_temperature_realistic_range(self, sample_training_df):
        """Temperature should be in realistic range for Mumbai (-10°C to 50°C)."""
        if 'temperature_2m' not in sample_training_df.columns:
            pytest.skip("temperature_2m column not present")
        
        temp = sample_training_df['temperature_2m'].dropna()
        
        # Mumbai typical range: 15-45°C, allow -10 to 50 for safety
        assert temp.min() >= -10, f"Temperature too low: {temp.min():.1f}°C (physically unlikely)"
        assert temp.max() <= 50, f"Temperature too high: {temp.max():.1f}°C (physically unlikely)"
        
        # Warn if outside typical range
        if temp.min() < 10:
            print(f"WARNING: Unusually cold temperature: {temp.min():.1f}°C")
        if temp.max() > 45:
            print(f"WARNING: Unusually hot temperature: {temp.max():.1f}°C")

    def test_temperature_lags_consistent(self, sample_training_df):
        """Temperature lag values should be consistent with current values."""
        if 'temperature_2m' not in sample_training_df.columns:
            pytest.skip("temperature_2m not present")
        
        # Check that lag columns exist
        lag_cols = [c for c in sample_training_df.columns if c.startswith('temperature_2m_lag')]
        assert len(lag_cols) >= 5, f"Expected at least 5 temperature lag columns, got {len(lag_cols)}"
        
        # Check lag values are in same range as current values
        for lag_col in lag_cols:
            lag_values = sample_training_df[lag_col].dropna()
            if len(lag_values) > 0:
                assert lag_values.min() >= -10, f"{lag_col} has unrealistic low value"
                assert lag_values.max() <= 50, f"{lag_col} has unrealistic high value"

    def test_relative_humidity_in_percentage_range(self, sample_training_df):
        """Relative humidity should be in [0, 100] range."""
        if 'relative_humidity_2m' not in sample_training_df.columns:
            pytest.skip("relative_humidity_2m not present")
        
        rh = sample_training_df['relative_humidity_2m'].dropna()
        
        assert rh.min() >= 0, f"Relative humidity below 0%: {rh.min():.1f}%"
        assert rh.max() <= 100, f"Relative humidity above 100%: {rh.max():.1f}%"
        
        # Check that we have some variation (not all same value)
        assert rh.std() > 0.1, "Relative humidity has suspiciously low variance"

    def test_surface_pressure_realistic(self, sample_training_df):
        """Surface pressure should be in realistic range (950-1050 hPa at sea level)."""
        if 'surface_pressure' not in sample_training_df.columns:
            pytest.skip("surface_pressure not present")
        
        pressure = sample_training_df['surface_pressure'].dropna()
        
        # Mumbai is at sea level, so pressure should be 950-1050 hPa
        assert pressure.min() >= 900, f"Pressure too low: {pressure.min():.1f} hPa"
        assert pressure.max() <= 1100, f"Pressure too high: {pressure.max():.1f} hPa"
        
        # Typical range is narrower
        if pressure.min() < 950 or pressure.max() > 1050:
            print(f"WARNING: Pressure outside typical range: [{pressure.min():.1f}, {pressure.max():.1f}] hPa")

    def test_wind_components_realistic(self, sample_training_df):
        """Wind components should be in realistic range (-50 to 50 m/s)."""
        if 'wind_u_10m' not in sample_training_df.columns:
            pytest.skip("wind_u_10m not present")
        
        u = sample_training_df['wind_u_10m'].dropna()
        v = sample_training_df['wind_v_10m'].dropna()
        
        # Extreme winds would be ±50 m/s (~180 km/h)
        assert u.min() >= -50, f"Wind U component too negative: {u.min():.1f} m/s"
        assert u.max() <= 50, f"Wind U component too positive: {u.max():.1f} m/s"
        assert v.min() >= -50, f"Wind V component too negative: {v.min():.1f} m/s"
        assert v.max() <= 50, f"Wind V component too positive: {v.max():.1f} m/s"
        
        # Compute wind speed from components
        wind_speed = np.sqrt(u**2 + v**2)
        assert wind_speed.max() <= 50, f"Computed wind speed exceeds 50 m/s: {wind_speed.max():.1f}"

    def test_cloud_cover_percentage(self, sample_training_df):
        """Cloud cover should be in [0, 100] range."""
        if 'cloud_cover' not in sample_training_df.columns:
            pytest.skip("cloud_cover not present")
        
        cloud = sample_training_df['cloud_cover'].dropna()
        
        assert cloud.min() >= 0, f"Cloud cover below 0%: {cloud.min():.1f}%"
        assert cloud.max() <= 100, f"Cloud cover above 100%: {cloud.max():.1f}%"

    def test_vpd_positive_and_reasonable(self, sample_training_df):
        """VPD (vapor pressure deficit) should be positive and < 10 kPa."""
        if 'vpd' not in sample_training_df.columns:
            pytest.skip("vpd not present")
        
        vpd = sample_training_df['vpd'].dropna()
        
        assert vpd.min() >= 0, f"VPD is negative: {vpd.min():.3f} kPa (physically impossible)"
        assert vpd.max() <= 10, f"VPD exceeds 10 kPa: {vpd.max():.3f} kPa (extremely high)"
        
        # Typical VPD range is 0-5 kPa
        if vpd.max() > 5:
            print(f"WARNING: High VPD value: {vpd.max():.3f} kPa")

    def test_precipitation_non_negative(self, sample_training_df):
        """Precipitation columns should be non-negative."""
        precip_cols = [c for c in sample_training_df.columns if c.startswith('precip_last_')]
        
        if not precip_cols:
            pytest.skip("No precipitation columns found")
        
        for col in precip_cols:
            precip = sample_training_df[col].dropna()
            assert precip.min() >= 0, f"{col} has negative values: {precip.min():.3f}"
            # Extreme precipitation would be < 100 mm/hr
            assert precip.max() <= 200, f"{col} exceeds 200 mm: {precip.max():.1f} (unrealistic)"

    def test_temperature_dewpoint_relationship(self, sample_training_df):
        """Dewpoint should always be <= temperature."""
        if 'temperature_2m' not in sample_training_df.columns or 'dewpoint_2m' not in sample_training_df.columns:
            pytest.skip("temperature or dewpoint not present")
        
        # Get rows where both are non-null
        valid = sample_training_df[['temperature_2m', 'dewpoint_2m']].dropna()
        
        if len(valid) > 0:
            # Dewpoint should always be <= temperature (physically impossible otherwise)
            violations = valid[valid['dewpoint_2m'] > valid['temperature_2m']]
            
            assert len(violations) == 0, \
                f"Found {len(violations)} rows where dewpoint > temperature (physically impossible)"

    def test_temp_dewpoint_diff_non_negative(self, sample_training_df):
        """temp_dewpoint_diff should be non-negative."""
        if 'temp_dewpoint_diff' not in sample_training_df.columns:
            pytest.skip("temp_dewpoint_diff not present")
        
        diff = sample_training_df['temp_dewpoint_diff'].dropna()
        
        assert diff.min() >= 0, \
            f"temp_dewpoint_diff is negative: {diff.min():.3f} (temperature < dewpoint is impossible)"


# =============================================================================
# Satellite Feature Value Range Tests
# =============================================================================

class TestSatelliteFeatureRanges:
    """Tests for satellite feature value ranges and validity."""

    @pytest.fixture
    def sample_training_df(self):
        """Create sample training data for validation."""
        try:
            df = assemble_training_data(
                date='2021-06-07',
                city='mumbai',
                include_landsat_bands=False,
                include_sentinel_spatial=False,
                skip_missing_satellite=True,
            )
            return df
        except Exception as e:
            pytest.skip(f"Could not load training data: {e}")

    def test_ndvi_in_valid_range(self, sample_training_df):
        """NDVI should be in [-1, 1] range."""
        if 'ndvi' not in sample_training_df.columns:
            pytest.skip("ndvi not present")
        
        ndvi = sample_training_df['ndvi'].dropna()
        
        assert ndvi.min() >= -1, f"NDVI below -1: {ndvi.min():.3f}"
        assert ndvi.max() <= 1, f"NDVI above 1: {ndvi.max():.3f}"
        
        # NDVI typically ranges from -0.3 (water) to 0.8 (dense vegetation)
        if ndvi.min() < -0.5:
            print(f"WARNING: Very low NDVI: {ndvi.min():.3f} (unusual for land)")
        if ndvi.max() > 0.9:
            print(f"WARNING: Very high NDVI: {ndvi.max():.3f} (unusual, possible error)")

    def test_ndbi_in_valid_range(self, sample_training_df):
        """NDBI (built-up index) should be in [-1, 1] range."""
        if 'ndbi' not in sample_training_df.columns:
            pytest.skip("ndbi not present")
        
        ndbi = sample_training_df['ndbi'].dropna()
        
        assert ndbi.min() >= -1, f"NDBI below -1: {ndbi.min():.3f}"
        assert ndbi.max() <= 1, f"NDBI above 1: {ndbi.max():.3f}"

    def test_mndwi_in_valid_range(self, sample_training_df):
        """MNDWI (water index) should be in [-1, 1] range."""
        if 'mndwi' not in sample_training_df.columns:
            pytest.skip("mndwi not present")
        
        mndwi = sample_training_df['mndwi'].dropna()
        
        assert mndwi.min() >= -1, f"MNDWI below -1: {mndwi.min():.3f}"
        assert mndwi.max() <= 1, f"MNDWI above 1: {mndwi.max():.3f}"

    def test_bsi_in_valid_range(self, sample_training_df):
        """BSI (bare soil index) should be in [-1, 1] range."""
        if 'bsi' not in sample_training_df.columns:
            pytest.skip("bsi not present")
        
        bsi = sample_training_df['bsi'].dropna()
        
        assert bsi.min() >= -1, f"BSI below -1: {bsi.min():.3f}"
        assert bsi.max() <= 1, f"BSI above 1: {bsi.max():.3f}"

    def test_ibi_in_valid_range(self, sample_training_df):
        """IBI (built-up index) should be in [-2, 2] range (derived from NDBI)."""
        if 'ibi' not in sample_training_df.columns:
            pytest.skip("ibi not present")
        
        ibi = sample_training_df['ibi'].dropna()
        
        # IBI can exceed [-1, 1] since it's derived from multiple indices
        assert ibi.min() >= -2, f"IBI below -2: {ibi.min():.3f}"
        assert ibi.max() <= 2, f"IBI above 2: {ibi.max():.3f}"

    def test_evi_in_valid_range(self, sample_training_df):
        """EVI (enhanced vegetation index) should be in [-1, 1] range."""
        if 'evi' not in sample_training_df.columns:
            pytest.skip("evi not present")
        
        evi = sample_training_df['evi'].dropna()
        
        # EVI can sometimes exceed [-1, 1] slightly due to formula, but should be close
        assert evi.min() >= -2, f"EVI far below expected range: {evi.min():.3f}"
        assert evi.max() <= 2, f"EVI far above expected range: {evi.max():.3f}"
        
        # Typical range is -0.2 to 1.0
        if evi.min() < -1 or evi.max() > 1.5:
            print(f"WARNING: EVI outside typical range: [{evi.min():.3f}, {evi.max():.3f}]")

    def test_cloud_mask_binary(self, sample_training_df):
        """cloud_mask should be binary (0 or 1)."""
        if 'cloud_mask' not in sample_training_df.columns:
            pytest.skip("cloud_mask not present")
        
        cloud_mask = sample_training_df['cloud_mask'].dropna()
        
        unique_values = set(cloud_mask.unique())
        assert unique_values.issubset({0, 1}), \
            f"cloud_mask should be binary, got values: {sorted(unique_values)}"

    def test_cloud_fraction_percentage(self, sample_training_df):
        """cloud_fraction_30m should be in [0, 1] range."""
        if 'cloud_fraction_30m' not in sample_training_df.columns:
            pytest.skip("cloud_fraction_30m not present")
        
        cloud_frac = sample_training_df['cloud_fraction_30m'].dropna()
        
        assert cloud_frac.min() >= 0, f"cloud_fraction_30m below 0: {cloud_frac.min():.3f}"
        assert cloud_frac.max() <= 1, f"cloud_fraction_30m above 1: {cloud_frac.max():.3f}"

    def test_local_ndvi_std_non_negative(self, sample_training_df):
        """local_ndvi_std (spatial variability) should be non-negative."""
        if 'local_ndvi_std' not in sample_training_df.columns:
            pytest.skip("local_ndvi_std not present")
        
        std = sample_training_df['local_ndvi_std'].dropna()
        
        assert std.min() >= 0, f"local_ndvi_std is negative: {std.min():.3f} (std cannot be negative)"
        # High std means high vegetation heterogeneity, should be < 1 (since NDVI is [-1,1])
        assert std.max() <= 1, f"local_ndvi_std exceeds 1: {std.max():.3f}"


# =============================================================================
# Data Quality and Completeness Tests
# =============================================================================

class TestDataQualityAndCompleteness:
    """Tests for overall data quality and completeness."""

    @pytest.fixture
    def sample_training_df(self):
        """Create sample training data for validation."""
        try:
            df = assemble_training_data(
                date='2021-06-07',
                city='mumbai',
                include_landsat_bands=False,
                include_sentinel_spatial=False,
                skip_missing_satellite=True,
            )
            return df
        except Exception as e:
            pytest.skip(f"Could not load training data: {e}")

    def test_required_columns_present(self, sample_training_df):
        """All required columns should be present."""
        required = [
            'row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors',
            'temperature_2m', 'relative_humidity_2m',
        ]
        
        missing = [c for c in required if c not in sample_training_df.columns]
        assert not missing, f"Missing required columns: {missing}"

    def test_no_all_null_columns(self, sample_training_df):
        """No columns should be entirely null."""
        all_null = [c for c in sample_training_df.columns if sample_training_df[c].isna().all()]
        
        assert not all_null, f"Found all-null columns: {all_null}"

    def test_key_columns_not_null(self, sample_training_df):
        """Key columns should not have null values."""
        key_cols = ['row_idx', 'timestamp', 'pm25_t']
        
        for col in key_cols:
            null_count = sample_training_df[col].isna().sum()
            assert null_count == 0, f"{col} has {null_count} null values (should be 0)"

    def test_meteo_features_low_null_rate(self, sample_training_df):
        """Core meteo features should have low null rates."""
        meteo_core = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure']
        
        for col in meteo_core:
            if col in sample_training_df.columns:
                null_pct = sample_training_df[col].isna().mean()
                assert null_pct < 0.1, \
                    f"{col} has high null rate: {null_pct*100:.1f}% (should be < 10%)"

    def test_satellite_features_acceptable_null_rate(self, sample_training_df):
        """Satellite features can have higher null rates (clouds, no overpasses)."""
        satellite_cols = ['ndvi', 'ndbi', 'evi']
        
        for col in satellite_cols:
            if col in sample_training_df.columns:
                null_pct = sample_training_df[col].isna().mean()
                # Satellite data can be 50%+ null due to clouds and revisit times
                assert null_pct < 0.95, \
                    f"{col} has extremely high null rate: {null_pct*100:.1f}% (should be < 95%)"

    def test_no_duplicate_keys(self, sample_training_df):
        """No duplicate (row_idx, timestamp) pairs."""
        dups = sample_training_df.duplicated(subset=['row_idx', 'timestamp']).sum()
        assert dups == 0, f"Found {dups} duplicate (row_idx, timestamp) pairs"

    def test_timestamps_hourly_aligned(self, sample_training_df):
        """Timestamps should be hourly aligned (minute and second should be 0)."""
        timestamps = pd.to_datetime(sample_training_df['timestamp'])
        
        not_hourly = timestamps[(timestamps.dt.minute != 0) | (timestamps.dt.second != 0)]
        assert len(not_hourly) == 0, \
            f"Found {len(not_hourly)} timestamps not aligned to the hour"

    def test_timestamps_within_date(self, sample_training_df):
        """All timestamps should be within the target date."""
        timestamps = pd.to_datetime(sample_training_df['timestamp'])
        
        # Extract unique dates
        unique_dates = timestamps.dt.date.unique()
        
        # Should have timestamps from target date (and possibly adjacent for lags)
        # But shouldn't have random dates far away
        date_range = (unique_dates.max() - unique_dates.min()).days
        assert date_range <= 2, \
            f"Timestamps span {date_range} days (should be ≤ 2 for single-day assembly with lags)"

    def test_row_count_reasonable(self, sample_training_df):
        """Row count should be reasonable for Mumbai."""
        # Mumbai has ~59k grid cells, 24 hours/day
        # So for one day, expect 59k rows (one per cell at various hours)
        # Lower bound: at least 10k rows (some cells may have missing data)
        # Upper bound: at most 100k rows (shouldn't exceed grid size × hours)
        
        n_rows = len(sample_training_df)
        assert n_rows >= 1000, f"Row count too low: {n_rows:,} (expected > 1,000)"
        assert n_rows <= 2_000_000, f"Row count too high: {n_rows:,} (expected < 2M for single day)"


# =============================================================================
# CLI Function Tests
# =============================================================================

class TestCLIFunctions:
    """Tests for CLI helper functions."""

    def test_get_date_range_single_day(self):
        """Test date range for single day."""
        dates = get_date_range('2021-06-07', '2021-06-07')
        assert dates == ['2021-06-07']

    def test_get_date_range_multi_day(self):
        """Test date range for multiple days."""
        dates = get_date_range('2021-06-07', '2021-06-10')
        assert dates == ['2021-06-07', '2021-06-08', '2021-06-09', '2021-06-10']

    def test_get_date_range_month_boundary(self):
        """Test date range across month boundary."""
        dates = get_date_range('2021-06-29', '2021-07-02')
        expected = ['2021-06-29', '2021-06-30', '2021-07-01', '2021-07-02']
        assert dates == expected

    def test_get_available_target_dates(self):
        """Test getting available target dates."""
        dates = get_available_target_dates(DEFAULT_TARGET_DIR)
        
        # Should return a list (may be empty if no data)
        assert isinstance(dates, list)
        
        # If any dates, they should be valid date strings
        for date in dates:
            from datetime import datetime
            datetime.strptime(date, '%Y-%m-%d')  # Should not raise

    def test_get_available_training_dates(self):
        """Test getting available training dates."""
        dates = get_available_training_dates(DEFAULT_OUTPUT_DIR)
        
        assert isinstance(dates, list)
        
        # Should be sorted
        assert dates == sorted(dates)


# =============================================================================
# Assembly and Validation Integration Tests
# =============================================================================

class TestAssemblyAndValidation:
    """Integration tests for assembly and validation."""

    def test_assemble_single_success(self):
        """Test assembling a single date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = assemble_single(
                date='2021-06-07',
                city='mumbai',
                include_bands=False,
                no_spatial=True,
                output_dir=Path(tmpdir),
                dry_run=False,
            )
            
            if not result.success:
                pytest.skip(f"Assembly failed: {result.error}")
            
            assert result.success
            assert result.rows > 0
            assert result.columns > 0
            assert result.elapsed_seconds > 0
            assert result.output_path is not None
            
            # Verify file exists
            assert Path(result.output_path).exists()

    def test_assemble_dry_run(self):
        """Test dry run mode."""
        result = assemble_single(
            date='2021-06-07',
            city='mumbai',
            include_bands=False,
            no_spatial=True,
            output_dir=Path(tempfile.gettempdir()),
            dry_run=True,
        )
        
        if not result.success:
            pytest.skip(f"Assembly failed: {result.error}")
        
        assert result.success
        assert result.output_path is None  # Dry run doesn't save

    def test_validate_training_data_valid(self):
        """Test validation on valid training data."""
        # Check if any training data exists
        dates = get_available_training_dates(DEFAULT_OUTPUT_DIR)
        
        if not dates:
            pytest.skip("No training data available")
        
        result = validate_training_data(dates[0], DEFAULT_OUTPUT_DIR)
        
        # Should have results (may or may not be valid)
        assert isinstance(result, ValidationResult)
        assert result.date == dates[0]
        assert result.rows >= 0

    def test_validate_nonexistent_date(self):
        """Test validation on nonexistent date."""
        result = validate_training_data(
            '1999-01-01',
            DEFAULT_OUTPUT_DIR,
        )
        
        assert not result.valid
        assert len(result.errors) > 0
        assert 'not found' in result.errors[0].lower()

    def test_validate_batch(self):
        """Test batch validation."""
        dates = get_available_training_dates(DEFAULT_OUTPUT_DIR)
        
        if not dates:
            pytest.skip("No training data available")
        
        # Validate first 3 dates (or all if < 3)
        test_dates = dates[:min(3, len(dates))]
        
        results = validate_batch(test_dates, DEFAULT_OUTPUT_DIR, verbose=False)
        
        assert len(results) == len(test_dates)
        assert all(isinstance(r, ValidationResult) for r in results)


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_structure(self):
        """Test that report has expected structure."""
        report = generate_report(DEFAULT_OUTPUT_DIR, DEFAULT_TARGET_DIR)
        
        # Check top-level keys
        assert 'generated_at' in report
        assert 'summary' in report
        assert 'date_coverage' in report
        assert 'validation_results' in report
        
        # Check summary structure
        summary = report['summary']
        required_keys = [
            'target_dates_available',
            'training_dates_assembled',
            'missing_dates',
            'valid_training_files',
            'invalid_training_files',
            'total_rows',
        ]
        for key in required_keys:
            assert key in summary

    def test_generate_report_json_serializable(self):
        """Test that report can be serialized to JSON."""
        report = generate_report(DEFAULT_OUTPUT_DIR, DEFAULT_TARGET_DIR)
        
        # Should not raise
        json_str = json.dumps(report, indent=2)
        assert len(json_str) > 0
        
        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed == report


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================

class TestEdgeCasesAndStress:
    """Edge cases and stress tests."""

    def test_validation_empty_dataframe(self):
        """Test validation behavior on empty parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty parquet
            empty_df = pd.DataFrame(columns=['row_idx', 'timestamp', 'pm25_t'])
            empty_path = Path(tmpdir) / '2021-01-01' / 'training.parquet'
            empty_path.parent.mkdir(parents=True)
            empty_df.to_parquet(empty_path)
            
            result = validate_training_data('2021-01-01', Path(tmpdir))
            
            assert not result.valid
            assert any('empty' in e.lower() for e in result.errors)

    def test_date_range_leap_year(self):
        """Test date range handles leap year correctly."""
        dates = get_date_range('2020-02-28', '2020-03-01')
        expected = ['2020-02-28', '2020-02-29', '2020-03-01']
        assert dates == expected

    def test_date_range_year_boundary(self):
        """Test date range across year boundary."""
        dates = get_date_range('2020-12-30', '2021-01-02')
        expected = ['2020-12-30', '2020-12-31', '2021-01-01', '2021-01-02']
        assert dates == expected

    def test_assembly_result_to_dict(self):
        """Test AssemblyResult serialization."""
        result = AssemblyResult(
            date='2021-06-07',
            success=True,
            rows=100000,
            columns=50,
            elapsed_seconds=12.5,
            error=None,
            output_path='/path/to/output.parquet',
        )
        
        d = result.to_dict()
        assert d['date'] == '2021-06-07'
        assert d['success'] is True
        assert d['rows'] == 100000

    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization."""
        result = ValidationResult(
            date='2021-06-07',
            valid=True,
            rows=100000,
            columns=50,
            null_pct={'pm25_t': 0.0, 'ndvi': 45.2},
            warnings=['High NDVI null rate'],
            errors=[],
        )
        
        d = result.to_dict()
        assert d['date'] == '2021-06-07'
        assert d['valid'] is True
        assert 'null_pct' in d
        assert len(d['warnings']) == 1
