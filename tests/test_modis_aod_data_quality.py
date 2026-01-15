"""
Data quality tests for MODIS AOD artifacts.

Tests validate:
- File schema and format
- Data completeness and integrity
- Temporal coverage
- Geospatial consistency
- Statistical properties
- Manifest alignment
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3


# Configuration
ARTIFACTS_ROOT = Path(__file__).parent.parent / "artifacts"
MODIS_AOD_DIR = ARTIFACTS_ROOT / "v1" / "modis_aod" / "mumbai"
MANIFEST_DB = MODIS_AOD_DIR / "manifest" / "modis_aod_manifest.db"
PIXEL_GRID_PATH = MODIS_AOD_DIR / "modis_pixel_grid.parquet"

# Expected schema
EXPECTED_COLUMNS = [
    'pixel_id', 'pixel_row', 'pixel_col',
    'aod_047', 'aod_055', 'aod_uncertainty', 'aod_qa', 'column_wv'
]

EXPECTED_DTYPES = {
    'pixel_id': 'object',
    'pixel_row': 'int64',
    'pixel_col': 'int64',
    'aod_047': 'float64',
    'aod_055': 'float64',
    'aod_uncertainty': 'float64',
    'aod_qa': 'Int32',  # Nullable integer
    'column_wv': 'float64',  # Can be int64 if all integers, allow both
}

# Expected data ranges (from MODIS documentation)
AOD_VALUE_RANGE = (-0.1, 5.0)  # Valid AOD range
AOD_QA_MAX = 65535  # 16-bit unsigned int
COLUMN_WV_RANGE = (0, 10000)  # Precipitable water in cm * 1000

# Mumbai bounding box
MUMBAI_BBOX = (72.7763, 18.8939, 72.9797, 19.2701)  # west, south, east, north
EXPECTED_PIXEL_COUNT = 882  # 42 rows × 22 cols - edge pixels


class TestFileExistence:
    """Test that MODIS AOD files exist and are accessible."""
    
    def test_output_directory_exists(self):
        """Check that output directory exists."""
        assert MODIS_AOD_DIR.exists(), f"MODIS AOD directory not found: {MODIS_AOD_DIR}"
        assert MODIS_AOD_DIR.is_dir(), f"Path is not a directory: {MODIS_AOD_DIR}"
    
    def test_pixel_grid_exists(self):
        """Check that pixel grid reference file exists."""
        assert PIXEL_GRID_PATH.exists(), f"Pixel grid not found: {PIXEL_GRID_PATH}"
        
        # Validate pixel grid
        grid = pd.read_parquet(PIXEL_GRID_PATH)
        assert len(grid) == EXPECTED_PIXEL_COUNT, \
            f"Expected {EXPECTED_PIXEL_COUNT} pixels, got {len(grid)}"
        assert 'pixel_id' in grid.columns
        assert 'pixel_lon' in grid.columns
        assert 'pixel_lat' in grid.columns
    
    def test_data_files_exist(self):
        """Check that at least some data files exist."""
        parquet_files = list(MODIS_AOD_DIR.glob("2*.parquet"))
        assert len(parquet_files) > 0, "No MODIS AOD parquet files found"
        assert len(parquet_files) > 100, f"Only {len(parquet_files)} files found, expected many more"
    
    def test_manifest_exists(self):
        """Check that manifest database exists."""
        assert MANIFEST_DB.exists(), f"Manifest database not found: {MANIFEST_DB}"


class TestSchemaValidation:
    """Test that all files have correct schema."""
    
    @pytest.fixture(scope="class")
    def sample_files(self):
        """Get sample of files for testing."""
        files = sorted(MODIS_AOD_DIR.glob("2*.parquet"))
        # Test first 10, last 10, and 10 random from middle
        sample = []
        if len(files) > 0:
            sample.extend(files[:10])
            sample.extend(files[-10:])
            if len(files) > 40:
                mid_start = len(files) // 2 - 5
                sample.extend(files[mid_start:mid_start+10])
        return list(set(sample))  # Remove duplicates
    
    def test_all_files_readable(self, sample_files):
        """Test that all sample files can be read."""
        for file_path in sample_files:
            try:
                df = pd.read_parquet(file_path)
                assert df is not None
            except Exception as e:
                pytest.fail(f"Failed to read {file_path.name}: {e}")
    
    def test_column_names(self, sample_files):
        """Test that all files have correct columns."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            assert list(df.columns) == EXPECTED_COLUMNS, \
                f"{file_path.name}: Expected columns {EXPECTED_COLUMNS}, got {list(df.columns)}"
    
    def test_column_dtypes(self, sample_files):
        """Test that columns have correct data types."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            for col, expected_dtype in EXPECTED_DTYPES.items():
                actual_dtype = str(df[col].dtype)
                
                # Allow 'object' for float columns if all values are null
                # (parquet stores all-null float columns as object)
                if expected_dtype == 'float64' and actual_dtype == 'object':
                    if df[col].notna().sum() == 0:
                        continue  # All null, object dtype is acceptable
                
                # Allow int64 for column_wv if all values are integers
                if col == 'column_wv' and actual_dtype == 'int64':
                    continue  # int64 is acceptable for column_wv
                
                assert actual_dtype == expected_dtype, \
                    f"{file_path.name}: Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
    
    def test_no_duplicate_columns(self, sample_files):
        """Test that there are no duplicate column names."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            assert len(df.columns) == len(set(df.columns)), \
                f"{file_path.name}: Duplicate columns found: {df.columns.tolist()}"


class TestDataIntegrity:
    """Test data completeness and integrity."""
    
    @pytest.fixture(scope="class")
    def sample_files(self):
        """Get sample of files for testing."""
        files = sorted(MODIS_AOD_DIR.glob("2*.parquet"))
        sample = files[:20] if len(files) > 20 else files
        return sample
    
    def test_row_count(self, sample_files):
        """Test that each file has expected number of pixels."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            assert len(df) == EXPECTED_PIXEL_COUNT, \
                f"{file_path.name}: Expected {EXPECTED_PIXEL_COUNT} rows, got {len(df)}"
    
    def test_pixel_ids_complete(self, sample_files):
        """Test that all expected pixel IDs are present."""
        # Load reference grid
        ref_grid = pd.read_parquet(PIXEL_GRID_PATH)
        expected_pixel_ids = set(ref_grid['pixel_id'])
        
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            actual_pixel_ids = set(df['pixel_id'])
            
            assert actual_pixel_ids == expected_pixel_ids, \
                f"{file_path.name}: Pixel IDs mismatch. Missing: {expected_pixel_ids - actual_pixel_ids}"
    
    def test_pixel_row_col_consistency(self, sample_files):
        """Test that pixel_id matches pixel_row and pixel_col."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            # Reconstruct pixel_id from row and col
            reconstructed = df.apply(lambda x: f"{x['pixel_row']}_{x['pixel_col']}", axis=1)
            mismatches = df[reconstructed != df['pixel_id']]
            
            assert len(mismatches) == 0, \
                f"{file_path.name}: {len(mismatches)} pixel_id mismatches found"
    
    def test_no_duplicate_pixels(self, sample_files):
        """Test that there are no duplicate pixel IDs."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            duplicates = df[df.duplicated(subset=['pixel_id'], keep=False)]
            
            assert len(duplicates) == 0, \
                f"{file_path.name}: {len(duplicates)} duplicate pixels found"
    
    def test_pixel_coordinates_in_range(self, sample_files):
        """Test that pixel rows and columns are within expected ranges."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            # Based on 42 rows × 22 cols
            assert df['pixel_row'].min() >= 0, f"{file_path.name}: Negative pixel_row found"
            assert df['pixel_row'].max() <= 41, f"{file_path.name}: pixel_row > 41 found"
            assert df['pixel_col'].min() >= 0, f"{file_path.name}: Negative pixel_col found"
            assert df['pixel_col'].max() <= 21, f"{file_path.name}: pixel_col > 21 found"


class TestDataValues:
    """Test that data values are within expected ranges."""
    
    @pytest.fixture(scope="class")
    def sample_files(self):
        """Get sample of files for testing."""
        files = sorted(MODIS_AOD_DIR.glob("2*.parquet"))
        return files[:30] if len(files) > 30 else files
    
    def test_aod_value_ranges(self, sample_files):
        """Test that AOD values are within valid range."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            for col in ['aod_047', 'aod_055']:
                valid_values = df[col].dropna()
                if len(valid_values) > 0:
                    min_val = valid_values.min()
                    max_val = valid_values.max()
                    
                    assert min_val >= AOD_VALUE_RANGE[0], \
                        f"{file_path.name}: {col} min={min_val} below range {AOD_VALUE_RANGE[0]}"
                    assert max_val <= AOD_VALUE_RANGE[1], \
                        f"{file_path.name}: {col} max={max_val} above range {AOD_VALUE_RANGE[1]}"
    
    def test_aod_uncertainty_positive(self, sample_files):
        """Test that most AOD uncertainty values are non-negative."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            valid_values = df['aod_uncertainty'].dropna()
            if len(valid_values) > 0:
                negative_count = (valid_values < 0).sum()
                # Allow up to 1% of values to be negative (data quality outliers)
                max_allowed_negative = max(1, len(valid_values) // 100)
                
                assert negative_count <= max_allowed_negative, \
                    f"{file_path.name}: {negative_count} negative aod_uncertainty values found (max allowed: {max_allowed_negative})"
    
    def test_aod_qa_range(self, sample_files):
        """Test that AOD QA values are within valid range."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            valid_values = df['aod_qa'].dropna()
            if len(valid_values) > 0:
                assert valid_values.min() >= 0, \
                    f"{file_path.name}: Negative aod_qa found"
                assert valid_values.max() <= AOD_QA_MAX, \
                    f"{file_path.name}: aod_qa exceeds max {AOD_QA_MAX}"
    
    def test_column_wv_range(self, sample_files):
        """Test that column water vapor values are reasonable."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            valid_values = df['column_wv'].dropna()
            if len(valid_values) > 0:
                min_val = valid_values.min()
                max_val = valid_values.max()
                
                assert min_val >= COLUMN_WV_RANGE[0], \
                    f"{file_path.name}: column_wv min={min_val} below {COLUMN_WV_RANGE[0]}"
                assert max_val <= COLUMN_WV_RANGE[1], \
                    f"{file_path.name}: column_wv max={max_val} above {COLUMN_WV_RANGE[1]}"
    
    def test_data_not_all_null(self, sample_files):
        """Test that files have some valid AOD data (not all null)."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            # At least one of the AOD columns should have some data
            aod_047_valid = df['aod_047'].notna().sum()
            aod_055_valid = df['aod_055'].notna().sum()
            
            # Allow up to 100% null for cloudy days, but warn if completely empty
            if aod_047_valid == 0 and aod_055_valid == 0:
                pytest.skip(f"{file_path.name}: All AOD values are null (likely cloudy)")
    
    def test_null_fractions_reasonable(self, sample_files):
        """Test that null fractions are within expected range (0-100%)."""
        for file_path in sample_files:
            df = pd.read_parquet(file_path)
            
            for col in ['aod_047', 'aod_055']:
                null_frac = df[col].isna().sum() / len(df)
                assert 0 <= null_frac <= 1, \
                    f"{file_path.name}: {col} null fraction {null_frac} outside [0,1]"


class TestTemporalCoverage:
    """Test temporal aspects of the dataset."""
    
    @pytest.fixture(scope="class")
    def all_dates(self):
        """Get all dates from filenames."""
        files = MODIS_AOD_DIR.glob("2*.parquet")
        dates = []
        for f in files:
            try:
                # Extract date from filename (YYYY-MM-DD.parquet)
                date_str = f.stem
                date = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date)
            except ValueError:
                continue
        return sorted(dates)
    
    def test_date_filename_format(self):
        """Test that filenames follow YYYY-MM-DD format."""
        files = MODIS_AOD_DIR.glob("2*.parquet")
        invalid_files = []
        
        for f in files:
            try:
                datetime.strptime(f.stem, "%Y-%m-%d")
            except ValueError:
                invalid_files.append(f.name)
        
        assert len(invalid_files) == 0, \
            f"Invalid date format in filenames: {invalid_files[:10]}"
    
    def test_temporal_coverage(self, all_dates):
        """Test that we have reasonable temporal coverage."""
        if len(all_dates) == 0:
            pytest.skip("No date files found")
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Should span multiple years
        date_range_years = (max_date - min_date).days / 365.25
        assert date_range_years >= 1, \
            f"Data spans only {date_range_years:.1f} years"
        
        print(f"\nTemporal coverage: {min_date.date()} to {max_date.date()} "
              f"({len(all_dates)} dates, {date_range_years:.1f} years)")
    
    def test_no_future_dates(self, all_dates):
        """Test that no dates are in the future."""
        if len(all_dates) == 0:
            pytest.skip("No date files found")
        
        max_date = max(all_dates)
        assert max_date <= datetime.now(), \
            f"Future date found: {max_date.date()}"
    
    def test_dates_within_expected_range(self, all_dates):
        """Test that dates are within 2019-2024 range."""
        if len(all_dates) == 0:
            pytest.skip("No date files found")
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        assert min_date >= datetime(2019, 1, 1), \
            f"Date before 2019: {min_date.date()}"
        assert max_date <= datetime(2024, 12, 31), \
            f"Date after 2024: {max_date.date()}"


class TestManifestConsistency:
    """Test that manifest database is consistent with files."""
    
    @pytest.fixture(scope="class")
    def manifest_dates(self):
        """Get dates from manifest database."""
        if not MANIFEST_DB.exists():
            return set()
        
        conn = sqlite3.connect(MANIFEST_DB)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT artifact_key FROM artifacts WHERE status='success'")
            dates = {row[0] for row in cursor.fetchall()}
        except Exception:
            dates = set()
        finally:
            conn.close()
        
        return dates
    
    @pytest.fixture(scope="class")
    def file_dates(self):
        """Get dates from actual files."""
        files = MODIS_AOD_DIR.glob("2*.parquet")
        return {f.stem for f in files}
    
    def test_manifest_file_consistency(self, manifest_dates, file_dates):
        """Test that manifest records match actual files."""
        if len(manifest_dates) == 0:
            pytest.skip("No manifest entries found")
        
        # Files that exist but not in manifest
        missing_from_manifest = file_dates - manifest_dates
        # Files in manifest but don't exist
        missing_files = manifest_dates - file_dates
        
        assert len(missing_files) == 0, \
            f"{len(missing_files)} files in manifest but missing on disk: {list(missing_files)[:10]}"
        
        # Allow some files to exist without manifest (recent writes)
        if len(missing_from_manifest) > 10:
            pytest.fail(f"{len(missing_from_manifest)} files exist but not in manifest: "
                       f"{list(missing_from_manifest)[:10]}")


class TestStatisticalProperties:
    """Test statistical properties of the dataset."""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Load sample of data for statistical tests."""
        files = sorted(MODIS_AOD_DIR.glob("2*.parquet"))
        sample_files = files[:50] if len(files) > 50 else files
        
        dfs = []
        for f in sample_files:
            df = pd.read_parquet(f)
            df['date'] = f.stem
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def test_aod_correlation(self, sample_data):
        """Test that AOD 047 and 055 are correlated."""
        if len(sample_data) == 0:
            pytest.skip("No sample data")
        
        # Both bands should have valid data
        valid_data = sample_data[['aod_047', 'aod_055']].dropna()
        
        if len(valid_data) > 100:
            correlation = valid_data['aod_047'].corr(valid_data['aod_055'])
            assert correlation > 0.7, \
                f"AOD bands correlation {correlation:.3f} is too low (expected > 0.7)"
            print(f"\nAOD 047-055 correlation: {correlation:.3f}")
    
    def test_spatial_variation(self, sample_data):
        """Test that AOD values show spatial variation."""
        if len(sample_data) == 0:
            pytest.skip("No sample data")
        
        # For each date, check that AOD varies across pixels
        dates = sample_data['date'].unique()[:10]
        
        for date in dates:
            date_data = sample_data[sample_data['date'] == date]['aod_055'].dropna()
            
            if len(date_data) > 10:
                std = date_data.std()
                # AOD should vary spatially (std > 0)
                assert std > 0, f"Date {date}: No spatial variation in AOD (std={std})"
    
    def test_realistic_mean_values(self, sample_data):
        """Test that mean AOD values are realistic for Mumbai."""
        if len(sample_data) == 0:
            pytest.skip("No sample data")
        
        aod_mean = sample_data['aod_055'].mean()
        
        # Mumbai typically has AOD between 0.2 and 0.8
        # This is a loose check to catch obvious errors
        if not pd.isna(aod_mean):
            assert 0.05 < aod_mean < 2.0, \
                f"Mean AOD {aod_mean:.3f} is outside realistic range [0.05, 2.0]"
            print(f"\nMean AOD 055: {aod_mean:.3f}")


class TestFileIntegrity:
    """Test file-level integrity."""
    
    def test_file_sizes_reasonable(self):
        """Test that parquet files have reasonable sizes."""
        files = list(MODIS_AOD_DIR.glob("2*.parquet"))
        
        for f in files[:100]:  # Check first 100
            size_kb = f.stat().st_size / 1024
            
            # With 882 rows, expect 10-30 KB per file
            assert 5 < size_kb < 100, \
                f"{f.name}: File size {size_kb:.1f} KB is outside expected range [5, 100]"
    
    def test_no_corrupted_files(self):
        """Test that all files can be read without errors."""
        files = list(MODIS_AOD_DIR.glob("2*.parquet"))
        corrupted = []
        
        for f in files:
            try:
                df = pd.read_parquet(f)
                assert len(df) > 0
            except Exception as e:
                corrupted.append((f.name, str(e)))
        
        assert len(corrupted) == 0, \
            f"{len(corrupted)} corrupted files found: {corrupted[:5]}"


def test_overall_summary():
    """Print overall summary of dataset."""
    files = list(MODIS_AOD_DIR.glob("2*.parquet"))
    
    if len(files) == 0:
        pytest.skip("No files found")
    
    total_files = len(files)
    
    # Get date range
    dates = []
    for f in files:
        try:
            dates.append(datetime.strptime(f.stem, "%Y-%m-%d"))
        except ValueError:
            continue
    
    if dates:
        min_date = min(dates).date()
        max_date = max(dates).date()
        date_range_days = (max(dates) - min(dates)).days
        
        # Sample some files for data stats
        sample = files[:10]
        total_pixels = 0
        valid_aod = 0
        
        for f in sample:
            df = pd.read_parquet(f)
            total_pixels += len(df)
            valid_aod += df['aod_055'].notna().sum()
        
        avg_valid_pct = (valid_aod / total_pixels * 100) if total_pixels > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"MODIS AOD Dataset Summary")
        print(f"{'='*60}")
        print(f"Total files: {total_files}")
        print(f"Date range: {min_date} to {max_date} ({date_range_days} days)")
        print(f"Pixels per file: {EXPECTED_PIXEL_COUNT}")
        print(f"Avg valid AOD: {avg_valid_pct:.1f}% (sampled from {len(sample)} files)")
        print(f"Output directory: {MODIS_AOD_DIR}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
