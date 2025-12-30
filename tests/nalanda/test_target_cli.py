"""Tests for Phase 3: Target CLI and parquet output."""

from pathlib import Path
import tempfile
import shutil

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nalanda.target import LAG_HOURS, TargetAssembler
from nalanda.target_cli import (
    date_range,
    get_aqi_artifact_path,
    get_output_path,
    process_date,
    process_date_worker,
    write_summary_report,
    write_target_parquet,
)


# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
SENSOR_JSON = PROJECT_ROOT / "cache" / "openaq_sensor_ids.json"
GRID_PARQUET = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"
AQI_DIR = PROJECT_ROOT / "artifacts" / "v1" / "aqi"


class TestDateRange:
    """Tests for date range generation."""
    
    def test_single_day(self):
        """Verify single day range."""
        dates = date_range('2021-06-07', '2021-06-07')
        assert dates == ['2021-06-07']
        
    def test_multi_day(self):
        """Verify multi-day range."""
        dates = date_range('2021-06-07', '2021-06-10')
        assert dates == ['2021-06-07', '2021-06-08', '2021-06-09', '2021-06-10']
        
    def test_month_boundary(self):
        """Verify range across month boundary."""
        dates = date_range('2021-06-29', '2021-07-02')
        assert dates == ['2021-06-29', '2021-06-30', '2021-07-01', '2021-07-02']


class TestPathHelpers:
    """Tests for path helper functions."""
    
    def test_aqi_artifact_path(self):
        """Verify AQI artifact path construction."""
        path = get_aqi_artifact_path('2021-06-07')
        assert path.name == '2021-06-07.parquet'
        
    def test_output_path(self):
        """Verify output path construction."""
        path = get_output_path('2021-06-07')
        assert 'training' in str(path)
        assert '2021-06-07' in str(path)
        assert path.name == 'target.parquet'


class TestWriteTargetParquet:
    """Tests for parquet writing."""
    
    def test_write_creates_file(self):
        """Verify parquet file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test' / 'target.parquet'
            
            df = pd.DataFrame({
                'row_idx': [1, 2, 3],
                'timestamp': pd.to_datetime(['2021-06-07 10:00', '2021-06-07 10:00', '2021-06-07 10:00']),
                'pm25_t': [42.5, 43.0, 41.0],
                'target_trust': [1.0, 0.9, 0.8],
                'n_sensors': [1, 1, 1],
            })
            
            metadata = {'radius_m': '250.0', 'trust_kernel': 'epanechnikov'}
            write_target_parquet(df, output_path, metadata)
            
            assert output_path.exists()
            
    def test_write_stores_metadata(self):
        """Verify metadata is stored in parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'target.parquet'
            
            df = pd.DataFrame({
                'row_idx': [1],
                'timestamp': pd.to_datetime(['2021-06-07 10:00']),
                'pm25_t': [42.5],
                'target_trust': [1.0],
                'n_sensors': [1],
            })
            
            metadata = {'radius_m': '250.0', 'trust_kernel': 'epanechnikov', 'custom_key': 'custom_value'}
            write_target_parquet(df, output_path, metadata)
            
            # Read back metadata
            table = pq.read_table(output_path)
            stored_metadata = table.schema.metadata
            
            assert b'radius_m' in stored_metadata
            assert stored_metadata[b'radius_m'] == b'250.0'
            assert b'custom_key' in stored_metadata
            
    def test_write_optimizes_dtypes(self):
        """Verify dtypes are optimized for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'target.parquet'
            
            df = pd.DataFrame({
                'row_idx': [1],
                'timestamp': pd.to_datetime(['2021-06-07 10:00']),
                'pm25_t': [42.5],
                'target_trust': [1.0],
                'n_sensors': [1],
            })
            
            write_target_parquet(df, output_path, {})
            
            # Read back and check schema
            table = pq.read_table(output_path)
            schema = table.schema
            
            # n_sensors should be int8
            assert str(schema.field('n_sensors').type) == 'int8'
            # pm25_t should be float (float32)
            assert 'float' in str(schema.field('pm25_t').type)


class TestProcessDate:
    """Tests for single date processing."""
    
    @pytest.fixture
    def assembler(self):
        """Create target assembler fixture."""
        return TargetAssembler(SENSOR_JSON, GRID_PARQUET)
    
    def test_process_success(self, assembler):
        """Verify successful processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = process_date(
                date='2021-06-07',
                assembler=assembler,
                aqi_dir=AQI_DIR,
                output_dir=Path(tmpdir),
                force=False,
            )
            
            assert result['status'] == 'success'
            assert result['input_rows'] > 0
            assert result['output_rows'] > 0
            assert result['unique_cells'] > 0
            
            # Verify file created
            output_path = Path(tmpdir) / '2021-06-07' / 'target.parquet'
            assert output_path.exists()
            
    def test_process_missing_input(self, assembler):
        """Verify handling of missing AQI artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = process_date(
                date='1999-01-01',  # Non-existent date
                assembler=assembler,
                aqi_dir=AQI_DIR,
                output_dir=Path(tmpdir),
                force=False,
            )
            
            assert result['status'] == 'skipped'
            assert 'not found' in result['error']
            
    def test_process_skip_existing(self, assembler):
        """Verify skipping existing output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run - should succeed
            result1 = process_date(
                date='2021-06-07',
                assembler=assembler,
                aqi_dir=AQI_DIR,
                output_dir=Path(tmpdir),
                force=False,
            )
            assert result1['status'] == 'success'
            
            # Second run - should skip
            result2 = process_date(
                date='2021-06-07',
                assembler=assembler,
                aqi_dir=AQI_DIR,
                output_dir=Path(tmpdir),
                force=False,
            )
            assert result2['status'] == 'skipped'
            assert 'exists' in result2['error']
            
    def test_process_force_overwrite(self, assembler):
        """Verify force overwrite works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run
            result1 = process_date(
                date='2021-06-07',
                assembler=assembler,
                aqi_dir=AQI_DIR,
                output_dir=Path(tmpdir),
                force=False,
            )
            assert result1['status'] == 'success'
            
            # Second run with force
            result2 = process_date(
                date='2021-06-07',
                assembler=assembler,
                aqi_dir=AQI_DIR,
                output_dir=Path(tmpdir),
                force=True,
            )
            assert result2['status'] == 'success'


class TestOutputValidation:
    """Tests for validating output file contents."""
    
    def test_output_schema(self):
        """Verify output file has correct schema including lag columns."""
        output_path = PROJECT_ROOT / 'artifacts' / 'v1' / 'training' / '2021-06-07' / 'target.parquet'
        
        if not output_path.exists():
            pytest.skip('Output file not generated yet')
        
        table = pq.read_table(output_path)
        column_names = [f.name for f in table.schema]
        
        # Core columns
        assert 'row_idx' in column_names
        assert 'timestamp' in column_names
        assert 'pm25_t' in column_names
        assert 'target_trust' in column_names
        assert 'n_sensors' in column_names
        
        # Lag columns
        for lag_h in LAG_HOURS:
            assert f'pm25_t_minus_{lag_h}h' in column_names
        
    def test_output_values_valid(self):
        """Verify output values are valid."""
        output_path = PROJECT_ROOT / 'artifacts' / 'v1' / 'training' / '2021-06-07' / 'target.parquet'
        
        if not output_path.exists():
            pytest.skip('Output file not generated yet')
        
        df = pd.read_parquet(output_path)
        
        # PM2.5 should be positive
        assert (df['pm25_t'] >= 0).all()
        
        # Trust should be positive
        assert (df['target_trust'] > 0).all()
        
        # n_sensors should be at least 1
        assert (df['n_sensors'] >= 1).all()
        
    def test_output_metadata_present(self):
        """Verify metadata is present in output."""
        output_path = PROJECT_ROOT / 'artifacts' / 'v1' / 'training' / '2021-06-07' / 'target.parquet'
        
        if not output_path.exists():
            pytest.skip('Output file not generated yet')
        
        table = pq.read_table(output_path)
        metadata = table.schema.metadata
        
        assert b'radius_m' in metadata
        assert b'trust_kernel' in metadata
        assert b'date' in metadata


class TestParallelProcessing:
    """Tests for parallel processing features."""
    
    def test_worker_function(self):
        """Verify worker function can process a date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = (
                '2021-06-07',
                SENSOR_JSON,
                GRID_PARQUET,
                250.0,
                AQI_DIR,
                Path(tmpdir),
                False,
            )
            
            result = process_date_worker(args)
            
            assert result['status'] == 'success'
            assert result['date'] == '2021-06-07'
            
    def test_summary_report_generation(self):
        """Verify summary report is written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                {'date': '2021-06-07', 'status': 'success', 'input_rows': 100, 'output_rows': 1000, 'unique_cells': 50, 'error': None},
                {'date': '2021-06-08', 'status': 'skipped', 'input_rows': 0, 'output_rows': 0, 'unique_cells': 0, 'error': 'Missing input'},
            ]
            
            summary_path = Path(tmpdir) / 'summary.csv'
            write_summary_report(results, summary_path)
            
            assert summary_path.exists()
            
            # Read back and verify
            df = pd.read_csv(summary_path)
            assert len(df) == 2
            assert 'date' in df.columns
            assert 'status' in df.columns
            assert df.loc[0, 'status'] == 'success'
            assert df.loc[1, 'status'] == 'skipped'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
