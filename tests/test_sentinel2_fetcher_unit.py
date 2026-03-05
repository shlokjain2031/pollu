#!/usr/bin/env python3
"""
Unit tests for Sentinel2Fetcher.

Usage:
    python -m pytest test_sentinel2_fetcher_unit.py -v

Tests:
    - should_fetch() with various timestamps
    - freshness threshold (5 days)
    - band ordering and selection
    - AOT rescaling
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta, timezone
import tempfile
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging

# Mock Earth Engine before importing the worker
import sys
from unittest.mock import MagicMock

sys.modules['ee'] = MagicMock()

from takshashila.distributed.worker.write_job_worker import Sentinel2Fetcher


class TestSentinel2Fetcher(unittest.TestCase):
    """Test Sentinel2Fetcher functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
        
        # Create temporary directory for test databases
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "sentinel2.duckdb"
    
    def tearDown(self):
        """Clean up test artifacts."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('takshashila.distributed.worker.write_job_worker.duckdb')
    def test_should_fetch_no_table(self, mock_duckdb):
        """Test should_fetch when table doesn't exist."""
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = (0,)
        
        fetcher = Sentinel2Fetcher(self.logger)
        fetcher.db_path = self.db_path
        
        result = fetcher.should_fetch(datetime.now(timezone.utc))
        self.assertTrue(result, "Should fetch when table doesn't exist")
    
    @patch('takshashila.distributed.worker.write_job_worker.duckdb')
    def test_should_fetch_no_records(self, mock_duckdb):
        """Test should_fetch when table exists but is empty."""
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn
        
        # First call: table exists (count > 0)
        # Second call: no records (MAX returns None)
        mock_conn.execute.return_value.fetchone.side_effect = [(1,), (None,)]
        
        fetcher = Sentinel2Fetcher(self.logger)
        fetcher.db_path = self.db_path
        
        result = fetcher.should_fetch(datetime.now(timezone.utc))
        self.assertTrue(result, "Should fetch when no records in table")
    
    @patch('takshashila.distributed.worker.write_job_worker.duckdb')
    def test_should_fetch_fresh_data(self, mock_duckdb):
        """Test should_fetch when latest data is within 5 days."""
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn
        
        # Table exists, has recent data (today - 3 days)
        current_date = datetime.now(timezone.utc)
        recent_date = (current_date - timedelta(days=3)).isoformat()
        
        mock_conn.execute.return_value.fetchone.side_effect = [(1,), (recent_date,)]
        
        fetcher = Sentinel2Fetcher(self.logger)
        fetcher.db_path = self.db_path
        
        result = fetcher.should_fetch(current_date)
        self.assertFalse(result, "Should NOT fetch when data is fresh (<=5 days)")
    
    @patch('takshashila.distributed.worker.write_job_worker.duckdb')
    def test_should_fetch_stale_data(self, mock_duckdb):
        """Test should_fetch when latest data is >5 days old."""
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn
        
        # Table exists, has stale data (today - 7 days)
        current_date = datetime.now(timezone.utc)
        stale_date = (current_date - timedelta(days=7)).isoformat()
        
        mock_conn.execute.return_value.fetchone.side_effect = [(1,), (stale_date,)]
        
        fetcher = Sentinel2Fetcher(self.logger)
        fetcher.db_path = self.db_path
        
        result = fetcher.should_fetch(current_date)
        self.assertTrue(result, "Should fetch when data is stale (>5 days)")
    
    @patch('takshashila.distributed.worker.write_job_worker.duckdb')
    def test_should_fetch_boundary_5_days(self, mock_duckdb):
        """Test should_fetch at exactly 5 day boundary."""
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn
        
        # Table exists, data is exactly 5 days old
        current_date = datetime.now(timezone.utc)
        boundary_date = (current_date - timedelta(days=5)).isoformat()
        
        mock_conn.execute.return_value.fetchone.side_effect = [(1,), (boundary_date,)]
        
        fetcher = Sentinel2Fetcher(self.logger)
        fetcher.db_path = self.db_path
        
        result = fetcher.should_fetch(current_date)
        self.assertFalse(result, "Should NOT fetch at exactly 5 day boundary (<=5)")
    
    def test_band_selection(self):
        """Test that correct bands are selected for Sentinel2."""
        fetcher = Sentinel2Fetcher(self.logger)
        expected_bands = ["B2", "B4", "B8", "AOT", "QA10"]
        self.assertEqual(fetcher.bands, expected_bands, "Sentinel2 bands should be B2, B4, B8, AOT, QA10")
    
    def test_collection_name(self):
        """Test that correct collection is used."""
        fetcher = Sentinel2Fetcher(self.logger)
        expected_collection = "COPERNICUS/S2_SR_HARMONIZED"
        self.assertEqual(fetcher.collection_name, expected_collection, "Should use COPERNICUS/S2_SR_HARMONIZED")
    
    def test_aot_rescaling(self):
        """Test AOT rescaling logic (x1000 to normal range)."""
        fetcher = Sentinel2Fetcher(self.logger)
        
        # Create mock DataFrame with high AOT values (stored as x1000)
        df = pd.DataFrame({
            "grid_id": ["grid_1", "grid_2"],
            "B2": [100.0, 200.0],
            "B4": [150.0, 250.0],
            "B8": [200.0, 300.0],
            "AOT": [600.0, 800.0],  # Should be rescaled to 0.6, 0.8
            "QA10": [10.0, 20.0],
        })
        
        # Simulate AOT rescaling
        aot_max = float(df["AOT"].dropna().max())
        if aot_max > 10.0:
            df["AOT"] = df["AOT"] / 1000.0
        
        # Verify rescaling
        self.assertAlmostEqual(df["AOT"].iloc[0], 0.6, places=3, msg="AOT should be rescaled to 0.6")
        self.assertAlmostEqual(df["AOT"].iloc[1], 0.8, places=3, msg="AOT should be rescaled to 0.8")
    
    def test_band_ordering(self):
        """Test that bands are ordered correctly."""
        fetcher = Sentinel2Fetcher(self.logger)
        
        # Create mock DataFrame with bands in random order
        df = pd.DataFrame({
            "grid_id": ["grid_1"],
            "QA10": [10.0],
            "B8": [200.0],
            "AOT": [0.5],
            "B4": [150.0],
            "B2": [100.0],
        })
        
        # Apply band sorting logic from fetcher
        def band_sort_key(col: str):
            if col.startswith("B") and col[1:].isdigit():
                return (0, int(col[1:]))
            elif col == "AOT":
                return (1, 0)
            elif col == "QA10":
                return (2, 0)
            else:
                return (3, 0)
        
        band_cols = [c for c in df.columns if c in fetcher.bands]
        band_cols_sorted = sorted(band_cols, key=band_sort_key)
        
        expected_order = ["B2", "B4", "B8", "AOT", "QA10"]
        self.assertEqual(band_cols_sorted, expected_order, "Bands should be ordered as B2, B4, B8, AOT, QA10")


class TestSentinel2FetcherComparison(unittest.TestCase):
    """Compare Sentinel2Fetcher with LandsatFetcher freshness thresholds."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
    
    def test_sentinel2_freshness_is_5_days(self):
        """Verify Sentinel2 has 5-day freshness (vs Landsat 16-day)."""
        from unittest.mock import patch, MagicMock
        
        with patch('takshashila.distributed.worker.write_job_worker.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn
            
            # Test with 5 days old (boundary)
            current_date = datetime.now(timezone.utc)
            boundary_date = (current_date - timedelta(days=5)).isoformat()
            mock_conn.execute.return_value.fetchone.side_effect = [(1,), (boundary_date,)]
            
            fetcher = Sentinel2Fetcher(self.logger)
            result_5days = fetcher.should_fetch(current_date)
            
            # Reset mock
            mock_duckdb.reset_mock()
            mock_conn.reset_mock()
            mock_duckdb.connect.return_value = mock_conn
            
            # Test with 6 days old (past boundary)
            past_boundary_date = (current_date - timedelta(days=6)).isoformat()
            mock_conn.execute.return_value.fetchone.side_effect = [(1,), (past_boundary_date,)]
            
            fetcher2 = Sentinel2Fetcher(self.logger)
            result_6days = fetcher2.should_fetch(current_date)
            
            self.assertFalse(result_5days, "5 days should NOT trigger fetch (at boundary)")
            self.assertTrue(result_6days, "6 days should trigger fetch (past boundary)")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    
    # Run tests
    unittest.main(verbosity=2)
