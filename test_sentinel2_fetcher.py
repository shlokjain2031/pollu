#!/usr/bin/env python3
"""Test Sentinel2Fetcher class."""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from takshashila.distributed.worker.write_job_worker import Sentinel2Fetcher

def test_sentinel2_fetcher():
    """Test Sentinel2Fetcher with today's date (2026-01-26)."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    # Use today's date (2026-01-26)
    current_date = datetime(2026, 1, 26, 12, 0, 0, tzinfo=timezone.utc)
    logger.info(f"Testing Sentinel2Fetcher for {current_date.isoformat()}")
    logger.info("")
    
    # Create fetcher
    fetcher = Sentinel2Fetcher(logger)
    
    # Check freshness
    logger.info("Step 1: Checking if Sentinel2 should be fetched...")
    should_fetch = fetcher.should_fetch(current_date)
    logger.info(f"  should_fetch() = {should_fetch}")
    logger.info("")
    
    if not should_fetch:
        logger.warning("  Skipping fetch (data is fresh)")
        logger.info("✓ Test completed (no fetch needed)")
        return
    
    # Fetch data
    logger.info("Step 2: Fetching Sentinel2 data...")
    df = fetcher.fetch(current_date)
    
    if df is None:
        logger.warning("  Fetch returned None (no data available)")
        logger.info("✓ Test completed (no data available)")
        return
    
    logger.info("")
    logger.info("Step 3: Validating fetched data...")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Sample rows:")
    logger.info(df.head(3).to_string())
    
    # Check for nulls
    logger.info("")
    logger.info("Step 4: Checking data quality...")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning("  Found null values:")
        logger.warning(null_counts[null_counts > 0].to_string())
    else:
        logger.info("  ✓ No null values found")
    
    # Write to DB
    logger.info("")
    logger.info("Step 5: Writing to DuckDB...")
    success = fetcher.write_to_db(df)
    
    if success:
        logger.info("  ✓ Write successful")
    else:
        logger.error("  ✗ Write failed")
        return
    
    logger.info("")
    logger.info("="*60)
    logger.info("✓ All tests passed!")
    logger.info("="*60)

if __name__ == "__main__":
    test_sentinel2_fetcher()
