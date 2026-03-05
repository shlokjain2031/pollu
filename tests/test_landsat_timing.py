#!/usr/bin/env python3
"""Test LandsatFetcher timing on 2026-01-07."""

import sys
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from takshashila.distributed.worker.write_job_worker import LandsatFetcher

def test_landsat_timing():
    """Test LandsatFetcher on 2026-01-07."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    # Use 2026-01-07 date
    test_date = datetime(2026, 1, 7, 12, 0, 0, tzinfo=timezone.utc)
    logger.info(f"Testing LandsatFetcher for {test_date.isoformat()}")
    logger.info("")
    
    # Create fetcher
    fetcher = LandsatFetcher(logger)
    
    # Check freshness
    logger.info("Step 1: Checking if Landsat8 should be fetched...")
    should_fetch = fetcher.should_fetch(test_date)
    logger.info(f"  should_fetch() = {should_fetch}")
    logger.info("")
    
    if not should_fetch:
        logger.warning("  Skipping fetch (data is fresh)")
        logger.info("✓ Test completed (no fetch needed)")
        return
    
    # Fetch data with timing
    logger.info("Step 2: Fetching Landsat8 data...")
    start_time = time.time()
    df = fetcher.fetch(test_date)
    fetch_duration = time.time() - start_time
    
    if df is None:
        logger.warning("  Fetch returned None (no data available)")
        logger.info(f"  Fetch time: {fetch_duration:.2f} seconds")
        logger.info("✓ Test completed (no data available)")
        return
    
    logger.info(f"  Fetch completed in {fetch_duration:.2f} seconds")
    logger.info("")
    
    # Validate
    logger.info("Step 3: Validating fetched data...")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Sample rows:")
    logger.info(df.head(3).to_string())
    
    # Write to DB
    logger.info("")
    logger.info("Step 4: Writing to DuckDB...")
    start_write = time.time()
    success = fetcher.write_to_db(df)
    write_duration = time.time() - start_write
    
    if success:
        logger.info(f"  Write completed in {write_duration:.2f} seconds")
    else:
        logger.error("  Write failed")
        return
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("✓ Test completed successfully!")
    logger.info("="*60)
    logger.info(f"Fetch time:  {fetch_duration:.2f} seconds")
    logger.info(f"Write time:  {write_duration:.2f} seconds")
    logger.info(f"Total time:  {fetch_duration + write_duration:.2f} seconds")
    logger.info("="*60)

if __name__ == "__main__":
    test_landsat_timing()
