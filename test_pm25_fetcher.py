#!/usr/bin/env python3
"""Test PM25Fetcher class."""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from takshashila.distributed.worker.write_job_worker import PM25Fetcher

def test_pm25_fetcher():
    """Test PM25Fetcher with current time."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    # Use current time
    current_date = datetime.now(timezone.utc)
    logger.info(f"Testing PM25Fetcher for {current_date.isoformat()}")
    logger.info("")
    
    # Create fetcher
    fetcher = PM25Fetcher(logger)
    
    # Check freshness
    logger.info("Step 1: Checking if PM2.5 should be fetched...")
    should_fetch = fetcher.should_fetch(current_date)
    logger.info(f"  should_fetch() = {should_fetch}")
    logger.info("")
    
    if not should_fetch:
        logger.warning("  Skipping fetch (data is fresh)")
        logger.info("✓ Test completed (no fetch needed)")
        return
    
    # Fetch data
    logger.info("Step 2: Fetching PM2.5 data from CPCB API...")
    df = fetcher.fetch(current_date)
    
    if df is None:
        logger.warning("  Fetch returned None (no data available)")
        logger.info("✓ Test completed (no data available)")
        return
    
    logger.info("")
    logger.info("Step 3: Validating fetched data...")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  PM2.5 statistics:")
    logger.info(f"    Min:  {df['pm25'].min():.2f} µg/m³")
    logger.info(f"    Max:  {df['pm25'].max():.2f} µg/m³")
    logger.info(f"    Mean: {df['pm25'].mean():.2f} µg/m³")
    logger.info(f"    Std:  {df['pm25'].std():.2f} µg/m³")
    
    logger.info("")
    logger.info("  Sample rows:")
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
    test_pm25_fetcher()
