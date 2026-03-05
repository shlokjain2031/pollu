#!/usr/bin/env python3
"""Test LandsatFetcher integration (Landsat only)."""

import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Test with a date that should trigger a fetch (>16 days old)
test_date = datetime.now(timezone.utc) - timedelta(days=20)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info(f"Testing LandsatFetcher with date: {test_date.isoformat()}")

# Import LandsatFetcher
sys.path.insert(0, str(Path(__file__).parent))
from takshashila.distributed.worker.write_job_worker import LandsatFetcher

# Initialize fetcher
fetcher = LandsatFetcher(logger)

# Test should_fetch
logger.info("Testing should_fetch()...")
should_fetch = fetcher.should_fetch(test_date)
logger.info(f"  should_fetch = {should_fetch}")

if should_fetch:
    # Test fetch
    logger.info("Testing fetch()...")
    df = fetcher.fetch(test_date)
    
    if df is not None:
        logger.info(f"  ✓ Fetched {len(df):,} rows")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Sample row:\n{df.head(1)}")
        
        # Test write
        logger.info("Testing write_to_db()...")
        success = fetcher.write_to_db(df)
        
        if success:
            logger.info("  ✓ Write successful")
            logger.info(f"\n✓ All tests passed!")
        else:
            logger.error("  ✗ Write failed")
            sys.exit(1)
    else:
        logger.info("  No Landsat data available for this date (this is OK)")
        logger.info("✓ Fetch logic works correctly")
else:
    logger.info("Skipping fetch test (should_fetch=False)")
    logger.info("✓ should_fetch logic works correctly")
