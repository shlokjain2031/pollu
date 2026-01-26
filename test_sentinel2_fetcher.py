#!/usr/bin/env python3
"""
Test script for Sentinel2Fetcher.

Usage:
    python test_sentinel2_fetcher.py

Steps:
    1. Fetch and populate Sentinel2 data from 2025-12-06 (creates table)
    2. Test with 2025-12-12 (6 days later, should fetch since freshness is 5 days)
    3. Test with 2025-12-11 (5 days later, should NOT fetch, freshness boundary)
"""

import subprocess
import sys

if __name__ == "__main__":
    # Step 1: Fetch from 2025-12-06 to create the table and populate data
    print("=" * 80)
    print("Step 1: Fetching Sentinel2 data from 2025-12-06 (creates table)")
    print("=" * 80)
    fetch_date = "2025-12-06 12:00:00"
    
    cmd_fetch = [
        sys.executable,
        "-m",
        "takshashila.distributed.worker.write_job_worker",
        "--current-date",
        fetch_date,
    ]
    
    print(f"Running: {' '.join(cmd_fetch)}")
    print()
    
    result = subprocess.run(cmd_fetch, cwd="/Users/shlokjain/pollu")
    if result.returncode != 0:
        print(f"ERROR: Failed to fetch Sentinel2 data from {fetch_date}")
        sys.exit(1)
    
    # Step 2: Test with 2025-12-12 (6 days after initial fetch, should fetch)
    print("\n" + "=" * 80)
    print("Step 2: Testing with 2025-12-12 (6 days after initial fetch - should FETCH)")
    print("=" * 80)
    test_date_fetch = "2025-12-12 12:00:00"
    
    cmd_test_fetch = [
        sys.executable,
        "-m",
        "takshashila.distributed.worker.write_job_worker",
        "--current-date",
        test_date_fetch,
    ]
    
    print(f"Running: {' '.join(cmd_test_fetch)}")
    print()
    
    result = subprocess.run(cmd_test_fetch, cwd="/Users/shlokjain/pollu")
    if result.returncode != 0:
        print(f"ERROR: Failed to fetch Sentinel2 data from {test_date_fetch}")
        sys.exit(1)
    
    # Step 3: Test with 2025-12-11 (5 days after initial fetch, should NOT fetch)
    print("\n" + "=" * 80)
    print("Step 3: Testing with 2025-12-11 (5 days after initial fetch - should SKIP)")
    print("=" * 80)
    test_date_skip = "2025-12-11 12:00:00"
    
    cmd_test_skip = [
        sys.executable,
        "-m",
        "takshashila.distributed.worker.write_job_worker",
        "--current-date",
        test_date_skip,
    ]
    
    print(f"Running: {' '.join(cmd_test_skip)}")
    print("Expected: 'Sentinel2 record is 5 days old (<=5), skip fetch'")
    print()
    
    result = subprocess.run(cmd_test_skip, cwd="/Users/shlokjain/pollu")
    sys.exit(result.returncode)
