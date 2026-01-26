#!/usr/bin/env python3
"""
Test script for LandsatFetcher.

Usage:
    python test_landsat_fetcher.py

Steps:
    1. Fetch and populate Landsat data from 2025-12-06 (creates table)
    2. Test with 2025-12-22 (16 days later)
"""

import subprocess
import sys

if __name__ == "__main__":
    # Step 1: Fetch from 2025-12-06 to create the table and populate data
    print("=" * 80)
    print("Step 1: Fetching Landsat data from 2025-12-06 (creates table)")
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
    
    result = subprocess.run(cmd_fetch)
    if result.returncode != 0:
        print(f"ERROR: Failed to fetch Landsat data from {fetch_date}")
        sys.exit(1)
    
    # Step 2: Test with 2025-12-22 (16 days after initial fetch)
    print("\n" + "=" * 80)
    print("Step 2: Testing with 2025-12-22 (16 days after initial fetch)")
    print("=" * 80)
    test_date = "2025-12-22 12:00:00"
    
    cmd_test = [
        sys.executable,
        "-m",
        "takshashila.distributed.worker.write_job_worker",
        "--current-date",
        test_date,
    ]
    
    print(f"Running: {' '.join(cmd_test)}")
    print()
    
    result = subprocess.run(cmd_test, cwd="/Users/shlokjain/pollu")
    sys.exit(result.returncode)

