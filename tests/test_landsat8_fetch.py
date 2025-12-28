#!/usr/bin/env python3
"""
Test Landsat-8 fetch for multiple dates with a small grid subset.

This test validates:
1. Grid loading and filtering to small subset
2. EE image query for multiple dates
3. Batch fetching with reduceRegions
4. Parquet write with atomic rename
5. Manifest registration

Produces persistent artifacts for inspection.

Usage:
  python tests/test_landsat8_fetch.py \
    --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
    --test-dates 2024-01-15 2024-01-18 2024-02-10 \
    --n-cells 100 \
    --output-dir test_output/landsat8 \
    --project fast-archive-465917-m0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import json

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ee
import geopandas as gpd
import pandas as pd

from patliputra.landsat8_signals import (
    get_landsat8_image,
    fetch_band_values_batched,
    write_artifact_atomic,
)
from manifest.db import Manifest


def test_landsat8_fetch(
    grid_artifact: Path,
    test_dates: List[str],
    n_cells: int = 100,
    output_dir: Path = None,
    project: str = None,
    batch_size: int = 1000,
    max_workers: int = 10,
):
    """Test Landsat fetch with small grid subset for multiple dates."""
    
    print(f"{'='*60}")
    print(f"Landsat-8 Multi-Date Fetch Test")
    print(f"{'='*60}")
    print(f"Dates: {', '.join(test_dates)}")
    print(f"Grid: {grid_artifact}")
    print(f"Cells: {n_cells}")
    print(f"Output: {output_dir}")
    print()
    
    # Initialize EE
    try:
        ee.Initialize(project=project)
        print(f"✓ Earth Engine initialized (project: {project})")
    except Exception as e:
        print(f"✗ EE initialization failed: {e}")
        print("  Run: python scripts/auth_earth_engine.py")
        sys.exit(1)
    
    # Load grid and subset
    print(f"\nLoading grid from {grid_artifact}...")
    grid_gdf = gpd.read_parquet(grid_artifact)
    full_grid_count = len(grid_gdf)
    print(f"  ✓ Loaded {full_grid_count:,} total cells")
    
    # Take first N cells
    grid_subset = grid_gdf.head(n_cells).copy()
    print(f"  ✓ Using first {len(grid_subset)} cells for test")
    
    # Mumbai bbox (for EE query)
    bbox = (72.7763, 18.8939, 72.9797, 19.2701)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize manifest
    manifest_db = output_dir / "test_manifest.db"
    manifest = Manifest(db_path=manifest_db, artifacts_root=output_dir)
    print(f"✓ Test manifest: {manifest_db}")
    
    # Track results
    successful_dates = []
    failed_dates = []
    
    # Process each date
    for idx, test_date in enumerate(test_dates, start=1):
        print(f"\n{'─'*60}")
        print(f"[{idx}/{len(test_dates)}] Processing {test_date}")
        print(f"{'─'*60}")
        
        artifact_path = output_dir / f"{test_date}.parquet"
        
        # Skip if already exists
        if artifact_path.exists():
            print(f"  ✓ Already exists: {artifact_path}")
            successful_dates.append(test_date)
            continue
        
        try:
            # Get Landsat image
            print(f"  Querying Landsat-8 collection...")
            image = get_landsat8_image(test_date, bbox)
            print(f"  ✓ Image found")
            
            # Fetch band values
            print(f"  Fetching band values (batch_size={batch_size}, workers={max_workers})...")
            df = fetch_band_values_batched(image, grid_subset, batch_size=batch_size, max_workers=max_workers)
            print(f"  ✓ Fetched {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            
            # Show sample
            print(f"\n  Sample data (first 2 rows):")
            print(df.head(2).to_string(index=False))
            
            # Prepare metadata
            metadata = {
                "date": test_date,
                "city": "mumbai_test",
                "source": "LANDSAT/LC08/C02/T1_TOA",
                "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
                "grid_cells": len(df),
                "grid_cells_requested": n_cells,
                "full_grid_cells": full_grid_count,
                "test_mode": True,
                "bbox": bbox,
            }
            
            # Write artifact
            print(f"\n  Writing artifact to {artifact_path}...")
            write_artifact_atomic(
                df=df,
                artifact_path=artifact_path,
                metadata=metadata,
                manifest=manifest,
                date_str=test_date,
                city="mumbai_test",
            )
            
            # Verify
            assert artifact_path.exists(), "Artifact file not created"
            meta_path = artifact_path.with_suffix(".parquet.meta.json")
            assert meta_path.exists(), "Metadata file not created"
            
            # Read back and validate row count
            df_read = pd.read_parquet(artifact_path)
            expected_rows = len(grid_subset)
            if expected_rows >= full_grid_count:
                expected_rows = full_grid_count
            
            assert len(df_read) == expected_rows, f"Row count mismatch: expected {expected_rows}, got {len(df_read)}"
            assert list(df_read.columns) == list(df.columns), "Column mismatch"
            
            print(f"  ✓ Verified: {len(df_read)} rows (expected {expected_rows})")
            
            # Show file size
            file_size_mb = artifact_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ File size: {file_size_mb:.2f} MB")
            
            successful_dates.append(test_date)
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_dates.append((test_date, str(e)))
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total dates: {len(test_dates)}")
    print(f"  ✓ Successful: {len(successful_dates)}")
    print(f"  ✗ Failed: {len(failed_dates)}")
    
    if successful_dates:
        print(f"\n✓ Successful dates:")
        for date in successful_dates:
            artifact_path = output_dir / f"{date}.parquet"
            size_mb = artifact_path.stat().st_size / (1024 * 1024)
            print(f"  - {date}: {artifact_path} ({size_mb:.2f} MB)")
    
    if failed_dates:
        print(f"\n✗ Failed dates:")
        for date, error in failed_dates:
            print(f"  - {date}: {error[:80]}")
    
    print(f"\n{'='*60}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Manifest DB: {manifest_db}")
    print(f"\nInspect artifacts:")
    print(f"  ls -lh {output_dir}/*.parquet")
    print(f"  cat {output_dir}/*.meta.json")
    print(f"{'='*60}")
    
    if failed_dates:
        sys.exit(1)
    

def main():
    parser = argparse.ArgumentParser(description="Test Landsat-8 fetch with small grid for multiple dates")
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        required=True,
        help="Path to grid artifact",
    )
    parser.add_argument(
        "--test-dates",
        type=str,
        nargs="+",
        default=["2024-01-15", "2024-01-18"],
        help="Dates to test (YYYY-MM-DD), space-separated (default: 2024-01-15 2024-01-18)",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=100,
        help="Number of grid cells to test (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size (number of grid cells per EE reduceRegions request)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of concurrent workers for batch fetching (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_output/landsat8"),
        help="Output directory for test artifacts (default: test_output/landsat8)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="GEE project ID",
    )
    
    args = parser.parse_args()
    
    test_landsat8_fetch(
        grid_artifact=args.grid_artifact,
        test_dates=args.test_dates,
        n_cells=args.n_cells,
        output_dir=args.output_dir,
        project=args.project,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
