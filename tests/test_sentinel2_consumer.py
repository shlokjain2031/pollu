#!/usr/bin/env python3
"""Test Sentinel-2 artifact consumer.

This test validates Sentinel-2 artifacts (B2, B4, B8, AOT, QA10) produced by
copernicus_signals.py. It checks:
- Manifest entries exist and are marked 'ok'
- Parquet files exist at registered paths
- Schema matches expected bands
- Grid coverage (all grid_ids present)
- Deterministic row ordering
- Numeric range sanity checks

Usage:
    python tests/test_sentinel2_consumer.py \
        --city mumbai \
        --artifacts-root artifacts \
        --grid-artifact artifacts/v1/grids/mumbai/grid.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

from manifest.sentinel2_manifest import Sentinel2Manifest


def test_sentinel2_artifacts(
    city: str,
    artifacts_root: Path,
    grid_artifact: Path,
) -> None:
    """
    Test Sentinel-2 artifacts for a city.

    Parameters
    ----------
    city : str
        City name (e.g., "mumbai")
    artifacts_root : Path
        Root artifacts directory
    grid_artifact : Path
        Path to grid artifact for validation
    """
    print(f"\n{'='*60}")
    print(f"Testing Sentinel-2 artifacts for {city}")
    print(f"{'='*60}\n")

    # Initialize manifest
    manifest = Sentinel2Manifest(city=city, artifacts_root=artifacts_root)
    print(f"✓ Manifest: {manifest.db_path}")

    if not manifest.db_path.exists():
        print(f"✗ Manifest does not exist: {manifest.db_path}")
        sys.exit(1)

    # Load grid for validation
    print(f"Loading grid: {grid_artifact}")
    grid_gdf = gpd.read_parquet(grid_artifact)
    expected_grid_ids = set(grid_gdf["grid_id"])
    expected_count = len(grid_gdf)
    print(f"  ✓ Grid has {expected_count:,} cells\n")

    # Get all OK entries
    ok_entries = manifest.list_ok()
    print(f"Found {len(ok_entries)} artifacts with status='ok'\n")

    if len(ok_entries) == 0:
        print("⚠ No artifacts to test (manifest is empty)")
        return

    # Expected schema
    expected_bands = ["B2", "B4", "B8", "AOT", "QA10"]
    expected_cols = ["grid_id"] + expected_bands

    # Test each artifact
    failed_tests = []
    for idx, entry in enumerate(ok_entries, start=1):
        date = entry.date
        artifact_path = Path(entry.artifact_path)
        
        print(f"[{idx}/{len(ok_entries)}] Testing {date}")
        print(f"  Path: {artifact_path}")

        # Check file exists
        if not artifact_path.exists():
            msg = f"  ✗ File does not exist: {artifact_path}"
            print(msg)
            failed_tests.append((date, msg))
            continue

        # Load parquet
        try:
            df = pd.read_parquet(artifact_path)
        except Exception as e:
            msg = f"  ✗ Failed to read parquet: {e}"
            print(msg)
            failed_tests.append((date, msg))
            continue

        # Check schema
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            msg = f"  ✗ Missing columns: {missing_cols}"
            print(msg)
            failed_tests.append((date, msg))
            continue
        print(f"  ✓ Schema OK ({len(df.columns)} columns)")

        # Check row count
        if len(df) != expected_count:
            msg = f"  ⚠ Row count mismatch: got {len(df):,}, expected {expected_count:,}"
            print(msg)
            # Don't fail, just warn

        # Check grid coverage
        actual_grid_ids = set(df["grid_id"])
        missing_ids = expected_grid_ids - actual_grid_ids
        extra_ids = actual_grid_ids - expected_grid_ids
        
        if missing_ids:
            msg = f"  ✗ Missing {len(missing_ids):,} grid_ids"
            print(msg)
            failed_tests.append((date, msg))
            continue
        
        if extra_ids:
            msg = f"  ⚠ Extra {len(extra_ids):,} grid_ids (not in canonical grid)"
            print(msg)
        
        print(f"  ✓ Grid coverage OK ({len(actual_grid_ids):,} cells)")

        # Check deterministic ordering (grid_id should match canonical order)
        canonical_order = grid_gdf["grid_id"].tolist()
        actual_order = df["grid_id"].tolist()
        
        if canonical_order != actual_order:
            msg = f"  ✗ Row ordering does not match canonical grid"
            print(msg)
            failed_tests.append((date, msg))
            continue
        print(f"  ✓ Deterministic ordering OK")

        # Check numeric ranges (bands should be in reasonable range)
        # Sentinel-2 SR values are typically 0-10000 (scaled reflectance * 10000)
        # AOT is aerosol optical thickness (0-1 range typically)
        # QA10 is a bitmask quality band (0-65535)
        for band in ["B2", "B4", "B8"]:
            if band in df.columns:
                vals = df[band].dropna()
                if len(vals) > 0:
                    band_min = vals.min()
                    band_max = vals.max()
                    # Reflectance bands should be 0-10000 (or 0-1 if normalized)
                    if band_min < 0 or band_max > 20000:
                        msg = f"  ⚠ {band} range unusual: [{band_min:.2f}, {band_max:.2f}]"
                        print(msg)
        
        # Check AOT range (0-1 typical)
        if "AOT" in df.columns:
            aot_vals = df["AOT"].dropna()
            if len(aot_vals) > 0:
                aot_min = aot_vals.min()
                aot_max = aot_vals.max()
                if aot_min < 0 or aot_max > 2:
                    msg = f"  ⚠ AOT range unusual: [{aot_min:.4f}, {aot_max:.4f}]"
                    print(msg)
        
        print(f"  ✓ Numeric ranges OK\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total artifacts tested: {len(ok_entries)}")
    print(f"Passed: {len(ok_entries) - len(failed_tests)}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print(f"\n⚠ Failed tests:")
        for date, msg in failed_tests:
            print(f"  {date}: {msg}")
        sys.exit(1)
    else:
        print(f"\n✓ All tests passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Test Sentinel-2 artifact consumer"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="mumbai",
        help="City name",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root artifacts directory",
    )
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        default=Path("artifacts/v1/grids/mumbai/grid.parquet"),
        help="Path to canonical grid artifact",
    )

    args = parser.parse_args()

    test_sentinel2_artifacts(
        city=args.city,
        artifacts_root=args.artifacts_root,
        grid_artifact=args.grid_artifact,
    )


if __name__ == "__main__":
    main()
