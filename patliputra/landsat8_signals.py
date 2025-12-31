#!/usr/bin/env python3
"""
Fetch Landsat-8 raw band values for grid cells and write per-date Parquet.

NO intermediate TIFFs are stored. For each date, we:
1. Load the canonical grid
2. Query Landsat-8 TOA image collection
3. Batch-fetch band values (B1-B11) for all grid cells via reduceRegions
4. Write one parquet per date with schema: grid_id, B1, B2, ..., B11
5. Store in artifacts/v1/landsat8/{city}/{date}.parquet

Prerequisites:
  - Earth Engine authenticated (ee.Authenticate() or service account)
  - Grid artifact exists (artifacts/v1/grids/{city}/grid.parquet)
  
Usage:
  python patliputra/landsat8_signals.py \
    --dates-file patliputra/landsat8_image_dates.txt \
    --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
    --city mumbai \
    --project your-ee-project-id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import tempfile
import shutil
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import pandas as pd
import geopandas as gpd

from manifest.landsat_manifest import LandsatManifest


def read_dates(dates_file: Path) -> List[str]:
    """Read dates from text file, one per line."""
    with open(dates_file, "r") as f:
        dates = [line.strip() for line in f if line.strip()]
    return sorted(set(dates))


def get_landsat8_image(date_str: str, bbox: tuple) -> ee.Image:
    """
    Query Landsat-8 Collection 2 Level-1 TOA for given date and bbox.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format
    bbox : tuple
        (west, south, east, north) in EPSG:4326

    Returns
    -------
    ee.Image
        Landsat-8 TOA image with bands B1-B11

    Raises
    ------
    RuntimeError
        If no image found for date
    """
    west, south, east, north = bbox
    geometry = ee.Geometry.Rectangle([west, south, east, north])

    # Query Landsat-8 Collection 2 Level-1 TOA (no cloud filter - include all scenes)
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filterBounds(geometry)
        .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
        .sort("system:time_start")
    )

    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No Landsat-8 image found for {date_str}")

    if count == 1:
        image = collection.first()
        # Capture single scene cloud cover
        scene_cloud_cover = image.get('CLOUD_COVER')
    else:
        # Multiple scenes: mosaic and compute mean cloud cover
        image = collection.mosaic()
        scene_cloud_cover = collection.aggregate_mean('CLOUD_COVER')
    
    # Attach scene-level cloud cover to image for metadata tracking
    image = image.set('scene_cloud_cover', scene_cloud_cover)
    image = image.set('scene_count', count)

    # Select all TOA bands
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
    return image.select(bands)


def fetch_band_values_batched(
    image: ee.Image,
    grid_gdf: gpd.GeoDataFrame,
    batch_size: int = 1000,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Fetch raw band values for all grid cells using batched reduceRegions (concurrent).

    For each grid cell (30m point), samples Landsat bands at 30m resolution.
    Since Landsat is 30m native, this is essentially point sampling.
    Batches are fetched concurrently to improve performance.

    Parameters
    ----------
    image : ee.Image
        Landsat-8 image with bands B1-B11
    grid_gdf : gpd.GeoDataFrame
        Grid with columns: grid_id, geometry (Point)
    batch_size : int
        Number of cells per EE request (default 1000)
    max_workers : int
        Number of concurrent workers (default 10)

    Returns
    -------
    pd.DataFrame
        Columns: grid_id, B1, B2, ..., B11
    """
    print(f"  Fetching band values for {len(grid_gdf):,} cells (batch_size={batch_size}, workers={max_workers})...")
    
    # Ensure grid is in WGS84 for EE
    if grid_gdf.crs and grid_gdf.crs.to_string() != "EPSG:4326":
        grid_wgs84 = grid_gdf.to_crs("EPSG:4326")
    else:
        grid_wgs84 = grid_gdf

    total_batches = (len(grid_wgs84) + batch_size - 1) // batch_size
    
    def fetch_batch(batch_idx: int) -> tuple[int, list[dict]]:
        """Fetch a single batch of grid cells."""
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(grid_wgs84))
        batch_gdf = grid_wgs84.iloc[start_idx:end_idx]

        # Convert to EE FeatureCollection
        features = []
        for idx, row in batch_gdf.iterrows():
            pt = row.geometry
            # Attach image-level angle properties to each feature so they
            # are included in reduceRegions results. Compute solar_zenith as
            # (90 - SUN_ELEVATION). Sensor properties may be missing.
            feat = ee.Feature(ee.Geometry.Point([pt.x, pt.y]))
            feat = feat.set({
                "grid_id": row["grid_id"],
                "solar_azimuth": image.get("SUN_AZIMUTH"),
                "solar_zenith": ee.Number(90).subtract(ee.Number(image.get("SUN_ELEVATION"))),
                "sensor_azimuth": image.get("SENSOR_AZIMUTH"),
                "sensor_zenith": image.get("SENSOR_ZENITH"),
                "scene_cloud_cover": image.get("scene_cloud_cover"),
            })
            features.append(feat)

        fc = ee.FeatureCollection(features)

        # Reduce regions at 30m scale (matches Landsat native resolution)
        reduced = image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=30,  # Landsat native resolution
            tileScale=1,
        )

        # Fetch results with retry/backoff for transient EE failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_list = reduced.getInfo()["features"]
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"EE getInfo failed after {max_retries} attempts: {e}")
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                time.sleep(wait_time)
        
        # Extract properties
        batch_results = []
        for feat in result_list:
            batch_results.append(feat["properties"])
        
        return (batch_idx, batch_results)
    
    # Submit all batches concurrently
    all_results = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        futures = {
            executor.submit(fetch_batch, batch_idx): batch_idx
            for batch_idx in range(total_batches)
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            batch_idx, batch_results = future.result()
            all_results.extend(batch_results)
            completed_count += 1
            
            # Calculate cells processed
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(grid_wgs84))
            
            if completed_count % 10 == 0 or completed_count == total_batches:
                print(f"    Completed {completed_count}/{total_batches} batches ({len(all_results):,}/{len(grid_wgs84):,} cells)")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # EE reduceRegions returns raw band names (B1, B2, ..., B11)
    # Sort bands numerically (B1, B2, ..., B11) to preserve expected order
    band_cols = [c for c in df.columns if c.startswith("B") and c[1:].isdigit()]
    band_cols = sorted(band_cols, key=lambda c: int(c[1:]))

    # Ensure we have grid_id
    if "grid_id" not in df.columns:
        raise RuntimeError("grid_id missing from EE results")

    # Keep grid_id + scene_cloud_cover + angle properties + band columns
    keep_cols = ["grid_id"]
    
    # Add scene cloud cover
    if "scene_cloud_cover" in df.columns:
        keep_cols.append("scene_cloud_cover")
    
    # Add angle properties (solar/sensor geometry)
    angle_cols = ["solar_azimuth", "solar_zenith", "sensor_azimuth", "sensor_zenith"]
    for col in angle_cols:
        if col in df.columns:
            keep_cols.append(col)
    
    # Add bands in numeric order
    keep_cols.extend(band_cols)
    df = df[keep_cols]
    
    # Reindex to match canonical grid order (deterministic row ordering)
    # Merge with original grid to preserve row_idx order
    grid_order = grid_gdf[["grid_id"]].copy()
    grid_order["_order"] = range(len(grid_order))
    df = grid_order.merge(df, on="grid_id", how="left").sort_values("_order").drop(columns=["_order"])
    
    print(f"  ✓ Fetched {len(df):,} cells with {len(band_cols)} bands (reindexed to grid order)")
    return df


def write_artifact_atomic(
    df: pd.DataFrame,
    artifact_path: Path,
    metadata: dict,
    manifest: LandsatManifest,
    date_str: str,
) -> None:
    """
    Write parquet artifact atomically via temp file + rename, then register in manifest.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write
    artifact_path : Path
        Final artifact path
    metadata : dict
        Metadata to write alongside (as {artifact_path}.meta.json)
    manifest : LandsatManifest
        Landsat manifest instance for registration
    date_str : str
        Date (YYYY-MM-DD)
    """
    # Mark producing
    request_id = f"landsat8_{manifest.city}_{date_str}"
    manifest.mark_producing(date_str, request_id)
    
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=artifact_path.parent,
            suffix=".parquet.tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            df.to_parquet(tmp_path, engine="pyarrow", index=False)
        
        # Atomic rename
        shutil.move(str(tmp_path), str(artifact_path))
        
        # Write metadata
        meta_path = artifact_path.with_suffix(".parquet.meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Mark ok in manifest
        manifest.mark_ok(date_str, artifact_path)
        
        print(f"  ✓ Wrote {artifact_path} ({len(df):,} rows)")
        
    except Exception as e:
        # Mark error in manifest
        manifest.mark_error(date_str, str(e))
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Landsat-8 raw band values for grid cells (per-date Parquet)"
    )
    parser.add_argument(
        "--dates-file",
        type=Path,
        default=Path("artifacts/v1/landsat8/mumbai/landsat8_image_dates.txt"),
        help="Path to dates file (one date per line, YYYY-MM-DD)",
    )
    parser.add_argument(
        "--grid-artifact",
        type=Path,
        required=True,
        help="Path to grid artifact parquet (artifacts/v1/grids/{city}/grid.parquet)",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="mumbai",
        help="City name (used in output path)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory for artifacts",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="72.7763,18.8939,72.9797,19.2701",
        help="Bounding box as west,south,east,north in EPSG:4326",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of grid cells per EE request",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of concurrent workers for batch fetching (default: 10)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Earth Engine project ID",
    )

    args = parser.parse_args()

    # Initialize Landsat manifest for this city
    manifest = LandsatManifest(
        city=args.city,
        artifacts_root=args.artifacts_root,
        version="v1"
    )
    print(f"✓ Landsat manifest initialized: {manifest.db_path}")

    # Initialize Earth Engine
    try:
        ee.Initialize(project=args.project)
        print(f"✓ Earth Engine initialized (project: {args.project})")
    except Exception as e:
        print(f"✗ Failed to initialize Earth Engine: {e}")
        print("  Run: python scripts/auth_earth_engine.py")
        sys.exit(1)

    # Parse bbox
    bbox_parts = [float(x.strip()) for x in args.bbox.split(",")]
    if len(bbox_parts) != 4:
        print("Error: bbox must be west,south,east,north")
        sys.exit(1)
    bbox = tuple(bbox_parts)

    # Load grid artifact
    if not args.grid_artifact.exists():
        print(f"✗ Grid artifact not found: {args.grid_artifact}")
        sys.exit(1)

    print(f"Loading grid from {args.grid_artifact}...")
    grid_gdf = gpd.read_parquet(args.grid_artifact)
    print(f"  ✓ Loaded {len(grid_gdf):,} grid cells")
    
    # Check required columns
    if "grid_id" not in grid_gdf.columns:
        print("✗ Grid must have 'grid_id' column")
        sys.exit(1)
    if "geometry" not in grid_gdf.columns:
        print("✗ Grid must have 'geometry' column")
        sys.exit(1)

    # Read dates
    if not args.dates_file.exists():
        print(f"✗ Dates file not found: {args.dates_file}")
        sys.exit(1)

    dates = read_dates(args.dates_file)
    print(f"✓ Found {len(dates)} dates to process")

    # Prepare output directory
    output_dir = args.artifacts_root / "v1" / "landsat8" / args.city
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Track progress
    failed_dates = []
    processed_count = 0

    # Process each date
    for idx, date_str in enumerate(dates, start=1):
        print(f"\n[{idx}/{len(dates)}] Processing {date_str}")

        # Use canonical artifacts layout for Landsat: artifacts/v1/landsat8/{city}/{date}.parquet
        artifact_path = args.artifacts_root / "v1" / "landsat8" / args.city / f"{date_str}.parquet"

        # Skip if already exists
        if artifact_path.exists():
            print(f"  ✓ Already exists: {artifact_path}")
            processed_count += 1
            continue

        try:
            # Get Landsat image
            print(f"  Querying Landsat-8 collection...")
            image = get_landsat8_image(date_str, bbox)
            print(f"  ✓ Image found")

            # Fetch band values for all grid cells
            df = fetch_band_values_batched(
                image=image,
                grid_gdf=grid_gdf,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
            )

            # Prepare metadata
            scene_cloud_cover_val = (
                df['scene_cloud_cover'].iloc[0] 
                if 'scene_cloud_cover' in df.columns and len(df) > 0 
                else None
            )
            metadata = {
                "date": date_str,
                "city": args.city,
                "source": "LANDSAT/LC08/C02/T1_TOA",
                "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
                "scene_cloud_cover": scene_cloud_cover_val,
                "grid_cells": len(df),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "bbox": bbox,
            }

            # Write artifact
            write_artifact_atomic(df, artifact_path, metadata, manifest, date_str)
            processed_count += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_dates.append((date_str, str(e)))
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Processed {processed_count}/{len(dates)} dates")
    print(f"  - Successful: {processed_count}")
    print(f"  - Failed: {len(failed_dates)}")
    print(f"Output: {output_dir}")

    if failed_dates:
        print(f"\n⚠ Failed dates:")
        for date_str, error in failed_dates:
            print(f"  - {date_str}: {error[:80]}")
        
        # Write failed log
        failed_log = output_dir / "failed_dates.log"
        with open(failed_log, "w") as f:
            f.write("# Failed Landsat-8 fetches\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            for date_str, error in failed_dates:
                f.write(f"{date_str} | {error}\n")
        print(f"\nFailed log: {failed_log}")


if __name__ == "__main__":
    main()
