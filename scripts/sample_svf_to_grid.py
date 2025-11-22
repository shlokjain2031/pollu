#!/usr/bin/env python3
"""
Sample SVF raster values at grid centroids.

Takes a high-resolution SVF raster and samples it at grid cell centers,
producing a parquet file with grid_id and svf columns.

Usage:
    python scripts/sample_svf_to_grid.py \\
        --svf-raster cache/mumbai_svf_2023.tif \\
        --grid-parquet data/mumbai/grid_30m.parquet \\
        --output cache/mumbai_svf_grid.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.crs import CRS


def sample_raster_at_points(
    raster_path: Path,
    points_gdf: gpd.GeoDataFrame,
    band: int = 1,
) -> pd.Series:
    """
    Sample raster values at point geometries.

    Parameters
    ----------
    raster_path : Path
        Path to raster file
    points_gdf : gpd.GeoDataFrame
        Points to sample (must have geometry column with Point geometries)
    band : int
        Band number to sample (1-indexed)

    Returns
    -------
    pd.Series
        Sampled values at each point
    """
    with rasterio.open(raster_path) as src:
        # Reproject points to raster CRS if needed
        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)

        # Extract coordinates
        coords = [(geom.x, geom.y) for geom in points_gdf.geometry]

        # Sample raster
        values = [val[0] for val in src.sample(coords, indexes=band)]

    return pd.Series(values, index=points_gdf.index)


def sample_svf_to_grid(
    svf_raster_path: Path,
    grid_parquet_path: Path,
    output_path: Path,
) -> gpd.GeoDataFrame:
    """
    Sample SVF raster at grid centroids and save to parquet.

    Parameters
    ----------
    svf_raster_path : Path
        Path to SVF GeoTIFF
    grid_parquet_path : Path
        Path to grid parquet (must have grid_id and geometry columns)
    output_path : Path
        Where to save output parquet

    Returns
    -------
    gpd.GeoDataFrame
        Grid with svf column added
    """
    print(f"Loading grid from {grid_parquet_path}...")
    grid = gpd.read_parquet(grid_parquet_path)

    if "grid_id" not in grid.columns:
        raise ValueError("Grid must have grid_id column")

    # Get centroids for sampling
    print(f"Computing centroids for {len(grid)} grid cells...")
    centroids = grid.copy()
    centroids["geometry"] = grid.geometry.centroid

    # Sample SVF
    print(f"Sampling SVF from {svf_raster_path}...")
    svf_values = sample_raster_at_points(svf_raster_path, centroids)

    # Add to grid
    grid["svf"] = svf_values

    # Report statistics
    print(f"\nSVF Statistics:")
    print(f"  Min:  {grid['svf'].min():.3f}")
    print(f"  Mean: {grid['svf'].mean():.3f}")
    print(f"  Max:  {grid['svf'].max():.3f}")
    print(f"  Null: {grid['svf'].isna().sum()} cells")

    # Identify street canyons (low SVF)
    canyon_threshold = 0.5
    n_canyons = (grid["svf"] < canyon_threshold).sum()
    print(
        f"  Street canyons (SVF < {canyon_threshold}): {n_canyons} cells ({100*n_canyons/len(grid):.1f}%)"
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet (preserves geometry)
    grid.to_parquet(output_path)
    print(f"\n✓ Saved {len(grid)} rows to {output_path}")

    # Also save as CSV for easy inspection (without geometry)
    csv_path = output_path.with_suffix(".csv")
    grid.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"✓ Saved CSV to {csv_path}")

    return grid


def main():
    parser = argparse.ArgumentParser(description="Sample SVF raster at grid centroids")
    parser.add_argument(
        "--svf-raster",
        type=Path,
        required=True,
        help="Path to SVF GeoTIFF",
    )
    parser.add_argument(
        "--grid-parquet",
        type=Path,
        required=True,
        help="Path to grid parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet path",
    )

    args = parser.parse_args()

    # Check inputs exist
    if not args.svf_raster.exists():
        print(f"Error: SVF raster not found: {args.svf_raster}")
        sys.exit(1)

    if not args.grid_parquet.exists():
        print(f"Error: Grid parquet not found: {args.grid_parquet}")
        sys.exit(1)

    # Sample
    grid = sample_svf_to_grid(
        args.svf_raster,
        args.grid_parquet,
        args.output,
    )


if __name__ == "__main__":
    main()
