"""Create a 30 m centroid grid for Mumbai wards and save as GeoParquet.

Usage:
    python scripts/create_grid_mumbai.py

This script defaults to reading `resources/mumbai_wards.geojson` and writing
`data/mumbai/grid_30m.parquet` in the project root. Adjust paths with the
command-line options if needed.

Output includes:
- grid_30m.parquet: Full GeoParquet with geometry (for GIS workflows)
- grid_30m_nd.parquet: Lightweight non-geometry parquet (for ML/numeric ops)
- transform.json: Grid spatial parameters (origin, resolution, dimensions)
- grid_manifest.json: Provenance metadata (timestamp, git SHA, row count)
"""

from pathlib import Path
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

# Optional H3 import with graceful fallback
try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
    print("WARNING: h3-py not installed. h3_cell column will be None. Install with: pip install h3")


def create_grid(
    boundary_path: Path,
    out_path: Path,
    target_crs: str = "EPSG:32643",
    res: float = 30.0,
    max_cells: int = 3_000_000,
):
    """Create 30m grid with enhanced schema and metadata sidecars.
    
    Generates a regular 30m centroid grid clipped to city boundary with:
    - row_idx: Contiguous 0..N-1 index for clipped grid
    - grid_id: Zero-padded string identifier "g00000000"
    - x_m, y_m: Projected coordinates (float32, EPSG:32643)
    - centroid_lon, centroid_lat: WGS84 coordinates (float64)
    - col_idx, row_idx_grid: Grid indices in full pre-clipped lattice
    - h3_cell: H3 hexagon index at resolution 11 (if h3 available)
    - geometry: Shapely Point (projected, for GeoParquet)
    
    Also writes sidecar files:
    - transform.json: Grid spatial parameters
    - grid_manifest.json: Provenance metadata
    - grid_30m_nd.parquet: Lightweight non-geometry version
    """
    print(f"Reading boundary from {boundary_path}...")
    gdf = gpd.read_file(boundary_path)
    city_proj = gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = city_proj.total_bounds
    
    print(f"City bounds (EPSG:32643): x=[{minx:.1f}, {maxx:.1f}], y=[{miny:.1f}, {maxy:.1f}]")

    # Generate full grid coordinates (before clipping)
    # IMPORTANT: Use FIXED origin to match legacy grid (pre-boundary-clipping)
    # This ensures coordinates align with existing Landsat/signal cache files
    # Legacy origin detected from old grid: X=266229.294901, Y=2090522.466401
    origin_x = 266229.294901
    origin_y = 2090522.466401
    
    print(f"Using FIXED origin for backward compatibility: x={origin_x:.6f}, y={origin_y:.6f}")
    
    xs = np.arange(origin_x, maxx, res)
    ys = np.arange(origin_y, maxy, res)
    nx = len(xs)
    ny = len(ys)
    
    estimated = nx * ny
    print(f"Full grid dimensions: {nx} cols × {ny} rows = {estimated:,} cells")
    
    if estimated > max_cells:
        raise RuntimeError(
            f"Estimated grid cells ({estimated:,}) > max_cells ({max_cells:,}). Abort to avoid OOM"
        )

    # Create meshgrid and track original grid indices
    # xx[i, j] corresponds to col j, row i
    xx, yy = np.meshgrid(xs, ys)
    col_indices_2d, row_indices_2d = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Flatten to 1D (row-major order: bottom-to-top, left-to-right)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    col_idx_flat = col_indices_2d.ravel()
    row_idx_grid_flat = row_indices_2d.ravel()
    
    print(f"Creating {len(xx_flat):,} points before clipping...")
    pts = [Point(x, y) for x, y in zip(xx_flat, yy_flat)]

    grid_gdf = gpd.GeoDataFrame(
        {
            "x_m": xx_flat,      # Keep as float64 for backward compatibility
            "y_m": yy_flat,      # Keep as float64 for backward compatibility
            "col_idx": col_idx_flat,
            "row_idx_grid": row_idx_grid_flat
        },
        geometry=pts,
        crs=target_crs
    )

    # Clip to boundary union
    print("Clipping to city boundary...")
    city_union = city_proj.union_all()
    mask = grid_gdf.geometry.within(city_union)
    grid_gdf = grid_gdf[mask].reset_index(drop=True)
    
    n_points = len(grid_gdf)
    print(f"After clipping: {n_points:,} points retained ({100*n_points/estimated:.1f}% of full grid)")
    
    # Assign contiguous row_idx (0..N-1) for clipped grid
    # TODO: neighbor precompute will use row_idx to build spatial index and
    # x_m, y_m for KDTree queries. Ensure these columns are present and correct.
    grid_gdf['row_idx'] = np.arange(n_points, dtype=np.int32)
    
    # Create zero-padded grid_id string: "g00000000"
    # Purpose: Human-readable identifier for API responses, debugging, joins
    grid_gdf['grid_id'] = grid_gdf['row_idx'].apply(lambda i: f"g{i:08d}")
    
    # Keep coordinates as float64 for backward compatibility with legacy grid
    # (float32 precision loss prevents exact coordinate matching with old parquet files)
    # Cast indices to int32 to save space
    grid_gdf['col_idx'] = grid_gdf['col_idx'].astype('int32')
    grid_gdf['row_idx_grid'] = grid_gdf['row_idx_grid'].astype('int32')
    
    # Transform to WGS84 for centroid_lon, centroid_lat
    # Purpose: Geographic coordinates needed for H3 indexing, distance calculations,
    # and API responses (users query by lat/lon, not UTM)
    print("Transforming to WGS84 for centroid_lon/centroid_lat...")
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    lon_flat, lat_flat = transformer.transform(grid_gdf['x_m'].values, grid_gdf['y_m'].values)
    grid_gdf['centroid_lon'] = lon_flat.astype('float64')
    grid_gdf['centroid_lat'] = lat_flat.astype('float64')
    
    # Compute H3 cell index at resolution 11 (~20m edge, matches 30m grid)
    # Purpose: Fast spatial queries, regional indexing, future vector joins
    if HAS_H3:
        print("Computing H3 cells (resolution 11)...")
        # Use h3.latlng_to_cell (new API) instead of deprecated geo_to_h3
        grid_gdf['h3_cell'] = grid_gdf.apply(
            lambda row: h3.latlng_to_cell(row['centroid_lat'], row['centroid_lon'], 11),
            axis=1
        )
    else:
        print("Skipping H3 (library not available)")
        grid_gdf['h3_cell'] = None
    
    # Reorder columns for clarity
    column_order = [
        'row_idx', 'grid_id', 'x_m', 'y_m', 'centroid_lon', 'centroid_lat',
        'col_idx', 'row_idx_grid', 'h3_cell', 'geometry'
    ]
    grid_gdf = grid_gdf[column_order]
    
    # Write main GeoParquet with geometry
    print(f"Writing GeoParquet to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_gdf.to_parquet(out_path, index=False)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved {n_points:,} points ({file_size_mb:.2f} MB)")
    
    # Write lightweight non-geometry Parquet for fast numeric access
    # Purpose: ML/numeric pipelines don't need geometry overhead; this file
    # loads 10x faster for neighbor precompute, spatial aggregates, etc.
    nd_path = out_path.parent / f"{out_path.stem}_nd.parquet"
    print(f"Writing non-geometry parquet to {nd_path}...")
    grid_nd = grid_gdf.drop(columns=['geometry'])
    grid_nd.to_parquet(nd_path, index=False)
    nd_size_mb = nd_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved {n_points:,} points ({nd_size_mb:.2f} MB, no geometry)")
    
    # Compute grid bounding boxes for transform.json
    grid_bbox_proj = (
        float(grid_gdf['x_m'].min()),
        float(grid_gdf['y_m'].min()),
        float(grid_gdf['x_m'].max()),
        float(grid_gdf['y_m'].max())
    )
    grid_bbox_wgs84 = (
        float(grid_gdf['centroid_lon'].min()),
        float(grid_gdf['centroid_lat'].min()),
        float(grid_gdf['centroid_lon'].max()),
        float(grid_gdf['centroid_lat'].max())
    )
    
    # Write transform.json
    # Purpose: Documents grid spatial parameters for downstream raster alignment,
    # tiling, and coordinate transformations. Enables reproducible grid operations.
    transform_path = out_path.parent / "transform.json"
    transform_data = {
        "crs_epsg": int(target_crs.split(':')[1]),
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "res_m": float(res),
        "nx": int(nx),
        "ny": int(ny),
        "grid_bbox_proj": grid_bbox_proj,
        "grid_bbox_wgs84": grid_bbox_wgs84,
        "note": "Full grid dimensions (nx, ny) before clipping. Actual grid has fewer points due to city boundary clipping."
    }
    print(f"Writing transform metadata to {transform_path}...")
    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    print(f"  ✓ Saved grid transform parameters")
    
    # Get git SHA for provenance
    git_sha = "unknown"
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            check=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from repo root
        )
        git_sha = result.stdout.strip()
    except Exception:
        git_sha = os.environ.get("GIT_SHA", "unknown")
    
    # Write grid_manifest.json
    # Purpose: Provenance tracking for reproducibility. Downstream pipelines check
    # manifest to ensure grid hasn't changed (neighbor indices depend on exact grid).
    manifest_path = out_path.parent / "grid_manifest.json"
    manifest_data = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(Path.cwd())),
        "git_sha": git_sha,
        "n_points": int(n_points),
        "n_points_full_grid": int(estimated),
        "clipping_ratio": float(n_points / estimated),
        "boundary_file": str(boundary_path),
        "schema_version": "1.0.0",
        "note": "Downstream signal processors (landsat, evi, s5p, svf) expect this exact schema: row_idx, grid_id, x_m, y_m, centroid_lon, centroid_lat, col_idx, row_idx_grid, h3_cell. Changing grid requires recomputing all features."
    }
    print(f"Writing manifest to {manifest_path}...")
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  ✓ Saved grid manifest (git SHA: {git_sha[:8]})")
    
    # Validation and summary
    print("\n" + "="*60)
    print("GRID CREATION SUMMARY")
    print("="*60)
    print(f"Total points:     {n_points:,}")
    print(f"row_idx range:    [{grid_gdf['row_idx'].min()}, {grid_gdf['row_idx'].max()}]")
    print(f"col_idx range:    [{grid_gdf['col_idx'].min()}, {grid_gdf['col_idx'].max()}]")
    print(f"row_idx_grid range: [{grid_gdf['row_idx_grid'].min()}, {grid_gdf['row_idx_grid'].max()}]")
    print(f"Full grid size:   {nx} × {ny} = {estimated:,} (before clipping)")
    print(f"Clipping ratio:   {100*n_points/estimated:.2f}%")
    print(f"Lon range:        [{grid_gdf['centroid_lon'].min():.6f}, {grid_gdf['centroid_lon'].max():.6f}]")
    print(f"Lat range:        [{grid_gdf['centroid_lat'].min():.6f}, {grid_gdf['centroid_lat'].max():.6f}]")
    print(f"X (UTM) range:    [{grid_gdf['x_m'].min():.1f}, {grid_gdf['x_m'].max():.1f}] m")
    print(f"Y (UTM) range:    [{grid_gdf['y_m'].min():.1f}, {grid_gdf['y_m'].max():.1f}] m")
    print("="*60)
    print("\nFirst 3 rows:")
    print(grid_gdf.head(3).to_string())
    print("\n✓ Grid creation complete!")
    print(f"\nOutput files:")
    print(f"  - {out_path}")
    print(f"  - {nd_path}")
    print(f"  - {transform_path}")
    print(f"  - {manifest_path}")
    
    return grid_gdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boundary", default="resources/mumbai_wards.geojson")
    p.add_argument("--out", default="data/mumbai/grid_30m.parquet")
    p.add_argument("--crs", default="EPSG:32643")
    p.add_argument("--res", type=float, default=30.0)
    args = p.parse_args()

    create_grid(Path(args.boundary), Path(args.out), target_crs=args.crs, res=args.res)


if __name__ == "__main__":
    main()
