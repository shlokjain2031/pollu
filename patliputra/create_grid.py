"""Create a 30 m centroid grid for Mumbai wards and save as artifact.

This script creates a canonical grid and registers it in the grid manifest.
Grid artifacts are immutable and versioned.

Usage:
    python scripts/create_grid_mumbai.py
    
Artifacts produced:
    artifacts/v1/grids/mumbai/grid.parquet
    artifacts/v1/grids/mumbai/grid_nd.parquet
    artifacts/v1/grids/mumbai/transform.json
    artifacts/v1/grids/mumbai/metadata.json
    
Manifest entry:
    manifest/grid_manifest.db (grid_id='mumbai', grid_version='v1')
"""
from pathlib import Path
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import h3
import json
import os
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone

from manifest.grid_manifest import GridManifest, compute_grid_hash


def create_grid(
    boundary_path: Path,
    grid_id: str = "mumbai",
    grid_version: str = "v1",
    target_crs: str = "EPSG:32643",
    res: float = 30.0,
    max_cells: int = 3_000_000,
    artifacts_root: Path = Path("artifacts"),
):
    print(f"[1/10] Reading boundary from {boundary_path}...")
    gdf = gpd.read_file(boundary_path)
    print(f"       Loaded {len(gdf)} features")
    
    print(f"[2/10] Projecting to {target_crs}...")
    city_proj = gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = city_proj.total_bounds
    print(f"       Bounds: X=[{minx:.2f}, {maxx:.2f}], Y=[{miny:.2f}, {maxy:.2f}]")
    
    print(f"[3/10] Generating grid coordinates (resolution={res}m)...")
    xs = np.arange(minx + res / 2.0, maxx, res)
    ys = np.arange(miny + res / 2.0, maxy, res)
    print(f"       Grid dimensions: {len(xs)} cols × {len(ys)} rows")
    estimated = len(xs) * len(ys)
    print(f"       Estimated cells: {estimated:,}")
    if estimated > max_cells:
        raise RuntimeError(
            f"Estimated grid cells ({estimated}) > max_cells ({max_cells}). Abort to avoid OOM"
        )
    print(f"[4/10] Creating meshgrid...")
    xx, yy = np.meshgrid(xs, ys)
    
    # Store col_idx and row_idx_grid before raveling
    col_indices = np.arange(len(xs))
    row_indices = np.arange(len(ys))
    col_idx_grid, row_idx_grid = np.meshgrid(col_indices, row_indices)
    print(f"       Meshgrid created: {xx.shape}")
    
    print(f"[5/10] Flattening arrays...")
    xx = xx.ravel()
    yy = yy.ravel()
    col_idx_flat = col_idx_grid.ravel()
    row_idx_grid_flat = row_idx_grid.ravel()
    print(f"       Total points: {len(xx):,}")
    
    print(f"[6/10] Creating Point geometries...")
    pts = [Point(x, y) for x, y in zip(xx, yy)]
    print(f"       Created {len(pts):,} points")
    grid_gdf = gpd.GeoDataFrame(
        {
            "x_m": xx.astype(np.float64),  # Explicit float64
            "y_m": yy.astype(np.float64),  # Explicit float64
            "col_idx": col_idx_flat.astype(np.int32),
            "row_idx_grid": row_idx_grid_flat.astype(np.int32),
        },
        geometry=pts,
        crs=target_crs,
    )
    
    # Clip to boundary union
    print(f"[7/10] Clipping to boundary union...")
    city_union = city_proj.union_all()
    mask = grid_gdf.geometry.within(city_union)
    before_clip = len(grid_gdf)
    grid_gdf = grid_gdf[mask].reset_index(drop=True)
    print(f"       Retained {len(grid_gdf):,} of {before_clip:,} points ({100*len(grid_gdf)/before_clip:.1f}%)")
    
    # Add row_idx (contiguous 0-based index after clipping)
    print(f"[8/10] Adding row_idx and grid_id...")
    grid_gdf["row_idx"] = np.arange(len(grid_gdf), dtype=np.int32)
    
    # Add grid_id with zero-padding
    grid_gdf["grid_id"] = grid_gdf["row_idx"].apply(lambda x: f"g{x:08d}")
    print(f"       row_idx range: [0, {len(grid_gdf)-1}]")
    
    # Convert to WGS84 for lat/lon
    print(f"[9/10] Converting to WGS84 and computing H3 cells...")
    grid_wgs84 = grid_gdf.to_crs("EPSG:4326")
    grid_gdf["centroid_lon"] = grid_wgs84.geometry.x.astype(np.float64)
    grid_gdf["centroid_lat"] = grid_wgs84.geometry.y.astype(np.float64)
    print(f"       Lon range: [{grid_gdf['centroid_lon'].min():.6f}, {grid_gdf['centroid_lon'].max():.6f}]")
    print(f"       Lat range: [{grid_gdf['centroid_lat'].min():.6f}, {grid_gdf['centroid_lat'].max():.6f}]")
    
    # Add H3 cells at resolution 11
    grid_gdf["h3_cell"] = grid_wgs84.geometry.apply(
        lambda geom: h3.latlng_to_cell(geom.y, geom.x, 11)
    )
    print(f"       H3 cells computed at resolution 11")
    
    # Reorder columns to match required schema
    grid_gdf = grid_gdf[
        [
            "row_idx",
            "grid_id",
            "x_m",
            "y_m",
            "centroid_lon",
            "centroid_lat",
            "col_idx",
            "row_idx_grid",
            "h3_cell",
            "geometry",
        ]
    ]
    
    # Compute grid hash
    origin_x = minx + res / 2.0
    origin_y = miny + res / 2.0
    grid_hash = compute_grid_hash(target_crs, res, origin_x, origin_y, len(xs), len(ys))
    print(f"\n[10/13] Computed grid hash: {grid_hash[:16]}...")
    
    # Prepare artifact directory
    artifact_dir = artifacts_root / grid_version / "grids" / grid_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"[11/13] Artifact directory: {artifact_dir}")
    
    # Atomic write: use temp directory then rename
    n_points = len(grid_gdf)
    with tempfile.TemporaryDirectory(dir=artifact_dir.parent) as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Write main GeoParquet with geometry
        tmp_grid_path = tmpdir_path / "grid.parquet"
        print(f"[12/13] Writing grid.parquet...")
        grid_gdf.to_parquet(tmp_grid_path, index=False)
        file_size_mb = tmp_grid_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Wrote {n_points:,} points ({file_size_mb:.2f} MB)")
        
        # Write lightweight non-geometry Parquet
        tmp_nd_path = tmpdir_path / "grid_nd.parquet"
        print(f"  Writing grid_nd.parquet...")
        grid_nd = grid_gdf.drop(columns=['geometry'])
        grid_nd.to_parquet(tmp_nd_path, index=False)
        nd_size_mb = tmp_nd_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Wrote {len(grid_gdf):,} points ({nd_size_mb:.2f} MB, no geometry)")
    
        # Compute grid bounding boxes
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
        tmp_transform_path = tmpdir_path / "transform.json"
        transform_data = {
            "crs_epsg": int(target_crs.split(':')[1]),
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "res_m": float(res),
            "nx": int(len(xs)),
            "ny": int(len(ys)),
            "grid_bbox_proj": grid_bbox_proj,
            "grid_bbox_wgs84": grid_bbox_wgs84,
            "note": "Full grid dimensions (nx, ny) before clipping. Actual grid has fewer points due to city boundary clipping."
        }
        print(f"  Writing transform.json...")
        with open(tmp_transform_path, 'w') as f:
            json.dump(transform_data, f, indent=2)
        print(f"  ✓ Wrote transform.json")
    
        # Get git SHA for provenance
        git_sha = "unknown"
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                check=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            git_sha = result.stdout.strip()
        except Exception:
            git_sha = os.environ.get("GIT_SHA", "unknown")
        
        # Write metadata.json
        tmp_metadata_path = tmpdir_path / "metadata.json"
        metadata = {
            "grid_id": grid_id,
            "grid_version": grid_version,
            "grid_hash": grid_hash,
            "generated_ts": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).relative_to(Path.cwd())),
            "git_sha": git_sha,
            "n_points": int(n_points),
            "n_points_full_grid": int(estimated),
            "clipping_ratio": float(n_points / estimated),
            "boundary_file": str(boundary_path),
            "crs": target_crs,
            "resolution_m": float(res),
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "nx": int(len(xs)),
            "ny": int(len(ys)),
            "schema_version": "1.0.0",
            "note": "Canonical grid for spatial alignment. Changing grid requires recomputing all features."
        }
        print(f"  Writing metadata.json...")
        with open(tmp_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Wrote metadata.json (git SHA: {git_sha[:8]})")
        
        # Atomic rename: move all files to final destination
        print(f"\n[13/13] Atomic move to artifact directory...")
        final_grid_path = artifact_dir / "grid.parquet"
        final_nd_path = artifact_dir / "grid_nd.parquet"
        final_transform_path = artifact_dir / "transform.json"
        final_metadata_path = artifact_dir / "metadata.json"
        
        shutil.move(str(tmp_grid_path), str(final_grid_path))
        shutil.move(str(tmp_nd_path), str(final_nd_path))
        shutil.move(str(tmp_transform_path), str(final_transform_path))
        shutil.move(str(tmp_metadata_path), str(final_metadata_path))
        print(f"  ✓ Moved all files to {artifact_dir}")
    
    # Register in grid manifest
    print(f"\nRegistering grid in manifest...")
    manifest = GridManifest()
    returned_hash = manifest.register(
        grid_id=grid_id,
        grid_version=grid_version,
        crs=target_crs,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_m=res,
        nx=len(xs),
        ny=len(ys),
    )
    manifest.close()
    print(f"  ✓ Registered grid '{grid_id}' (hash: {returned_hash[:16]}...)")
    
    # Validation and summary
    print("\n" + "="*60)
    print("GRID CREATION SUMMARY")
    print("="*60)
    print(f"Grid ID:          {grid_id}")
    print(f"Grid version:     {grid_version}")
    print(f"Grid hash:        {grid_hash[:16]}...")
    print(f"Total points:     {n_points:,}")
    print(f"row_idx range:    [{grid_gdf['row_idx'].min()}, {grid_gdf['row_idx'].max()}]")
    print(f"col_idx range:    [{grid_gdf['col_idx'].min()}, {grid_gdf['col_idx'].max()}]")
    print(f"row_idx_grid range: [{grid_gdf['row_idx_grid'].min()}, {grid_gdf['row_idx_grid'].max()}]")
    print(f"Full grid size:   {len(xs)} × {len(ys)} = {estimated:,} (before clipping)")
    print(f"Clipping ratio:   {100*n_points/estimated:.2f}%")
    print(f"Lon range:        [{grid_gdf['centroid_lon'].min():.6f}, {grid_gdf['centroid_lon'].max():.6f}]")
    print(f"Lat range:        [{grid_gdf['centroid_lat'].min():.6f}, {grid_gdf['centroid_lat'].max():.6f}]")
    print(f"X (UTM) range:    [{grid_gdf['x_m'].min():.1f}, {grid_gdf['x_m'].max():.1f}] m")
    print(f"Y (UTM) range:    [{grid_gdf['y_m'].min():.1f}, {grid_gdf['y_m'].max():.1f}] m")
    print("="*60)
    print("\nFirst 3 rows:")
    print(grid_gdf.head(3).to_string())
    print("\n✓ Grid creation complete!")
    print(f"\nArtifact files:")
    print(f"  - {artifact_dir / 'grid.parquet'}")
    print(f"  - {artifact_dir / 'grid_nd.parquet'}")
    print(f"  - {artifact_dir / 'transform.json'}")
    print(f"  - {artifact_dir / 'metadata.json'}")
    print(f"\nManifest:")
    print(f"  - manifest/grid_manifest.db (grid_id='{grid_id}')")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boundary", default="resources/mumbai_wards.geojson", help="Path to boundary GeoJSON")
    p.add_argument("--grid-id", default="mumbai", help="Grid identifier (city name)")
    p.add_argument("--grid-version", default="v1", help="Grid version")
    p.add_argument("--crs", default="EPSG:32643", help="Target CRS")
    p.add_argument("--res", type=float, default=30.0, help="Grid resolution in meters")
    p.add_argument("--artifacts-root", default="artifacts", help="Artifacts root directory")
    args = p.parse_args()
    
    create_grid(
        boundary_path=Path(args.boundary),
        grid_id=args.grid_id,
        grid_version=args.grid_version,
        target_crs=args.crs,
        res=args.res,
        artifacts_root=Path(args.artifacts_root),
    )


if __name__ == "__main__":
    main()