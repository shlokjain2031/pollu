#!/usr/bin/env python3
"""
Sky View Factor (SVF) calculation using Google Earth Engine.

Computes SVF using horizon angle search in multiple azimuth directions.
Uses Google Open Buildings Temporal + NASA DEM to create DSM.

SVF Formula:
    SVF = (1/n) * Σ sin²(α_i)
    where α_i is the maximum horizon angle in azimuth direction i

References:
    - Zakšek et al. (2011) "Sky-View Factor as a Relief Visualization Technique"
    - Yokoyama et al. (2002) "Visualizing Topography by Openness"
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import ee


def create_dsm(
    year: int,
    geometry: ee.Geometry,
    resolution_m: float = 5.0,
    target_crs: str = "EPSG:32643",
) -> ee.Image:
    """
    Create Digital Surface Model from DEM + building heights.

    Parameters
    ----------
    year : int
        Year for building data (2016-2023)
    geometry : ee.Geometry
        Area of interest
    resolution_m : float
        Output resolution in meters
    target_crs : str
        Target CRS (e.g., 'EPSG:32643' for UTM 43N)

    Returns
    -------
    ee.Image
        DSM in meters above sea level, reprojected to target CRS
    """
    # Load DEM (elevation)
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

    # Load building heights
    buildings = (
        ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
        .filterBounds(geometry)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("building_height")
        .max()  # Get maximum building height per pixel
    )

    # Create DSM: ground elevation + building height
    # Where no buildings exist, building_height is masked, so DSM = DEM
    dsm = dem.add(buildings).unmask(dem)

    # Reproject to target CRS at desired resolution
    dsm = dsm.reproject(crs=target_crs, scale=resolution_m).rename("dsm")

    return dsm


def compute_horizon_angle(
    dsm: ee.Image,
    azimuth_deg: float,
    search_radius_m: float,
    resolution_m: float,
) -> ee.Image:
    """
    Compute maximum horizon angle in a given azimuth direction.

    Parameters
    ----------
    dsm : ee.Image
        Digital Surface Model in meter-based CRS
    azimuth_deg : float
        Azimuth direction in degrees (0° = North, 90° = East)
    search_radius_m : float
        Search radius in meters
    resolution_m : float
        Pixel resolution in meters

    Returns
    -------
    ee.Image
        Maximum horizon angle (radians) in this direction
    """
    # Convert azimuth to radians
    azimuth_rad = azimuth_deg * 3.14159265359 / 180.0

    # Create unit pixel offsets (dx, dy in pixels per step)
    # North = +Y, East = +X in standard raster coordinates
    dx = ee.Number(azimuth_rad).sin()  # pixels per step
    dy = ee.Number(azimuth_rad).cos()  # pixels per step

    # Number of search steps
    n_steps = int(search_radius_m / resolution_m)

    # Initialize maximum horizon angle
    max_angle = ee.Image.constant(0)

    # Search along the azimuth direction
    for step in range(1, n_steps + 1):
        # Offset distance in meters
        distance = step * resolution_m

        # Displace DSM by step pixels along azimuth
        displaced_dsm = dsm.displace(
            displacement=ee.Image.constant([dx.multiply(step), dy.multiply(step)]),
            mode="bilinear",
        )

        # Height difference between offset point and origin
        height_diff = displaced_dsm.subtract(dsm)

        # Horizon angle = arctan(height_diff / distance)
        angle = height_diff.divide(distance).atan()

        # Update maximum angle
        max_angle = max_angle.max(angle)

    return max_angle.rename(f"horizon_{int(azimuth_deg)}")


def compute_svf(
    year: int,
    geometry: ee.Geometry,
    n_directions: int = 16,
    search_radius_m: float = 100.0,
    resolution_m: float = 5.0,
    target_crs: str = "EPSG:32643",
) -> ee.Image:
    """
    Compute Sky View Factor using horizon angle integration.

    Parameters
    ----------
    year : int
        Year for building data (2016-2023)
    geometry : ee.Geometry
        Area of interest
    n_directions : int
        Number of azimuth directions (default 16 = 22.5° spacing)
    search_radius_m : float
        Horizon search radius in meters (default 100m)
    resolution_m : float
        Output resolution in meters (5m matches PM2.5 sensor scale)
    target_crs : str
        Target CRS for computation (meter-based, e.g., 'EPSG:32643')

    Returns
    -------
    ee.Image
        Sky View Factor (0-1, where 1 = completely open sky)
    """
    print(f"  [SVF] Creating DSM...")
    # Create DSM in target CRS at desired resolution
    dsm = create_dsm(year, geometry, resolution_m=resolution_m, target_crs=target_crs)

    print(
        f"  [SVF] Computing horizon angles in {n_directions} directions (step: {360.0/n_directions:.1f}°)..."
    )
    # Compute horizon angles in all directions
    azimuth_step = 360.0 / n_directions

    # Sum of sin²(horizon_angle) across all directions
    svf_sum = ee.Image.constant(0)

    for i in range(n_directions):
        azimuth = i * azimuth_step
        if i % 4 == 0:  # Log every 4th direction
            print(
                f"    [SVF] Computing direction {i+1}/{n_directions} ({azimuth:.0f}°)..."
            )
        horizon_angle = compute_horizon_angle(
            dsm, azimuth, search_radius_m, resolution_m
        )

        # SVF contribution: sin²(horizon_angle)
        # horizon_angle is measured from horizontal plane
        # SVF uses angle from zenith, so: zenith_angle = π/2 - horizon_angle
        # cos(zenith_angle) = sin(horizon_angle)
        # Therefore: cos²(zenith_angle) = sin²(horizon_angle)
        svf_contribution = horizon_angle.sin().pow(2)
        svf_sum = svf_sum.add(svf_contribution)

    print(f"  [SVF] Averaging over {n_directions} directions...")
    # Average over all directions
    svf = svf_sum.divide(n_directions).rename("svf")

    # Ensure proper projection
    svf = svf.reproject(crs=target_crs, scale=resolution_m)

    # Clamp to [0, 1]
    svf = svf.clamp(0, 1)

    print(f"  [SVF] ✓ SVF computation complete")
    return svf


def split_bbox_into_tiles(
    bbox: Tuple[float, float, float, float], tile_size_deg: float = 0.02
) -> List[Tuple[float, float, float, float]]:
    """
    Split bounding box into smaller tiles to avoid Earth Engine download limits.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in EPSG:4326
    tile_size_deg : float
        Size of each tile in degrees (default 0.02 ≈ 2.2km at Mumbai latitude)

    Returns
    -------
    List[tuple]
        List of tile bboxes (west, south, east, north)
    """
    west, south, east, north = bbox
    tiles = []

    x = west
    while x < east:
        y = south
        while y < north:
            tile_west = x
            tile_south = y
            tile_east = min(x + tile_size_deg, east)
            tile_north = min(y + tile_size_deg, north)
            tiles.append((tile_west, tile_south, tile_east, tile_north))
            y = tile_north
        x = tile_east

    return tiles


def download_svf_tile(
    svf_image: ee.Image,
    tile_bbox: Tuple[float, float, float, float],
    tile_path: Path,
    resolution_m: float = 3.0,
    target_crs: str = "EPSG:32643",
) -> Path:
    """
    Download a single SVF tile.

    Parameters
    ----------
    svf_image : ee.Image
        Computed SVF image
    tile_bbox : tuple
        (west, south, east, north) for this tile in EPSG:4326
    tile_path : Path
        Where to save the tile
    resolution_m : float
        Pixel resolution in meters
    target_crs : str
        Target CRS for export

    Returns
    -------
    Path
        Path to downloaded tile
    """
    import requests

    west, south, east, north = tile_bbox
    # Create geometry in WGS84
    geometry = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326")

    # Transform to target CRS for export (ensures consistent orientation)
    geometry_utm = geometry.transform(target_crs, maxError=1)

    url = svf_image.getDownloadURL(
        {
            "scale": resolution_m,
            "region": geometry_utm,
            "filePerBand": False,
            "format": "GeoTIFF",
            "crs": target_crs,
        }
    )

    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tile_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return tile_path


def download_svf_for_mumbai(
    year: int = 2023,
    output_path: Path | None = None,
    bbox: Tuple[float, float, float, float] = (72.7763, 18.8939, 72.9797, 19.2701),
    n_directions: int = 16,
    search_radius_m: float = 100.0,
    resolution_m: float = 3.0,
    tile_cache_dir: Path | None = None,
    target_crs: str = "EPSG:32643",
) -> Path:
    """
    Download SVF raster for Mumbai using tiled approach.

    Parameters
    ----------
    year : int
        Year for building data
    output_path : Path | None
        Where to save final merged GeoTIFF (default: cache/mumbai_svf_{year}_{resolution}m.tif)
    bbox : tuple
        (west, south, east, north) in EPSG:4326
    n_directions : int
        Azimuth directions for horizon search
    search_radius_m : float
        Search radius in meters
    resolution_m : float
        Pixel resolution in meters (2-3m recommended)
    tile_cache_dir : Path | None
        Directory to cache tiles (default: cache/svf_tiles)
    target_crs : str
        Target CRS for computation and export (meter-based)

    Returns
    -------
    Path
        Path to merged GeoTIFF
    """
    import rasterio
    from rasterio.merge import merge as rio_merge

    if output_path is None:
        output_path = Path(f"cache/mumbai_svf_{year}_{int(resolution_m)}m.tif")

    if tile_cache_dir is None:
        tile_cache_dir = Path("cache/svf_tiles")

    # Create geometry
    west, south, east, north = bbox
    geometry = ee.Geometry.Rectangle([west, south, east, north])

    # Compute SVF (compute once, download in tiles)
    print(
        f"Computing SVF for {year} with {n_directions} directions at {resolution_m}m resolution..."
    )
    svf = compute_svf(
        year,
        geometry,
        n_directions,
        search_radius_m,
        resolution_m,
        target_crs=target_crs,
    )

    # Split into tiles
    tiles = split_bbox_into_tiles(bbox, tile_size_deg=0.02)
    print(f"Downloading {len(tiles)} tiles...")

    # Prepare tile directory
    tile_dir = tile_cache_dir / f"svf_{year}_{int(resolution_m)}m"
    tile_dir.mkdir(parents=True, exist_ok=True)

    def download_tile_task(
        idx: int, tile_bbox: Tuple[float, float, float, float]
    ) -> Tuple[int, Path | None]:
        """Download a single tile and return (index, path or None)."""
        tile_path = tile_dir / f"tile_{idx:03d}.tif"

        if tile_path.exists():
            return (idx, tile_path)

        try:
            download_svf_tile(
                svf, tile_bbox, tile_path, resolution_m, target_crs=target_crs
            )
            return (idx, tile_path)
        except Exception as e:
            print(f"    ✗ Tile {idx+1}/{len(tiles)} failed: {e}")
            import traceback

            print(f"       Traceback: {traceback.format_exc()}")
            return (idx, None)

    # Download tiles in parallel
    tile_paths = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_tile_task, idx, tile_bbox): idx
            for idx, tile_bbox in enumerate(tiles)
        }

        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            idx, tile_path = future.result()
            if tile_path:
                tile_paths.append(tile_path)
                status = "cached" if tile_path.exists() else "downloaded"
                print(
                    f"    ✓ Tile {idx+1}/{len(tiles)} {status} ({completed_count}/{len(tiles)} complete)"
                )

    if not tile_paths:
        raise RuntimeError(f"Failed to download any tiles for SVF {year}")

    # Merge tiles
    print(f"Merging {len(tile_paths)} tiles...")

    # Open all tiles for merging
    src_files = []

    for tile_path in tile_paths:
        src = rasterio.open(tile_path)
        src_files.append(src)

    try:
        # Merge tiles
        mosaic, out_transform = rio_merge(src_files)

        # Get metadata from first tile
        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "deflate",
                "tiled": True,
            }
        )

        # Write merged file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"✓ SVF saved to {output_path}")

    except Exception as e:
        print(f"Merge failed: {e}")
        # Try GDAL fallback for large mosaics
        import subprocess
        import shutil

        gdalbuildvrt = shutil.which("gdalbuildvrt")
        gdal_translate = shutil.which("gdal_translate")

        if gdalbuildvrt and gdal_translate:
            print("Attempting GDAL VRT merge as fallback...")
            import tempfile

            with tempfile.TemporaryDirectory(prefix="svf_vrt_") as tmpdir:
                list_path = Path(tmpdir) / "tiles.txt"
                vrt_path = Path(tmpdir) / "mosaic.vrt"

                with open(list_path, "w") as f:
                    for p in tile_paths:
                        f.write(str(p.resolve()) + "\n")

                # Build VRT
                subprocess.check_call(
                    [gdalbuildvrt, "-input_file_list", str(list_path), str(vrt_path)]
                )

                # Translate to GeoTIFF
                output_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.check_call(
                    [
                        gdal_translate,
                        "-of",
                        "GTiff",
                        "-co",
                        "COMPRESS=DEFLATE",
                        "-co",
                        "TILED=YES",
                        "-co",
                        "BIGTIFF=YES",
                        str(vrt_path),
                        str(output_path),
                    ]
                )

                print(f"✓ SVF saved to {output_path} via GDAL")
        else:
            raise RuntimeError(
                "Merge failed and GDAL utilities not found. Install GDAL or reduce resolution."
            ) from e
    finally:
        # Clean up opened files
        for src in src_files:
            src.close()

    return output_path


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Sky View Factor using Earth Engine"
    )
    parser.add_argument(
        "--year", type=int, default=2023, help="Year for building data (2016-2023)"
    )
    parser.add_argument("--output", type=Path, help="Output GeoTIFF path")
    parser.add_argument(
        "--bbox",
        type=str,
        default="72.7763,18.8939,72.9797,19.2701",
        help="Bounding box: west,south,east,north",
    )
    parser.add_argument(
        "--directions", type=int, default=16, help="Number of azimuth directions"
    )
    parser.add_argument(
        "--radius", type=float, default=100.0, help="Search radius in meters"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=3.0,
        help="Pixel resolution in meters (2-3m recommended)",
    )
    parser.add_argument(
        "--tile-cache-dir",
        type=Path,
        default=Path("cache/svf_tiles"),
        help="Tile cache directory",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove existing tile cache before downloading",
    )

    args = parser.parse_args()

    # Initialize Earth Engine
    try:
        ee.Initialize(project="fast-archive-465917-m0")
        print("Earth Engine initialized")
    except Exception as e:
        print(f"Failed to initialize Earth Engine: {e}")
        return

    # Parse bbox
    bbox = tuple(float(x.strip()) for x in args.bbox.split(","))

    # Clean cache if requested
    if args.clean_cache:
        tile_cache = args.tile_cache_dir / f"svf_{args.year}_{int(args.resolution)}m"
        if tile_cache.exists():
            import shutil

            print(f"Removing existing tile cache: {tile_cache}")
            shutil.rmtree(tile_cache)

    # Download SVF
    output_path = download_svf_for_mumbai(
        year=args.year,
        output_path=args.output,
        bbox=bbox,
        n_directions=args.directions,
        search_radius_m=args.radius,
        resolution_m=args.resolution,
        tile_cache_dir=args.tile_cache_dir,
    )


if __name__ == "__main__":
    main()
