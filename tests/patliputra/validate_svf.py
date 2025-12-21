#!/usr/bin/env python3
"""
Validate SVF (Sky View Factor) GeoTIFF file.

Checks:
- File exists and is readable
- Valid GeoTIFF format
- Expected CRS (EPSG:32643 for Mumbai)
- Expected resolution (3m for high-res urban analysis)
- Value range [0, 1] (SVF is a ratio)
- No excessive nodata/NaN values
- Spatial coverage matches Mumbai bbox
- Statistics (mean, std, percentiles)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS


def validate_svf_file(
    svf_path: Path,
    expected_crs: str = "EPSG:32643",
    expected_resolution_m: float = 3.0,
    tolerance: float = 0.1,
    max_nodata_fraction: float = 0.1,
) -> dict:
    """
    Validate SVF GeoTIFF file.

    Parameters
    ----------
    svf_path : Path
        Path to SVF GeoTIFF
    expected_crs : str
        Expected CRS (default: EPSG:32643 for UTM 43N)
    expected_resolution_m : float
        Expected pixel resolution in meters (default: 3.0)
    tolerance : float
        Tolerance for resolution check in meters (default: 0.1)
    max_nodata_fraction : float
        Maximum allowed fraction of nodata/NaN pixels (default: 0.1 = 10%)

    Returns
    -------
    dict
        Validation results with status and statistics
    """
    results = {
        "file": str(svf_path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check file exists
    if not svf_path.exists():
        results["valid"] = False
        results["errors"].append(f"File does not exist: {svf_path}")
        return results

    try:
        with rasterio.open(svf_path) as src:
            # Check format
            if src.driver != "GTiff":
                results["warnings"].append(f"Expected GTiff format, got {src.driver}")

            # Check CRS
            expected_crs_obj = CRS.from_string(expected_crs)
            if src.crs != expected_crs_obj:
                results["errors"].append(
                    f"CRS mismatch: expected {expected_crs}, got {src.crs}"
                )
                results["valid"] = False

            # Check resolution
            transform = src.transform
            pixel_width = abs(transform.a)
            pixel_height = abs(transform.e)

            if abs(pixel_width - expected_resolution_m) > tolerance:
                results["errors"].append(
                    f"Pixel width mismatch: expected {expected_resolution_m}m, "
                    f"got {pixel_width:.2f}m"
                )
                results["valid"] = False

            if abs(pixel_height - expected_resolution_m) > tolerance:
                results["errors"].append(
                    f"Pixel height mismatch: expected {expected_resolution_m}m, "
                    f"got {pixel_height:.2f}m"
                )
                results["valid"] = False

            # Check orientation (should be north-up, e < 0)
            if transform.e > 0:
                results["errors"].append(
                    f"Raster is inverted (upside-down): transform.e = {transform.e} > 0"
                )
                results["valid"] = False

            # Read data
            data = src.read(1)
            results["stats"]["shape"] = data.shape
            results["stats"]["dtype"] = str(data.dtype)

            # Check for nodata
            nodata = src.nodata
            if nodata is not None:
                nodata_mask = data == nodata
            else:
                nodata_mask = np.isnan(data)

            nodata_fraction = nodata_mask.sum() / data.size
            results["stats"]["nodata_fraction"] = float(nodata_fraction)

            if nodata_fraction > max_nodata_fraction:
                results["warnings"].append(
                    f"High nodata fraction: {nodata_fraction:.2%} "
                    f"(threshold: {max_nodata_fraction:.2%})"
                )

            # Get valid data
            valid_data = data[~nodata_mask]

            if len(valid_data) == 0:
                results["errors"].append("No valid data pixels found")
                results["valid"] = False
                return results

            # Check value range [0, 1]
            min_val = float(valid_data.min())
            max_val = float(valid_data.max())

            results["stats"]["min"] = min_val
            results["stats"]["max"] = max_val
            results["stats"]["mean"] = float(valid_data.mean())
            results["stats"]["std"] = float(valid_data.std())
            results["stats"]["median"] = float(np.median(valid_data))

            # Percentiles
            percentiles = [5, 25, 50, 75, 95]
            for p in percentiles:
                results["stats"][f"p{p}"] = float(np.percentile(valid_data, p))

            if min_val < 0:
                results["errors"].append(f"SVF values below 0 found: min={min_val:.4f}")
                results["valid"] = False

            if max_val > 1:
                results["errors"].append(f"SVF values above 1 found: max={max_val:.4f}")
                results["valid"] = False

            # Check for suspicious patterns
            if min_val == max_val:
                results["errors"].append(f"All pixels have same value: {min_val:.4f}")
                results["valid"] = False

            # Check spatial extent
            bounds = src.bounds
            results["stats"]["bounds"] = {
                "west": bounds.left,
                "south": bounds.bottom,
                "east": bounds.right,
                "north": bounds.top,
            }

            # Mumbai bbox in EPSG:32643 (approximate)
            # Original WGS84: (72.7763, 18.8939, 72.9797, 19.2701)
            # Converted to UTM 43N
            mumbai_utm_west = 663000
            mumbai_utm_east = 686000
            mumbai_utm_south = 2091000
            mumbai_utm_north = 2133000

            if (
                bounds.left < mumbai_utm_west - 5000
                or bounds.right > mumbai_utm_east + 5000
                or bounds.bottom < mumbai_utm_south - 5000
                or bounds.top > mumbai_utm_north + 5000
            ):
                results["warnings"].append(
                    "Spatial extent differs significantly from expected Mumbai bbox"
                )

            # Additional metadata
            results["stats"]["width"] = src.width
            results["stats"]["height"] = src.height
            results["stats"]["count"] = src.count
            results["stats"]["crs"] = str(src.crs)
            results["stats"]["resolution_m"] = (pixel_width, pixel_height)

    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Error reading file: {str(e)}")

    return results


def print_validation_results(results: dict) -> None:
    """
    Pretty print validation results.

    Parameters
    ----------
    results : dict
        Validation results from validate_svf_file
    """
    print("=" * 80)
    print(f"SVF Validation Report: {Path(results['file']).name}")
    print("=" * 80)

    # Status
    if results["valid"]:
        print("✓ Status: VALID")
    else:
        print("✗ Status: INVALID")

    print()

    # Errors
    if results["errors"]:
        print(f"Errors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  ✗ {err}")
        print()

    # Warnings
    if results["warnings"]:
        print(f"Warnings ({len(results['warnings'])}):")
        for warn in results["warnings"]:
            print(f"  ⚠ {warn}")
        print()

    # Statistics
    if results["stats"]:
        print("Statistics:")
        stats = results["stats"]

        print(f"  Shape: {stats.get('shape', 'N/A')}")
        print(f"  CRS: {stats.get('crs', 'N/A')}")
        print(
            f"  Resolution: {stats.get('resolution_m', 'N/A')[0]:.2f} × "
            f"{stats.get('resolution_m', 'N/A')[1]:.2f} m"
        )
        print(f"  Valid pixels: {(1 - stats.get('nodata_fraction', 0)):.2%}")

        print("\n  Value Statistics:")
        print(f"    Min:    {stats.get('min', 'N/A'):.4f}")
        print(f"    Max:    {stats.get('max', 'N/A'):.4f}")
        print(f"    Mean:   {stats.get('mean', 'N/A'):.4f}")
        print(f"    Median: {stats.get('median', 'N/A'):.4f}")
        print(f"    Std:    {stats.get('std', 'N/A'):.4f}")

        print("\n  Percentiles:")
        for p in [5, 25, 50, 75, 95]:
            if f"p{p}" in stats:
                print(f"    P{p:2d}:    {stats[f'p{p}']:.4f}")

        if "bounds" in stats:
            print("\n  Spatial Bounds (UTM):")
            b = stats["bounds"]
            print(f"    West:  {b['west']:,.0f}")
            print(f"    South: {b['south']:,.0f}")
            print(f"    East:  {b['east']:,.0f}")
            print(f"    North: {b['north']:,.0f}")

    print("=" * 80)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate SVF GeoTIFF file")
    parser.add_argument(
        "svf_file",
        type=Path,
        help="Path to SVF GeoTIFF file",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:32643",
        help="Expected CRS (default: EPSG:32643)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=3.0,
        help="Expected resolution in meters (default: 3.0)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Resolution tolerance in meters (default: 0.1)",
    )
    parser.add_argument(
        "--max-nodata",
        type=float,
        default=0.1,
        help="Maximum allowed nodata fraction (default: 0.1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and warnings",
    )

    args = parser.parse_args()

    # Validate
    results = validate_svf_file(
        args.svf_file,
        args.crs,
        args.resolution,
        args.tolerance,
        args.max_nodata,
    )

    # Print results
    if not args.quiet:
        print_validation_results(results)
    else:
        if results["errors"]:
            for err in results["errors"]:
                print(f"ERROR: {err}", file=sys.stderr)
        if results["warnings"]:
            for warn in results["warnings"]:
                print(f"WARNING: {warn}", file=sys.stderr)

    # Exit code
    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()
