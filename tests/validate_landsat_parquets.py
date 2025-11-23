#!/usr/bin/env python3
"""
Comprehensive validation script for Landsat parquet files.

Validates:
1. TOA bands (B1-B7, B10, B11)
2. Computed indices (NDVI, IBI, NDBI, UI, etc.)
3. Sentinel-2 EVI columns
4. Sentinel-5P NO2 columns
5. OSM industrial features
6. OSM dust features
7. OpenAQ PM2.5 values (if available)
8. Grid geometry and CRS

Prints a comprehensive report and exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set

import geopandas as gpd
import numpy as np
import pandas as pd


# Expected column groups
TOA_BANDS = [
    "toa_b1",
    "toa_b2",
    "toa_b3",
    "toa_b4",
    "toa_b5",
    "toa_b6",
    "toa_b7",
    "toa_b10",
    "toa_b11",
]

INDICES = [
    "ndvi",
    "ibi",
    "ndbi",
    "ui",
    "mndwi",
    "bsi",
    "ebbi",
]

EVI_COLUMNS = [
    "sentinel2_evi",
    "sentinel2_evi_is_nodata",
    "sentinel2_evi_meta",
]

S5P_COLUMNS = [
    "s5p_no2",
    "s5p_no2_is_nodata",
    "s5p_no2_meta",
]

OSM_INDUSTRIAL_COLUMNS = [
    "industrial_area_m2",
    "industrial_building_area_m2",
    "industrial_building_count",
    "industrial_building_volume_m3",
    "power_plant_count",
    "chimney_count",
    "landfill_area_m2",
    "quarry_area_m2",
    "industrial_poi_count",
    "industrial_contrib_count",
    "industrial_index_raw",
    "industrial_index_combined",
    "industrial_confidence",
]

OSM_DUST_COLUMNS = [
    "total_road_m",
    "unpaved_road_m",
    "pct_unpaved",
    "road_weighted",
    "road_dust_index",
    "construction_area_m2",
    "construction_count",
    "construction_dust_index",
    "mining_area_m2",
    "mining_count",
    "mining_dust_index",
    "vacant_land_area_m2",
    "vacant_land_count",
    "vacant_dust_index",
    "dust_index_combined",
]

OPENAQ_COLUMNS = [
    "pm25",
    "sample_count",
]

REQUIRED_CORE_COLUMNS = [
    "grid_id",
    "geometry",
    "tile_id",
]

EXPECTED_CRS = "EPSG:32643"


def validate_file(
    path: Path,
    require_toa: bool = True,
    require_indices: bool = True,
    require_evi: bool = False,
    require_s5p: bool = False,
    require_osm_industrial: bool = False,
    require_osm_dust: bool = False,
    require_openaq: bool = False,
) -> List[str]:
    """
    Validate a single Landsat parquet file.

    Returns a list of validation issues (empty if all checks pass).
    """
    issues = []

    # 1. File exists and can be read
    if not path.exists():
        return [f"File not found: {path}"]

    try:
        gdf = gpd.read_parquet(path)
    except Exception as e:
        return [f"Failed to read parquet: {e}"]

    if len(gdf) == 0:
        issues.append("Empty dataframe (0 rows)")
        return issues

    # 2. Core columns
    missing_core = [col for col in REQUIRED_CORE_COLUMNS if col not in gdf.columns]
    if missing_core:
        issues.append(f"Missing core columns: {missing_core}")

    # 3. CRS
    if gdf.crs is None:
        issues.append("CRS is None")
    elif str(gdf.crs) != EXPECTED_CRS:
        issues.append(f"CRS mismatch: expected {EXPECTED_CRS}, got {gdf.crs}")

    # 4. TOA bands
    if require_toa:
        missing_toa = [col for col in TOA_BANDS if col not in gdf.columns]
        if missing_toa:
            issues.append(f"Missing TOA bands: {missing_toa}")
        else:
            # Check for reasonable values (0-1 for reflectance, or nodata)
            for band in TOA_BANDS:
                values = gdf[band].dropna()
                if len(values) > 0:
                    if values.min() < -0.1 or values.max() > 2.0:
                        issues.append(
                            f"TOA band {band} out of range: min={values.min():.3f}, max={values.max():.3f}"
                        )

    # 5. Indices
    if require_indices:
        missing_indices = [col for col in INDICES if col not in gdf.columns]
        if missing_indices:
            issues.append(f"Missing indices: {missing_indices}")
        else:
            # Check for reasonable values (-1 to 1 for most indices)
            for idx in INDICES:
                values = gdf[idx].dropna()
                if len(values) > 0:
                    if values.min() < -2.0 or values.max() > 2.0:
                        issues.append(
                            f"Index {idx} out of range: min={values.min():.3f}, max={values.max():.3f}"
                        )

    # 6. Sentinel-2 EVI
    if require_evi:
        missing_evi = [col for col in EVI_COLUMNS if col not in gdf.columns]
        if missing_evi:
            issues.append(f"Missing EVI columns: {missing_evi}")
        else:
            # Check EVI values
            evi_values = gdf["sentinel2_evi"]
            evi_nodata = gdf["sentinel2_evi_is_nodata"]

            # Ensure some non-nodata values
            if evi_nodata.all():
                issues.append("All EVI values are nodata")
            else:
                valid_evi = evi_values[~evi_nodata]
                if len(valid_evi) > 0:
                    if valid_evi.min() < -1.0 or valid_evi.max() > 1.5:
                        issues.append(
                            f"EVI out of range: min={valid_evi.min():.3f}, max={valid_evi.max():.3f}"
                        )

    # 7. Sentinel-5P NO2
    if require_s5p:
        missing_s5p = [col for col in S5P_COLUMNS if col not in gdf.columns]
        if missing_s5p:
            issues.append(f"Missing S5P NO2 columns: {missing_s5p}")
        else:
            # Check NO2 values
            no2_values = gdf["s5p_no2"]
            no2_nodata = gdf["s5p_no2_is_nodata"]

            # Ensure some non-nodata values
            if no2_nodata.all():
                issues.append("All NO2 values are nodata")
            else:
                valid_no2 = no2_values[~no2_nodata]
                if len(valid_no2) > 0:
                    # NO2 should be positive (mol/m²)
                    if valid_no2.min() < 0:
                        issues.append(
                            f"NO2 has negative values: min={valid_no2.min():.6e}"
                        )

    # 8. OSM Industrial
    if require_osm_industrial:
        missing_industrial = [
            col for col in OSM_INDUSTRIAL_COLUMNS if col not in gdf.columns
        ]
        if missing_industrial:
            issues.append(f"Missing OSM industrial columns: {missing_industrial}")
        else:
            # Check for non-negative values
            for col in OSM_INDUSTRIAL_COLUMNS:
                if col in ["industrial_confidence"]:
                    continue  # Can be 0-1
                values = gdf[col].dropna()
                if len(values) > 0 and values.min() < 0:
                    issues.append(f"OSM industrial column {col} has negative values")

    # 9. OSM Dust
    if require_osm_dust:
        missing_dust = [col for col in OSM_DUST_COLUMNS if col not in gdf.columns]
        if missing_dust:
            issues.append(f"Missing OSM dust columns: {missing_dust}")
        else:
            # Check for non-negative values
            for col in OSM_DUST_COLUMNS:
                if col == "pct_unpaved":
                    # Should be 0-100
                    values = gdf[col].dropna()
                    if len(values) > 0:
                        if values.min() < 0 or values.max() > 100:
                            issues.append(
                                f"pct_unpaved out of range [0,100]: min={values.min():.1f}, max={values.max():.1f}"
                            )
                else:
                    values = gdf[col].dropna()
                    if len(values) > 0 and values.min() < 0:
                        issues.append(f"OSM dust column {col} has negative values")

    # 10. OpenAQ PM2.5
    if require_openaq:
        missing_openaq = [col for col in OPENAQ_COLUMNS if col not in gdf.columns]
        if missing_openaq:
            issues.append(f"Missing OpenAQ columns: {missing_openaq}")
        else:
            pm25_values = gdf["pm25"].dropna()
            if len(pm25_values) == 0:
                issues.append("All PM2.5 values are NaN/missing")
            else:
                # PM2.5 should be positive (µg/m³)
                if pm25_values.min() < 0:
                    issues.append(
                        f"PM2.5 has negative values: min={pm25_values.min():.1f}"
                    )
                if pm25_values.max() > 1000:
                    issues.append(
                        f"PM2.5 has suspiciously high values: max={pm25_values.max():.1f}"
                    )

    return issues


def validate_directory(
    cache_root: Path,
    pattern: str = "landsat_*.parquet",
    max_files: Optional[int] = None,
    require_toa: bool = True,
    require_indices: bool = True,
    require_evi: bool = False,
    require_s5p: bool = False,
    require_osm_industrial: bool = False,
    require_osm_dust: bool = False,
    require_openaq: bool = False,
) -> int:
    """
    Validate all Landsat parquet files in a directory.

    Returns 0 if all files pass, 1 otherwise.
    """
    files = sorted(cache_root.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    if not files:
        print(f"No files matched pattern '{pattern}' in {cache_root}")
        return 1

    print(f"Validating {len(files)} files in {cache_root}\n")

    # Check for failed_dates.txt log
    failed_log = cache_root / "failed_dates.txt"
    if failed_log.exists():
        print("⚠ Found failed_dates.txt log - dates that failed processing:\n")
        with open(failed_log, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    print(f"  {line.strip()}")
        print()

    all_issues = {}
    for path in files:
        issues = validate_file(
            path,
            require_toa=require_toa,
            require_indices=require_indices,
            require_evi=require_evi,
            require_s5p=require_s5p,
            require_osm_industrial=require_osm_industrial,
            require_osm_dust=require_osm_dust,
            require_openaq=require_openaq,
        )
        if issues:
            all_issues[path.name] = issues

    # Print report
    if not all_issues:
        print("✓ All files passed validation!")
        print(f"\nValidated features:")
        print(f"  • TOA bands: {len(TOA_BANDS)} bands")
        print(f"  • Indices: {len(INDICES)} indices")
        if require_evi:
            print(f"  • Sentinel-2 EVI: {len(EVI_COLUMNS)} columns")
        if require_s5p:
            print(f"  • Sentinel-5P NO2: {len(S5P_COLUMNS)} columns")
        if require_osm_industrial:
            print(f"  • OSM Industrial: {len(OSM_INDUSTRIAL_COLUMNS)} columns")
        if require_osm_dust:
            print(f"  • OSM Dust: {len(OSM_DUST_COLUMNS)} columns")
        if require_openaq:
            print(f"  • OpenAQ PM2.5: {len(OPENAQ_COLUMNS)} columns")
        return 0
    else:
        print(f"✗ {len(all_issues)} file(s) failed validation:\n")
        for fname, issues in all_issues.items():
            print(f"{fname}:")
            for issue in issues:
                print(f"  - {issue}")
            print()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Landsat parquet files for completeness and correctness"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache"),
        help="Directory containing landsat_YYYY-MM-DD.parquet files",
    )
    parser.add_argument(
        "--pattern",
        default="landsat_*.parquet",
        help="Glob pattern to match files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to validate (for testing)",
    )
    parser.add_argument(
        "--no-toa",
        action="store_true",
        help="Skip TOA band validation",
    )
    parser.add_argument(
        "--no-indices",
        action="store_true",
        help="Skip index validation",
    )
    parser.add_argument(
        "--require-evi",
        action="store_true",
        help="Require Sentinel-2 EVI columns",
    )
    parser.add_argument(
        "--require-s5p",
        action="store_true",
        help="Require Sentinel-5P NO2 columns",
    )
    parser.add_argument(
        "--require-osm-industrial",
        action="store_true",
        help="Require OSM industrial columns",
    )
    parser.add_argument(
        "--require-osm-dust",
        action="store_true",
        help="Require OSM dust columns",
    )
    parser.add_argument(
        "--require-openaq",
        action="store_true",
        help="Require OpenAQ PM2.5 columns",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require all optional columns (EVI, S5P, OSM, OpenAQ)",
    )

    args = parser.parse_args(argv)

    require_evi = args.require_evi or args.require_all
    require_s5p = args.require_s5p or args.require_all
    require_osm_industrial = args.require_osm_industrial or args.require_all
    require_osm_dust = args.require_osm_dust or args.require_all
    require_openaq = args.require_openaq or args.require_all

    return validate_directory(
        cache_root=args.cache_root,
        pattern=args.pattern,
        max_files=args.max_files,
        require_toa=not args.no_toa,
        require_indices=not args.no_indices,
        require_evi=require_evi,
        require_s5p=require_s5p,
        require_osm_industrial=require_osm_industrial,
        require_osm_dust=require_osm_dust,
        require_openaq=require_openaq,
    )


if __name__ == "__main__":
    sys.exit(main())
