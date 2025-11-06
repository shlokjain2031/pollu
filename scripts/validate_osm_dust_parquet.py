#!/usr/bin/env python3
"""Validate an OSM dust parquet produced by `aggregate_osm_dust_30m.py`.

Checks performed:
- file exists and can be read as a GeoDataFrame
- required columns present
- CRS is EPSG:32643 (warn if different)
- geometry areas are ~900 m^2 (30x30) within tolerance
- numeric sanity checks (non-negative lengths/areas, percent ranges, index ranges)
- reasonable non-empty counts

Exits with code 0 if all checks pass, non-zero otherwise. Prints a concise report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np


REQUIRED_COLUMNS = [
    "tile_id",
    "geometry",
    "total_road_m",
    "unpaved_road_m",
    "pct_unpaved",
    "road_weighted",
    "road_dust_index",
    "construction_area_m2",
    "construction_count",
    "construction_dust_index",
    "dust_index_combined",
]


def validate(gdf_path: Path) -> int:
    failures: List[str] = []

    if not gdf_path.exists():
        print(f"ERROR: file not found: {gdf_path}")
        return 2

    try:
        gdf = gpd.read_parquet(gdf_path)
    except Exception as e:
        print(f"ERROR: failed to read parquet: {e}")
        return 3

    print(f"Read GeoDataFrame shape: {gdf.shape}")

    # required cols
    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        failures.append(f"missing columns: {missing}")

    # geometry / crs
    if "geometry" not in gdf.columns:
        failures.append("no geometry column")
    else:
        try:
            if gdf.crs is None:
                print("WARNING: GeoDataFrame has no CRS set")
            else:
                crs_str = str(gdf.crs)
                print(f"CRS: {crs_str}")
                if "32643" not in crs_str and "EPSG:32643" not in crs_str:
                    print(
                        "WARNING: expected CRS EPSG:32643 (UTM43N) for 30m grid; got different CRS"
                    )

            # area check: tile should be about 900 m^2
            areas = gdf.geometry.area.to_numpy(dtype=float)
            median_area = float(np.nanmedian(areas))
            print(f"Median tile area (m^2): {median_area:.1f}")
            if not (700 <= median_area <= 1100):
                failures.append(
                    f"unexpected median tile area: {median_area:.1f} m^2 (expected ~900)"
                )
        except Exception as e:
            failures.append(f"geometry/area check failed: {e}")

    # Numeric sanity checks
    def nonneg(col):
        if col not in gdf.columns:
            return
        vals = gdf[col].to_numpy(dtype=float)
        if np.any(vals < -1e-6):
            failures.append(f"negative values in {col}: min={float(np.nanmin(vals))}")

    nonneg("total_road_m")
    nonneg("unpaved_road_m")
    nonneg("road_weighted")
    nonneg("construction_area_m2")
    nonneg("construction_count")

    # relational checks
    if "total_road_m" in gdf.columns and "unpaved_road_m" in gdf.columns:
        tr = gdf["total_road_m"].to_numpy(dtype=float)
        ur = gdf["unpaved_road_m"].to_numpy(dtype=float)
        bad = np.where(ur - tr > 1e-6)[0]
        if bad.size > 0:
            failures.append(
                f"{bad.size} tiles have unpaved_road_m > total_road_m (first idx {int(bad[0])})"
            )

    # range checks
    def range_check(col, lo, hi):
        if col not in gdf.columns:
            return
        vals = gdf[col].to_numpy(dtype=float)
        if np.nanmax(vals) > hi + 1e-6 or np.nanmin(vals) < lo - 1e-6:
            failures.append(
                f"{col} out of expected range [{lo},{hi}]: min={float(np.nanmin(vals))}, max={float(np.nanmax(vals))}"
            )

    range_check("pct_unpaved", 0.0, 100.0)
    range_check("road_dust_index", 0.0, 100.0)
    range_check("construction_dust_index", 0.0, 100.0)
    range_check("dust_index_combined", 0.0, 100.0)

    # Missingness
    nan_report = {}
    for c in REQUIRED_COLUMNS:
        if c in gdf.columns:
            n_null = int(gdf[c].isna().sum())
            nan_report[c] = n_null
    print("Null counts (per column):")
    for k, v in nan_report.items():
        print(f"  {k}: {v}")

    # Summary stats
    try:
        total_tiles = len(gdf)
        tiles_with_roads = (
            int((gdf["total_road_m"].fillna(0) > 0).sum())
            if "total_road_m" in gdf.columns
            else 0
        )
        tiles_with_constr = (
            int((gdf["construction_area_m2"].fillna(0) > 0).sum())
            if "construction_area_m2" in gdf.columns
            else 0
        )
        print(
            f"Total tiles: {total_tiles}; tiles with roads: {tiles_with_roads}; tiles with construction area: {tiles_with_constr}"
        )
        # percentiles of combined index
        if "dust_index_combined" in gdf.columns:
            arr = gdf["dust_index_combined"].fillna(0).to_numpy(dtype=float)
            qs = np.nanpercentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
            print("Combined index percentiles: 0,1,5,25,50,75,95,99,100 ->")
            print(" ", ", ".join([f"{q:.2f}" for q in qs]))
    except Exception as e:
        failures.append(f"summary stats failed: {e}")

    # Print sample rows
    print("\nSample rows (first 10):")
    try:
        with gpd.option_context("display.max_columns", None):
            print(gdf.head(10).to_string())
    except Exception:
        print(gdf.head(10))

    # Print one random row to help spot-check non-head tiles
    try:
        if len(gdf) > 0:
            rng = np.random.default_rng()
            ridx = int(rng.integers(0, len(gdf)))
            print(f"\nRandom row (index {ridx}):")
            try:
                with gpd.option_context("display.max_columns", None):
                    print(gdf.iloc[[ridx]].to_string(index=False))
            except Exception:
                print(gdf.iloc[[ridx]])
    except Exception as e:
        print("Could not print random row:", e)

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(" -", f)
        return 4
    else:
        print("\nVALIDATION PASSED: basic checks OK")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate OSM dust parquet (30m tiles)"
    )
    parser.add_argument("parquet", help="path to parquet file")
    args = parser.parse_args()
    rc = validate(Path(args.parquet))
    sys.exit(rc)


if __name__ == "__main__":
    main()
