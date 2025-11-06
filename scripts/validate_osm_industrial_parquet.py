#!/usr/bin/env python3
"""Validate an OSM industrial parquet produced by `aggregate_osm_industrial_30m.py`.

Performs schema, CRS, geometry area, numeric range and basic-content checks.
Prints a concise human report and exits 0 on success, non-zero on failure.
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


def validate(path: Path) -> int:
    failures: List[str] = []

    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 2

    try:
        gdf = gpd.read_parquet(path)
    except Exception as e:
        print(f"ERROR: failed to read parquet: {e}")
        return 3

    print(f"Read GeoDataFrame shape: {gdf.shape}")

    # required cols
    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        failures.append(f"missing columns: {missing}")

    # geometry and crs
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
                    print("WARNING: expected CRS EPSG:32643 (UTM43N) for 30m grid; got different CRS")

            areas = gdf.geometry.area.to_numpy(dtype=float)
            median_area = float(np.nanmedian(areas))
            print(f"Median tile area (m^2): {median_area:.1f}")
            if not (700 <= median_area <= 1100):
                failures.append(f"unexpected median tile area: {median_area:.1f} m^2 (expected ~900)")
        except Exception as e:
            failures.append(f"geometry/area check failed: {e}")

    # non-negative checks
    def nonneg(col):
        if col not in gdf.columns:
            return
        vals = gdf[col].to_numpy(dtype=float)
        if np.any(vals < -1e-6):
            failures.append(f"negative values in {col}: min={float(np.nanmin(vals))}")

    for col in [
        "industrial_area_m2",
        "industrial_building_area_m2",
        "industrial_building_volume_m3",
        "landfill_area_m2",
        "quarry_area_m2",
    ]:
        nonneg(col)

    for col in ["industrial_building_count", "power_plant_count", "chimney_count", "industrial_poi_count", "industrial_contrib_count"]:
        if col in gdf.columns:
            vals = gdf[col].fillna(0).to_numpy(dtype=float)
            if np.any(vals < 0):
                failures.append(f"negative counts in {col}")

    # index ranges
    def range_check(col, lo, hi):
        if col not in gdf.columns:
            return
        vals = gdf[col].to_numpy(dtype=float)
        if np.nanmax(vals) > hi + 1e-6 or np.nanmin(vals) < lo - 1e-6:
            failures.append(f"{col} out of expected range [{lo},{hi}]: min={float(np.nanmin(vals))}, max={float(np.nanmax(vals))}")

    range_check("industrial_index_combined", 0.0, 100.0)

    # confidence values check
    if "industrial_confidence" in gdf.columns:
        allowed = set(["high", "medium", "low"])
        vals = set([str(x).lower() for x in gdf["industrial_confidence"].dropna().unique()])
        bad = vals - allowed
        if bad:
            failures.append(f"industrial_confidence has unexpected values: {bad}")

    # null counts
    nan_report = {}
    for c in REQUIRED_COLUMNS:
        if c in gdf.columns:
            n_null = int(gdf[c].isna().sum())
            nan_report[c] = n_null
    print("Null counts (per column):")
    for k, v in nan_report.items():
        print(f"  {k}: {v}")

    # summary stats
    try:
        total_tiles = len(gdf)
        tiles_with_industrial_area = int((gdf["industrial_area_m2"].fillna(0) > 0).sum()) if "industrial_area_m2" in gdf.columns else 0
        tiles_with_buildings = int((gdf["industrial_building_area_m2"].fillna(0) > 0).sum()) if "industrial_building_area_m2" in gdf.columns else 0
        print(f"Total tiles: {total_tiles}; tiles with industrial area: {tiles_with_industrial_area}; tiles with industrial buildings: {tiles_with_buildings}")
        if "industrial_index_combined" in gdf.columns:
            arr = gdf["industrial_index_combined"].fillna(0).to_numpy(dtype=float)
            qs = np.nanpercentile(arr, [0,1,5,25,50,75,95,99,100])
            print('Industrial combined percentiles (0,1,5,25,50,75,95,99,100):')
            print(' ', ', '.join([f'{q:.2f}' for q in qs]))
    except Exception as e:
        failures.append(f"summary stats failed: {e}")

    # sample rows
    print("\nSample rows (first 10):")
    try:
        with gpd.option_context('display.max_columns', None):
            print(gdf.head(10).to_string())
    except Exception:
        print(gdf.head(10))

    # print all columns and dtypes for quick inspection
    try:
        print('\nAll columns:')
        print(list(gdf.columns))
        print('\nColumn dtypes:')
        print(gdf.dtypes)
    except Exception as e:
        print('Could not print columns/dtypes:', e)

    # random row: print each column on its own line as 'column: value'
    try:
        if len(gdf) > 0:
            rng = np.random.default_rng()
            ridx = int(rng.integers(0, len(gdf)))
            print(f"\nRandom row (index {ridx}):")
            row = gdf.iloc[ridx]
            def fmt(v):
                # None / nan
                try:
                    import pandas as _pd

                    if _pd.isna(v):
                        return None
                except Exception:
                    pass
                # geometry
                try:
                    if hasattr(v, "wkt"):
                        return v.wkt
                except Exception:
                    pass
                # numpy scalars -> python native
                try:
                    import numpy as _np

                    if isinstance(v, _np.generic):
                        try:
                            return v.item()
                        except Exception:
                            return float(v)
                    if isinstance(v, _np.ndarray):
                        try:
                            return v.tolist()
                        except Exception:
                            return str(v)
                except Exception:
                    pass
                # pandas types
                try:
                    import pandas as _pd

                    if isinstance(v, _pd.Timestamp):
                        return str(v)
                except Exception:
                    pass
                # fallback: plain python
                try:
                    return v
                except Exception as e:
                    return f"<error reading value: {e}>"

            for col in gdf.columns:
                try:
                    val = row[col]
                    valfmt = fmt(val)
                    print(f"{col}: {valfmt}")
                except Exception as e:
                    print(f"{col}: <error reading value: {e}>")
    except Exception as e:
        print('Could not print random row:', e)

    if failures:
        print('\nVALIDATION FAILED:')
        for f in failures:
            print(' -', f)
        return 4
    else:
        print('\nVALIDATION PASSED: basic checks OK')
        return 0


def main():
    parser = argparse.ArgumentParser(description='Validate OSM industrial parquet (30m tiles)')
    parser.add_argument('parquet', help='path to parquet file')
    args = parser.parse_args()
    rc = validate(Path(args.parquet))
    sys.exit(rc)


if __name__ == '__main__':
    main()
