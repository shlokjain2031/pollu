"""Create a 30 m centroid grid for Mumbai wards and save as GeoParquet.

Usage:
    python scripts/create_grid_mumbai.py

This script defaults to reading `resources/mumbai_wards.geojson` and writing
`data/mumbai/grid_30m.parquet` in the project root. Adjust paths with the
command-line options if needed.
"""

from pathlib import Path
import argparse
import numpy as np

import geopandas as gpd
from shapely.geometry import Point


def create_grid(
    boundary_path: Path,
    out_path: Path,
    target_crs: str = "EPSG:32643",
    res: float = 30.0,
    max_cells: int = 3_000_000,
):
    gdf = gpd.read_file(boundary_path)
    city_proj = gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = city_proj.total_bounds

    xs = np.arange(minx + res / 2.0, maxx, res)
    ys = np.arange(miny + res / 2.0, maxy, res)
    estimated = len(xs) * len(ys)
    if estimated > max_cells:
        raise RuntimeError(
            f"Estimated grid cells ({estimated}) > max_cells ({max_cells}). Abort to avoid OOM"
        )

    xx, yy = np.meshgrid(xs, ys)
    xx = xx.ravel()
    yy = yy.ravel()
    pts = [Point(x, y) for x, y in zip(xx, yy)]

    grid_gdf = gpd.GeoDataFrame({"x": xx, "y": yy}, geometry=pts, crs=target_crs)

    # Clip to boundary union
    city_union = city_proj.union_all
    mask = [pt.within(city_union) for pt in grid_gdf.geometry]
    grid_gdf = grid_gdf[mask].reset_index(drop=True)
    grid_gdf["grid_id"] = grid_gdf.index.astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_gdf.to_parquet(out_path, index=False)
    print(f"Saved grid ({len(grid_gdf)} points) to {out_path}")


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
