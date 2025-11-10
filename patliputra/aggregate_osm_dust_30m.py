#!/usr/bin/env python3
"""Aggregate OSM dust signals to a 30 m tile grid.

Usage examples:
  python3 scripts/aggregate_osm_dust_30m.py --place "Mumbai, India" --out outputs/mumbai_osm_dust.parquet
  python3 scripts/aggregate_osm_dust_30m.py --bbox 72.5,18.8,73.2,19.4 --out outputs/mumbai_osm_dust.parquet

This script performs:
 - fetch OSM geometries (roads, constructions, landuse) via Overpass (osmnx)
 - build a 30 m grid in EPSG:32643 (UTM 43N)
 - for each way: intersect with overlapping tiles, accumulate length and dust-weighted length
 - for nodes/polys: accumulate construction counts and area per tile
 - produce a parquet (and geopackage) with per-tile aggregates and dust indexes

Notes:
 - For large areas prefer using a local PBF + pyrosm; this script uses Overpass and is best for city-scale.
 - Make sure `osmnx`, `geopandas`, `shapely`, `rtree`, and `pyproj` are installed.
"""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from typing import Tuple, List

import geopandas as gpd
import numpy as np

try:
    from pyrosm import OSM
except Exception:
    OSM = None
import pandas as pd
from shapely.geometry import box, LineString, Point, Polygon
from shapely.ops import unary_union
import warnings

try:
    # shapely >= 1.8
    from shapely.validation import make_valid
except Exception:
    make_valid = None

DEFAULT_CRS = "EPSG:32643"  # UTM zone 43N (Mumbai)
CELL_SIZE = 30  # meters


def build_30m_grid(
    bbox_wgs84: Tuple[float, float, float, float], crs=DEFAULT_CRS, cell_size=CELL_SIZE
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox_wgs84
    bbox_poly = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_poly]}, crs="EPSG:4326")
    bbox_proj = bbox_gdf.to_crs(crs).geometry[0]
    minx_m, miny_m, maxx_m, maxy_m = bbox_proj.bounds

    nx = int(math.ceil((maxx_m - minx_m) / cell_size))
    ny = int(math.ceil((maxy_m - miny_m) / cell_size))

    polys = []
    ids = []
    for i in range(nx):
        for j in range(ny):
            x0 = minx_m + i * cell_size
            y0 = miny_m + j * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            p = box(x0, y0, x1, y1)
            polys.append(p)
            ids.append(f"r{i}_c{j}")

    grid = gpd.GeoDataFrame({"tile_id": ids, "geometry": polys}, crs=crs)
    # clip grid to bbox exact extent
    grid = grid[grid.intersects(bbox_proj)].reset_index(drop=True)
    return grid


def fetch_osm_features_pbf_strict(
    pbf_path: str, bbox: Tuple[float, float, float, float] | None
) -> gpd.GeoDataFrame:
    """Read relevant OSM features from a local PBF using pyrosm and filter to bbox (WGS84).

    Returns a GeoDataFrame in EPSG:4326 containing nodes/ways/areas with tags we care about.
    """
    if OSM is None:
        raise RuntimeError(
            "pyrosm is required for PBF processing. Install with `pip install pyrosm`"
        )

    osm = OSM(pbf_path)

    # preferred: get data by custom criteria (pyrosm >=0.9 API)
    tags = {
        "highway": True,
        "surface": True,
        "tracktype": True,
        "smoothness": True,
        "lanes": True,
        "maxspeed": True,
        "construction": True,
        "landuse": True,
        "building:construction": True,
        "site": True,
        "amenity": True,
        "industrial": True,
        "natural": True,
    }

    try:
        gdf = osm.get_data_by_custom_criteria(tags=tags, filter_type="any")
    except Exception:
        # fallback: collect common layers
        parts = []
        try:
            roads = osm.get_network(network_type="driving")
            parts.append(roads)
        except Exception:
            pass
        try:
            buildings = osm.get_buildings()
            parts.append(buildings)
        except Exception:
            pass
        try:
            landuse = osm.get_landuse()
            parts.append(landuse)
        except Exception:
            pass
        if parts:
            import pandas as _pd

            gdf = _pd.concat(parts, ignore_index=True)
            gdf = gpd.GeoDataFrame(gdf)
        else:
            gdf = gpd.GeoDataFrame(columns=["geometry"])

    # Ensure geometry column and CRS
    if not gdf.empty and "geometry" in gdf.columns:
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

    # bbox filter if requested
    if bbox is not None and not gdf.empty:
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        gdf = gdf[~gdf.geometry.is_empty].copy()
        gdf = gdf[gdf.geometry.intersects(bbox_poly)]

    return gdf


def fetch_osm_features_pbf(
    pbf_path: str, bbox: Tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Fetch features from a local PBF using pyrosm and filter to bbox.

    This is the preferred path for large areas. Returns a GeoDataFrame in WGS84.
    """
    try:
        from pyrosm import OSM
    except Exception as e:
        raise RuntimeError(
            "pyrosm is required to read PBF files. Install with `pip install pyrosm`"
        ) from e

    north, south, east, west = bbox[3], bbox[1], bbox[2], bbox[0]
    osm = OSM(pbf_path)

    # tags of interest similar to Overpass path
    tags = {
        "highway": True,
        "surface": True,
        "tracktype": True,
        "smoothness": True,
        "lanes": True,
        "maxspeed": True,
        "construction": True,
        "landuse": True,
        "building:construction": True,
        "site": True,
        "amenity": True,
        "industrial": True,
        "natural": True,
    }

    # pyrosm provides get_data_by_custom_criteria which returns nodes/ways/polygons
    try:
        gdf = osm.get_data_by_custom_criteria(tags=tags, filter_type="any")
    except TypeError:
        # fallback if API signature differs
        # try to collect typical layers
        parts = []
        try:
            roads = osm.get_network(network_type="driving")
            parts.append(roads)
        except Exception:
            pass
        try:
            buildings = osm.get_buildings()
            parts.append(buildings)
        except Exception:
            pass
        try:
            landuse = osm.get_landuse()
            parts.append(landuse)
        except Exception:
            pass
        if parts:
            gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True)).set_crs(
                "EPSG:4326", allow_override=True
            )
        else:
            gdf = gpd.GeoDataFrame(columns=["geometry"])

    # filter by bbox
    bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    if not gdf.empty and "geometry" in gdf.columns:
        gdf = gdf[~gdf.geometry.is_empty].copy()
        # ensure CRS and filter
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        gdf = gdf[gdf.geometry.intersects(bbox_poly)]

    return gdf


def default_weights():
    return {
        "unpaved": 1.0,
        "dirt": 0.95,
        "gravel": 0.9,
        "sand": 0.9,
        "compacteddirt": 0.95,
        "ground": 0.8,
        "clay": 0.8,
        "pebblestone": 0.7,
        "rubble": 0.6,
        "track": 0.7,
        "residential": 0.35,
        "service": 0.35,
        "unclassified": 0.3,
        "paved": 0.05,
        "asphalt": 0.05,
        "concrete": 0.05,
    }


def compute_tile_aggregates(
    gdf: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, crs=DEFAULT_CRS
):
    # Prepare outputs
    tile_index = {i: row for i, row in grid.iterrows()}
    # accumulators per tile index
    accum = defaultdict(
        lambda: {
            "total_road_m": 0.0,
            "unpaved_road_m": 0.0,
            "road_weighted": 0.0,
            "construction_area_m2": 0.0,
            "construction_count": 0,
        }
    )

    weights = default_weights()

    # split gdf into lines, points, polys
    lines = gdf[
        gdf.geometry.type.isin(["LineString", "MultiLineString"])
        & gdf["highway"].notna()
    ].copy()
    points = gdf[gdf.geometry.type == "Point"].copy()
    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # reproject to metric
    lines = lines.to_crs(crs)
    points = points.to_crs(crs)
    polys = polys.to_crs(crs)

    # build rtree on grid for fast candidate selection
    try:
        from rtree import index as rindex
    except Exception:
        raise RuntimeError("rtree is required for spatial indexing; install rtree")

    ridx = rindex.Index()
    for i, row in grid.iterrows():
        ridx.insert(i, row.geometry.bounds)

    # helper: find candidate tile indices for geometry
    def candidate_tiles(geom):
        return list(ridx.intersection(geom.bounds))

    def safe_geom(g):
        """Return a geometry that is valid where possible (repair if invalid).

        Uses shapely.make_valid when available, falls back to buffer(0). If repair
        fails, returns the original geometry.
        """
        if g is None or g.is_empty:
            return g
        try:
            if g.is_valid:
                return g
        except Exception:
            # some geometries may raise on is_valid; try repair
            pass
        # try make_valid first
        if make_valid is not None:
            try:
                mg = make_valid(g)
                if mg is not None and not mg.is_empty:
                    return mg
            except Exception:
                pass
        # fallback to buffer(0)
        try:
            mg = g.buffer(0)
            if mg is not None and not mg.is_empty:
                return mg
        except Exception:
            pass
        # give up
        return g

    # Process lines (roads)
    dusty_surfaces = set(
        [
            "unpaved",
            "dirt",
            "gravel",
            "sand",
            "compacteddirt",
            "ground",
            "clay",
            "pebblestone",
            "rubble",
        ]
    )

    for idx, row in lines.iterrows():
        geom = row.geometry
        geom = safe_geom(geom)
        tags = {
            k: row.get(k)
            for k in [
                "surface",
                "highway",
                "tracktype",
                "smoothness",
                "maxspeed",
                "lanes",
            ]
        }
        cands = candidate_tiles(geom)
        for tid in cands:
            tile_geom = grid.loc[tid].geometry
            try:
                piece = geom.intersection(tile_geom)
            except Exception:
                # retry with repaired geometries
                piece = safe_geom(geom).intersection(safe_geom(tile_geom))
            if piece.is_empty:
                continue
            # piece may be multilinestring or geometrycollection; calculate total length
            piece_length = piece.length
            if piece_length <= 0:
                continue
            accum[tid]["total_road_m"] += piece_length

            surface = (
                str(tags.get("surface")) if tags.get("surface") is not None else ""
            ).lower()
            highway = (
                str(tags.get("highway")) if tags.get("highway") is not None else ""
            ).lower()

            # determine weight
            w = 0.0
            if surface in weights:
                w = weights[surface]
            elif highway in weights:
                w = weights[highway]
            else:
                # fallback heuristics
                if highway.startswith("track"):
                    w = 0.6
                else:
                    w = 0.1

            accum[tid]["road_weighted"] += piece_length * w
            if surface in dusty_surfaces:
                accum[tid]["unpaved_road_m"] += piece_length

    # Process polygons (construction / landuse)
    for idx, row in polys.iterrows():
        geom = row.geometry
        geom = safe_geom(geom)
        tags = {
            k: row.get(k)
            for k in [
                "construction",
                "landuse",
                "building:construction",
                "site",
                "industrial",
                "natural",
            ]
        }
        cands = candidate_tiles(geom)
        for tid in cands:
            tile_geom = grid.loc[tid].geometry
            try:
                inter = geom.intersection(tile_geom)
            except Exception:
                inter = safe_geom(geom).intersection(safe_geom(tile_geom))
            if inter.is_empty:
                continue
            area = inter.area
            if area <= 0:
                continue
            accum[tid]["construction_area_m2"] += area

    # Process points (construction nodes)
    for idx, row in points.iterrows():
        geom = row.geometry
        tags = {
            k: row.get(k) for k in ["construction", "site", "building:construction"]
        }
        cands = candidate_tiles(geom)
        for tid in cands:
            tile_geom = grid.loc[tid].geometry
            if geom.within(tile_geom):
                accum[tid]["construction_count"] += 1

    # Build results into grid dataframe
    total_road = []
    unpaved_road = []
    road_weighted = []
    construction_area = []
    construction_count = []
    road_index = []
    construction_index = []

    for i, row in grid.iterrows():
        a = accum.get(i, {})
        tr = a.get("total_road_m", 0.0)
        ur = a.get("unpaved_road_m", 0.0)
        rw = a.get("road_weighted", 0.0)
        ca = a.get("construction_area_m2", 0.0)
        cc = a.get("construction_count", 0)

        # normalized indices
        pct_unpaved = (100.0 * ur / tr) if tr > 0 else 0.0
        # road_index normalized to 0..100 using simple scale (tune later)
        # we normalize by cell diagonal length (approx) to keep comparable across tiles
        cell_diag = math.sqrt((CELL_SIZE**2) * 2)
        road_idx = min(100.0, (rw / (CELL_SIZE * cell_diag)) * 100.0)
        # construction index: fraction of tile area that is construction (0..100)
        tile_area = CELL_SIZE * CELL_SIZE
        constr_idx = min(100.0, (ca / tile_area) * 100.0)

        total_road.append(tr)
        unpaved_road.append(ur)
        road_weighted.append(rw)
        construction_area.append(ca)
        construction_count.append(cc)
        road_index.append(road_idx)
        construction_index.append(constr_idx)

    grid["total_road_m"] = total_road
    grid["unpaved_road_m"] = unpaved_road
    grid["pct_unpaved"] = np.where(
        np.array(total_road) > 0,
        100.0 * np.array(unpaved_road) / np.array(total_road),
        0.0,
    )
    grid["road_weighted"] = road_weighted
    grid["road_dust_index"] = road_index
    grid["construction_area_m2"] = construction_area
    grid["construction_count"] = construction_count
    grid["construction_dust_index"] = construction_index
    # combined simple index: weighted average (weights tunable)
    grid["dust_index_combined"] = (
        0.6 * grid["road_dust_index"] + 0.4 * grid["construction_dust_index"]
    )

    return grid


def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be minx,miny,maxx,maxy")
    return parts[0], parts[1], parts[2], parts[3]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate OSM dust signals at 30 m tiles"
    )
    parser.add_argument(
        "--bbox", help="bbox minx,miny,maxx,maxy in WGS84", default=None
    )
    parser.add_argument(
        "--pbf", help="local OSM PBF file to read instead of Overpass", default=None
    )
    parser.add_argument(
        "--out", help="output parquet path", default="outputs/osm_dust_30m.parquet"
    )
    parser.add_argument("--gpkg", help="also write geopackage", default=None)
    args = parser.parse_args()

    if args.bbox is None:
        parser.error("Please provide --bbox (minx,miny,maxx,maxy in WGS84)")

    bbox = parse_bbox(args.bbox)

    if not args.pbf:
        parser.error(
            "This script requires a local PBF file via --pbf for pyrosm-based processing"
        )

    print(f"Using local PBF: {args.pbf}")
    gdf = fetch_osm_features_pbf_strict(args.pbf, bbox)

    # build grid
    grid = build_30m_grid((bbox[0], bbox[1], bbox[2], bbox[3]))

    print("Aggregating features into 30 m tiles...")
    grid_out = compute_tile_aggregates(gdf, grid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Writing parquet: {args.out}")
    grid_out.to_parquet(args.out, index=False)
    if args.gpkg:
        print(f"Writing geopackage: {args.gpkg}")
        grid_out.to_file(args.gpkg, layer="tiles", driver="GPKG")

    print("Done.")


if __name__ == "__main__":
    main()
