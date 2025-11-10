#!/usr/bin/env python3
"""Aggregate OSM industrial-emissions signals to a 30 m tile grid.

Reads a local OSM PBF (pyrosm) and computes per-tile industrial proxies such as
industrial area, industrial building area/count, estimated building volume,
power-plant and chimney counts, landfill/quarry area, and a combined index.

Writes a Parquet (and optional GPKG) with per-tile metrics.

Example:
  python3 scripts/aggregate_osm_industrial_30m.py --pbf /path/Bombay.osm.pbf \
      --bbox 72.78,18.95,72.90,19.06 --out /tmp/mumbai_industrial.parquet --gpkg /tmp/mumbai_industrial.gpkg
"""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from typing import Tuple

import geopandas as gpd
import numpy as np

try:
    from pyrosm import OSM
except Exception:
    OSM = None
import pandas as pd

from shapely.geometry import box
from shapely.validation import make_valid as _make_valid


DEFAULT_CRS = "EPSG:32643"
CELL_SIZE = 30


def build_30m_grid(bbox_wgs84: Tuple[float, float, float, float], crs=DEFAULT_CRS, cell_size=CELL_SIZE) -> gpd.GeoDataFrame:
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
            polys.append(box(x0, y0, x1, y1))
            ids.append(f"r{i}_c{j}")

    grid = gpd.GeoDataFrame({"tile_id": ids, "geometry": polys}, crs=crs)
    grid = grid[grid.intersects(bbox_proj)].reset_index(drop=True)
    return grid


def safe_make_valid(g):
    try:
        if g is None or g.is_empty:
            return g
        if g.is_valid:
            return g
    except Exception:
        pass
    try:
        mg = _make_valid(g)
        if mg is not None and not mg.is_empty:
            return mg
    except Exception:
        pass
    try:
        mg = g.buffer(0)
        if mg is not None and not mg.is_empty:
            return mg
    except Exception:
        pass
    return g


def fetch_osm_for_industry(pbf_path: str, bbox: Tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
    """Read industrial-relevant features from PBF. Returns GeoDataFrame in WGS84."""
    if OSM is None:
        raise RuntimeError("pyrosm is required. Install with `pip install pyrosm`")

    osm = OSM(pbf_path)

    # tags we care about (broad)
    tags = {
        "landuse": True,
        "building": True,
        "power": True,
        "man_made": True,
        "industrial": True,
        "site": True,
        "amenity": True,
    }

    try:
        gdf = osm.get_data_by_custom_criteria(tags=tags, filter_type="any")
    except Exception:
        # fallbacks: buildings, landuse, points
        parts = []
        try:
            parts.append(osm.get_buildings())
        except Exception:
            pass
        try:
            parts.append(osm.get_landuse())
        except Exception:
            pass
        try:
            parts.append(osm.get_pois())
        except Exception:
            pass
        if parts:
            gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True)).set_crs("EPSG:4326", allow_override=True)
        else:
            gdf = gpd.GeoDataFrame(columns=["geometry"])

    if not gdf.empty and "geometry" in gdf.columns:
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

    if bbox is not None and not gdf.empty:
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        gdf = gdf[~gdf.geometry.is_empty].copy()
        gdf = gdf[gdf.geometry.intersects(bbox_poly)]

    return gdf


def compute_industrial_aggregates(gdf: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, crs=DEFAULT_CRS) -> gpd.GeoDataFrame:
    # Prepare accumulators
    accum = defaultdict(lambda: {
        "industrial_area_m2": 0.0,
        "industrial_building_area_m2": 0.0,
        "industrial_building_count": 0,
        "industrial_building_volume_m3": 0.0,
        "power_plant_count": 0,
        "chimney_count": 0,
        "landfill_area_m2": 0.0,
        "quarry_area_m2": 0.0,
        "industrial_poi_count": 0,
        "industrial_contrib_count": 0,
    })

    if gdf.empty:
        # ensure all zero columns
        for col in list(accum[0].keys()):
            grid[col] = 0
        grid["industrial_index_raw"] = 0.0
        grid["industrial_index_combined"] = 0.0
        grid["industrial_confidence"] = "low"
        return grid

    # categorize features
    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    points = gdf[gdf.geometry.type == "Point"].copy()

    # reproject
    polys = polys.to_crs(crs)
    lines = lines.to_crs(crs)
    points = points.to_crs(crs)

    # spatial index on grid
    try:
        from rtree import index as rindex
    except Exception:
        raise RuntimeError("rtree is required; install rtree")
    ridx = rindex.Index()
    for i, row in grid.iterrows():
        ridx.insert(i, row.geometry.bounds)

    def cands(geom):
        return list(ridx.intersection(geom.bounds))

    # helper to estimate building height
    def estimate_height(row):
        # try building:height then building:levels*3 else fallback 6 m
        h = None
        if "building:height" in row and row.get("building:height"):
            try:
                h = float(str(row.get("building:height")).split(";")[0])
            except Exception:
                h = None
        if h is None and "building:levels" in row and row.get("building:levels"):
            try:
                h = float(row.get("building:levels")) * 3.0
            except Exception:
                h = None
        if h is None:
            h = 6.0
        return h

    # process polygons
    for idx, row in polys.iterrows():
        geom = safe_make_valid(row.geometry)
        if geom is None or geom.is_empty:
            continue
        tags = {k: row.get(k) for k in row.index if isinstance(k, str)}
        cand_ids = cands(geom)
        for tid in cand_ids:
            tile_geom = grid.loc[tid].geometry
            try:
                inter = geom.intersection(tile_geom)
            except Exception:
                inter = safe_make_valid(geom).intersection(safe_make_valid(tile_geom))
            if inter.is_empty:
                continue
            area = inter.area
            if area <= 0:
                continue
            # landuse industrial
            lu = (str(tags.get("landuse")) if tags.get("landuse") is not None else "").lower()
            if lu == "industrial":
                accum[tid]["industrial_area_m2"] += area
                accum[tid]["industrial_contrib_count"] += 1
            if lu == "landfill":
                accum[tid]["landfill_area_m2"] += area
            if lu == "quarry":
                accum[tid]["quarry_area_m2"] += area
            # building footprints
            btag = (str(tags.get("building")) if tags.get("building") is not None else "").lower()
            if btag in {"industrial", "factory", "works"}:
                accum[tid]["industrial_building_area_m2"] += area
                accum[tid]["industrial_building_count"] += 1
                h = estimate_height(tags)
                accum[tid]["industrial_building_volume_m3"] += area * h
                accum[tid]["industrial_contrib_count"] += 1

    # process points and POIs
    for idx, row in points.iterrows():
        geom = row.geometry
        tags = {k: row.get(k) for k in row.index if isinstance(k, str)}
        cand_ids = cands(geom)
        for tid in cand_ids:
            tile_geom = grid.loc[tid].geometry
            if geom.within(tile_geom):
                # power plant node?
                if str(tags.get("power")).lower() == "plant":
                    accum[tid]["power_plant_count"] += 1
                    accum[tid]["industrial_contrib_count"] += 1
                if str(tags.get("man_made")).lower() == "chimney" or str(tags.get("chimney")).lower() == "yes":
                    accum[tid]["chimney_count"] += 1
                    accum[tid]["industrial_contrib_count"] += 1
                # node tagged industrial/site
                if (str(tags.get("industrial")).lower() == "yes") or (str(tags.get("site")).lower() == "industrial"):
                    accum[tid]["industrial_poi_count"] += 1
                    accum[tid]["industrial_contrib_count"] += 1

    # Also inspect lines for power infrastructure (rare) and count chimney-like tags
    for idx, row in lines.iterrows():
        geom = row.geometry
        tags = {k: row.get(k) for k in row.index if isinstance(k, str)}
        cand_ids = cands(geom)
        for tid in cand_ids:
            tile_geom = grid.loc[tid].geometry
            try:
                piece = geom.intersection(tile_geom)
            except Exception:
                piece = safe_make_valid(geom).intersection(safe_make_valid(tile_geom))
            if piece.is_empty:
                continue
            # if line tagged power=plant or man_made chimney (unlikely), count it
            if str(tags.get("power")).lower() == "plant":
                accum[tid]["power_plant_count"] += 1
                accum[tid]["industrial_contrib_count"] += 1

    # assemble columns
    industrial_area = []
    industrial_building_area = []
    industrial_building_count = []
    industrial_building_volume = []
    power_plant_count = []
    chimney_count = []
    landfill_area = []
    quarry_area = []
    industrial_poi_count = []
    industrial_contrib_count = []
    industrial_index_raw = []

    for i, row in grid.iterrows():
        a = accum.get(i, {})
        ia = a.get("industrial_area_m2", 0.0)
        iba = a.get("industrial_building_area_m2", 0.0)
        ibc = a.get("industrial_building_count", 0)
        ibv = a.get("industrial_building_volume_m3", 0.0)
        ppc = a.get("power_plant_count", 0)
        chc = a.get("chimney_count", 0)
        lfa = a.get("landfill_area_m2", 0.0)
        qra = a.get("quarry_area_m2", 0.0)
        ipoi = a.get("industrial_poi_count", 0)
        icc = a.get("industrial_contrib_count", 0)

        industrial_area.append(ia)
        industrial_building_area.append(iba)
        industrial_building_count.append(ibc)
        industrial_building_volume.append(ibv)
        power_plant_count.append(ppc)
        chimney_count.append(chc)
        landfill_area.append(lfa)
        quarry_area.append(qra)
        industrial_poi_count.append(ipoi)
        industrial_contrib_count.append(icc)

        # raw score (weights can be tuned)
        raw = 0.0
        raw += (ibv / 1000.0) * 1.0  # volume per 1000 m3
        raw += (ia / 1000.0) * 0.8  # industrial area per 1000 m2
        raw += ppc * 50.0
        raw += chc * 20.0
        raw += (lfa / 1000.0) * 0.2
        raw += (qra / 1000.0) * 0.2
        raw += ipoi * 5.0

        industrial_index_raw.append(raw)

    grid["industrial_area_m2"] = industrial_area
    grid["industrial_building_area_m2"] = industrial_building_area
    grid["industrial_building_count"] = industrial_building_count
    grid["industrial_building_volume_m3"] = industrial_building_volume
    grid["power_plant_count"] = power_plant_count
    grid["chimney_count"] = chimney_count
    grid["landfill_area_m2"] = landfill_area
    grid["quarry_area_m2"] = quarry_area
    grid["industrial_poi_count"] = industrial_poi_count
    grid["industrial_contrib_count"] = industrial_contrib_count
    grid["industrial_index_raw"] = industrial_index_raw

    # normalize raw -> 0..100 using 99th percentile cap to reduce outlier influence
    arr = np.array(industrial_index_raw, dtype=float)
    if arr.size == 0:
        grid["industrial_index_combined"] = 0.0
    else:
        p99 = float(np.nanpercentile(arr, 99))
        if p99 <= 0:
            p99 = float(arr.max()) if arr.max() > 0 else 1.0
        norm = np.clip((arr / p99) * 100.0, 0.0, 100.0)
        grid["industrial_index_combined"] = norm

    # confidence
    conf = []
    for i, row in grid.iterrows():
        if row["power_plant_count"] > 0 or row["chimney_count"] > 0 or row["industrial_building_area_m2"] > 0:
            conf.append("high")
        elif row["industrial_area_m2"] > 0 or row["industrial_poi_count"] > 0:
            conf.append("medium")
        else:
            conf.append("low")
    grid["industrial_confidence"] = conf

    return grid


def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be minx,miny,maxx,maxy")
    return parts[0], parts[1], parts[2], parts[3]


def main():
    parser = argparse.ArgumentParser(description="Aggregate OSM industrial signals at 30 m tiles")
    parser.add_argument("--bbox", help="bbox minx,miny,maxx,maxy in WGS84", default=None)
    parser.add_argument("--pbf", help="local OSM PBF file to read (required)", required=True)
    parser.add_argument("--out", help="output parquet path", default="outputs/osm_industrial_30m.parquet")
    parser.add_argument("--gpkg", help="also write geopackage", default=None)
    args = parser.parse_args()

    if args.bbox is None:
        # read broad features and compute bounds from PBF
        print("No bbox provided: reading PBF to infer bounds (may be slow for large PBFs)...")
        gdf = fetch_osm_for_industry(args.pbf, bbox=None)
        if gdf.empty:
            raise RuntimeError("Could not infer bbox from PBF; please provide --bbox")
        bounds = gdf.total_bounds  # minx,miny,maxx,maxy
        bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    else:
        bbox = parse_bbox(args.bbox)

    print(f"Using bbox: {bbox}")
    print(f"Reading PBF: {args.pbf}")
    gdf = fetch_osm_for_industry(args.pbf, bbox=bbox)

    print("Building 30m grid...")
    grid = build_30m_grid(bbox)

    print("Aggregating industrial features into tiles...")
    grid_out = compute_industrial_aggregates(gdf, grid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Writing parquet: {args.out}")
    grid_out.to_parquet(args.out, index=False)
    if args.gpkg:
        print(f"Writing geopackage: {args.gpkg}")
        grid_out.to_file(args.gpkg, layer="tiles", driver="GPKG")

    print("Done.")


if __name__ == "__main__":
    main()
