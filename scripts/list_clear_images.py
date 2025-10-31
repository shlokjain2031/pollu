#!/usr/bin/env python3
"""List clear Landsat-8 C02 images for a given bbox and date range.

This script queries the Earth Engine ImageCollection and filters by the
`CLOUD_COVER` property. It prints the number of images found and the
first 200 image ids.

Usage: edit bbox/start/end/cloud_threshold in the script or call directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_env(path: Path):
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    load_env(repo_root / ".env")

    import ee

    # Initialize with explicit project if available
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    # Bounding box for Mumbai area (lon/lat)
    bbox = {
        "minx": 72.77633295153348,
        "miny": 18.89395643371942,
        "maxx": 72.97973149704592,
        "maxy": 19.270176667777736,
    }

    start = "2013-01-01"
    end = "2025-10-31"
    cloud_threshold = 10  # percent
    collection_id = "LANDSAT/LC08/C02/T1_TOA"

    geom = ee.Geometry.Rectangle(
        [bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]]
    )
    col = ee.ImageCollection(collection_id).filterDate(start, end).filterBounds(geom)

    import argparse

    # Filter by CLOUD_COVER property
    try:
        filtered = col.filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold))
    except Exception:
        # Some collections may expose CLOUD_COVER_LAND instead
        filtered = col.filter(ee.Filter.lt("CLOUD_COVER_LAND", cloud_threshold))

    size = filtered.size().getInfo()
    print(
        json.dumps(
            {
                "collection": collection_id,
                "start": start,
                "end": end,
                "bbox": bbox,
                "cloud_threshold": cloud_threshold,
                "count": size,
            },
            indent=2,
        )
    )

    n = min(size, 200)
    if n > 0:
        lst = filtered.sort("system:time_start").toList(n)
        ids = []
        # Allow overriding/deriving bbox from a GeoJSON file (useful for other cities)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--geojson", help="Path to wards GeoJSON (will derive bbox)", default=None
        )
        args = parser.parse_args()

        # Default hardcoded bbox for Mumbai (conservative)
        default_bbox = {
            "minx": 72.77633295153348,
            "miny": 18.89395643371942,
            "maxx": 72.97973149704592,
            "maxy": 19.270176667777736,
        }

        def bbox_from_geojson(path: Path):
            obj = json.loads(path.read_text())
            minx = float("inf")
            miny = float("inf")
            maxx = float("-inf")
            maxy = float("-inf")

            def proc(coords):
                nonlocal minx, miny, maxx, maxy
                if isinstance(coords[0], list):
                    for c in coords:
                        proc(c)
                else:
                    x, y = coords
                    minx = min(minx, x)
                    miny = min(miny, y)
                    maxx = max(maxx, x)
                    maxy = max(maxy, y)

            for feat in obj.get("features", []):
                geom = feat.get("geometry")
                proc(geom["coordinates"])
            return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}

        # Determine which GeoJSON to use: CLI arg -> resources/mumbai_wards.geojson -> None
        geojson_path = None
        if args.geojson:
            geojson_path = Path(args.geojson)
        else:
            candidate = repo_root / "resources" / "mumbai_wards.geojson"
            if candidate.exists():
                geojson_path = candidate

        if geojson_path and geojson_path.exists():
            try:
                bbox = bbox_from_geojson(geojson_path)
                print(f"Using bbox derived from {geojson_path}: {bbox}")
            except Exception as e:
                print(
                    f"Failed to compute bbox from {geojson_path}, falling back to default: {e}"
                )
                bbox = default_bbox
        else:
            bbox = default_bbox
            info = item.getInfo()
            ids.append(info.get("id"))
        print(json.dumps({"ids_count": len(ids), "ids": ids}, indent=2))


if __name__ == "__main__":
    main()
