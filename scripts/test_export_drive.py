#!/usr/bin/env python3
"""Test script for EarthEngineClient.export_to_drive

This script attempts to start a Drive export for a small Landsat TOA image
and prints the returned task status or the error.
"""
import sys
import traceback
from pathlib import Path

try:
    from patliputra.backends.earthengine import EarthEngineClient
    import ee
except Exception as e:
    print("Failed to import EarthEngineClient or ee:", e)
    traceback.print_exc()
    sys.exit(2)


def main():
    ee_client = EarthEngineClient()
    try:
        ee_client.initialize()
    except Exception as e:
        print("EE initialize failed:", e)
        traceback.print_exc()
        # continue to see behavior

    # Mumbai small bbox (same as fetch_landsat56)
    bbox = {"minx": 72.7763, "miny": 18.8939, "maxx": 72.9797, "maxy": 19.2701}

    try:
        # Build an image for one day (may produce an image even if single-scene)
        img = ee_client.landsat_toa_image(
            collection="LANDSAT/LC08/C02/T1_TOA",
            bbox=bbox,
            start="2019-01-04",
            end="2019-01-05",
            reducer="median",
        )
    except Exception as e:
        print("Failed to build landsat image:", e)
        traceback.print_exc()
        sys.exit(3)

    try:
        region = ee.Geometry.Rectangle(
            [bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]]
        )
        status = ee_client.export_to_drive(
            img,
            description="test_export_mumbai_20190101",
            folder=None,
            file_name_prefix="test_export_mumbai_20190101",
            region=region,
            crs="EPSG:32643",
            scale=30,
            wait=False,
        )
        print("Export task started; status:", status)
    except Exception as e:
        print("Export to Drive failed:", e)
        traceback.print_exc()
        sys.exit(4)


if __name__ == "__main__":
    main()
