#!/usr/bin/env python3
"""
Fetch all cloud-free Landsat-8 B2 image dates for the Mumbai bounding box
and save them to a file under:
    /Users/shlokjain/pollu/patliputra/signals/toa_b2/image_dates.txt
"""

import argparse
import ee
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch cloud-free Landsat-8 image dates for Mumbai"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Earth Engine project ID",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("patliputra/landsat8_image_dates.txt"),
        help="Output file path for image dates",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="72.7763,18.8939,72.9797,19.2701",
        help="Bounding box as west,south,east,north",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-11-01",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=15.0,
        help="Maximum cloud cover percentage",
    )
    args = parser.parse_args()

    print(f"Initializing Earth Engine with project: {args.project}")
    ee.Initialize(project=args.project)

    # Parse bbox
    bbox_parts = [float(x.strip()) for x in args.bbox.split(",")]
    if len(bbox_parts) != 4:
        print("Error: bbox must be west,south,east,north")
        return
    bbox = ee.Geometry.Rectangle(bbox_parts)

    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filterBounds(bbox)
        .filterDate(args.start_date, args.end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", args.max_cloud_cover))
        .sort("system:time_start")
    )

    count = collection.size().getInfo()

    image_list = collection.toList(count)
    all_dates = []
    for i in range(count):
        img = ee.Image(image_list.get(i))
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        all_dates.append(date)
    unique_dates = sorted(set(all_dates))

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for d in unique_dates:
            f.write(d + "\n")

    print(f"Saved {len(unique_dates)} image dates to {args.output}")


if __name__ == "__main__":
    main()
