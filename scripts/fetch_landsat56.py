#!/usr/bin/env python3
"""
Fetch all cloud-free Landsat-8 B2 image dates for the Mumbai bounding box
and save them to a file under:
    /Users/shlokjain/pollu/patliputra/signals/toa_b2/image_dates.txt
"""

import ee
from pathlib import Path


def main():
    print("Landsat8 Initialized")
    ee.Initialize(project="fast-archive-465917-m0")

    bbox = ee.Geometry.Rectangle([72.7763, 18.8939, 72.9797, 19.2701])

    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filterBounds(bbox)
        .filterDate("2018-01-01", "2025-11-01")
        .filter(ee.Filter.lt("CLOUD_COVER", 15))
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

    output_dir = Path("/Users/shlokjain/pollu/patliputra")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "landsat8_image_dates.txt"

    with open(output_file, "w") as f:
        for d in unique_dates:
            f.write(d + "\n")

    print(f"Saved {len(unique_dates)} image dates to {output_file}")


if __name__ == "__main__":
    main()
