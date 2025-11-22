#!/usr/bin/env python3
"""Find Sentinel-2 dates for specific missing months."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List, Tuple

from patliputra.backends.earthengine import EarthEngineClient

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_BBOX = (72.7763, 18.8939, 72.9797, 19.2701)
DEFAULT_OUTPUT = Path("patliputra/sentinel2_low_cloud_dates.txt")

# Missing months from validation
MISSING_MONTHS = [
    (2018, 4),  # April 2018
    (2018, 5),  # May 2018
    (2018, 9),  # September 2018
    (2022, 6),  # June 2022
]


def find_best_date_for_month(
    client: EarthEngineClient,
    year: int,
    month: int,
    bbox: Tuple[float, float, float, float],
    max_cloud: float = 50.0,  # More lenient for missing months
) -> Tuple[int, int, dt.date | None, float | None]:
    """Find best Sentinel-2 date for a specific month."""
    ee = __import__("ee")

    start = dt.date(year, month, 1)
    if month == 12:
        end = dt.date(year + 1, 1, 1)
    else:
        end = dt.date(year, month + 1, 1)

    geom = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])

    collection = (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = int(collection.size().getInfo())
    if count == 0:
        print(f"  ✗ {year}-{month:02d}: No imagery found (cloud < {max_cloud}%)")
        return year, month, None, None

    image = collection.first()
    info = image.getInfo()
    props = info.get("properties", {}) if info else {}

    timestamp = props.get("system:time_start")
    cloud_pct = props.get("CLOUDY_PIXEL_PERCENTAGE")

    if timestamp is None:
        print(f"  ✗ {year}-{month:02d}: Missing timestamp")
        return year, month, None, None

    chosen_date = dt.datetime.utcfromtimestamp(timestamp / 1000).date()
    print(
        f"  ✓ {year}-{month:02d}: {chosen_date} (cloud: {cloud_pct:.1f}%, {count} images available)"
    )

    return year, month, chosen_date, cloud_pct


def append_to_file(dates: List[dt.date], output_path: Path) -> None:
    """Append new dates to existing file and re-sort."""
    # Read existing dates
    existing = []
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = [
                dt.date.fromisoformat(line.strip()) for line in f if line.strip()
            ]

    # Combine and sort
    all_dates = sorted(set(existing + dates))

    # Write back
    with open(output_path, "w") as f:
        for date in all_dates:
            f.write(date.isoformat() + "\n")

    print(f"\n✓ Updated {output_path} with {len(dates)} new dates")
    print(f"  Total: {len(all_dates)} dates")


def main():
    parser = argparse.ArgumentParser(
        description="Find Sentinel-2 dates for missing months"
    )
    parser.add_argument("--ee-project", required=True, help="Earth Engine project ID")
    parser.add_argument(
        "--bbox",
        default=",".join(str(v) for v in DEFAULT_BBOX),
        help="Bounding box: west,south,east,north",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=50.0,
        help="Maximum cloud percentage (default: 50.0)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output file to append to"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't write to file, just show results"
    )
    args = parser.parse_args()

    bbox = tuple(float(x.strip()) for x in args.bbox.split(","))

    # Initialize Earth Engine
    client = EarthEngineClient(project_id=args.ee_project)
    client.initialize()
    print("Earth Engine initialized\n")

    print("Searching for Sentinel-2 imagery in missing months:")
    print("=" * 60)

    found_dates = []
    for year, month in MISSING_MONTHS:
        _, _, date, cloud = find_best_date_for_month(
            client, year, month, bbox, args.max_cloud
        )
        if date:
            found_dates.append(date)

    print("\n" + "=" * 60)
    print(f"Found {len(found_dates)} / {len(MISSING_MONTHS)} missing months\n")

    if found_dates and not args.dry_run:
        append_to_file(found_dates, args.output)
    elif args.dry_run:
        print("DRY RUN - would add these dates:")
        for date in sorted(found_dates):
            print(f"  {date}")
    else:
        print("⚠️  No new dates found")

    # Show what's still missing
    still_missing = []
    for year, month in MISSING_MONTHS:
        if not any(d.year == year and d.month == month for d in found_dates):
            still_missing.append((year, month))

    if still_missing:
        print(f"\n⚠️  Still missing after search:")
        for year, month in still_missing:
            print(f"  {year}-{month:02d}")
        print(f"\nTip: Increase --max-cloud (currently {args.max_cloud}%) or")
        print("     these months may have no Sentinel-2 coverage")


if __name__ == "__main__":
    main()
