#!/usr/bin/env python3
"""Find low-cloud Sentinel-2 dates for each Landsat processing month."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from patliputra.backends.earthengine import EarthEngineClient

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_DATES_PATH = Path("patliputra/landsat8_image_dates.txt")
DEFAULT_OUTPUT_PATH = Path("patliputra/sentinel2_low_cloud_dates.txt")
DEFAULT_BBOX = (72.7763, 18.8939, 72.9797, 19.2701)
DEFAULT_MAX_CLOUD = 20.0

logger = logging.getLogger("sentinel2_low_cloud_dates")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_dates(path: Path) -> List[dt.date]:
    if not path.exists():
        raise FileNotFoundError(f"Dates file not found: {path}")
    with open(path, "r") as fh:
        return [dt.date.fromisoformat(line.strip()) for line in fh if line.strip()]


def unique_months(dates: Iterable[dt.date]) -> List[dt.date]:
    seen = {date.replace(day=1) for date in dates}
    return sorted(seen)


def next_month(date_val: dt.date) -> dt.date:
    if date_val.month == 12:
        return dt.date(date_val.year + 1, 1, 1)
    return dt.date(date_val.year, date_val.month + 1, 1)


def _bbox_geometry(bbox: Tuple[float, float, float, float]):
    ee = __import__("ee")
    return ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])


def find_low_cloud_date(
    client: EarthEngineClient,
    month_start: dt.date,
    bbox: Tuple[float, float, float, float],
    max_cloud: Optional[float] = None,
) -> Optional[dt.date]:
    ee = __import__("ee")
    start = ee.Date(month_start.isoformat())
    end = ee.Date(next_month(month_start).isoformat())
    geom = _bbox_geometry(bbox)

    collection = (
        ee.ImageCollection(COLLECTION_ID).filterDate(start, end).filterBounds(geom)
    )
    if max_cloud is not None:
        collection = collection.filter(
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)
        )
    collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE")

    image = collection.first()
    if image is None:
        logger.warning(
            "No Sentinel-2 image found for %s", month_start.strftime("%Y-%m")
        )
        return None

    info = image.getInfo()
    props = info.get("properties", {}) if info else {}
    timestamp = props.get("system:time_start")
    cloud_pct = props.get("CLOUDY_PIXEL_PERCENTAGE")
    if timestamp is None:
        logger.warning("Missing timestamp for %s", month_start.strftime("%Y-%m"))
        return None
    chosen_date = dt.datetime.utcfromtimestamp(timestamp / 1000).date()
    return chosen_date


def write_results(chosen_dates: Sequence[dt.date], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for date_val in chosen_dates:
            fh.write(date_val.isoformat() + "\n")
    logger.info("Wrote %d entries to %s", len(chosen_dates), output_path)


def process(
    dates_path: Path,
    output_path: Path,
    bbox: Tuple[float, float, float, float],
    max_cloud: Optional[float],
    ee_project: Optional[str],
) -> List[dt.date]:
    dates = read_dates(dates_path)
    months = unique_months(dates)
    client = (
        EarthEngineClient(project_id=ee_project) if ee_project else EarthEngineClient()
    )
    client.initialize()
    results: List[dt.date] = []
    for month_start in months:
        try:
            chosen_date = find_low_cloud_date(client, month_start, bbox, max_cloud)
            if chosen_date is not None:
                results.append(chosen_date)
            else:
                logger.warning(
                    "Skipping %s because no date passed selection",
                    month_start.strftime("%Y-%m"),
                )
        except Exception as exc:
            logger.error(
                "Failed to find date for %s: %s",
                month_start.strftime("%Y-%m"),
                exc,
                exc_info=True,
            )
    write_results(results, output_path)
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Sentinel-2 low-cloud dates for each month present in landsat8_image_dates.txt"
    )
    parser.add_argument(
        "--dates-path",
        type=Path,
        default=DEFAULT_DATES_PATH,
        help="Path to landsat8_image_dates.txt",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the resulting text file",
    )
    parser.add_argument(
        "--bbox",
        default=",".join(str(v) for v in DEFAULT_BBOX),
        help="minx,miny,maxx,maxy in lon/lat",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=DEFAULT_MAX_CLOUD,
        help="Maximum CLOUDY_PIXEL_PERCENTAGE to consider",
    )
    parser.add_argument(
        "--ee-project", default=None, help="Optional Earth Engine project id"
    )
    parser.add_argument(
        "--no-cloud-filter",
        action="store_true",
        help="Disable the max cloud percentage filter",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    bbox = tuple(float(x) for x in args.bbox.split(","))
    max_cloud = None if args.no_cloud_filter else args.max_cloud
    process(
        dates_path=args.dates_path,
        output_path=args.out_path,
        bbox=bbox,  # type: ignore[arg-type]
        max_cloud=max_cloud,
        ee_project=args.ee_project,
    )


if __name__ == "__main__":
    main()
