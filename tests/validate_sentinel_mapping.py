#!/usr/bin/env python3
"""Validate Sentinel-2 to Landsat date mapping.

Checks which Landsat dates have corresponding Sentinel-2 monthly mappings
and flags any missing months.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_LANDSAT_DATES = Path("patliputra/landsat8_image_dates.txt")
DEFAULT_SENTINEL_DATES = Path("patliputra/sentinel2_low_cloud_dates.txt")


def read_dates(path: Path) -> List[dt.date]:
    """Read dates from text file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return [dt.date.fromisoformat(line.strip()) for line in f if line.strip()]


def build_monthly_mapping(
    sentinel_dates: List[dt.date],
) -> Dict[Tuple[int, int], dt.date]:
    """Build mapping from (year, month) to Sentinel-2 date."""
    mapping = {}
    for date in sentinel_dates:
        key = (date.year, date.month)
        if key in mapping:
            # Keep the one with lower cloud cover (assumed to be sorted)
            continue
        mapping[key] = date
    return mapping


def validate_mapping(
    landsat_dates: List[dt.date], sentinel_mapping: Dict[Tuple[int, int], dt.date]
) -> Tuple[List[dt.date], List[Tuple[int, int]]]:
    """
    Check which Landsat dates have Sentinel-2 mappings.

    Returns
    -------
    tuple
        (mapped_dates, missing_months)
    """
    mapped = []
    missing_months = set()

    for landsat_date in landsat_dates:
        key = (landsat_date.year, landsat_date.month)
        if key in sentinel_mapping:
            mapped.append(landsat_date)
        else:
            missing_months.add(key)

    return mapped, sorted(missing_months)


def print_report(
    landsat_dates: List[dt.date],
    sentinel_dates: List[dt.date],
    mapped: List[dt.date],
    missing_months: List[Tuple[int, int]],
) -> None:
    """Print validation report."""
    print(f"\n{'='*60}")
    print("SENTINEL-2 TO LANDSAT MAPPING VALIDATION")
    print(f"{'='*60}\n")

    print(f"Total Landsat dates: {len(landsat_dates)}")
    print(f"Total Sentinel-2 dates: {len(sentinel_dates)}")
    print(
        f"Mapped Landsat dates: {len(mapped)} ({100*len(mapped)/len(landsat_dates):.1f}%)"
    )
    print(f"Missing months: {len(missing_months)}\n")

    if missing_months:
        print(f"{'='*60}")
        print("MISSING MONTHS (no Sentinel-2 data)")
        print(f"{'='*60}\n")

        # Group by year
        by_year = {}
        for year, month in missing_months:
            by_year.setdefault(year, []).append(month)

        for year in sorted(by_year.keys()):
            months = sorted(by_year[year])
            month_names = [dt.date(year, m, 1).strftime("%b") for m in months]
            print(f"{year}: {', '.join(month_names)} ({len(months)} months)")

        print(f"\n{'='*60}")
        print("AFFECTED LANDSAT DATES")
        print(f"{'='*60}\n")

        # Show which Landsat dates are affected
        for year, month in missing_months:
            affected = [
                d.isoformat()
                for d in landsat_dates
                if d.year == year and d.month == month
            ]
            month_name = dt.date(year, month, 1).strftime("%Y-%m")
            print(f"{month_name}: {', '.join(affected)}")

    print(f"\n{'='*60}")
    print("YEAR-BY-YEAR SUMMARY")
    print(f"{'='*60}\n")

    # Year-by-year breakdown
    year_stats = {}
    for date in landsat_dates:
        year = date.year
        year_stats.setdefault(year, {"total": 0, "mapped": 0})
        year_stats[year]["total"] += 1
        if (date.year, date.month) in {(y, m) for y, m in missing_months}:
            pass
        else:
            year_stats[year]["mapped"] += 1

    print(f"{'Year':<8} {'Landsat':<10} {'Mapped':<10} {'Coverage':<10}")
    print("-" * 40)
    for year in sorted(year_stats.keys()):
        stats = year_stats[year]
        total = stats["total"]
        mapped = stats["mapped"]
        pct = 100 * mapped / total if total > 0 else 0
        print(f"{year:<8} {total:<10} {mapped:<10} {pct:>6.1f}%")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Sentinel-2 to Landsat date mapping"
    )
    parser.add_argument(
        "--landsat-dates",
        type=Path,
        default=DEFAULT_LANDSAT_DATES,
        help="Path to landsat8_image_dates.txt",
    )
    parser.add_argument(
        "--sentinel-dates",
        type=Path,
        default=DEFAULT_SENTINEL_DATES,
        help="Path to sentinel2_low_cloud_dates.txt",
    )
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="List missing months in machine-readable format",
    )
    args = parser.parse_args()

    # Read dates
    landsat_dates = read_dates(args.landsat_dates)
    sentinel_dates = read_dates(args.sentinel_dates)

    # Build mapping
    sentinel_mapping = build_monthly_mapping(sentinel_dates)

    # Validate
    mapped, missing_months = validate_mapping(landsat_dates, sentinel_mapping)

    if args.list_missing:
        # Machine-readable output
        for year, month in missing_months:
            print(f"{year}-{month:02d}")
    else:
        # Human-readable report
        print_report(landsat_dates, sentinel_dates, mapped, missing_months)

        if missing_months:
            print("⚠️  WARNING: Some Landsat dates have no Sentinel-2 monthly mapping!")
            print(
                "   These dates will use interpolation or fall back to nearest neighbor.\n"
            )
        else:
            print("✅ All Landsat dates have Sentinel-2 monthly mappings!\n")


if __name__ == "__main__":
    main()
