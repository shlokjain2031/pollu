#!/usr/bin/env python3
"""
Validate OpenAQ PM2.5 data file.

Checks structure, required columns, value ranges, fallback distribution, and data quality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def validate_openaq_pm25(
    file_path: Path,
    expected_dates_path: Path | None = None,
    min_coverage_pct: float = 50.0,
    quiet: bool = False,
) -> bool:
    """
    Validate OpenAQ PM2.5 parquet file.

    Parameters
    ----------
    file_path : Path
        Path to the parquet file to validate
    expected_dates_path : Path, optional
        Path to file with expected dates (one per line)
    min_coverage_pct : float
        Minimum percentage of non-null PM2.5 values required
    quiet : bool
        If True, suppress detailed output

    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    if not quiet:
        print(f"Validating OpenAQ PM2.5 file: {file_path}")
        print("=" * 60)

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ Failed to read parquet file: {e}")
        return False

    valid = True

    # Check required columns
    required_columns = [
        "date",
        "sensor_id",
        "sensor_name",
        "latitude",
        "longitude",
        "first_date",
        "last_date",
        "pm25",
        "sample_count",
        "actual_date",
        "fallback_type",
        "days_offset",
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        valid = False
    else:
        if not quiet:
            print(f"✓ All required columns present ({len(required_columns)} columns)")

    # Check basic structure
    n_rows = len(df)
    n_sensors = df["sensor_id"].nunique() if "sensor_id" in df.columns else 0
    n_dates = df["date"].nunique() if "date" in df.columns else 0

    if not quiet:
        print(f"\nDataset Structure:")
        print(f"  Total rows: {n_rows:,}")
        print(f"  Unique sensors: {n_sensors}")
        print(f"  Unique dates: {n_dates}")

    if n_rows == 0:
        print("❌ No data rows found")
        return False

    # Validate PM2.5 values
    if "pm25" in df.columns:
        non_null_pm25 = df["pm25"].notna().sum()
        null_pm25 = df["pm25"].isna().sum()
        coverage_pct = 100 * non_null_pm25 / n_rows

        if not quiet:
            print(f"\nPM2.5 Coverage:")
            print(f"  Non-null values: {non_null_pm25:,} ({coverage_pct:.1f}%)")
            print(f"  Null values: {null_pm25:,} ({100 - coverage_pct:.1f}%)")

        if coverage_pct < min_coverage_pct:
            print(
                f"⚠️  Warning: PM2.5 coverage ({coverage_pct:.1f}%) below minimum ({min_coverage_pct}%)"
            )
            valid = False

        # Check value ranges for non-null PM2.5
        valid_pm25 = df["pm25"].dropna()
        if len(valid_pm25) > 0:
            pm25_min = valid_pm25.min()
            pm25_max = valid_pm25.max()
            pm25_mean = valid_pm25.mean()
            pm25_median = valid_pm25.median()
            pm25_std = valid_pm25.std()

            if not quiet:
                print(f"\nPM2.5 Statistics (non-null values):")
                print(f"  Min: {pm25_min:.2f} µg/m³")
                print(f"  Max: {pm25_max:.2f} µg/m³")
                print(f"  Mean: {pm25_mean:.2f} µg/m³")
                print(f"  Median: {pm25_median:.2f} µg/m³")
                print(f"  Std Dev: {pm25_std:.2f} µg/m³")

            # Validate ranges (PM2.5 should be 0-1000 µg/m³, warn if >500)
            if pm25_min < 0:
                print(f"❌ Invalid: PM2.5 minimum is negative ({pm25_min:.2f})")
                valid = False
            if pm25_max > 1000:
                print(f"❌ Invalid: PM2.5 maximum exceeds 1000 µg/m³ ({pm25_max:.2f})")
                valid = False
            if pm25_max > 500:
                print(f"⚠️  Warning: PM2.5 maximum is very high ({pm25_max:.2f} µg/m³)")

            # Check for suspicious values
            extremely_low = (valid_pm25 < 1).sum()
            very_high = (valid_pm25 > 300).sum()
            if not quiet and (extremely_low > 0 or very_high > 0):
                print(f"\n  Flagged values:")
                if extremely_low > 0:
                    print(
                        f"    < 1 µg/m³: {extremely_low} ({100*extremely_low/len(valid_pm25):.1f}%)"
                    )
                if very_high > 0:
                    print(
                        f"    > 300 µg/m³: {very_high} ({100*very_high/len(valid_pm25):.1f}%)"
                    )

    # Validate fallback type distribution
    if "fallback_type" in df.columns:
        fallback_counts = df["fallback_type"].value_counts()
        if not quiet:
            print(f"\nFallback Type Distribution:")
            for fb_type, count in fallback_counts.items():
                pct = 100 * count / n_rows
                print(f"  {fb_type}: {count:,} ({pct:.1f}%)")

        # Check that fallback types are valid
        valid_types = {"exact", "window", "seasonal", "spatial_idw", "knn_fallback", "temporal_fallback", "none"}
        invalid_types = set(fallback_counts.index) - valid_types
        if invalid_types:
            print(f"❌ Invalid fallback types found: {invalid_types}")
            valid = False

    # Validate sample counts
    if "sample_count" in df.columns:
        sample_counts = df["sample_count"]
        if not quiet:
            print(f"\nSample Count Statistics:")
            print(f"  Min: {sample_counts.min()}")
            print(f"  Max: {sample_counts.max()}")
            print(f"  Mean: {sample_counts.mean():.1f}")
            print(f"  Median: {sample_counts.median():.0f}")

        zero_samples = (sample_counts == 0).sum()
        if zero_samples > 0 and not quiet:
            print(f"  Zero samples: {zero_samples} ({100*zero_samples/n_rows:.1f}%)")

    # Validate coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        valid_coords = df["latitude"].notna() & df["longitude"].notna()
        invalid_coords = (~valid_coords).sum()

        if not quiet:
            print(f"\nCoordinate Coverage:")
            print(f"  Valid coordinates: {valid_coords.sum():,}")
            if invalid_coords > 0:
                print(f"  Missing coordinates: {invalid_coords}")

        # Check Mumbai bounds (rough bbox: 18.9-19.3°N, 72.8-73.0°E)
        valid_lats = df.loc[valid_coords, "latitude"]
        valid_lons = df.loc[valid_coords, "longitude"]

        if len(valid_lats) > 0:
            lat_range = (valid_lats.min(), valid_lats.max())
            lon_range = (valid_lons.min(), valid_lons.max())

            if not quiet:
                print(f"  Latitude range: {lat_range[0]:.4f}° to {lat_range[1]:.4f}°")
                print(f"  Longitude range: {lon_range[0]:.4f}° to {lon_range[1]:.4f}°")

            # Sanity check for Mumbai region
            if not (18.0 <= lat_range[0] <= 20.0 and 18.0 <= lat_range[1] <= 20.0):
                print(f"⚠️  Warning: Latitude range outside Mumbai bounds")
            if not (72.0 <= lon_range[0] <= 74.0 and 72.0 <= lon_range[1] <= 74.0):
                print(f"⚠️  Warning: Longitude range outside Mumbai bounds")

    # Compare with expected dates if provided
    if expected_dates_path and expected_dates_path.exists():
        with open(expected_dates_path) as f:
            expected_dates = {line.strip() for line in f if line.strip()}

        # Convert df dates to string format for comparison
        actual_dates = (
            set(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
            if "date" in df.columns
            else set()
        )
        missing_dates = expected_dates - actual_dates
        extra_dates = actual_dates - expected_dates

        if not quiet:
            print(f"\nDate Coverage (vs {expected_dates_path.name}):")
            print(f"  Expected dates: {len(expected_dates)}")
            print(f"  Actual dates: {len(actual_dates)}")

        if missing_dates:
            print(f"  ❌ Missing dates: {len(missing_dates)}")
            if not quiet and len(missing_dates) <= 10:
                for date in sorted(missing_dates)[:10]:
                    print(f"    - {date}")
            valid = False
        else:
            if not quiet:
                print(f"  ✓ All expected dates present")

        if extra_dates and not quiet:
            print(f"  Extra dates: {len(extra_dates)}")

    # Validate days_offset consistency
    if "days_offset" in df.columns and "fallback_type" in df.columns:
        exact_with_offset = df[
            (df["fallback_type"] == "exact") & (df["days_offset"] != 0)
        ]
        if len(exact_with_offset) > 0:
            print(
                f"⚠️  Warning: {len(exact_with_offset)} exact matches have non-zero days_offset"
            )

    # Final validation result
    if not quiet:
        print("\n" + "=" * 60)

    if valid:
        print("✓ Validation PASSED")
    else:
        print("❌ Validation FAILED")

    return valid


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Validate OpenAQ PM2.5 parquet file")
    parser.add_argument(
        "file",
        type=Path,
        help="Path to OpenAQ PM2.5 parquet file",
    )
    parser.add_argument(
        "--expected-dates",
        type=Path,
        help="Path to file with expected dates (one per line)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=50.0,
        help="Minimum PM2.5 coverage percentage required (default: 50)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    valid = validate_openaq_pm25(
        args.file,
        args.expected_dates,
        args.min_coverage,
        args.quiet,
    )

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
