#!/usr/bin/env python3
"""
Validate Open-Meteo meteorology parquet file.

Checks:
- File exists and is readable
- Expected columns present
- Date coverage matches expected Landsat dates
- Value ranges are physically reasonable
- No excessive missing values
- Statistics (mean, std, percentiles)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def validate_meteo_file(
    meteo_path: Path,
    expected_dates_file: Path | None = None,
    min_dates: int = 1,
) -> dict:
    """
    Validate Open-Meteo meteorology parquet file.

    Parameters
    ----------
    meteo_path : Path
        Path to meteorology parquet file
    expected_dates_file : Path | None
        Optional path to file with expected dates (one per line)
    min_dates : int
        Minimum number of dates required (default: 1)

    Returns
    -------
    dict
        Validation results with status, errors, warnings, and statistics
    """
    results = {
        "file": str(meteo_path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check file exists
    if not meteo_path.exists():
        results["valid"] = False
        results["errors"].append(f"File does not exist: {meteo_path}")
        return results

    try:
        # Read parquet
        df = pd.read_parquet(meteo_path)

        results["stats"]["n_rows"] = len(df)
        results["stats"]["n_columns"] = len(df.columns)

        # Check minimum dates
        if len(df) < min_dates:
            results["errors"].append(f"Insufficient dates: {len(df)} < {min_dates}")
            results["valid"] = False

        # Expected columns (at overpass time)
        required_columns = [
            "date",
            "datetime_utc",
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover",
        ]

        # Optional daily aggregate columns
        optional_columns = [
            "temp_mean",
            "temp_max",
            "temp_min",
            "rh_mean",
            "wind_speed_mean",
            "wind_speed_max",
            "precip_total",
            "cloud_cover_mean",
            "pressure_mean",
        ]

        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results["errors"].append(
                f"Missing required columns: {', '.join(missing_cols)}"
            )
            results["valid"] = False

        # Check optional columns
        present_optional = [col for col in optional_columns if col in df.columns]
        results["stats"]["has_daily_aggregates"] = len(present_optional) > 0

        # Validate date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            results["stats"]["date_range"] = {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            }

            # Check for duplicates
            duplicates = df["date"].duplicated().sum()
            if duplicates > 0:
                results["errors"].append(f"Duplicate dates found: {duplicates}")
                results["valid"] = False

            # Check date ordering
            if not df["date"].is_monotonic_increasing:
                results["warnings"].append("Dates are not sorted")

            # Compare with expected dates file
            if expected_dates_file and expected_dates_file.exists():
                with open(expected_dates_file, "r") as f:
                    expected_dates = [
                        pd.to_datetime(line.strip()) for line in f if line.strip()
                    ]

                expected_set = set(expected_dates)
                actual_set = set(df["date"])

                missing_dates = expected_set - actual_set
                extra_dates = actual_set - expected_set

                if missing_dates:
                    results["warnings"].append(
                        f"{len(missing_dates)} dates missing from expected: "
                        f"{sorted([d.strftime('%Y-%m-%d') for d in list(missing_dates)[:5]])}"
                    )

                if extra_dates:
                    results["warnings"].append(
                        f"{len(extra_dates)} extra dates not in expected file"
                    )

                results["stats"]["expected_dates"] = len(expected_dates)
                results["stats"]["matched_dates"] = len(expected_set & actual_set)

        # Validate meteorology values
        if "temperature_2m" in df.columns:
            temp = df["temperature_2m"]
            results["stats"]["temperature"] = {
                "min": float(temp.min()),
                "max": float(temp.max()),
                "mean": float(temp.mean()),
                "std": float(temp.std()),
            }

            # Mumbai: typical range 15-40°C
            if temp.min() < 0:
                results["errors"].append(
                    f"Temperature below freezing: {temp.min():.1f}°C"
                )
                results["valid"] = False
            elif temp.min() < 10:
                results["warnings"].append(
                    f"Unusually low temperature: {temp.min():.1f}°C"
                )

            if temp.max() > 50:
                results["errors"].append(
                    f"Unrealistic high temperature: {temp.max():.1f}°C"
                )
                results["valid"] = False
            elif temp.max() > 45:
                results["warnings"].append(f"Very high temperature: {temp.max():.1f}°C")

        if "relative_humidity_2m" in df.columns:
            rh = df["relative_humidity_2m"]
            results["stats"]["humidity"] = {
                "min": float(rh.min()),
                "max": float(rh.max()),
                "mean": float(rh.mean()),
            }

            # Valid range: 0-100%
            if rh.min() < 0 or rh.max() > 100:
                results["errors"].append(
                    f"Humidity out of valid range [0,100]: {rh.min():.1f}-{rh.max():.1f}%"
                )
                results["valid"] = False

        if "wind_speed_10m" in df.columns:
            wind = df["wind_speed_10m"]
            results["stats"]["wind_speed"] = {
                "min": float(wind.min()),
                "max": float(wind.max()),
                "mean": float(wind.mean()),
            }

            # Typical range: 0-20 m/s for Mumbai
            if wind.min() < 0:
                results["errors"].append(f"Negative wind speed: {wind.min():.1f} m/s")
                results["valid"] = False

            if wind.max() > 30:
                results["warnings"].append(
                    f"Very high wind speed: {wind.max():.1f} m/s (>100 km/h)"
                )

        if "wind_direction_10m" in df.columns:
            wind_dir = df["wind_direction_10m"]

            # Valid range: 0-360 degrees
            if wind_dir.min() < 0 or wind_dir.max() > 360:
                results["errors"].append(
                    f"Wind direction out of valid range [0,360]: "
                    f"{wind_dir.min():.1f}-{wind_dir.max():.1f}°"
                )
                results["valid"] = False

        if "surface_pressure" in df.columns:
            pressure = df["surface_pressure"]
            results["stats"]["pressure"] = {
                "min": float(pressure.min()),
                "max": float(pressure.max()),
                "mean": float(pressure.mean()),
            }

            # Typical range: 990-1030 hPa
            if pressure.min() < 950 or pressure.max() > 1050:
                results["warnings"].append(
                    f"Unusual pressure range: {pressure.min():.1f}-{pressure.max():.1f} hPa"
                )

        if "precipitation" in df.columns:
            precip = df["precipitation"]
            results["stats"]["precipitation"] = {
                "min": float(precip.min()),
                "max": float(precip.max()),
                "mean": float(precip.mean()),
                "days_with_rain": int((precip > 0).sum()),
            }

            if precip.min() < 0:
                results["errors"].append(
                    f"Negative precipitation: {precip.min():.1f} mm"
                )
                results["valid"] = False

        if "cloud_cover" in df.columns:
            clouds = df["cloud_cover"]

            # Valid range: 0-100%
            if clouds.min() < 0 or clouds.max() > 100:
                results["errors"].append(
                    f"Cloud cover out of valid range [0,100]: "
                    f"{clouds.min():.1f}-{clouds.max():.1f}%"
                )
                results["valid"] = False

        # Check for missing values
        missing_summary = {}
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                missing_summary[col] = int(n_missing)

        if missing_summary:
            results["warnings"].append(
                f"Columns with missing values: {missing_summary}"
            )
            results["stats"]["missing_values"] = missing_summary

    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Error reading file: {str(e)}")

    return results


def print_validation_results(results: dict) -> None:
    """
    Pretty print validation results.

    Parameters
    ----------
    results : dict
        Validation results from validate_meteo_file
    """
    print("=" * 80)
    print(f"Meteorology Validation Report: {Path(results['file']).name}")
    print("=" * 80)

    # Status
    if results["valid"]:
        print("✓ Status: VALID")
    else:
        print("✗ Status: INVALID")

    print()

    # Errors
    if results["errors"]:
        print(f"Errors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  ✗ {err}")
        print()

    # Warnings
    if results["warnings"]:
        print(f"Warnings ({len(results['warnings'])}):")
        for warn in results["warnings"]:
            print(f"  ⚠ {warn}")
        print()

    # Statistics
    if results["stats"]:
        print("Statistics:")
        stats = results["stats"]

        print(f"  Rows: {stats.get('n_rows', 'N/A')}")
        print(f"  Columns: {stats.get('n_columns', 'N/A')}")

        if "date_range" in stats:
            dr = stats["date_range"]
            print(f"  Date Range: {dr['start']} to {dr['end']}")

        if "expected_dates" in stats:
            print(
                f"  Date Match: {stats['matched_dates']}/{stats['expected_dates']} "
                f"({100*stats['matched_dates']/stats['expected_dates']:.1f}%)"
            )

        print(
            f"  Daily Aggregates: {'Yes' if stats.get('has_daily_aggregates') else 'No'}"
        )

        if "temperature" in stats:
            print("\n  Temperature (°C):")
            t = stats["temperature"]
            print(f"    Min:  {t['min']:.1f}")
            print(f"    Max:  {t['max']:.1f}")
            print(f"    Mean: {t['mean']:.1f}")
            print(f"    Std:  {t['std']:.1f}")

        if "humidity" in stats:
            print("\n  Relative Humidity (%):")
            h = stats["humidity"]
            print(f"    Min:  {h['min']:.1f}")
            print(f"    Max:  {h['max']:.1f}")
            print(f"    Mean: {h['mean']:.1f}")

        if "wind_speed" in stats:
            print("\n  Wind Speed (m/s):")
            w = stats["wind_speed"]
            print(f"    Min:  {w['min']:.1f}")
            print(f"    Max:  {w['max']:.1f}")
            print(f"    Mean: {w['mean']:.1f}")

        if "pressure" in stats:
            print("\n  Surface Pressure (hPa):")
            p = stats["pressure"]
            print(f"    Min:  {p['min']:.1f}")
            print(f"    Max:  {p['max']:.1f}")
            print(f"    Mean: {p['mean']:.1f}")

        if "precipitation" in stats:
            print("\n  Precipitation:")
            pr = stats["precipitation"]
            print(f"    Max:  {pr['max']:.1f} mm")
            print(f"    Mean: {pr['mean']:.2f} mm")
            print(f"    Days with rain: {pr['days_with_rain']}")

    print("=" * 80)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Open-Meteo meteorology parquet file"
    )
    parser.add_argument(
        "meteo_file",
        type=Path,
        help="Path to meteorology parquet file",
    )
    parser.add_argument(
        "--expected-dates",
        type=Path,
        help="Path to file with expected dates (one per line)",
    )
    parser.add_argument(
        "--min-dates",
        type=int,
        default=1,
        help="Minimum number of dates required (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and warnings",
    )

    args = parser.parse_args()

    # Validate
    results = validate_meteo_file(
        args.meteo_file,
        args.expected_dates,
        args.min_dates,
    )

    # Print results
    if not args.quiet:
        print_validation_results(results)
    else:
        if results["errors"]:
            for err in results["errors"]:
                print(f"ERROR: {err}", file=sys.stderr)
        if results["warnings"]:
            for warn in results["warnings"]:
                print(f"WARNING: {warn}", file=sys.stderr)

    # Exit code
    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()
