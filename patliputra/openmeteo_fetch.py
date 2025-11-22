#!/usr/bin/env python3
"""
Fetch meteorological data from Open-Meteo Historical Weather API.

Uses Open-Meteo's ECMWF IFS reanalysis data at 9km resolution (free, no API key required).
Fetches hourly meteorology for Mumbai bbox and joins to Landsat dates.

Resolution:
    - ECMWF IFS: 9km (default, 2017-present, highest resolution)
    - ERA5-Land: 11km (1950-present)
    - ERA5: 25km (1940-present)

Variables:
    - temperature_2m: Air temperature at 2m (°C)
    - relative_humidity_2m: Relative humidity at 2m (%)
    - dewpoint_2m: Dewpoint temperature at 2m (°C)
    - surface_pressure: Surface pressure (hPa)
    - wind_speed_10m: Wind speed at 10m (m/s)
    - wind_direction_10m: Wind direction at 10m (°)
    - precipitation: Total precipitation (mm)
    - cloud_cover: Cloud cover (%)

References:
    - https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


def fetch_openmeteo_hourly(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    variables: List[str] | None = None,
    models: str = "ecmwf_ifs",
) -> pd.DataFrame:
    """
    Fetch hourly meteorology from Open-Meteo for a single location.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    latitude : float
        Latitude in degrees
    longitude : float
        Longitude in degrees
    variables : List[str] | None
        List of variable names to fetch. If None, fetches default set.
    models : str
        Reanalysis model to use. Options:
        - 'ecmwf_ifs': 9km resolution, 2017-present (recommended)
        - 'era5_land': 11km resolution, 1950-present
        - 'era5': 25km resolution, 1940-present
        Default: 'ecmwf_ifs' for highest resolution

    Returns
    -------
    pd.DataFrame
        Hourly meteorology with datetime index and variable columns
    """
    if variables is None:
        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover",
        ]

    # Open-Meteo Historical Weather API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "models": models,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    # Parse hourly data
    hourly = data["hourly"]
    times = pd.to_datetime(hourly["time"])

    # Build dataframe
    df = pd.DataFrame({"datetime": times})
    for var in variables:
        if var in hourly:
            df[var] = hourly[var]

    df.set_index("datetime", inplace=True)

    return df


def fetch_openmeteo_for_bbox(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    variables: List[str] | None = None,
    models: str = "ecmwf_ifs",
) -> pd.DataFrame:
    """
    Fetch meteorology for Mumbai bbox (fetches center point).

    Open-Meteo free tier provides point data, not spatial grids.
    For air quality modeling, using city center is reasonable.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    bbox : tuple
        (west, south, east, north) in degrees
    variables : List[str] | None
        Variables to fetch
    models : str
        Reanalysis model ('ecmwf_ifs', 'era5_land', or 'era5')

    Returns
    -------
    pd.DataFrame
        Hourly meteorology for bbox center
    """
    west, south, east, north = bbox

    # Use bbox center
    lat = (south + north) / 2
    lon = (west + east) / 2

    print(f"Fetching Open-Meteo data for center point: ({lat:.4f}, {lon:.4f})")
    print(f"Period: {start_date} to {end_date}")
    print(
        f"Model: {models} (9km resolution)"
        if models == "ecmwf_ifs"
        else f"Model: {models}"
    )

    df = fetch_openmeteo_hourly(start_date, end_date, lat, lon, variables, models)

    return df


def match_meteo_to_landsat_overpass(
    meteo_df: pd.DataFrame,
    landsat_dates: List[str],
    overpass_hour_utc: int = 5,
) -> pd.DataFrame:
    """
    Match meteorology to Landsat overpass times.

    Landsat 8 overpasses Mumbai around 05:00-05:30 UTC.

    Parameters
    ----------
    meteo_df : pd.DataFrame
        Hourly meteorology with datetime index
    landsat_dates : List[str]
        List of Landsat dates in YYYY-MM-DD format
    overpass_hour_utc : int
        Approximate overpass hour in UTC (default 5 for Mumbai)

    Returns
    -------
    pd.DataFrame
        Meteorology matched to Landsat dates, indexed by date
    """
    records = []

    for date_str in landsat_dates:
        # Create datetime for overpass hour
        overpass_dt = pd.Timestamp(f"{date_str} {overpass_hour_utc:02d}:00:00")

        # Find closest hour in meteo data
        if overpass_dt in meteo_df.index:
            row = meteo_df.loc[overpass_dt].to_dict()
            row["date"] = date_str
            row["datetime_utc"] = overpass_dt
            records.append(row)
        else:
            # Find nearest hour (within ±3 hours)
            time_diffs = abs(meteo_df.index - overpass_dt)
            min_diff_idx = time_diffs.argmin()
            min_diff = time_diffs[min_diff_idx]

            if min_diff <= pd.Timedelta(hours=3):
                row = meteo_df.iloc[min_diff_idx].to_dict()
                row["date"] = date_str
                row["datetime_utc"] = meteo_df.index[min_diff_idx]
                row["hours_offset"] = min_diff.total_seconds() / 3600
                records.append(row)
            else:
                print(f"Warning: No meteo data within 3 hours of {date_str} overpass")

    if not records:
        raise ValueError("No meteorology matched to any Landsat dates")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    return df


def compute_daily_aggregates(
    meteo_df: pd.DataFrame,
    landsat_dates: List[str],
) -> pd.DataFrame:
    """
    Compute daily aggregate meteorology (24-hour means/max/min).

    Useful for daily exposure modeling.

    Parameters
    ----------
    meteo_df : pd.DataFrame
        Hourly meteorology with datetime index
    landsat_dates : List[str]
        List of Landsat dates in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame
        Daily aggregate meteorology
    """
    records = []

    for date_str in landsat_dates:
        date = pd.Timestamp(date_str)

        # Get all hours for this date
        day_mask = meteo_df.index.date == date.date()
        day_data = meteo_df[day_mask]

        if len(day_data) == 0:
            print(f"Warning: No meteo data for {date_str}")
            continue

        record = {"date": date_str}

        # Temperature stats
        if "temperature_2m" in day_data.columns:
            record["temp_mean"] = day_data["temperature_2m"].mean()
            record["temp_max"] = day_data["temperature_2m"].max()
            record["temp_min"] = day_data["temperature_2m"].min()

        # Humidity stats
        if "relative_humidity_2m" in day_data.columns:
            record["rh_mean"] = day_data["relative_humidity_2m"].mean()

        # Wind stats
        if "wind_speed_10m" in day_data.columns:
            record["wind_speed_mean"] = day_data["wind_speed_10m"].mean()
            record["wind_speed_max"] = day_data["wind_speed_10m"].max()

        # Total precipitation
        if "precipitation" in day_data.columns:
            record["precip_total"] = day_data["precipitation"].sum()

        # Mean cloud cover
        if "cloud_cover" in day_data.columns:
            record["cloud_cover_mean"] = day_data["cloud_cover"].mean()

        # Mean pressure
        if "surface_pressure" in day_data.columns:
            record["pressure_mean"] = day_data["surface_pressure"].mean()

        records.append(record)

    if not records:
        raise ValueError("No daily aggregates computed")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    return df


def fetch_and_cache_openmeteo(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    cache_dir: Path | None = None,
    models: str = "ecmwf_ifs",
) -> pd.DataFrame:
    """
    Fetch Open-Meteo data with local caching.

    Parameters
    ----------
    start_date : str
        Start date YYYY-MM-DD
    end_date : str
        End date YYYY-MM-DD
    bbox : tuple
        (west, south, east, north)
    cache_dir : Path | None
        Cache directory (default: cache/openmeteo)
    models : str
        Reanalysis model (default: 'ecmwf_ifs' for 9km resolution)

    Returns
    -------
    pd.DataFrame
        Hourly meteorology
    """
    if cache_dir is None:
        cache_dir = Path("cache/openmeteo")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache file name based on dates, location, and model
    lat = (bbox[1] + bbox[3]) / 2
    lon = (bbox[0] + bbox[2]) / 2
    cache_file = (
        cache_dir / f"openmeteo_{models}_{start_date}_{end_date}_"
        f"{lat:.2f}_{lon:.2f}.parquet"
    )

    if cache_file.exists():
        print(f"Loading cached Open-Meteo data from {cache_file}")
        df = pd.read_parquet(cache_file)
        df.set_index("datetime", inplace=True)
        return df

    # Fetch fresh data
    print(f"Fetching Open-Meteo data from {start_date} to {end_date}...")
    df = fetch_openmeteo_for_bbox(start_date, end_date, bbox, models=models)

    # Cache
    df_to_save = df.reset_index()
    df_to_save.to_parquet(cache_file, index=False)
    print(f"✓ Cached to {cache_file}")

    return df


def integrate_meteo_with_landsat(
    landsat_dates: List[str],
    bbox: Tuple[float, float, float, float] = (72.7763, 18.8939, 72.9797, 19.2701),
    overpass_hour: int = 5,
    include_daily_aggs: bool = True,
    cache_dir: Path | None = None,
    models: str = "ecmwf_ifs",
) -> pd.DataFrame:
    """
    Fetch Open-Meteo data and match to Landsat dates.

    Parameters
    ----------
    landsat_dates : List[str]
        List of Landsat acquisition dates (YYYY-MM-DD)
    bbox : tuple
        Mumbai bounding box
    overpass_hour : int
        Landsat overpass hour in UTC (default 5)
    include_daily_aggs : bool
        Whether to include 24-hour aggregates
    cache_dir : Path | None
        Cache directory
    models : str
        Reanalysis model (default: 'ecmwf_ifs' for 9km resolution)

    Returns
    -------
    pd.DataFrame
        Meteorology matched to Landsat dates
    """
    if not landsat_dates:
        raise ValueError("No Landsat dates provided")

    # Determine date range
    dates = pd.to_datetime(landsat_dates)
    start_date = (dates.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch hourly data
    hourly_df = fetch_and_cache_openmeteo(start_date, end_date, bbox, cache_dir, models)

    # Match to overpass times
    print(f"\nMatching meteorology to {len(landsat_dates)} Landsat dates...")
    overpass_df = match_meteo_to_landsat_overpass(
        hourly_df, landsat_dates, overpass_hour
    )

    # Optionally add daily aggregates
    if include_daily_aggs:
        print("Computing daily aggregates...")
        daily_df = compute_daily_aggregates(hourly_df, landsat_dates)

        # Merge
        overpass_df = overpass_df.merge(daily_df, on="date", how="left")

    print(f"✓ Matched {len(overpass_df)} dates with meteorology")

    return overpass_df


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch meteorology from Open-Meteo and match to Landsat dates"
    )
    parser.add_argument(
        "--dates",
        type=str,
        help="Comma-separated Landsat dates (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dates-file",
        type=Path,
        help="Path to file with Landsat dates (one per line, YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="72.7763,18.8939,72.9797,19.2701",
        help="Bounding box: west,south,east,north",
    )
    parser.add_argument(
        "--overpass-hour",
        type=int,
        default=5,
        help="Landsat overpass hour UTC (default 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cache/landsat_meteo.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/openmeteo"),
        help="Cache directory",
    )
    parser.add_argument(
        "--no-daily-aggs",
        action="store_true",
        help="Skip daily aggregate computation",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ecmwf_ifs",
        choices=["ecmwf_ifs", "era5_land", "era5"],
        help="Reanalysis model: ecmwf_ifs (9km, default), era5_land (11km), era5 (25km)",
    )

    args = parser.parse_args()

    # Parse inputs
    if args.dates_file:
        # Read dates from file
        with open(args.dates_file, "r") as f:
            landsat_dates = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(landsat_dates)} dates from {args.dates_file}")
    elif args.dates:
        # Parse comma-separated dates
        landsat_dates = [d.strip() for d in args.dates.split(",")]
    else:
        parser.error("Either --dates or --dates-file must be provided")

    bbox = tuple(float(x.strip()) for x in args.bbox.split(","))

    # Fetch and integrate
    meteo_df = integrate_meteo_with_landsat(
        landsat_dates,
        bbox,
        args.overpass_hour,
        not args.no_daily_aggs,
        args.cache_dir,
        args.models,
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meteo_df.to_parquet(args.output, index=False)
    print(f"\n✓ Saved meteorology to {args.output}")

    # Print summary
    print("\nSummary:")
    print(f"  Dates: {len(meteo_df)}")
    print(f"  Variables: {list(meteo_df.columns)}")
    print("\nSample:")
    print(meteo_df.head())


if __name__ == "__main__":
    main()
