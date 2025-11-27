#!/usr/bin/env python3
"""
Spatial interpolation for missing PM2.5 values in OpenAQ data.

Fills gaps using Inverse Distance Weighting (IDW) from nearby sensors with valid data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points on Earth in kilometers.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point in degrees
    lat2, lon2 : float
        Latitude and longitude of second point in degrees

    Returns
    -------
    float
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def spatial_interpolate_idw(
    target_lat: float,
    target_lon: float,
    neighbor_data: pd.DataFrame,
    max_distance_km: float = 10.0,
    min_neighbors: int = 3,
    power: float = 2.0,
) -> Optional[float]:
    """
    Interpolate PM2.5 value using Inverse Distance Weighting.

    Parameters
    ----------
    target_lat, target_lon : float
        Coordinates of point to interpolate
    neighbor_data : pd.DataFrame
        DataFrame with columns: latitude, longitude, pm25 (valid values only)
    max_distance_km : float
        Maximum distance to search for neighbors
    min_neighbors : int
        Minimum number of neighbors required
    power : float
        Power parameter for IDW (default: 2.0 for inverse square)

    Returns
    -------
    float or None
        Interpolated PM2.5 value, or None if insufficient neighbors
    """
    if len(neighbor_data) == 0:
        return None

    # Calculate distances to all neighbors
    distances = neighbor_data.apply(
        lambda row: haversine_distance(
            target_lat, target_lon, row["latitude"], row["longitude"]
        ),
        axis=1,
    )

    # Filter by max distance
    valid_neighbors = neighbor_data[distances <= max_distance_km].copy()
    valid_distances = distances[distances <= max_distance_km]

    if len(valid_neighbors) < min_neighbors:
        return None

    # Handle case where target is at exact sensor location
    zero_dist = valid_distances < 0.001  # Less than 1 meter
    if zero_dist.any():
        return valid_neighbors.loc[zero_dist, "pm25"].iloc[0]

    # IDW calculation: weight = 1 / distance^power
    weights = 1.0 / (valid_distances**power)
    weighted_sum = (valid_neighbors["pm25"] * weights).sum()
    weight_sum = weights.sum()

    return weighted_sum / weight_sum


def interpolate_missing_pm25(
    input_path: Path,
    max_distance_km: float = 10.0,
    min_neighbors: int = 3,
    power: float = 2.0,
) -> None:
    """
    Interpolate missing PM2.5 values using spatial IDW interpolation.
    Modifies the input file in-place.

    Parameters
    ----------
    input_path : Path
        Input parquet with PM2.5 data (will be modified in-place)
    max_distance_km : float
        Maximum distance to search for neighbors (default: 10 km)
    min_neighbors : int
        Minimum neighbors required for interpolation (default: 3)
    power : float
        IDW power parameter (default: 2.0)
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Count initial missing values
    initial_missing = df["pm25"].isna().sum()
    total_rows = len(df)
    print(
        f"Initial missing PM2.5 values: {initial_missing}/{total_rows} ({100*initial_missing/total_rows:.1f}%)"
    )

    if initial_missing == 0:
        print("No missing values to interpolate!")
        return

    print(f"\nSpatial interpolation settings:")
    print(f"  Max distance: {max_distance_km} km")
    print(f"  Min neighbors: {min_neighbors}")
    print(f"  IDW power: {power}")

    # Process each date separately
    interpolated_count = 0
    df_list = []

    for date in sorted(df["date"].unique()):
        date_df = df[df["date"] == date].copy()

        # Get sensors with valid data on this date
        valid_data = date_df[date_df["pm25"].notna()][
            ["sensor_id", "latitude", "longitude", "pm25"]
        ].copy()

        # Get sensors with missing data on this date
        missing_mask = date_df["pm25"].isna()

        if missing_mask.sum() == 0:
            df_list.append(date_df)
            continue

        # Interpolate for each missing sensor
        for idx in date_df[missing_mask].index:
            target_lat = date_df.loc[idx, "latitude"]
            target_lon = date_df.loc[idx, "longitude"]

            # Skip if coordinates are missing
            if pd.isna(target_lat) or pd.isna(target_lon):
                continue

            # Interpolate using nearby sensors (excluding self)
            neighbor_data = valid_data[
                valid_data["sensor_id"] != date_df.loc[idx, "sensor_id"]
            ]

            interpolated_value = spatial_interpolate_idw(
                target_lat,
                target_lon,
                neighbor_data,
                max_distance_km,
                min_neighbors,
                power,
            )

            if interpolated_value is not None:
                date_df.loc[idx, "pm25"] = interpolated_value
                date_df.loc[idx, "fallback_type"] = "spatial_idw"
                date_df.loc[idx, "days_offset"] = 0  # Same date
                interpolated_count += 1

        df_list.append(date_df)

    # Combine all dates
    df_final = pd.concat(df_list, ignore_index=True)
    df_final = df_final.sort_values(["sensor_id", "date"])

    # Final statistics
    final_missing = df_final["pm25"].isna().sum()
    filled_count = initial_missing - final_missing

    print(f"\nInterpolation Results:")
    print(f"  Values interpolated: {interpolated_count}")
    print(
        f"  Total filled: {filled_count}/{initial_missing} ({100*filled_count/initial_missing:.1f}% of missing)"
    )
    print(
        f"  Remaining missing: {final_missing}/{total_rows} ({100*final_missing/total_rows:.1f}% of total)"
    )

    # Show breakdown by fallback type
    print(f"\nFallback Type Distribution:")
    fallback_counts = df_final["fallback_type"].value_counts()
    for fb_type, count in fallback_counts.items():
        print(f"  {fb_type}: {count}")

    # Save back to input file (in-place modification)
    df_final.to_parquet(input_path, index=False)
    print(f"\n✓ Updated {input_path} with interpolated values")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Spatial interpolation for missing PM2.5 values using IDW (modifies file in-place)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cache/openaq_daily_pm25.parquet"),
        help="Input parquet with PM2.5 data (will be modified in-place)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=10.0,
        help="Maximum distance in km to search for neighbors (default: 10)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=3,
        help="Minimum neighbors required for interpolation (default: 3)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="IDW power parameter (default: 2.0 for inverse square)",
    )

    args = parser.parse_args()

    interpolate_missing_pm25(
        args.input,
        args.max_distance,
        args.min_neighbors,
        args.power,
    )


if __name__ == "__main__":
    main()
