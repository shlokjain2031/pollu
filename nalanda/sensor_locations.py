"""Sensor location loading and validation for Pollu target assembly.

This module handles loading sensor metadata from OpenAQ JSON exports
and preparing it for spatial operations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def load_sensor_locations(
    json_path: str | Path,
    require_site_id: bool = True,
) -> pd.DataFrame:
    """Load sensor metadata from openaq_sensor_ids.json.
    
    Args:
        json_path: Path to the OpenAQ sensor JSON file.
        require_site_id: If True, filter out sensors without site_id.
            These sensors can't be matched to CPCB AQI artifacts.
    
    Returns:
        DataFrame with columns:
            - site_id: str, CPCB site identifier
            - latitude: float, WGS84 latitude
            - longitude: float, WGS84 longitude
            - location_name: str, human-readable station name
            
    Raises:
        FileNotFoundError: If json_path doesn't exist.
        ValueError: If no valid sensors found after filtering.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Sensor metadata not found: {json_path}")
    
    with open(json_path, 'r') as f:
        raw_sensors = json.load(f)
    
    # Filter to sensors with required fields
    valid_sensors = []
    for sensor in raw_sensors:
        # Must have coordinates
        if sensor.get('latitude') is None or sensor.get('longitude') is None:
            continue
        
        # Optionally require site_id for CPCB matching
        if require_site_id and sensor.get('site_id') is None:
            continue
            
        valid_sensors.append({
            'site_id': str(sensor.get('site_id', '')),
            'latitude': float(sensor['latitude']),
            'longitude': float(sensor['longitude']),
            'location_name': sensor.get('location_name', ''),
        })
    
    if not valid_sensors:
        raise ValueError(
            f"No valid sensors found in {json_path}. "
            f"Total entries: {len(raw_sensors)}, require_site_id={require_site_id}"
        )
    
    df = pd.DataFrame(valid_sensors)
    
    # Validate coordinates are within Mumbai region (rough bounds)
    # Mumbai: lat ~18.85-19.35, lon ~72.75-73.05
    lat_valid = (df['latitude'] >= 18.5) & (df['latitude'] <= 19.5)
    lon_valid = (df['longitude'] >= 72.5) & (df['longitude'] <= 73.5)
    
    invalid_count = (~(lat_valid & lon_valid)).sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} sensors have coordinates outside Mumbai bounds")
    
    # Remove duplicates by site_id
    df = df.drop_duplicates(subset='site_id', keep='first')
    
    return df.reset_index(drop=True)


def get_sensor_by_site_id(
    sensors_df: pd.DataFrame,
    site_id: str,
) -> Optional[pd.Series]:
    """Look up a sensor by its site_id.
    
    Args:
        sensors_df: DataFrame from load_sensor_locations().
        site_id: CPCB site identifier to look up.
        
    Returns:
        Series with sensor data, or None if not found.
    """
    matches = sensors_df[sensors_df['site_id'] == str(site_id)]
    if len(matches) == 0:
        return None
    return matches.iloc[0]
