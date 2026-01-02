"""Sensor location loading and validation for Pollu target assembly.

This module handles loading sensor metadata from OpenAQ API
and preparing it for spatial operations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


OPENAQ_API_BASE = "https://api.openaq.org/v3"
CPCB_DATA_DIR = Path(__file__).parent.parent / "cpcb_aqi_data"


def _normalize_location_name(location_name: str) -> str:
    """Convert location name to filename format.
    
    Rules:
    - Spaces and commas are always converted to underscores
    - Other special characters are retained if at least one immediately adjacent side
      has an alphanumeric character (letter or digit)
    - Leading/trailing underscores and consecutive underscores are collapsed
    
    Examples:
        "Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB" 
        -> "Chhatrapati_Shivaji_Intl._Airport_(T2)_Mumbai_MPCB"
        
        "Mindspace-Malad West, Mumbai - MPCB" 
        -> "Mindspace-Malad_West_Mumbai_MPCB"
    """
    result = []
    
    for i, char in enumerate(location_name):
        if char in (',', ' '):
            # Comma and space always become underscore
            result.append('_')
        elif char.isalnum():
            # Keep alphanumeric characters
            result.append(char)
        else:
            # Special character - keep only if at least one adjacent side is alphanumeric
            left_is_alnum = i > 0 and location_name[i-1].isalnum()
            right_is_alnum = i < len(location_name) - 1 and location_name[i+1].isalnum()
            
            if left_is_alnum or right_is_alnum:
                result.append(char)
            else:
                result.append('_')
    
    # Join and clean up
    result_str = ''.join(result).strip('_')
    
    # Collapse consecutive underscores to single underscore
    result_str = re.sub(r'_+', '_', result_str)
    
    return result_str


def _extract_site_id_from_filename(location_name_updated: str, data_dir: Path = CPCB_DATA_DIR) -> Optional[str]:
    """Find first file matching location_name_updated and extract site_id.
    
    Args:
        location_name_updated: Normalized location name (e.g., "Colaba_Mumbai_MPCB")
        data_dir: Directory containing CPCB AQI data files
        
    Returns:
        Site ID as string, or None if no matching file found
    """
    # Pattern: Raw_data_1Hr_{year}_site_{site_id}_{location_name_updated}
    pattern = re.compile(r'Raw_data_1Hr_\d{4}_site_(\d+)_' + re.escape(location_name_updated))
    
    if not data_dir.exists():
        print(f"Warning: CPCB data directory not found: {data_dir}")
        return None
    
    for filepath in sorted(data_dir.glob("Raw_data_*.csv")):
        match = pattern.search(filepath.name)
        if match:
            return match.group(1)
    print("faulty location name:", location_name_updated)

    return None


def load_sensor_locations(
    openaq_token: str,
    bbox: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """Load sensor metadata from OpenAQ API and match to CPCB site IDs.
    
    Fetches PM2.5 sensor locations from OpenAQ API, then matches each sensor
    to a CPCB site ID by finding the corresponding file in cpcb_aqi_data/.
    
    Args:
        openaq_token: OpenAQ API token for authentication
        bbox: Bounding box as "west,south,east,north" (EPSG:4326)
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with columns:
            - sensor_id: int, OpenAQ sensor identifier
            - location_id: int, OpenAQ location identifier
            - location_name: str, human-readable station name
            - latitude: float, WGS84 latitude
            - longitude: float, WGS84 longitude
            - site_id: str, CPCB site identifier (matched from filenames)
            
    Raises:
        requests.RequestException: If API request fails
        ValueError: If no valid sensors found
    """
    headers = {"X-API-Key": openaq_token}
    
    sensor_rows = []
    page = 1
    limit = 100
    
    print(f"Fetching PM2.5 sensors from OpenAQ API (bbox={bbox})...")
    
    while True:
        params = {"bbox": bbox, "limit": limit, "page": page}
        
        try:
            r = requests.get(
                f"{OPENAQ_API_BASE}/locations",
                params=params,
                headers=headers,
                timeout=timeout,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"✗ API request failed: {e}")
            break
        
        results = r.json().get("results", [])
        if not results:
            break
        
        for location in results:
            loc_id = location.get("id")
            if loc_id is None:
                continue
            
            # Fetch detailed location info to check sensors
            detail_url = f"{OPENAQ_API_BASE}/locations/{loc_id}"
            try:
                detail_r = requests.get(detail_url, headers=headers, timeout=timeout)
                detail_r.raise_for_status()
            except requests.RequestException:
                continue
            
            detail_results = detail_r.json().get("results", [])
            if not detail_results:
                continue
            
            loc = detail_results[0]
            sensors = loc.get("sensors", [])
            
            # Extract location name and coordinates
            loc_name = loc.get("name") or loc.get("location") or loc.get("city")
            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude") or coords.get("lat") or loc.get("latitude")
            lon = coords.get("longitude") or coords.get("lon") or loc.get("longitude")
            
            # Filter for PM2.5 sensors
            for sensor in sensors:
                param = sensor.get("parameter")
                if isinstance(param, dict):
                    param_name = param.get("name") or param.get("parameter")
                else:
                    param_name = param
                
                sensor_id = sensor.get("id")
                if sensor_id is None:
                    continue
                
                try:
                    sid = int(sensor_id)
                except (ValueError, TypeError):
                    continue
                
                if isinstance(param_name, str) and param_name.lower() == "pm25":
                    sensor_rows.append({
                        "sensor_id": sid,
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "latitude": lat,
                        "longitude": lon,
                    })
        
        if len(results) < limit:
            break
        page += 1
    
    print(f"  → Fetched {len(sensor_rows)} sensor records")
    
    # Create DataFrame and clean
    df = pd.DataFrame(sensor_rows)
    
    if df.empty:
        raise ValueError("No sensors found from OpenAQ API")
    
    # Filter to valid coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    
    # Deduplicate by rounded lat/lon (unique sensor locations)
    df["lat_round"] = df["latitude"].round(6)
    df["lon_round"] = df["longitude"].round(6)
    df["latlon_key"] = df["lat_round"].astype(str) + "_" + df["lon_round"].astype(str)
    df = df.sort_values(["latlon_key", "sensor_id"])
    df = df.drop_duplicates(subset=["latlon_key"]).reset_index(drop=True)
    df = df.drop(columns=["lat_round", "lon_round", "latlon_key"])
    
    print(f"  ✓ {len(df)} unique PM2.5 sensor locations")
    
    # Match each sensor to CPCB site_id by finding corresponding file
    print("Matching sensors to CPCB site IDs...")
    
    site_ids = []
    matched_count = 0
    
    for idx, row in df.iterrows():
        loc_name = row.get("location_name") or ""
        location_name_updated = _normalize_location_name(loc_name)
        site_id = _extract_site_id_from_filename(location_name_updated)
        
        if site_id:
            matched_count += 1
        
        site_ids.append(site_id)
    
    df["site_id"] = site_ids
    
    # Print all normalized location names used for matching
    location_names_normalized = [
        _normalize_location_name(str(n)) for n in df["location_name"].fillna("").tolist()
    ]

    print(f"  ✓ Matched {matched_count}/{len(df)} sensors to CPCB site IDs")
    
    # Filter to only sensors with site_id
    df_matched = df[df["site_id"].notna()].copy()
    
    if df_matched.empty:
        raise ValueError("No sensors could be matched to CPCB site IDs")
    
    return df_matched.reset_index(drop=True)


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
