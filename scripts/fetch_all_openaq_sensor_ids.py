#!/usr/bin/env python3
"""
Fetch all OpenAQ sensor IDs in a bounding box and save to cache/openaq_sensor_ids.json.

OpenAQ API v3 Reference:
- Locations endpoint: GET https://api.openaq.org/v3/locations
- Parameters: bbox (bounding box), parameter (pollutant name), limit, offset
- Response: {results: [{id, name, coordinates: {latitude, longitude}, sensors: [{id, parameter: {name}}]}]}
"""
import requests
import pandas as pd
import json
import time
from pathlib import Path

API_TOKEN = None  # OpenAQ v3 works without API key for basic queries
BBOX = "72.7763,18.8939,72.9797,19.2701"  # Mumbai region (west,south,east,north)
OUT_JSON = "cache/openaq_sensor_ids.json"

# Session for connection pooling and retry
session = requests.Session()
if API_TOKEN:
    session.headers.update({"X-API-Key": API_TOKEN})

sensor_rows = []
offset = 0
limit = 100
consecutive_empty = 0

print("Fetching PM2.5 sensors from OpenAQ API v3...")
print(f"Region: bbox={BBOX}")
print()

while consecutive_empty < 2:
    try:
        # Query: locations with PM2.5 parameter in Mumbai bbox
        params = {
            "bbox": BBOX,
            "parameter": "pm25",  # Filter for PM2.5 only
            "limit": limit,
            "offset": offset,
        }
        
        url = "https://api.openaq.org/v3/locations"
        print(f"Fetching offset={offset}, limit={limit}...")
        
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            consecutive_empty += 1
            print(f"  ✓ No results at offset {offset} (consecutive empty: {consecutive_empty})")
            offset += limit
            time.sleep(0.5)
            continue
        
        consecutive_empty = 0
        
        print(f"  ✓ Found {len(results)} locations with PM2.5 sensors")
        
        for loc in results:
            loc_id = loc.get("id")
            loc_name = loc.get("name")
            
            coords = loc.get("coordinates", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            
            # Skip if missing coordinates
            if lat is None or lon is None:
                continue
            
            # Extract PM2.5 sensors
            sensors = loc.get("sensors", [])
            
            for sensor in sensors:
                # Parameter can be string or dict
                param = sensor.get("parameter", {})
                if isinstance(param, dict):
                    param_name = param.get("name", "").lower()
                else:
                    param_name = str(param).lower()
                
                # Only PM2.5 sensors
                if "pm25" not in param_name and "pm2.5" not in param_name:
                    continue
                
                sensor_id = sensor.get("id")
                if sensor_id is None:
                    continue
                
                try:
                    sensor_id = int(sensor_id)
                except (ValueError, TypeError):
                    continue
                
                sensor_rows.append({
                    "sensor_id": sensor_id,
                    "location_id": loc_id,
                    "location_name": loc_name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                })
        
        offset += limit
        time.sleep(0.5)  # Rate limiting
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request failed: {e}")
        print(f"  Retrying in 5 seconds...")
        time.sleep(5)
        continue
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON decode error: {e}")
        print(f"  Skipping this batch...")
        offset += limit
        time.sleep(1)
        continue
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        break

print()
print(f"Found {len(sensor_rows)} total sensor records")

# Deduplicate by sensor_id (keep one copy of each unique sensor)
if sensor_rows:
    df = pd.DataFrame(sensor_rows)
    
    # Drop duplicates by sensor_id, keeping first occurrence
    df = df.drop_duplicates(subset=["sensor_id"], keep="first")
    
    print(f"After deduplication: {len(df)} unique sensors")
    print()
    print("Sample sensors:")
    for idx, row in df.head(10).iterrows():
        print(f"  {row['location_name']:<40} ({row['latitude']:.4f}, {row['longitude']:.4f})")
    
    # Write JSON
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    
    print()
    print(f"✓ Wrote {len(df)} sensors to {OUT_JSON}")
else:
    print("✗ No sensors found!")
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump([], f)

