#!/usr/bin/env python3
"""
Fetch all WAQI air quality stations in Mumbai and save their UIDs to cache.
"""
import requests
import json
from pathlib import Path

WAQI_TOKEN = "5c88c4c3a4397350d3afdd1e47115fa2b1a3878e"
WAQI_URL = "https://api.waqi.info/search/"
KEYWORD = "mumbai"
OUT_FILE = "cache/waqi_station_uids.json"

print(f"Fetching WAQI stations for '{KEYWORD}'...")

try:
    response = requests.get(
        WAQI_URL,
        params={"token": WAQI_TOKEN, "keyword": KEYWORD},
        timeout=30
    )
    response.raise_for_status()
    data = response.json()
    
    print(f"Response status: {data.get('status')}")
    
    results = data.get("data", [])
    print(f"Found {len(results)} results")
    
    # Extract UIDs
    uids = []
    stations = []
    
    for result in results:
        uid = result.get("uid")
        if uid:
            uids.append(uid)
            stations.append({
                "uid": uid,
                "station": result.get("station", {}).get("name", "Unknown"),
                "country": result.get("station", {}).get("country", "Unknown"),
            })
            print(f"  ✓ {result.get('station', {}).get('name', 'Unknown')} (uid: {uid})")
    
    print(f"\nExtracted {len(uids)} UIDs")
    
    # Save to cache
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUT_FILE, "w") as f:
        json.dump({"uids": uids, "stations": stations}, f, indent=2)
    
    print(f"✓ Saved to {OUT_FILE}")
    
except requests.exceptions.RequestException as e:
    print(f"✗ Request failed: {e}")
except json.JSONDecodeError as e:
    print(f"✗ JSON decode error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
