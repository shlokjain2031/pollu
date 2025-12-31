#!/usr/bin/env python3
"""
Fetch all OpenAQ sensor IDs in a bounding box and save to cache/openaq_sensor_ids.parquet.
"""
import requests
import pandas as pd
import json
from pathlib import Path

API_TOKEN = "5881288ec6eee8a04c8029d8aa1ff991cf7a91c7d10537f8c3288ab5918700cc"  # Replace with your token or pass as argument
BBOX = "72.7763,18.8939,72.9797,19.2701"  # Mumbai region
OUT_PATH = "cache/openaq_sensor_ids.parquet"
OUT_JSON = "cache/openaq_sensor_ids.json"

headers = {"X-API-Key": API_TOKEN}
sensor_rows = []
page = 1
limit = 100



while True:
    params = {"bbox": BBOX, "limit": limit, "page": page}
    r = requests.get("https://api.openaq.org/v3/locations", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        break
    for s in results:
        loc_id = s.get("id")
        if loc_id is None:
            continue
        # Fetch detailed location info to check sensors
        detail_url = f"https://api.openaq.org/v3/locations/{loc_id}"
        detail_r = requests.get(detail_url, headers=headers, timeout=30)
        detail_r.raise_for_status()
        detail_results = detail_r.json().get("results", [])
        if not detail_results:
            continue
        loc = detail_results[0]
        sensors = loc.get("sensors", [])
        # extract location name and coordinates (robust to several key names)
        loc_name = loc.get("name") or loc.get("location") or loc.get("city") or None
        coords = loc.get("coordinates") or {}
        lat = coords.get("latitude") or coords.get("lat") or loc.get("latitude") or loc.get("lat")
        lon = coords.get("longitude") or coords.get("lon") or loc.get("longitude") or loc.get("lon")

        for sensor in sensors:
            # parameter may be a dict or a string depending on API
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
            except Exception:
                # skip non-integer ids
                continue

            if isinstance(param_name, str) and param_name.lower() == "pm25":
                sensor_rows.append(
                    {
                        "sensor_id": sid,
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
    if len(results) < limit:
        break
    page += 1

print(f"Found {len(sensor_rows)} sensor records (including duplicates by sensor id).")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(sensor_rows)
if not df.empty:
    # Keep only records with coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    # Ensure numeric coords
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]) 
    # Deduplicate by rounded lat/lon pairs (unique sensor locations)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["lat_round"] = df["latitude"].round(6)
    df["lon_round"] = df["longitude"].round(6)
    df["latlon_key"] = df["lat_round"].astype(str) + "_" + df["lon_round"].astype(str)
    # keep the record with the lowest sensor_id for each unique lat/lon
    df = df.sort_values(["latlon_key", "sensor_id"]).drop_duplicates(subset=["latlon_key"]).reset_index(drop=True)
    df = df.drop(columns=["lat_round", "lon_round", "latlon_key"])
else:
    df = pd.DataFrame(columns=["sensor_id", "location_id", "location_name", "latitude", "longitude"])

df.to_parquet(OUT_PATH, index=False)
# Also write JSON (array of records) for easy inspection
out_json_path = Path(OUT_JSON)
with out_json_path.open("w") as jf:
    json.dump(df.to_dict(orient="records"), jf, indent=2)
print(f"Wrote sensor info to {OUT_PATH} ({len(df)} unique sensors)")
print(f"Wrote sensor info JSON to {OUT_JSON}")
