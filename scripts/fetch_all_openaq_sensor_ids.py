#!/usr/bin/env python3
"""
Fetch all OpenAQ sensor IDs in a bounding box and save to cache/openaq_sensor_ids.parquet.
"""
import requests
import pandas as pd
from pathlib import Path

API_TOKEN = "5881288ec6eee8a04c8029d8aa1ff991cf7a91c7d10537f8c3288ab5918700cc"  # Replace with your token or pass as argument
BBOX = "72.65,18.85,73.05,19.35"  # Mumbai region
OUT_PATH = "cache/openaq_sensor_ids.parquet"

headers = {"X-API-Key": API_TOKEN}
sensor_ids = set()
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
        sensors = detail_results[0].get("sensors", [])
        for sensor in sensors:
            param = sensor.get("parameter", {})
            if param.get("name") == "pm25" and sensor.get("id") is not None:
                sensor_ids.add(int(sensor["id"]))
    if len(results) < limit:
        break
    page += 1

print(f"Found {len(sensor_ids)} sensors.")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"sensor_id": sorted(sensor_ids)}).to_parquet(OUT_PATH, index=False)
print(f"Wrote sensor IDs to {OUT_PATH}")
