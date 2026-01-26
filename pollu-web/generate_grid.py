#!/usr/bin/env python3
"""
Generate 100m x 100m grid cells for Mumbai.

Each cell is approximately 100m x 100m on the ground.
Converts to degrees: lat ≈ 0.0009°, lon ≈ 0.000955°
"""

import json
import math
from pathlib import Path

def generate_mumbai_grid(output_path="data/mumbai_grid.geojson", sample_size=None):
    """
    Generate 100m x 100m grid cells covering Mumbai.
    
    Args:
        output_path: Path to save GeoJSON
        sample_size: If set, only generate first N cells (for testing)
    """
    
    # Mumbai bounds (from boundary GeoJSON)
    min_lat, max_lat = 18.92, 19.28
    min_lon, max_lon = 72.82, 72.95
    
    # Cell size in degrees (100m x 100m)
    # At Mumbai's latitude (19°N):
    # 1° latitude ≈ 111 km
    # 1° longitude ≈ 111 km * cos(19°) ≈ 105 km
    
    lat_delta = 100 / 111000  # ~0.0009°
    lon_delta = 100 / (111000 * math.cos(math.radians(19)))  # ~0.000955°
    
    print(f"Cell size: {lat_delta:.6f}° lat × {lon_delta:.6f}° lon")
    print(f"Coverage: {min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E")
    
    features = []
    cell_count = 0
    grid_id_counter = 1
    
    # Generate grid
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            cell_count += 1
            
            # Create cell polygon (counterclockwise for GeoJSON)
            coords = [
                [lon, lat],
                [lon + lon_delta, lat],
                [lon + lon_delta, lat + lat_delta],
                [lon, lat + lat_delta],
                [lon, lat]  # Close polygon
            ]
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "grid_id": f"g{grid_id_counter:06d}",
                    "pm25_value": None  # Will be filled by API
                }
            }
            
            features.append(feature)
            grid_id_counter += 1
            
            if sample_size and len(features) >= sample_size:
                break
            
            lon += lon_delta
        
        if sample_size and len(features) >= sample_size:
            break
        
        lat += lat_delta
    
    # Create GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "description": "30m x 30m grid cells for Mumbai",
            "cell_size_meters": 30,
            "total_cells": len(features),
            "coverage": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            }
        }
    }
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"\n✅ Generated {len(features)} grid cells")
    print(f"📁 Saved to: {output_file}")
    
    return len(features)

if __name__ == "__main__":
    import sys
    
    # Optional: pass sample size as argument
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if sample:
        print(f"\n📊 Generating {sample} sample cells (testing mode)\n")
        generate_mumbai_grid(sample_size=sample)
    else:
        print("\n📊 Generating full grid (this will take a moment)\n")
        generate_mumbai_grid()
