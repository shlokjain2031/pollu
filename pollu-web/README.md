# Mumbai Air Quality Heatmap

A simple, static web application to visualize AQI air quality predictions across Mumbai in 30m x 30m grid cells.

## Features

- Interactive map with OpenStreetMap basemap
- 30m x 30m grid cells colored by AQI levels
- AQI color coding (Good → Moderate → Unhealthy → Very Unhealthy → Hazardous)
- Click on any grid cell to see detailed AQI value
- Mumbai boundary overlay

## Local Development

Simply open `index.html` in your browser:

```bash
open index.html
```

Or use a simple HTTP server:

```bash
# Python 3
python3 -m http.server 8000

# Then visit http://localhost:8000
```

## Data Files

You need to create two GeoJSON files in the `data/` directory:

### 1. Mumbai Boundary (`data/mumbai_boundary.geojson`)
Copy from `../resources/mumbai_wards.geojson` or create a simplified version

### 2. Grid Cells (`data/mumbai_grid.geojson`)
Format:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lng1, lat1], [lng2, lat2], ...]]
      },
      "properties": {
        "grid_id": "g000001",
        "pm25_value": 45.2
      }
    }
  ]
}
```

## Deployment to Vercel

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Deploy:
   ```bash
   vercel
   ```

3. Follow prompts to link project

## Technology Stack

- **Leaflet.js**: Interactive map library
- **OpenStreetMap**: Free map tiles
- **Plain HTML/CSS/JS**: No build step required
- **Vercel**: Static site hosting with CDN

## Future Enhancements

- Real-time updates from API
- Time slider to see historical data
- Search/filter by location
- Export data as CSV
