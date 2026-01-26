# Pollu Web - Prediction API Integration

## Setup Complete ✅

The web app now fetches real-time PM2.5 predictions from the coordinator API!

## Files Modified/Created

### New Files:
- `js/prediction-api.js` - API integration module
- `test-api.html` - API testing page
- `grid.parquet` - Symlink to Mumbai grid data (509,209 cells)

### Modified Files:
- `index.html` - Added prediction-api.js script
- `js/map.js` - Integrated real API calls instead of mock data

## How It Works

1. **Grid Loading**: Uses mock grid data (~509K cells) covering Mumbai
2. **Location Click**: When user clicks on map:
   - Finds nearest grid cell to lat/lon
   - Gets current UTC hour timestamp
   - Calls `/predict` API with grid_id and timestamp
   - Displays real PM2.5 value

3. **API Endpoint**: 
   ```
   POST https://geneva-reverse-nothing-concentrations.trycloudflare.com/predict
   ```

## Testing

### Option 1: Test Page
Open `test-api.html` in your browser to test the API integration:
- Test direct API calls
- Test specific Mumbai locations (Mumbai Central, Bandra)

### Option 2: Main App
Open `index.html` and click anywhere on the Mumbai map to get real PM2.5 data!

## Current Implementation

**✅ Working:**
- API integration with coordinator
- Mock grid data (509K cells)
- Lat/lon → Grid ID conversion
- Current timestamp generation
- Real PM2.5 values from API
- Error handling

**⚠️ Note:**
- Using mock grid data (evenly distributed 30m grid)
- For production: Convert `grid.parquet` to JSON for better web performance
- Or use proper parquet-wasm library to parse the .parquet file

## API Response Example

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "pm25": 124.91,
  "source": "duckdb"
}
```

## Next Steps (Optional)

1. **Convert Grid to JSON**: For faster loading
   ```bash
   python -c "import pandas as pd; df = pd.read_parquet('grid.parquet'); df.to_json('grid.json', orient='records')"
   ```

2. **Add Spatial Index**: Use R-tree or similar for faster nearest-neighbor search

3. **Add Caching**: Cache grid lookups in browser localStorage

4. **Add Loading Indicators**: Better UX while fetching predictions

## Serving the Web App

```bash
# Simple Python server
cd pollu-web
python3 -m http.server 8000

# Then open: http://localhost:8000
```

Or use any static file server (nginx, Apache, etc.)
