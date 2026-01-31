/**
 * Pollu Prediction API Integration
 * 
 * Handles:
 * - Loading grid data from parquet
 * - Converting lat/lon to grid_id
 * - Fetching PM2.5 predictions from coordinator API
 */

// API Configuration
// Use localhost:8080 for local development, or set to cloudflare tunnel for production
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8080'
    : 'https://tries-satellite-extra-configuring.trycloudflare.com';

// Grid data cache
let gridData = null;
let gridLoading = false;
let gridLoadError = null;

/**
 * Load grid parquet data using parquet-wasm
 */
async function loadGridData() {
    if (gridData) return gridData;
    if (gridLoading) {
        // Wait for existing load to complete
        while (gridLoading) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return gridData;
    }
    
    gridLoading = true;
    try {
        const response = await fetch('grid.parquet');
        if (!response.ok) {
            throw new Error(`Failed to fetch grid.parquet: ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        
        // Use parquet-wasm to read the file
        // Note: You'll need to include parquet-wasm library in your HTML
        // <script src="https://unpkg.com/parquet-wasm@0.6.0/esm/parquet_wasm.js" type="module"></script>
        
        // For now, we'll use a simple CSV-like structure
        // In production, use proper parquet parsing
        
        // Convert to simple array of objects
        // Each row: {grid_id, centroid_lon, centroid_lat, ...}
        
        // TEMPORARY: Parse using DuckDB WASM or convert to JSON first
        // For this implementation, let's assume we have a JSON version
        // Return empty for now - you'll need to convert the parquet to JSON
        gridData = [];
        gridLoading = false;
        
        return gridData;
    } catch (error) {
        gridLoadError = error;
        gridLoading = false;
        throw error;
    }
}

/**
 * Find nearest grid cell for given lat/lon coordinates
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @returns {string|null} - Grid ID (e.g., "g00000001") or null if not found
 */
function findNearestGridCell(lat, lon) {
    if (!gridData || gridData.length === 0) {
        return null;
    }
    
    let nearestCell = null;
    let minDistance = Infinity;
    
    // Simple euclidean distance (works for small areas like Mumbai)
    for (const cell of gridData) {
        const dLat = cell.centroid_lat - lat;
        const dLon = cell.centroid_lon - lon;
        const distance = Math.sqrt(dLat * dLat + dLon * dLon);
        
        if (distance < minDistance) {
            minDistance = distance;
            nearestCell = cell;
        }
    }
    
    return nearestCell ? nearestCell.grid_id : null;
}

/**
 * Get current timestamp rounded to the hour in ISO format
 * @returns {string} - ISO timestamp (e.g., "2026-01-26T19:00:00Z")
 */
function getCurrentHourUTC() {
    const now = new Date();
    // Round down to the nearest hour
    now.setMinutes(0, 0, 0);
    return now.toISOString();
}

/**
 * Fetch AQI prediction from the coordinator API
 * @param {string} gridId - Grid ID (e.g., "g00000001")
 * @param {string} timestampHour - ISO timestamp (e.g., "2026-01-26T19:00:00Z")
 * @returns {Promise<Object>} - Prediction response {request_id, pm25}
 */
async function fetchPrediction(gridId, timestampHour) {
    const url = `${API_BASE_URL}/predict`;
    
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                grid_id: gridId,
                timestamp_hour: timestampHour
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error?.message || `API request failed: ${response.statusText}`);
        }
        
        return data;
    } catch (error) {
        throw error;
    }
}

/**
 * Get AQI prediction for a specific location
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @returns {Promise<Object>} - {pm25: number, grid_id: string, timestamp: string, request_id: string}
 */
async function getPM25ForLocation(lat, lon) {
    // Ensure grid data is loaded
    if (!gridData || gridData.length === 0) {
        await loadGridData();
    }
    
    // Find nearest grid cell
    const gridId = findNearestGridCell(lat, lon);
    if (!gridId) {
        throw new Error('Could not find grid cell for this location');
    }
    
    // Get current timestamp
    const timestamp = getCurrentHourUTC();
    
    // Fetch prediction
    const result = await fetchPrediction(gridId, timestamp);
    
    return {
        pm25: Math.round(result.pm25),
        grid_id: gridId,
        timestamp: timestamp,
        request_id: result.request_id
    };
}

/**
 * TEMPORARY: Use mock grid data for testing
 * This generates a simple grid covering Mumbai
 */
function useMockGridData() {
    
    // Generate mock grid data
    // Mumbai bounds: lat 18.8939-19.2701, lon 72.7763-72.9797
    // 30m resolution ≈ 0.00027° (approximate)
    
    const minLat = 18.8939;
    const maxLat = 19.2701;
    const minLon = 72.7763;
    const maxLon = 72.9797;
    const resolution = 0.00027; // ~30 meters
    
    gridData = [];
    let gridIdx = 0;
    
    for (let lat = minLat; lat <= maxLat; lat += resolution) {
        for (let lon = minLon; lon <= maxLon; lon += resolution) {
            gridIdx++;
            gridData.push({
                grid_id: `g${String(gridIdx).padStart(8, '0')}`,
                centroid_lat: lat,
                centroid_lon: lon
            });
            
            // Limit to ~500K cells (similar to actual grid)
            if (gridIdx >= 509209) break;
        }
        if (gridIdx >= 509209) break;
    }
    
}

// Auto-initialize with mock data for now
// TODO: Replace with actual parquet parsing
useMockGridData();
