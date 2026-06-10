/**
 * Pollu Prediction API client.
 *
 * The coordinator resolves lat/lon to the nearest Delhi grid cell server-side,
 * so the frontend just sends coordinates and the current hour.
 */

// Use localhost:8080 for local development, or the tunnel for production
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8080'
    : 'https://treated-fred-encountered-supplied.trycloudflare.com';

/**
 * Current UTC hour in the API's canonical format (e.g. "2026-06-09T14:00:00Z").
 */
function currentTimestampHour() {
    const now = new Date();
    now.setUTCMinutes(0, 0, 0);
    return now.toISOString().replace(/\.\d{3}Z$/, 'Z');
}

/**
 * Fetch the PM2.5 prediction for a location.
 *
 * @param {number} lat
 * @param {number} lon
 * @returns {Promise<{pm25: number, lat: number, lon: number, request_id: string}>}
 * @throws {Error} with .retryable and .errorType set when the API returns an error
 */
async function getPM25ForLocation(lat, lon) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon, timestamp_hour: currentTimestampHour() }),
    });

    const body = await response.json();
    if (!response.ok) {
        const detail = body && body.error ? body.error : {};
        const error = new Error(detail.message || `API error ${response.status}`);
        error.errorType = detail.error_type || 'Unknown';
        error.retryable = Boolean(detail.retryable);
        throw error;
    }

    return {
        pm25: Math.round(body.pm25 * 10) / 10,
        lat: body.lat,
        lon: body.lon,
        request_id: body.request_id,
    };
}

/**
 * Check API health. Returns the health document or null if unreachable.
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        return response.ok ? await response.json() : null;
    } catch {
        return null;
    }
}
