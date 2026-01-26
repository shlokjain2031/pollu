// Initialize map with Mumbai bounding box
// BBOX: "72.7763,18.8939,72.9797,19.2701"
const southWest = L.latLng(18.8939, 72.7763);
const northEast = L.latLng(19.2701, 72.9797);
const bounds = L.latLngBounds(southWest, northEast);

const map = L.map('map', {
    maxBounds: bounds,
    maxBoundsViscosity: 1.0,
    minZoom: 12
}).fitBounds(bounds, { maxZoom: 13 });

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
    minZoom: 10
}).addTo(map);


// Add grey overlay for areas outside BBOX
const maxBounds = L.latLngBounds(
    L.latLng(-90, -180),
    L.latLng(90, 180)
);

// Create rectangles for north, south, east, west areas outside bbox
const overlayStyle = {
    color: '#999',
    weight: 0,
    fillColor: '#ccc',
    fillOpacity: 0.5
};

// North overlay
L.rectangle([[bounds.getNorth(), -180], [90, 180]], overlayStyle).addTo(map);
// South overlay
L.rectangle([[-90, -180], [bounds.getSouth(), 180]], overlayStyle).addTo(map);
// West overlay
L.rectangle([[bounds.getSouth(), -180], [bounds.getNorth(), bounds.getWest()]], overlayStyle).addTo(map);
// East overlay
L.rectangle([[bounds.getSouth(), bounds.getEast()], [bounds.getNorth(), 180]], overlayStyle).addTo(map);

// Load Mumbai outer boundary only
fetch('data/mumbai_boundary.geojson')
    .then(response => response.json())
    .then(data => {
        console.log('Mumbai boundary loaded');
    })
    .catch(error => {
        console.error('Error loading boundary:', error);
    });

// Function to place marker at location
function placeMarker(lat, lon, locationName = null) {
    // Remove previous marker if exists
    if (currentMarker) {
        map.removeLayer(currentMarker);
    }
    
    // Add new marker at location
    currentMarker = L.marker([lat, lon], {
        icon: L.icon({
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        })
    }).addTo(map);
    
    if (locationName) {
        currentMarker.bindPopup(locationName);
    }
    
    // Update search input with lat/long
    searchInput.value = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    
    // TODO: Show loading state in PM2.5 info box while fetching from API
    // Display "Loading..." or spinner in the heading until API response is received
    
    // TODO: Send /predict request to API with lat/lon coordinates
    // Example: fetch('/predict', { method: 'POST', body: JSON.stringify({ lat, lon }) })
    //   .then(response => response.json())
    //   .then(data => {
    //     const pm25Value = data.pm25; // or appropriate field from API response
    //     updatePM25Display(pm25Value);
    //   })
    //   .catch(error => {
    //     // TODO: Show error message in PM2.5 info box if API request fails
    //     // Display error text like "Failed to fetch data" in the heading
    //     console.error('API Error:', error);
    //   })
    // For now, using random PM2.5 value as placeholder
    const randomPM25 = Math.floor(Math.random() * 400) + 10; // 10-410
    
    // Update PM2.5 display
    updatePM25Display(randomPM25);
    
    // Show PM2.5 info box
    pm25Info.style.display = 'block';
    
    // Clear search results
    searchResults.innerHTML = '';
}

// Function to get PM2.5 category and color
function getPM25Info(value) {
    let category, color;
    
    if (value <= 35) {
        category = 'Good';
        color = '#2ecc71';
    } else if (value <= 75) {
        category = 'Satisfactory';
        color = '#a9d82f';
    } else if (value <= 115) {
        category = 'Moderately Polluted';
        color = '#ffd700';
    } else if (value <= 150) {
        category = 'Poor';
        color = '#ff8c00';
    } else if (value <= 250) {
        category = 'Very Poor';
        color = '#e74c3c';
    } else {
        category = 'Severe';
        color = '#8b0000';
    }
    
    return { category, color };
}

// Function to update PM2.5 info display
function updatePM25Display(value) {
    const { category, color } = getPM25Info(value);
    
    // Update heading and caption with dynamic color
    document.getElementById('pm25-value').textContent = `PM2.5 is ${value} µg/m³ here`;
    document.getElementById('pm25-value').innerHTML = `PM<sub>2.5</sub> is ${value} µg/m³ here`;
    document.getElementById('pm25-value').style.color = color;
    document.getElementById('pm25-category').textContent = category;
    document.getElementById('pm25-category').style.color = color;
    
    // Update scale indicator position
    // Scale: 0-500, map to 0-100%
    const percentage = Math.min((value / 500) * 100, 100);
    const indicator = document.getElementById('scale-indicator');
    indicator.style.left = `${percentage}%`;
    
    // Update timestamp - UTC minutes since start of hour
    const now = new Date();
    const minutesSinceHour = now.getUTCMinutes();
    document.getElementById('pm25-updated').textContent = `Updated ${minutesSinceHour} minutes ago`;
}

// Handle map clicks
map.on('click', (e) => {
    placeMarker(e.latlng.lat, e.latlng.lng);
    map.setView([e.latlng.lat, e.latlng.lng], 15);
});

// Search functionality using Nominatim (OSM)
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const pm25Info = document.getElementById('pm25-info');
let searchTimeout;
let currentMarker = null;

searchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    const query = e.target.value.trim();
    
    if (query.length < 2) {
        searchResults.innerHTML = '';
        return;
    }
    
    // Debounce search
    searchTimeout = setTimeout(() => {
        // Search within Mumbai bbox
        const bbox = '72.7763,18.8939,72.9797,19.2701';
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&bounded=1&viewbox=${bbox}&limit=5&accept-language=en`;
        
        fetch(url, {
            headers: {
                'Accept': 'application/json',
                'User-Agent': 'Pollu-Web (https://github.com/shlokjain/pollu)'
            }
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`API error: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                searchResults.innerHTML = '';
                if (!Array.isArray(data) || data.length === 0) {
                    searchResults.innerHTML = '<li>No results found</li>';
                    return;
                }
                
                data.forEach(result => {
                    const li = document.createElement('li');
                    li.textContent = result.display_name;
                    li.addEventListener('click', () => {
                        // Zoom to location
                        const lat = parseFloat(result.lat);
                        const lon = parseFloat(result.lon);
                        map.setView([lat, lon], 15);
                        
                        // Place marker using the helper function
                        placeMarker(lat, lon, result.display_name);
                    });
                    searchResults.appendChild(li);
                });
            })
            .catch(error => {
                console.error('Search error:', error);
                searchResults.innerHTML = '<li style="color: #d32f2f;">Search error - please try again</li>';
            });
    }, 300);
});
