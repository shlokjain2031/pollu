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

const AQI_LEVELS = [
    { max: 35, category: 'Good', color: '#2ecc71' },
    { max: 75, category: 'Satisfactory', color: '#a9d82f' },
    { max: 115, category: 'Moderately Polluted', color: '#ffd700' },
    { max: 150, category: 'Poor', color: '#ff8c00' },
    { max: 250, category: 'Very Poor', color: '#e74c3c' },
    { max: 500, category: 'Severe', color: '#8b0000' }
];

const AQI_LEVEL_WIDTH_PERCENT = 100 / AQI_LEVELS.length;

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
    
    // Show AQI info box with loading state
    pm25Info.style.display = 'block';
    document.getElementById('pm25-value').innerHTML = 'Loading AQI data...';
    document.getElementById('pm25-category').textContent = 'Please wait';
    
    // Fetch AQI prediction from API
    getPM25ForLocation(lat, lon)
        .then(result => {
            console.log('AQI prediction received:', result);
            updatePM25Display(result.pm25);
            
            // Log additional info
            console.log(`Grid ID: ${result.grid_id}, Source: ${result.source}, Timestamp: ${result.timestamp}`);
        })
        .catch(error => {
            console.error('Failed to fetch AQI prediction:', error);
            
            // Show error in UI
            document.getElementById('pm25-value').innerHTML = 'Failed to load AQI data';
            document.getElementById('pm25-category').textContent = 'Error';
            document.getElementById('pm25-value').style.color = '#d32f2f';
            document.getElementById('pm25-category').style.color = '#d32f2f';
        });
    
    // Clear search results
    searchResults.innerHTML = '';
}

// Function to get AQI category and color
function getPM25Info(value) {
    const normalizedValue = Math.max(0, value);
    const level = AQI_LEVELS.find(level => normalizedValue <= level.max) || AQI_LEVELS[AQI_LEVELS.length - 1];
    return { category: level.category, color: level.color };
}

// Function to update AQI info display
function updatePM25Display(value) {
    const { category, color } = getPM25Info(value);
    
    // Update heading and caption with dynamic color
    document.getElementById('pm25-value').textContent = `AQI is ${value} µg/m³ here`;
    document.getElementById('pm25-value').innerHTML = `AQI is ${value} µg/m³ here`;
    document.getElementById('pm25-value').style.color = color;
    document.getElementById('pm25-category').textContent = category;
    document.getElementById('pm25-category').style.color = color;
    
    // Update scale indicator position using equal width bands
    const indicator = document.getElementById('scale-indicator');
    const clampedValue = Math.max(0, Math.min(value, AQI_LEVELS[AQI_LEVELS.length - 1].max));
    let indicatorPercent = 100;
    let cumulativePercent = 0;
    let previousMax = 0;

    for (const level of AQI_LEVELS) {
        if (clampedValue <= level.max) {
            const levelRange = level.max - previousMax || 1;
            const levelProgress = Math.min(Math.max(clampedValue - previousMax, 0), levelRange);
            const fraction = levelProgress / levelRange;
            indicatorPercent = cumulativePercent + fraction * AQI_LEVEL_WIDTH_PERCENT;
            break;
        }

        cumulativePercent += AQI_LEVEL_WIDTH_PERCENT;
        previousMax = level.max;
    }

    indicator.style.left = `${Math.min(indicatorPercent, 100)}%`;
    
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
        const nominatimUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&bounded=1&viewbox=${bbox}&limit=5&accept-language=en`;
        
        // Use CORS proxy to bypass browser restrictions
        const corsProxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(nominatimUrl)}`;
        
        fetch(corsProxyUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`API error: ${response.status}`);
                }
                return response.json();
            })
            .then(result => {
                let data;
                try {
                    data = JSON.parse(result.contents);
                } catch (e) {
                    throw new Error('Failed to parse API response');
                }
                
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
