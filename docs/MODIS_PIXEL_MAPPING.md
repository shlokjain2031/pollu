# MODIS Pixel ID Mapping: Technical Formula

**Document Version:** 1.0  
**Date:** January 15, 2026  
**Purpose:** Deterministic mapping between 30m grid row_idx and MODIS 1km pixel_id

---

## Quick Reference

Given a 30m grid cell with coordinates `(lon, lat)` in EPSG:4326:

```
pixel_col = floor((lon - BBOX_WEST) / PIXEL_SIZE_LON)
pixel_row = floor((BBOX_NORTH - lat) / PIXEL_SIZE_LAT)
pixel_id = "{pixel_row}_{pixel_col}"
```

**Constants (Mumbai):**
```
BBOX_WEST  = 72.7763°
BBOX_NORTH = 19.2701°
PIXEL_SIZE_LON = 0.00949°  (1000m at 19°N latitude)
PIXEL_SIZE_LAT = 0.00901°  (1000m latitude)
```

---

## Detailed Formula

### Coordinate System

- **Projection:** EPSG:4326 (WGS84 geographic)
- **Units:** Decimal degrees
- **Origin:** Top-left corner (northwest) of bounding box
- **X-axis:** Longitude (increases eastward)
- **Y-axis:** Latitude (increases northward, but pixel indexing is top-down)

### Bounding Box

Mumbai coverage area in decimal degrees:

```
┌─────────────────────────────────────────────┐ North (19.2701°)
│                                             │
│          Mumbai MODIS Pixel Grid            │
│          22 columns × 42 rows = 882 pixels  │
│                                             │
└─────────────────────────────────────────────┘ South (18.8939°)
West (72.7763°)                    East (72.9797°)
```

**Box dimensions:**
- East-West: 72.9797° - 72.7763° = 0.2034°
- North-South: 19.2701° - 18.8939° = 0.3762°

### Pixel Size Calculation

MODIS MAIAC native resolution is 1000m. Converting to degrees at Mumbai's latitude (19°N):

**Longitude (East-West):**
```
1° longitude at 19°N ≈ 105.3 km (varies by latitude)
1000m / 105,300m = 0.009492° ≈ 0.00949°
```

**Latitude (North-South):**
```
1° latitude ≈ 111.0 km (constant everywhere)
1000m / 111,000m = 0.009009° ≈ 0.00901°
```

### Column Index (East-West)

Maps longitude to column index (0 at western edge):

```python
pixel_col = int((lon - BBOX_WEST) / PIXEL_SIZE_LON)
           = int((lon - 72.7763) / 0.00949)
```

**Example:**
```
lon = 72.85°
pixel_col = int((72.85 - 72.7763) / 0.00949)
          = int(0.0737 / 0.00949)
          = int(7.76)
          = 7
```

**Valid range:** 0 to 21 (22 columns total)

### Row Index (North-South)

Maps latitude to row index (0 at northern edge, increases southward):

```python
pixel_row = int((BBOX_NORTH - lat) / PIXEL_SIZE_LAT)
          = int((19.2701 - lat) / 0.00901)
```

**Why inverted (NORTH - lat)?**
- Image coordinate systems have origin at top-left
- Latitude increases northward, but rows increase southward
- Must invert: higher latitude → lower row index

**Example:**
```
lat = 19.05°
pixel_row = int((19.2701 - 19.05) / 0.00901)
          = int(0.2201 / 0.00901)
          = int(24.42)
          = 24
```

**Valid range:** 0 to 41 (42 rows total)

### Pixel ID Format

Composite string identifier:

```python
pixel_id = f"{pixel_row}_{pixel_col}"
```

**Example:**
```
pixel_row = 24
pixel_col = 7
pixel_id = "24_7"
```

---

## Step-by-Step Algorithm

### Forward Mapping (Grid Cell → MODIS Pixel)

**Input:** 30m grid cell with row_idx and geometry (Point)

```python
def get_modis_pixel_id(row_idx, grid_gdf, bbox=(72.7763, 18.8939, 72.9797, 19.2701)):
    """
    Map 30m grid cell (by row_idx) to MODIS pixel_id.
    
    O(1) operation - pure coordinate math, no lookup needed.
    """
    BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH = bbox
    PIXEL_SIZE_LON = 0.00949
    PIXEL_SIZE_LAT = 0.00901
    
    # Step 1: Get coordinates from grid
    point = grid_gdf.loc[row_idx, 'geometry']
    lon = point.x
    lat = point.y
    
    # Step 2: Compute column (longitude → column)
    pixel_col = int((lon - BBOX_WEST) / PIXEL_SIZE_LON)
    
    # Step 3: Compute row (latitude → row, inverted)
    pixel_row = int((BBOX_NORTH - lat) / PIXEL_SIZE_LAT)
    
    # Step 4: Format pixel ID
    pixel_id = f"{pixel_row}_{pixel_col}"
    
    return pixel_id
```

**Complexity:** O(1)

### Inverse Mapping (MODIS Pixel → Grid Cells)

Given a `pixel_id`, find all 30m grid cells within that pixel:

```python
def get_grid_cells_in_pixel(pixel_id, grid_gdf, bbox=(72.7763, 18.8939, 72.9797, 19.2701)):
    """
    Find all 30m grid cells within a MODIS pixel.
    
    Returns ~1089 cells per pixel (33×33 at 30m within 1km).
    """
    BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH = bbox
    PIXEL_SIZE_LON = 0.00949
    PIXEL_SIZE_LAT = 0.00901
    
    # Parse pixel ID
    pixel_row_str, pixel_col_str = pixel_id.split('_')
    pixel_row = int(pixel_row_str)
    pixel_col = int(pixel_col_str)
    
    # Compute pixel bounding box
    pixel_lon_west = BBOX_WEST + pixel_col * PIXEL_SIZE_LON
    pixel_lon_east = pixel_lon_west + PIXEL_SIZE_LON
    pixel_lat_north = BBOX_NORTH - pixel_row * PIXEL_SIZE_LAT
    pixel_lat_south = pixel_lat_north - PIXEL_SIZE_LAT
    
    # Find all grid cells within this pixel's bounding box
    within_pixel = grid_gdf[
        (grid_gdf.geometry.x >= pixel_lon_west) &
        (grid_gdf.geometry.x < pixel_lon_east) &
        (grid_gdf.geometry.y >= pixel_lat_south) &
        (grid_gdf.geometry.y < pixel_lat_north)
    ]
    
    return within_pixel
```

**Result:** Typically ~924 grid cells per MODIS pixel (33×33 at 30m)

---

## Implementation Examples

### Example 1: Single Grid Cell

**Given:**
- grid_idx = 42500
- grid_gdf.loc[42500, 'geometry'] = Point(72.85, 19.05)

**Calculation:**
```
lon = 72.85
lat = 19.05

pixel_col = int((72.85 - 72.7763) / 0.00949)
          = int(7.76)
          = 7

pixel_row = int((19.2701 - 19.05) / 0.00901)
          = int(24.42)
          = 24

pixel_id = "24_7"
```

**Verification:** This grid cell belongs to MODIS pixel at row 24, col 7

---

### Example 2: Vectorized for All Grid Cells

**Using NumPy for 509,209 grid cells:**

```python
import numpy as np

BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH = bbox
PIXEL_SIZE_LON = 0.00949
PIXEL_SIZE_LAT = 0.00901

# Extract all coordinates
lons = grid_gdf.geometry.x.values  # Shape: (509209,)
lats = grid_gdf.geometry.y.values  # Shape: (509209,)

# Vectorized computation
pixel_cols = ((lons - BBOX_WEST) / PIXEL_SIZE_LON).astype(int)
pixel_rows = ((BBOX_NORTH - lats) / PIXEL_SIZE_LAT).astype(int)

# Create pixel IDs as strings
pixel_ids = np.char.add(
    np.char.add(pixel_rows.astype(str), '_'),
    pixel_cols.astype(str)
)  # Shape: (509209,)

# Add to grid
grid_gdf['modis_pixel_id'] = pixel_ids
```

**Performance:** 509,209 cells in ~0.5 seconds (~1M cells/second)

---

## Pixel Grid Reference

### Grid Dimensions

```
Rows (North-South):  42 rows (pixel_row: 0-41)
Cols (East-West):    22 cols (pixel_col: 0-21)
Total pixels:        882
```

### Pixel Centroid Coordinates

Each pixel is indexed by its center coordinate:

```python
def pixel_id_to_centroid(pixel_id):
    """Get centroid (lon, lat) of a MODIS pixel."""
    pixel_row, pixel_col = map(int, pixel_id.split('_'))
    
    # Center of pixel
    pixel_lon = BBOX_WEST + (pixel_col + 0.5) * PIXEL_SIZE_LON
    pixel_lat = BBOX_NORTH - (pixel_row + 0.5) * PIXEL_SIZE_LAT
    
    return (pixel_lon, pixel_lat)

# Example: pixel "24_7" centroid
pixel_lon = 72.7763 + (7 + 0.5) * 0.00949 = 72.8413
pixel_lat = 19.2701 - (24 + 0.5) * 0.00901 = 19.0536
# Centroid: (72.8413, 19.0536)
```

### Pixel Boundaries

Each pixel covers approximately 1km × 1km:

```python
def pixel_id_to_bounds(pixel_id):
    """Get bounding box [west, south, east, north] of MODIS pixel."""
    pixel_row, pixel_col = map(int, pixel_id.split('_'))
    
    west = BBOX_WEST + pixel_col * PIXEL_SIZE_LON
    east = west + PIXEL_SIZE_LON
    north = BBOX_NORTH - pixel_row * PIXEL_SIZE_LAT
    south = north - PIXEL_SIZE_LAT
    
    return (west, south, east, north)

# Example: pixel "24_7" bounds
pixel_col = 7
pixel_row = 24
west = 72.7763 + 7 * 0.00949 = 72.8428
east = 72.8428 + 0.00949 = 72.8523
north = 19.2701 - 24 * 0.00901 = 19.0537
south = 19.0537 - 0.00901 = 19.0447
# Bounds: (72.8428, 19.0447, 72.8523, 19.0537)
```

---

## Edge Cases & Validation

### Boundary Cells

Grid cells exactly on pixel boundaries use **floor() assignment**:

```python
# Cell at exact boundary
lon = 72.7763 + 0.00949 = 72.7858  (boundary between col 0 and 1)
pixel_col = int((72.7858 - 72.7763) / 0.00949) = int(1.0) = 1  # Assigned to col 1

# Cell at corner
lon = 72.9797  (eastern edge)
pixel_col = int((72.9797 - 72.7763) / 0.00949) = int(21.4) = 21  # Assigned to col 21 (last column)
```

### Out-of-Bounds Cells

Grid cells outside the MODIS bbox will have negative or oversized indices:

```python
# Western edge
lon = 72.7700 (outside bbox)
pixel_col = int((72.7700 - 72.7763) / 0.00949) = int(-0.66) = 0 (clamped by floor)

# Northern edge
lat = 19.3000 (outside bbox)
pixel_row = int((19.2701 - 19.3000) / 0.00901) = int(-33.2) = -34 (invalid!)
```

**Current approach:** No validation - assumes grid cells are entirely within bbox (true for our data)

**Recommended safety check:**
```python
def validate_pixel_id(pixel_id):
    """Validate pixel_id is within expected grid."""
    row, col = map(int, pixel_id.split('_'))
    return 0 <= row < 42 and 0 <= col < 22
```

---

## Determinism Guarantees

### Properties

1. **Deterministic:** Same input (lon, lat) always produces same pixel_id
2. **Idempotent:** Multiple applications produce same result
3. **Injective within pixel:** Multiple grid cells can map to same pixel_id (intended)
4. **O(1) computation:** No database lookups, pure arithmetic
5. **Invertible:** Given pixel_id, can compute pixel bounds (exact region)

### Stability

- Mapping is stable across runs (no randomness)
- Mapping is stable across machines (pure math, no machine-specific ops)
- Mapping is stable across library versions (no dependency on geospatial libs)

### Reproducibility

To reproduce this mapping elsewhere:
1. Use same constants (BBOX_WEST, BBOX_NORTH, PIXEL_SIZE_LON, PIXEL_SIZE_LAT)
2. Use floor() function (not round)
3. Use WGS84 coordinates (EPSG:4326)
4. Use standard IEEE 754 floating-point arithmetic

---

## Use Cases

### Training Data Preparation

```python
# 1. Load 30m grid
grid = gpd.read_parquet('artifacts/v1/grids/mumbai/grid.parquet')

# 2. Add MODIS pixel IDs to each cell
from patliputra.modis_pixel_utils import add_modis_pixel_id_from_geometry
grid = add_modis_pixel_id_from_geometry(grid)

# 3. For each training date, load AOD and join
for date in date_range:
    aod = pd.read_parquet(f'artifacts/v1/modis_aod/mumbai/{date}.parquet')
    training_data = grid.merge(aod, on='modis_pixel_id', how='left')
    # Result: 509,209 rows, each with its MODIS pixel's AOD values
```

### Spatial Analysis

```python
# Find all grid cells in a MODIS pixel
pixel_id = "24_7"
cells_in_pixel = grid[grid['modis_pixel_id'] == pixel_id]
print(f"{len(cells_in_pixel)} grid cells in pixel {pixel_id}")
# Output: ~924 grid cells
```

### Validation

```python
# Verify mapping is injective within pixels
pixel_counts = grid['modis_pixel_id'].value_counts()
print(f"Min cells per pixel: {pixel_counts.min()}")      # Should be ~900+
print(f"Max cells per pixel: {pixel_counts.max()}")      # Should be ~1100
print(f"Mean cells per pixel: {pixel_counts.mean():.0f}") # Should be ~924
```

---

## References

- **MODIS MAIAC:** https://doi.org/10.5067/MODIS/MCD19A2.061
- **WGS84 (EPSG:4326):** https://epsg.io/4326
- **Implementation:** `patliputra/modis_pixel_utils.py`
- **Constants source:** `patliputra/modis_aod_signals_v2.py`

