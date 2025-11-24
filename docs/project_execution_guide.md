# Pollu Project Execution Guide

**Date Started**: November 23, 2025  
**Purpose**: Step-by-step guide to run the Pollu pollution modeling pipeline from scratch

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Pipeline Execution](#pipeline-execution)
5. [Validation](#validation)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.8+ (we're using Python 3.12.10)
- ~10GB disk space for data cache
- Internet connection (for Earth Engine, OpenAQ, Open-Meteo APIs)

### Required Accounts
- Google Earth Engine service account (optional, for satellite data)
- OpenAQ API token (optional, for ground truth PM2.5)

---

## Environment Setup

### Python Dependencies

The following packages are required (from `pyproject.toml`):

**Core Packages:**
- `numpy` (2.3.5) - Numerical computing
- `pandas` (2.3.3) - Data manipulation
- `geopandas` (1.1.1) - Geospatial data handling

**Geospatial Libraries:**
- `shapely` (2.1.2) - Geometric operations
- `pyproj` (3.7.2) - Coordinate reference system transformations
- `rasterio` (1.4.3) - Raster data I/O
- `rioxarray` (0.20.0) - Raster operations with xarray
- `pyogrio` (0.11.1) - Fast vector I/O
- `pyarrow` - Parquet file format support (required for GeoParquet)

**Machine Learning:**
- `scikit-learn` (1.7.2) - ML algorithms
- `mgwr` (2.2.1) - Geographically Weighted Regression
- `scipy` (1.16.3) - Scientific computing

**API & Storage:**
- `fastapi` (0.121.3) - REST API framework
- `uvicorn` (0.38.0) - ASGI server
- `sqlalchemy` (2.0.44) - Database ORM
- `duckdb` (1.4.2) - Embedded analytics database
- `earthengine-api` - Google Earth Engine Python API

**Utilities:**
- `requests` (2.32.5) - HTTP library
- `python-dotenv` (1.2.1) - Environment variable management

### Installation

```powershell
# Navigate to project directory
# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -e .

# Install pyarrow for Parquet file support
pip install pyarrow

# Install Earth Engine API for satellite data
pip install earthengine-api

# Install pyrosm for OSM PBF file processing
pip install pyrosm
```

**What this does:**
- Creates an isolated Python environment in `.venv/`
- Installs all packages listed in `pyproject.toml`
- Installs the `pollu` package in editable mode (changes take effect immediately)

### Earth Engine Authentication

Before using satellite data, authenticate with Google Earth Engine:

```powershell
python -c "import ee; ee.Authenticate()"
```

**What this does:**
- Opens a browser window for Google authentication
- Saves credentials locally for future use
- Required only once per machine

---

## Data Preparation

### Step 1: Create 30m Grid for Mumbai

**File:** `scripts/create_grid_mumbai.py`

**Command:**
```powershell
python scripts/create_grid_mumbai.py
```

**What it does:**
1. Reads Mumbai ward boundaries from `resources/mumbai_wards.geojson` (WGS84/EPSG:4326)
2. Reprojects to UTM Zone 43N (EPSG:32643) for metric calculations
3. Creates a regular 30m × 30m grid covering the bounding box
4. Generates centroid points at the center of each cell (offset by 15m from edges)
5. Clips grid to Mumbai boundary using spatial intersection
6. Assigns sequential `grid_id` starting from 0
7. Saves as GeoParquet to `data/mumbai/grid_30m.parquet`

**Key Parameters:**
- `--boundary`: Input boundary file (default: `resources/mumbai_wards.geojson`)
- `--out`: Output parquet file (default: `data/mumbai/grid_30m.parquet`)
- `--crs`: Target CRS (default: `EPSG:32643`)
- `--res`: Grid resolution in meters (default: `30.0`)

**Expected output:**
```
Saved grid (509209 points) to data/mumbai/grid_30m.parquet
```

**Why we need this:**
All feature extraction (Landsat, Sentinel, SVF, OSM) will be sampled at these exact grid points to ensure perfect spatial alignment across all datasets.

---

### Step 2: Fetch Landsat-8 Image Dates

**File:** `scripts/fetch_landsat56.py`

**Prerequisites:**
- Earth Engine authentication completed (see [Earth Engine Authentication](#earth-engine-authentication))
- Google Earth Engine project ID

**Command:**
```powershell
python scripts/fetch_landsat56.py --project "fast-archive-465917-m0"
```

**What it does:**
1. Connects to Google Earth Engine with your project ID
2. Queries Landsat-8 Collection 2 Level-1 TOA imagery
3. Filters by Mumbai bounding box (72.7763°, 18.8939° to 72.9797°, 19.2701°)
4. Filters by date range (2018-01-01 to 2025-11-01)
5. Filters by cloud cover (< 15%)
6. Extracts unique image dates
7. Saves dates to `patliputra/landsat8_image_dates.txt`

**Key Parameters:**
- `--project`: Google Earth Engine project ID (required)
- `--output`: Output file path (default: `patliputra/landsat8_image_dates.txt`)
- `--bbox`: Bounding box as west,south,east,north (default: Mumbai)
- `--start-date`: Start date YYYY-MM-DD (default: `2018-01-01`)
- `--end-date`: End date YYYY-MM-DD (default: `2025-11-01`)
- `--max-cloud-cover`: Maximum cloud cover % (default: `15.0`)

**Expected output:**
```
Initializing Earth Engine with project: fast-archive-465917-m0
Saved XXX image dates to patliputra/landsat8_image_dates.txt
```

**Why we need this:**
This generates a list of all cloud-free Landsat-8 images available for Mumbai. These dates will be used to download and process the actual satellite imagery in the next step.

---

### Step 3: Download and Process Landsat-8 Imagery

**File:** `patliputra/landsat8_signals.py`

**Prerequisites:**
- Grid parquet created (`data/mumbai/grid_30m.parquet`)
- Landsat image dates file (`patliputra/landsat8_image_dates.txt`)
- Earth Engine authenticated
- Sufficient disk space (~50-100GB for cache)

**Command:**
```powershell
.\.venv\Scripts\python.exe patliputra/landsat8_signals.py --grid-parquet data/mumbai/grid_30m.parquet --project fast-archive-465917-m0
```

**What it does:**
1. Reads all cloud-free dates from `patliputra/landsat8_image_dates.txt` (~105 dates)
2. For each date:
   - Queries Landsat-8 TOA imagery from Earth Engine
   - Splits Mumbai bounding box into ~40 tiles (to avoid 50MB download limit)
   - Downloads tiles in parallel (8 workers)
   - Merges tiles into single GeoTIFF
   - Reprojects to EPSG:32643 at 30m resolution
   - Samples all 11 TOA bands (B1-B11) at grid centroids
   - Computes spectral indices (NDVI, NDBI, MNDWI, IBI)
   - Saves to `cache/landsat_processed/landsat8_YYYY-MM-DD_signals.parquet`
3. Caches downloaded GeoTIFFs in `cache/landsat_downloads/`

**Key Parameters:**
- `--grid-parquet`: Path to grid file (required)
- `--project`: Google Earth Engine project ID (required)
- `--dates-file`: Dates file (default: `patliputra/landsat8_image_dates.txt`)
- `--output-dir`: Processed output directory (default: `cache/landsat_processed`)
- `--cache-dir`: Download cache directory (default: `cache/landsat_downloads`)
- `--bbox`: Bounding box (default: Mumbai coordinates)
- `--target-crs`: Target CRS (default: `EPSG:32643`)
- `--target-res`: Target resolution in meters (default: `30.0`)

**Expected output:**
```
Earth Engine initialized with project: fast-archive-465917-m0
Found 105 dates to process

[1/105] Processing 2018-01-01
  Downloading 40 tiles for 2018-01-01...
    ✓ Tile 1/40 downloaded (1/40 complete)
    ...
  Merging 40 tiles...
  → Merged to cache/landsat_downloads/landsat8_2018-01-01.tif
  → Processing raster...
  → Computing spectral indices...
  → Writing to cache/landsat_processed/landsat8_2018-01-01_signals.parquet...
  ✓ Saved 509209 rows to cache/landsat_processed/landsat8_2018-01-01_signals.parquet

[2/105] Processing 2018-01-17
  ...
```

**Processing Time:**
- ~5-10 minutes per date (40 tiles × download + merge + sample)
- Total: **8-17 hours** for all 105 dates
- Resumes automatically if interrupted (checks for existing outputs)

**Output Schema:**
Each parquet file contains:
- `grid_id`: Grid cell identifier
- `x`, `y`: Coordinates in EPSG:32643
- `geometry`: Point geometry
- `toa_b1` through `toa_b11`: Landsat-8 TOA band reflectances
- `ndvi`: Normalized Difference Vegetation Index (B5-B4)/(B5+B4)
- `ndbi`: Normalized Difference Built-up Index (B6-B5)/(B6+B5)
- `mndwi`: Modified Normalized Difference Water Index (B3-B6)/(B3+B6)
- `ibi`: Index-based Built-up Index (Xu 2008 formula)
- `is_nodata`: Boolean flag for missing/invalid data
- `solar_azimuth`, `solar_zenith`: Sun angles
- `sensor_azimuth`, `sensor_zenith`: Sensor viewing angles
- `raster_meta`: JSON metadata

**Why we need this:**
Landsat-8 provides the core predictors for PM2.5 modeling: land surface characteristics (NDVI for vegetation, NDBI/IBI for built-up areas), which correlate with pollution sources and sinks. The 30m resolution enables hyperlocal predictions.

---

### Step 4: Find Sentinel-2 Low-Cloud Dates

**File:** `scripts/sentinel2_low_cloud_dates.py`

**Prerequisites:**
- Earth Engine authenticated
- Google Earth Engine project ID

**Command:**
```powershell
python scripts/sentinel2_low_cloud_dates.py --project fast-archive-465917-m0
```

**What it does:**
1. Connects to Google Earth Engine with your project ID
2. Queries Sentinel-2 SR Harmonized collection (`COPERNICUS/S2_SR_HARMONIZED`)
3. Filters by Mumbai bounding box
4. Applies QA60 cloud mask to filter cloudy pixels
5. For each month in the date range, selects the date with lowest cloud coverage
6. Saves monthly representative dates to `patliputra/sentinel2_low_cloud_dates.txt`

**Key Parameters:**
- `--project`: Google Earth Engine project ID (required)
- `--output`: Output file path (default: `patliputra/sentinel2_low_cloud_dates.txt`)
- `--bbox`: Bounding box (default: Mumbai coordinates)
- `--start-date`: Start date (default: matches Landsat range)
- `--end-date`: End date (default: matches Landsat range)

**Expected output:**
```
Initializing Earth Engine with project: fast-archive-465917-m0
Saved XX monthly Sentinel-2 dates to patliputra/sentinel2_low_cloud_dates.txt
```

**Why we need this:**
Sentinel-2 provides higher resolution (10m) vegetation data through EVI (Enhanced Vegetation Index). Since Sentinel-2 has more frequent revisits than Landsat, we select the best monthly image to map to Landsat dates.

---

### Step 5: Process Sentinel-2 EVI Signals

**File:** `patliputra/evi_signals.py`

**Prerequisites:**
- Grid parquet created (`data/mumbai/grid_30m.parquet`)
- Landsat image dates file (`patliputra/landsat8_image_dates.txt`)
- Sentinel-2 low-cloud dates file (`patliputra/sentinel2_low_cloud_dates.txt`)
- Landsat parquet files processed (`cache/landsat_processed/landsat8_*_signals.parquet`)
- Earth Engine authenticated

**Command:**
```powershell
.\.venv\Scripts\python.exe patliputra/evi_signals.py --ee-project fast-archive-465917-m0
```

**What it does:**
1. Reads all Landsat processing dates from `patliputra/landsat8_image_dates.txt`
2. Reads monthly Sentinel-2 low-cloud dates from `patliputra/sentinel2_low_cloud_dates.txt`
3. For each Landsat date:
   - Maps to closest Sentinel-2 monthly date (within 7-day window by default)
   - Downloads Sentinel-2 SR Harmonized imagery from Earth Engine
   - Applies QA60 cloud mask to remove cloudy pixels
   - Computes Enhanced Vegetation Index (EVI): `2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)`
   - Reprojects to EPSG:32643 at 30m resolution to match grid
   - Samples EVI values at all grid centroids
   - Updates existing Landsat parquet files with new EVI columns
4. For missing dates (no Sentinel-2 within window), estimates EVI from temporal neighbors

**Key Parameters:**
- `--ee-project`: Google Earth Engine project ID (required)
- `--dates-path`: Landsat dates file (default: `patliputra/landsat8_image_dates.txt`)
- `--grid-parquet`: Grid file (default: `data/mumbai/grid_30m.parquet`)
- `--sentinel-dates-path`: Sentinel-2 dates file (default: `patliputra/sentinel2_low_cloud_dates.txt`)
- `--cache-root`: Cache directory containing Landsat parquets (default: `cache`)
- `--bbox`: Bounding box (default: Mumbai coordinates)
- `--estimate-window`: Days to look for neighbors when estimating (default: 7)
- `--force`: Recompute even if EVI columns already exist

**Expected output:**
```
Running evi_signals with dates=patliputra/landsat8_image_dates.txt grid=data/mumbai/grid_30m.parquet cache=cache force=False

[1/105] Processing 2018-01-01
  → Mapped to Sentinel-2 date: 2018-01-05
  → Downloading Sentinel-2 imagery...
  → Computing EVI...
  → Sampling at grid points...
  → Updating cache/landsat_processed/landsat8_2018-01-01_signals.parquet...
  ✓ Added EVI column (509209 rows)

[2/105] Processing 2018-01-17
  ...
```

**Processing Time:**
- ~2-5 minutes per date (single image download, no tiling needed due to smaller size)
- Total: **3-9 hours** for all 105 dates
- Resumes automatically if interrupted

**Output Schema:**
Updates existing Landsat parquet files with additional columns:
- `evi`: Enhanced Vegetation Index from Sentinel-2 (10m resolution, resampled to 30m)
- `evi_estimated`: Boolean flag indicating if EVI was estimated from neighbors (no direct Sentinel-2 match)

**Why we need this:**
EVI from Sentinel-2 provides a superior vegetation metric compared to Landsat NDVI due to:
1. Higher native resolution (10m vs 30m)
2. Better sensitivity in high-biomass areas
3. Reduced atmospheric and soil background effects
By mapping monthly Sentinel-2 dates to Landsat dates, we enrich each observation with the best available vegetation data.

---

### Step 6: Process Sentinel-5P NO2 Signals

**File:** `patliputra/s5p_no2_signals.py`

**Prerequisites:**
- Grid parquet created (`data/mumbai/grid_30m.parquet`)
- Landsat image dates file (`patliputra/landsat8_image_dates.txt`)
- Landsat parquet files processed (`cache/landsat_processed/landsat8_*_signals.parquet`)
- Earth Engine authenticated

**Command:**
```powershell
.\.venv\Scripts\python.exe patliputra/s5p_no2_signals.py --ee-project fast-archive-465917-m0
```

**What it does:**
1. Reads all Landsat processing dates from `patliputra/landsat8_image_dates.txt`
2. For each date:
   - Queries Sentinel-5P NO2 offline product (`COPERNICUS/S5P/OFFL/L3_NO2`)
   - Selects tropospheric NO2 column number density band
   - Downloads image from Earth Engine (scale: 1113.2m native resolution)
   - Reprojects to EPSG:32643 at 30m resolution to match grid
   - Samples NO2 values at all grid centroids
   - Updates existing Landsat parquet files with new S5P NO2 columns
3. For dates with no S5P coverage (satellite revisit gaps):
   - Finds nearest previous and next available S5P observations (within 7-day window)
   - Performs linear temporal interpolation between neighbors
   - Marks interpolated values with metadata flag
4. Caches downloaded GeoTIFFs in `cache/copernicus_YYYY-MM-DD/`

**Key Parameters:**
- `--dates-path`: Landsat dates file (default: `patliputra/landsat8_image_dates.txt`)
- `--grid-parquet`: Grid file (default: `data/mumbai/grid_30m.parquet`)
- `--cache-root`: Cache directory containing Landsat parquets (default: `cache`)
- `--ee-project`: Google Earth Engine project ID
- `--bbox`: Bounding box (default: Mumbai coordinates)
- `--estimate-window`: Days to look for neighbors when estimating (default: 7)
- `--force`: Recompute even if S5P columns already exist

**Expected output:**
```
INFO: Queueing processing for dates listed in patliputra/landsat8_image_dates.txt
INFO: Reading dates from patliputra/landsat8_image_dates.txt
INFO: Loaded 105 dates

[Processing 2018-01-01]
INFO: Downloading S5P image to cache/copernicus_2018-01-01/copernicus_2018-01-01.tif
INFO: Saved GeoTIFF: cache/copernicus_2018-01-01/copernicus_2018-01-01.tif
INFO: Updated cache/landsat_processed/landsat8_2018-01-01_signals.parquet with S5P NO2 columns

[Processing 2018-01-17]
INFO: No S5P image available for 2018-01-17
INFO: Finding neighbors within 7-day window...
INFO: Interpolating between prev=2018-01-14 and next=2018-01-20
INFO: Updated cache/landsat_processed/landsat8_2018-01-17_signals.parquet with S5P NO2 columns
  ...
```

**Processing Time:**
- ~2-3 minutes per date with direct S5P coverage (single image download)
- ~4-6 minutes per date requiring interpolation (3 images: target neighbors)
- Total: **4-8 hours** for all 105 dates
- Resumes automatically if interrupted (checks for existing S5P columns)

**Output Schema:**
Updates existing Landsat parquet files with additional columns:
- `s5p_no2`: Tropospheric NO2 column number density (mol/m²)
  - Native Sentinel-5P resolution: ~7km × 3.5km at nadir
  - Downsampled to 30m to match grid (each grid cell gets spatially nearest NO2 value)
- `s5p_no2_is_nodata`: Boolean flag for missing/invalid NO2 data
- `s5p_no2_meta`: JSON metadata string containing:
  - `source`: Collection ID (`COPERNICUS/S5P/OFFL/L3_NO2`)
  - `band`: Band name (`tropospheric_NO2_column_number_density`)
  - `method`: Processing method (`observed` or `interpolated`)
  - `estimated_from`: For interpolated dates, contains `{"prev": "YYYY-MM-DD", "next": "YYYY-MM-DD"}`

**Why we need this:**
Sentinel-5P provides atmospheric NO2 measurements which serve as a direct proxy for combustion-related air pollution (vehicles, industry, power plants). Key benefits:
1. **Direct pollution indicator**: NO2 correlates strongly with PM2.5 from fossil fuel combustion
2. **Daily global coverage**: TROPOMI instrument provides near-daily observations
3. **Complements surface features**: While Landsat/Sentinel-2 capture land characteristics (vegetation, built-up areas), S5P captures actual atmospheric pollutants
4. **Temporal dynamics**: Captures day-to-day variability in emissions and atmospheric chemistry

**Technical Notes:**
- **Spatial resolution gap**: S5P native resolution (~7km) is coarser than our 30m grid. Each grid cell receives the NO2 value from the spatially nearest S5P pixel. This is acceptable because:
  - NO2 has spatial autocorrelation (neighboring areas have similar concentrations)
  - The model learns spatial patterns from high-resolution features (Landsat, OSM, SVF)
  - S5P provides the temporal signal of pollution events
- **Temporal interpolation**: S5P has 1-day revisit time but occasional gaps due to clouds or processing. Linear interpolation is used for gaps ≤7 days, which is reasonable given NO2's sub-weekly temporal autocorrelation
- **Data format**: Downloaded as GeoTIFF from Earth Engine, then sampled using the same `raster_to_grid_df` utility as Landsat processing

**Resume behavior:**
If processing is interrupted, rerunning the command will:
- Skip dates that already have `s5p_no2` column in their Landsat parquet
- Resume from the first date without S5P data
- Use the `--force` flag to reprocess all dates (useful if you want to change interpolation parameters)

---

### Step 7: Compute Sky View Factor (SVF) for Multiple Years

**File:** `patliputra/svf_earth_engine.py`

**Prerequisites:**
- Earth Engine authenticated
- Google Earth Engine project ID
- Sufficient disk space (~5-10GB per year for cache)

**Command:**
```powershell
.\.venv\Scripts\python.exe patliputra\svf_earth_engine.py --project fast-archive-465917-m0
```

**What it does:**
1. Automatically processes years 2018-2022 (no year parameter needed)
2. For each year:
   - Creates a **proxy Digital Surface Model (DSM)** by combining:
     - **NASA DEM** (ground elevation): Provides terrain elevation baseline
     - **Google Open Buildings Temporal** (building heights): Adds 3D building structures
     - Formula: `DSM = DEM + building_height` (where buildings exist)
   - Computes **Sky View Factor (SVF)** using horizon angle search:
     - Searches in 16 azimuth directions (22.5° spacing)
     - For each direction, traces a 100m radius to find maximum horizon angle
     - SVF = (1/16) × Σ sin²(horizon_angle_i)
   - Splits Mumbai bounding box into tiles (~100 tiles at 3m resolution)
   - Downloads tiles in parallel (4 concurrent workers with rate limiting)
   - Merges tiles into single GeoTIFF
   - Saves to `cache/mumbai_svf_{year}_3m.tif`
3. Skips years with existing output files (resume capability)

**Key Parameters:**
- `--project`: Google Earth Engine project ID (required)
- `--years`: Comma-separated years (default: `2018,2019,2020,2021,2022`)
- `--output-dir`: Output directory (default: `cache/`)
- `--resolution`: Pixel resolution in meters (default: `3.0`)
- `--directions`: Number of azimuth directions (default: `16`)
- `--radius`: Horizon search radius in meters (default: `100.0`)
- `--bbox`: Bounding box west,south,east,north (default: Mumbai coordinates)
- `--tile-cache-dir`: Tile cache directory (default: `cache/svf_tiles`)
- `--clean-cache`: Remove existing tile cache before processing

**Expected output:**
```
Earth Engine initialized with project: fast-archive-465917-m0

============================================================
Processing year 2018 (1/5)
============================================================
Computing SVF for 2018 with 16 directions at 3.0m resolution...
  [SVF] Creating DSM...
  [SVF] Computing horizon angles in 16 directions (step: 22.5°)...
    [SVF] Computing direction 1/16 (0°)...
    [SVF] Computing direction 5/16 (90°)...
    [SVF] Computing direction 9/16 (180°)...
    [SVF] Computing direction 13/16 (270°)...
  [SVF] Averaging over 16 directions...
  [SVF] ✓ SVF computation complete
Downloading 100 tiles...
    ✓ Tile 1/100 downloaded (1/100 complete)
    ✓ Tile 2/100 cached (2/100 complete)
    ...
Merging 100 tiles...
✓ SVF saved to cache/mumbai_svf_2018_3m.tif

============================================================
Processing year 2019 (2/5)
============================================================
...

============================================================
All 5 years processed successfully!
============================================================
```

**Processing Time:**
- ~30-60 minutes per year (100 tiles × download + merge)
- SVF computation (server-side): ~2-3 minutes
- Tile downloads: ~25-50 minutes (with rate limiting and retries)
- Tile merging: ~3-5 minutes
- Total for all 5 years: **2.5-5 hours**
- Resumes automatically if interrupted (checks for existing outputs and cached tiles)

**Output Files:**
- `cache/mumbai_svf_2018_3m.tif` (and 2019, 2020, 2021, 2022)
- `cache/svf_tiles/svf_2018_3m/tile_000.tif` through `tile_099.tif` (cached for resume)

**Understanding the Proxy DSM Approach:**

Our SVF computation requires a **Digital Surface Model (DSM)** that represents the actual height of all surfaces (ground + buildings + vegetation). However, high-resolution DSM data is not readily available globally. We create a **proxy DSM** by combining two freely available datasets:

**1. NASA DEM (Digital Elevation Model)**
- Provides ground elevation (terrain topography)
- Resolution: ~30m
- Represents "bare earth" without buildings or vegetation

**2. Google Open Buildings Temporal**
- Provides building height estimates from 2016-2023
- Resolution: Variable (typically 1-5m)
- Machine learning-derived from satellite imagery
- Temporal: Captures building construction/demolition over time

**Proxy DSM Formula:**
```python
DSM = DEM + building_height
```

Where buildings exist, we add their height to ground elevation. Where no buildings exist (parks, roads, open spaces), the DSM equals the DEM.

**Why This Works:**

1. **Urban focus**: Mumbai is highly urbanized. Buildings are the dominant vertical structures affecting sky visibility at street level.

2. **Missing vegetation is acceptable**: 
   - Trees/vegetation are not included in our proxy DSM
   - This is a **conservative approximation** (underestimates obstruction)
   - For PM2.5 modeling, this may actually be beneficial since vegetation acts as a sink (removes pollutants)
   - We capture vegetation effects separately through NDVI/EVI indices

3. **Temporal alignment**:
   - Google Open Buildings provides yearly data (2016-2023)
   - We compute SVF for years matching our Landsat observations (2018-2022)
   - Captures urban development over time (new construction reduces SVF)

4. **Resolution tradeoff**:
   - Final SVF at 3m resolution captures local-scale urban geometry
   - Much higher resolution than alternatives (e.g., global DSM at 30-90m)
   - Sufficient to distinguish street canyons, courtyards, and open squares

**SVF Interpretation:**
- **SVF = 1.0**: Completely open sky (rooftop, open field, wide road)
- **SVF = 0.5**: Moderate obstruction (low-density residential, tree-lined streets)
- **SVF = 0.2**: Heavy obstruction (narrow street canyon, dense urban core)
- **SVF = 0.0**: Fully enclosed (theoretically; rare in practice)

**Why SVF Matters for PM2.5:**
1. **Ventilation proxy**: Lower SVF = reduced wind flow = pollutant accumulation
2. **Thermal effects**: SVF affects urban heat island, which influences atmospheric mixing
3. **Traffic channeling**: Street geometry (captured by SVF) concentrates vehicle emissions
4. **Microclimate**: SVF modulates temperature inversions that trap pollutants

**Rate Limiting & Robustness:**
- **Parallel downloads**: 4 concurrent workers (reduced from 8 to avoid Earth Engine rate limits)
- **Retry logic**: Each tile retries up to 3 times with exponential backoff (5s, 10s, 20s)
- **Timeout**: 15-minute timeout per tile download
- **Staggered requests**: 0.5-second delays between worker starts
- **Cached tiles**: Existing tiles are reused on resume

**Resume behavior:**
If processing is interrupted:
```powershell
# Simply rerun the same command - it will skip completed years
.\.venv\Scripts\python.exe patliputra\svf_earth_engine.py --project fast-archive-465917-m0

# To reprocess specific years
.\.venv\Scripts\python.exe patliputra\svf_earth_engine.py --project fast-archive-465917-m0 --years "2020,2021"

# To force reprocessing (clears tile cache first)
.\.venv\Scripts\python.exe patliputra\svf_earth_engine.py --project fast-archive-465917-m0 --clean-cache
```

---

## Pipeline Execution

_[To be filled as we execute remaining steps]_

---
