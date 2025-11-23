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

## Pipeline Execution

_[To be filled as we execute remaining steps]_

---
